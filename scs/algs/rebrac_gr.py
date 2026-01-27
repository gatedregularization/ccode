from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from d3rlpy.algos.qlearning.base import QLearningAlgoBase
from d3rlpy.algos.qlearning.torch.ddpg_impl import DDPGModules
from d3rlpy.base import (
    DeviceArg,
    LearnableConfig,
    register_learnable,
)
from d3rlpy.constants import (
    IMPL_NOT_INITIALIZED_ERROR,
    ActionSpace,
)
from d3rlpy.models.builders import (
    create_continuous_q_function,
    create_deterministic_policy,
)
from d3rlpy.models.encoders import (
    EncoderFactory,
    make_encoder_field,
)
from d3rlpy.models.q_functions import (
    QFunctionFactory,
    make_q_func_field,
)
from d3rlpy.optimizers.optimizers import (
    OptimizerFactory,
    make_optimizer_field,
)

from scs.algs.torch.rebrac_gr_impl import ReBRACgRImpl
from scs.torch_utility import TorchMiniBatchWithPolicy

if TYPE_CHECKING:
    from d3rlpy.types import Shape

    from scs.datasets.mini_batch import TransitionMiniBatchWithPolicy

__all__ = ["ReBRACgR", "ReBRACgRConfig"]


@dataclasses.dataclass()
class ReBRACgRConfig(LearnableConfig):
    r"""Config of ReBRAC algorithm with gated regularization.

    References:
        * `Tarasov et al., Revisiting the Minimalist Approach to Offline
          Reinforcement Learning. <https://arxiv.org/abs/2305.09836>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for a policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        tau (float): Target network synchronization coefficiency.
        n_critics (int): Number of Q functions for ensemble.
        target_smoothing_sigma (float): Standard deviation for target noise.
        target_smoothing_clip (float): Clipping range for target noise.
        actor_beta (float): :math:`\beta_1` value.
        critic_beta (float): :math:`\beta_2` value.
        update_actor_interval (int): Interval to update policy function
            described as `delayed policy update` in the paper.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 1024
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    target_smoothing_sigma: float = 0.2
    target_smoothing_clip: float = 0.5
    actor_beta: float = 0.001
    critic_beta: float = 0.01
    update_actor_interval: int = 2
    gate_weight: float = 2.0

    def create(self, device: DeviceArg = False, enable_ddp: bool = False) -> ReBRACgR:
        return ReBRACgR(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "rebrac_gr"


class ReBRACgR(QLearningAlgoBase[ReBRACgRImpl, ReBRACgRConfig]):
    def inner_create_impl(self, observation_shape: Shape, action_size: int) -> None:
        policy = create_deterministic_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_policy = create_deterministic_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        q_funcs, q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_q_funcs, targ_q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )

        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(),
            lr=self._config.actor_learning_rate,
            compiled=self.compiled,
        )
        critic_optim = self._config.critic_optim_factory.create(
            q_funcs.named_modules(),
            lr=self._config.critic_learning_rate,
            compiled=self.compiled,
        )

        modules = DDPGModules(
            policy=policy,
            targ_policy=targ_policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
        )

        self._impl = ReBRACgRImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            target_smoothing_sigma=self._config.target_smoothing_sigma,
            target_smoothing_clip=self._config.target_smoothing_clip,
            actor_beta=self._config.actor_beta,
            critic_beta=self._config.critic_beta,
            update_actor_interval=self._config.update_actor_interval,
            gate_weight=self._config.gate_weight,
            compiled=self.compiled,
            device=self._device,
        )

    def update(self, batch: TransitionMiniBatchWithPolicy) -> dict[str, float]:
        """Update parameters with mini-batch of data.

        IMPORTANT: Calls the WithPolicy version of the mini-batch.

        Args:
            batch: Mini-batch data.

        Returns:
            Dictionary of metrics.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        torch_batch = TorchMiniBatchWithPolicy.from_batch(
            batch=batch,
            gamma=self._config.gamma,
            compute_returns_to_go=self.need_returns_to_go,
            device=self._device,
            observation_scaler=self._config.observation_scaler,
            action_scaler=self._config.action_scaler,
            reward_scaler=self._config.reward_scaler,
        )
        loss = self._impl.update(torch_batch, self._grad_step)
        self._grad_step += 1
        return loss

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(ReBRACgRConfig)
