from __future__ import annotations

from dataclasses import (
    dataclass,
)
import math
from typing import TYPE_CHECKING

from d3rlpy.algos.qlearning.base import QLearningAlgoBase
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
    create_normal_policy,
    create_parameter,
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

from scs.algs.torch.cql_gr_impl import (
    CQLgRImpl,
    CQLModules,
)
from scs.torch_utility import TorchMiniBatchWithPolicy

if TYPE_CHECKING:
    from d3rlpy.types import Shape

    from scs.datasets.mini_batch import TransitionMiniBatchWithPolicy

__all__ = ["CQLgR", "CQLgRConfig"]


@dataclass()
class CQLgRConfig(LearnableConfig):
    r"""Config of Conservative Q-Learning algorithm with gated regularization.

    References:
        * `Kumar et al., Conservative Q-Learning for Offline Reinforcement
          Learning. <https://arxiv.org/abs/2006.04779>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        temp_learning_rate (float):
            Learning rate for temperature parameter of SAC.
        alpha_learning_rate (float): Learning rate for :math:`\alpha`.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        temp_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the temperature.
        alpha_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for :math:`\alpha`.
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
        initial_temperature (float): Initial temperature value.
        initial_alpha (float): Initial :math:`\alpha` value.
        alpha_threshold (float): Threshold value described as :math:`\tau`.
        conservative_weight (float): Constant weight to scale conservative loss.
        n_action_samples (int): Number of sampled actions to compute
            :math:`\log{\sum_a \exp{Q(s, a)}}`.
        soft_q_backup (bool): Flag to use SAC-style backup.
        max_q_backup (bool): Flag to sample max Q-values for target.
        gate_weight (float): Weight injected to the gate in the loss function and
            changes the aggressiveness of the gate.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    temp_learning_rate: float = 1e-4
    alpha_learning_rate: float = 1e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    temp_optim_factory: OptimizerFactory = make_optimizer_field()
    alpha_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    initial_temperature: float = 1.0
    initial_alpha: float = 1.0
    alpha_threshold: float = 10.0
    conservative_weight: float = 5.0
    n_action_samples: int = 10
    soft_q_backup: bool = False
    max_q_backup: bool = False
    gate_weight: float = 2.0

    def create(self, device: DeviceArg = False, enable_ddp: bool = False) -> CQLgR:
        return CQLgR(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "cql_gr"


class CQLgR(QLearningAlgoBase[CQLgRImpl, CQLgRConfig]):
    def inner_create_impl(self, observation_shape: Shape, action_size: int) -> None:
        assert not (self._config.soft_q_backup and self._config.max_q_backup), (
            "soft_q_backup and max_q_backup are mutually exclusive."
        )
        policy = create_normal_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        q_funcs, q_func_fowarder = create_continuous_q_function(
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
        log_temp = create_parameter(
            (1, 1),
            math.log(self._config.initial_temperature),
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        log_alpha = create_parameter(
            (1, 1),
            math.log(self._config.initial_alpha),
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
        if self._config.temp_learning_rate > 0:
            temp_optim = self._config.temp_optim_factory.create(
                log_temp.named_modules(),
                lr=self._config.temp_learning_rate,
                compiled=self.compiled,
            )
        else:
            temp_optim = None
        if self._config.alpha_learning_rate > 0:
            alpha_optim = self._config.alpha_optim_factory.create(
                log_alpha.named_modules(),
                lr=self._config.alpha_learning_rate,
                compiled=self.compiled,
            )
        else:
            alpha_optim = None

        modules = CQLModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            log_temp=log_temp,
            log_alpha=log_alpha,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            temp_optim=temp_optim,
            alpha_optim=alpha_optim,
        )

        self._impl = CQLgRImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_fowarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            alpha_threshold=self._config.alpha_threshold,
            conservative_weight=self._config.conservative_weight,
            n_action_samples=self._config.n_action_samples,
            soft_q_backup=self._config.soft_q_backup,
            max_q_backup=self._config.max_q_backup,
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


register_learnable(CQLgRConfig)
