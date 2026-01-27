# pylint: disable=too-many-ancestors
from __future__ import annotations

from typing import TYPE_CHECKING

from d3rlpy.algos.qlearning.torch.rebrac_impl import ReBRACImpl
from d3rlpy.algos.qlearning.torch.td3_plus_bc_impl import TD3PlusBCActorLoss
import torch

from scs.gates import std_gate

if TYPE_CHECKING:
    from d3rlpy.algos.qlearning.torch.ddpg_impl import DDPGModules
    from d3rlpy.models.torch import (
        ActionOutput,
        ContinuousEnsembleQFunctionForwarder,
    )
    from d3rlpy.types import Shape

    from scs.torch_utility import TorchMiniBatchWithPolicy

__all__ = ["ReBRACgRImpl"]


class ReBRACgRImpl(ReBRACImpl):
    """ReBRAC with gated regularization (ReBRACgR).

    Adapted from d3rlpy's ReBRACImpl. Applies std_gate to:
    1. Actor loss: scales the Q-value term by behavior policy confidence.
    2. Target: scales the BC penalty term by behavior policy confidence.

    See compute_actor_loss and compute_target.
    """

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DDPGModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        actor_beta: float,
        critic_beta: float,
        update_actor_interval: int,
        gate_weight: float,
        compiled: bool,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
            tau=tau,
            target_smoothing_sigma=target_smoothing_sigma,
            target_smoothing_clip=target_smoothing_clip,
            actor_beta=actor_beta,
            critic_beta=critic_beta,
            update_actor_interval=update_actor_interval,
            compiled=compiled,
            device=device,
        )
        self._gate_weight = gate_weight

    def compute_actor_loss(
        self, batch: TorchMiniBatchWithPolicy, action: ActionOutput
    ) -> TD3PlusBCActorLoss:
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations,
            action.squashed_mu,
            reduction="min",
        )
        lam = 1 / (q_t.abs().mean()).detach()
        # gR: Scale Q-value by behavior policy confidence (Eq. 11).
        with torch.no_grad():
            gate = std_gate(batch.stds, self._gate_weight)
        bc_loss = ((batch.actions - action.squashed_mu) ** 2).sum(dim=1, keepdim=True)
        return TD3PlusBCActorLoss(
            actor_loss=(gate * lam * -q_t + self._actor_beta * bc_loss).mean(),
            bc_loss=bc_loss.mean(),
        )

    def compute_target(self, batch: TorchMiniBatchWithPolicy) -> torch.Tensor:
        with torch.no_grad():
            action = self._modules.targ_policy(batch.next_observations)
            # smoothing target
            noise = torch.randn(action.mu.shape, device=batch.device)
            scaled_noise = self._target_smoothing_sigma * noise
            clipped_noise = scaled_noise.clamp(
                -self._target_smoothing_clip, self._target_smoothing_clip
            )
            smoothed_action = action.squashed_mu + clipped_noise
            clipped_action = smoothed_action.clamp(-1.0, 1.0)
            next_q = self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                clipped_action,
                reduction="min",
            )

            # gR: Scale BC penalty by behavior policy confidence (Eq. 11).
            with torch.no_grad():
                gate = std_gate(batch.stds, self._gate_weight)
            bc_penalty = ((clipped_action - batch.next_actions) ** 2).sum(
                dim=1, keepdim=True
            )
            return next_q - self._critic_beta * gate * bc_penalty
