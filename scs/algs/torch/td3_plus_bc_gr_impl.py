# pylint: disable=too-many-ancestors
from __future__ import annotations

from typing import TYPE_CHECKING

from d3rlpy.algos.qlearning.torch.td3_plus_bc_impl import (
    TD3PlusBCActorLoss,
    TD3PlusBCImpl,
)
import torch

from scs.gates import std_gate

if TYPE_CHECKING:
    from d3rlpy.algos.qlearning.torch.ddpg_impl import (
        DDPGModules,
    )
    from d3rlpy.models.torch import (
        ActionOutput,
        ContinuousEnsembleQFunctionForwarder,
    )
    from d3rlpy.types import Shape

    from scs.torch_utility import TorchMiniBatchWithPolicy

__all__ = ["TD3PlusBCgRImpl"]


class TD3PlusBCgRImpl(TD3PlusBCImpl):
    """TD3+BC with gated regularization (TD3+BCgR).

    Adapted from d3rlpy's TD3PlusBCImpl. Applies std_gate to the Q-value term
    in the actor loss, scaling policy improvement by behavior policy confidence.

    See compute_actor_loss.
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
        alpha: float,
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
            alpha=alpha,
            update_actor_interval=update_actor_interval,
            compiled=compiled,
            device=device,
        )
        self._gate_weight = gate_weight

    def compute_actor_loss(
        self, batch: TorchMiniBatchWithPolicy, action: ActionOutput
    ) -> TD3PlusBCActorLoss:
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations, action.squashed_mu, "none"
        )[0]
        lam = self._alpha / (q_t.abs().mean()).detach()
        # gR: Scale Q-value by behavior policy confidence (Eq. 11).
        with torch.no_grad():
            gate = std_gate(batch.stds, self._gate_weight)
        bc_loss = ((batch.actions - action.squashed_mu) ** 2).mean()
        scaled_q_t = q_t * gate
        return TD3PlusBCActorLoss(
            actor_loss=lam * -scaled_q_t.mean() + bc_loss, bc_loss=bc_loss
        )
