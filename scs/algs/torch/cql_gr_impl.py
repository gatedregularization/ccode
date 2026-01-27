from __future__ import annotations

from typing import TYPE_CHECKING

from d3rlpy.algos.qlearning.torch.cql_impl import (
    CQLCriticLoss,
    CQLImpl,
    CQLModules,
)
from d3rlpy.algos.qlearning.torch.sac_impl import SACImpl
from d3rlpy.models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    get_parameter,
)
import torch

from scs.gates import std_gate

if TYPE_CHECKING:
    from d3rlpy.types import (
        Shape,
        TorchObservation,
    )

    from scs.torch_utility import TorchMiniBatchWithPolicy

__all__ = ["CQLgRImpl"]


class CQLgRImpl(CQLImpl):
    """CQL with gated regularization (CQLgR).

    Adapted from d3rlpy's CQLImpl. Applies std_gate to the conservative loss
    term, scaling it by behavior policy confidence.

    See _compute_conservative_loss.
    """

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: CQLModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        alpha_threshold: float,
        conservative_weight: float,
        n_action_samples: int,
        soft_q_backup: bool,
        max_q_backup: bool,
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
            alpha_threshold=alpha_threshold,
            conservative_weight=conservative_weight,
            n_action_samples=n_action_samples,
            soft_q_backup=soft_q_backup,
            max_q_backup=max_q_backup,
            compiled=compiled,
            device=device,
        )
        self._gate_weight: float = gate_weight

    def compute_critic_loss(
        self, batch: TorchMiniBatchWithPolicy, q_tpn: torch.Tensor
    ) -> CQLCriticLoss:
        # call loss from CQLImpl parent class SACImpl
        loss = SACImpl.compute_critic_loss(self, batch, q_tpn)
        conservative_loss = self._compute_conservative_loss(
            obs_t=batch.observations,
            act_t=batch.actions,
            obs_tp1=batch.next_observations,
            returns_to_go=batch.returns_to_go,
            stds=batch.stds,
        )

        if self._modules.alpha_optim:
            self.update_alpha(conservative_loss.detach())

        # clip for stability
        log_alpha = get_parameter(self._modules.log_alpha)
        clipped_alpha = log_alpha.exp().clamp(0, 1e6)[0][0]
        scaled_conservative_loss = clipped_alpha * conservative_loss

        return CQLCriticLoss(
            critic_loss=loss.critic_loss + scaled_conservative_loss.sum(),
            conservative_loss=scaled_conservative_loss.sum(),
            alpha=clipped_alpha,
        )

    def _compute_conservative_loss(
        self,
        obs_t: TorchObservation,
        act_t: torch.Tensor,
        obs_tp1: TorchObservation,
        returns_to_go: torch.Tensor,
        stds: torch.Tensor,
    ) -> torch.Tensor:
        policy_values_t, log_probs_t = self._compute_policy_is_values(
            policy_obs=obs_t,
            value_obs=obs_t,
            returns_to_go=returns_to_go,
        )
        policy_values_tp1, log_probs_tp1 = self._compute_policy_is_values(
            policy_obs=obs_tp1,
            value_obs=obs_t,
            returns_to_go=returns_to_go,
        )
        random_values, random_log_probs = self._compute_random_is_values(obs_t)

        # compute logsumexp
        # (n critics, batch, 3 * n samples) -> (n critics, batch, 1)
        target_values = torch.cat(
            [
                policy_values_t - log_probs_t,
                policy_values_tp1 - log_probs_tp1,
                random_values - random_log_probs,
            ],
            dim=2,
        )
        logsumexp = torch.logsumexp(target_values, dim=2, keepdim=True)

        # estimate action-values for data actions
        data_values = self._q_func_forwarder.compute_expected_q(obs_t, act_t, "none")
        pre_gate_loss = logsumexp - data_values
        # gR: Scale conservative loss by behavior policy confidence (Eq. 11).
        with torch.no_grad():
            gate = std_gate(stds, self._gate_weight)
        loss = (gate * pre_gate_loss).mean(dim=[1, 2])
        return self._conservative_weight * (loss - self._alpha_threshold)
