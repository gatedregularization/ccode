from __future__ import annotations

import torch


def std_gate(
    stds: torch.Tensor,
    weight: float,
    floor: float = 0.0,
    means: torch.Tensor | None = None,
    *,
    per_dim: bool = False,
    normalize: bool = True,
    gate_mode: str = "std",
) -> torch.Tensor:
    """Computes gate values from behavior policy standard deviations or ablation modes.

    Extends Equation (11) in the paper with tanh-squashing Jacobian correction,
    scale-free per-dimension standardization, per-dimension output, and batch
    mean-normalization (matching baseline regularization strength via
    ``beta_max = beta_base / E[Phi]``):
    * ``gate_mode="std"``: Scale-free, mean-normalized behavioral gate.
    * ``gate_mode="constant"``: Constant gate equal to ``weight`` (e.g. ``0.5``)
      for all transitions and dimensions.
    * ``gate_mode="uniform"``: Stochastic gate sampled i.i.d. from
      ``Uniform(floor, 1.0)`` (mean ``0.5`` when ``floor=0.0``).

    Args:
        stds: Behavior policy standard deviations, shape ``(batch, action_dim)``.
        weight: Dimensionless scale parameter ``rho > 0`` controlling decay with
            relative standard deviation (or the constant gate value when
            ``gate_mode="constant"``).
        floor: Minimum gate value in ``[0, 1]`` before mean-normalization.
        means: Unsquashed behavior policy means, shape ``(batch, action_dim)``.
        per_dim: If ``True``, return per-dimension gate of shape
            ``(batch, action_dim)``; otherwise average over dimensions to shape
            ``(batch, 1)``.
        normalize: If ``True``, normalize ``effective_stds`` and the resulting
            gate by their per-dimension batch means so the gate is scale-free
            with expected value ``1.0``.
        gate_mode: Gate computation mode: ``"std"``, ``"constant"``, or
            ``"uniform"``.

    Returns:
        Gate tensor of shape ``(batch, action_dim)`` if ``per_dim`` is ``True``,
        else ``(batch, 1)``.
    """
    out_shape = stds.shape if per_dim else (stds.shape[0], 1)
    if gate_mode == "constant":
        return torch.full(
            out_shape, fill_value=weight, dtype=stds.dtype, device=stds.device
        )
    if gate_mode == "uniform":
        return torch.empty(out_shape, dtype=stds.dtype, device=stds.device).uniform_(
            floor, 1.0
        )

    effective_stds = (
        (1.0 - torch.tanh(means).square()) * stds if means is not None else stds
    )
    if normalize:
        effective_stds = effective_stds / effective_stds.mean(
            dim=0, keepdim=True
        ).clamp(min=1e-8)
    gate_per_dim = torch.clamp(torch.exp(-effective_stds / weight), min=floor)
    if normalize:
        gate_per_dim = gate_per_dim / gate_per_dim.mean(dim=0, keepdim=True).clamp(
            min=1e-8
        )
    if per_dim:
        return gate_per_dim
    return torch.mean(gate_per_dim, dim=1, keepdim=True)
