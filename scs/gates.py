from __future__ import annotations

import torch


def std_gate(stds: torch.Tensor, weight: float) -> torch.Tensor:
    """Computes gate values from behavior policy standard deviations.

    See Equation (11) in the paper.

    Returns:
        Gate values in [0, 1], shape (batch, 1).
    """
    exp_stds = torch.exp(-stds / weight)
    return torch.mean(exp_stds, dim=1, keepdim=True)


def std_gate_inverse(stds: torch.Tensor, weight: float) -> torch.Tensor:
    """Computes inverted gate values from behavior policy standard deviations.

    Returns:
        Inverted gate values in [0, 1], shape (batch, 1).
    """
    exp_stds = -torch.exp(-weight * stds) + 1
    return torch.mean(exp_stds, dim=1, keepdim=True)
