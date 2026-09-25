from __future__ import annotations

import torch


def std_gate(stds: torch.Tensor, weight: float, floor: float = 0.0) -> torch.Tensor:
    """Computes gate values from behavior policy standard deviations.

    See Equation (11) in the paper.
    Also adds a floor to keep a minimum level of regularization as safety margin.

    Returns:
        Gate values in [0, 1], shape (batch, 1).
    """
    exp_stds = torch.exp(-stds / weight)
    gate_values = torch.mean(exp_stds, dim=1, keepdim=True)
    return torch.clamp(gate_values, min=floor, max=1.0)
