from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import TYPE_CHECKING

from d3rlpy.torch_utility import (
    TorchMiniBatch,
    _compute_return_to_go,
    convert_to_torch,
    convert_to_torch_recursively,
    copy_recursively,
)
import numpy as np
import torch
from typing_extensions import Self

if TYPE_CHECKING:
    from d3rlpy.preprocessing import (
        ActionScaler,
        ObservationScaler,
        RewardScaler,
    )

    from scs.datasets.mini_batch import TransitionMiniBatchWithPolicy


@dataclass(frozen=True)
class TorchMiniBatchWithPolicy(TorchMiniBatch):
    """Torch mini-batch with policy mean and std fields.

    Adapted from ``d3rlpy.torch_utility.TorchMiniBatch`` to handle policy data.

    Attributes:
        means: Batched policy means.
        stds: Batched policy standard deviations.
    """

    means: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    stds: torch.Tensor = field(default_factory=lambda: torch.tensor([]))

    @classmethod
    def from_batch(
        cls,
        batch: TransitionMiniBatchWithPolicy,
        gamma: float,
        compute_returns_to_go: bool,
        device: str,
        observation_scaler: ObservationScaler | None = None,
        action_scaler: ActionScaler | None = None,
        reward_scaler: RewardScaler | None = None,
    ) -> TorchMiniBatchWithPolicy:
        """Creates a TorchMiniBatchWithPolicy from a numpy batch.

        Returns:
            A TorchMiniBatchWithPolicy on the specified device.
        """
        # convert numpy array to torch tensor
        observations = convert_to_torch_recursively(batch.observations, device)
        actions = convert_to_torch(batch.actions, device)
        next_actions = convert_to_torch(batch.next_actions, device)
        rewards = convert_to_torch(batch.rewards, device)
        next_observations = convert_to_torch_recursively(
            batch.next_observations, device
        )
        terminals = convert_to_torch(batch.terminals, device)
        intervals = convert_to_torch(batch.intervals, device)
        means = convert_to_torch(batch.means, device)
        stds = convert_to_torch(batch.stds, device)

        if compute_returns_to_go:
            returns_to_go = convert_to_torch(
                np.array(
                    [
                        _compute_return_to_go(
                            gamma=gamma,
                            rewards_to_go=transition.rewards_to_go,
                            reward_scaler=reward_scaler,
                        )
                        for transition in batch.transitions
                    ]
                ),
                device,
            )
        else:
            returns_to_go = torch.zeros_like(rewards)

        # apply scaler
        if (
            observation_scaler
            and isinstance(observations, torch.Tensor)
            and isinstance(next_observations, torch.Tensor)
        ):
            observations = observation_scaler.transform(observations)
            next_observations = observation_scaler.transform(next_observations)
        if action_scaler:
            actions = action_scaler.transform(actions)
            next_actions = action_scaler.transform(next_actions)
        if reward_scaler:
            rewards = reward_scaler.transform(rewards)

        return TorchMiniBatchWithPolicy(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_actions=next_actions,
            returns_to_go=returns_to_go,
            terminals=terminals,
            intervals=intervals,
            device=device,
            means=means,
            stds=stds,
            numpy_batch=batch,
        )

    def copy_(self, src: Self) -> None:
        """Copies data from another mini-batch in-place.

        Args:
            src: Source mini-batch to copy from.
        """
        assert self.device == src.device, "incompatible device"
        copy_recursively(src.observations, self.observations)
        self.actions.copy_(src.actions)
        self.rewards.copy_(src.rewards)
        copy_recursively(src.next_observations, self.next_observations)
        self.next_actions.copy_(src.next_actions)
        self.returns_to_go.copy_(src.returns_to_go)
        self.terminals.copy_(src.terminals)
        self.intervals.copy_(src.intervals)
        self.stds.copy_(src.stds)
        self.means.copy_(src.means)
