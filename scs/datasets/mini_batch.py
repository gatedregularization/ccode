from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from d3rlpy.dataset.mini_batch import TransitionMiniBatch
from d3rlpy.dataset.utils import (
    cast_recursively,
    check_dtype,
    check_non_1d_array,
    stack_observations,
)
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from d3rlpy.types import Float32NDArray

    from scs.datasets.components import TransitionWithPolicy


@dataclass(frozen=True)
class TransitionMiniBatchWithPolicy(TransitionMiniBatch):
    """Mini-batch of transitions with added policy.

    Adapted from ``d3rlpy.dataset.mini_batch.TransitionMiniBatch`` to handle
    policy data.

    Args:
        observations: Batched observations.
        actions: Batched actions.
        rewards: Batched rewards.
        next_observations: Batched next observations.
        returns_to_go: Batched returns-to-go.
        terminals: Batched environment terminal flags.
        intervals: Batched timesteps between observations and next
            observations.
        transitions: List of transitions.
        means: Batched policy means.
        stds: Batched policy standard deviations.
    """

    means: Float32NDArray  # (B, ...)
    stds: Float32NDArray  # (B, ...)

    def __post_init__(self) -> None:
        super().__post_init__()
        assert check_non_1d_array(self.means)
        assert check_dtype(self.means, np.float32)
        assert check_non_1d_array(self.stds)
        assert check_dtype(self.stds, np.float32)

    @classmethod
    def from_transitions(
        cls, transitions: Sequence[TransitionWithPolicy]
    ) -> TransitionMiniBatchWithPolicy:
        r"""Constructs mini-batch from list of transitions.

        Args:
            transitions: List of transitions.

        Returns:
            Mini-batch.
        """
        observations = stack_observations(
            [transition.observation for transition in transitions]
        )
        actions = np.stack([transition.action for transition in transitions], axis=0)
        rewards = np.stack([transition.reward for transition in transitions], axis=0)
        next_observations = stack_observations(
            [transition.next_observation for transition in transitions]
        )
        next_actions = np.stack(
            [transition.next_action for transition in transitions], axis=0
        )
        terminals = np.reshape(
            np.array([transition.terminal for transition in transitions]),
            (-1, 1),
        )
        intervals = np.reshape(
            np.array([transition.interval for transition in transitions]),
            (-1, 1),
        )
        means = np.stack([transition.mean for transition in transitions], axis=0)
        stds = np.stack([transition.std for transition in transitions], axis=0)
        return TransitionMiniBatchWithPolicy(
            observations=cast_recursively(observations, np.float32),
            actions=cast_recursively(actions, np.float32),
            rewards=cast_recursively(rewards, np.float32),
            next_observations=cast_recursively(next_observations, np.float32),
            next_actions=cast_recursively(next_actions, np.float32),
            terminals=cast_recursively(terminals, np.float32),
            intervals=cast_recursively(intervals, np.float32),
            transitions=transitions,
            means=cast_recursively(means, np.float32),
            stds=cast_recursively(stds, np.float32),
        )

    @property
    def mean_shape(self) -> Sequence[int]:
        """Returns the shape of a single mean vector."""
        return self.means.shape[1:]

    @property
    def std_shape(self) -> Sequence[int]:
        """Returns the shape of a single std vector."""
        return self.stds.shape[1:]
