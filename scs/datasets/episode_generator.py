from __future__ import annotations

from typing import TYPE_CHECKING

from d3rlpy.dataset.episode_generator import EpisodeGenerator
from d3rlpy.dataset.utils import slice_observations
import numpy as np

from scs.datasets.components import EpisodeWithPolicy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from d3rlpy.types import (
        Float32NDArray,
        NDArray,
        ObservationSequence,
    )


class EpisodeWithPolicyGenerator(EpisodeGenerator):
    """Episode generator with policy data.

    Adapted from ``d3rlpy.dataset.episode_generator.EpisodeGenerator`` to handle
    policy data.
    Adds fields for the mean and standard deviation of the policy.

    Args:
        observations: Sequence of observations.
        actions: Sequence of actions.
        rewards: Sequence of rewards.
        terminals: Sequence of environment terminal flags.
        timeouts: Sequence of timeout flags.
        means: Sequence of policy means.
        stds: Sequence of policy standard deviations.
    """

    _means: NDArray
    _stds: NDArray

    def __init__(
        self,
        observations: ObservationSequence,
        actions: NDArray,
        rewards: Float32NDArray,
        terminals: Float32NDArray,
        means: NDArray,
        stds: NDArray,
        timeouts: Float32NDArray | None = None,
    ) -> None:
        if actions.ndim == 1:
            actions = np.reshape(actions, (-1, 1))

        if rewards.ndim == 1:
            rewards = np.reshape(rewards, (-1, 1))

        if terminals.ndim > 1:
            terminals = np.reshape(terminals, (-1))

        if means.ndim == 1:
            means = np.reshape(means, (-1, 1))

        if stds.ndim == 1:
            stds = np.reshape(stds, (-1, 1))

        if timeouts is None:
            timeouts = np.zeros_like(terminals)

        if np.sum(np.logical_and(terminals, timeouts)) == 0:
            # Use terminal flag if timeout and terminal happen at the same time
            timeouts[np.asarray(terminals, dtype=np.uint32)] = False

        assert (np.sum(terminals) + np.sum(timeouts)) > 0, (
            "No episode termination was found. Either terminals"
            " or timeouts must include non-zero values."
        )

        self._observations = observations
        self._actions = actions
        self._rewards = rewards
        self._terminals = terminals
        self._means = means
        self._stds = stds
        self._timeouts = timeouts

    def __call__(self) -> Sequence[EpisodeWithPolicy]:
        """Generates episodes by splitting at terminal/timeout flags.

        Returns:
            List of EpisodeWithPolicy instances.
        """
        start = 0
        episodes = []
        for i in range(self._terminals.shape[0]):
            if self._terminals[i] or self._timeouts[i]:
                end = i + 1
                episode = EpisodeWithPolicy(
                    observations=slice_observations(self._observations, start, end),
                    actions=self._actions[start:end],
                    rewards=self._rewards[start:end],
                    terminated=bool(self._terminals[i]),
                    means=self._means[start:end],  # Added mean
                    stds=self._stds[start:end],  # Added std
                )
                episodes.append(episode)
                start = end
        return episodes
