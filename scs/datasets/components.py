from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
)

from d3rlpy.dataset.components import (
    Episode,
    Signature,
    Transition,
)

if TYPE_CHECKING:
    from d3rlpy.types import NDArray


@dataclass(frozen=True)
class TransitionWithPolicy(Transition):
    """Transition with policy mean and std.

    Adapted from ``d3rlpy.dataset.components.Transition``.
    Adds fields for the mean and standard deviation of the policy.

    Args:
        observation: Observation.
        action: Action
        reward: Reward. This could be a multi-step discounted return.
        next_observation: Observation at next timestep. This could be
            observation at multi-step ahead.
        next_action: Action at next timestep. This could be action at
            multi-step ahead.
        terminal: Flag of environment termination.
        interval: Timesteps between ``observation`` and ``next_observation``.
        rewards_to_go: Remaining rewards till the end of an episode, which is
            used to compute returns_to_go.
        mean: Mean of the policy.
        std: Standard deviation of the policy.
    """

    mean: NDArray  # (...)
    std: NDArray  # (...)

    @property
    def mean_signature(self) -> Signature:
        """Returns the signature for the mean array."""
        return Signature(dtype=[self.mean.dtype], shape=[self.mean.shape[1:]])

    @property
    def std_signature(self) -> Signature:
        """Returns the signature for the std array."""
        return Signature(dtype=[self.std.dtype], shape=[self.std.shape[1:]])


@dataclass(frozen=True)
class EpisodeWithPolicy(Episode):
    """Episode with policy mean and std.

    Adapted from ``d3rlpy.dataset.components.Episode``.
    Adds fields for the mean and standard deviation of the policy.

    Args:
        observations: Sequence of observations.
        actions: Sequence of actions.
        rewards: Sequence of rewards.
        terminated: Flag of environment termination.
        means: Mean of the policy.
        stds: Standard deviation of the policy.
    """

    means: NDArray
    stds: NDArray

    @property
    def mean_signature(self) -> Signature:
        """Returns the signature for the means array."""
        return Signature(dtype=[self.means.dtype], shape=[self.means.shape[1:]])

    @property
    def std_signature(self) -> Signature:
        """Returns the signature for the stds array."""
        return Signature(dtype=[self.stds.dtype], shape=[self.stds.shape[1:]])

    def serialize(self) -> dict[str, Any]:
        """Serializes the episode to a dictionary."""
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "terminated": self.terminated,
            "means": self.means,
            "stds": self.stds,
        }

    @classmethod
    def deserialize(cls, serializedData: dict[str, Any]) -> EpisodeWithPolicy:
        """Deserializes a dictionary to an EpisodeWithPolicy.

        Args:
            serializedData: Dictionary with episode data.

        Returns:
            An EpisodeWithPolicy instance.
        """
        return cls(
            observations=serializedData["observations"],
            actions=serializedData["actions"],
            rewards=serializedData["rewards"],
            terminated=serializedData["terminated"],
            means=serializedData["means"],
            stds=serializedData["stds"],
        )
