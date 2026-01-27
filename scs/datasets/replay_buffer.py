from __future__ import annotations

from typing import TYPE_CHECKING

from d3rlpy.dataset.replay_buffer import ReplayBuffer
from d3rlpy.dataset.trajectory_slicers import (
    BasicTrajectorySlicer,
    TrajectorySlicerProtocol,
)
from d3rlpy.dataset.writers import (
    BasicWriterPreprocess,
    WriterPreprocessProtocol,
)
import numpy as np

from scs.datasets.mini_batch import TransitionMiniBatchWithPolicy
from scs.datasets.transition_pickers import TransitionPickerWithPolicy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from d3rlpy.constants import ActionSpace
    from d3rlpy.dataset.components import Signature
    from d3rlpy.dataset.transition_pickers import TransitionPickerProtocol
    from d3rlpy.types import GymEnv

    from scs.datasets.buffers import InfiniteBufferWithPolicy
    from scs.datasets.components import (
        EpisodeWithPolicy,
        TransitionWithPolicy,
    )


class ReplayBufferWithPolicy(ReplayBuffer):
    """Replay buffer for experience replay, expanded to contain policy.

    Adapted from ``d3rlpy.dataset.replay_buffer.ReplayBuffer`` to handle
    policy data.

    To determine shapes of observations, actions and rewards, one of
    ``episodes``, ``env``, or signatures must be provided.

    Example:
        from d3rlpy.dataset import FIFOBuffer, ReplayBuffer, Signature

        buffer = FIFOBuffer(limit=1000000)

        # initialize with pre-collected episodes
        replay_buffer = ReplayBuffer(buffer=buffer, episodes=<episodes>)

        # initialize with Gym
        replay_buffer = ReplayBuffer(buffer=buffer, env=<env>)

        # initialize with manually specified signatures
        replay_buffer = ReplayBuffer(
            buffer=buffer,
            observation_signature=Signature(dtype=[<dtype>], shape=[<shape>]),
            action_signature=Signature(dtype=[<dtype>], shape=[<shape>]),
            reward_signature=Signature(dtype=[<dtype>], shape=[<shape>]),
        )

    Args:
        buffer (d3rlpy.dataset.BufferProtocol): Buffer implementation.
        transition_picker (Optional[d3rlpy.dataset.TransitionPickerProtocol]):
            Transition picker implementation for Q-learning-based algorithms.
            If ``None`` is given, ``TransitionPickerWithPolicy`` is used by default.
        trajectory_slicer (Optional[d3rlpy.dataset.TrajectorySlicerProtocol]):
            Trajectory slicer implementation for Transformer-based algorithms.
            If ``None`` is given, ``BasicTrajectorySlicer`` is used by default.
        writer_preprocessor (Optional[d3rlpy.dataset.WriterPreprocessProtocol]):
            Writer preprocessor implementation. If ``None`` is given,
            ``BasicWriterPreprocess`` is used by default.
        episodes (Optional[Sequence[d3rlpy.dataset.EpisodeBase]]):
            List of episodes to initialize replay buffer.
        env (Optional[GymEnv]): Gym environment to extract shapes of
            observations and action.
        observation_signature (Optional[d3rlpy.dataset.Signature]):
            Signature of observation.
        action_signature (Optional[d3rlpy.dataset.Signature]):
            Signature of action.
        reward_signature (Optional[d3rlpy.dataset.Signature]):
            Signature of reward.
        action_space (Optional[d3rlpy.constants.ActionSpace]):
            Action-space type.
        action_size (Optional[int]): Size of action-space. For continuous
            action-space, this represents dimension of action vectors. For
            discrete action-space, this represents the number of discrete
            actions.
        cache_size (int): Size of cache to record active episode history used
            for online training. ``cache_size`` needs to be greater than the
            maximum possible episode length.
        write_at_termination (bool): Flag to write experiences to the buffer at
            the end of an episode all at once.
    """

    # TODO: Make this more elegant; Overrides type signatures
    _buffer: InfiniteBufferWithPolicy
    _transition_picker: TransitionPickerWithPolicy

    def __init__(
        self,
        buffer: InfiniteBufferWithPolicy,
        transition_picker: TransitionPickerProtocol | None = None,
        trajectory_slicer: TrajectorySlicerProtocol | None = None,
        writer_preprocessor: WriterPreprocessProtocol | None = None,
        episodes: Sequence[EpisodeWithPolicy] | None = None,
        env: GymEnv | None = None,
        observation_signature: Signature | None = None,
        action_signature: Signature | None = None,
        reward_signature: Signature | None = None,
        action_space: ActionSpace | None = None,
        action_size: int | None = None,
        cache_size: int = 10000,
        write_at_termination: bool = False,
    ) -> None:
        super().__init__(
            buffer=buffer,
            transition_picker=transition_picker or TransitionPickerWithPolicy(),
            trajectory_slicer=trajectory_slicer or BasicTrajectorySlicer(),
            writer_preprocessor=writer_preprocessor or BasicWriterPreprocess(),
            episodes=episodes,
            env=env,
            observation_signature=observation_signature,
            action_signature=action_signature,
            reward_signature=reward_signature,
            action_space=action_space,
            action_size=action_size,
            cache_size=cache_size,
            write_at_termination=write_at_termination,
        )
        self._rng: np.random.Generator = np.random.default_rng()

    def sample_transition(self) -> TransitionWithPolicy:
        """Samples a single transition uniformly from the buffer."""
        index = self._rng.integers(self._buffer.transition_count).astype(int)
        episode, transition_index = self._buffer[index]
        return self._transition_picker(episode, transition_index)

    def sample_transition_batch(self, batch_size: int) -> TransitionMiniBatchWithPolicy:
        """Samples a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            A TransitionMiniBatchWithPolicy of the sampled transitions.
        """
        return TransitionMiniBatchWithPolicy.from_transitions(
            [self.sample_transition() for _ in range(batch_size)]
        )
