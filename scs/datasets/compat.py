from d3rlpy.constants import ActionSpace
from d3rlpy.dataset.trajectory_slicers import TrajectorySlicerProtocol
from d3rlpy.dataset.transition_pickers import TransitionPickerProtocol
from d3rlpy.types import (
    Float32NDArray,
    NDArray,
    ObservationSequence,
)

from scs.datasets.buffers import InfiniteBufferWithPolicy
from scs.datasets.episode_generator import EpisodeWithPolicyGenerator
from scs.datasets.replay_buffer import ReplayBufferWithPolicy
from scs.datasets.transition_pickers import TransitionPickerWithPolicy


class MDPDatasetWithPolicy(ReplayBufferWithPolicy):
    """MDP dataset with policy mean and std in transitions.

    Args:
        observations (ObservationSequence): Observations.
        actions (np.ndarray): Actions.
        rewards (np.ndarray): Rewards.
        terminals (np.ndarray): Environmental terminal flags.
        means (np.ndarray): Sequence of policy means.
        stds (np.ndarray): Sequence of policy standard deviations.
        timeouts (np.ndarray): Timeouts.
        transition_picker (Optional[TransitionPickerProtocol]):
            Transition picker implementation for Q-learning-based algorithms.
            If ``None`` is given, ``BasicTransitionPicker`` is used by default.
        trajectory_slicer (Optional[TrajectorySlicerProtocol]):
            Trajectory slicer implementation for Transformer-based algorithms.
            If ``None`` is given, ``BasicTrajectorySlicer`` is used by default.
        action_space (Optional[d3rlpy.constants.ActionSpace]):
            Action-space type.
        action_size (Optional[int]): Size of action-space. For continuous
            action-space, this represents dimension of action vectors. For
            discrete action-space, this represents the number of discrete
            actions.
    """

    _transition_picker: TransitionPickerWithPolicy

    def __init__(
        self,
        observations: ObservationSequence,
        actions: NDArray,
        rewards: Float32NDArray,
        terminals: Float32NDArray,
        means: NDArray,
        stds: NDArray,
        timeouts: Float32NDArray | None = None,
        transition_picker: TransitionPickerProtocol | None = None,
        trajectory_slicer: TrajectorySlicerProtocol | None = None,
        action_space: ActionSpace | None = None,
        action_size: int | None = None,
    ) -> None:
        episode_generator = EpisodeWithPolicyGenerator(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            timeouts=timeouts,
            means=means,
            stds=stds,
        )
        buffer = InfiniteBufferWithPolicy()
        super().__init__(
            buffer,
            episodes=episode_generator(),
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            action_space=action_space,
            action_size=action_size,
        )
