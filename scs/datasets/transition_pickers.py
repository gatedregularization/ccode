from __future__ import annotations

from d3rlpy.dataset.transition_pickers import (
    TransitionPickerProtocol,
    _validate_index,
)
from d3rlpy.dataset.utils import (
    create_zero_observation,
    retrieve_observation,
)
import numpy as np

from scs.datasets.components import (
    EpisodeWithPolicy,
    TransitionWithPolicy,
)


class TransitionPickerWithPolicy(TransitionPickerProtocol):
    """Transition picker with policy data.

    Adapted from ``d3rlpy.dataset.transition_pickers.BasicTransitionPicker`` to
    handle policy data.
    """

    def __call__(self, episode: EpisodeWithPolicy, index: int) -> TransitionWithPolicy:
        """Picks a transition from an episode at the given index.

        Args:
            episode: Episode to pick from.
            index: Transition index within the episode.

        Returns:
            A TransitionWithPolicy at the given index.
        """
        _validate_index(episode, index)

        observation = retrieve_observation(episode.observations, index)
        is_terminal = episode.terminated and index == episode.size() - 1
        if is_terminal:
            next_observation = create_zero_observation(observation)
            next_action = np.zeros_like(episode.actions[index])
        else:
            next_observation = retrieve_observation(episode.observations, index + 1)
            next_action = episode.actions[index + 1]

        return TransitionWithPolicy(
            observation=observation,
            action=episode.actions[index],
            reward=episode.rewards[index],
            next_observation=next_observation,
            next_action=next_action,
            terminal=float(is_terminal),
            interval=1,
            rewards_to_go=episode.rewards[index:],
            mean=episode.means[index],
            std=episode.stds[index],
        )
