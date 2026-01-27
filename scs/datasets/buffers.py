from collections.abc import Sequence

from d3rlpy.dataset.buffers import BufferProtocol

from scs.datasets.components import EpisodeWithPolicy


class InfiniteBufferWithPolicy(BufferProtocol):
    """Buffer with unlimited capacity and modified type signatures.

    Adapted from ``d3rlpy.dataset.buffers.InfiniteBuffer`` to handle data including
    the policy.
    """

    _transitions: list[tuple[EpisodeWithPolicy, int]]
    _episodes: list[EpisodeWithPolicy]

    def __init__(self) -> None:
        self._transitions = []
        self._episodes = []
        self._transition_count = 0

    def append(self, episode: EpisodeWithPolicy, index: int) -> None:
        self._transitions.append((episode, index))
        if not self._episodes or episode is not self._episodes[-1]:
            self._episodes.append(episode)

    @property
    def episodes(self) -> Sequence[EpisodeWithPolicy]:
        """Returns all stored episodes."""
        return self._episodes

    @property
    def transition_count(self) -> int:
        return len(self._transitions)

    def __len__(self) -> int:
        return len(self._transitions)

    def __getitem__(self, index: int) -> tuple[EpisodeWithPolicy, int]:
        return self._transitions[index]
