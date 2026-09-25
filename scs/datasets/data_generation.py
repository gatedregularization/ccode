from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import d3rlpy
import minari
from minari.data_collector import EpisodeBuffer
from minari.data_collector.callbacks import StepDataCallback
import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

    from d3rlpy.dataset.components import EpisodeBase
    from gymnasium import Env
    from minari.dataset.minari_dataset import MinariDataset
    from minari.dataset.step_data import StepData


class CollectPolicy(StepDataCallback):
    actor: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]

    def __call__(
        self,
        env: Env,
        obs: Any,
        info: dict[str, Any],
        action: Any | None = None,
        rew: Any | None = None,
        terminated: bool | None = None,
        truncated: bool | None = None,
    ) -> StepData:
        """Callback method.

        Collects the policy data in form of the mean and standard deviation of
        of the parameterized Gaussian policy. Stores the data in the `info` dict
        to ensure minimal interference with the environment step logic.
        """
        mean, std = self.actor(obs)
        step_data: StepData = {
            "action": action,
            "observation": obs,
            "reward": rew,
            "terminated": terminated,
            "truncated": truncated,
            "info": {
                "policy_mean": mean,
                "policy_std": std,
            },
        }
        return step_data

    @classmethod
    def create(
        cls,
        actor: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    ) -> type[CollectPolicy]:
        """Creates a CollectPolicy subclass bound to a given actor."""
        name = f"{cls.__name__}[{actor.__name__}]"
        return type(name, (cls,), {"actor": staticmethod(actor)})


def collector_generate_dataset(
    env: Env,
    get_action: Callable[[np.ndarray], np.ndarray],
    dataset_id: str,
    n_episodes: int = 1000,
    max_steps: int = 1000,
    collect_timesteps: int = 0,
    step_data_callback: type[StepDataCallback] = StepDataCallback,
    **kwargs: Any,
) -> tuple[MinariDataset, list[float]]:
    """Generates a Minari dataset by collecting from an environment.

    Uses minari.DataCollector to gather episodes. Stops after n_episodes
    or when collect_timesteps is reached.

    Args:
        env: Gymnasium environment.
        get_action: Policy function returning an action.
        dataset_id: Identifier for the Minari dataset.
        n_episodes: Target number of episodes.
        max_steps: Maximum steps per episode.
        collect_timesteps: Stop after this many timesteps (0 disables).
        step_data_callback: Callback class for step data collection.
        **kwargs: Passed to minari.create_dataset.

    Returns:
        Tuple of (MinariDataset, list of episode rewards).
    """
    collector_env = minari.DataCollector(
        env,
        step_data_callback=step_data_callback,
        record_infos=True,
    )
    if collect_timesteps > 0:
        n_episodes = collect_timesteps

    rewards = []
    pbar = tqdm(range(n_episodes), desc="Generating data")
    for _ in pbar:
        obs, _ = collector_env.reset()
        sum_reward = 0.0
        for _ts in range(max_steps):
            action = get_action(obs)
            next_obs, reward, terminated, truncated, _ = collector_env.step(action)
            sum_reward += float(reward)
            pbar.set_postfix(steps=collector_env._storage.total_steps)
            if terminated or truncated:
                break
            obs = next_obs
        rewards.append(sum_reward)
        if (
            collect_timesteps > 0
            and collector_env._storage.total_steps >= collect_timesteps
        ):
            break
    return collector_env.create_dataset(dataset_id, env, **kwargs), rewards


def _collect_episode(
    env: Env,
    get_action: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    episode: int,
    max_steps: int,
) -> tuple[EpisodeBuffer, int]:
    """Collects a single episode from the environment.

    Args:
        env: Gymnasium environment.
        get_action: Policy returning (action, mean, std).
        episode: Episode index for the buffer ID.
        max_steps: Maximum steps per episode.

    Returns:
        Tuple of (EpisodeBuffer, episode_length).
    """
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminations: list[bool] = []
    truncations: list[bool] = []
    infos: dict[str, list[np.ndarray]] = {
        "policy_mean": [],
        "policy_std": [],
    }
    obs, _ = env.reset()
    observations.append(obs)
    for ts in range(max_steps):
        action, mean, std = get_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(next_obs)
        actions.append(action)
        rewards.append(float(reward))
        terminations.append(terminated)
        truncations.append(truncated)
        infos["policy_mean"].append(mean)
        infos["policy_std"].append(std)
        if terminated or truncated:
            break
        obs = next_obs
    return EpisodeBuffer(
        id=episode,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        truncations=truncations,
        infos=infos,
    ), ts + 1


def collect_dataset(
    env: Env,
    get_action: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    dataset_id: str,
    n_episodes: int = 1000,
    max_steps: int = 1000,
    collect_timesteps: int = 0,
    **kwargs: Any,
) -> tuple[MinariDataset, list[float]]:
    """Collects multiple episodes into a Minari dataset.

    Args:
        env: Gymnasium environment.
        get_action: Policy returning (action, mean, std).
        dataset_id: Identifier for the Minari dataset.
        n_episodes: Target number of episodes.
        max_steps: Maximum steps per episode.
        collect_timesteps: Stop after this many timesteps (0 disables).
        **kwargs: Passed to minari.create_dataset_from_buffers.

    Returns:
        Tuple of (MinariDataset, list of episode rewards).
    """
    max_episodes = collect_timesteps if collect_timesteps > 0 else n_episodes
    timesteps = 0
    episode_rewards: list[float] = []
    episodes: list[EpisodeBuffer] = []
    pbar = tqdm(range(max_episodes), desc="Generating data")
    for episode in pbar:
        episode_data, len_episode = _collect_episode(
            env,
            get_action,
            episode,
            max_steps,
        )
        timesteps += len_episode
        pbar.set_postfix(steps=timesteps)
        episodes.append(episode_data)
        episode_rewards.append(float(np.sum(episode_data.rewards)))
        if collect_timesteps > 0 and timesteps >= collect_timesteps:
            break
    return minari.create_dataset_from_buffers(
        dataset_id,
        episodes,
        env,
        **kwargs,
    ), episode_rewards


def _d4rl_episode_to_minari_buffer(
    n_episode: int,
    episode: EpisodeBase,
    policy: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> EpisodeBuffer:
    """Converts a d3rlpy episode to a Minari EpisodeBuffer.

    Args:
        n_episode: Episode index for the buffer ID.
        episode: d3rlpy EpisodeBase to convert.
        policy: Callable returning (mean, std) for observations.

    Returns:
        An EpisodeBuffer with policy mean/std in infos.
    """
    observations = np.asarray(episode.observations)
    means, stds = policy(observations)
    terminations = np.zeros(episode.actions.shape[0], dtype=bool)
    truncations = np.zeros(episode.actions.shape[0], dtype=bool)
    if episode.terminated:
        terminations[-1] = True
    else:
        truncations[-1] = True
    return EpisodeBuffer(
        id=n_episode,
        observations=list(observations),
        actions=list(episode.actions),
        rewards=list(episode.rewards),
        terminations=list(terminations),
        truncations=list(truncations),
        infos={
            "policy_mean": list(means),
            "policy_std": list(stds),
        },
    )


def d4rl_to_minari_with_policy(
    dataset: str,
    dataset_id: str,
    env_name: str,
    policy: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    **kwargs: Any,
) -> MinariDataset:
    """Converts a D4RL dataset to Minari format with policy annotations.

    Args:
        dataset: D4RL dataset name.
        dataset_id: Identifier for the new Minari dataset.
        env_name: Gymnasium environment name.
        policy: Callable returning (mean, std) for observations.
        **kwargs: Passed to minari.create_dataset_from_buffers.

    Returns:
        The created MinariDataset.
    """
    raw_dataset, _env_old = d3rlpy.datasets.get_d4rl(dataset)
    n_episodes = raw_dataset.size()
    new_dataset = []
    for n_episode, episode in tqdm(
        enumerate(raw_dataset.episodes),
        total=n_episodes,
    ):
        episode_buffer = _d4rl_episode_to_minari_buffer(n_episode, episode, policy)
        new_dataset.append(episode_buffer)
    return minari.create_dataset_from_buffers(
        dataset_id,
        new_dataset,
        env=env_name,
        eval_env=env_name,
        **kwargs,
    )
