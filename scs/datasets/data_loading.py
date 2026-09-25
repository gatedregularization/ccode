from typing import Any

from d3rlpy.datasets import _MinariEnvType
from d3rlpy.envs import GoalConcatWrapper
import gymnasium
from gymnasium.spaces import (
    Box as GymnasiumBox,
    Dict as GymnasiumDictSpace,
)
from gymnasium.wrappers import TimeLimit as GymnasiumTimeLimit
import numpy as np

from scs.datasets.compat import MDPDatasetWithPolicy
from scs.datasets.replay_buffer import ReplayBufferWithPolicy
from scs.datasets.transition_pickers import TransitionPickerWithPolicy


def get_minari(
    env_name: str,
    render_mode: str | None = None,
    tuple_observation: bool = False,
) -> tuple[ReplayBufferWithPolicy, gymnasium.Env[Any, Any]]:
    """Returns minari dataset and environment, modified to contain policy.

    Adapted from ``d3rlpy.datasets.get_minari`` to allow handling of policy data.

    The dataset is provided through minari, that has policy parameters stored in
    its ``infos`` dictionary.

    Example:
        from d3rlpy.datasets import get_minari

        dataset, env = get_minari("door-cloned-v1")

    Args:
        env_name: environment id of minari dataset.
        transition_picker: TransitionPickerProtocol object.
        trajectory_slicer: TrajectorySlicerProtocol object.
        render_mode: Mode of rendering (``human``, ``rgb_array``).
        tuple_observation: Flag to include goals as tuple element.

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.
    """
    try:
        import minari

        _dataset = minari.load_dataset(env_name, download=True)
        env = _dataset.recover_environment()
        unwrapped_env = env.unwrapped
        unwrapped_env.render_mode = render_mode

        if isinstance(env.observation_space, GymnasiumBox):
            env_type = _MinariEnvType.BOX
        elif (
            isinstance(env.observation_space, GymnasiumDictSpace)
            and "observation" in env.observation_space.spaces
            and "desired_goal" in env.observation_space.spaces
        ):
            env_type = _MinariEnvType.GOAL_CONDITIONED
            unwrapped_env = GoalConcatWrapper(
                unwrapped_env, tuple_observation=tuple_observation
            )
        else:
            raise ValueError(f"Unsupported observation space: {env.observation_space}")

        observations = []
        actions = []
        rewards = []
        terminals = []
        timeouts = []
        means = []
        stds = []

        for ep in _dataset:
            if env_type == _MinariEnvType.BOX:
                _observations = ep.observations
            elif env_type == _MinariEnvType.GOAL_CONDITIONED:
                assert isinstance(ep.observations, dict)
                if isinstance(ep.observations["desired_goal"], dict):
                    sorted_keys = sorted(ep.observations["desired_goal"].keys())
                    goal_obs = np.concatenate(
                        [ep.observations["desired_goal"][key] for key in sorted_keys],
                        axis=-1,
                    )
                else:
                    goal_obs = ep.observations["desired_goal"]
                if tuple_observation:
                    _observations = (ep.observations["observation"], goal_obs)
                else:
                    _observations = np.concatenate(
                        [
                            ep.observations["observation"],
                            goal_obs,
                        ],
                        axis=-1,
                    )
            else:
                raise ValueError("Unsupported observation format.")
            observations.append(_observations)
            actions.append(ep.actions)
            rewards.append(ep.rewards)
            terminals.append(ep.terminations)
            timeouts.append(ep.truncations)
            means.append(ep.infos["policy_mean"])
            stds.append(ep.infos["policy_std"])

        if tuple_observation:
            stacked_observations = tuple(
                np.concatenate([observation[i] for observation in observations])
                for i in range(2)
            )
        else:
            stacked_observations = np.concatenate(observations)

        dataset = MDPDatasetWithPolicy(
            observations=stacked_observations,
            actions=np.concatenate(actions),
            rewards=np.concatenate(rewards),
            terminals=np.concatenate(terminals),
            means=np.concatenate(means),
            stds=np.concatenate(stds),
            timeouts=np.concatenate(timeouts),
            transition_picker=TransitionPickerWithPolicy(),
            trajectory_slicer=None,
        )

        #  Workaround to account for imprecise type signatures
        if env.spec is not None and env.spec.max_episode_steps is not None:
            return dataset, GymnasiumTimeLimit(
                unwrapped_env, max_episode_steps=env.spec.max_episode_steps
            )
        else:
            return dataset, unwrapped_env

    except ImportError as e:
        raise ImportError("minari is not installed.\n$ d3rlpy install minari") from e
