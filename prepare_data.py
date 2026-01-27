from __future__ import annotations

from scs.datasets.data_generation import d4rl_to_minari_with_policy
from scs.datasets.policy_model import TanhGaussianFromNormal

if __name__ == "__main__":
    for n, (env_name, dataset_name) in enumerate(
        [
            ("Ant-v2", "ant-medium-v2"),
            ("Hopper-v2", "hopper-medium-v2"),
            ("HalfCheetah-v2", "halfcheetah-medium-v2"),
            ("Walker2d-v2", "walker2d-medium-v2"),
        ]
    ):
        print(f"Adding policy to D4RL dataset: {dataset_name}")

        policy_model = TanhGaussianFromNormal.create_from_name(
            dataset_name,
            use_gpu=False,
        )
        dataset_id = f"d4rl/{dataset_name.split('-')[0]}/medium-p-v2"

        new_dataset = d4rl_to_minari_with_policy(
            dataset_name,
            dataset_id=dataset_id,
            env_name=env_name,
            policy=policy_model.get_policy,
            algorithm_name="D4RL dataset annotated with recovered policy",
            author="GatedRegularization",
            author_email="gatedregularization@protonmail.com",
            description="Test dataset generated from D4RL policy",
            code_permalink="soon",
        )

        print(f"Storing as Minari dataset: {dataset_id}")
        print(f"Completed {n + 1} of 4 datasets")
