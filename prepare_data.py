from __future__ import annotations

import subprocess
import sys
import types

_GYM_D4RL = "0.24.1"
_GYM_D3RLPY = "0.26.2"



def _install_gym(version: str) -> None:
    print(f"Installing gym=={version}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", f"gym=={version}", "-q"],
    )


if __name__ == "__main__":
    _install_gym(_GYM_D4RL)

    # Bypass d3rlpy healthcheck that rejects gym<0.26.0
    # (needed for D4RL data loading in ``get_d4rl``).
    _fake_healthcheck = types.ModuleType("d3rlpy.healthcheck")
    _fake_healthcheck.run_healthcheck = lambda: None
    sys.modules["d3rlpy.healthcheck"] = _fake_healthcheck

    from scs.datasets.data_generation import d4rl_to_minari_with_policy
    from scs.datasets.policy_model import TanhGaussianFromNormal

    import d4rl as _

    try:
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
    finally:
        _install_gym(_GYM_D3RLPY)
