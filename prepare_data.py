from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import types

_GYM_D4RL_VERSION = "0.24.1"
_GYM_D4RL_HASH = "9b23bbaa96ba0771a16a9f80062edb79b6f176ffdc84f39ea4645b03a8b72291"
_GYM_D3RLPY_VERSION = "0.26.2"
_GYM_D3RLPY_HASH = "e0d882f4b54f0c65f203104c24ab8a38b039f1289986803c7d02cdbe214fbcc4"


def _install_pinned_gym(version: str, sha256_hash: str) -> None:
    print(f"Installing gym=={version}")
    requirement_text = f"gym=={version} --hash=sha256:{sha256_hash}\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as req_file:
        req_file.write(requirement_text)
        requirements_path = req_file.name

    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise RuntimeError("uv is required but was not found in PATH")

    env = os.environ.copy()
    env["UV_NO_CACHE"] = "1"
    print("Using UV_NO_CACHE=1 for package install")

    try:
        subprocess.check_call(
            [
                uv_executable,
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-deps",
                "--require-hashes",
                "-r",
                requirements_path,
                "-q",
            ],
            env=env,
        )
    finally:
        os.remove(requirements_path)


if __name__ == "__main__":
    _install_pinned_gym(_GYM_D4RL_VERSION, _GYM_D4RL_HASH)

    # Bypass d3rlpy healthcheck that rejects gym<0.26.0
    # (needed for D4RL data loading in ``get_d4rl``).
    _fake_healthcheck = types.ModuleType("d3rlpy.healthcheck")
    _fake_healthcheck.run_healthcheck = lambda: None
    sys.modules["d3rlpy.healthcheck"] = _fake_healthcheck

    from scs.datasets.data_generation import d4rl_to_minari_with_policy
    from scs.datasets.policy_model import TanhGaussianFromNormal

    try:
        for n, (env_name, dataset_name) in enumerate(
            [
                ("Ant-v2", "ant-medium-v2"),
                ("Hopper-v2", "hopper-medium-v2"),
                ("HalfCheetah-v2", "halfcheetah-medium-v2"),
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
        _install_pinned_gym(_GYM_D3RLPY_VERSION, _GYM_D3RLPY_HASH)
