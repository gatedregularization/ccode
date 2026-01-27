from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def get_experiments(results_path: Path) -> dict[Path, list[Path]]:
    """Returns a dictionary mapping from experiments to the seeded runs.

    Experiments as well as the seeded runs are represented as the paths to their
    respective directories. This structure allows to store the processed results
    across seeded experiments to be saved in the same experiment directory.

    Tailored to the specific structure of the stored results.
    """
    experiments_seeds: dict[Path, list[Path]] = {}
    for algorithm_path in results_path.iterdir():
        if not algorithm_path.is_dir():
            continue
        for environment_path in algorithm_path.iterdir():
            if not environment_path.is_dir():
                continue
            experiments_seeds[environment_path] = [
                seed_path
                for seed_path in environment_path.iterdir()
                if seed_path.is_dir()
            ]
    return experiments_seeds


def aggregate_results(seed_paths: list[Path]) -> pd.DataFrame:
    """Aggregates reward progression data from multiple seeded experiment runs.

    Reads reward_progression.csv from each seed directory and combines them into
    a single DataFrame with seed identifiers extracted from directory names.

    Args:
        seed_paths: Paths to seed-specific experiment directories, each containing
            a reward_progression.csv file.

    Returns:
        DataFrame with columns ["timestep", "reward", "seed"] containing the
        concatenated data from all seeds.
    """
    exerpiment_runs = []
    for seed_path in seed_paths:
        data = pd.read_csv(
            seed_path / "reward_progression.csv",
            names=["timestep", "reward"],
            header=None,
        )
        data["seed"] = int(seed_path.name.split("_")[-2])
        exerpiment_runs.append(data)
    return pd.concat(exerpiment_runs)


def construct_rolling_stats(
    experiment_rewards: pd.DataFrame, window_size: int = 1
) -> pd.DataFrame:
    """Computes rolling statistics over reward data across timesteps.

    Groups timesteps into non-overlapping windows and computes summary statistics
    (mean, std, variance, standard error) over all reward values within each window.

    When window_size=1, this computes per-timestep statistics across seeds.

    Args:
        experiment_rewards: DataFrame with columns ["timestep", "reward", "seed"].
        window_size: Number of consecutive timesteps to include in each window.

    Returns:
        DataFrame with columns ["timestep", "mean", "std", "variance",
        "standard_error", "n_seeds"]. The timestep column contains the last
        timestep in each window.
    """
    timesteps = sorted(experiment_rewards["timestep"].unique())
    rolling_stats: dict[str, list[float]] = {
        "timestep": [],
        "mean": [],
        "std": [],
        "variance": [],
        "standard_error": [],
        "n_seeds": [],
    }
    for t in range(0, len(timesteps), window_size):
        window = timesteps[t : t + window_size]
        window_indices = experiment_rewards["timestep"].isin(window)
        window_data = experiment_rewards[window_indices]
        window_rewards = window_data["reward"].values
        rolling_stats["timestep"].append(timesteps[t + window_size - 1])
        rolling_stats["mean"].append(np.mean(window_rewards))
        rolling_stats["std"].append(np.std(window_rewards, ddof=1))
        rolling_stats["variance"].append(np.var(window_rewards, ddof=1))
        rolling_stats["standard_error"].append(
            rolling_stats["std"][-1] / np.sqrt(len(window_rewards))
        )
        rolling_stats["n_seeds"].append(len(window_data["seed"].unique()))
    return pd.DataFrame(rolling_stats)


def process_algorithm_experiments(
    experiment_direcotry: Path,
    experiment_paths: list[Path],
    window_size: int,
) -> None:
    """Processes and saves aggregated statistics for a single experiment.

    Aggregates reward progression data across seeds, computes per-timestep
    statistics, and optionally computes rolling statistics with a larger window.
    Results are saved as CSV files in the experiment directory.

    Args:
        experiment_direcotry: Path to the experiment directory (algorithm/environment).
        experiment_paths: Paths to individual seed run directories of the experiment.
        window_size: Window size for rolling statistics. If greater than 1, an
            additional rolling statistics file is generated.
    """
    # Extract naming components from paths
    algorithm_and_data = experiment_direcotry.parts[-2]
    environment = experiment_direcotry.parts[-1]
    behvarioal_policy = experiment_paths[0].name.split("_")[0]
    experiment_name = f"{algorithm_and_data}_{environment}_{behvarioal_policy}"

    print(f"Processing experiment: {experiment_name}")
    aggregated_results = aggregate_results(experiment_paths)
    ar_path = (
        experiment_direcotry / f"reward_progression_{experiment_name}_aggregate.csv"
    )
    aggregated_results.to_csv(ar_path, index=False)
    print(f"    Saved aggregated results to: {ar_path}")

    reward_stats = construct_rolling_stats(aggregated_results, window_size=1)
    rs_path = experiment_direcotry / f"reward_progression_{experiment_name}_stats.csv"
    reward_stats.to_csv(
        rs_path,
        index=False,
    )
    print(f"    Saved reward statistics to: {rs_path}")

    if window_size > 1:
        reward_rolling_stats = construct_rolling_stats(
            aggregated_results, window_size=window_size
        )
        rrs_path = (
            experiment_direcotry
            / f"reward_progression_{experiment_name}_rolling_{window_size}k.csv"
        )
        reward_rolling_stats.to_csv(
            rrs_path,
            index=False,
        )
        print(
            f"    Saved reward rolling statistics (window {window_size}k) to: "
            f"{rrs_path}"
        )


def main() -> None:
    """Parses arguments and processes all experiments in the results directory."""
    parser = argparse.ArgumentParser(
        description="Process experiment results and compute statistics"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="data/d3rlpy_logdata",
        help="Path to the directory containing experiment results",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=1,
        help="Window size for rolling statistics",
    )
    args = parser.parse_args()

    print(f"Processing experiment results in {args.results_path}")
    results_path = Path(args.results_path)
    experiments = get_experiments(results_path)
    for experiment_dir, seed_paths in experiments.items():
        process_algorithm_experiments(
            experiment_dir, seed_paths, window_size=args.window_size
        )
    print("Processing complete.")


if __name__ == "__main__":
    main()
