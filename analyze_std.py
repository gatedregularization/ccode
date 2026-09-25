from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import minari
import numpy as np
import pandas as pd

DATASETS = (
    "d4rl/hopper/medium-p-v2",
    "d4rl/halfcheetah/medium-p-v2",
    "d4rl/ant/medium-p-v2",
)
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def _load_stds(dataset_id: str) -> np.ndarray:
    """Loads the behavior policy standard deviations of a Minari dataset.

    Reads the ``policy_std`` entries stored in the episode ``infos`` in the same way
    as ``scs.datasets.data_loading.get_minari``, so the values match those seen by
    the gate during training.

    Args:
        dataset_id: Minari dataset id, e.g. ``d4rl/hopper/medium-p-v2``.

    Returns:
        Standard deviations of all transitions, shape (N, action_dim).

    Raises:
        ValueError: If the standard deviations are not aligned with the actions.
    """
    dataset = minari.load_dataset(dataset_id)
    stds = []
    actions = []
    for ep in dataset:
        stds.append(ep.infos["policy_std"])
        actions.append(ep.actions)
    stacked_stds = np.concatenate(stds).astype(np.float64)
    stacked_actions = np.concatenate(actions)
    if stacked_stds.shape != stacked_actions.shape:
        raise ValueError(
            f"{dataset_id}: std shape {stacked_stds.shape} does not match "
            f"action shape {stacked_actions.shape}"
        )
    return stacked_stds


def _summarize(values: np.ndarray) -> dict[str, Any]:
    """Computes summary statistics of a distribution of standard deviations.

    Besides mean, std, variance and percentiles, the scale-free coefficient of
    variation (std / mean) and spread ratio ((p90 - p10) / p50) are included,
    since the std scales differ across datasets and absolute spread alone is not
    comparable.

    Args:
        values: Standard deviations, shape (N,).

    Returns:
        Dictionary with keys ``sample_count``, ``mean``, ``std``, ``var``, ``cv``,
        ``min``, ``max``, ``p01`` to ``p99``, ``iqr`` and ``spread_ratio``.
    """
    mean = float(np.mean(values))
    std = float(np.std(values))
    percentiles = np.percentile(values, PERCENTILES)
    stats: dict[str, Any] = {
        "sample_count": values.shape[0],
        "mean": mean,
        "std": std,
        "var": float(np.var(values)),
        "cv": std / mean,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    for q, value in zip(PERCENTILES, percentiles):
        stats[f"p{q:02d}"] = float(value)
    stats["iqr"] = stats["p75"] - stats["p25"]
    stats["spread_ratio"] = (stats["p90"] - stats["p10"]) / stats["p50"]
    return stats


def main() -> None:
    """Parses arguments and writes the std statistics of all datasets to CSV."""
    parser = argparse.ArgumentParser(
        description="Summarize behavior policy std distributions per dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gate_analysis/std_summary.csv"),
    )
    args = parser.parse_args()

    rows = []
    for dataset_id in DATASETS:
        stds = _load_stds(dataset_id)
        action_dimension = stds.shape[1]
        # Matches the averaging over action dims in ``std_gate``.
        rows.append(
            {
                "dataset": dataset_id,
                "aggregation": "mean_over_dims",
                "action_dimension": action_dimension,
                **_summarize(stds.mean(axis=1)),
            }
        )
        for dim in range(action_dimension):
            rows.append(
                {
                    "dataset": dataset_id,
                    "aggregation": f"dim_{dim}",
                    "action_dimension": action_dimension,
                    **_summarize(stds[:, dim]),
                }
            )

    summary = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
