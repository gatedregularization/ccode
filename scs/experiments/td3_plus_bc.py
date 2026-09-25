from __future__ import annotations

import argparse

import d3rlpy
from d3rlpy.logging import FileAdapterFactory

from scs.datasets.data_loading import get_minari
from scs.experiments.defaults import (
    DATASET,
    GATE_WEIGHT,
    SEED,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TD3+BC experiment")
    parser.add_argument("--dataset", type=str, default=DATASET, help="Dataset to use")
    parser.add_argument(
        "--gate_weight", type=float, default=GATE_WEIGHT, help="Weight for the gate"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument(
        "--compile", action="store_true", help="Compile the model graph"
    )
    args = parser.parse_args()

    dataset, env = get_minari(args.dataset)

    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    td3bc = d3rlpy.algos.TD3PlusBCConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        batch_size=256,
        target_smoothing_sigma=0.2,
        target_smoothing_clip=0.5,
        alpha=2.5,
        update_actor_interval=2,
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        compile_graph=args.compile,
    ).create(device=args.gpu)

    td3bc.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        logger_adapter=FileAdapterFactory(root_dir="data/d3rlpy_logdata"),
        evaluators={"reward_progression": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"TD3PlusBC_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
