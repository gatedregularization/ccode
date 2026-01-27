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
    parser = argparse.ArgumentParser(description="Run CQL experiment")
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

    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])

    if "medium" in args.dataset:
        conservative_weight = 10.0
    else:
        conservative_weight = 5.0

    cql = d3rlpy.algos.CQLConfig(
        actor_learning_rate=3e-5,
        critic_learning_rate=3e-4,
        temp_learning_rate=1e-4,
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder,
        batch_size=256,
        n_action_samples=10,
        alpha_learning_rate=0.0,
        conservative_weight=conservative_weight,
        compile_graph=args.compile,
    ).create(device=args.gpu)

    cql.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        logger_adapter=FileAdapterFactory(root_dir="data/d3rlpy_logdata"),
        evaluators={"reward_progression": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"CQL_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
