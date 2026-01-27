#!/usr/bin/env zsh
set -euo pipefail

# Configuration constants
readonly SCRIPT_DIR=${0:A:h}
readonly REPO_ROOT="${SCRIPT_DIR}/"
readonly LOG_ROOT="data/logs_with_gate"

# Source shared utilities
source "${SCRIPT_DIR}/scs/shell_utils.sh"

# Experiment methods and datasets
readonly -a ALGOS=(cql_gr rebrac_gr td3_plus_bc_gr)
readonly -a DATASETS=(
    d4rl/hopper/medium-p-v2
    d4rl/ant/medium-p-v2
    d4rl/walker2d/medium-p-v2
    d4rl/halfcheetah/medium-p-v2
)
readonly -a SEEDS=(0 1 2 3 4)

main() {
    cd "$REPO_ROOT"
    mkdir -p "$LOG_ROOT"

    echo "Starting runs at $(timestamp)"

    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for algo in "${ALGOS[@]}"; do
                run_experiment "$algo" "$dataset" "$seed" "$LOG_ROOT"
            done
        done
    done

    echo "All runs completed at $(timestamp)"
    summarize_d3rlpy_logdata
}

main "$@"
