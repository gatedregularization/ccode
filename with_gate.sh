#!/usr/bin/env zsh
set -euo pipefail

# Configuration constants
readonly SCRIPT_DIR=${0:A:h}
readonly REPO_ROOT="${SCRIPT_DIR}/"
readonly LOG_ROOT="data/logs_with_gate"
readonly TRAINING_LOG_ROOT="data/logs"

# Source shared utilities
source "${SCRIPT_DIR}/scs/shell_utils.sh"

# Experiment methods and datasets
readonly -a ALGOS=(cql_gr rebrac_gr td3_plus_bc_gr)
readonly -a DATASETS=(
    d4rl/hopper/medium-p-v2
    d4rl/ant/medium-p-v2
    d4rl/halfcheetah/medium-p-v2
)
readonly -a SEEDS=(0 1 2 3 4)

main() {
    local batch_timestamp training_log_file

    cd "$REPO_ROOT"
    mkdir -p "$LOG_ROOT" "$TRAINING_LOG_ROOT"

    batch_timestamp=$(timestamp)
    training_log_file="${TRAINING_LOG_ROOT}/experiment_${batch_timestamp}.log"

    log_message "Starting runs for with_gate.sh" "$training_log_file"

    for seed in "${SEEDS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for algo in "${ALGOS[@]}"; do
                run_experiment "$algo" "$dataset" "$seed" "$LOG_ROOT" "$training_log_file"
            done
        done
    done

    log_message "DONE sweep with_gate.sh" "$training_log_file"
    summarize_d3rlpy_logdata
}

main "$@"
