#!/usr/bin/env zsh
set -euo pipefail

# Runs stochastic Uniform(0, 1) gate ablation (mean 0.5) across all 3 algorithms
# (cql_gr, rebrac_gr, td3_plus_bc_gr), all 3 datasets, and seeds 0..4 using
# gate_mode="uniform".
# Results are written to:
#   data/d3rlpy_logdata/stoch_CQLgR_0.5_d4rl/
#   data/d3rlpy_logdata/stoch_ReBRACgR_0.5_d4rl/
#   data/d3rlpy_logdata/stoch_TD3PlusBCgR_0.5_d4rl/

readonly SCRIPT_DIR=${0:A:h}
readonly REPO_ROOT="${SCRIPT_DIR}/"
readonly LOG_ROOT="data/logs_ablation_stochastic"
readonly TRAINING_LOG_ROOT="data/logs"

source "${SCRIPT_DIR}/scs/shell_utils.sh"

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
    training_log_file="${TRAINING_LOG_ROOT}/ablation_stochastic_${batch_timestamp}.log"

    log_message "Starting runs for ablation_stochastic.sh (Uniform[0, 1] random gate, mean 0.5)" "$training_log_file"

    for seed in "${SEEDS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            run_ablation_experiment "cql_gr" "stoch_CQLgR" "$dataset" "$seed" "uniform" "0.5" "$LOG_ROOT" "$training_log_file"
            run_ablation_experiment "rebrac_gr" "stoch_ReBRACgR" "$dataset" "$seed" "uniform" "0.5" "$LOG_ROOT" "$training_log_file"
            run_ablation_experiment "td3_plus_bc_gr" "stoch_TD3PlusBCgR" "$dataset" "$seed" "uniform" "0.5" "$LOG_ROOT" "$training_log_file"
        done
    done

    log_message "DONE sweep ablation_stochastic.sh" "$training_log_file"
    summarize_d3rlpy_logdata
}

main "$@"
