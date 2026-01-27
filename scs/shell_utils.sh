#!/usr/bin/env zsh

timestamp() {
    date +"%Y-%m-%d_%H-%M-%S"
}

safe_dataset_name() {
    local dataset="$1"
    printf '%s\n' "${dataset//\//-}"
}

log_message() {
    local message="$1"
    local log_file="$2"
    printf '[%s] %s\n' "$(timestamp)" "$message" | tee -a "$log_file"
}

summarize_d3rlpy_logdata() {
    local log_data_dir=${1:-data/d3rlpy_logdata}

    [[ -d "$log_data_dir" ]] || {
        print "No d3rlpy logs found at $log_data_dir"
        return 0
    }

    print "D3RLPY Log Summary:"
    print "==================="

    local exp_dir env_dir run_dir

    for exp_dir in "$log_data_dir"/[!.]*(/N); do
        [[ -d "$exp_dir" ]] || continue
        local exp_name=${exp_dir:t}

        for env_dir in "$exp_dir"/[!.]*(/N); do
            [[ -d "$env_dir" ]] || continue
            local env_name=${env_dir:t}

            # Skip unreadable/unsearchable dirs rather than letting the shell error out
            if [[ ! -r "$env_dir" || ! -x "$env_dir" ]]; then
                printf '%s/%s/ : skipped (permission denied)\n' "$exp_name" "$env_name"
                continue
            fi

            local -a files=("$env_dir"/[!.]*(.N))   # regular files only
            local -a dirs=("$env_dir"/[!.]*(/N))    # directories only
            printf '%s/%s/ : %d files and %d folders\n' "$exp_name" "$env_name" ${#files[@]} ${#dirs[@]}

            for run_dir in "$env_dir"/[!.]*(/N); do
                [[ -d "$run_dir" ]] || continue
                local run_name=${run_dir:t}

                if [[ ! -r "$run_dir" || ! -x "$run_dir" ]]; then
                    printf '|-- %s : skipped (permission denied)\n' "$run_name"
                    continue
                fi

                local -a rfiles=("$run_dir"/[!.]*(.N))
                local -a rdirs=("$run_dir"/[!.]*(/N))
                printf '|-- %s : %d files and %d folders\n' "$run_name" ${#rfiles[@]} ${#rdirs[@]}
            done
        done
    done
}




run_experiment() {
    local algo="$1" dataset="$2" seed="$3" log_root="$4"
    local safe_dataset run_id log_dir

    safe_dataset=$(safe_dataset_name "$dataset")
    run_id="${algo}_${safe_dataset}_s${seed}"
    log_dir="${log_root}/${run_id}"

    mkdir -p "$log_dir"

    log_message "RUN algo=${algo} dataset=${dataset} seed=${seed}" "${log_dir}/run.log"

    python -m scs.experiments."${algo}" \
        --dataset "$dataset" \
        --seed "$seed" \
        --gpu \
        --compile

    log_message "DONE ${run_id}" "${log_dir}/run.log"
}
