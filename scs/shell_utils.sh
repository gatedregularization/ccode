#!/usr/bin/env zsh

zmodload zsh/datetime

timestamp() {
    date +"%Y-%m-%d_%H-%M-%S"
}

safe_dataset_name() {
    local dataset="$1"
    printf '%s\n' "${dataset//\//-}"
}

log_message() {
    local message="$1"
    shift
    local -a log_files=("$@")
    local line
    line=$(printf '[%s] %s' "$(timestamp)" "$message")

    if (( ${#log_files[@]} > 0 )); then
        printf '%s\n' "$line" | tee -a "${log_files[@]}"
    else
        printf '%s\n' "$line"
    fi
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
    local algo="$1" dataset="$2" seed="$3" log_root="$4" batch_log_file="${5:-}"
    local safe_dataset run_id log_dir
    local start_epoch end_epoch elapsed_seconds elapsed_hms exit_code
    local -a log_targets

    safe_dataset=$(safe_dataset_name "$dataset")
    run_id="${algo}_${safe_dataset}_s${seed}"
    log_dir="${log_root}/${run_id}"
    log_targets=("${log_dir}/run.log")

    mkdir -p "$log_dir"
    if [[ -n "$batch_log_file" ]]; then
        mkdir -p "${batch_log_file:h}"
        log_targets+=("$batch_log_file")
    fi

    log_message "RUN algo=${algo} dataset=${dataset} seed=${seed}" "${log_targets[@]}"

    start_epoch=$EPOCHSECONDS

    if python -m scs.experiments."${algo}" \
        --dataset "$dataset" \
        --seed "$seed" \
        --gpu \
        --compile; then
        exit_code=0
    else
        exit_code=$?
    fi

    end_epoch=$EPOCHSECONDS
    elapsed_seconds=$((end_epoch - start_epoch))
    elapsed_hms=$(printf '%02d:%02d:%02d' \
        $((elapsed_seconds / 3600)) \
        $(((elapsed_seconds % 3600) / 60)) \
        $((elapsed_seconds % 60)))

    log_message "TIME ${run_id} elapsed=${elapsed_hms} (${elapsed_seconds}s)" "${log_targets[@]}"

    if (( exit_code == 0 )); then
        log_message "DONE ${run_id}" "${log_targets[@]}"
    else
        log_message "FAIL ${run_id} exit_code=${exit_code}" "${log_targets[@]}"
    fi

    return "$exit_code"
}
