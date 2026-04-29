#!/bin/bash

cleanup_training_processes() {
    local sig
    local patterns=(
        'python3 train.py'
        '/usr/local/bin/ray job submit'
        '/usr/local/bin/ray job logs'
        'sglang.launch_server'
        'sglang_router.launch_server'
        'sgl-router'
    )

    for sig in TERM KILL; do
        for pattern in "${patterns[@]}"; do
            pkill "-${sig}" -f "${pattern}" >/dev/null 2>&1 || true
        done
        sleep 1
    done
}

start_fresh_ray_head() {
    local master_addr="$1"
    local num_gpus="$2"

    cleanup_training_processes
    ray stop --force >/dev/null 2>&1 || true
    if [ "${SLIME_MODAL_DIRECT:-0}" = "1" ]; then
        ray start --head --node-ip-address "${master_addr}" --num-gpus "${num_gpus}" \
            --disable-usage-stats --include-dashboard=False
    else
        ray start --head --node-ip-address "${master_addr}" --num-gpus "${num_gpus}" --disable-usage-stats
    fi
}
