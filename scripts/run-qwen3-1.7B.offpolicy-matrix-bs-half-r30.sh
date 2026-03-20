#!/bin/bash

set -euo pipefail

export PYTHONUNBUFFERED=1

N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-4096}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.70}
NUM_ROLLOUT=${NUM_ROLLOUT:-30}
EVAL_INTERVAL=${EVAL_INTERVAL:-5}
INPUT_DATA=${INPUT_DATA:-/root/gsm8k/data/gsm8k-train.parquet}
EVAL_DATA=${EVAL_DATA:-/root/gsm8k/data/gsm8k-test.parquet}
RUN_ROOT_BASE=${RUN_ROOT_BASE:-/root/APRIL/runs/offpolicy-matrix-bs-half-r${NUM_ROLLOUT}-qwen3-1.7b-len${ROLLOUT_MAX_RESPONSE_LEN}}
FORCE_RAY_RESTART=${FORCE_RAY_RESTART:-1}
WATCHDOG_ENABLED=${WATCHDOG_ENABLED:-1}
WATCHDOG_POLL_SECONDS=${WATCHDOG_POLL_SECONDS:-20}
WATCHDOG_STALL_SECONDS=${WATCHDOG_STALL_SECONDS:-180}
WATCHDOG_MAX_RESTARTS=${WATCHDOG_MAX_RESTARTS:-5}
WATCHDOG_HEALTHCHECK_GRACE_SECONDS=${WATCHDOG_HEALTHCHECK_GRACE_SECONDS:-180}
WATCHDOG_FAULT_INJECT_GROUP=${WATCHDOG_FAULT_INJECT_GROUP:-}
WATCHDOG_FAULT_INJECT_DELAY_SECONDS=${WATCHDOG_FAULT_INJECT_DELAY_SECONDS:-0}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-1.7B.sh"

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l || true)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

mkdir -p "${RUN_ROOT_BASE}"

COMMON_CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-1.7B
   --ref-load /root/Qwen3-1.7B_torch_dist
)

COMMON_DATA_ARGS=(
   --prompt-data "${INPUT_DATA}"
   --input-key source_prompt
   --label-key answer
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout "${NUM_ROLLOUT}"
   --save-interval "${EVAL_INTERVAL}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 0.8
   --balance-data
   --eval-interval "${EVAL_INTERVAL}"
   --eval-prompt-data gsm8k_test "${EVAL_DATA}"
   --n-samples-per-eval-prompt 4
   --eval-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --eval-top-p 0.7
)

COMMON_PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

COMMON_GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

COMMON_SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
   --sglang-disable-cuda-graph
)

COMMON_MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

start_ray_cluster() {
    ray stop --force || true
    ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 1 --disable-usage-stats
}

stop_job_and_logs() {
    local submission_id="$1"
    local log_pid="${2:-}"

    if [ -n "${submission_id}" ]; then
        ray job stop "${submission_id}" >/dev/null 2>&1 || true
    fi

    if [ -n "${log_pid}" ] && kill -0 "${log_pid}" >/dev/null 2>&1; then
        kill "${log_pid}" >/dev/null 2>&1 || true
        wait "${log_pid}" >/dev/null 2>&1 || true
    fi
}

follow_job_logs() {
    local submission_id="$1"
    local run_root="$2"

    ray job logs "${submission_id}" -f >>"${run_root}/job_output.log" 2>&1 &
    echo $!
}

get_worker_pid_on_port_10000() {
    ss -ltnp 2>/dev/null | awk '
        /:10000 / {
            if (match($0, /pid=([0-9]+)/, m)) {
                print m[1];
                exit;
            }
        }
    '
}

watch_job() {
    local submission_id="$1"
    local run_name="$2"
    local run_root="$3"
    local log_pid="$4"
    local driver_log="/tmp/ray/session_latest/logs/job-driver-${submission_id}.log"
    local start_ts
    local last_progress_ts
    local last_size
    local worker_started=0
    local injected=0

    start_ts=$(date +%s)
    last_progress_ts="${start_ts}"
    last_size=$(stat -c %s "${run_root}/job_output.log" 2>/dev/null || echo 0)

    while true; do
        sleep "${WATCHDOG_POLL_SECONDS}"

        local now
        local status_output
        local current_size
        now=$(date +%s)
        status_output=$(ray job status "${submission_id}" 2>&1 || true)

        if echo "${status_output}" | grep -q "Status for job '.*': SUCCEEDED"; then
            wait "${log_pid}" >/dev/null 2>&1 || true
            return 0
        fi

        if echo "${status_output}" | grep -q "Status for job '.*': FAILED\|Status for job '.*': STOPPED"; then
            echo "[watchdog] ${run_name}: Ray job entered terminal failure state."
            return 1
        fi

        current_size=$(stat -c %s "${run_root}/job_output.log" 2>/dev/null || echo 0)
        if [ "${current_size}" -gt "${last_size}" ]; then
            last_progress_ts="${now}"
            last_size="${current_size}"
        fi

        if grep -q "Uvicorn running on http://.*:10000" "${run_root}/job_output.log" 2>/dev/null; then
            worker_started=1
        fi

        if [ "${worker_started}" = "1" ] && [ "${WATCHDOG_FAULT_INJECT_GROUP}" = "${run_name}" ] \
            && [ "${WATCHDOG_FAULT_INJECT_DELAY_SECONDS}" -gt 0 ] && [ "${injected}" = "0" ] \
            && [ $((now - start_ts)) -ge "${WATCHDOG_FAULT_INJECT_DELAY_SECONDS}" ]; then
            local worker_pid
            worker_pid=$(get_worker_pid_on_port_10000 || true)
            if [ -n "${worker_pid}" ]; then
                echo "[watchdog] ${run_name}: fault-inject killing rollout worker pid ${worker_pid} on :10000"
                kill -9 "${worker_pid}" >/dev/null 2>&1 || true
                injected=1
            fi
        fi

        if [ "${worker_started}" = "1" ] && ! ss -ltn 2>/dev/null | grep -q ':10000 '; then
            echo "[watchdog] ${run_name}: rollout worker on :10000 disappeared."
            return 2
        fi

        if [ -f "${driver_log}" ] && grep -q "Removing failed worker\|PoisonError" "${driver_log}" 2>/dev/null; then
            echo "[watchdog] ${run_name}: detected router failed-worker/PoisonError pattern."
            return 3
        fi

        if [ $((now - last_progress_ts)) -gt "${WATCHDOG_STALL_SECONDS}" ]; then
            echo "[watchdog] ${run_name}: no log progress for ${WATCHDOG_STALL_SECONDS}s."
            return 4
        fi
    done
}

run_job() {
    local run_name="$1"
    local rollout_batch_size="$2"
    shift 2

    local run_root="${RUN_ROOT_BASE}/${run_name}"
    local load_path="/root/.slime-nonexistent-qwen3-1.7B-offpolicy-matrix-bs-half-load"
    mkdir -p "${run_root}"

    if [ -f "${run_root}/latest_checkpointed_iteration.txt" ]; then
        load_path="${run_root}"
        echo "=== Resuming ${run_name} from ${load_path} ==="
    else
        echo "=== Starting ${run_name} from pretrained weights ==="
    fi

    if [ "${FORCE_RAY_RESTART}" = "1" ]; then
        start_ray_cluster
    fi

    local attempt=0
    while true; do
        local submission_id="${run_name}-$(date +%s)"
        local log_pid=""
        local submit_output=""

        echo "=== ${run_name}: launch attempt ${attempt} ===" | tee -a "${run_root}/job_output.log"

        submit_output=$(
            ray job submit --no-wait \
               --submission-id "${submission_id}" \
               --address="http://127.0.0.1:8265" \
               --runtime-env-json="${RUNTIME_ENV_JSON}" \
               -- python3 train.py \
               --actor-num-nodes 1 \
               --actor-num-gpus-per-node 1 \
               --colocate \
               --save "${run_root}" \
               --load "${load_path}" \
               --rollout-batch-size "${rollout_batch_size}" \
               --global-batch-size "$((rollout_batch_size * N_SAMPLES_PER_PROMPT))" \
               "${MODEL_ARGS[@]}" \
               "${COMMON_CKPT_ARGS[@]}" \
               "${COMMON_DATA_ARGS[@]}" \
               "${COMMON_GRPO_ARGS[@]}" \
               "${OPTIMIZER_ARGS[@]}" \
               "${COMMON_PERF_ARGS[@]}" \
               "${COMMON_SGLANG_ARGS[@]}" \
               "${COMMON_MISC_ARGS[@]}" \
               "$@" 2>&1
        )
        echo "${submit_output}" | tee -a "${run_root}/job_output.log"

        log_pid=$(follow_job_logs "${submission_id}" "${run_root}")

        if [ "${WATCHDOG_ENABLED}" != "1" ]; then
            wait "${log_pid}" >/dev/null 2>&1 || true
            return 0
        fi

        if watch_job "${submission_id}" "${run_name}" "${run_root}" "${log_pid}"; then
            echo "=== ${run_name}: completed successfully ===" | tee -a "${run_root}/job_output.log"
            return 0
        fi

        stop_job_and_logs "${submission_id}" "${log_pid}"
        attempt=$((attempt + 1))
        if [ "${attempt}" -gt "${WATCHDOG_MAX_RESTARTS}" ]; then
            echo "=== ${run_name}: exceeded watchdog restart budget (${WATCHDOG_MAX_RESTARTS}) ===" | tee -a "${run_root}/job_output.log"
            return 1
        fi

        echo "=== ${run_name}: restarting after watchdog trigger (attempt ${attempt}) ===" | tee -a "${run_root}/job_output.log"
        start_ray_cluster
    done
}

run_job "qwen3-1.7b-train-non-provision-bs24-r${NUM_ROLLOUT}" 24

run_job "qwen3-1.7b-train-provision-2p0x-bs12-r${NUM_ROLLOUT}" 12 \
    --partial-rollout \
    --over-sampling-batch-size 24

run_job "qwen3-1.7b-train-provision-3p0x-bs8-r${NUM_ROLLOUT}" 8 \
    --partial-rollout \
    --over-sampling-batch-size 24
