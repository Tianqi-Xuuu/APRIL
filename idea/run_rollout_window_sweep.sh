#!/bin/bash

set -euo pipefail

export PYTHONUNBUFFERED=1

IDEA_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${IDEA_DIR}/.." &>/dev/null && pwd)"
source "${IDEA_DIR}/lib.sh"
source "${REPO_ROOT}/scripts/lib/train_cleanup.sh"

MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}
SGLANG_PYTHON_ROOT=${SGLANG_PYTHON_ROOT:-}
RUNTIME_PYTHONPATH="${MEGATRON_ROOT}"
if [ -n "${SGLANG_PYTHON_ROOT}" ]; then
    RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH}:${SGLANG_PYTHON_ROOT}"
fi

MODEL_TAG=${MODEL_TAG:-qwen3-1.7b}
MODEL_SCRIPT=${MODEL_SCRIPT:-${REPO_ROOT}/scripts/models/qwen3-1.7B.sh}
HF_CHECKPOINT=${HF_CHECKPOINT:-/root/Qwen3-1.7B}
REF_LOAD=${REF_LOAD:-/root/Qwen3-1.7B_torch_dist}
LOAD_PATH=${LOAD_PATH:-/root/.slime-nonexistent-${MODEL_TAG}-idea-rollout-window-load}

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-4096}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.70}
NUM_ROLLOUT=${NUM_ROLLOUT:-20}
INPUT_DATA=${INPUT_DATA:-/root/gsm8k/data/gsm8k-train.parquet}
TASK_TAG=${TASK_TAG:-gsm8k}
WINDOW_RATIOS=${WINDOW_RATIOS:-"1.0 1.25 1.5 1.75 2.0 2.5 3.0"}
OVER_SAMPLING_BATCH_SIZES=${OVER_SAMPLING_BATCH_SIZES:-}
FORCE_RAY_RESTART=${FORCE_RAY_RESTART:-1}
RUN_ANALYSIS=${RUN_ANALYSIS:-1}
DRY_RUN=${DRY_RUN:-0}
INCLUDE_BASELINE=${INCLUDE_BASELINE:-1}
# When 1, keep existing run_status.csv and skip window jobs that already succeeded; re-run others.
RESUME_FAILED_WINDOW_SWEEP=${RESUME_FAILED_WINDOW_SWEEP:-0}
RUN_ROOT_BASE=${RUN_ROOT_BASE:-/root/APRIL/runs/idea-rollout-window-${MODEL_TAG}-${TASK_TAG}-bs${ROLLOUT_BATCH_SIZE}-len${ROLLOUT_MAX_RESPONSE_LEN}}
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}

source "${MODEL_SCRIPT}"

EXTRA_TRAIN_ARGS_ARRAY=()
if [ -n "${EXTRA_TRAIN_ARGS}" ]; then
    # shellcheck disable=SC2206
    EXTRA_TRAIN_ARGS_ARRAY=(${EXTRA_TRAIN_ARGS})
fi

if [ -n "${OVER_SAMPLING_BATCH_SIZES}" ]; then
    BATCH_SIZES="${OVER_SAMPLING_BATCH_SIZES}"
else
    # shellcheck disable=SC2086
    BATCH_SIZES=$(batch_sizes_from_ratios "${ROLLOUT_BATCH_SIZE}" ${WINDOW_RATIOS})
fi
if [ -z "${BATCH_SIZES}" ]; then
    echo "No batch sizes resolved for the window sweep." >&2
    exit 1
fi
if [ "${INCLUDE_BASELINE}" = "1" ]; then
    # shellcheck disable=SC2086
    BATCH_SIZES=$(ensure_baseline_batch_size "${ROLLOUT_BATCH_SIZE}" ${BATCH_SIZES})
fi

HAS_NVLINK=0
if command -v nvidia-smi >/dev/null 2>&1; then
    NVLINK_COUNT="$(nvidia-smi 2>/dev/null | grep -c NVLink || true)"
    NVLINK_COUNT="${NVLINK_COUNT:-0}"
    if [ "${NVLINK_COUNT}" -gt 0 ] 2>/dev/null; then
        HAS_NVLINK=1
    fi
fi

mkdir -p "${RUN_ROOT_BASE}"

COMMON_CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --load "${LOAD_PATH}"
)

COMMON_ROLLOUT_ARGS=(
   --prompt-data "${INPUT_DATA}"
   --input-key source_prompt
   --label-key answer
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout "${NUM_ROLLOUT}"
   --save-interval 1
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 0.8
   --global-batch-size "$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))"
   --balance-data
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
)

COMMON_GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
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

if [ "${DRY_RUN}" != "1" ]; then
    if [ "${SLIME_MODAL_DIRECT:-0}" = "1" ]; then
        cleanup_training_processes
        ray stop --force >/dev/null 2>&1 || true
    elif [ "${FORCE_RAY_RESTART}" = "1" ]; then
        start_fresh_ray_head "${MASTER_ADDR}" 1
    fi
fi

RUNTIME_ENV_JSON="{
  \"working_dir\": \"${REPO_ROOT}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"${RUNTIME_PYTHONPATH}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SLIME_SGLANG_DIRECT\": \"${SLIME_SGLANG_DIRECT:-0}\"
  }
}"

run_job() {
    local over_bs="$1"
    local ratio_label="$2"
    local run_name="$3"
    shift 3
    local run_root="${RUN_ROOT_BASE}/${run_name}"
    mkdir -p "${run_root}" "${run_root}/debug_rollout"

    if [ "${RESUME_FAILED_WINDOW_SWEEP:-0}" = "1" ] && [ -f "${RUN_ROOT_BASE}/run_status.csv" ]; then
        if awk -F"," -v r="${run_name}" '
            NR > 1 && $1 == r && $2 == "succeeded" && ($3 == 0 || $3 == "0") { found = 1 }
            END { exit(found ? 0 : 1) }
        ' "${RUN_ROOT_BASE}/run_status.csv" 2>/dev/null; then
            echo "=== Skip ${run_name} (already succeeded) ==="
            return
        fi
        local tmp_status
        tmp_status="$(mktemp)"
        awk -F"," -v r="${run_name}" 'NR==1 || $1!=r' "${RUN_ROOT_BASE}/run_status.csv" >"${tmp_status}"
        mv "${tmp_status}" "${RUN_ROOT_BASE}/run_status.csv"
    fi

    local train_args=(
        python3 train.py
        --debug-rollout-only
        --actor-num-nodes 1
        --actor-num-gpus-per-node 1
        --colocate
        --save "${run_root}"
        --save-debug-rollout-data "${run_root}/debug_rollout/rollout_{rollout_id:06d}.pkl"
        "${MODEL_ARGS[@]}"
        "${COMMON_CKPT_ARGS[@]}"
        "${COMMON_ROLLOUT_ARGS[@]}"
        "${COMMON_GRPO_ARGS[@]}"
        "${COMMON_PERF_ARGS[@]}"
        "${COMMON_SGLANG_ARGS[@]}"
        "${COMMON_MISC_ARGS[@]}"
        "${EXTRA_TRAIN_ARGS_ARRAY[@]}"
        "$@"
    )

    local cmd
    if [ "${SLIME_MODAL_DIRECT:-0}" = "1" ]; then
        cmd=(
            env PYTHONPATH="${RUNTIME_PYTHONPATH}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
            CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE="${HAS_NVLINK}"
            SLIME_SGLANG_DIRECT="${SLIME_SGLANG_DIRECT:-0}" SLIME_MODAL_DIRECT="${SLIME_MODAL_DIRECT:-0}"
            "${train_args[@]}"
        )
    else
        cmd=(
            ray job submit --address="http://127.0.0.1:8265"
            --runtime-env-json="${RUNTIME_ENV_JSON}"
            -- "${train_args[@]}"
        )
    fi

    echo "=== Starting ${run_name} (window=${over_bs}, ratio=${ratio_label}) ==="
    if [ "${DRY_RUN}" = "1" ]; then
        quote_cmd "${cmd[@]}"
        echo "${run_name},dry_run,0" >> "${RUN_ROOT_BASE}/run_status.csv"
        return
    fi

    set +e
    (cd "${REPO_ROOT}" && "${cmd[@]}") 2>&1 | tee "${run_root}/job_output.log"
    local rc=${PIPESTATUS[0]}
    set -e

    if [ "${rc}" -ne 0 ]; then
        echo "${run_name},failed,${rc}" | tee -a "${RUN_ROOT_BASE}/run_status.csv"
        echo "=== ${run_name} failed with exit code ${rc}, continuing ==="
    else
        echo "${run_name},succeeded,0" | tee -a "${RUN_ROOT_BASE}/run_status.csv"
        echo "=== ${run_name} succeeded ==="
    fi
}

if [ "${RESUME_FAILED_WINDOW_SWEEP:-0}" != "1" ] || [ ! -f "${RUN_ROOT_BASE}/run_status.csv" ]; then
    echo "run_name,status,exit_code" >"${RUN_ROOT_BASE}/run_status.csv"
fi

for OVER_BS in ${BATCH_SIZES}; do
    if [ "${OVER_BS}" -lt "${ROLLOUT_BATCH_SIZE}" ]; then
        echo "Skip invalid over_sampling_batch_size=${OVER_BS} < rollout_batch_size=${ROLLOUT_BATCH_SIZE}" >&2
        continue
    fi

    RATIO_LABEL=$(ratio_label_from_sizes "${ROLLOUT_BATCH_SIZE}" "${OVER_BS}")
    if [ "${OVER_BS}" -eq "${ROLLOUT_BATCH_SIZE}" ]; then
        RUN_NAME="${MODEL_TAG}-${TASK_TAG}-len${ROLLOUT_MAX_RESPONSE_LEN}-bs${ROLLOUT_BATCH_SIZE}-nonprovision"
        run_job "${OVER_BS}" "${RATIO_LABEL}" "${RUN_NAME}"
    else
        RUN_NAME="${MODEL_TAG}-${TASK_TAG}-len${ROLLOUT_MAX_RESPONSE_LEN}-bs${ROLLOUT_BATCH_SIZE}-over${OVER_BS}-${RATIO_LABEL}"
        run_job "${OVER_BS}" "${RATIO_LABEL}" "${RUN_NAME}" \
            --partial-rollout \
            --over-sampling-batch-size "${OVER_BS}"
    fi
done

if [ "${RUN_ANALYSIS}" = "1" ] && [ "${DRY_RUN}" != "1" ]; then
    python "${IDEA_DIR}/analyze_window_experiments.py" \
        --run-root-base "${RUN_ROOT_BASE}" \
        --output-dir "${RUN_ROOT_BASE}/analysis"
fi
