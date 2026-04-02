#!/bin/bash

set -euo pipefail

export PYTHONUNBUFFERED=1

IDEA_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${IDEA_DIR}/.." &>/dev/null && pwd)"
source "${IDEA_DIR}/lib.sh"
source "${REPO_ROOT}/scripts/lib/train_cleanup.sh"

MODEL_TAG=${MODEL_TAG:-qwen2.5-3b}
MODEL_SCRIPT=${MODEL_SCRIPT:-${REPO_ROOT}/scripts/models/qwen2.5-3B.sh}
HF_CHECKPOINT=${HF_CHECKPOINT:-/root/Qwen2.5-3B}
REF_LOAD=${REF_LOAD:-/root/Qwen2.5-3B_torch_dist}
BASE_LOAD_PATH=${BASE_LOAD_PATH:-/root/.slime-nonexistent-${MODEL_TAG}-idea-train-window-load}

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-4096}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.70}
NUM_ROLLOUT=${NUM_ROLLOUT:-30}
EVAL_INTERVAL=${EVAL_INTERVAL:-5}
INPUT_DATA=${INPUT_DATA:-/root/math_level12/data/math-level4-train.stepthink.parquet}
EVAL_DATA=${EVAL_DATA:-/root/math_level12/data/math-level4-test.subset128.seed1234.parquet}
TASK_TAG=${TASK_TAG:-math-level4}
EVAL_NAME=${EVAL_NAME:-math_level4_eval}
WINDOW_RATIOS=${WINDOW_RATIOS:-"1.0 1.5 2.0 2.5 3.0"}
OVER_SAMPLING_BATCH_SIZES=${OVER_SAMPLING_BATCH_SIZES:-}
FORCE_RAY_RESTART=${FORCE_RAY_RESTART:-1}
RUN_ANALYSIS=${RUN_ANALYSIS:-1}
DRY_RUN=${DRY_RUN:-0}
RUN_ROOT_BASE=${RUN_ROOT_BASE:-/root/APRIL/runs/idea-train-window-${MODEL_TAG}-${TASK_TAG}-bs${ROLLOUT_BATCH_SIZE}-len${ROLLOUT_MAX_RESPONSE_LEN}}

source "${MODEL_SCRIPT}"

if [ -n "${OVER_SAMPLING_BATCH_SIZES}" ]; then
    BATCH_SIZES="${OVER_SAMPLING_BATCH_SIZES}"
else
    # shellcheck disable=SC2086
    BATCH_SIZES=$(batch_sizes_from_ratios "${ROLLOUT_BATCH_SIZE}" ${WINDOW_RATIOS})
fi
if [ -z "${BATCH_SIZES}" ]; then
    echo "No batch sizes resolved for the training sweep." >&2
    exit 1
fi
# shellcheck disable=SC2086
BATCH_SIZES=$(ensure_baseline_batch_size "${ROLLOUT_BATCH_SIZE}" ${BATCH_SIZES})

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l || true)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

mkdir -p "${RUN_ROOT_BASE}"

COMMON_CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
)

COMMON_ROLLOUT_ARGS=(
   --prompt-data "${INPUT_DATA}"
   --input-key source_prompt
   --label-key answer
   --metadata-key metadata
   --apply-chat-template
   --rm-type deepscaler
   --num-rollout "${NUM_ROLLOUT}"
   --save-interval "${EVAL_INTERVAL}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 0.8
   --global-batch-size "$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))"
   --balance-data
)

COMMON_EVAL_ARGS=(
   --eval-prompt-data "${EVAL_NAME}" "${EVAL_DATA}"
   --n-samples-per-eval-prompt 1
   --eval-interval "${EVAL_INTERVAL}"
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

COMMON_TRAIN_ARGS=(
   --lr 1e-6
   --min-lr 1e-7
   --lr-decay-style cosine
   --weight-decay 0.01
   --clip-grad 1.0
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

if [ "${FORCE_RAY_RESTART}" = "1" ] && [ "${DRY_RUN}" != "1" ]; then
    start_fresh_ray_head "${MASTER_ADDR}" 1
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

run_job() {
    local over_bs="$1"
    local ratio_label="$2"
    local run_name="$3"
    shift 3
    local run_root="${RUN_ROOT_BASE}/${run_name}"
    local debug_dir="${run_root}/debug_rollout"
    mkdir -p "${run_root}" "${debug_dir}"

    local load_path="${BASE_LOAD_PATH}"
    if [ -f "${run_root}/latest_checkpointed_iteration.txt" ]; then
        load_path="${run_root}"
    fi

    local cmd=(
        ray job submit --address="http://127.0.0.1:8265"
        --runtime-env-json="${RUNTIME_ENV_JSON}"
        -- python3 train.py
        --actor-num-nodes 1
        --actor-num-gpus-per-node 1
        --colocate
        --keep-only-latest-checkpoint
        --load "${load_path}"
        --save "${run_root}"
        --save-debug-rollout-data "${debug_dir}/rollout_{rollout_id:06d}.pkl"
        "${MODEL_ARGS[@]}"
        "${COMMON_CKPT_ARGS[@]}"
        "${COMMON_ROLLOUT_ARGS[@]}"
        "${COMMON_EVAL_ARGS[@]}"
        "${COMMON_GRPO_ARGS[@]}"
        "${COMMON_TRAIN_ARGS[@]}"
        "${COMMON_PERF_ARGS[@]}"
        "${COMMON_SGLANG_ARGS[@]}"
        "${COMMON_MISC_ARGS[@]}"
        "$@"
    )

    echo "=== Starting ${run_name} (window=${over_bs}, ratio=${ratio_label}) ==="
    if [ "${DRY_RUN}" = "1" ]; then
        quote_cmd "${cmd[@]}"
        echo "${run_name},dry_run,0" >> "${RUN_ROOT_BASE}/run_status.csv"
        return
    fi

    set +e
    "${cmd[@]}" | tee "${run_root}/job_output.log"
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

echo "run_name,status,exit_code" > "${RUN_ROOT_BASE}/run_status.csv"

for OVER_BS in ${BATCH_SIZES}; do
    if [ "${OVER_BS}" -lt "${ROLLOUT_BATCH_SIZE}" ]; then
        echo "Skip invalid over_sampling_batch_size=${OVER_BS} < rollout_batch_size=${ROLLOUT_BATCH_SIZE}" >&2
        continue
    fi

    RATIO_LABEL=$(ratio_label_from_sizes "${ROLLOUT_BATCH_SIZE}" "${OVER_BS}")
    if [ "${OVER_BS}" -eq "${ROLLOUT_BATCH_SIZE}" ]; then
        RUN_NAME="${MODEL_TAG}-${TASK_TAG}-train-len${ROLLOUT_MAX_RESPONSE_LEN}-bs${ROLLOUT_BATCH_SIZE}-nonprovision"
        run_job "${OVER_BS}" "${RATIO_LABEL}" "${RUN_NAME}"
    else
        RUN_NAME="${MODEL_TAG}-${TASK_TAG}-train-len${ROLLOUT_MAX_RESPONSE_LEN}-bs${ROLLOUT_BATCH_SIZE}-over${OVER_BS}-${RATIO_LABEL}"
        run_job "${OVER_BS}" "${RATIO_LABEL}" "${RUN_NAME}" \
            --partial-rollout \
            --over-sampling-batch-size "${OVER_BS}"
    fi
done

if [ "${RUN_ANALYSIS}" = "1" ] && [ "${DRY_RUN}" != "1" ]; then
    python "${IDEA_DIR}/analyze_window_experiments.py" \
        --run-root-base "${RUN_ROOT_BASE}" \
        --output-dir "${RUN_ROOT_BASE}/analysis" \
        --skip-decode-proxy

    python "${REPO_ROOT}/scripts/analysis/plot_offpolicy_training_compare.py" \
        --run-root-base "${RUN_ROOT_BASE}" \
        --output-dir "${RUN_ROOT_BASE}/analysis/training_compare"
fi
