#!/bin/bash

set -euo pipefail

export PYTHONUNBUFFERED=1

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-256}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.70}
NUM_ROLLOUT=${NUM_ROLLOUT:-1}
RUN_MODE=${RUN_MODE:-rollout_only}
INPUT_DATA=${INPUT_DATA:-/root/gsm8k/data/gsm8k-train.parquet}
RUN_ROOT_BASE=${RUN_ROOT_BASE:-/root/APRIL/runs/partial-rollout-sweep-qwen3-1.7b-bs${ROLLOUT_BATCH_SIZE}}
OVER_SAMPLING_BATCH_SIZES=${OVER_SAMPLING_BATCH_SIZES:-"32 40 48 56 64 80 96"}
FORCE_RAY_RESTART=${FORCE_RAY_RESTART:-1}

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
   --load /root/.slime-nonexistent-qwen3-1.7B-partial-sweep-load
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
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 0.8
   --global-batch-size "$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))"
   --balance-data
   --partial-rollout
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
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

TRAIN_MODE_ARGS=()
if [ "${RUN_MODE}" = "rollout_only" ]; then
    TRAIN_MODE_ARGS+=(--debug-rollout-only)
elif [ "${RUN_MODE}" = "train" ]; then
    :
else
    echo "Unsupported RUN_MODE=${RUN_MODE}. Use rollout_only or train." >&2
    exit 1
fi

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

if [ "${FORCE_RAY_RESTART}" = "1" ]; then
    ray stop --force || true
    ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 1 --disable-usage-stats
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

for OVER_SAMPLING_BATCH_SIZE in ${OVER_SAMPLING_BATCH_SIZES}; do
    if [ "${OVER_SAMPLING_BATCH_SIZE}" -lt "${ROLLOUT_BATCH_SIZE}" ]; then
        echo "Skip invalid over_sampling_batch_size=${OVER_SAMPLING_BATCH_SIZE} < rollout_batch_size=${ROLLOUT_BATCH_SIZE}" >&2
        continue
    fi

    RATIO_LABEL=$(python - <<PY
rollout_bs = int("${ROLLOUT_BATCH_SIZE}")
over_bs = int("${OVER_SAMPLING_BATCH_SIZE}")
ratio = over_bs / rollout_bs
if ratio.is_integer():
    print(f"{int(ratio)}p0x")
else:
    print(str(ratio).replace(".", "p") + "x")
PY
)

    RUN_NAME="qwen3-1.7b-partial-gsm8k-bs${ROLLOUT_BATCH_SIZE}-over${OVER_SAMPLING_BATCH_SIZE}-${RATIO_LABEL}"
    RUN_ROOT="${RUN_ROOT_BASE}/${RUN_NAME}"
    mkdir -p "${RUN_ROOT}"

    echo "=== Starting ${RUN_NAME} ==="
    echo "rollout_batch_size=${ROLLOUT_BATCH_SIZE}, over_sampling_batch_size=${OVER_SAMPLING_BATCH_SIZE}, run_mode=${RUN_MODE}"

    ray job submit --address="http://127.0.0.1:8265" \
       --runtime-env-json="${RUNTIME_ENV_JSON}" \
       -- python3 train.py \
       "${TRAIN_MODE_ARGS[@]}" \
       --actor-num-nodes 1 \
       --actor-num-gpus-per-node 1 \
       --colocate \
       --save "${RUN_ROOT}" \
       --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}" \
       "${MODEL_ARGS[@]}" \
       "${COMMON_CKPT_ARGS[@]}" \
       "${COMMON_ROLLOUT_ARGS[@]}" \
       "${COMMON_GRPO_ARGS[@]}" \
       "${COMMON_PERF_ARGS[@]}" \
       "${COMMON_SGLANG_ARGS[@]}" \
       "${COMMON_MISC_ARGS[@]}" | tee "${RUN_ROOT}/job_output.log"
done
