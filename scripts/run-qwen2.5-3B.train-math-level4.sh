#!/bin/bash

set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/lib/train_cleanup.sh"

ROLL_OUT_BATCH_SIZE=${ROLL_OUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-4096}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.70}
NUM_ROLLOUT=${NUM_ROLLOUT:-30}
EVAL_INTERVAL=${EVAL_INTERVAL:-5}
PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-0}
OVERSAMPLING_BATCH_SIZE=${OVERSAMPLING_BATCH_SIZE:-$((ROLL_OUT_BATCH_SIZE * 2))}
INPUT_DATA=${INPUT_DATA:-/root/math_level12/data/math-level4-train.stepthink.parquet}
EVAL_DATA=${EVAL_DATA:-/root/math_level12/data/math-level4-test.subset128.seed1234.parquet}
TRAIN_DATA_TAG=${TRAIN_DATA_TAG:-$(basename "${INPUT_DATA}" .parquet)}
EVAL_DATA_TAG=${EVAL_DATA_TAG:-$(basename "${EVAL_DATA}" .parquet)}
EVAL_NAME=${EVAL_NAME:-${EVAL_DATA_TAG//-/_}}
PROVISION_TAG=${PROVISION_TAG:-}
if [ -z "${PROVISION_TAG}" ] && [ "${PARTIAL_ROLLOUT}" = "1" ]; then
  if [ "${ROLL_OUT_BATCH_SIZE}" -gt 0 ]; then
    provision_x10=$((OVERSAMPLING_BATCH_SIZE * 10 / ROLL_OUT_BATCH_SIZE))
    PROVISION_TAG="-prov${provision_x10}"
  fi
fi
RUN_NAME=${RUN_NAME:-qwen2.5-3b-train-${TRAIN_DATA_TAG}${PROVISION_TAG}-bs${ROLL_OUT_BATCH_SIZE}-n${N_SAMPLES_PER_PROMPT}-r${NUM_ROLLOUT}}
RUN_ROOT=${RUN_ROOT:-/root/APRIL/runs/${RUN_NAME}}
DEBUG_DIR=${DEBUG_DIR:-${RUN_ROOT}/debug_rollout}
ANALYSIS_DIR=${ANALYSIS_DIR:-${RUN_ROOT}/analysis}
JOB_LOG=${JOB_LOG:-${RUN_ROOT}/job_output.log}
LOAD_PATH=${LOAD_PATH:-/root/.slime-nonexistent-qwen2.5-3B-train-load}

mkdir -p "${RUN_ROOT}" "${DEBUG_DIR}" "${ANALYSIS_DIR}"

if [ -f "${RUN_ROOT}/latest_checkpointed_iteration.txt" ]; then
  LOAD_PATH="${RUN_ROOT}"
fi

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l || true)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi

source "${SCRIPT_DIR}/models/qwen2.5-3B.sh"

CKPT_ARGS=(
  --hf-checkpoint /root/Qwen2.5-3B
  --ref-load /root/Qwen2.5-3B_torch_dist
  --load "${LOAD_PATH}"
  --save "${RUN_ROOT}"
)

ROLLOUT_ARGS=(
  --prompt-data "${INPUT_DATA}"
  --input-key source_prompt
  --label-key answer
  --metadata-key metadata
  --apply-chat-template
  --rm-type deepscaler
  --num-rollout "${NUM_ROLLOUT}"
  --save-interval "${EVAL_INTERVAL}"
  --rollout-batch-size "${ROLL_OUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature 0.8
  --global-batch-size "$((ROLL_OUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))"
  --balance-data
  --save-debug-rollout-data "${DEBUG_DIR}/rollout_{rollout_id:06d}.pkl"
)

if [ "${PARTIAL_ROLLOUT}" = "1" ]; then
  ROLLOUT_ARGS+=(
    --partial-rollout
    --over-sampling-batch-size "${OVERSAMPLING_BATCH_SIZE}"
  )
fi

EVAL_ARGS=(
  --eval-prompt-data "${EVAL_NAME}" "${EVAL_DATA}"
  --n-samples-per-eval-prompt 1
  --eval-interval "${EVAL_INTERVAL}"
)

PERF_ARGS=(
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

TRAIN_ARGS=(
  --lr 1e-6
  --min-lr 1e-7
  --lr-decay-style cosine
  --weight-decay 0.01
  --clip-grad 1.0
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-loss-coef 0.00
  --kl-coef 0.00
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
  --sglang-disable-cuda-graph
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
start_fresh_ray_head "${MASTER_ADDR}" 1

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 1 \
  --colocate \
  --keep-only-latest-checkpoint \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${TRAIN_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" | tee "${JOB_LOG}"
