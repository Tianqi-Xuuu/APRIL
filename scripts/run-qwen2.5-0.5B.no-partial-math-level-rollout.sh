#!/bin/bash

set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/lib/train_cleanup.sh"

ROLL_OUT_BATCH_SIZE=${ROLL_OUT_BATCH_SIZE:-64}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-8192}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.70}
RUN_NAME=${RUN_NAME:-qwen2.5-0.5b-no-partial-math-level-bs64-n1}
RUN_ROOT=${RUN_ROOT:-/root/APRIL/runs/${RUN_NAME}}
INPUT_DATA=${INPUT_DATA:-/root/math_level12/data/math-level4-train.shortprompt.parquet}
DEBUG_DIR=${DEBUG_DIR:-${RUN_ROOT}/debug_rollout}
ANALYSIS_DIR=${ANALYSIS_DIR:-${RUN_ROOT}/analysis}
JOB_LOG=${JOB_LOG:-${RUN_ROOT}/job_output.log}

mkdir -p "${RUN_ROOT}" "${DEBUG_DIR}" "${ANALYSIS_DIR}"

NUM_ROWS=$(python3 - <<PY
import pandas as pd
df = pd.read_parquet("${INPUT_DATA}")
print(len(df))
PY
)
NUM_ROLLOUT=${NUM_ROLLOUT:-$(((NUM_ROWS + ROLL_OUT_BATCH_SIZE - 1) / ROLL_OUT_BATCH_SIZE))}

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l || true)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi

source "${SCRIPT_DIR}/models/qwen2.5-0.5B.sh"

CKPT_ARGS=(
  --hf-checkpoint /root/Qwen2.5-0.5B
  --ref-load /root/Qwen2.5-0.5B_torch_dist
  --load /root/.slime-nonexistent-qwen2.5-0.5B-no-partial-math-load
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
  --save-interval 1
  --rollout-batch-size "${ROLL_OUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature 0.8
  --global-batch-size "$((ROLL_OUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))"
  --balance-data
  --save-debug-rollout-data "${DEBUG_DIR}/rollout_{rollout_id:06d}.pkl"
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
  --debug-rollout-only \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 1 \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" | tee "${JOB_LOG}"

python3 /root/APRIL/scripts/analysis/analyze_rollout_debug_data.py \
  --debug-dir "${DEBUG_DIR}" \
  --output-dir "${ANALYSIS_DIR}" \
  --log-path "${JOB_LOG}" | tee "${ANALYSIS_DIR}/analysis_stdout.log"
