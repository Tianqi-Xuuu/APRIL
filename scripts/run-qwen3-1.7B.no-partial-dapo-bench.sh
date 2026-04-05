#!/bin/bash

set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/lib/train_cleanup.sh"
APRIL_ROOT=${APRIL_ROOT:-$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)}
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}
SGLANG_PYTHON_ROOT=${SGLANG_PYTHON_ROOT:-}
QWEN3_HF_CHECKPOINT=${QWEN3_HF_CHECKPOINT:-/root/Qwen3-1.7B}
QWEN3_REF_LOAD=${QWEN3_REF_LOAD:-/root/Qwen3-1.7B_torch_dist}

ROLL_OUT_BATCH_SIZE=${ROLL_OUT_BATCH_SIZE:-128}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-256}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.72}
DEBUG_ROLLOUT_ONLY=${DEBUG_ROLLOUT_ONLY:-1}
RUN_NAME=${RUN_NAME:-qwen3-1.7b-no-partial-dapo-bs128-n2}
RUN_ROOT=${RUN_ROOT:-${APRIL_ROOT}/runs/${RUN_NAME}}
INPUT_DATA=${INPUT_DATA:-/root/dapo-math-17k/data/train-00000-of-00001.parquet}
PADDED_DATA=${PADDED_DATA:-${RUN_ROOT}/data/dapo_math_17k_padded_bs${ROLL_OUT_BATCH_SIZE}.parquet}
DEBUG_DIR=${DEBUG_DIR:-${RUN_ROOT}/debug_rollout}
ANALYSIS_DIR=${ANALYSIS_DIR:-${RUN_ROOT}/analysis}
JOB_LOG=${JOB_LOG:-${RUN_ROOT}/job_output.log}

mkdir -p "${RUN_ROOT}" "${DEBUG_DIR}" "${ANALYSIS_DIR}"

python "${APRIL_ROOT}/scripts/analysis/prepare_padded_dataset.py" \
  --input "${INPUT_DATA}" \
  --output "${PADDED_DATA}" \
  --batch-size "${ROLL_OUT_BATCH_SIZE}" | tee "${RUN_ROOT}/dataset_prepare.log"

NUM_ROWS=$(python - <<PY
import pandas as pd
df = pd.read_parquet("${PADDED_DATA}")
print(len(df))
PY
)
NUM_ROLLOUT=${NUM_ROLLOUT:-$((NUM_ROWS / ROLL_OUT_BATCH_SIZE))}

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l || true)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi

source "${SCRIPT_DIR}/models/qwen3-1.7B.sh"

CKPT_ARGS=(
  --hf-checkpoint "${QWEN3_HF_CHECKPOINT}"
  --ref-load "${QWEN3_REF_LOAD}"
  --load /root/.slime-nonexistent-qwen3-1.7B-no-partial-bench-load
  --save "${RUN_ROOT}"
)

ROLLOUT_ARGS=(
  --prompt-data "${PADDED_DATA}"
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
  --global-batch-size $((ROLL_OUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
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

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
  --sglang-disable-cuda-graph
)

MISC_ARGS=(
  --no-gradient-accumulation-fusion
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

DEBUG_ARGS=()
if [ "${DEBUG_ROLLOUT_ONLY}" = "1" ]; then
  DEBUG_ARGS+=(--debug-rollout-only)
fi

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
start_fresh_ray_head "${MASTER_ADDR}" 1

RUNTIME_PYTHONPATH="${MEGATRON_ROOT}"
if [ -n "${SGLANG_PYTHON_ROOT}" ]; then
  RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH}:${SGLANG_PYTHON_ROOT}"
fi

RUNTIME_ENV_JSON="{
  \"working_dir\": \"${APRIL_ROOT}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"${RUNTIME_PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  "${DEBUG_ARGS[@]}" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 1 \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" | tee "${JOB_LOG}"

python "${APRIL_ROOT}/scripts/analysis/analyze_rollout_debug_data.py" \
  --debug-dir "${DEBUG_DIR}" \
  --output-dir "${ANALYSIS_DIR}" \
  --log-path "${JOB_LOG}" | tee "${ANALYSIS_DIR}/analysis_stdout.log"
