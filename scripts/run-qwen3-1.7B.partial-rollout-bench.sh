#!/bin/bash

set -ex

export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/lib/train_cleanup.sh"

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * 2))}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-256}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.7}

source "${SCRIPT_DIR}/models/qwen3-1.7B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-1.7B
   --ref-load /root/Qwen3-1.7B_torch_dist
   --load /root/.slime-nonexistent-qwen3-1.7B-partial-bench-load
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/data/train-00000-of-00001.parquet
   --input-key source_prompt
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 1
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}
   --rollout-temperature 0.8
   --global-batch-size $((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
   --balance-data
   --partial-rollout
   --over-sampling-batch-size ${OVER_SAMPLING_BATCH_SIZE}
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
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
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION}
   --sglang-disable-cuda-graph
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
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
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
