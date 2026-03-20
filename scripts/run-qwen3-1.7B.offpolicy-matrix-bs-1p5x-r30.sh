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
RUN_ROOT_BASE=${RUN_ROOT_BASE:-/root/APRIL/runs/offpolicy-matrix-bs-1p5x-r${NUM_ROLLOUT}-qwen3-1.7b-len${ROLLOUT_MAX_RESPONSE_LEN}}
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
   --load /root/.slime-nonexistent-qwen3-1.7B-offpolicy-matrix-bs-1p5x-load
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

run_job() {
    local run_name="$1"
    local rollout_batch_size="$2"
    shift 2

    local run_root="${RUN_ROOT_BASE}/${run_name}"
    mkdir -p "${run_root}"

    if [ "${FORCE_RAY_RESTART}" = "1" ]; then
        ray stop --force || true
        ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 1 --disable-usage-stats
    fi

    echo "=== Starting ${run_name} ==="
    ray job submit --address="http://127.0.0.1:8265" \
       --runtime-env-json="${RUNTIME_ENV_JSON}" \
       -- python3 train.py \
       --actor-num-nodes 1 \
       --actor-num-gpus-per-node 1 \
       --colocate \
       --save "${run_root}" \
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
       "$@" | tee "${run_root}/job_output.log"
}

run_job "qwen3-1.7b-train-non-provision-bs36-r${NUM_ROLLOUT}" 36

run_job "qwen3-1.7b-train-provision-2p0x-bs18-r${NUM_ROLLOUT}" 18 \
    --partial-rollout \
    --over-sampling-batch-size 36

run_job "qwen3-1.7b-train-provision-3p0x-bs12-r${NUM_ROLLOUT}" 12 \
    --partial-rollout \
    --over-sampling-batch-size 36
