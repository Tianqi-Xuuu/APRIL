#!/bin/bash

set -euo pipefail

export PYTHONUNBUFFERED=1

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-4096}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.70}
NUM_ROLLOUT=${NUM_ROLLOUT:-3}
INPUT_DATA=${INPUT_DATA:-/root/gsm8k/data/gsm8k-train.parquet}
RUN_ROOT_BASE=${RUN_ROOT_BASE:-/root/APRIL/runs/provision-throughput-bench-qwen3-1.7b-bs${ROLLOUT_BATCH_SIZE}-len${ROLLOUT_MAX_RESPONSE_LEN}}
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
   --load /root/.slime-nonexistent-qwen3-1.7B-provision-bench-load
)

COMMON_ROLLOUT_ARGS=(
   --prompt-data "${INPUT_DATA}"
   --input-key source_prompt
   --label-key answer
   --metadata-key metadata
   --apply-chat-template
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

run_job() {
    local run_name="$1"
    shift
    local run_root="${RUN_ROOT_BASE}/${run_name}"
    mkdir -p "${run_root}" "${run_root}/debug_rollout"

    echo "=== Starting ${run_name} ==="
    set +e
    ray job submit --address="http://127.0.0.1:8265" \
       --runtime-env-json="${RUNTIME_ENV_JSON}" \
       -- python3 train.py \
       --debug-rollout-only \
       --actor-num-nodes 1 \
       --actor-num-gpus-per-node 1 \
       --colocate \
       --save "${run_root}" \
       "${MODEL_ARGS[@]}" \
       "${COMMON_CKPT_ARGS[@]}" \
       "${COMMON_ROLLOUT_ARGS[@]}" \
       "${COMMON_GRPO_ARGS[@]}" \
       "${COMMON_PERF_ARGS[@]}" \
       "${COMMON_SGLANG_ARGS[@]}" \
       "${COMMON_MISC_ARGS[@]}" \
       "$@" | tee "${run_root}/job_output.log"
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

run_job "qwen3-1.7b-non-provision-bs${ROLLOUT_BATCH_SIZE}" \
    --save-debug-rollout-data "${RUN_ROOT_BASE}/qwen3-1.7b-non-provision-bs${ROLLOUT_BATCH_SIZE}/debug_rollout/rollout_{rollout_id:06d}.pkl"

run_job "qwen3-1.7b-provision-1p5x-bs${ROLLOUT_BATCH_SIZE}" \
    --save-debug-rollout-data "${RUN_ROOT_BASE}/qwen3-1.7b-provision-1p5x-bs${ROLLOUT_BATCH_SIZE}/debug_rollout/rollout_{rollout_id:06d}.pkl" \
    --partial-rollout \
    --over-sampling-batch-size "$((ROLLOUT_BATCH_SIZE * 3 / 2))"

run_job "qwen3-1.7b-provision-2p0x-bs${ROLLOUT_BATCH_SIZE}" \
    --save-debug-rollout-data "${RUN_ROOT_BASE}/qwen3-1.7b-provision-2p0x-bs${ROLLOUT_BATCH_SIZE}/debug_rollout/rollout_{rollout_id:06d}.pkl" \
    --partial-rollout \
    --over-sampling-batch-size "$((ROLLOUT_BATCH_SIZE * 2))"

run_job "qwen3-1.7b-provision-2p5x-bs${ROLLOUT_BATCH_SIZE}" \
    --save-debug-rollout-data "${RUN_ROOT_BASE}/qwen3-1.7b-provision-2p5x-bs${ROLLOUT_BATCH_SIZE}/debug_rollout/rollout_{rollout_id:06d}.pkl" \
    --partial-rollout \
    --over-sampling-batch-size "$((ROLLOUT_BATCH_SIZE * 5 / 2))"

run_job "qwen3-1.7b-provision-3p0x-bs${ROLLOUT_BATCH_SIZE}" \
    --save-debug-rollout-data "${RUN_ROOT_BASE}/qwen3-1.7b-provision-3p0x-bs${ROLLOUT_BATCH_SIZE}/debug_rollout/rollout_{rollout_id:06d}.pkl" \
    --partial-rollout \
    --over-sampling-batch-size "$((ROLLOUT_BATCH_SIZE * 3))"

python /root/APRIL/scripts/analysis/plot_provision_throughput.py \
    --run-root-base "${RUN_ROOT_BASE}" \
    --output-csv "${RUN_ROOT_BASE}/provision_benchmark.csv" \
    --output-plot "${RUN_ROOT_BASE}/tokens_per_sec_vs_provision.png"
