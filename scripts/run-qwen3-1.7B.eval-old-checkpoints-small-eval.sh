#!/bin/bash

set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH=${PYTHONPATH:-/root/Megatron-LM/}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

RUN_ROOT=${RUN_ROOT:-/root/APRIL/runs/offpolicy-matrix-bs-half-r30-qwen3-1.7b-len4096/qwen3-1.7b-train-non-provision-bs24-r30}
CHECKPOINT_ITERS=${CHECKPOINT_ITERS:-4,9,14,19,24,29}
EVAL_DATA=${EVAL_DATA:-/root/gsm8k/data/gsm8k-test.subset256.seed1234.parquet}
EVAL_DATA_NAME=${EVAL_DATA_NAME:-gsm8k_test_subset256}
N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT:-1}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-4096}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.70}
NUM_ROLLOUT=${NUM_ROLLOUT:-30}
TMP_LOAD_BASE=${TMP_LOAD_BASE:-/tmp/april_eval_loads}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-1.7B.sh"

ANALYSIS_DIR="${RUN_ROOT}/analysis"
LOG_PATH="${ANALYSIS_DIR}/checkpoint_eval_subset256.log"
JSON_PATH="${ANALYSIS_DIR}/checkpoint_eval_subset256_results.json"
MD_PATH="${ANALYSIS_DIR}/checkpoint_eval_subset256_results.md"

mkdir -p "${ANALYSIS_DIR}" "${TMP_LOAD_BASE}"
: > "${LOG_PATH}"

echo "[eval] stopping old ray processes" | tee -a "${LOG_PATH}"
ray stop --force >> "${LOG_PATH}" 2>&1 || true
ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats >> "${LOG_PATH}" 2>&1

RESULTS_JSONL="${ANALYSIS_DIR}/checkpoint_eval_subset256_results.jsonl"
: > "${RESULTS_JSONL}"
find "${ANALYSIS_DIR}" -maxdepth 1 -name 'checkpoint_eval_iter_*.json' -delete

IFS=',' read -r -a iter_array <<< "${CHECKPOINT_ITERS}"

for iter in "${iter_array[@]}"; do
    iter_padded=$(printf "%07d" "${iter}")
    ckpt_dir="${RUN_ROOT}/iter_${iter_padded}"
    if [ ! -d "${ckpt_dir}" ]; then
        echo "[eval] missing checkpoint ${ckpt_dir}, skipping" | tee -a "${LOG_PATH}"
        continue
    fi

    load_dir="$(mktemp -d "${TMP_LOAD_BASE}/iter_${iter_padded}.XXXXXX")"
    ln -s "${ckpt_dir}" "${load_dir}/iter_${iter_padded}"
    printf "%s\n" "${iter}" > "${load_dir}/latest_checkpointed_iteration.txt"

    echo "[eval] evaluating checkpoint iter_${iter_padded}" | tee -a "${LOG_PATH}"
    python3 /root/APRIL/scripts/eval_checkpoint_subset.py \
        --actor-num-nodes 1 \
        --actor-num-gpus-per-node 1 \
        --colocate \
        --save "${RUN_ROOT}" \
        --load "${load_dir}" \
        --rollout-batch-size 24 \
        --global-batch-size 192 \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint /root/Qwen3-1.7B \
        --ref-load /root/Qwen3-1.7B_torch_dist \
        --prompt-data /root/gsm8k/data/gsm8k-train.parquet \
        --input-key source_prompt \
        --label-key answer \
        --metadata-key metadata \
        --apply-chat-template \
        --rollout-shuffle \
        --rm-type deepscaler \
        --num-rollout "${NUM_ROLLOUT}" \
        --save-interval 100 \
        --n-samples-per-prompt 8 \
        --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
        --rollout-temperature 0.8 \
        --balance-data \
        --eval-interval 100 \
        --eval-prompt-data "${EVAL_DATA_NAME}" "${EVAL_DATA}" \
        --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}" \
        --eval-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
        --eval-top-p 0.7 \
        --advantage-estimator grpo \
        --kl-loss-coef 0.00 \
        --kl-coef 0.00 \
        --entropy-coef 0.00 \
        --eps-clip 0.2 \
        --eps-clip-high 0.28 \
        --optimizer adam \
        --lr 1e-6 \
        --lr-decay-style constant \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.98 \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --context-parallel-size 1 \
        --expert-model-parallel-size 1 \
        --expert-tensor-parallel-size 1 \
        --recompute-granularity full \
        --recompute-method uniform \
        --recompute-num-layers 1 \
        --use-dynamic-batch-size \
        --max-tokens-per-gpu 2048 \
        --optimizer-cpu-offload \
        --overlap-cpu-optimizer-d2h-h2d \
        --use-precision-aware-optimizer \
        --rollout-num-gpus-per-engine 1 \
        --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}" \
        --sglang-disable-cuda-graph \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --accumulate-allreduce-grads-in-fp32 \
        --attention-softmax-in-fp32 \
        --attention-backend flash \
        --eval-output-json "${ANALYSIS_DIR}/checkpoint_eval_iter_${iter_padded}.json" \
        --eval-rollout-id "${iter}" \
        >> "${LOG_PATH}" 2>&1

    cat "${ANALYSIS_DIR}/checkpoint_eval_iter_${iter_padded}.json" >> "${RESULTS_JSONL}"
    rm -rf "${load_dir}"
    ray stop --force >> "${LOG_PATH}" 2>&1 || true
    ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats >> "${LOG_PATH}" 2>&1
done

RUN_ROOT_FOR_SUMMARY="${RUN_ROOT}" python3 - <<'PY'
import json
import os
from pathlib import Path

analysis_dir = Path(os.environ["RUN_ROOT_FOR_SUMMARY"]) / "analysis"
jsonl_path = analysis_dir / "checkpoint_eval_subset256_results.jsonl"
json_path = analysis_dir / "checkpoint_eval_subset256_results.json"
md_path = analysis_dir / "checkpoint_eval_subset256_results.md"

rows = []
for line in jsonl_path.read_text().splitlines():
    line = line.strip()
    if line:
        rows.append(json.loads(line))

rows.sort(key=lambda x: x["checkpoint_rollout_id"])
json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

lines = [
    "# Checkpoint Eval on GSM8K subset256",
    "",
    "| checkpoint | metric | reward_mean | truncated_ratio | num_samples |",
    "|---|---:|---:|---:|---:|",
]
for row in rows:
    for metric_name, metric in row["metrics"].items():
        lines.append(
            f"| {row['checkpoint_rollout_id']} | {metric_name} | "
            f"{metric['reward_mean']:.6f} | {metric.get('truncated_ratio', 0.0):.6f} | {metric['num_samples']} |"
        )
md_path.write_text("\n".join(lines) + "\n")
PY

echo "[cleanup] removing evaluated checkpoint directories" | tee -a "${LOG_PATH}"
for iter in "${iter_array[@]}"; do
    iter_padded=$(printf "%07d" "${iter}")
    rm -rf "${RUN_ROOT}/iter_${iter_padded}"
done
rm -f "${RUN_ROOT}/latest_checkpointed_iteration.txt"

ray stop --force >> "${LOG_PATH}" 2>&1 || true

echo "[done] wrote ${JSON_PATH} and ${MD_PATH}, removed old checkpoint directories" | tee -a "${LOG_PATH}"
