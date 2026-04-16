#!/bin/bash

set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/lib/train_cleanup.sh"

# Repository and dependency roots.
APRIL_ROOT=${APRIL_ROOT:-$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)}
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}
SGLANG_PYTHON_ROOT=${SGLANG_PYTHON_ROOT:-}
QWEN25_HF_CHECKPOINT=${QWEN25_HF_CHECKPOINT:-/root/Qwen2.5-3B}
QWEN25_REF_LOAD=${QWEN25_REF_LOAD:-/root/Qwen2.5-3B_torch_dist}
SLIME_SGLANG_DIRECT=${SLIME_SGLANG_DIRECT:-0}
DEBUG_ROLLOUT_ONLY=${DEBUG_ROLLOUT_ONLY:-0}
SAVE_HF_WEIGHTS=${SAVE_HF_WEIGHTS:-0}
SAVE_HF_ONLY_FINAL=${SAVE_HF_ONLY_FINAL:-0}
TRAIN_SEED=${TRAIN_SEED:-1234}
DISABLE_CKPT_SAVE=${DISABLE_CKPT_SAVE:-0}
USE_BEHAVIOR_LOGPROBS_FOR_PPO_CLIP=${USE_BEHAVIOR_LOGPROBS_FOR_PPO_CLIP:-0}
LR=${LR:-1e-6}
MIN_LR=${MIN_LR:-1e-7}
CLIP_GRAD=${CLIP_GRAD:-1.0}
EPS_CLIP=${EPS_CLIP:-0.2}
EPS_CLIP_HIGH=${EPS_CLIP_HIGH:-0.28}

# Training and rollout defaults.
ROLL_OUT_BATCH_SIZE=${ROLL_OUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-4096}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.70}
SGLANG_WEIGHT_LOADER_DISABLE_MMAP=${SGLANG_WEIGHT_LOADER_DISABLE_MMAP:-1}
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
LOAD_PATH=${LOAD_PATH:-/root/.slime-nonexistent-qwen2.5-3B-train-load}

find_current_libstdcpp() {
  local ld_dir=""

  IFS=':' read -r -a ld_dirs <<< "${LD_LIBRARY_PATH:-}"
  for ld_dir in "${ld_dirs[@]}"; do
    if [ -r "${ld_dir}/libstdc++.so.6" ]; then
      printf '%s\n' "${ld_dir}/libstdc++.so.6"
      return 0
    fi
  done

  if command -v g++ >/dev/null 2>&1; then
    ld_dir=$(g++ -print-file-name=libstdc++.so.6 2>/dev/null || true)
    if [ -n "${ld_dir}" ] && [ -r "${ld_dir}" ]; then
      printf '%s\n' "${ld_dir}"
      return 0
    fi
  fi

  return 1
}

ensure_glibcxx_3_4_32() {
  # cumem_allocator requires a newer libstdc++ than many cluster defaults provide.
  local required_symbol="GLIBCXX_3.4.32"
  local current_libstdcpp=""
  local modules_init=""

  current_libstdcpp="$(find_current_libstdcpp || true)"
  if [ -n "${current_libstdcpp}" ] && strings "${current_libstdcpp}" 2>/dev/null | grep -q "${required_symbol}"; then
    return 0
  fi

  if ! type module >/dev/null 2>&1; then
    for modules_init in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash /opt/Modules/init/bash; do
      if [ -f "${modules_init}" ]; then
        # shellcheck source=/dev/null
        source "${modules_init}"
        break
      fi
    done
  fi

  if type module >/dev/null 2>&1; then
    module load gcc/13.3.1-p20240614 >/dev/null 2>&1 || \
      module load gcc/13.2.1-p20240113 >/dev/null 2>&1 || true
  fi

  current_libstdcpp="$(find_current_libstdcpp || true)"
  if [ -n "${current_libstdcpp}" ] && strings "${current_libstdcpp}" 2>/dev/null | grep -q "${required_symbol}"; then
    return 0
  fi

  echo "warning: failed to locate libstdc++ with ${required_symbol}; cumem_allocator may fail to import" >&2
}

compute_provision_tag() {
  if [ -n "${PROVISION_TAG}" ] || [ "${PARTIAL_ROLLOUT}" != "1" ] || [ "${ROLL_OUT_BATCH_SIZE}" -le 0 ]; then
    return 0
  fi

  local provision_x10
  provision_x10=$((OVERSAMPLING_BATCH_SIZE * 10 / ROLL_OUT_BATCH_SIZE))
  PROVISION_TAG="-prov${provision_x10}"
}

prepare_run_layout() {
  # Reuse the same run directory when resuming from a prior checkpoint.
  RUN_NAME=${RUN_NAME:-qwen2.5-3b-train-${TRAIN_DATA_TAG}${PROVISION_TAG}-bs${ROLL_OUT_BATCH_SIZE}-n${N_SAMPLES_PER_PROMPT}-r${NUM_ROLLOUT}}
  RUN_ROOT=${RUN_ROOT:-${APRIL_ROOT}/runs/${RUN_NAME}}
  DEBUG_DIR=${DEBUG_DIR:-${RUN_ROOT}/debug_rollout}
  ANALYSIS_DIR=${ANALYSIS_DIR:-${RUN_ROOT}/analysis}
  JOB_LOG=${JOB_LOG:-${RUN_ROOT}/job_output.log}

  mkdir -p "${RUN_ROOT}" "${DEBUG_DIR}" "${ANALYSIS_DIR}"

  if [ -f "${RUN_ROOT}/latest_checkpointed_iteration.txt" ]; then
    LOAD_PATH="${RUN_ROOT}"
  fi
}

detect_nvlink_support() {
  local nvlink_count
  nvlink_count=$(nvidia-smi | grep -o "NVLink" | wc -l || true)

  if [ "${nvlink_count}" -gt 0 ]; then
    HAS_NVLINK=1
  else
    HAS_NVLINK=0
  fi
}

build_checkpoint_args() {
  CKPT_ARGS=(
    --hf-checkpoint "${QWEN25_HF_CHECKPOINT}"
    --ref-load "${QWEN25_REF_LOAD}"
    --load "${LOAD_PATH}"
    --save "${RUN_ROOT}"
  )
}

build_data_args() {
  local checkpoint_save_interval="${EVAL_INTERVAL}"
  if [ "${DISABLE_CKPT_SAVE}" = "1" ]; then
    checkpoint_save_interval="$((NUM_ROLLOUT + 1))"
  fi

  ROLLOUT_ARGS=(
    --prompt-data "${INPUT_DATA}"
    --input-key source_prompt
    --label-key answer
    --metadata-key metadata
    --apply-chat-template
    --rm-type deepscaler
    --num-rollout "${NUM_ROLLOUT}"
    --save-interval "${checkpoint_save_interval}"
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
}

build_training_args() {
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
    --seed "${TRAIN_SEED}"
    --optimizer adam
    --lr "${LR}"
    --min-lr "${MIN_LR}"
    --lr-decay-style cosine
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad "${CLIP_GRAD}"
  )

  GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.00
    --kl-coef 0.00
    --entropy-coef 0.00
    --eps-clip "${EPS_CLIP}"
    --eps-clip-high "${EPS_CLIP_HIGH}"
  )

  SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
    --sglang-disable-cuda-graph
  )

  if [ "${SGLANG_WEIGHT_LOADER_DISABLE_MMAP}" = "1" ]; then
    SGLANG_ARGS+=(--sglang-weight-loader-disable-mmap)
  fi

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
  if [ "${USE_BEHAVIOR_LOGPROBS_FOR_PPO_CLIP}" = "1" ]; then
    DEBUG_ARGS+=(--use-behavior-logprobs-for-ppo-clip)
  fi

  SAVE_ARGS=()
  if [ "${SAVE_HF_WEIGHTS}" = "1" ]; then
    SAVE_ARGS+=(--save-hf-weights)
    if [ "${SAVE_HF_ONLY_FINAL}" = "1" ]; then
      SAVE_ARGS+=(--save-hf-only-final)
    fi
  else
    SAVE_ARGS+=(--keep-only-latest-checkpoint)
  fi
}

prepare_runtime_env() {
  # Keep Ray workers on the same runtime paths as the current shell session.
  export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
  if [ "${SLIME_MODAL_DIRECT:-0}" = "1" ]; then
    cleanup_training_processes
    ray stop --force >/dev/null 2>&1 || true
  else
    start_fresh_ray_head "${MASTER_ADDR}" 1
  fi

  RUNTIME_PYTHONPATH="${MEGATRON_ROOT}"
  if [ -n "${SGLANG_PYTHON_ROOT}" ]; then
    RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH}:${SGLANG_PYTHON_ROOT}"
  fi

  RUNTIME_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  RUNTIME_ENV_JSON="{
    \"working_dir\": \"${APRIL_ROOT}\",
    \"env_vars\": {
      \"PYTHONPATH\": \"${RUNTIME_PYTHONPATH}\",
      \"LD_LIBRARY_PATH\": \"${RUNTIME_LD_LIBRARY_PATH}\",
      \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
      \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
      \"SLIME_SGLANG_DIRECT\": \"${SLIME_SGLANG_DIRECT}\"
    }
  }"
}

submit_training_job() {
  if [ "${SLIME_MODAL_DIRECT:-0}" = "1" ]; then
    PYTHONPATH="${RUNTIME_PYTHONPATH}" \
      LD_LIBRARY_PATH="${RUNTIME_LD_LIBRARY_PATH}" \
      CUDA_DEVICE_MAX_CONNECTIONS=1 \
      NCCL_NVLS_ENABLE="${HAS_NVLINK}" \
      SLIME_SGLANG_DIRECT="${SLIME_SGLANG_DIRECT}" \
      SLIME_MODAL_DIRECT="${SLIME_MODAL_DIRECT:-0}" \
      python3 train.py \
      "${DEBUG_ARGS[@]}" \
      "${SAVE_ARGS[@]}" \
      --actor-num-nodes 1 \
      --actor-num-gpus-per-node 1 \
      --colocate \
      "${MODEL_ARGS[@]}" \
      "${CKPT_ARGS[@]}" \
      "${ROLLOUT_ARGS[@]}" \
      "${EVAL_ARGS[@]}" \
      "${GRPO_ARGS[@]}" \
      "${TRAIN_ARGS[@]}" \
      "${PERF_ARGS[@]}" \
      "${SGLANG_ARGS[@]}" \
      "${MISC_ARGS[@]}" | tee "${JOB_LOG}"
    return
  fi

  ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
    "${DEBUG_ARGS[@]}" \
    "${SAVE_ARGS[@]}" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 1 \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" | tee "${JOB_LOG}"
}

main() {
  compute_provision_tag
  ensure_glibcxx_3_4_32
  prepare_run_layout
  detect_nvlink_support

  source "${SCRIPT_DIR}/models/qwen2.5-3B.sh"

  build_checkpoint_args
  build_data_args
  build_training_args
  prepare_runtime_env
  submit_training_job
}

main "$@"
