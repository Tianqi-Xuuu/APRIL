#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
APRIL_ROOT=${APRIL_ROOT:-$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)}

export PYTHONUNBUFFERED=1

OCEAN_ROOT=${OCEAN_ROOT:-/ocean/projects/cis260009p/txu7/april}
MATH_LEVEL12_ROOT=${MATH_LEVEL12_ROOT:-${OCEAN_ROOT}/data/math_level12}
MEGATRON_ROOT=${MEGATRON_ROOT:-${OCEAN_ROOT}/src/Megatron-LM}
SGLANG_PYTHON_ROOT=${SGLANG_PYTHON_ROOT:-${OCEAN_ROOT}/src/sglang/python}
QWEN25_HF_CHECKPOINT=${QWEN25_HF_CHECKPOINT:-${OCEAN_ROOT}/models/Qwen2.5-3B}
QWEN25_REF_LOAD=${QWEN25_REF_LOAD:-${OCEAN_ROOT}/models/Qwen2.5-3B_torch_dist}

export HF_HOME=${HF_HOME:-${OCEAN_ROOT}/hf}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export TMPDIR=${TMPDIR:-/ocean/projects/cis260009p/txu7/t}
export RAY_TMPDIR=${RAY_TMPDIR:-/ocean/projects/cis260009p/txu7/r}

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TMPDIR}" "${RAY_TMPDIR}" "${MATH_LEVEL12_ROOT}"

INPUT_DATA=${INPUT_DATA:-${MATH_LEVEL12_ROOT}/math-level2-train.parquet}
EVAL_SOURCE=${EVAL_SOURCE:-${MATH_LEVEL12_ROOT}/math-level2-test.parquet}
EVAL_DATA=${EVAL_DATA:-${MATH_LEVEL12_ROOT}/math-level2-test.subset128.seed1234.parquet}
EVAL_SAMPLE_SIZE=${EVAL_SAMPLE_SIZE:-128}
EVAL_SAMPLE_SEED=${EVAL_SAMPLE_SEED:-1234}

ROLL_OUT_BATCH_SIZE=${ROLL_OUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
NUM_ROLLOUT=${NUM_ROLLOUT:-30}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-4096}
EVAL_INTERVAL=${EVAL_INTERVAL:-5}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.70}
SLIME_SGLANG_DIRECT=${SLIME_SGLANG_DIRECT:-1}
DEBUG_ROLLOUT_ONLY=${DEBUG_ROLLOUT_ONLY:-0}

# User-facing alias: PROVISION_SIZE is the oversampling batch size used by partial rollout.
PROVISION_SIZE=${PROVISION_SIZE:-${OVERSAMPLING_BATCH_SIZE:-}}
PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-0}
if [ -n "${PROVISION_SIZE}" ]; then
  PARTIAL_ROLLOUT=1
  OVERSAMPLING_BATCH_SIZE=${OVERSAMPLING_BATCH_SIZE:-${PROVISION_SIZE}}
fi

if [ ! -f "${INPUT_DATA}" ]; then
  echo "Missing input dataset: ${INPUT_DATA}" >&2
  exit 1
fi

if [ ! -f "${EVAL_SOURCE}" ]; then
  echo "Missing eval source dataset: ${EVAL_SOURCE}" >&2
  exit 1
fi

if [ ! -f "${EVAL_DATA}" ]; then
  python - <<PY
import pandas as pd

src = "${EVAL_SOURCE}"
dst = "${EVAL_DATA}"
n = int("${EVAL_SAMPLE_SIZE}")
seed = int("${EVAL_SAMPLE_SEED}")

df = pd.read_parquet(src)
subset = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
subset.to_parquet(dst, index=False)
print({"output": dst, "rows": len(subset), "source": src, "seed": seed})
PY
fi

if [ ! -f "${QWEN25_REF_LOAD}/latest_checkpointed_iteration.txt" ]; then
  PYTHONPATH="${MEGATRON_ROOT}" python "${APRIL_ROOT}/tools/convert_hf_to_torch_dist.py" \
    --hf-checkpoint "${QWEN25_HF_CHECKPOINT}" \
    --save "${QWEN25_REF_LOAD}"
fi

TRAIN_DATA_TAG=${TRAIN_DATA_TAG:-math-level2}
EVAL_DATA_TAG=${EVAL_DATA_TAG:-math-level2-test-subset${EVAL_SAMPLE_SIZE}}
if [ "${PARTIAL_ROLLOUT}" = "1" ]; then
  RUN_NAME=${RUN_NAME:-qwen2.5-3b-grpo-${TRAIN_DATA_TAG}-prov${OVERSAMPLING_BATCH_SIZE}-bs${ROLL_OUT_BATCH_SIZE}-n${N_SAMPLES_PER_PROMPT}-r${NUM_ROLLOUT}}
else
  RUN_NAME=${RUN_NAME:-qwen2.5-3b-grpo-${TRAIN_DATA_TAG}-bs${ROLL_OUT_BATCH_SIZE}-n${N_SAMPLES_PER_PROMPT}-r${NUM_ROLLOUT}}
fi
RUN_ROOT=${RUN_ROOT:-${OCEAN_ROOT}/runs/${RUN_NAME}}

export APRIL_ROOT
export MEGATRON_ROOT
export SGLANG_PYTHON_ROOT
export QWEN25_HF_CHECKPOINT
export QWEN25_REF_LOAD
export INPUT_DATA
export EVAL_DATA
export TRAIN_DATA_TAG
export EVAL_DATA_TAG
export ROLL_OUT_BATCH_SIZE
export N_SAMPLES_PER_PROMPT
export NUM_ROLLOUT
export ROLLOUT_MAX_RESPONSE_LEN
export EVAL_INTERVAL
export SGLANG_MEM_FRACTION
export SLIME_SGLANG_DIRECT
export DEBUG_ROLLOUT_ONLY
export PARTIAL_ROLLOUT
export OVERSAMPLING_BATCH_SIZE
export RUN_NAME
export RUN_ROOT

bash "${APRIL_ROOT}/scripts/run-qwen2.5-3B.train-math-level4.sh"
