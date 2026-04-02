#!/bin/bash

set -euo pipefail

IDEA_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

MODEL_TAG=${MODEL_TAG:-qwen3-1.7b}
TASK_TAG=${TASK_TAG:-gsm8k}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
TAIL_RESPONSE_LENS=${TAIL_RESPONSE_LENS:-"256 1024 4096"}
RUN_ANALYSIS=${RUN_ANALYSIS:-1}
DRY_RUN=${DRY_RUN:-0}
FORCE_RAY_RESTART=${FORCE_RAY_RESTART:-1}
RUN_ROOT_BASE=${RUN_ROOT_BASE:-/root/APRIL/runs/idea-tail-window-${MODEL_TAG}-${TASK_TAG}-bs${ROLLOUT_BATCH_SIZE}}

mkdir -p "${RUN_ROOT_BASE}"

first=1
for RESPONSE_LEN in ${TAIL_RESPONSE_LENS}; do
    child_force_restart=0
    if [ "${first}" = "1" ]; then
        child_force_restart="${FORCE_RAY_RESTART}"
        first=0
    fi

    echo "=== Tail sweep len=${RESPONSE_LEN} ==="
    env \
        ROLLOUT_MAX_RESPONSE_LEN="${RESPONSE_LEN}" \
        RUN_ROOT_BASE="${RUN_ROOT_BASE}/len${RESPONSE_LEN}" \
        RUN_ANALYSIS=0 \
        FORCE_RAY_RESTART="${child_force_restart}" \
        DRY_RUN="${DRY_RUN}" \
        MODEL_TAG="${MODEL_TAG}" \
        TASK_TAG="${TASK_TAG}" \
        ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}" \
        bash "${IDEA_DIR}/run_rollout_window_sweep.sh"
done

if [ "${RUN_ANALYSIS}" = "1" ] && [ "${DRY_RUN}" != "1" ]; then
    python "${IDEA_DIR}/analyze_window_experiments.py" \
        --run-root-base "${RUN_ROOT_BASE}" \
        --recursive \
        --output-dir "${RUN_ROOT_BASE}/analysis"
fi
