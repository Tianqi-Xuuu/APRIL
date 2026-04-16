#!/bin/bash
#
# Run idea/run_rollout_window_sweep.sh once per HuggingFace weights directory
# (e.g. every .../hf/iter_* from a training run), using the same window grid.
#
# Primary metric after each sweep remains rollout group goodput; see
# idea/analyze_window_experiments.py (written under each sweep's RUN_ROOT_BASE/analysis).
#
# Required: set HF_SCAN_DIR to a directory that contains only iter_* checkpoint
# folders (typical layout: <run_root>/hf), or set HF_CKPT_LIST to explicit paths.
#
# Example (Modal-style paths, math level-2, Qwen2.5-3B, ratio 1.0..4.0 step 0.5):
#
#   export HF_SCAN_DIR=/root/runs/qwen25-modal-level2-bs8-n8-r50-hfonly/hf
#   export MODEL_TAG=qwen2.5-3b
#   export MODEL_SCRIPT=/root/APRIL/scripts/models/qwen2.5-3B.sh
#   export HF_CHECKPOINT=/root/models/Qwen2.5-3B
#   export REF_LOAD=/root/models/Qwen2.5-3B_torch_dist
#   export INPUT_DATA=/root/data/math_level12/math-level2-train.parquet
#   export TASK_TAG=math-level2
#   export ROLLOUT_BATCH_SIZE=8
#   export INCLUDE_BASE_HF=1
#   bash idea/run_rollout_window_sweep_all_ckpts.sh
#

set -euo pipefail

export PYTHONUNBUFFERED=1

IDEA_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${IDEA_DIR}/.." &>/dev/null && pwd)"

WINDOW_RATIO_MIN=${WINDOW_RATIO_MIN:-1.0}
WINDOW_RATIO_MAX=${WINDOW_RATIO_MAX:-4.0}
WINDOW_RATIO_STEP=${WINDOW_RATIO_STEP:-0.5}

WINDOW_RATIOS=${WINDOW_RATIOS:-}
if [ -z "${WINDOW_RATIOS}" ]; then
    WINDOW_RATIOS="$(
        python3 - <<PY
import os
start = float(os.environ.get("WINDOW_RATIO_MIN", "1.0"))
end = float(os.environ.get("WINDOW_RATIO_MAX", "4.0"))
step = float(os.environ.get("WINDOW_RATIO_STEP", "0.5"))
xs = []
x = start
while x <= end + 1e-9:
    xs.append(f"{x:g}")
    x += step
print(" ".join(xs))
PY
    )"
fi
export WINDOW_RATIOS

MODEL_TAG=${MODEL_TAG:-qwen2.5-3b}
MODEL_SCRIPT=${MODEL_SCRIPT:-${REPO_ROOT}/scripts/models/qwen2.5-3B.sh}
HF_CHECKPOINT=${HF_CHECKPOINT:-/root/Qwen2.5-3B}
REF_LOAD=${REF_LOAD:-/root/Qwen2.5-3B_torch_dist}
LOAD_PATH=${LOAD_PATH:-/root/.slime-nonexistent-${MODEL_TAG}-idea-rollout-window-load}

RUN_ROOT_BASE_PARENT=${RUN_ROOT_BASE_PARENT:-${REPO_ROOT}/runs/idea-rollout-window-${MODEL_TAG}-${TASK_TAG:-sweep}-bs${ROLLOUT_BATCH_SIZE:-8}-allckpts}
INCLUDE_BASE_HF=${INCLUDE_BASE_HF:-0}
# When 1, skip checkpoints whose run_status.csv already has 7 succeeded and 0 failed;
# otherwise re-run that checkpoint's sweep with RESUME_FAILED_WINDOW_SWEEP=1 (only re-run failed/missing windows).
RESUME_ONLY_FAILED_CKPTS=${RESUME_ONLY_FAILED_CKPTS:-0}

CKPT_PATHS_FILE="$(mktemp)"
cleanup() {
    rm -f "${CKPT_PATHS_FILE}"
}
trap cleanup EXIT

if [ -n "${HF_CKPT_LIST:-}" ]; then
    # shellcheck disable=SC2086
    printf '%s\n' ${HF_CKPT_LIST} >"${CKPT_PATHS_FILE}"
elif [ -n "${HF_SCAN_DIR:-}" ]; then
    if [ ! -d "${HF_SCAN_DIR}" ]; then
        echo "HF_SCAN_DIR is not a directory: ${HF_SCAN_DIR}" >&2
        exit 1
    fi
    python3 - "${HF_SCAN_DIR}" <<'PY' >"${CKPT_PATHS_FILE}"
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])


def sort_key(p: Path) -> tuple[int, str]:
    m = re.search(r"(\d+)$", p.name)
    return (int(m.group(1)) if m else 0, p.name)


candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("iter_")]
for p in sorted(candidates, key=sort_key):
    print(p.resolve())
PY
else
    echo "Set HF_SCAN_DIR (directory of iter_* HF exports) or HF_CKPT_LIST (space-separated paths)." >&2
    exit 1
fi

if [ ! -s "${CKPT_PATHS_FILE}" ] && [ "${INCLUDE_BASE_HF}" != "1" ]; then
    echo "No checkpoints found under ${HF_SCAN_DIR:-}; nothing to do." >&2
    exit 1
fi

mkdir -p "${RUN_ROOT_BASE_PARENT}"

export MODEL_TAG
export MODEL_SCRIPT
export REF_LOAD
export LOAD_PATH
export WINDOW_RATIOS

meta_path="${RUN_ROOT_BASE_PARENT}/multi_ckpt_sweep_meta.txt"
{
    echo "WINDOW_RATIOS=${WINDOW_RATIOS}"
    echo "HF_SCAN_DIR=${HF_SCAN_DIR:-}"
    echo "INCLUDE_BASE_HF=${INCLUDE_BASE_HF}"
    echo "HF_CHECKPOINT(base)=${HF_CHECKPOINT}"
    echo "--- checkpoints ---"
    if [ "${INCLUDE_BASE_HF}" = "1" ]; then
        echo "${HF_CHECKPOINT}"
    fi
    cat "${CKPT_PATHS_FILE}"
} | tee "${meta_path}"

run_one() {
    local ckpt_path="$1"
    local slug_override="${2:-}"
    local slug
    if [ -n "${slug_override}" ]; then
        slug="${slug_override}"
    else
        slug="$(python3 - "$ckpt_path" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1]).resolve()
name = path.name
if name.startswith("iter_"):
    m = re.match(r"iter_0*(\d+)$", name)
    print(f"iter{m.group(1)}" if m else re.sub(r"[^A-Za-z0-9_.-]+", "_", name))
else:
    print(re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:80])
PY
)"
    fi

    export HF_CHECKPOINT="${ckpt_path}"
    export RUN_ROOT_BASE="${RUN_ROOT_BASE_PARENT}/${slug}"
    if [ "${RESUME_ONLY_FAILED_CKPTS}" = "1" ]; then
        local rs="${RUN_ROOT_BASE}/run_status.csv"
        if [ -f "${rs}" ]; then
            local ok bad
            ok="$(grep -c ',succeeded,0' "${rs}" 2>/dev/null || true)"
            bad="$(grep -c ',failed,' "${rs}" 2>/dev/null || true)"
            ok="${ok:-0}"
            bad="${bad:-0}"
            if [ "${ok}" -eq 7 ] && [ "${bad}" -eq 0 ] 2>/dev/null; then
                echo "Skip checkpoint ${slug} (7/7 windows already succeeded)."
                return
            fi
        fi
        export RESUME_FAILED_WINDOW_SWEEP=1
    else
        export RESUME_FAILED_WINDOW_SWEEP=0
    fi
    echo ""
    echo "=============================="
    echo "Checkpoint: ${ckpt_path}"
    echo "RUN_ROOT_BASE: ${RUN_ROOT_BASE}"
    echo "=============================="
    bash "${IDEA_DIR}/run_rollout_window_sweep.sh"
}

if [ "${INCLUDE_BASE_HF}" = "1" ]; then
    run_one "${HF_CHECKPOINT}" "base_hf"
fi

while IFS= read -r line; do
    [ -z "${line}" ] && continue
    [ -d "${line}" ] || {
        echo "Skip missing directory: ${line}" >&2
        continue
    }
    run_one "${line}"
done <"${CKPT_PATHS_FILE}"

echo ""
echo "All sweeps finished. Per-checkpoint analysis is under each RUN_ROOT_BASE/*/analysis/"
echo "Parent: ${RUN_ROOT_BASE_PARENT}"
