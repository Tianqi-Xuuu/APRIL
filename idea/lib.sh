#!/bin/bash

ratio_label_from_sizes() {
    local rollout_bs="$1"
    local over_bs="$2"
    python - "$rollout_bs" "$over_bs" <<'PY'
import sys

rollout_bs = int(sys.argv[1])
over_bs = int(sys.argv[2])
ratio = over_bs / rollout_bs
if abs(ratio - round(ratio)) < 1e-9:
    print(f"{int(round(ratio))}p0x")
else:
    text = f"{ratio:.4f}".rstrip("0").rstrip(".")
    print(text.replace(".", "p") + "x")
PY
}

batch_sizes_from_ratios() {
    local rollout_bs="$1"
    shift
    python - "$rollout_bs" "$@" <<'PY'
import math
import sys

rollout_bs = int(sys.argv[1])
ratios = [float(arg) for arg in sys.argv[2:]]
batch_sizes = sorted({max(rollout_bs, int(math.ceil(rollout_bs * ratio))) for ratio in ratios})
print(" ".join(str(size) for size in batch_sizes))
PY
}

ensure_baseline_batch_size() {
    local rollout_bs="$1"
    shift
    python - "$rollout_bs" "$@" <<'PY'
import sys

rollout_bs = int(sys.argv[1])
sizes = [int(arg) for arg in sys.argv[2:]]
if rollout_bs not in sizes:
    sizes.append(rollout_bs)
sizes = sorted(set(sizes))
print(" ".join(str(size) for size in sizes))
PY
}

quote_cmd() {
    printf '%q ' "$@"
    printf '\n'
}
