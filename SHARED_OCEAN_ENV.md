# Shared Ocean Environment

This document is for other users who want to reuse the shared APRIL environment under `/ocean/projects/cis260009p/txu7/april`.

## Shared Assets

### Code

Pull this repository yourself. Do not depend on `/jet/home/txu7/APRIL`.

### Shared Python Environment

- Venv: `/ocean/projects/cis260009p/txu7/april/venvs/april-py310`
- Megatron-LM: `/ocean/projects/cis260009p/txu7/april/src/Megatron-LM`
- SGLang Python package: `/ocean/projects/cis260009p/txu7/april/src/sglang/python`

### Shared Models

#### Qwen3-1.7B

- HF checkpoint: `/ocean/projects/cis260009p/txu7/april/models/Qwen3-1.7B`
- Megatron `torch_dist`: `/ocean/projects/cis260009p/txu7/april/models/Qwen3-1.7B_torch_dist`

#### Qwen2.5-3B

- HF checkpoint: `/ocean/projects/cis260009p/txu7/april/models/Qwen2.5-3B`
- Megatron `torch_dist`: `/ocean/projects/cis260009p/txu7/april/models/Qwen2.5-3B_torch_dist`

If `Qwen2.5-3B_torch_dist` does not exist yet, you need to generate it on a GPU node before training.

### Shared Datasets

#### GSM8K

- Train: `/ocean/projects/cis260009p/txu7/april/data/gsm8k/gsm8k-train.parquet`
- Test: `/ocean/projects/cis260009p/txu7/april/data/gsm8k/gsm8k-test.parquet`
- Test subset256: `/ocean/projects/cis260009p/txu7/april/data/gsm8k/gsm8k-test.subset256.seed1234.parquet`

#### competition_math

- Level 2 minimal train: `/ocean/projects/cis260009p/txu7/april/data/math_level12/math-level2-train.min1.parquet`
- Level 2 minimal test: `/ocean/projects/cis260009p/txu7/april/data/math_level12/math-level2-test.min1.parquet`

## Important Rule

Reuse the shared environment as read-only.

Do not write to:

- `/ocean/projects/cis260009p/txu7/april/models`
- `/ocean/projects/cis260009p/txu7/april/data`
- `/ocean/projects/cis260009p/txu7/april/venvs`
- `/ocean/projects/cis260009p/txu7/april/src`
- `/ocean/projects/cis260009p/txu7/april/runs`

Use your own directories for:

- `RUN_ROOT`
- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- `TMPDIR`
- `RAY_TMPDIR`

## Basic Setup

After pulling the repo:

```bash
module load gcc/13.3.1-p20240614 cuda/12.6.1
source /ocean/projects/cis260009p/txu7/april/venvs/april-py310/bin/activate

export MEGATRON_ROOT=/ocean/projects/cis260009p/txu7/april/src/Megatron-LM
export SGLANG_PYTHON_ROOT=/ocean/projects/cis260009p/txu7/april/src/sglang/python

export HF_HOME=/ocean/projects/cis260009p/$USER/april/hf
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TMPDIR=/ocean/projects/cis260009p/$USER/t
export RAY_TMPDIR=/ocean/projects/cis260009p/$USER/r

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR" "$RAY_TMPDIR"
```

## Example: Qwen3-1.7B GSM8K Smoke

```bash
export APRIL_ROOT=$PWD
export RUN_ROOT=/ocean/projects/cis260009p/$USER/april/runs/qwen3-smoke-1
export QWEN3_HF_CHECKPOINT=/ocean/projects/cis260009p/txu7/april/models/Qwen3-1.7B
export QWEN3_REF_LOAD=/ocean/projects/cis260009p/txu7/april/models/Qwen3-1.7B_torch_dist
export INPUT_DATA=/ocean/projects/cis260009p/txu7/april/data/gsm8k/gsm8k-train.parquet
export DEBUG_ROLLOUT_ONLY=1
export SLIME_SGLANG_DIRECT=1
export NUM_ROLLOUT=1
export ROLL_OUT_BATCH_SIZE=1
export N_SAMPLES_PER_PROMPT=1
export ROLLOUT_MAX_RESPONSE_LEN=256

bash scripts/run-qwen3-1.7B.no-partial-dapo-bench.sh
```

## Example: Qwen2.5-3B Minimal Train

```bash
export APRIL_ROOT=$PWD
export RUN_ROOT=/ocean/projects/cis260009p/$USER/april/runs/qwen25-min-train-1
export QWEN25_HF_CHECKPOINT=/ocean/projects/cis260009p/txu7/april/models/Qwen2.5-3B
export QWEN25_REF_LOAD=/ocean/projects/cis260009p/txu7/april/models/Qwen2.5-3B_torch_dist
export INPUT_DATA=/ocean/projects/cis260009p/txu7/april/data/math_level12/math-level2-train.min1.parquet
export EVAL_DATA=/ocean/projects/cis260009p/txu7/april/data/math_level12/math-level2-test.min1.parquet
export DEBUG_ROLLOUT_ONLY=0
export SLIME_SGLANG_DIRECT=1
export NUM_ROLLOUT=1
export EVAL_INTERVAL=1
export ROLL_OUT_BATCH_SIZE=1
export N_SAMPLES_PER_PROMPT=1
export ROLLOUT_MAX_RESPONSE_LEN=32

bash scripts/run-qwen2.5-3B.train-math-level4.sh
```

## If `Qwen2.5-3B_torch_dist` Is Missing

Run this on a GPU node:

```bash
PYTHONPATH="$MEGATRON_ROOT" python tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint /ocean/projects/cis260009p/txu7/april/models/Qwen2.5-3B \
  --save /ocean/projects/cis260009p/txu7/april/models/Qwen2.5-3B_torch_dist
```
