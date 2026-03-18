# DAPO Partial Rollout Single-GPU Repro Notes (2026-03-18)

This note records the exact dataset, code paths, fixes, scripts, and observed batch-capacity results used to test DAPO-style partial rollout on a single A100 80GB GPU.

## Environment

- Repo: `/root/APRIL`
- Megatron-LM: `/root/Megatron-LM`
- GPU: `1 x NVIDIA A100 80GB PCIe`
- Date: `2026-03-18`

## Dataset Download

The dataset repo that worked is:

- `RyanYr/DAPO-Math-17k`

Download command:

```bash
hf download RyanYr/DAPO-Math-17k \
  --repo-type dataset \
  --local-dir /root/dapo-math-17k
```

Downloaded file used in APRIL:

- `/root/dapo-math-17k/data/train-00000-of-00001.parquet`

Basic dataset check:

- Rows: `17398`
- Important columns:
  - `source_prompt`
  - `answer`
  - `problem`
  - `solution`

APRIL can read the parquet file directly, so the run scripts use:

```bash
--prompt-data /root/dapo-math-17k/data/train-00000-of-00001.parquet
--input-key source_prompt
--label-key answer
--apply-chat-template
```

## Partial Rollout Code Paths

The main partial-rollout related code paths are:

- `slime/utils/arguments.py`
  - `--partial-rollout`
  - `--over-sampling-batch-size`
  - `--dynamic-sampling-filter-path`
  - `--debug-rollout-only`
- `slime/rollout/sglang_example.py`
  - partial rollout data-buffer requeue logic
  - over-sampling fetch logic
  - rollout request generation
- `scripts/partial_rollout/qwen/dapo/run-qwen3-4B-dapo-partial.sh`
  - official example using:
    - `--partial-rollout`
    - `--over-sampling-batch-size 64`
    - `--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`
- `README_zh.md`
  - high-level usage description

Useful implementation detail:

- In `slime/rollout/sglang_example.py`, the target fetch size becomes `over_sampling_batch_size` when a sampling filter is enabled.
- With partial rollout enabled, unfinished groups can be pushed back into the buffer and resumed later.

## Fixes Required For Single-GPU Partial Rollout

### Fix 1: Make wandb optional

Files changed:

- `slime/utils/wandb_utils.py`
- `slime/backends/megatron_utils/initialize.py`
- `slime/backends/megatron_utils/data.py`
- `slime/backends/megatron_utils/model.py`
- `slime/ray/buffer.py`

Reason:

- Non-wandb runs should not require `wandb` to be installed.

### Fix 2: SGLang router Prometheus port conflict

File changed:

- `slime/ray/rollout.py`

Reason:

- Router startup could fail because Prometheus metrics tried to bind a busy port.

Fix:

- Use a dynamically allocated local metrics port.

### Fix 3: Single-GPU partial rollout invalid CUDA device ordinal

File changed:

- `slime/backends/sglang_utils/sglang_engine.py`

Problem:

- In partial-rollout rollout-only runs, `base_gpu_id` could be inferred as `1` on a machine that only had `cuda:0`.
- This crashed SGLang with:

```text
RuntimeError: CUDA error: invalid device ordinal
```

Fix:

- Prefer `ray.get_gpu_ids()` first.
- Fallback to `CUDA_VISIBLE_DEVICES`.
- Only use the old placement-based heuristic if neither is available.

This is the key single-GPU bug fix for this round of testing.

## Repro Scripts Added

- `scripts/run-qwen3-1.7B.partial-rollout-bench.sh`
- `scripts/run-qwen3-0.6B.partial-rollout-bench.sh`

Shared settings in these scripts:

- `--partial-rollout`
- `--debug-rollout-only`
- `--rm-type deepscaler`
- `--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`
- `--rollout-max-response-len 256`
- `--n-samples-per-prompt 8`
- `--sglang-mem-fraction-static 0.7`

Both scripts also intentionally use a non-existent `--load` path so APRIL falls back to clean weight-only loading from `ref_load`.

## Commands Used

### Qwen3-1.7B

Default DAPO-style partial rollout:

```bash
bash /root/APRIL/scripts/run-qwen3-1.7B.partial-rollout-bench.sh
```

Higher batch examples:

```bash
ROLLOUT_BATCH_SIZE=64 OVER_SAMPLING_BATCH_SIZE=128 \
  bash /root/APRIL/scripts/run-qwen3-1.7B.partial-rollout-bench.sh

ROLLOUT_BATCH_SIZE=128 OVER_SAMPLING_BATCH_SIZE=256 \
  bash /root/APRIL/scripts/run-qwen3-1.7B.partial-rollout-bench.sh
```

### Qwen3-0.6B

```bash
bash /root/APRIL/scripts/run-qwen3-0.6B.partial-rollout-bench.sh
```

Higher batch examples:

```bash
ROLLOUT_BATCH_SIZE=128 OVER_SAMPLING_BATCH_SIZE=256 \
  bash /root/APRIL/scripts/run-qwen3-0.6B.partial-rollout-bench.sh

ROLLOUT_BATCH_SIZE=256 OVER_SAMPLING_BATCH_SIZE=512 \
  bash /root/APRIL/scripts/run-qwen3-0.6B.partial-rollout-bench.sh
```

## Observed Capacity Results

These results are for:

- `rollout-max-response-len=256`
- `n-samples-per-prompt=8`
- single GPU
- rollout-only debug mode

### Qwen3-1.7B

Confirmed to launch and enter rollout initialization at:

- `rollout_batch_size=32`, `over_sampling_batch_size=64`
- `rollout_batch_size=48`, `over_sampling_batch_size=96`
- `rollout_batch_size=64`, `over_sampling_batch_size=128`
- `rollout_batch_size=96`, `over_sampling_batch_size=192`
- `rollout_batch_size=128`, `over_sampling_batch_size=256`

Practical conclusion:

- On this machine and config, `Qwen3-1.7B` supports at least `rollout_batch_size=128` in the tested range.

### Qwen3-0.6B

Confirmed to launch and enter rollout initialization at:

- `rollout_batch_size=128`, `over_sampling_batch_size=256`
- `rollout_batch_size=256`, `over_sampling_batch_size=512`

Practical conclusion:

- On this machine and config, `Qwen3-0.6B` supports at least `rollout_batch_size=256` in the tested range.

## Important Caveat

These are not full end-to-end PPO training benchmarks.

They are:

- DAPO-Math-17k based
- partial-rollout enabled
- rollout-only capacity checks
- with `rollout-max-response-len=256`

If you change any of the following, the stable batch ceiling may drop:

- `rollout-max-response-len`
- `n-samples-per-prompt`
- `sglang-mem-fraction-static`
- model size
- whether actor training is colocated

## Recommended Next Step

If you want a more realistic production estimate, run the same scripts with:

- the target `rollout-max-response-len`
- your target `n-samples-per-prompt`
- one fixed batch size per model

and then record:

- `partial_rollout/rollout_time`
- `total_tokens`
- peak `nvidia-smi` memory

That will give you a more deployment-relevant ceiling than a startup-only capacity sweep.
