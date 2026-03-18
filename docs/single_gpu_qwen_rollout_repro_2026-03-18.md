# Single-GPU Qwen Rollout Repro Notes (2026-03-18)

This note records the fixes, commands, and test results used to make APRIL run on a single GPU with Qwen3 models.

## Environment

- Repo: `/root/APRIL`
- Megatron-LM: `/root/Megatron-LM`
- GPU: `1 x NVIDIA A100 80GB PCIe`
- Date: `2026-03-18`

## Files Changed

- `slime/utils/wandb_utils.py`
- `slime/backends/megatron_utils/initialize.py`
- `slime/backends/megatron_utils/data.py`
- `slime/backends/megatron_utils/model.py`
- `slime/ray/buffer.py`
- `slime/ray/rollout.py`
- `scripts/run-qwen3-4B.sh`
- `scripts/run-qwen3-1.7B.single-gpu.sh`
- `scripts/run-qwen3-1.7B.single-gpu-batch8.sh`
- `scripts/run-qwen3-0.6B.single-gpu-batch8.sh`
- `data/minimal/train.jsonl`
- `data/minimal/eval.jsonl`
- `data/minimal/train.batch8.jsonl`

## Fix 1: Make wandb optional

Problem:

- Training code imported `wandb` directly.
- If `wandb` was not installed, even a non-wandb run failed.

Fix:

- Added `slime/utils/wandb_utils.py`.
- Replaced direct imports with guarded access.
- If `--use-wandb` is not passed, the code now runs without requiring `wandb`.

Verification:

- `python -m py_compile` passed for the modified Python files.

## Fix 2: Single-GPU safe router startup

Problem:

- `sglang_router` tried to bind a Prometheus metrics port and sometimes crashed with:

```text
failed to install Prometheus metrics exporter: FailedToCreateHTTPListener("Address already in use (os error 98)")
```

Root cause:

- In this installed `sglang_router` version, passing `prometheus_port=None` was not enough to reliably disable metrics binding.

Fix:

- In `slime/ray/rollout.py`, router startup was changed to allocate a dynamic free Prometheus port:

```python
router_args = RouterArgs(
    host=self.args.sglang_router_ip,
    port=self.args.sglang_router_port,
    balance_abs_threshold=0,
    prometheus_port=find_available_port(random.randint(28000, 29000)),
    prometheus_host="127.0.0.1",
)
```

Why this fix:

- It avoids the fixed-port collision while keeping router startup logic small and local.

## Fix 3: Avoid broken `slime` checkpoint metadata during tests

Problem:

- Some test runs failed because an old save directory contained an invalid or partial:

```text
latest_checkpointed_iteration.txt
```

- This caused errors such as:

```text
AssertionError: error parsing metadata file /root/Qwen3-1.7B_slime/latest_checkpointed_iteration.txt
```

Fix used for reproducible rollout tests:

- For batch-size benchmark scripts, intentionally set `--load` to a non-existent path.
- `slime.utils.arguments.parse_args()` then automatically falls back to:
  - `args.load = args.ref_load`
  - `args.finetune = True`
  - `args.no_load_optim = True`
  - `args.no_load_rng = True`

This avoids restoring optimizer/RNG state and only loads model weights from the clean `torch_dist` checkpoint.

Example:

```bash
--load /root/.slime-nonexistent-qwen3-1.7B-batch8-load
```

## Fix 4: Safer process cleanup

Problem:

- Some original scripts used:

```bash
pkill -9 python
```

- This kills every Python process on the machine, not just APRIL.

Safer choice used here:

- Use `ray stop --force || true`
- Avoid global `pkill -9 python`

Note:

- `ray stop --force` can still interrupt an in-progress checkpoint save.
- If you want to preserve save integrity, wait for the run to fully finish before stopping Ray.

## Model Download and Conversion

### Qwen3-1.7B

Downloaded to:

- `/root/Qwen3-1.7B`

Converted checkpoint:

- `/root/Qwen3-1.7B_torch_dist`

### Qwen3-0.6B

Downloaded with:

```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /root/Qwen3-0.6B
```

Converted with:

```bash
PYTHONPATH=/root/Megatron-LM python /root/APRIL/tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint /root/Qwen3-0.6B \
  --save /root/Qwen3-0.6B_torch_dist
```

Converted checkpoint:

- `/root/Qwen3-0.6B_torch_dist`

Verification:

```bash
cat /root/Qwen3-0.6B_torch_dist/latest_checkpointed_iteration.txt
```

Expected output:

```text
release
```

## Benchmark Scripts Added

### 1.7B

- `scripts/run-qwen3-1.7B.single-gpu.sh`
- `scripts/run-qwen3-1.7B.single-gpu-batch8.sh`

### 0.6B

- `scripts/run-qwen3-0.6B.single-gpu-batch8.sh`

## Benchmark Dataset

Used a tiny local prompt file for repeatable rollout timing:

- `data/minimal/train.batch8.jsonl`

This file contains `8` prompts.

Benchmark setting:

- `rollout-batch-size=8`
- `n-samples-per-prompt=2`
- Total generations per rollout: `16`
- `rollout-max-response-len=512`

## Commands Used

### Run 1.7B single-GPU batch benchmark

```bash
bash /root/APRIL/scripts/run-qwen3-1.7B.single-gpu-batch8.sh
```

### Run 0.6B single-GPU batch benchmark

```bash
bash /root/APRIL/scripts/run-qwen3-0.6B.single-gpu-batch8.sh
```

## Observed Rollout Results

### Qwen3-1.7B

Job:

- `raysubmit_bV7wf1PhLpA9EUPw`

Observed metrics:

- `partial_rollout/total_tokens = 8192`
- `partial_rollout/rollout_time = 11.638416051864624`
- `partial_rollout/tokens_throughput = 703.8758507595659`
- `perf/total_train_time = 23.483569145202637`

Source log:

- `/tmp/ray/session_latest/logs/job-driver-raysubmit_bV7wf1PhLpA9EUPw.log`

### Qwen3-0.6B

Job:

- `raysubmit_iGDkzXjiUuhgwgBT`

Observed metrics:

- `partial_rollout/total_tokens = 5985`
- `partial_rollout/rollout_time = 11.082377433776855`
- `partial_rollout/tokens_throughput = 540.0465771684444`
- `perf/total_train_time = 21.940219163894653`

Source log:

- `/tmp/ray/session_latest/logs/job-driver-raysubmit_iGDkzXjiUuhgwgBT.log`

Important note:

- `0.6B` often finished answers earlier, so it did not consume the full `16 x 512 = 8192` completion tokens in this run.
- That means the measured `0.6B` rollout time is an actual run result, but not a perfectly apples-to-apples “all requests forced to full length” comparison.

## Rollout Timing Interpretation

Code path:

- `slime/rollout/sglang_example.py`
- `slime/ray/buffer.py`
- `slime/backends/utils/data.py`

Key behavior:

- `rollout_time` is measured only around rollout generation.
- It scales mostly with:
  - number of active requests
  - completion length
  - how long requests stay alive together before tail effects

Why larger batch is efficient here:

- At `batch8 x samples2 = 16` concurrent generations, the GPU stays much busier.
- For `1.7B`, moving from the earlier tiny test (`2 x 2 = 4` generations) to `16` generations increased total generated tokens by about `4x`, but rollout time stayed near `11s`.

## Recommended Reproduction Order

1. Ensure the repo changes in this note are present.
2. Download HF weights.
3. Convert HF weights to `torch_dist`.
4. Verify `latest_checkpointed_iteration.txt` contains `release`.
5. Run the desired benchmark script.
6. Read metrics from the Ray job log under `/tmp/ray/session_latest/logs/`.

## Useful Grep Commands

```bash
rg -n "partial_rollout 0:|perf 0:|step 0:" /tmp/ray/session_latest/logs -g '*.out' -g '*.log'
```

```bash
rg -n "entrypoint command exited|SUCCEEDED|FAILED" /tmp/ray/session_latest/logs -g '*.log'
```

## Practical Takeaways

- `Qwen3-1.7B` is viable on a single 80GB GPU for this APRIL setup.
- `Qwen3-0.6B` is even more comfortable on memory and training-side model load.
- The router dynamic-port fix is important if you repeatedly stop/start Ray in the same environment.
- For repeatable single-GPU tests, use a non-existent `--load` path so APRIL automatically enters the clean finetune-style initialization path.
