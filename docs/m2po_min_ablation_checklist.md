# Baseline vs M2PO Minimal Ablation Checklist

## Which script should I use?
- Use scripts/run-qwen3-4B-amd.sh only on AMD/ROCm machines.
- Use scripts/run-qwen3-4B.sh on NVIDIA/CUDA machines.
- For your proposal hardware (PSC A100), prefer scripts/run-qwen3-4B.sh.

## One-time prep
- Confirm you are on branch `xiaoqiw`.
- Verify checkpoint/data paths in the selected script.

## Minimal A/B commands

### Option A: NVIDIA/CUDA
1. Baseline (APRIL + GRPO)
```bash
cd /shared/scratch/0/home/v_xiaoqi_wu/llmsys_project/APRIL
EXTRA_ALGO_ARGS="" bash scripts/run-qwen3-4B.sh
```

2. M2PO (carry-over only, mask top 1% clipped outliers)
```bash
cd /shared/scratch/0/home/v_xiaoqi_wu/llmsys_project/APRIL
EXTRA_ALGO_ARGS="--m2po-enable --m2po-only-carryover --m2po-mask-topk-percent 0.01 --m2po-min-mask-tokens 1" \
  bash scripts/run-qwen3-4B.sh
```

### Option B: AMD/ROCm
1. Baseline (APRIL + GRPO)
```bash
cd /shared/scratch/0/home/v_xiaoqi_wu/llmsys_project/APRIL
EXTRA_ALGO_ARGS="" bash scripts/run-qwen3-4B-amd.sh
```

2. M2PO (carry-over only, mask top 1% clipped outliers)
```bash
cd /shared/scratch/0/home/v_xiaoqi_wu/llmsys_project/APRIL
EXTRA_ALGO_ARGS="--m2po-enable --m2po-only-carryover --m2po-mask-topk-percent 0.01 --m2po-min-mask-tokens 1" \
  bash scripts/run-qwen3-4B-amd.sh
```

## What to compare (minimal)
- train/m2po_eligible_tokens
- train/m2po_masked_tokens
- train/m2po_masked_ratio
- train/pg_clipfrac
- partial_rollout/tokens_throughput

## Fast tuning knobs
- Too aggressive masking: reduce --m2po-mask-topk-percent to 0.005.
- Too weak masking: increase --m2po-mask-topk-percent to 0.02.
- For strict outlier-only behavior, keep --m2po-only-carryover enabled.
