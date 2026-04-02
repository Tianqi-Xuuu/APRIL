# APRIL Window Experiments

This folder packages the experiment plan for the window-cost idea into runnable scripts.

## What is included

- `run_rollout_window_sweep.sh`
  - rollout-only sweep over different window sizes `N`
  - measures the overall tradeoff between fewer long-tail stalls and higher per-step system cost
- `run_tail_window_sweep.sh`
  - repeats the rollout-only sweep across several `rollout_max_response_len` settings
  - tests whether the best window shifts as the tail gets heavier
- `run_train_window_sweep.sh`
  - train-mode sweep over a smaller set of window sizes
  - checks whether the system-optimal window is still acceptable once off-policy side effects are included
- `analyze_window_experiments.py`
  - merges rollout metrics, debug rollout pickles, and decode trace logs
  - writes `summary.csv`, `summary.json`, and several plots

## Main outputs

The analysis script writes these plots under each run root's `analysis/` directory:

- `goodput_vs_window.png`
- `c_hat_vs_window.png`
- `t_hat_vs_window.png`
- `off_policy_vs_window.png`
- `carryover_vs_window.png`
- `tokens_per_sec_vs_window.png`

Here:

- `c_hat = rollout_time / decode_batches_per_rollout_proxy`
- `t_hat = decode_batches_per_rollout_proxy`
- `group_goodput = rollout_batch_size / rollout_time`

`c_hat` and `t_hat` are proxies, not kernel-level ground truth. They are most trustworthy for rollout-only runs without eval traffic mixed into the same log.

## Example commands

Rollout-only sweep:

```bash
bash /root/APRIL/idea/run_rollout_window_sweep.sh
```

Tail sweep:

```bash
TAIL_RESPONSE_LENS="256 1024 4096" \
bash /root/APRIL/idea/run_tail_window_sweep.sh
```

Train-mode sweep:

```bash
WINDOW_RATIOS="1.0 1.5 2.0 2.5 3.0" \
bash /root/APRIL/idea/run_train_window_sweep.sh
```

Dry-run smoke test:

```bash
DRY_RUN=1 WINDOW_RATIOS="1.0 2.0" \
bash /root/APRIL/idea/run_rollout_window_sweep.sh
```

Analyze existing runs:

```bash
python /root/APRIL/idea/analyze_window_experiments.py \
  --run-root-base /root/APRIL/runs \
  --run-name-regex 'qwen2.5-3b-train-math-level2-shortprompt.*' \
  --output-dir /root/APRIL/idea/test_output \
  --skip-decode-proxy
```

## Notes

- All runner scripts save `debug_rollout/rollout_*.pkl` so the carry-over and response-length analysis can be reproduced later.
- The train-mode script reuses APRIL's existing `plot_offpolicy_training_compare.py` to produce reward and eval curves after the sweep.
- If you want a finer window grid, set `OVER_SAMPLING_BATCH_SIZES` directly or change `WINDOW_RATIOS`.
