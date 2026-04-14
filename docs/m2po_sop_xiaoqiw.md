# M2PO SOP (APRIL + GRPO)

## Why SOP (not PRD)
This task is implementation and experiment execution on an existing codebase. SOP is better than PRD because it gives step-by-step run and validation actions.

## Goal
Implement and run a minimal M2PO-style selective clip-masking method that only masks extreme outlier tokens.

## Scope
- Base: APRIL + GRPO pipeline already runnable.
- New behavior: in policy update, only among trust-region clipped tokens, mask top outliers by score `(log_ratio)^2`.
- Optional restriction: only apply to carry-over samples (detected by sample metadata).

## Implementation Design
1. Eligible token set T:
   - Reuse existing clipped indicator from GRPO PPO objective (`clipfrac > 0`).
2. Outlier score:
   - Use `(ppo_kl)^2` as proxy of `(log r)^2` where `log r = -ppo_kl`.
3. Selective masking:
   - Among eligible tokens, mask top-k by percentile (`m2po_mask_topk_percent`).
   - Keep a minimum mask count (`m2po_min_mask_tokens`) when eligible tokens exist.
4. Carry-over gating (optional):
   - If `m2po_only_carryover` is enabled, only tokens from samples with
     `start_rollout_id != final_rollout_id` are eligible.

## New CLI Flags
- `--m2po-enable`
- `--m2po-only-carryover`
- `--m2po-mask-topk-percent` (default 0.01)
- `--m2po-min-mask-tokens` (default 1)

## Run SOP
1. Start from your branch and existing APRIL run script.
2. Add M2PO flags, recommended initial config:
   - `--m2po-enable`
   - `--m2po-only-carryover`
   - `--m2po-mask-topk-percent 0.01`
   - `--m2po-min-mask-tokens 1`
3. Run short sanity training (small rollout count) and check logs:
   - `train/m2po_eligible_tokens`
   - `train/m2po_masked_tokens`
   - `train/m2po_masked_ratio`
4. If masked ratio is too high, reduce percentile; if too low, increase percentile.
5. Full run ablation:
   - Baseline APRIL+GRPO (no M2PO)
   - M2PO carry-over only
   - (Optional) M2PO on all clipped tokens

## Acceptance Criteria
- Training runs without crash.
- M2PO metrics are logged and non-zero under partial rollout.
- Masking mostly affects a small fraction of tokens.
- Throughput drop is limited while stability signals improve (clip-related tails).

## Task Checklist (Executed)
- [x] Confirm branch is `xiaoqiw`.
- [x] Locate loss and rollout metadata pipeline.
- [x] Add M2PO CLI arguments.
- [x] Implement selective clip-mask on outlier tokens in policy loss.
- [x] Add M2PO logging metrics.
- [ ] Run a full training ablation.
- [ ] Analyze quality-throughput tradeoff figures.

## Notes
Current implementation masks token contributions in policy loss only and keeps KL/entropy logic unchanged.
