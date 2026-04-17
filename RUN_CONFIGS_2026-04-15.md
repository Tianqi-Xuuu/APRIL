# Run Config Snapshot (2026-04-15)

## Shared Base Config

- Launch entry: `modal_app.py::train_qwen25_level2_base`
- Model: `Qwen2.5-3B`
- Train seed: `1234`
- Rollout seed (runtime arg): `42`
- Rollout settings: `rollout_batch_size=8`, `n_samples_per_prompt=8`, `num_rollout=100`, `eval_interval=5`
- Max response length: `rollout_max_response_len=4096`
- PPO behavior-logprob path: `use_behavior_logprobs_for_ppo_clip=true`
- Checkpoint saving: `disable_ckpt_save=true` (no ckpt save)

## Dataset Config

- Train dataset: `/root/data/math_level12/math-level2-train.parquet`
- Eval dataset (subset): `/root/data/math_level12/math-level2-test.subset64.seed1234.parquet`
- Eval mode: subset eval (level2 test subset64)

## Run-by-Run Config

| Run name | Mode | partial_rollout | oversampling_batch_size | LR | min_lr | clip_grad | eps_clip | eps_clip_high |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen25-modal-level2-bs8-n8-r100-prov100-collapse-test` | prov=10 (aggressive no-clip test) | true | 80 | 5e-6 | 1e-7 | 1000 | 100 | 100 |
| `qwen25-modal-level2-bs8-n8-r100-nonprov-lr5e-6` | non-prov (aggressive no-clip test) | false | 8 | 5e-6 | 1e-7 | 1000 | 100 | 100 |
| `qwen25-modal-level2-bs8-n8-r100-prov5-lr5e-6` | prov=5 (aggressive no-clip test) | true | 40 | 5e-6 | 1e-7 | 1000 | 100 | 100 |
| `qwen25-modal-level2-bs8-n8-r100-nonprov-lr5e-6-normalclip` | non-prov (normal clip) | false | 8 | 5e-6 | 1e-7 | 1.0 | 0.2 | 0.28 |
| `qwen25-modal-level2-bs8-n8-r100-prov5-lr5e-6-normalclip` | prov=5 (normal clip) | true | 40 | 5e-6 | 1e-7 | 1.0 | 0.2 | 0.28 |
| `qwen25-modal-level2-bs8-n8-r100-prov10-lr5e-6-normalclip` | prov=10 (normal clip) | true | 80 | 5e-6 | 1e-7 | 1.0 | 0.2 | 0.28 |

## Notes

- `prov` is controlled by `partial_rollout=true` + `oversampling_batch_size / rollout_batch_size`.
- Here:
  - non-prov: `partial_rollout=false` (no provision path)
  - prov=5: `40 / 8 = 5`
  - prov=10: `80 / 8 = 10`
