# PPO Clip 改动说明（Behavior Logprob 路径）

## 目标
让 `train/pg_clipfrac` 基于 rollout 时的真实 behavior policy logprob 计算，而不是训练时再用当前 actor 重算 `old_log_probs`。

## 核心改动

1. 新增开关  
在参数中新增：
`--use-behavior-logprobs-for-ppo-clip`  
关闭时保持原行为，打开时启用 behavior logprob 路径。

2. Rollout 侧采集真实 logprob  
在 `slime/rollout/sglang_example.py`：
- 请求 sglang 返回 token 级 logprob（top-level 字段）
- 解析 `output_token_logprobs`
- partial rollout/continuation 场景下按 token 顺序拼接各段 logprob
- 写入 `sample.metadata["behavior_log_probs"]`

3. 训练数据链路透传  
把 `behavior_log_probs` 从 sample metadata 透传到训练 batch：
- `slime/ray/buffer.py`
- `slime/backends/utils/data.py`
- `slime/backends/megatron_utils/model.py`

4. PPO loss 使用 behavior logprob  
在 `slime/backends/megatron_utils/loss.py`：
- 开关开启且 batch 中有 `behavior_log_probs` 时，使用其作为 `old_log_probs`
- 否则回退到原先逻辑
- 修复 non-GSPO 分支中 `old_log_probs` 被错误覆盖回 `batch["log_probs"]` 的问题

5. 诊断日志增强  
新增/强化以下日志以便判定路径是否生效：
- `train/pg_clipfrac`
- `train/ppo_kl`
- `train/ppo_kl_abs`
- `train/old_log_probs_behavior_sample_frac`

## 最小验证
新增静态回归测试：
`tests/test_behavior_clip_path_static.py`

用于检查 non-GSPO 路径不会再把 `old_log_probs` 错误回退到当前 logprob。

## 使用方式
训练命令增加：

```bash
--use-behavior-logprobs-for-ppo-clip
```

建议同时观察：
- `train/old_log_probs_behavior_sample_frac`（样本级覆盖）
- `train/pg_clipfrac` 与 `train/ppo_kl_abs`（clip 是否被触发）

## 备注
`train/old_log_probs_is_behavior` 在动态 batch 下更接近 microbatch 统计，不宜直接当成样本覆盖率解读。样本覆盖请优先看 `train/old_log_probs_behavior_sample_frac`。
