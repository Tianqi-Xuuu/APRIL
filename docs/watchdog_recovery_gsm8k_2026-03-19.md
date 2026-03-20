# Watchdog Recovery Notes

日期：`2026-03-19`

这份文档记录 `Qwen3-1.7B + GSM8K + single GPU` 训练里，`sglang rollout worker :10000` 掉线、router `PoisonError`、以及 watchdog 自动恢复的处理过程，方便后续复现。

## 1. 问题现象

训练在 rollout 或 eval 过程中，`sglang` 的唯一 rollout worker 会先失联，随后 `sglang-router` 进入异常状态。典型日志特征：

- `Generate request to http://172.17.0.2:10000 failed`
- `Removing failed worker: http://172.17.0.2:10000`
- `PoisonError`

没有 watchdog 时，常见表现是：

- `train.py` 主进程还活着
- GPU 上已经没有 running process
- `job_output.log` 长时间不再推进
- Ray job 没有及时 fail fast

也就是：训练会静默卡住。

## 2. 根因排序

本地复盘的故障链更像：

`rollout worker crash -> router bug -> driver stall`

而不是：

`router bug -> worker crash`

`PoisonError` 更像 router 在 failed worker removal 路径上的二次故障，不是第一因。

## 3. 加入的 watchdog 能力

当前主脚本：

- [run-qwen3-1.7B.offpolicy-matrix-bs-half-r30.sh](/root/APRIL/scripts/run-qwen3-1.7B.offpolicy-matrix-bs-half-r30.sh)
- [run-qwen3-1.7B.offpolicy-matrix-bs-1p5x-r30.sh](/root/APRIL/scripts/run-qwen3-1.7B.offpolicy-matrix-bs-1p5x-r30.sh)

现在都具备以下逻辑：

- `ray job submit --no-wait` 提交后，后台持续 `tail` job logs
- 周期性检查：
  - Ray job 状态
  - `job_output.log` 是否继续增长
  - router driver log 是否出现 `Removing failed worker` / `PoisonError`
  - rollout worker `:10000` 是否在健康启动后持续消失
- 发生异常时：
  - 停止当前 Ray job
  - 清理当前 `run_root` 相关的旧 `train.py` / `ray job logs`
  - 重启 Ray cluster
  - 从 `run_root` 最新 checkpoint 自动 resume

## 4. 关键修复点

这次 watchdog 最终修到可用，靠的是这几条：

1. 只在 worker 真正健康启动后再做健康检查

- 通过 `Uvicorn running on http://...:10000` 确认 worker ready
- 避免启动初期的误判

2. `:10000` 消失要连续多次才判故障

- 用 `WATCHDOG_MISSING_WORKER_POLLS`
- 避免瞬时端口抖动导致误重启

3. 故障注入只打一次

- 用 `.watchdog_fault_injected` 标记
- 避免每次重启后又立刻重复注入

4. 重启前按 `run_root` 精确清理残留训练进程

- 不是只 `ray stop --force`
- 还会枚举并清理同一 `run_root` 对应的 `train.py` / `ray job logs`

5. 恢复测试时，checkpoint 参数必须和训练配置一致

- 例如用 `r30` 训练出来的 checkpoint，就必须用 `NUM_ROLLOUT=30` 恢复
- 否则 Megatron 会报 `OptimizerParamScheduler ... do not match`

## 5. 已验证的恢复链

在 `r30` 配置上，已经实际验证通过：

1. 从 `iter_0000024` 恢复
2. 主动故障注入，杀掉 `:10000` rollout worker
3. watchdog 检测到 worker 消失
4. 自动重启为 `attempt 1`
5. `attempt 1` 再次加载：

```text
loading distributed checkpoint ... at iteration 24
```

6. 恢复后重新进入正常 rollout generation

也就是：

`worker crash -> watchdog detect -> restart -> resume checkpoint -> continue rollout`

## 6. 小 eval subset

为了避免每次 eval 太重，现在训练脚本默认切到固定小评估集：

- `/root/gsm8k/data/gsm8k-test.subset256.seed1234.parquet`

默认评估参数：

- `EVAL_DATA=/root/gsm8k/data/gsm8k-test.subset256.seed1234.parquet`
- `N_SAMPLES_PER_EVAL_PROMPT=1`

这样每次 eval 更轻，而且不同实验组之间仍然公平。

## 7. 推荐启动方式

只跑 provision 组：

```bash
RUN_GROUPS=2p0x,3p0x \
EVAL_DATA=/root/gsm8k/data/gsm8k-test.subset256.seed1234.parquet \
N_SAMPLES_PER_EVAL_PROMPT=1 \
bash /root/APRIL/scripts/run-qwen3-1.7B.offpolicy-matrix-bs-half-r30.sh
```

如果要调 watchdog：

```bash
WATCHDOG_ENABLED=1
WATCHDOG_POLL_SECONDS=10
WATCHDOG_STALL_SECONDS=120
WATCHDOG_HEALTHCHECK_GRACE_SECONDS=30
WATCHDOG_MISSING_WORKER_POLLS=2
WATCHDOG_MAX_RESTARTS=5
```

## 8. 备注

- `/tmp/ray/session...` 接近满盘时，Ray 更容易出额外问题，建议定期清理
- 当前 watchdog 目标是“避免静默卡死并自动恢复到最近 checkpoint”
- 它不是精确恢复到故障发生时刻，而是恢复到最近一次 save checkpoint
