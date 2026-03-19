# GSM8K Qwen No-Partial Rollout Repro

## 目标

这份文档记录 2026-03-18 在单卡环境下，把 `GSM8K` 适配到当前 APRIL/Qwen3 no-partial-rollout 流程的排错过程、修复点、有效配置和当前正式任务状态，供之后在别的环境复现。

## 关键结论

- `GSM8K` 最开始的异常超长生成，不是数据集天然太长，而是 `source_prompt` 数据格式和 `--apply-chat-template` 不匹配。
- 修复后，`Qwen3-1.7B` 和 `Qwen3-0.6B` 都能稳定跑 `GSM8K` 的 `batch_size=64`、`maxlen=4096`。
- 这组配置下：
  - `1.7B` 的 rollout 时间约 `82.96s`
  - `0.6B` 的 rollout 时间约 `83.51s`
  - `0.6B` 不比 `1.7B` 更快，且质量更不稳定

## 根因

### 错误现象

最开始用 `GSM8K` 跑时，出现了这些异常：

- sample rollout 开头直接是 `<|im_start|>assistant`
- 输出成了泛化客服话术或无关长文本
- `maxlen=20k` 时出现极长输出，看起来像死循环

### 真正原因

代码在 [data.py](/root/APRIL/slime/utils/data.py) 里会在 `apply_chat_template=True` 时执行：

```python
prompt = tokenizer.apply_chat_template(prompt, tools, tokenize=False, add_generation_prompt=True)
```

而：

- DAPO 的 `source_prompt` 本来就是消息列表，形如 `[{role: user, content: ...}]`
- 我们最早准备的 `GSM8K` `source_prompt` 是普通字符串

对 `Qwen3` tokenizer 来说，给普通字符串调用 `apply_chat_template`，会错误地产出只带 assistant 起始标记的内容，等于把用户问题吃掉了。

实际验证结果是：

```text
INPUT TYPE <class 'str'>
<|im_start|>assistant
```

所以异常超长生成并不是任务自然长尾，而是输入格式错了。

## 修复内容

### 1. 把 GSM8K prompt 改成消息列表

修改文件：

- [prepare_gsm8k_dataset.py](/root/APRIL/scripts/analysis/prepare_gsm8k_dataset.py)

修复前：

- `source_prompt` 是字符串

修复后：

- `source_prompt` 是：

```python
[{"role": "user", "content": prompt}]
```

修复后再次验证：

```text
<|im_start|>user
...题目文本...
<|im_end|>
<|im_start|>assistant
```

这才是当前流程需要的输入格式。

### 2. 修复 rollout logging 对字符串列表求平均导致的报错

修改文件：

- [data.py](/root/APRIL/slime/backends/megatron_utils/data.py)

问题：

- 为了保存 sample rollout，`rollout_data` 里新增了 `prompts` / `responses`
- 旧逻辑会把所有 list 都当成数值求平均
- 结果在 `log_rollout_data()` 里触发：

```text
TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

修复：

- 在聚合日志时跳过 `prompts` 和 `responses`

### 3. 新增 0.6B no-partial 对照脚本

新增文件：

- [run-qwen3-0.6B.no-partial-dapo-bench.sh](/root/APRIL/scripts/run-qwen3-0.6B.no-partial-dapo-bench.sh)

用途：

- 用和 `1.7B` 同一套 no-partial benchmark 逻辑跑 `0.6B`

## 相关提交

- `02b90b9` `Fix GSM8K chat formatting and rollout logging`
- `a54282b` `Fix rollout timing import for GSM8K smoke`
- `c0d4c64` `Record sample rollouts without wallclock metadata`

## 数据准备

### 生成 GSM8K parquet

命令：

```bash
python /root/APRIL/scripts/analysis/prepare_gsm8k_dataset.py \
  --output /root/gsm8k/data/gsm8k-train.parquet
```

输出字段：

- `source_prompt`
- `answer`
- `solution`
- `metadata`

当前输出文件：

- [/root/gsm8k/data/gsm8k-train.parquet](/root/gsm8k/data/gsm8k-train.parquet)

## 有效 smoke 配置

### Qwen3-1.7B

命令：

```bash
RUN_ROOT=/root/APRIL/runs/qwen3-1.7b-no-partial-gsm8k-smoke-len4096-bs64-fixed \
INPUT_DATA=/root/gsm8k/data/gsm8k-train.parquet \
NUM_ROLLOUT=1 \
ROLL_OUT_BATCH_SIZE=64 \
N_SAMPLES_PER_PROMPT=1 \
ROLLOUT_MAX_RESPONSE_LEN=4096 \
bash /root/APRIL/scripts/run-qwen3-1.7B.no-partial-dapo-bench.sh
```

结果：

- `rollout_time ≈ 82.96s`
- `tokens_throughput ≈ 761.07 tok/s`
- `response_length_mean ≈ 985.53`
- `p50 ≈ 644.5`
- `p90 ≈ 2080.3`
- `p99 = 4096`
- `truncated = 0.046875`

分析文件：

- [summary.json](/root/APRIL/runs/qwen3-1.7b-no-partial-gsm8k-smoke-len4096-bs64-fixed/analysis/summary.json)
- [rollout_000000.jsonl](/root/APRIL/runs/qwen3-1.7b-no-partial-gsm8k-smoke-len4096-bs64-fixed/analysis/sample_records/rollout_000000.jsonl)

### Qwen3-0.6B

命令：

```bash
RUN_ROOT=/root/APRIL/runs/qwen3-0.6b-no-partial-gsm8k-smoke-len4096-bs64-fixed \
INPUT_DATA=/root/gsm8k/data/gsm8k-train.parquet \
NUM_ROLLOUT=1 \
ROLL_OUT_BATCH_SIZE=64 \
N_SAMPLES_PER_PROMPT=1 \
ROLLOUT_MAX_RESPONSE_LEN=4096 \
bash /root/APRIL/scripts/run-qwen3-0.6B.no-partial-dapo-bench.sh
```

结果：

- `rollout_time ≈ 83.51s`
- `tokens_throughput ≈ 686.35 tok/s`
- `response_length_mean ≈ 894.64`
- `p50 = 584.0`
- `p90 ≈ 1441.7`
- `p99 = 4096`
- `truncated = 0.03125`

分析文件：

- [summary.json](/root/APRIL/runs/qwen3-0.6b-no-partial-gsm8k-smoke-len4096-bs64-fixed/analysis/summary.json)
- [rollout_000000.jsonl](/root/APRIL/runs/qwen3-0.6b-no-partial-gsm8k-smoke-len4096-bs64-fixed/analysis/sample_records/rollout_000000.jsonl)

## 质量观察

- `1.7B` 首条 sample 正常解题并正确收尾。
- `0.6B` 也能正常输出简单题，但长尾样本更容易出现推理漂移。
- `0.6B` 的一个尾部样例里，把加班工资题算错了，而且回复长度接近 `4096`，说明它更容易在长推理里发散。

## DAPO 和 GSM8K 的区别

### DAPO

之前 `1.7B + DAPO + bs=64 + maxlen=20k` 的现象主要是任务真实长尾：

- `response_length_mean ≈ 11723`
- `p50 ≈ 11490`
- `p90 = 20000`
- `truncated ≈ 15.6%`

因此 DAPO 的问题主要是“题目和推理链本来就长”。

### GSM8K

GSM8K 最开始的异常不是天然长尾，而是 prompt/chat-template 格式错位导致的错误生成。

## 已清理的无效 runs

已经删除这些已被替代或无效的目录：

- `/root/APRIL/runs/qwen3-1.7b-no-partial-gsm8k-smoke-len20000-bs64`
- `/root/APRIL/runs/qwen3-1.7b-no-partial-gsm8k-smoke-len4096-bs64`
- `/root/APRIL/runs/qwen3-1.7b-no-partial-smoke`
- `/root/APRIL/runs/qwen3-1.7b-no-partial-smoke-len1024`
- `/root/APRIL/runs/qwen3-1.7b-no-partial-smoke-len20000`
- `/root/APRIL/runs/qwen3-1.7b-no-partial-smoke-len8192`

## 当前正在运行的正式任务

当前已经启动：

- 模型：`Qwen3-1.7B`
- 数据：`GSM8K`
- `batch_size=64`
- `n_sample=1`
- `maxlen=4096`
- `num_rollout=117`

运行目录：

- [qwen3-1.7b-no-partial-gsm8k-full-len4096-bs64](/root/APRIL/runs/qwen3-1.7b-no-partial-gsm8k-full-len4096-bs64)

Ray job：

- `raysubmit_AUmy2tDrdqgJzJG4`

查看状态：

```bash
ray job status raysubmit_AUmy2tDrdqgJzJG4
ray job logs raysubmit_AUmy2tDrdqgJzJG4 | tail -n 200
```

## 建议

- 如果目标是继续看 `GSM8K` 长尾，可以把 `maxlen` 提到 `8192`
- 如果目标是做稳定对比，`4096` 已经足够区分 `1.7B` 和 `0.6B`
- 如果后续要做 `partial rollout` 对比，建议直接沿用这份已经修好的 `GSM8K` parquet
