# APRIL Setup Guide

如果你的目标是先在另一台机器上尽量复现当前这套工作环境，先看 `/root/APRIL/ENVIRONMENT_REPRODUCTION_GUIDE.md`。

这份文档面向之后在新环境里从零开始使用 APRIL 的场景。当前仓库里已经实际跑过两条主线：

- `Qwen3-1.7B + GSM8K`
- `Qwen2.5-3B + competition_math (Level 2 shortprompt / Level 4 variants)`

默认硬件假设仍然是：

- 单张 GPU

如果你的目标是先跑通，再做 `non-provision / provision` 对照，这份文档可以直接照着执行。当前最推荐的主线是：

- `Qwen2.5-3B + Level 2 shortprompt + subset64 eval`

## 1. 目录约定

下面的命令默认使用这些路径：

- 代码目录：`/root/APRIL`
- Megatron-LM：`/root/Megatron-LM`
- `Qwen3-1.7B` HF 目录：`/root/Qwen3-1.7B`
- `Qwen3-1.7B` `torch_dist`：`/root/Qwen3-1.7B_torch_dist`
- `Qwen2.5-3B` HF 目录：`/root/Qwen2.5-3B`
- `Qwen2.5-3B` `torch_dist`：`/root/Qwen2.5-3B_torch_dist`
- GSM8K 数据目录：`/root/gsm8k/data`
- competition_math 数据目录：`/root/math_level12/data`

如果你想改路径，后面的命令里同步替换即可。

## 2. 环境准备

### 2.1 基础要求

建议至少满足：

- Linux
- CUDA 可用
- 单卡 GPU
- Python 3.10
- 已安装 `git`
- 已安装 `huggingface-cli`

### 2.2 拉代码

如果是新环境：

```bash
cd /root
git clone https://github.com/Tianqi-Xuuu/APRIL
cd /root/APRIL
```

如果代码已经在本地，直接：

```bash
cd /root/APRIL
```

### 2.3 安装 APRIL

```bash
cd /root/APRIL
pip install -e .
```

如果你还没装数据处理相关依赖，建议顺手补上：

```bash
pip install datasets pyarrow pandas
```

### 2.4 准备 Megatron-LM

APRIL 里有一部分工具和训练逻辑依赖 `Megatron-LM`，并且会通过 `PYTHONPATH=/root/Megatron-LM` 找到它。

如果新环境里还没有：

```bash
cd /root
git clone https://github.com/NVIDIA/Megatron-LM.git
```

后续运行转换脚本时，记得显式带上：

```bash
PYTHONPATH=/root/Megatron-LM
```

## 3. 模型准备

### 3.1 下载 Hugging Face 模型

#### Qwen3-1.7B

```bash
huggingface-cli download Qwen/Qwen3-1.7B --local-dir /root/Qwen3-1.7B
```

下载完成后，训练脚本里的：

- `--hf-checkpoint /root/Qwen3-1.7B`

就会直接使用这个目录。

#### Qwen2.5-3B

```bash
huggingface-cli download Qwen/Qwen2.5-3B --local-dir /root/Qwen2.5-3B
```

### 3.2 转换成 Megatron 可加载的 `torch_dist`

#### Qwen3-1.7B

APRIL 的 actor / reference 初始化依赖 `--ref-load`，这里需要先把 HF 权重转成 `torch_dist`。

执行：

```bash
cd /root/APRIL
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint /root/Qwen3-1.7B \
  --save /root/Qwen3-1.7B_torch_dist
```

转换完成后，建议确认一下：

```bash
ls /root/Qwen3-1.7B_torch_dist
cat /root/Qwen3-1.7B_torch_dist/latest_checkpointed_iteration.txt
```

正常情况下，目录里会有 `.distcp` 文件和 `latest_checkpointed_iteration.txt`。

#### Qwen2.5-3B

```bash
cd /root/APRIL
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint /root/Qwen2.5-3B \
  --save /root/Qwen2.5-3B_torch_dist
```

### 3.3 `hf-checkpoint` 和 `ref-load` 的区别

这两个参数很容易混：

- `--hf-checkpoint`：给 `SGLang` 用的 HF 权重目录，也会从这里读 tokenizer
- `--ref-load`：给 Megatron 侧加载的 `torch_dist` checkpoint

对当前这套 `Qwen3-1.7B` 配置来说，建议固定成：

```text
--hf-checkpoint /root/Qwen3-1.7B
--ref-load /root/Qwen3-1.7B_torch_dist
```

对 `Qwen2.5-3B` 来说，对应就是：

```text
--hf-checkpoint /root/Qwen2.5-3B
--ref-load /root/Qwen2.5-3B_torch_dist
```

## 4. 数据集准备

当前仓库里已经准备好的主要数据有两类：

- `GSM8K`
- `competition_math` 的 level 子集和 prompt 变体

### 4.1 生成 GSM8K parquet

```bash
cd /root/APRIL
python /root/APRIL/scripts/analysis/prepare_gsm8k_dataset.py \
  --output /root/gsm8k/data/gsm8k-train.parquet
```

这个脚本会自动从 Hugging Face datasets 读取 `gsm8k/main` 的 `train` split，并输出成：

- `source_prompt`
- `answer`
- `solution`
- `metadata`

### 4.2 为什么这里要特别注意 `source_prompt`

APRIL 当前脚本里启用了：

```text
--apply-chat-template
```

所以 `source_prompt` 不能是普通字符串，必须是 OpenAI message list 这种结构。  
我们已经在 `scripts/analysis/prepare_gsm8k_dataset.py` 里修好了这一点，当前生成的是：

```python
[{"role": "user", "content": prompt}]
```

如果你在别的环境自己重新准备数据，一定要保留这个格式。  
不然 `Qwen3` 的 `apply_chat_template` 会把 prompt 弄坏，表现出来就是：

- 一开头只有 `<|im_start|>assistant`
- 模型开始乱生成
- 长度异常长，看起来像死循环

### 4.3 快速检查数据

可以抽一条看一下：

```bash
python - <<'PY'
import pandas as pd
df = pd.read_parquet('/root/gsm8k/data/gsm8k-train.parquet')
print(df.iloc[0].to_dict())
PY
```

### 4.4 固定小评估集

为了避免训练时 `eval` 比 rollout 更慢，当前推荐固定使用一个小的 GSM8K test subset：

- `/root/gsm8k/data/gsm8k-test.subset256.seed1234.parquet`

如果这个文件还不存在，可以按固定随机种子从 `gsm8k-test.parquet` 抽出 `256` 条，供不同实验组共用。当前仓库脚本默认已经优先使用这个 subset。

### 4.5 competition_math / MATH level 数据

当前已经准备好的核心文件在：

- `Level 2 train`: `/root/math_level12/data/math-level2-train.parquet`
- `Level 2 test`: `/root/math_level12/data/math-level2-test.parquet`
- `Level 2 shortprompt train`: `/root/math_level12/data/math-level2-train.shortprompt.parquet`
- `Level 2 shortprompt test`: `/root/math_level12/data/math-level2-test.shortprompt.parquet`
- `Level 2 shortprompt subset64`: `/root/math_level12/data/math-level2-test.shortprompt.subset64.seed1234.parquet`
- `Level 4 train`: `/root/math_level12/data/math-level4-train.parquet`
- `Level 4 stepthink train`: `/root/math_level12/data/math-level4-train.stepthink.parquet`
- `Level 4 shortprompt train`: `/root/math_level12/data/math-level4-train.shortprompt.parquet`
- `Level 4 subset64`: `/root/math_level12/data/math-level4-test.subset64.seed1234.parquet`
- `Level 5 train`: `/root/math_level12/data/math-level5-train.parquet`
- `Level 5 short / minprompt 变体`

当前 prompt 变体里最常用的是：

- `stepthink`
  - 适合观察更长推理
- `shortprompt`
  - 更适合控制长度
- `minprompt`
  - 进一步压缩输出

如果你要做 `Qwen2.5-3B` 的 `non-prov / provision` 对照，当前推荐优先使用：

- train: `/root/math_level12/data/math-level2-train.shortprompt.parquet`
- eval: `/root/math_level12/data/math-level2-test.shortprompt.subset64.seed1234.parquet`

### 4.6 reward 对齐说明

当前 `deepscaler` 已经修过，不再要求 response 里必须先出现 `</think>` 或 `###Response`。

现在这几种形式都会被正常抽答案：

- `Answer: \boxed{123}`
- 只有 `\boxed{123}`
- 没有 `</think>` 的普通解题回答

所以当前数学数据建议统一用：

```text
Answer: \boxed{123}
```

不用再依赖 `step-by-step + </think>` 这种格式。

## 5. 快速开始

### 5.0 先清场

现在 `1.7B` 和 `Qwen2.5-3B` 主脚本都会先调用：

- `/root/APRIL/scripts/lib/train_cleanup.sh`

它会在每次启动前：

- 清残留 `train.py`
- 清残留 `ray job submit/logs`
- `ray stop --force`
- 再重启新的 Ray head

这样能避免旧 session 污染新实验。

### 5.1 Qwen2.5-3B 训练主线

这是当前最推荐的正式训练入口。它已经支持：

- 启动前自动清场
- 自动根据数据集生成 `RUN_NAME`
- 小 eval subset
- `--keep-only-latest-checkpoint`
- 如果同一 `RUN_ROOT` 下已有 checkpoint，则自动 resume

当前推荐配置：

- train: `/root/math_level12/data/math-level2-train.shortprompt.parquet`
- eval: `/root/math_level12/data/math-level2-test.shortprompt.subset64.seed1234.parquet`

直接启动：

```bash
cd /root/APRIL
INPUT_DATA=/root/math_level12/data/math-level2-train.shortprompt.parquet \
EVAL_DATA=/root/math_level12/data/math-level2-test.shortprompt.subset64.seed1234.parquet \
ROLL_OUT_BATCH_SIZE=8 \
N_SAMPLES_PER_PROMPT=8 \
NUM_ROLLOUT=30 \
bash /root/APRIL/scripts/run-qwen2.5-3B.train-math-level4.sh
```

如果要跑 provision 版本，只需要额外带上：

```bash
PARTIAL_ROLLOUT=1
OVERSAMPLING_BATCH_SIZE=<倍数后的 batch>
```

例如 `2.0x provision`：

```bash
cd /root/APRIL
PARTIAL_ROLLOUT=1 \
OVERSAMPLING_BATCH_SIZE=16 \
INPUT_DATA=/root/math_level12/data/math-level2-train.shortprompt.parquet \
EVAL_DATA=/root/math_level12/data/math-level2-test.shortprompt.subset64.seed1234.parquet \
ROLL_OUT_BATCH_SIZE=8 \
N_SAMPLES_PER_PROMPT=8 \
NUM_ROLLOUT=30 \
RUN_NAME=qwen2.5-3b-train-math-level2-shortprompt-prov20-bs8-n8-r30 \
bash /root/APRIL/scripts/run-qwen2.5-3B.train-math-level4.sh
```

其他 provision 档位对应：

- `3.0x`: `OVERSAMPLING_BATCH_SIZE=24`
- `4.0x`: `OVERSAMPLING_BATCH_SIZE=32`
- `5.0x`: `OVERSAMPLING_BATCH_SIZE=40`

### 5.2 No-partial rollout smoke

这是最推荐的第一步。先确认模型、数据、`torch_dist` 和 Ray/SGLang 都能正常起来。

```bash
cd /root/APRIL
RUN_ROOT=/root/APRIL/runs/qwen3-1.7b-no-partial-gsm8k-smoke-len4096-bs64 \
INPUT_DATA=/root/gsm8k/data/gsm8k-train.parquet \
NUM_ROLLOUT=1 \
ROLL_OUT_BATCH_SIZE=64 \
N_SAMPLES_PER_PROMPT=1 \
ROLLOUT_MAX_RESPONSE_LEN=4096 \
bash /root/APRIL/scripts/run-qwen3-1.7B.no-partial-dapo-bench.sh
```

这条命令的含义：

- 使用 `GSM8K`
- 单轮 rollout
- `batch_size=64`
- 每题采样 `1` 条
- 最长输出 `4096`
- 只跑 rollout，不进入正式训练更新

输出重点看：

- `${RUN_ROOT}/job_output.log`
- `${RUN_ROOT}/analysis/summary.json`
- `${RUN_ROOT}/analysis/sample_records/`

### 5.3 全量 no-partial rollout

如果 smoke 没问题，可以跑完整个 `GSM8K`：

```bash
cd /root/APRIL
RUN_ROOT=/root/APRIL/runs/qwen3-1.7b-no-partial-gsm8k-full-len4096-bs64 \
INPUT_DATA=/root/gsm8k/data/gsm8k-train.parquet \
ROLL_OUT_BATCH_SIZE=64 \
N_SAMPLES_PER_PROMPT=1 \
ROLLOUT_MAX_RESPONSE_LEN=4096 \
bash /root/APRIL/scripts/run-qwen3-1.7B.no-partial-dapo-bench.sh
```

这个脚本会先把数据 pad 到 batch size 的整数倍，然后自动推导 `NUM_ROLLOUT`。

### 5.4 Partial-rollout 对照 sweep

如果你要比较不同 `over_sampling_batch_size` 的影响，直接用现成脚本：

```bash
cd /root/APRIL
INPUT_DATA=/root/gsm8k/data/gsm8k-train.parquet \
ROLLOUT_BATCH_SIZE=32 \
N_SAMPLES_PER_PROMPT=8 \
ROLLOUT_MAX_RESPONSE_LEN=256 \
OVER_SAMPLING_BATCH_SIZES="32 40 48 56 64 80 96" \
RUN_MODE=rollout_only \
NUM_ROLLOUT=1 \
bash /root/APRIL/scripts/run-qwen3-1.7B.partial-rollout-sweep.sh
```

如果你想模拟真实训练而不是 rollout-only，把：

```text
RUN_MODE=rollout_only
```

改成：

```text
RUN_MODE=train
```

## 6. 常用脚本说明

当前你主要会用到这几个：

- `scripts/run-qwen2.5-3B.train-math-level4.sh`
  - 当前 `Qwen2.5-3B` 的通用训练入口
  - 虽然文件名里还是 `level4`，但现在已经支持通过 `INPUT_DATA/EVAL_DATA` 切到 `Level 2 shortprompt`
- `scripts/lib/train_cleanup.sh`
  - 所有主训练脚本启动前都会调用的清场脚本
- `scripts/run-qwen3-1.7B.no-partial-dapo-bench.sh`
  - 当前最稳定的 `Qwen3-1.7B + GSM8K + no-partial` 启动脚本
- `scripts/run-qwen3-1.7B.partial-rollout-sweep.sh`
  - 用于批量测试不同 `over_sampling_batch_size`
- `scripts/run-qwen2.5-3B.no-partial-math-level-rollout.sh`
  - 当前 `Qwen2.5-3B` 的单轮 rollout-only 测试入口
- `scripts/analysis/prepare_gsm8k_dataset.py`
  - 生成 APRIL 可直接读取的 `GSM8K` parquet
- `scripts/analysis/prepare_padded_dataset.py`
  - 把数据 pad 到 batch size 整数倍
- `tools/convert_hf_to_torch_dist.py`
  - 把 HF checkpoint 转成 `torch_dist`

## 7. 结果怎么看

每次跑一个实验，建议先固定好一个 `RUN_ROOT`。  
结果通常会出现在这些位置：

- `job_output.log`
  - Ray job submit 和训练主日志
- `analysis/summary.json`
  - 聚合后的关键指标
- `analysis/sample_records/rollout_*.jsonl`
  - sample rollout，方便检查输出质量
- `analysis/eval_metrics.jsonl`
  - eval 聚合指标
- `analysis/eval_sample_records/<dataset>_rollout_<id>.jsonl`
  - 每次 eval 的样本级记录
- `debug_rollout/rollout_*.pkl`
  - 原始 debug rollout 数据

如果是 Ray job 在后台跑，也可以用：

```bash
ray job status <job_id>
ray job logs <job_id> | tail -n 200
```

## 8. 常见问题

### 8.1 报错：`ref_load ... does not exist`

这基本说明 `torch_dist` 还没准备好，或者路径不对。

先检查：

```bash
ls /root/Qwen3-1.7B_torch_dist
cat /root/Qwen3-1.7B_torch_dist/latest_checkpointed_iteration.txt
```

如果目录不存在，就重新执行第 3.2 步转换。

### 8.2 GSM8K 生成特别长，像死循环

先检查是不是数据格式坏了。  
如果 `source_prompt` 不是 message list，而是普通字符串，就很容易触发这个问题。

正确格式应该像：

```python
[{"role": "user", "content": "..."}]
```

### 8.3 单卡显存 / KV cache 压力过大

如果你把 `batch_size` 和 `max_response_len` 都拉得很大，`SGLang` 可能出现：

- `KV cache pool is full`
- `Retract requests`

单卡下更稳的做法通常是：

- 先减小 `ROLL_OUT_BATCH_SIZE`
- 保持 `N_SAMPLES_PER_PROMPT=1`
- 适当降低 `ROLLOUT_MAX_RESPONSE_LEN`
- 必要时下调 `SGLANG_MEM_FRACTION`

### 8.4 `ray stop --force`

现有脚本会在启动前执行：

```bash
ray stop --force
```

这意味着如果机器上已经有别的 Ray 任务，它会被停掉。  
所以并发跑多个实验前，先确认你是否真的想重启 Ray。

### 8.5 checkpoint 占空间很快

当前大模型训练时，checkpoint 可能非常大，尤其是：

- `Qwen2.5-3B`

因此当前主训练脚本已经默认带：

```text
--keep-only-latest-checkpoint
```

它会在保存新 checkpoint 后，清掉更老 checkpoint 的大权重分片，避免磁盘线性膨胀。

## 9. 推荐工作流

如果是在新环境第一次使用，我建议按这个顺序：

1. 安装 APRIL 和依赖
2. 准备 `Megatron-LM`
3. 下载 `Qwen2.5-3B`
4. 转成 `/root/Qwen2.5-3B_torch_dist`
5. 准备 `Level 2 shortprompt` 训练和 `subset64` eval
6. 先跑一轮 rollout-only smoke
7. 再跑 `Qwen2.5-3B + Level 2 shortprompt` 正式训练
8. 之后再做 `non-provision / provision` 对照

## 10. 当前默认结论

按我们已经验证过的配置：

- `Qwen2.5-3B` 是当前推荐继续推进的主模型
- `Level 2 shortprompt` 是当前推荐的主训练数据集
- `subset64 eval` 是当前推荐的快速评估配置
- `Qwen3-1.7B + GSM8K` 仍然可用，但现在更偏历史基线
- `0.6B` 这条线已经不再继续维护

如果你后面继续沿用当前主线，最关键的是保证下面四项一直一致：

- `--hf-checkpoint /root/Qwen2.5-3B`
- `--ref-load /root/Qwen2.5-3B_torch_dist`
- `INPUT_DATA=/root/math_level12/data/math-level2-train.shortprompt.parquet`
- `EVAL_DATA=/root/math_level12/data/math-level2-test.shortprompt.subset64.seed1234.parquet`
