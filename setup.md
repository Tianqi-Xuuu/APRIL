# APRIL Setup Guide

这份文档面向之后在新环境里从零开始使用 APRIL 的场景，默认只使用：

- 模型：`Qwen3-1.7B`
- 数据集：`GSM8K`
- 硬件：单张 GPU

如果你的目标是先跑通、再做 no-partial / partial-rollout 对照，这份文档可以直接照着执行。

## 1. 目录约定

下面的命令默认使用这些路径：

- 代码目录：`/root/APRIL`
- Megatron-LM：`/root/Megatron-LM`
- HF 模型目录：`/root/Qwen3-1.7B`
- Megatron `torch_dist` 目录：`/root/Qwen3-1.7B_torch_dist`
- GSM8K 输出目录：`/root/gsm8k/data`

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
git clone https://github.com/RLsys-Foundation/APRIL.git
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

```bash
huggingface-cli download Qwen/Qwen3-1.7B --local-dir /root/Qwen3-1.7B
```

下载完成后，训练脚本里的：

- `--hf-checkpoint /root/Qwen3-1.7B`

就会直接使用这个目录。

### 3.2 转换成 Megatron 可加载的 `torch_dist`

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

### 3.3 `hf-checkpoint` 和 `ref-load` 的区别

这两个参数很容易混：

- `--hf-checkpoint`：给 `SGLang` 用的 HF 权重目录，也会从这里读 tokenizer
- `--ref-load`：给 Megatron 侧加载的 `torch_dist` checkpoint

对当前这套 `Qwen3-1.7B` 配置来说，建议固定成：

```text
--hf-checkpoint /root/Qwen3-1.7B
--ref-load /root/Qwen3-1.7B_torch_dist
```

## 4. 数据集准备

你之后只用 `GSM8K`，这里直接准备成 APRIL 现在能吃的 parquet 格式。

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

## 5. 快速开始

### 5.1 No-partial rollout smoke

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

### 5.2 全量 no-partial rollout

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

### 5.3 Partial-rollout 对照 sweep

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

- `scripts/run-qwen3-1.7B.no-partial-dapo-bench.sh`
  - 当前最稳定的 `Qwen3-1.7B + GSM8K + no-partial` 启动脚本
- `scripts/run-qwen3-1.7B.partial-rollout-sweep.sh`
  - 用于批量测试不同 `over_sampling_batch_size`
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

## 9. 推荐工作流

如果是在新环境第一次使用，我建议按这个顺序：

1. 安装 APRIL 和依赖
2. 准备 `Megatron-LM`
3. 下载 `Qwen3-1.7B`
4. 转成 `/root/Qwen3-1.7B_torch_dist`
5. 生成 `/root/gsm8k/data/gsm8k-train.parquet`
6. 跑一轮 no-partial smoke
7. 确认 sample rollout 正常
8. 再跑 full no-partial 或 partial-rollout sweep

## 10. 当前默认结论

按我们已经验证过的配置：

- `Qwen3-1.7B` 是当前建议保留的主模型
- `GSM8K` 已经适配好，不需要再做额外格式修补
- `0.6B` 这条线已经不再继续维护

如果你后面继续沿用这套路径，最关键的是保证下面三项一直一致：

- `--hf-checkpoint /root/Qwen3-1.7B`
- `--ref-load /root/Qwen3-1.7B_torch_dist`
- `INPUT_DATA=/root/gsm8k/data/gsm8k-train.parquet`
