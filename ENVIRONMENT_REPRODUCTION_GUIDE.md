# APRIL Environment Reproduction Guide

这份文档用于在另一台机器上尽量准确地复现当前 `/root/APRIL` 工作环境。

适用目标：

- 复现当前这台机器上已经跑通的 `APRIL + Megatron-LM + sglang` 训练环境
- 保持和现有脚本兼容的目录布局
- 避免只装出“版本名相近、行为不一致”的环境

如果你后续还要准备模型、数据和训练命令，继续看 `/root/APRIL/setup.md`。

## 1. 当前环境指纹

这份 guide 基于下面这台机器的实际状态整理：

| Item | Value |
| --- | --- |
| APRIL commit | `067588941f9cd576f093bc129d31f7a75cc795a7` |
| OS | `Ubuntu 22.04.4 LTS` |
| Python | `3.10.12` |
| Python executable | `/usr/bin/python` |
| pip | `25.1.1` |
| GPU | `NVIDIA A100 80GB PCIe` |
| Driver | `590.48.01` |
| CUDA runtime | `12.6` |
| nvcc | `12.6.68` |
| gcc | `11.4.0` |
| torch | `2.7.1+cu126` |
| transformers | `4.53.0` |
| ray | `2.47.1` |
| sglang | `0.4.9` |
| flash-attn | `2.7.4.post1` |
| megatron-core | `0.14.0rc0` |

当前环境没有可导入的 `vllm`，现有单卡 `Qwen2.5-3B / Qwen3-1.7B` 主线不依赖它。

## 2. 目录布局

为了直接复用仓库里的现成脚本，建议保持和当前机器一致的路径：

- APRIL: `/root/APRIL`
- Megatron-LM: `/root/Megatron-LM`
- sglang source: `/sgl-workspace/sglang`
- Qwen2.5-3B: `/root/Qwen2.5-3B`
- Qwen2.5-3B torch_dist: `/root/Qwen2.5-3B_torch_dist`
- GSM8K data: `/root/gsm8k/data`
- competition_math data: `/root/math_level12/data`

如果你不想复用这些绝对路径，就要同步修改脚本里的默认值，尤其是：

- `/root/APRIL/scripts/run-qwen2.5-3B.train-math-level4.sh`
- `/root/APRIL/setup.md`

## 3. 方式 A：用 Docker 快速起一个相近环境

如果你的目标是先尽快拉起一个可用环境，Docker 是最快的路径。  
但要说明白：当前 `docker/Dockerfile` 没有把所有外部仓库都固定到这份 guide 记录的 commit，所以它更适合“快速接近”，不是“最严格的精确复现”。

如果你要尽量严格复刻当前工作区状态，优先走第 4 节，或者直接同步第 8 节列出的几个目录。

### 3.1 主机要求

- Ubuntu 22.04
- NVIDIA Driver 可正常驱动 A100
- `nvidia-smi` 可用
- Docker + NVIDIA Container Toolkit 可用

### 3.2 拉代码并固定到当前 commit

```bash
cd /root
git clone https://github.com/RLsys-Foundation/APRIL.git
cd /root/APRIL
git checkout 067588941f9cd576f093bc129d31f7a75cc795a7
```

### 3.3 构建镜像

```bash
cd /root/APRIL
docker build -f docker/Dockerfile -t april-env:cu126 .
```

这个 `Dockerfile` 会做几件关键事情：

- 安装当前环境需要的 CUDA 侧 Python 依赖
- 克隆 `/root/Megatron-LM` 并应用 `/root/APRIL/docker/patch/megatron-sandwich_norm.patch`
- 准备 `/sgl-workspace/sglang` 并应用 `/root/APRIL/docker/patch/sglang.patch`

### 3.4 启动容器

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -it april-env:cu126 /bin/bash
```

### 3.5 在容器内安装 APRIL 仓库本身

如果镜像里还没有你的 APRIL 工作树，容器内执行：

```bash
cd /root
git clone https://github.com/RLsys-Foundation/APRIL.git
cd /root/APRIL
git checkout 067588941f9cd576f093bc129d31f7a75cc795a7
pip install -e .
```

然后跳到第 5 节做验收。

## 4. 方式 B：裸机复现

如果你不打算用 Docker，可以按下面的顺序搭一套尽量接近当前机器的环境。

### 4.1 系统前置条件

- Ubuntu 22.04.x
- NVIDIA Driver 590 系列或兼容 CUDA 12.6 的版本
- CUDA toolkit 12.6，确保 `nvcc --version` 正常
- Python 3.10
- `git`
- `build-essential`
- `curl`
- `pkg-config`
- `libssl-dev`

一个可用的起步命令：

```bash
sudo apt-get update
sudo apt-get install -y git build-essential curl pkg-config libssl-dev
```

### 4.2 拉 APRIL 并固定 commit

```bash
cd /root
git clone https://github.com/RLsys-Foundation/APRIL.git
cd /root/APRIL
git checkout 067588941f9cd576f093bc129d31f7a75cc795a7
```

### 4.3 准备 Megatron-LM

当前工作环境里的 `megatron-core` 不是纯净上游版本，而是：

- commit: `cc0bdfbddebf709c78c0bc002e8579b8853f7ff9`
- 再额外应用 `/root/APRIL/docker/patch/megatron-sandwich_norm.patch`

所以不要只靠 `pip install megatron-core`。

```bash
cd /root
git clone https://github.com/NVIDIA/Megatron-LM.git
cd /root/Megatron-LM
git checkout cc0bdfbddebf709c78c0bc002e8579b8853f7ff9
git apply /root/APRIL/docker/patch/megatron-sandwich_norm.patch
```

### 4.4 准备 sglang

当前工作环境里的 `sglang` 也是本地源码 editable 安装，不是普通 wheel：

- source path: `/sgl-workspace/sglang`
- commit: `625018d2594422f7d35cdd51fd2e492d92a1fc3a`
- 再额外应用 `/root/APRIL/docker/patch/sglang.patch`

执行：

```bash
mkdir -p /sgl-workspace
cd /sgl-workspace
git clone https://github.com/sgl-project/sglang.git
cd /sgl-workspace/sglang
git checkout 625018d2594422f7d35cdd51fd2e492d92a1fc3a
git apply /root/APRIL/docker/patch/sglang.patch
```

### 4.5 安装 Python 依赖

`/root/APRIL/requirements.txt` 已经是当前环境的 Python 依赖快照，但里面有两行是“editable 本地源码安装”：

- `sglang`
- `megatron-core`

这两项要走第 4.3 和第 4.4 节的本地路径，不要让 `pip` 自己拉到别的目录。

先过滤掉这两行：

```bash
grep -Ev '#egg=sglang|#egg=megatron_core' /root/APRIL/requirements.txt > /tmp/april.requirements.no_local.txt
```

然后安装依赖。为了让 `torch==2.7.1+cu126`、`torchvision==0.22.1+cu126`、`torchaudio==2.7.1+cu126` 能解析到正确源，建议显式带上 PyTorch 的 cu126 index：

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r /tmp/april.requirements.no_local.txt
```

如果你是在全新机器上从零装，`apex`、`grouped_gemm` 这类包会走编译流程。  
一旦编译失败，先优先检查：

- `nvcc --version`
- `CUDA_HOME`
- `gcc --version`

### 4.6 安装本地 editable 包

按当前环境的方式安装：

```bash
pip install -e /root/Megatron-LM
pip install -e /sgl-workspace/sglang/python
pip install -e /root/APRIL
```

## 5. 验收

先确认基础版本：

```bash
python --version
pip --version
nvidia-smi
nvcc --version
```

再确认 Python 包和源码路径：

```bash
python - <<'PY'
import pathlib
import sglang
import torch
import transformers
import ray
import flash_attn
import importlib.metadata as md

print("torch:", torch.__version__)
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("ray:", ray.__version__)
print("flash_attn:", flash_attn.__version__)
print("sglang file:", pathlib.Path(sglang.__file__).resolve())
print("sglang direct_url:", md.distribution("sglang").read_text("direct_url.json"))
print("megatron direct_url:", md.distribution("megatron-core").read_text("direct_url.json"))
PY
```

期望看到的关键点：

- `torch.cuda.is_available: True`
- `torch.version.cuda: 12.6`
- `sglang file` 指向 `/sgl-workspace/sglang/python/...`
- `megatron direct_url` 指向 `/root/Megatron-LM`

## 6. 模型和数据

当前脚本默认依赖这些路径：

- `/root/Qwen2.5-3B`
- `/root/Qwen2.5-3B_torch_dist`
- `/root/gsm8k/data`
- `/root/math_level12/data`

模型下载、`torch_dist` 转换、`GSM8K` / `competition_math` 数据准备步骤，直接看：

- `/root/APRIL/setup.md`

## 7. 最小 smoke test

模型和数据就绪后，可以用当前最常用的单卡入口做一次 smoke test：

```bash
cd /root/APRIL
INPUT_DATA=/root/math_level12/data/math-level2-train.shortprompt.parquet \
EVAL_DATA=/root/math_level12/data/math-level2-test.shortprompt.subset64.seed1234.parquet \
ROLL_OUT_BATCH_SIZE=8 \
N_SAMPLES_PER_PROMPT=8 \
NUM_ROLLOUT=1 \
bash /root/APRIL/scripts/run-qwen2.5-3B.train-math-level4.sh
```

日志默认会落到：

- `/root/APRIL/runs/<run_name>/job_output.log`

注意：这个脚本在启动前会执行 `ray stop --force`，会清掉同机已有 Ray 任务。

## 8. 额外说明

- 当前 `requirements.txt` 是这台机器的快照，优先级高于 README 里更泛化的安装说明。
- `Dockerfile` 里也记录了环境构建方式，但那更偏“构建镜像”；如果你想复刻当前工作区状态，除了装 Python 包，还要确保 `Megatron-LM` 和 `sglang` 的 commit 与 patch 一致。
- 如果你想做到最接近“原封不动复制”，最稳的办法仍然是直接同步这三个目录：
  - `/root/APRIL`
  - `/root/Megatron-LM`
  - `/sgl-workspace/sglang`
