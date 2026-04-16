import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Optional

import modal


APP_NAME = "april-modal"
APRIL_NV_DOCKER_IMAGE = "rlsys/april:NV_exp_docker_image"
APRIL_ROOT = Path("/root/APRIL")
SGLANG_PYTHON_ROOT = Path("/sgl-workspace/sglang/python")
MODELS_ROOT = Path("/root/models")
DATA_ROOT = Path("/root/data")
RUNS_ROOT = Path("/root/runs")
CACHE_ROOT = Path("/root/cache")
MATH_LEVEL12_ROOT = DATA_ROOT / "math_level12"

QWEN25_HF_CHECKPOINT = MODELS_ROOT / "Qwen2.5-3B"
QWEN25_REF_LOAD = MODELS_ROOT / "Qwen2.5-3B_torch_dist"
LEVEL2_TRAIN = MATH_LEVEL12_ROOT / "math-level2-train.parquet"
LEVEL2_TEST_SUBSET64 = MATH_LEVEL12_ROOT / "math-level2-test.subset64.seed1234.parquet"

MODELS_VOLUME = modal.Volume.from_name("april-models", create_if_missing=True)
DATA_VOLUME = modal.Volume.from_name("april-data", create_if_missing=True)
RUNS_VOLUME = modal.Volume.from_name("april-runs", create_if_missing=True)
CACHE_VOLUME = modal.Volume.from_name("april-cache", create_if_missing=True)

VOLUME_MOUNTS = {
    str(MODELS_ROOT): MODELS_VOLUME,
    str(DATA_ROOT): DATA_VOLUME,
    str(RUNS_ROOT): RUNS_VOLUME,
    str(CACHE_ROOT): CACHE_VOLUME,
}

REPO_IGNORE = [
    ".git",
    ".git/**",
    ".venv",
    ".venv/**",
    ".venv-modal",
    ".venv-modal/**",
    "__pycache__",
    "**/__pycache__/**",
    "*.pyc",
    "results",
    "results/**",
    "runs",
    "runs/**",
]

image = (
    modal.Image.from_registry(APRIL_NV_DOCKER_IMAGE)
    .add_local_dir(".", remote_path=str(APRIL_ROOT), copy=True, ignore=REPO_IGNORE)
)
app = modal.App(APP_NAME)


def _quote(v: Path | str) -> str:
    return shlex.quote(str(v))


def _run(command: str, *, env: Optional[Dict[str, str]] = None) -> None:
    merged = os.environ.copy()
    if env:
        merged.update({k: str(v) for k, v in env.items()})
    subprocess.run(["bash", "-lc", command], check=True, env=merged)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 24,
    volumes=VOLUME_MOUNTS,
)
def train_qwen25_level2_base(
    run_name: str,
    partial_rollout: bool = False,
    oversampling_batch_size: int = 16,
    rollout_batch_size: int = 8,
    n_samples_per_prompt: int = 8,
    num_rollout: int = 100,
    eval_interval: int = 5,
    rollout_max_response_len: int = 4096,
    sglang_mem_fraction: float = 0.7,
    train_seed: int = 1234,
    disable_ckpt_save: bool = False,
    save_hf_weights: bool = False,
    save_hf_only_final: bool = False,
    use_behavior_logprobs_for_ppo_clip: bool = False,
    lr: str = "1e-6",
    min_lr: str = "1e-7",
    clip_grad: str = "1.0",
    eps_clip: str = "0.2",
    eps_clip_high: str = "0.28",
) -> str:
    run_root = RUNS_ROOT / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    env = {
        "APRIL_ROOT": str(APRIL_ROOT),
        "MEGATRON_ROOT": "/root/Megatron-LM",
        "SGLANG_PYTHON_ROOT": str(SGLANG_PYTHON_ROOT),
        "QWEN25_HF_CHECKPOINT": str(QWEN25_HF_CHECKPOINT),
        "QWEN25_REF_LOAD": str(QWEN25_REF_LOAD),
        "INPUT_DATA": str(LEVEL2_TRAIN),
        "EVAL_DATA": str(LEVEL2_TEST_SUBSET64),
        "RUN_NAME": run_name,
        "RUN_ROOT": str(run_root),
        "TRAIN_DATA_TAG": LEVEL2_TRAIN.stem,
        "EVAL_DATA_TAG": LEVEL2_TEST_SUBSET64.stem,
        "HF_HOME": str(CACHE_ROOT / "hf"),
        "HUGGINGFACE_HUB_CACHE": str(CACHE_ROOT / "hf" / "hub"),
        "TMPDIR": "/tmp",
        "RAY_TMPDIR": "/tmp/ray",
        "RAY_NUM_GPUS": "1",
        "RAY_PLASMA_DIRECTORY": "/tmp",
        "RAY_OBJECT_STORE_MEMORY": str(256 * 1024 * 1024),
        "SLIME_SGLANG_DIRECT": "1",
        "SLIME_MODAL_DIRECT": "1",
        "SAVE_HF_WEIGHTS": "1" if save_hf_weights else "0",
        "SAVE_HF_ONLY_FINAL": "1" if save_hf_only_final else "0",
        "TRAIN_SEED": str(train_seed),
        "DISABLE_CKPT_SAVE": "1" if disable_ckpt_save else "0",
        "DEBUG_ROLLOUT_ONLY": "0",
        "ROLL_OUT_BATCH_SIZE": str(rollout_batch_size),
        "N_SAMPLES_PER_PROMPT": str(n_samples_per_prompt),
        "NUM_ROLLOUT": str(num_rollout),
        "EVAL_INTERVAL": str(eval_interval),
        "ROLLOUT_MAX_RESPONSE_LEN": str(rollout_max_response_len),
        "SGLANG_MEM_FRACTION": str(sglang_mem_fraction),
        "PARTIAL_ROLLOUT": "1" if partial_rollout else "0",
        "OVERSAMPLING_BATCH_SIZE": str(oversampling_batch_size),
        "USE_BEHAVIOR_LOGPROBS_FOR_PPO_CLIP": "1" if use_behavior_logprobs_for_ppo_clip else "0",
        "LR": str(lr),
        "MIN_LR": str(min_lr),
        "CLIP_GRAD": str(clip_grad),
        "EPS_CLIP": str(eps_clip),
        "EPS_CLIP_HIGH": str(eps_clip_high),
    }

    _run(f"cd {_quote(APRIL_ROOT)} && bash scripts/run-qwen2.5-3B.train-math-level4.sh", env=env)
    RUNS_VOLUME.commit()
    return str(run_root)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=86400,
    volumes=VOLUME_MOUNTS,
)
def idea_rollout_window_sweep_all_ckpts(
    hf_parent_run_name: str,
    include_base_hf: bool = True,
    resume_only_failed_ckpts: bool = False,
    num_rollout: int = 20,
    rollout_batch_size: int = 8,
    n_samples_per_prompt: int = 8,
    rollout_max_response_len: int = 4096,
    sglang_mem_fraction: float = 0.7,
    window_ratio_min: float = 1.0,
    window_ratio_max: float = 4.0,
    window_ratio_step: float = 0.5,
    run_analysis: bool = True,
) -> str:
    """
    For each HF export under ``/root/runs/<hf_parent_run_name>/hf/iter_*``, run
    ``idea/run_rollout_window_sweep.sh`` (goodput vs window), optionally including the base
    Qwen2.5-3B weights first. Requires that run directory on ``april-runs`` volume.
    """
    if not hf_parent_run_name.strip():
        raise ValueError("hf_parent_run_name must be non-empty (RUN_NAME of a run that has hf/iter_*)")

    hf_scan = RUNS_ROOT / hf_parent_run_name.strip() / "hf"
    run_root_parent = (
        RUNS_ROOT
        / f"idea-rollout-window-allckpts-{hf_parent_run_name.strip()}-bs{rollout_batch_size}"
    )

    env = {
        "APRIL_ROOT": str(APRIL_ROOT),
        "MEGATRON_ROOT": "/root/Megatron-LM",
        "SGLANG_PYTHON_ROOT": str(SGLANG_PYTHON_ROOT),
        "HF_CHECKPOINT": str(QWEN25_HF_CHECKPOINT),
        "REF_LOAD": str(QWEN25_REF_LOAD),
        "HF_SCAN_DIR": str(hf_scan),
        "INCLUDE_BASE_HF": "1" if include_base_hf else "0",
        "MODEL_TAG": "qwen2.5-3b",
        "MODEL_SCRIPT": str(APRIL_ROOT / "scripts/models/qwen2.5-3B.sh"),
        "INPUT_DATA": str(LEVEL2_TRAIN),
        "TASK_TAG": "math-level2",
        "ROLLOUT_BATCH_SIZE": str(rollout_batch_size),
        "N_SAMPLES_PER_PROMPT": str(n_samples_per_prompt),
        "NUM_ROLLOUT": str(num_rollout),
        "ROLLOUT_MAX_RESPONSE_LEN": str(rollout_max_response_len),
        "SGLANG_MEM_FRACTION": str(sglang_mem_fraction),
        "WINDOW_RATIO_MIN": str(window_ratio_min),
        "WINDOW_RATIO_MAX": str(window_ratio_max),
        "WINDOW_RATIO_STEP": str(window_ratio_step),
        "RUN_ROOT_BASE_PARENT": str(run_root_parent),
        "RUN_ANALYSIS": "1" if run_analysis else "0",
        "FORCE_RAY_RESTART": "1",
        "DRY_RUN": "0",
        "HF_HOME": str(CACHE_ROOT / "hf"),
        "HUGGINGFACE_HUB_CACHE": str(CACHE_ROOT / "hf" / "hub"),
        "TMPDIR": "/tmp",
        "RAY_TMPDIR": "/tmp/ray",
        "RAY_NUM_GPUS": "1",
        "RAY_PLASMA_DIRECTORY": "/tmp",
        "RAY_OBJECT_STORE_MEMORY": str(256 * 1024 * 1024),
        "SLIME_SGLANG_DIRECT": "1",
        "SLIME_MODAL_DIRECT": "1",
        "RESUME_ONLY_FAILED_CKPTS": "1" if resume_only_failed_ckpts else "0",
    }

    _run(f"cd {_quote(APRIL_ROOT)} && bash idea/run_rollout_window_sweep_all_ckpts.sh", env=env)
    RUNS_VOLUME.commit()
    return str(run_root_parent)
