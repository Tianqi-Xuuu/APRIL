import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--origin-hf-dir", type=str, default=None)
    parser.add_argument("--converted-model-dir", type=str, default=None)
    parser.add_argument("--force-convert", action="store_true", default=False)
    parser.add_argument("--load-batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=2 * 1024**3)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--load-mode", type=str, default="staged", choices=["staged", "baseline"])
    parser.add_argument("--prompt", type=str, default="What is 2 + 2?")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--max-gpu-memory", type=str, default=None)
    parser.add_argument("--max-cpu-memory", type=str, default="120GiB")
    return parser.parse_args()


def get_dtype(dtype_name):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def report_cuda_peak(prefix, device):
    if device.type != "cuda":
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(
        f"{prefix}: allocated={allocated:.3f} GiB, reserved={reserved:.3f} GiB, "
        f"peak_allocated={peak_allocated:.3f} GiB, peak_reserved={peak_reserved:.3f} GiB"
    )


def move_inputs_to_device(inputs, device):
    return {key: value.to(device) for key, value in inputs.items()}


def is_hf_model_dir(model_dir: Path):
    return (model_dir / "config.json").exists() and (
        (model_dir / "model.safetensors.index.json").exists() or any(model_dir.glob("model-*.safetensors"))
    )


def is_torch_dist_iter_dir(model_dir: Path):
    metadata_path = model_dir / "metadata.json"
    common_path = model_dir / "common.pt"
    if not metadata_path.exists() or not common_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception:
        return False
    return metadata.get("sharded_backend") == "torch_dist"


def prepare_model_dir(args):
    model_dir = Path(args.model_dir).resolve()
    if is_hf_model_dir(model_dir):
        return model_dir

    if not is_torch_dist_iter_dir(model_dir):
        raise ValueError(
            f"{model_dir} is neither an HF model dir nor a torch_dist iter checkpoint dir that this loader can convert."
        )

    if args.origin_hf_dir is None:
        raise ValueError("--origin-hf-dir is required when --model-dir points to a torch_dist iter checkpoint.")

    converted_model_dir = (
        Path(args.converted_model_dir).resolve() if args.converted_model_dir else model_dir.parent / f"{model_dir.name}_hf_streaming"
    )

    need_convert = args.force_convert or not is_hf_model_dir(converted_model_dir)
    if need_convert:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "convert_torch_dist_to_hf.py"),
            "--input-dir",
            str(model_dir),
            "--output-dir",
            str(converted_model_dir),
            "--origin-hf-dir",
            str(Path(args.origin_hf_dir).resolve()),
            "--load-batch-size",
            str(args.load_batch_size),
            "--chunk-size",
            str(args.chunk_size),
            "--force",
        ]
        print("auto_convert_start")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        print("auto_convert_end")
    else:
        print(f"reusing_converted_model_dir={converted_model_dir}")

    return converted_model_dir


def main():
    args = parse_args()
    resolved_model_dir = prepare_model_dir(args)
    device = torch.device(args.device)
    torch_dtype = get_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_dir, trust_remote_code=args.trust_remote_code)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    load_start = time.time()
    if args.load_mode == "staged":
        device_map = {"": str(device)} if device.type == "cuda" else None
        max_memory = None
        if device.type == "cuda" and args.max_gpu_memory is not None:
            device_map = "auto"
            max_memory = {torch.cuda.current_device(): args.max_gpu_memory, "cpu": args.max_cpu_memory}
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )
        model.eval()
        if device.type == "cuda":
            model.to(device)

    model.eval()
    load_time = time.time() - load_start
    print(f"load_time_sec={load_time:.3f}")
    report_cuda_peak("after_load", device)

    inputs = tokenizer(args.prompt, return_tensors="pt")
    if device.type == "cuda":
        inputs = move_inputs_to_device(inputs, device)
        torch.cuda.reset_peak_memory_stats()

    gen_start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    gen_time = time.time() - gen_start
    print(f"generate_time_sec={gen_time:.3f}")
    report_cuda_peak("after_generate", device)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("generated_text_start")
    print(decoded)
    print("generated_text_end")


if __name__ == "__main__":
    main()
