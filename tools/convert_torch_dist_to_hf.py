import argparse
import importlib.util
import json
import os
import pickle
import re
import shutil
import time
from pathlib import Path
from typing import Optional

import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp
from transformers import AutoConfig
from typing_extensions import override


def _load_module(module_name, relative_path):
    module_path = Path(__file__).resolve().parent.parent / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_CONVERT_GLM4 = _load_module(
    "april_megatron_to_hf_glm4", "slime/backends/megatron_utils/megatron_to_hf/glm4.py"
).convert_glm4_to_hf
_CONVERT_QWEN2 = _load_module(
    "april_megatron_to_hf_qwen2", "slime/backends/megatron_utils/megatron_to_hf/qwen2.py"
).convert_qwen2_to_hf
_CONVERT_QWEN3MOE = _load_module(
    "april_megatron_to_hf_qwen3moe", "slime/backends/megatron_utils/megatron_to_hf/qwen3moe.py"
).convert_qwen3moe_to_hf
_CONVERT_DEEPSEEKV3 = _load_module(
    "april_megatron_to_hf_deepseekv3", "slime/backends/megatron_utils/megatron_to_hf/deepseekv3.py"
).convert_deepseekv3_to_hf
_CONVERT_LLAMA = _load_module(
    "april_megatron_to_hf_llama", "slime/backends/megatron_utils/megatron_to_hf/llama.py"
).convert_llama_to_hf


def quantize_param(name, weight, weight_block_size):
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"
    fp8_min = torch.finfo(torch.float8_e4m3fn).min
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    if weight_block_size is not None:
        block_n, block_k = weight_block_size[0], weight_block_size[1]
        n_tiles = weight.shape[0] // block_n
        k_tiles = weight.shape[1] // block_k
        qweight = weight.reshape(n_tiles, block_n, k_tiles, block_k)
        block_max = torch.max(torch.abs(qweight), dim=1, keepdim=True)[0]
        block_max = torch.max(block_max, dim=3, keepdim=True)[0]
        scale = block_max.to(torch.float32) / fp8_min
        qweight = (qweight / scale).clamp(min=fp8_min, max=fp8_max).reshape(weight.shape).to(torch.float8_e4m3fn)
        scale = scale.squeeze()
        scale_name = name.replace(".weight", ".weight_scale_inv")
    else:
        scale = weight.abs().max().clamp(min=1e-12).to(torch.float32) / fp8_max
        qweight = (weight / scale).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
        scale = scale.view(1)
        scale_name = name.replace(".weight", ".weight_scale")
    return [(name, qweight), (scale_name, scale)]


def quantize_params(args, megatron_name, converted_named_params, quantization_config):
    if quantization_config is None:
        return converted_named_params

    assert quantization_config["quant_method"] == "fp8"
    assert quantization_config["fmt"] == "e4m3"
    assert quantization_config["activation_scheme"] == "dynamic"
    weight_block_size = quantization_config.get("weight_block_size", None)

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, megatron_name)
    if not match:
        return converted_named_params

    _, rest = match.groups()
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, _ = match.groups()
        if rest in ["linear_fc1", "linear_fc2"]:
            quantized = []
            for converted_name, param in converted_named_params:
                if converted_name.endswith("_scale"):
                    continue
                quantized.extend(quantize_param(converted_name, param, weight_block_size))
            return quantized

    shared_expert_pattern = r"mlp.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest = match.groups()[0]
        if rest in ["linear_fc1.weight", "linear_fc2.weight"]:
            quantized = []
            for converted_name, param in converted_named_params:
                quantized.extend(quantize_param(converted_name, param, weight_block_size))
            return quantized

    if rest in [
        "self_attention.linear_proj.weight",
        "self_attention.linear_qkv.weight",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.weight",
    ]:
        quantized = []
        for converted_name, param in converted_named_params:
            quantized.extend(quantize_param(converted_name, param, weight_block_size))
        return quantized

    return converted_named_params


def convert_to_hf(args, model_name, name, param, quantization_config=None):
    if "glm4" in model_name:
        converted_named_tensors = _CONVERT_GLM4(args, name, param)
    elif "qwen3moe" in model_name:
        converted_named_tensors = _CONVERT_QWEN3MOE(args, name, param)
    elif "qwen2" in model_name or "qwen3" in model_name:
        converted_named_tensors = _CONVERT_QWEN2(args, name, param)
    elif "deepseekv3" in model_name:
        converted_named_tensors = _CONVERT_DEEPSEEKV3(args, name, param)
    elif "llama" in model_name:
        converted_named_tensors = _CONVERT_LLAMA(args, name, param)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if not quantization_config:
        return converted_named_tensors

    return quantize_params(args, name, converted_named_tensors, quantization_config)


def remove_padding(name, param, vocab_size):
    if name == "module.module.embedding.word_embeddings.weight" or name == "module.module.output_layer.weight":
        return param[:vocab_size]
    return param


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


pickle.Unpickler = UnpicklerWrapper


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: Optional[dist_cp.metadata.Metadata] = None,
        is_coordinator: bool = False,
    ) -> None:
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k or "_state" in k:
                continue
            print(f"find {k} in torch_dist ckpt")
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def get_expert_param(args, name, param):
    if ".experts." not in name:
        yield name, param
        return

    num_experts = args.num_experts
    match = re.search(r"mlp.experts\.(.+)\.weight(\d+)", name)
    if not match:
        assert param.shape[0] == num_experts
        for expert_id in range(num_experts):
            expert_name = name.replace(".experts.experts.", f".experts.") + str(expert_id)
            expert_param = param[expert_id]
            yield expert_name, expert_param
    else:
        yield name, param


def get_layer_param(args, name, param):
    if ".layers." not in name:
        yield name, param
        return

    num_layers = args.num_layers
    match = re.search(r"\.layers\.(\d+)\.", name)
    if not match:
        assert param.shape[0] == num_layers
        for layer_id in range(num_layers):
            layer_name = name.replace(".layers.", f".layers.{layer_id}.")
            layer_param = param[layer_id]
            yield from get_expert_param(args, layer_name, layer_param)
    else:
        yield from get_expert_param(args, name, param)


def get_named_params(args, state_dict):
    for name, param in state_dict.items():
        name = f"module.module.{name}"
        yield from get_layer_param(args, name, param)


def save_tensors(args, model_name, state_dict, output_dir, chunk_size, vocab_size=None):
    # for slime update_weight compatible
    setattr(args, "sglang_enable_ep_moe", False)

    print(f"start saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    # 2GB
    current_size = 0
    total_size = 0
    modeltensors = [{}]
    for name, param in get_named_params(args, state_dict):
        if vocab_size:
            param = remove_padding(name, param, vocab_size)
        converted_named_tensors = convert_to_hf(args, model_name, name, param)
        for converted_name, converted_param in converted_named_tensors:
            tensor_size = converted_param.numel() * converted_param.element_size()
            if tensor_size + current_size > chunk_size:
                modeltensors.append({})
                current_size = 0
            modeltensors[-1][converted_name] = converted_param
            current_size += tensor_size
            total_size += tensor_size

    metadata = {"metadata": {"total_size": total_size}, "weight_map": {}}

    num_files = len(modeltensors)
    for i, tensors in enumerate(modeltensors):
        filename = f"model-{i:05d}-of-{num_files:05d}.safetensors"
        for key in tensors.keys():
            metadata["weight_map"][key] = filename
    index_filepath = os.path.join(output_dir, "model.safetensors.index.json")
    json.dump(metadata, open(index_filepath, "w"), indent=2)
    print(f"{index_filepath} saved.")

    for i, tensors in enumerate(modeltensors):
        filename = f"model-{i:05d}-of-{num_files:05d}.safetensors"
        t = time.time()
        filepath = os.path.join(output_dir, filename)
        safetensors.torch.save_file(tensors, filepath)
        print(f"{filename} saved in {time.time() - t:.2f} sec.")


def get_model_tensor_entries(input_dir):
    storage_reader = WrappedStorageReader(input_dir)
    metadata = storage_reader.read_metadata()
    entries = []
    for name, storage_meta in metadata.state_dict_metadata.items():
        if "optimizer" in name or "_state" in name:
            continue
        if isinstance(storage_meta, dist_cp.metadata.TensorStorageMetadata):
            entries.append((name, storage_meta))
    return entries


def load_tensor_entries(input_dir, entries):
    storage_reader = WrappedStorageReader(input_dir)
    state_dict = {}
    for name, storage_meta in entries:
        state_dict[name] = torch.empty(storage_meta.size, dtype=storage_meta.properties.dtype)

    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=storage_reader,
        planner=dist_cp.default_planner.DefaultLoadPlanner(),
        no_dist=True,
    )
    return state_dict


def save_tensors_streaming(args, model_name, input_dir, output_dir, chunk_size, vocab_size=None, load_batch_size=1):
    # for slime update_weight compatible
    setattr(args, "sglang_enable_ep_moe", False)

    os.makedirs(output_dir, exist_ok=True)
    entries = get_model_tensor_entries(input_dir)
    print(
        f"start streaming conversion from {input_dir} to {output_dir}, "
        f"{len(entries)} tensor entries, load_batch_size={load_batch_size}"
    )

    current_size = 0
    total_size = 0
    shard_id = 0
    current_tensors = {}
    weight_map = {}

    def flush_current_tensors():
        nonlocal current_size, shard_id
        if not current_tensors:
            return

        shard_id += 1
        filename = f"model-{shard_id:05d}.safetensors"
        filepath = os.path.join(output_dir, filename)
        t = time.time()
        safetensors.torch.save_file(current_tensors, filepath)
        for key in current_tensors.keys():
            weight_map[key] = filename
        print(
            f"{filename} saved in {time.time() - t:.2f} sec, "
            f"tensors={len(current_tensors)}, shard_size={current_size / 1024**3:.2f} GiB"
        )
        current_tensors.clear()
        current_size = 0

    for start in range(0, len(entries), load_batch_size):
        batch_entries = entries[start : start + load_batch_size]
        batch_desc = ", ".join(name for name, _ in batch_entries[:3])
        if len(batch_entries) > 3:
            batch_desc += ", ..."
        print(
            f"loading raw tensor batch {start // load_batch_size + 1}/"
            f"{(len(entries) + load_batch_size - 1) // load_batch_size}: {batch_desc}"
        )
        batch_state_dict = load_tensor_entries(input_dir, batch_entries)

        for name, param in get_named_params(args, batch_state_dict):
            if vocab_size:
                param = remove_padding(name, param, vocab_size)
            converted_named_tensors = convert_to_hf(args, model_name, name, param)
            for converted_name, converted_param in converted_named_tensors:
                converted_param = converted_param.contiguous()
                tensor_size = converted_param.numel() * converted_param.element_size()
                if current_tensors and current_size + tensor_size > chunk_size:
                    flush_current_tensors()
                current_tensors[converted_name] = converted_param
                current_size += tensor_size
                total_size += tensor_size

        del batch_state_dict

    flush_current_tensors()

    metadata = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    index_filepath = os.path.join(output_dir, "model.safetensors.index.json")
    json.dump(metadata, open(index_filepath, "w"), indent=2)
    print(f"{index_filepath} saved.")


def copy_assets(origin_hf_dir, output_dir):
    for filename in os.listdir(origin_hf_dir):
        if filename == "model.safetensors.index.json" or filename.endswith(".safetensors"):
            continue
        origin_filename = os.path.join(origin_hf_dir, filename)
        if not os.path.isfile(origin_filename):
            print(f"Skip {filename}, not a file.")
            continue
        src, dst = origin_filename, os.path.join(output_dir, filename)
        print(f"copy from {src} to {dst}")
        shutil.copy(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        default=None,
        help="use the origin hf dir to copy files like tokenizer, config.json, etc.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5 * 1024**3,
        help="Chunk size for saving tensors, default is 5GB.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocab size for removing padding, if applicable. If not provided, no padding will be removed.",
    )
    parser.add_argument(
        "--streaming",
        dest="streaming",
        action="store_true",
        help="Stream raw torch_dist tensors in small batches instead of loading the full state_dict at once.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable streaming conversion and load the full non-optimizer state_dict at once.",
    )
    parser.add_argument(
        "--load-batch-size",
        type=int,
        default=1,
        help="Number of raw torch_dist tensor entries to load at a time when --streaming is enabled.",
    )
    parser.set_defaults(streaming=True)
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    if args.model_name is None and args.origin_hf_dir is None:
        raise ValueError(
            "Either --model-name or --origin-hf-dir must be provided, so that we can know the name of the params."
        )

    if args.model_name is None:
        hf_config = AutoConfig.from_pretrained(args.origin_hf_dir, trust_remote_code=True)
        args.model_name = type(hf_config).__name__.lower()

    megatron_args = torch.load(os.path.join(args.input_dir, "common.pt"), weights_only=False)["args"]
    if args.streaming:
        save_tensors_streaming(
            megatron_args,
            args.model_name,
            args.input_dir,
            args.output_dir,
            args.chunk_size,
            args.vocab_size,
            args.load_batch_size,
        )
    else:
        state_dict = {}
        print(f"loading model from {args.input_dir}")
        t = time.time()
        dist_cp.state_dict_loader._load_state_dict(
            state_dict,
            storage_reader=WrappedStorageReader(args.input_dir),
            planner=EmptyStateDictLoadPlanner(),
            no_dist=True,
        )
        print(f"model loaded in {time.time()-t:.2f} sec.")

        save_tensors(megatron_args, args.model_name, state_dict, args.output_dir, args.chunk_size, args.vocab_size)

    if args.origin_hf_dir:
        copy_assets(args.origin_hf_dir, args.output_dir)
