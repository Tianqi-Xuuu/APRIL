import json
import os
import shutil

import safetensors.torch
import torch
import torch.distributed as dist
from megatron.core import mpu
from transformers import AutoConfig, AutoTokenizer

from . import update_weight_utils

DEFAULT_HF_CHUNK_SIZE = 5 * 1024**3
_SKIP_ASSET_FILENAMES = {
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}


def _should_skip_asset(filename):
    if filename in _SKIP_ASSET_FILENAMES:
        return True
    return filename.endswith(".safetensors") or filename.endswith(".bin")


def _copy_hf_assets(origin_hf_dir, output_dir):
    for filename in os.listdir(origin_hf_dir):
        if _should_skip_asset(filename):
            continue

        src = os.path.join(origin_hf_dir, filename)
        if not os.path.isfile(src):
            continue

        dst = os.path.join(output_dir, filename)
        shutil.copy(src, dst)


class HFShardWriter:
    def __init__(self, output_dir, chunk_size):
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.shard_id = 1
        self.current_size = 0
        self.total_size = 0
        self.current_tensors = {}
        self.weight_map = {}

        os.makedirs(output_dir, exist_ok=True)

    def _flush(self):
        if not self.current_tensors:
            return

        filename = f"model-{self.shard_id:05d}.safetensors"
        filepath = os.path.join(self.output_dir, filename)
        safetensors.torch.save_file(self.current_tensors, filepath)
        for key in self.current_tensors.keys():
            self.weight_map[key] = filename

        self.shard_id += 1
        self.current_tensors = {}
        self.current_size = 0

    def add_tensor(self, name, param):
        tensor = _materialize_cpu_tensor(name, param)
        tensor_size = tensor.numel() * tensor.element_size()

        if self.current_tensors and self.current_size + tensor_size > self.chunk_size:
            self._flush()

        self.current_tensors[name] = tensor
        self.current_size += tensor_size
        self.total_size += tensor_size

    def finalize(self):
        self._flush()
        metadata = {"metadata": {"total_size": self.total_size}, "weight_map": self.weight_map}
        with open(os.path.join(self.output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(metadata, f, indent=2)


def _materialize_cpu_tensor(name, param):
    tensor = param.detach()
    try:
        if tensor.is_cuda:
            # Saving happens right after training/eval work on the same device. Make the
            # device state explicit before copying weights back to host memory.
            torch.cuda.synchronize(tensor.device)
            tensor = tensor.contiguous().clone()
            torch.cuda.synchronize(tensor.device)
            return tensor.to(device="cpu", copy=True).contiguous()
        return tensor.contiguous().cpu()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to materialize tensor {name} "
            f"(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}, stride={tensor.stride()})"
        ) from exc


def export_hf_checkpoint(args, model, iteration, chunk_size=DEFAULT_HF_CHUNK_SIZE, cpu_state_dict=None):
    output_root = os.path.join(args.save, "hf")
    output_dir = os.path.join(output_root, f"iter_{iteration:07d}")
    is_save_rank = (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
    )

    if is_save_rank:
        os.makedirs(output_root, exist_ok=True)
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        writer = HFShardWriter(output_dir, chunk_size)
    else:
        writer = None

    if is_save_rank:
        hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        model_name = type(hf_config).__name__.lower() if args.model_name is None else args.model_name
        quantization_config = getattr(hf_config, "quantization_config", None)
        vocab_size = args.vocab_size
        if vocab_size is None:
            vocab_size = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True).vocab_size
    else:
        model_name = None
        quantization_config = None
        vocab_size = args.vocab_size

    vocab_size_list = [vocab_size]
    dist.broadcast_object_list(vocab_size_list, src=0)
    vocab_size = vocab_size_list[0]

    rank = dist.get_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()

    if cpu_state_dict is not None and dist.get_world_size() == 1:
        for name in sorted(cpu_state_dict.keys()):
            param = cpu_state_dict[name]
            merged_param = update_weight_utils.remove_padding(name, param, vocab_size)
            if is_save_rank:
                converted_named_tensors = update_weight_utils.convert_to_hf(
                    args,
                    model_name,
                    name,
                    merged_param,
                    quantization_config,
                )
                for converted_name, converted_param in converted_named_tensors:
                    writer.add_tensor(converted_name, converted_param)

        if is_save_rank:
            writer.finalize()
            _copy_hf_assets(args.hf_checkpoint, output_dir)

            with open(os.path.join(output_root, "latest_checkpointed_iteration.txt"), "w") as f:
                f.write(str(iteration))

            with open(os.path.join(output_root, "latest_checkpointed_path.txt"), "w") as f:
                f.write(output_dir)

            print(f"Saved HF safetensors checkpoint to {output_dir}")

        dist.barrier()
        return

    param_infos = update_weight_utils.get_param_infos(args, model)
    model_params = dict(update_weight_utils.named_parameters(args, model))

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for info in param_infos:
        if rank == info.src_rank:
            param = model_params[info.name]
        else:
            param = torch.empty(info.shape, dtype=info.dtype, device=torch.cuda.current_device())

        if pp_size > 1:
            pp_group_ranks = dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group())
            if info.src_rank in pp_group_ranks:
                dist.broadcast(param, src=info.src_rank, group=mpu.get_pipeline_model_parallel_group())

        if ep_size > 1 and ".experts." in info.name:
            ep_group_ranks = dist.get_process_group_ranks(mpu.get_expert_model_parallel_group())
            src_rank = info.src_rank if info.src_rank in ep_group_ranks else rank
            dist.broadcast(param, src=src_rank, group=mpu.get_expert_model_parallel_group())

        for key, value in info.attrs.items():
            setattr(param, key, value)

        merged_param = update_weight_utils.all_gather_param(info.name, param)
        merged_param = update_weight_utils.remove_padding(info.name, merged_param, vocab_size)

        if is_save_rank:
            converted_named_tensors = update_weight_utils.convert_to_hf(
                args,
                model_name,
                info.name,
                merged_param,
                quantization_config,
            )
            for converted_name, converted_param in converted_named_tensors:
                writer.add_tensor(converted_name, converted_param)

    if is_save_rank:
        writer.finalize()
        _copy_hf_assets(args.hf_checkpoint, output_dir)

        with open(os.path.join(output_root, "latest_checkpointed_iteration.txt"), "w") as f:
            f.write(str(iteration))

        with open(os.path.join(output_root, "latest_checkpointed_path.txt"), "w") as f:
            f.write(output_dir)

        print(f"Saved HF safetensors checkpoint to {output_dir}")

    dist.barrier()
