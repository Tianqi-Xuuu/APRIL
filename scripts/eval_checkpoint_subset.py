import json
import os
from pathlib import Path

import ray

from slime.ray.placement_group import create_actor_group, create_placement_groups, create_rollout_group
from slime.utils.arguments import parse_args


def add_custom_arguments(parser):
    parser.add_argument("--eval-output-json", type=str, required=True)
    parser.add_argument("--eval-rollout-id", type=int, required=True)
    return parser


def _to_builtin(value):
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    try:
        import numpy as np
        import torch

        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def main():
    args = parse_args(add_custom_arguments)

    pgs = create_placement_groups(args)
    actor_model = create_actor_group(args, pgs["actor"])
    rollout_generator = create_rollout_group(args, pgs["rollout"])

    try:
        start_rollout_ids = ray.get(
            actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
        )
        assert len(set(start_rollout_ids)) == 1

        ray.get(actor_model.async_init_weight_update_connections(rollout_generator))

        if args.offload:
            ray.get(rollout_generator.async_onload())

        ray.get(actor_model.async_update_weights())

        eval_rollout_id = args.eval_rollout_id
        ray.get(rollout_generator.async_generate(eval_rollout_id, evaluation=True))
        data = ray.get(rollout_generator.data_buffer.get_data.remote(eval_rollout_id, evaluation=True))
        output = {
            "checkpoint_rollout_id": eval_rollout_id,
            "eval_data_name": args.eval_prompt_data[0],
            "eval_data_path": args.eval_prompt_data[1],
            "metrics": {},
        }
        for key, value in data.items():
            rewards = value["rewards"]
            truncated = value.get("truncated", [])
            output["metrics"][key] = {
                "reward_mean": float(sum(rewards) / len(rewards)),
                "num_samples": int(len(rewards)),
            }
            if truncated:
                output["metrics"][key]["truncated_ratio"] = float(sum(truncated) / len(truncated))
            print(
                f"eval {eval_rollout_id}: "
                f"{{'eval/{key}': {output['metrics'][key]['reward_mean']}, "
                f"'eval/{key}-truncated_ratio': {output['metrics'][key].get('truncated_ratio', 0.0)}}}"
            )

        output_path = Path(args.eval_output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_to_builtin(output), indent=2, sort_keys=True) + "\n")
        print(json.dumps(output, sort_keys=True))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
