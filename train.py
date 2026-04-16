import os

import ray

from slime.ray.placement_group import create_actor_group, create_placement_groups, create_rollout_group
from slime.utils.arguments import parse_args


def train(args):
    if not ray.is_initialized():
        if os.environ.get("SLIME_MODAL_DIRECT", "0") == "1":
            ray_tmpdir = os.environ.get("RAY_TMPDIR", "/tmp/ray")
            spill_dir = os.path.join(ray_tmpdir, "spill")
            os.makedirs(spill_dir, exist_ok=True)
            object_store_memory = int(os.environ.get("RAY_OBJECT_STORE_MEMORY", str(256 * 1024 * 1024)))

            ray.init(
                num_gpus=int(os.environ.get("RAY_NUM_GPUS", "1")),
                include_dashboard=False,
                ignore_reinit_error=True,
                _temp_dir=ray_tmpdir,
                _plasma_directory=os.environ.get("RAY_PLASMA_DIRECTORY", "/tmp"),
                object_spilling_directory=spill_dir,
                object_store_memory=object_store_memory,
            )
        else:
            ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), ignore_reinit_error=True)

    # allocate the GPUs
    pgs = create_placement_groups(args)

    actor_model = create_actor_group(args, pgs["actor"])

    # create the rollout generator, with sglang engines inside.
    rollout_generator = create_rollout_group(args, pgs["rollout"])

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_generator.data_buffer.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
    assert args.num_rollout > 0

    # sync the initialization (model initalization, load checkpoint, etc.)
    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
    )
    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if args.rollout_global_dataset:
        ray.get(rollout_generator.data_buffer.load.remote(args.start_rollout_id - 1))

    # initialize the connection for weight update during training
    ray.get(actor_model.async_init_weight_update_connections(rollout_generator))

    if args.offload:
        ray.get(rollout_generator.async_onload())

    # always update weight first so that sglang has the loaded weights from training.
    ray.get(actor_model.async_update_weights())

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0:
            ray.get(rollout_generator.async_generate(rollout_id, evaluation=True))
            ray.get(actor_model.async_eval(rollout_id))

        ray.get(rollout_generator.async_generate(rollout_id))

        if args.offload:
            ray.get(rollout_generator.async_offload())

        ray.get(actor_model.async_train(rollout_id))

        if (
            not getattr(args, "save_hf_only_final", False)
            and args.save_interval is not None
            and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
            )
        ):
            ray.get(actor_model.async_save_model(rollout_id))
            if args.rollout_global_dataset:
                ray.get(rollout_generator.data_buffer.save.remote(rollout_id))

        if args.offload:
            ray.get(actor_model.async_offload())
            ray.get(rollout_generator.async_onload())

        ray.get(actor_model.async_update_weights())

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(rollout_generator.async_generate(rollout_id, evaluation=True))
            ray.get(actor_model.async_eval(rollout_id))

    if getattr(args, "save_hf_only_final", False) and args.num_rollout > 0:
        final_rollout_id = args.num_rollout - 1
        ray.get(actor_model.async_save_model(final_rollout_id))
        if args.rollout_global_dataset:
            ray.get(rollout_generator.data_buffer.save.remote(final_rollout_id))


if __name__ == "__main__":
    args = parse_args()
    train(args)
