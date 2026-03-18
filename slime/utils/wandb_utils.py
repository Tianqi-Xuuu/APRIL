def get_wandb_module():
    try:
        import wandb
    except ImportError:
        return None
    return wandb


def require_wandb(args=None):
    wandb = get_wandb_module()
    if wandb is None:
        message = "wandb is not installed."
        if args is not None and getattr(args, "use_wandb", False):
            message += " Install it or run without --use-wandb."
        raise ImportError(message)
    return wandb
