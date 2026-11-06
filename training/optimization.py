"""AdamW and polynomial LR matching I-MMSeg's training loop."""
import torch


OPTIMIZATION_PROTOCOL = {
    "id": "im_mseg_adamw_poly_v1",
    "optimizer": "AdamW",
    "weight_decay_scope": "all_parameters",
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "scheduler": "polynomial_after_update_zero_based_iteration",
    "power": 0.9,
}


def build_optimizer(model, base_lr, weight_decay):
    return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay,
                             betas=(0.9, 0.999), eps=1e-8)


def build_scheduler(optimizer, total_updates):
    if not isinstance(total_updates, int) or total_updates < 1:
        raise ValueError("total_updates must be a positive integer")
    # LambdaLR starts at step 0. I-MMSeg computes f(iter_num=0) *after*
    # the first optimizer update, so its first two updates use the base LR.
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: max(0.0, 1.0 - max(0, step - 1) / total_updates) ** 0.9,
    )


def validate_optimization_checkpoint(checkpoint):
    if checkpoint.get("optimization_protocol") != OPTIMIZATION_PROTOCOL:
        raise ValueError("Resume optimizer/LR scheduler protocol differs from I-MMSeg. "
                         "Start a new run; use --init-weights to reuse model weights.")
