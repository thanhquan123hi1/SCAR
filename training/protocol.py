"""Versioned training policy, separate from the unchanged evaluation metrics."""
from __future__ import annotations

from training.metrics.surface_distance import protocol_for_dataset


NO_VAL_PROTOCOL = {
    "id": "myops380_train304_test76_last_v1",
    "validation": False,
    "checkpoint_selection": "last_completed_epoch",
    "budget_unit": "epochs",
    "train_patients": 304,
    "test_patients": 76,
}


def training_protocol(no_val=False):
    if no_val:
        return dict(NO_VAL_PROTOCOL)
    return {"id": "validation_inclusive_dice_v1", "validation": True,
            "checkpoint_selection": "avg_pathology_inclusive_dice"}


def metric_protocol(dataset_id, no_val=False):
    protocol = dict(protocol_for_dataset(dataset_id))
    if no_val:
        if dataset_id != "myops380":
            raise ValueError("--no-val currently supports only myops380")
        protocol.pop("selection_metric")
    return protocol


def validate_training_checkpoint(checkpoint, *, require_complete=False):
    """Reject mixed policy metadata and incomplete fixed-budget paper results."""
    args = checkpoint.get("args", {})
    no_val = args.get("no_val", False)
    saved = checkpoint.get("training_protocol")
    if saved is not None and saved != training_protocol(no_val):
        raise ValueError("Checkpoint training protocol disagrees with no_val")
    if not no_val:
        return False
    if saved != NO_VAL_PROTOCOL or args.get("dataset_id", "myops380") != "myops380":
        raise ValueError("Missing or invalid no-val training protocol")
    planned = args.get("max_epochs")
    completed = checkpoint.get("epoch", -1) + 1
    if (not isinstance(planned, int) or planned < 1 or not 0 < completed <= planned
            or checkpoint.get("planned_epochs") != planned
            or checkpoint.get("completed_epochs") != completed
            or checkpoint.get("training_complete") is not (completed == planned)):
        raise ValueError("Inconsistent fixed training budget in checkpoint")
    if require_complete and completed != planned:
        raise ValueError(f"Incomplete training budget: {completed}/{planned} epochs; resume the final checkpoint before test")
    total = checkpoint.get("total_updates", 0)
    successful = checkpoint.get("global_step", -1)
    skipped = checkpoint.get("amp_skipped_updates", -1)
    batch_size, accum_steps = args.get("batch_size"), args.get("accum_steps")
    if any(type(value) is not int or value < 1 for value in (batch_size, accum_steps)):
        raise ValueError("Invalid batch size / accumulation in fixed-budget checkpoint")
    # The data contract locks 1,972 slices. The optional rare sampler also draws
    # exactly this many samples per epoch. Derive the budget independently of
    # the saved total, so mutually consistent but incorrect counters fail.
    batches = (1972 + batch_size - 1) // batch_size
    updates_per_epoch = (batches + accum_steps - 1) // accum_steps
    if (any(type(value) is not int for value in (total, successful, skipped))
            or total != planned * updates_per_epoch or successful < 1 or skipped < 0
            or successful + skipped != completed * updates_per_epoch
            or checkpoint.get("scheduler", {}).get("last_epoch") != successful):
        raise ValueError("Inconsistent optimizer updates / scheduler in fixed-budget checkpoint")
    return True
