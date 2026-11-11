"""Versioned training policy, separate from the unchanged evaluation metrics."""
from __future__ import annotations

from training.metrics.surface_distance import protocol_for_dataset
from training.dataset.benchmark_profiles import MYOPSPP_DATASET


NO_VAL_PROTOCOL = {
    "id": "myops380_train304_test76_last_v1",
    "validation": False,
    "checkpoint_selection": "last_completed_epoch",
    "budget_unit": "epochs",
    "train_patients": 304,
    "test_patients": 76,
}


def training_protocol(no_val=False, dataset_id="myops380"):
    if dataset_id == MYOPSPP_DATASET:
        if not no_val:
            raise ValueError("MyoPS++ predicted ROI requires no-val 64/16 training")
        return dict(NO_VAL_PROTOCOL, id=f"{MYOPSPP_DATASET}_train64_test16_last_v1",
                    train_patients=64, test_patients=16)
    if no_val:
        return dict(NO_VAL_PROTOCOL)
    return {"id": "validation_inclusive_dice_v1", "validation": True,
            "checkpoint_selection": "avg_pathology_inclusive_dice"}


def metric_protocol(dataset_id, no_val=False):
    protocol = dict(protocol_for_dataset(dataset_id))
    if no_val:
        if dataset_id not in ("myops380", MYOPSPP_DATASET):
            raise ValueError("Unsupported no-val dataset")
        protocol.pop("selection_metric", None)
    return protocol


def validate_training_checkpoint(checkpoint, *, require_complete=False):
    """Reject mixed policy metadata and incomplete fixed-budget paper results."""
    args = checkpoint.get("args", {})
    no_val = args.get("no_val", False)
    dataset_id = args.get("dataset_id", "myops380")
    saved = checkpoint.get("training_protocol")
    expected = training_protocol(no_val, dataset_id)
    if saved is not None and saved != expected:
        raise ValueError("Checkpoint training protocol disagrees with no_val")
    if not no_val:
        return False
    if saved != expected or dataset_id not in ("myops380", MYOPSPP_DATASET):
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
    # MyoPS380 locks 1,972 slices; predicted ROI records its locked slice count.
    # The optional rare sampler draws that many samples. Derive the budget independently of
    # the saved total, so mutually consistent but incorrect counters fail.
    samples = 1972
    if dataset_id == MYOPSPP_DATASET:
        data = checkpoint.get("benchmark_data", {})
        if data.get("dataset_id") != dataset_id or data.get("patient_counts") != {"train": 64, "test_vol": 16}:
            raise ValueError("Missing or invalid MyoPS++ locked data")
        samples = data.get("sample_counts", {}).get("train")
        if type(samples) is not int or samples < 64:
            raise ValueError("Invalid MyoPS++ training slice count")
    batches = (samples + batch_size - 1) // batch_size
    updates_per_epoch = (batches + accum_steps - 1) // accum_steps
    if (any(type(value) is not int for value in (total, successful, skipped))
            or total != planned * updates_per_epoch or successful < 1 or skipped < 0
            or successful + skipped != completed * updates_per_epoch
            or checkpoint.get("scheduler", {}).get("last_epoch") != successful):
        raise ValueError("Inconsistent optimizer updates / scheduler in fixed-budget checkpoint")
    return True
