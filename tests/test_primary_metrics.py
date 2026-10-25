"""Primary metric semantics and strict compatibility with old reporting metadata."""
from copy import deepcopy

import numpy as np
import pytest

from training.dataset.benchmark_profiles import DATASET_IDS
from training.metrics.surface_distance import (
    BENCHMARK_PROTOCOL, benchmark_protocol_matches, dataset_rows,
    protocol_for_dataset, summarize_dataset_rows,
)


def legacy_protocol():
    return dict(BENCHMARK_PROTOCOL, id="myops380_voxel_v1",
                selection_metric="avg_pathology_dice",
                official_reference="I_MMSeg@90f46c4eb72924509895fcda6bc6a3b8c3316e66/utils.py:171-182",
                official_empty_policy="if either empty: Dice=1 when pred<200 and target<=200, else 0; HD95=0")


@pytest.mark.parametrize("dataset", DATASET_IDS)
def test_primary_reports_penalize_small_misses_and_keep_distance_counts(dataset):
    target = np.zeros((8, 8, 4), dtype=np.uint8)
    target[2:4, 2:4, 1:3] = 3
    missed = np.zeros_like(target)
    rows = dataset_rows(missed, target, "missed", dataset_id=dataset, spacing=[1, 1, 5])
    rows += dataset_rows(target, target, "exact", dataset_id=dataset, spacing=[1, 1, 5])
    rows += dataset_rows(missed, missed, "negative", dataset_id=dataset, spacing=[1, 1, 5])
    summary = summarize_dataset_rows(rows, dataset_id=dataset)
    scar = summary["scar"]
    assert scar["mean_dice"] == 0.5  # miss=0, exact=1, both-empty excluded
    assert scar["dice_defined_cases"] == 2
    assert scar["dice_undefined_cases"] == 1
    assert scar["status_counts"] == {"prediction_empty": 1, "ok": 1, "both_empty": 1}
    unit = "voxel" if dataset == "myops380" else "mm"
    assert scar[f"mean_hd95_{unit}"] == 0
    assert scar[f"hd95_{unit}_defined_cases"] == 1
    assert scar[f"hd95_{unit}_undefined_cases"] == 2
    assert summary["avg_pathology_dice"] is None  # no defined exclusive-edema cases
    assert summary["avg_pathology_inclusive_dice"] == 0.5
    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)
    assert not any(k.startswith("official_") for k in keys([rows, summary, protocol_for_dataset(dataset)]))


@pytest.mark.parametrize("dataset", DATASET_IDS)
def test_false_positives_on_negative_patients_are_zero_dice(dataset):
    target = np.zeros((8, 8, 4), dtype=np.uint8)
    prediction = target.copy()
    prediction[2:4, 2:4, 1:3] = 3
    row = next(r for r in dataset_rows(prediction, target, "fp", dataset_id=dataset, spacing=[1, 1, 5]) if r["region"] == "scar")
    assert row["dice"] == row["iou"] == row["precision"] == 0
    assert row["status"] == "target_empty"
    assert row["hd95"] is None


def test_only_exact_legacy_reporting_protocol_is_accepted():
    original = legacy_protocol()
    assert benchmark_protocol_matches(original)
    assert original == legacy_protocol()  # validation must not mutate checkpoint metadata
    for dataset in DATASET_IDS:
        assert benchmark_protocol_matches(protocol_for_dataset(dataset), dataset)
    for key, value in (("primary_empty_policy", "both_empty=1"), ("distance_unit", "mm"),
                       ("selection_metric", "test_dice"), ("official_empty_policy", "changed"),
                       ("id", "unknown")):
        changed = dict(original, **{key: value})
        assert not benchmark_protocol_matches(changed), key
    changed = deepcopy(original)
    changed["regions"]["scar"] = [2]
    assert not benchmark_protocol_matches(changed)
    assert not benchmark_protocol_matches(dict(original, unexpected=True))
    assert not benchmark_protocol_matches(None)
    assert not benchmark_protocol_matches({"id": "myops380_voxel_v1"})
    assert not benchmark_protocol_matches(original, "myopspp_bc80")


@pytest.mark.parametrize("dataset", DATASET_IDS)
def test_old_selection_is_readable_but_cannot_resume(dataset):
    current = protocol_for_dataset(dataset)
    assert current["selection_metric"] == "avg_pathology_inclusive_dice"
    assert benchmark_protocol_matches(current, dataset, allow_legacy_selection=False)
    old_id = ("myops380_voxel_v2" if dataset == "myops380" else
              f"{dataset}_roi_mm_v1" if "evaluation_grid" in current else "myopspp_bc80_mm_v1")
    old = dict(current, id=old_id, selection_metric="avg_pathology_dice")
    assert benchmark_protocol_matches(old, dataset)
    assert not benchmark_protocol_matches(old, dataset, allow_legacy_selection=False)
    old["regions"] = dict(old["regions"], edema_inclusive=[2])
    assert not benchmark_protocol_matches(old, dataset)


def test_trainer_rejects_old_selection_before_reading_data(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch
    from training.trainer.trainer import trainer_Myops
    args = SimpleNamespace(dataset_id="myops380", resume="last.pth")
    with patch("training.trainer.trainer.load_checkpoint", return_value={"benchmark_protocol": legacy_protocol()}):
        with pytest.raises(ValueError, match="--init-weights"):
            trainer_Myops(args, None, tmp_path / "untouched")
    assert not (tmp_path / "untouched").exists()


@pytest.mark.parametrize("source", ["cli", "yaml"])
def test_periodic_checkpoints_are_rejected(source, tmp_path):
    from training.train import parse_args
    if source == "cli":
        argv = ["--save-every", "10"]
    else:
        config = tmp_path / "periodic.yaml"
        config.write_text("train:\n  save_every: 10\n")
        argv = ["--config", str(config)]
    with pytest.raises(SystemExit if source == "cli" else ValueError):
        parse_args(argv)
