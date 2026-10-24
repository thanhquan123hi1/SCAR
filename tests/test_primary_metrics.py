"""Primary metric semantics and strict compatibility with old reporting metadata."""
from copy import deepcopy
import ast
from pathlib import Path

import numpy as np
import pytest

from training.dataset.benchmark_profiles import DATASET_IDS
from training.metrics.surface_distance import (
    BENCHMARK_PROTOCOL, benchmark_protocol_matches, dataset_rows,
    protocol_for_dataset, summarize_dataset_rows,
)


def legacy_protocol():
    return dict(BENCHMARK_PROTOCOL, id="myops380_voxel_v1",
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


def extract_function(filename, name):
    """Exercise report functions without orchestrator training/backup side effects."""
    path = Path(__file__).resolve().parents[1] / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    env = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    return env[name]


def test_v7_primary_report_handles_undefined_distances_without_test_winner():
    render = extract_function("run_v7_goal_loop.py", "format_report")
    report = render([dict(label="best", epoch=3, scar_dice=0., scar_hd95=None,
                          aar_dice=0., aar_hd95=None, myo_dice=80.,
                          scar_hd95_defined=0, aar_hd95_defined=0, cases=76)])
    assert "undefined" in report
    assert "0/76" in report
    assert "Best Overall" not in report
    assert "Off / Std" not in report


def test_threshold_monitor_does_not_fall_back_to_author_compatibility_scores():
    verify = extract_function("monitor_and_evaluate_all.py", "verify_criteria")
    verify.__globals__.update(TARGET_SCAR_DICE=.7364, TARGET_SCAR_HD95=3.66,
                             TARGET_EDEMA_INC_DICE=.7544, TARGET_EDEMA_INC_HD95=3.89,
                             TARGET_MYO_DICE=.868)
    metrics = {r: dict(mean_dice=0., mean_hd95_voxel=10., mean_official_dice=1.,
                       mean_official_hd95_voxel=0., status_counts={})
               for r in ("scar", "edema_inclusive", "myocardial_ring")}
    passed, detail = verify(metrics)
    assert not passed
    assert detail["scar_dice"][0] == 0
    assert detail["scar_hd95"][0] == 10


def test_parallel_report_preserves_undefined_values():
    render = extract_function("run_parallel_train_and_eval.py", "format_results_markdown")
    report = render([dict(model="M3", label="best", epoch=1, scar_dice=None,
                          scar_hd95=None, edema_inc_dice=None, edema_inc_hd95=None,
                          myo_dice=80., empty_count=76)])
    assert "undefined" in report
    assert "not a matched-evaluator comparison" in report
