"""Physical surface distance and Hausdorff (HD95) metrics."""
from __future__ import annotations

import numpy as np
from collections import Counter
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


BENCHMARK_PROTOCOL = {
    "id": "myops380_voxel_v1",
    "distance_unit": "voxel",
    "voxelspacing": None,
    "axis_order": "HWD",
    "regions": {"normal_myocardium": [1], "edema": [2], "scar": [3],
                "edema_inclusive": [2, 3], "myocardial_ring": [1, 2, 3]},
    "primary_empty_policy": "both_empty: excluded; one_empty: Dice/IoU=0, distances undefined",
    "aggregation": "equal patient weight within each region; exclude undefined values and report counts",
    "selection_metric": "avg_pathology_dice",
    "pathology_regions": ["scar", "edema"],
    "pathology_aggregation": "arithmetic mean of scar and exclusive-edema patient means; both required",
    "official_reference": "I_MMSeg@90f46c4eb72924509895fcda6bc6a3b8c3316e66/utils.py:171-182",
    "official_empty_policy": "if either empty: Dice=1 when pred<200 and target<=200, else 0; HD95=0",
}


def protocol_for_dataset(dataset_id="myops380"):
    from training.dataset.benchmark_profiles import ROI_PROFILES
    if dataset_id == "myops380":
        return BENCHMARK_PROTOCOL
    if dataset_id in ROI_PROFILES:
        profile = ROI_PROFILES[dataset_id]
        return {k: v for k, v in dict(BENCHMARK_PROTOCOL,
                id=f"{dataset_id}_roi_mm_v1", distance_unit="mm",
                voxelspacing="preprocessed ROI NIfTI spacing in mm, orthogonal grid",
                evaluation_grid="preprocessed_roi", localization="oracle_gt_bbox",
                crop_mm=profile["crop_mm"], model_hw=[128, 128],
                spacing_xy_mm=profile["spacing_xy"]).items() if not k.startswith("official_")}
    if dataset_id != "myopspp_bc80":
        raise ValueError(f"Unknown metric dataset profile: {dataset_id}")
    return {k: v for k, v in dict(BENCHMARK_PROTOCOL,
            id="myopspp_bc80_mm_v1", distance_unit="mm",
            voxelspacing="native NIfTI spacing in mm, orthogonal grid").items() if not k.startswith("official_")}


def dataset_rows(prediction, target, case, compute_distance=True, dataset_id="myops380", spacing=None):
    if dataset_id == "myops380":
        return benchmark_rows(prediction, target, case, compute_distance=compute_distance)
    protocol = protocol_for_dataset(dataset_id)
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("MyoPS++ metrics require positive mm spacing of the evaluation grid")
    # Reuse canonical validation and overlap definitions, without importing the
    # MyoPS380 author-specific empty-mask metrics into the new benchmark.
    rows = benchmark_rows(prediction, target, case, compute_distance=False)
    for row in rows:
        ids = protocol["regions"][row["region"]]
        physical = binary_metrics(np.isin(prediction, ids), np.isin(target, ids), spacing=spacing,
                                  compute_distance=compute_distance)
        for key in ("hd95_voxel", "asd_voxel", "official_dice", "official_hd95_voxel"):
            row.pop(key)
        row.update(physical, hd95_unit="mm", hd95_mm=physical["hd95"], asd_mm=physical["asd"])
    return rows


def summarize_dataset_rows(rows, dataset_id="myops380"):
    if dataset_id == "myops380":
        return summarize_rows(rows)
    protocol = protocol_for_dataset(dataset_id)
    identities = [(r["case"], r["region"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate case/region rows would bias patient means")
    for case in {r["case"] for r in rows}:
        if {r["region"] for r in rows if r["case"] == case} != set(protocol["regions"]):
            raise ValueError(f"Incomplete benchmark regions for {case}")
    summary = {}
    for name in protocol["regions"]:
        subset = [r for r in rows if r["region"] == name]
        region = {"cases": len(subset), "status_counts": dict(Counter(r["status"] for r in subset))}
        for metric in ("dice", "iou", "precision", "recall", "hd95_mm", "asd_mm"):
            values = [r[metric] for r in subset if r[metric] is not None]
            if not all(np.isfinite(v) for v in values):
                raise ValueError(f"Non-finite {metric}")
            region[f"mean_{metric}"] = float(np.mean(values)) if values else None
            region[f"{metric}_defined_cases"] = len(values)
            region[f"{metric}_undefined_cases"] = len(subset) - len(values)
        summary[name] = region
    for metric in ("dice", "iou", "hd95_mm"):
        values = [summary[name][f"mean_{metric}"] for name in ("scar", "edema")]
        summary[f"avg_pathology_{metric}"] = float(np.mean(values)) if all(v is not None for v in values) else None
    return summary


def benchmark_rows(prediction, target, case, compute_distance=True):
    """Two evaluators on identical canonical HWD predictions; never mutate masks."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.ndim != 3 or prediction.shape != target.shape or not prediction.size:
        raise ValueError("Benchmark requires matching nonempty HWD volumes")
    if not np.isin(prediction, [0, 1, 2, 3]).all() or not np.isin(target, [0, 1, 2, 3]).all():
        raise ValueError("Benchmark requires canonical integer labels 0..3")
    rows = []
    for region, ids in BENCHMARK_PROTOCOL["regions"].items():
        pred, truth = np.isin(prediction, ids), np.isin(target, ids)
        metrics = binary_metrics(pred, truth, compute_distance=compute_distance)
        n_pred, n_true = int(pred.sum()), int(truth.sum())
        overlap = int(np.logical_and(pred, truth).sum())
        metrics["precision"] = overlap / n_pred if n_pred else None
        metrics["recall"] = overlap / n_true if n_true else None
        official_dice, official_hd95 = metrics["dice"], metrics["hd95"]
        if not n_pred or not n_true:
            official_dice = float(n_pred < 200 and n_true <= 200)
            official_hd95 = 0.0 if compute_distance else None
        rows.append(dict(case=case, region=region, **metrics,
                         prediction_voxels=n_pred, target_voxels=n_true,
                         hd95_unit="voxel", hd95_voxel=metrics["hd95"],
                         asd_voxel=metrics["asd"],
                         official_dice=official_dice, official_hd95_voxel=official_hd95))
    return rows


def summarize_rows(rows):
    """Case means, defined counts and separately named official reproduction metrics."""
    summary = {}
    identities = [(r["case"], r["region"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate case/region rows would bias patient means")
    expected = set(BENCHMARK_PROTOCOL["regions"])
    for case in {r["case"] for r in rows}:
        if {r["region"] for r in rows if r["case"] == case} != expected:
            raise ValueError(f"Incomplete benchmark regions for {case}")
    for name in BENCHMARK_PROTOCOL["regions"]:
        subset = [r for r in rows if r["region"] == name]
        region = {"cases": len(subset), "status_counts": dict(Counter(r["status"] for r in subset))}
        for metric in ("dice", "iou", "precision", "recall", "hd95_voxel", "asd_voxel", "official_dice", "official_hd95_voxel"):
            values = [r[metric] for r in subset if r[metric] is not None]
            if not all(np.isfinite(v) for v in values):
                raise ValueError(f"Non-finite {metric}; undefined values must be None")
            region[f"mean_{metric}"] = float(np.mean(values)) if values else None
            region[f"{metric}_defined_cases"] = len(values)
            region[f"{metric}_undefined_cases"] = len(subset) - len(values)
        summary[name] = region
    for metric in ("dice", "iou", "official_dice", "hd95_voxel", "official_hd95_voxel"):
        values = [summary[name][f"mean_{metric}"] for name in ("scar", "edema")]
        key = (f"official_avg_pathology_{metric.removeprefix('official_')}"
               if metric.startswith("official_") else f"avg_pathology_{metric}")
        summary[key] = float(np.mean(values)) if all(v is not None for v in values) else None
    return summary


def binary_metrics(prediction, target, spacing=None, compute_distance=True, empty_mode="undefined"):
    """Symmetric HD95 and ASD on 1-connected surfaces; spacing follows array axis order.

    Missing surfaces have no finite distance. Report their status separately instead
    of assigning zero distance to a completely missed lesion.
    """
    pred, truth = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and target shapes differ.")
    if spacing is not None:
        spacing = np.asarray(spacing, dtype=float)
        if spacing.shape != (pred.ndim,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError("Spacing must contain one finite positive value per array axis.")
    n_pred, n_true = int(pred.sum()), int(truth.sum())
    if n_pred == 0 and n_true == 0:
        if empty_mode == "defined":
            return {"dice": 1.0, "iou": 1.0, "hd95": 0.0, "asd": 0.0, "status": "both_empty"}
        return {"dice": None, "iou": None, "hd95": None, "asd": None, "status": "both_empty"}
    overlap = np.logical_and(pred, truth).sum()
    result = {
        "dice": float(2 * overlap / (n_pred + n_true)),
        "iou": float(overlap / (n_pred + n_true - overlap)),
        "hd95": None,
        "asd": None,
        "status": "ok",
    }
    if not n_pred or not n_true:
        if empty_mode == "defined":
            result["hd95"] = float("inf")
            result["asd"] = float("inf")
            result["status"] = "one_empty"
        else:
            result["status"] = "prediction_empty" if not n_pred else "target_empty"
        return result
    if not compute_distance:
        return result
    structure = generate_binary_structure(pred.ndim, 1)
    pred_surface = pred ^ binary_erosion(pred, structure=structure, border_value=0)
    true_surface = truth ^ binary_erosion(truth, structure=structure, border_value=0)
    distances = np.concatenate((
        distance_transform_edt(~true_surface, sampling=spacing)[pred_surface],
        distance_transform_edt(~pred_surface, sampling=spacing)[true_surface],
    ))
    result["hd95"] = float(np.percentile(distances, 95))
    result["asd"] = float(np.mean(distances))
    return result


def calculate_metric_percase(pred, gt, spacing=None):
    """Legacy tuple API, honest empty-mask behavior, no input mutation."""
    result = binary_metrics(pred, gt, spacing, empty_mode="undefined")
    distance = result["hd95"]
    if distance is None:
        distance = np.nan if result["status"] == "both_empty" else np.inf
    return (result["dice"] if result["dice"] is not None else np.nan), distance
