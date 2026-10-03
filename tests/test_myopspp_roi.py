"""ROI exports must keep their own geometry, splits and already normalized values."""
import hashlib
import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pytest
import torch

PROFILES = [("myopspp_roi128_76", "roi128mm_76cases", 128, 1.0, [48, 13, 15]),
            ("myopspp_roi160_80", "roi160mm_to128_80cases", 160, 1.25, [51, 13, 16])]
EXCLUDED = {"Case2013", "Case2017", "Case2018", "Case2031"}


@pytest.fixture(params=PROFILES, ids=[p[0] for p in PROFILES])
def roi_export(tmp_path, request):
    from training.dataset.benchmark_profiles import fixed_myopspp_splits
    profile, export_id, crop, spacing, counts = request.param
    root = tmp_path / "export"
    root.mkdir()
    splits = {s: [c for c in names if crop == 160 or c not in EXCLUDED]
              for s, names in fixed_myopspp_splits().items()}
    ids = sorted(c for names in splits.values() for c in names)
    (root / "splits.json").write_text(json.dumps({"patients": splits}))
    (root / "dataset_manifest.json").write_text(json.dumps(dict(
        dataset_id=export_id, patients=ids, patient_count=len(ids), crop_mm=crop,
        model_hw=[128, 128], output_inplane_spacing_mm=spacing,
        excluded_cases=sorted(EXCLUDED) if crop == 128 else [],
        localization="oracle GT bbox; NOT automatic inference",
        normalization="p1/p99 after crop/resize; [0,1]")))
    image = np.linspace(0, 1, 128 * 128, dtype=np.float32).reshape(128, 128, 1)
    target = np.zeros(image.shape, dtype=np.uint16)
    target[10:40, 10:40] = 200
    target[15:20, 15:20] = 1220
    target[25:30, 25:30] = 2221
    target[1, 1], target[2, 2] = 500, 600
    affine = np.diag([-spacing, -spacing, 7, 1])
    for case in ids:
        center = "CenterB" if case.startswith("Case2") else "CenterC"
        directory = root / "cases" / center / case
        directory.mkdir(parents=True)
        for suffix in ("C0", "LGE", "T2", "gd"):
            volume = nib.Nifti1Image(target if suffix == "gd" else image, affine)
            volume.header.set_xyzt_units("mm")
            nib.save(volume, directory / f"{case}_{suffix}.nii.gz")
        metadata = dict(case=case, center=center, split=next(s for s in splits if case in splits[s]),
                        crop_mm=crop, output_shape=list(image.shape), output_spacing_mm=[spacing, spacing, 7],
                        output_affine=affine.tolist(), native_spacing_mm=[1.5, 1.5, 7],
                        native_affine=np.diag([-1.5, -1.5, 7, 1]).tolist(),
                        localization="raw GT >0, all slices, bbox midpoint; oracle")
        (directory / "metadata.json").write_text(json.dumps(metadata))
    refresh_checksums(root)
    return root, profile, counts, image, target


def refresh_checksums(root):
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            records.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}\n")
    (root / "checksums.sha256").write_text("".join(records))


def test_adapter_preserves_export_values_and_locks_roi_protocol(roi_export, tmp_path):
    from preprocessing.myopspp_roi import preprocess_myopspp_roi
    from preprocessing.verify import verify_dataset
    from training.dataset.data_contract import lock_benchmark_data
    from training.metrics.surface_distance import dataset_rows
    root, profile, counts, image, raw = roi_export
    cache = tmp_path / "cache"
    info = preprocess_myopspp_roi(root, cache, profile, show_progress=False)
    assert [len(info["patients"][s]) for s in ("train", "val", "test_vol")] == counts
    case = info["patients"]["test_vol"][0]
    with h5py.File(cache / f"LGE/test_vol_h5/{case}.npy.h5") as volume:
        np.testing.assert_array_equal(volume["image"][:], image)
        label = volume["label"][:]
        assert label[1, 1, 0] == label[2, 2, 0] == 0
        assert label[11, 11, 0] == 1 and label[16, 16, 0] == 2 and label[26, 26, 0] == 3
    sample_name = (cache / "lists/train.txt").read_text().splitlines()[0]
    with np.load(cache / f"bSSFP/train_npz/{sample_name}.npz") as volume:
        np.testing.assert_array_equal(volume["image"], image[:, :, 0])
    report = verify_dataset(cache, cache / "lists", "canonical", profile)
    assert report["benchmark_protocol"]["evaluation_grid"] == "preprocessed_roi"
    xy = 1 if profile.endswith("76") else 1.25
    target = np.zeros((5, 5, 1), np.uint8)
    pred = target.copy()
    target[2, 2, 0], pred[3, 2, 0] = 3, 3
    rows = dataset_rows(pred, target, "test", dataset_id=profile, spacing=[xy, xy, 7])
    assert next(r for r in rows if r["region"] == "scar")["hd95_mm"] == pytest.approx(xy)
    with pytest.raises(ValueError):
        lock_benchmark_data(cache, cache / "lists", "canonical", "myopspp_bc80")
    before = report["benchmark_data"]
    with h5py.File(cache / f"LGE/test_vol_h5/{case}.npy.h5", "r+") as volume:
        volume["image"][0, 0, 0] = 0.9
    assert lock_benchmark_data(cache, cache / "lists", "canonical", profile) != before
    with pytest.raises(FileExistsError):
        preprocess_myopspp_roi(root, cache, profile, show_progress=False)


@pytest.mark.parametrize("problem", ["checksum", "geometry", "intensity", "split", "profile"])
def test_bad_export_never_publishes_cache(roi_export, tmp_path, problem):
    from preprocessing.myopspp_roi import preprocess_myopspp_roi
    root, profile, _, _, _ = roi_export
    path = root / "cases/CenterB/Case2001/Case2001_LGE.nii.gz"
    if problem in ("checksum", "geometry", "intensity"):
        volume = nib.load(path)
        image, affine = volume.get_fdata().astype(np.float32), volume.affine.copy()
        if problem == "geometry":
            affine[0, 3] = 10
        else:
            image[0, 0, 0] = 2
        nib.save(nib.Nifti1Image(image, affine, volume.header), path)
    elif problem == "split":
        path = root / "splits.json"
        record = json.loads(path.read_text())
        record["patients"]["train"].pop()
        path.write_text(json.dumps(record))
    else:
        profile = "myopspp_roi160_80" if profile.endswith("76") else "myopspp_roi128_76"
    if problem != "checksum":
        refresh_checksums(root)
    cache = tmp_path / "bad_cache"
    with pytest.raises(ValueError):
        preprocess_myopspp_roi(root, cache, profile, show_progress=False)
    assert not cache.exists()


@pytest.mark.parametrize("profile,export,crop,xy,counts", PROFILES)
def test_profile_selects_separate_paths_and_preserves_380_defaults(profile, export, crop, xy, counts):
    from training.train import parse_args
    args, _ = parse_args(["--dataset", profile])
    assert args.img_size == 128 and args.label_order == "canonical"
    assert profile in args.data_root and profile in args.list_dir and profile in args.run_root
    assert args.pathology_reduction == "sample"
    old, _ = parse_args([])
    assert old.dataset_id == "myops380" and old.label_order == "legacy"
    assert old.pathology_reduction == "positive"


def test_sample_pathology_reduction_accumulates_with_uneven_positive_counts():
    from training.loss.losses import SegmentationLoss
    torch.manual_seed(1234)
    logits = torch.randn(4, 4, 4, 4, requires_grad=True)
    target = torch.zeros(4, 4, 4, dtype=torch.long)
    target[0, 1:3, 1:3] = 2
    target[2, 1:3, 1:3] = 3
    loss = SegmentationLoss(aar_weight=0.2, scar_weight=0.2, pathology_reduction="sample")
    full = loss(logits, target)
    grad = torch.autograd.grad(full["loss"], logits)[0]
    pieces = [loss(logits[i:i+1], target[i:i+1])["loss"] for i in range(4)]
    accumulated = sum(pieces) / 4
    np.testing.assert_allclose(accumulated.detach(), full["loss"].detach(), rtol=1e-6)
    torch.testing.assert_close(torch.autograd.grad(accumulated, logits)[0], grad)
    legacy = SegmentationLoss(aar_weight=0.2, scar_weight=0.2)(logits, target)
    torch.testing.assert_close(full["aar_loss"], legacy["aar_loss"] * 0.5)
    torch.testing.assert_close(full["scar_loss"], legacy["scar_loss"] * 0.25)


def test_resume_rejects_changed_reduction_and_accepts_legacy_missing_setting():
    from argparse import Namespace
    from training.trainer.trainer import _validate_resume_loss_config
    _validate_resume_loss_config({}, Namespace(pathology_reduction="positive"))
    with pytest.raises(ValueError, match="pathology_reduction"):
        _validate_resume_loss_config({}, Namespace(pathology_reduction="sample"))


@pytest.mark.parametrize("problem", ["negative_spacing", "nan_spacing", "singular_affine", "spacing_affine_mismatch"])
def test_invalid_native_provenance_is_rejected(roi_export, tmp_path, problem):
    from preprocessing.myopspp_roi import preprocess_myopspp_roi
    root, profile, _, _, _ = roi_export
    path = root / "cases/CenterB/Case2001/metadata.json"
    record = json.loads(path.read_text())
    if problem == "negative_spacing":
        record["native_spacing_mm"][0] = -1
    elif problem == "nan_spacing":
        record["native_spacing_mm"][1] = float("nan")
    elif problem == "singular_affine":
        record["native_affine"] = np.zeros((4, 4)).tolist()
    else:
        record["native_affine"][0][0] = -2
    path.write_text(json.dumps(record))
    refresh_checksums(root)
    cache = tmp_path / "bad_cache"
    with pytest.raises(ValueError, match="native"):
        preprocess_myopspp_roi(root, cache, profile, show_progress=False)
    assert not cache.exists()


def test_warm_start_transfers_hierarchical_base_classifier_to_standard_head(tmp_path):
    from training.models import CMSPANet, get_testing
    from training.train import _load_initial_weights
    from training.dataset.data_contract import CLASS_NAMES
    source = CMSPANet(get_testing(), img_size=32, ablation="M2-MAX-V6")
    destination = CMSPANet(get_testing(), img_size=32, ablation="M2-MAX-V4")
    with torch.no_grad():
        source.segmentation_head.base_head.weight.fill_(0.125)
        source.segmentation_head.base_head.bias.fill_(0.25)
    path = tmp_path / "init.pth"
    torch.save(dict(format_version=1, class_names=CLASS_NAMES, model=source.state_dict()), path)
    _load_initial_weights(destination, path)
    torch.testing.assert_close(destination.segmentation_head[0].weight, source.segmentation_head.base_head.weight)
    torch.testing.assert_close(destination.segmentation_head[0].bias, source.segmentation_head.base_head.bias)


def test_roi_train_resume_evaluate_and_objective_guard(roi_export, tmp_path):
    from preprocessing.myopspp_roi import preprocess_myopspp_roi
    from training.train import main as train
    from training.evaluate import main as evaluate
    from training.trainer.trainer import load_checkpoint
    root, profile, counts, _, _ = roi_export
    cache, run = tmp_path / "cache", tmp_path / "run"
    preprocess_myopspp_roi(root, cache, profile, show_progress=False)
    argv = ["--dataset", profile, "--config", "testing", "--data-root", str(cache),
            "--list-dir", str(cache / "lists"), "--output-dir", str(run),
            "--epochs", "2", "--epochs-per-run", "1", "--batch-size", "32", "--accum-steps", "2",
            "--ablation", "M2-MAX-V7", "--aar-weight", "0.2", "--scar-weight", "0.2",
            "--wall-weight", "0.2", "--inclusion-weight", "0.1",
            "--num-workers", "0", "--cpu-threads", "2", "--device", "cpu", "--amp", "none", "--no-tensorboard"]
    first = train(argv)
    last = run / "checkpoints/last.pth"
    with pytest.raises(ValueError, match="pathology_reduction"):
        train(argv + ["--resume", str(last), "--pathology-reduction", "positive"])
    second = train(argv + ["--resume", str(last)])
    assert second["global_step"] > first["global_step"]
    assert load_checkpoint(last)["epoch"] == 1
    assert load_checkpoint(last)["benchmark_protocol"]["evaluation_grid"] == "preprocessed_roi"
    summary = evaluate(["--checkpoint", str(run / "checkpoints/best.pth"), "--data-root", str(cache),
                        "--device", "cpu", "--amp", "none", "--cpu-threads", "2", "--no-save-predictions"])
    assert summary["dataset_id"] == profile and summary["case_count"] == counts[2]
    assert "mean_hd95_mm" in summary["scar"]
    assert "without native restoration" in summary["hd95_note"]


def test_run_all_can_package_train_and_evaluate_roi_export(roi_export, tmp_path):
    from run_all import main
    root, profile, counts, _, _ = roi_export
    run = main(["--dataset", profile, "--config", "testing", "--raw-root", str(root),
                "--data-root", str(tmp_path / "cache"), "--run-root", str(tmp_path / "runs"), "--run-id", "smoke",
                "--epochs", "1", "--batch-size", "32", "--accum-steps", "1", "--device", "cpu",
                "--amp", "none", "--num-workers", "0", "--cpu-threads", "2", "--no-tensorboard"])
    from training.run_layout import RunLayout
    summary = json.loads((RunLayout.from_checkpoint(run / "checkpoints/best.pth").evaluation("test_vol") / "metrics.json").read_text())
    assert summary["dataset_id"] == profile and summary["case_count"] == counts[2]
