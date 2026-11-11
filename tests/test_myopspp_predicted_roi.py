"""Predicted ROI export -> locked 64/16 cache -> train/resume/evaluate."""
import hashlib
import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pytest
import yaml

from preprocessing.myopspp_roi import preprocess_myopspp_roi, verified_export_files
from training.dataset.benchmark_profiles import (
    DATASET_IDS, MYOPSPP_DATASET as PROFILE, MYOPSPP_EXPORT_ID, MYOPSPP_LOCALIZATION, fixed_myopspp_splits,
)
from training.dataset.data_contract import CLASS_NAMES, ensure_patient_splits, lock_benchmark_data
from training.dataset.myops_dataset import MyopsDataset
from training.train import parse_args


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_checksums(root):
    records = [f"{sha(p)}  {p.relative_to(root).as_posix()}\n"
               for p in sorted(root.rglob("*")) if p.is_file() and p.name != "checksums.sha256"]
    (root / "checksums.sha256").write_text("".join(records), encoding="utf-8")


@pytest.fixture
def roi_export(tmp_path):
    root = tmp_path / "export"
    splits = fixed_myopspp_splits()
    ids = sorted(sum(splits.values(), []))
    write_json(root / "splits.json", {"patients": splits})
    models = {"fold0": {"train": splits["train"][::2]},
              "fold1": {"train": splits["train"][1::2]}, "final": {"train": splits["train"]}}
    protocol = root / "reproducibility/protocol.json"
    write_json(protocol, {"models": models})
    model_hashes = {k: hashlib.sha256(k.encode()).hexdigest() for k in models}
    predictions = {}
    for case in ids:
        model = ("fold1" if case in models["fold0"]["train"] else "fold0") if case in splits["train"] else "final"
        predictions[case] = {"case": case, "model": model, "model_sha256": model_hashes[model],
                             "training_patients": models[model]["train"], "center_hw_1mm": [64, 64]}
    predicted = root / "reproducibility/predicted_centers.json"
    write_json(predicted, predictions)
    lock = dict(centers_sha256=sha(predicted), protocol_sha256=sha(protocol), models_sha256=model_hashes)
    lock_path = root / "reproducibility/prediction_lock.json"
    write_json(lock_path, lock)
    write_json(root / "dataset_manifest.json", dict(
        dataset_id=MYOPSPP_EXPORT_ID, patients=ids, patient_count=80, crop_mm=128,
        model_hw=[128, 128], output_inplane_spacing_mm=1.0, excluded_cases=[],
        labels=[0, 200, 500, 600, 1220, 2221], localization="predicted U-Net2D; synthetic test folds",
        normalization="p1/p99 after crop/resize; [0,1]", prediction_lock=lock, protocol_sha256=sha(protocol)))
    image = np.linspace(0, 1, 128 * 128, dtype=np.float32).reshape(128, 128, 1)
    raw = np.zeros(image.shape, dtype=np.uint16)
    raw[10:40, 10:40] = 200
    raw[15:20, 15:20], raw[25:30, 25:30] = 1220, 2221
    raw[1, 1], raw[2, 2] = 500, 600
    affine = np.diag([-1., -1., 7., 1.])
    for case in ids:
        center = "CenterB" if case.startswith("Case2") else "CenterC"
        folder = root / "cases" / center / case
        folder.mkdir(parents=True)
        for i, suffix in enumerate(("C0", "LGE", "T2", "gd")):
            volume = nib.Nifti1Image(raw if suffix == "gd" else image * (1 - .1 * i), affine)
            volume.header.set_xyzt_units("mm")
            nib.save(volume, folder / f"{case}_{suffix}.nii.gz")
        write_json(folder / "metadata.json", dict(
            case=case, center=center, split="train" if case in splits["train"] else "test_vol",
            crop_mm=128, output_shape=list(raw.shape), output_spacing_mm=[1, 1, 7],
            output_affine=affine.tolist(), native_affine=affine.tolist(), native_spacing_mm=[1, 1, 7],
            localization=MYOPSPP_LOCALIZATION, localizer=predictions[case], prediction_lock_sha256=sha(lock_path)))
    refresh_checksums(root)
    return root, image, raw


@pytest.fixture
def cache(roi_export, tmp_path):
    root = tmp_path / "cache"
    preprocess_myopspp_roi(roi_export[0], root, show_progress=False)
    return root


def test_cache_preserves_values_labels_geometry_and_source(roi_export, cache, tmp_path):
    from preprocessing.verify import verify_dataset
    source, image, raw = roi_export
    verified_export_files(source.resolve())
    info = json.loads((cache / "dataset_metadata.json").read_text())
    assert info["patients"] == fixed_myopspp_splits()
    assert info["sample_counts"] == {"train": 64, "test_vol": 16}
    assert info["no_val"] and info["localization"] == MYOPSPP_LOCALIZATION
    first = fixed_myopspp_splits()["train"][0]
    for i, modality in enumerate(("bSSFP", "LGE", "T2w")):
        with np.load(cache / modality / "train_npz" / f"{first}_slice000.npz") as data:
            np.testing.assert_array_equal(data["image"], image[:, :, 0] * (1 - .1 * i))
            labels = data["label"]
            assert labels.dtype == np.uint8
            assert labels[1, 1] == labels[2, 2] == 0  # LV/RV, never edema or scar
            assert (labels[11, 11], labels[16, 16], labels[26, 26]) == (1, 2, 3)
            assert (labels == 2).sum() == (raw == 1220).sum()
            assert (labels == 3).sum() == (raw == 2221).sum()
    sample = MyopsDataset(*[cache / m / "test_vol_h5" for m in ("bSSFP", "LGE", "T2w")],
                          cache / "lists", "test_vol", label_order="canonical")[0]
    np.testing.assert_allclose(sample["spacing"], [1, 1, 7])
    assert sample["has_geometry"] and sample["label"].shape == (128, 128, 1)
    report = verify_dataset(cache, cache / "lists", "canonical", PROFILE, no_val=True)
    assert report["patient_counts"] == {"train": 64, "test_vol": 16}
    assert report["benchmark_protocol"]["localization"] == MYOPSPP_LOCALIZATION
    assert "selection_metric" not in report["benchmark_protocol"]
    for seed in (1, 42):
        saved = ensure_patient_splits(cache / "lists", seed=seed, no_val=True,
                                      dataset_id=PROFILE, output_dir=tmp_path / f"seed{seed}")
        for name in ("train.txt", "test_vol.txt"):
            assert (saved / name).read_bytes() == (cache / "lists" / name).read_bytes()
    with pytest.raises(FileExistsError):
        preprocess_myopspp_roi(source, cache)
    with pytest.raises(ValueError, match="outside"):
        preprocess_myopspp_roi(source, source / "cache")
    assert not (source / "cache").exists()


@pytest.mark.parametrize("problem", ["checksum", "oracle", "split", "label", "nan", "geometry", "localizer", "missing"])
def test_bad_export_never_publishes_cache(roi_export, tmp_path, problem):
    root = roi_export[0]
    folder = root / "cases/CenterB/Case2001"
    if problem in ("checksum", "oracle"):
        path = root / "dataset_manifest.json"
        record = json.loads(path.read_text())
        record["localization"] = "oracle GT bbox"
        write_json(path, record)
    elif problem == "split":
        path = root / "splits.json"
        record = json.loads(path.read_text())
        record["patients"]["val"] = [record["patients"]["train"].pop()]
        write_json(path, record)
    elif problem == "localizer":
        path = folder / "metadata.json"
        record = json.loads(path.read_text())
        record["localizer"]["training_patients"].append("Case2001")
        write_json(path, record)
    elif problem == "missing":
        (folder / "Case2001_T2.nii.gz").unlink()
    else:
        path = folder / ("Case2001_gd.nii.gz" if problem == "label" else "Case2001_LGE.nii.gz")
        original = nib.load(path)
        data = original.get_fdata(dtype=np.float32)
        affine = original.affine.copy()
        if problem == "geometry":
            affine[0, 3] = 5
        else:
            data[0, 0, 0] = 999 if problem == "label" else np.nan
        changed = nib.Nifti1Image(data, affine)
        changed.header.set_xyzt_units("mm")
        nib.save(changed, path)
    if problem != "checksum":
        refresh_checksums(root)
    destination = tmp_path / "bad_cache"
    with pytest.raises((ValueError, FileNotFoundError)):
        preprocess_myopspp_roi(root, destination, show_progress=False)
    assert not destination.exists()
    assert not list(tmp_path.glob("bad_cache.building-*"))


def test_cache_identity_split_and_bytes_are_locked(cache):
    before = lock_benchmark_data(cache, cache / "lists", "canonical", PROFILE, no_val=True)
    for label in ("legacy", "auto"):
        with pytest.raises(ValueError, match="canonical"):
            lock_benchmark_data(cache, cache / "lists", label, PROFILE, no_val=True)
    with pytest.raises(ValueError, match="no-val"):
        lock_benchmark_data(cache, cache / "lists", "canonical", PROFILE)
    with pytest.raises(ValueError, match="MyoPS380"):
        lock_benchmark_data(cache, cache / "lists", "canonical", "myops380", no_val=True)
    path = next((cache / "LGE/test_vol_h5").glob("*.h5"))
    with h5py.File(path, "r+") as f:
        f["image"][0, 0, 0] = .5
    assert lock_benchmark_data(cache, cache / "lists", "canonical", PROFILE, no_val=True) != before
    (cache / "lists/val.txt").write_text("Case2001_slice000\n")
    with pytest.raises(ValueError, match="validation"):
        lock_benchmark_data(cache, cache / "lists", "canonical", PROFILE, no_val=True)


def test_profiles_and_no_val_keep_myops380_defaults(tmp_path):
    assert DATASET_IDS == ("myops380", PROFILE)
    old, _ = parse_args([])
    new, _ = parse_args(["--dataset", PROFILE])
    assert old.no_val and old.label_order == "legacy" and "MyoPS380" in old.data_root
    assert new.no_val and new.label_order == "canonical" and PROFILE in new.data_root
    assert old.pathology_reduction == "positive" and new.pathology_reduction == "positive"
    for flags in (["--with-val"], ["--label-order", "legacy"], ["--patience", "2"]):
        with pytest.raises(SystemExit):
            parse_args(["--dataset", PROFILE, *flags])
    for retired in ("myopspp_bc80", "myopspp_roi128_76", "myopspp_roi128_80", "myopspp_roi160_80"):
        with pytest.raises(SystemExit):
            parse_args(["--dataset", retired])
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"data": {"dataset_id": PROFILE, "data_root": str(tmp_path / "cache")}}))
    relocated, _ = parse_args(["--config", str(config)])
    assert Path(relocated.list_dir) == tmp_path / "cache/lists"
    original, _ = parse_args(["--config", str(config), "--dataset", "myops380"])
    assert original.dataset_id == "myops380" and original.label_order == "legacy"


def test_mm_metrics_keep_myops380_voxel_semantics():
    from training.metrics.surface_distance import dataset_rows
    target = np.zeros((5, 5, 3), np.uint8)
    pred = target.copy()
    target[2, 2, 0], pred[2, 2, 1] = 3, 3
    roi = dataset_rows(pred, target, "case", dataset_id=PROFILE, spacing=[1, 1, 7])
    native = dataset_rows(pred, target, "case", dataset_id="myops380")
    assert next(r for r in roi if r["region"] == "scar")["hd95_mm"] == pytest.approx(7)
    assert next(r for r in native if r["region"] == "scar")["hd95_voxel"] == pytest.approx(1)


@pytest.mark.parametrize("variant", ["m2_plus", "lightweight_m2_plus", "paper_formula_v1"])
def test_train_resume_final_checkpoint_and_evaluate(cache, tmp_path, variant):
    from training.train import main as train
    from training.evaluate import main as evaluate
    from training.trainer.trainer import load_checkpoint
    config = yaml.safe_load(Path("training/config/models/testing.yaml").read_text())
    config["train"]["no_val"] = True
    if variant == "lightweight_m2_plus":
        config["model"].update(architecture=variant, ablation="LIGHTWEIGHT_M2_PLUS", sspanet_gate="optimized")
    elif variant == "paper_formula_v1":
        config["model"]["sspanet_gate"] = variant
    path = tmp_path / "testing.yaml"
    path.write_text(yaml.safe_dump(config))
    run = tmp_path / "run"
    argv = ["--dataset", PROFILE, "--config", str(path), "--data-root", str(cache),
            "--list-dir", str(cache / "lists"), "--output-dir", str(run), "--epochs", "2", "--epochs-per-run", "1",
            "--batch-size", "32", "--accum-steps", "1", "--device", "cpu", "--amp", "none",
            "--num-workers", "0", "--cpu-threads", "2", "--no-tensorboard", "--aar-weight", "0.2", "--scar-weight", "0.2"]
    first = train(argv)
    last = run / "checkpoints/testingsmall_seed1234.pt"
    checkpoint = load_checkpoint(last)
    if variant == "paper_formula_v1":
        assert checkpoint["model_config"]["sspanet_gate"] == variant
    assert checkpoint["class_names"] == CLASS_NAMES
    assert checkpoint["training_protocol"]["train_patients"] == 64
    assert not checkpoint["training_complete"]
    assert set(checkpoint["split_hashes"]) == {"train", "test_vol"}
    assert not (last.parent / "best.pt").exists()
    eval_args = ["--checkpoint", str(last), "--data-root", str(cache), "--device", "cpu", "--amp", "none",
                 "--cpu-threads", "2", "--no-save-predictions"]
    with pytest.raises(ValueError, match="Incomplete training budget"):
        evaluate(eval_args)
    with pytest.raises(ValueError, match="scar_weight"):
        train(argv + ["--resume", str(last), "--scar-weight", "0.3"])
    second = train(argv + ["--resume", str(last)])
    assert second["global_step"] > first["global_step"] and second["training_complete"]
    summary = evaluate(eval_args)
    assert summary["dataset_id"] == PROFILE and summary["case_count"] == 16
    assert summary["benchmark_protocol"]["localization"] == MYOPSPP_LOCALIZATION
    assert "without native restoration" in summary["hd95_note"]
    assert "mean_hd95_mm" in summary["scar"]


def test_run_all_packages_trains_and_evaluates(roi_export, tmp_path):
    from run_all import main
    from training.run_layout import RunLayout
    run = main(["--dataset", PROFILE, "--config", "testing", "--no-val", "--raw-root", str(roi_export[0]),
                "--data-root", str(tmp_path / "cache"), "--run-root", str(tmp_path / "runs"), "--run-id", "smoke",
                "--epochs", "1", "--batch-size", "32", "--accum-steps", "1", "--device", "cpu", "--amp", "none",
                "--num-workers", "0", "--cpu-threads", "2", "--no-tensorboard"])
    last = run / "checkpoints/testingsmall_seed1234.pt"
    summary = json.loads((RunLayout.from_checkpoint(last).evaluation("test_vol") / "metrics.json").read_text())
    assert summary["dataset_id"] == PROFILE and summary["case_count"] == 16
    assert not (last.parent / "best.pt").exists()
