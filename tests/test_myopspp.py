"""Separate complete-modality benchmark: raw labels, patient isolation and mm metrics."""
import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pytest

from training.dataset.data_contract import lock_benchmark_data
from training.dataset.myops_dataset import MyopsDataset, RandomGenerator


def write_case(root, case, center, affine=None, missing=None, unit="mm"):
    directory = root / center / case
    directory.mkdir(parents=True, exist_ok=True)
    affine = np.diag([1.5, 1.5, 7.0, 1.0]) if affine is None else affine
    image = np.arange(32 * 32, dtype=np.float32).reshape(32, 32, 1)
    target = np.zeros(image.shape, dtype=np.int16)
    target[8:24, 8:24] = 200
    target[10:14, 10:14] = 1220
    target[16:20, 16:20] = 2221
    target[1, 1] = 500
    target[2, 2] = 600
    for suffix in ("C0", "LGE", "T2", "gd"):
        if suffix == missing:
            continue
        volume = nib.Nifti1Image(target if suffix == "gd" else image, affine)
        volume.header.set_xyzt_units(unit)
        nib.save(volume, directory / f"{case}_{suffix}.nii.gz")


@pytest.fixture
def raw_cohort(tmp_path):
    root = tmp_path / "raw" / "MyoPS_train"
    for center, start, count in (("CenterB", 2001, 35), ("CenterC", 3001, 45)):
        for number in range(start, start + count):
            write_case(root, f"Case{number}", center)
    write_case(root, "Case1001", "CenterA", missing="T2")
    return root


@pytest.fixture
def packaged(raw_cohort, tmp_path):
    from preprocessing.myopspp import preprocess_myopspp
    cache = tmp_path / "cache"
    info = preprocess_myopspp(raw_cohort, cache, show_progress=False)
    return cache, info


def test_raw_mapping_preserves_pathology_and_rejects_unknown_labels():
    from preprocessing.myopspp import canonicalize_myopspp
    raw = np.asarray([0, 200, 500, 600, 1220, 2221], dtype=np.int16)
    np.testing.assert_array_equal(canonicalize_myopspp(raw), [0, 1, 0, 0, 2, 3])
    np.testing.assert_array_equal(raw, [0, 200, 500, 600, 1220, 2221])
    for invalid in ([999], [1220.5], [np.nan]):
        with pytest.raises(ValueError):
            canonicalize_myopspp(np.asarray(invalid))


def test_only_bc_complete_cases_are_joined_by_id(raw_cohort):
    from preprocessing.myopspp import discover_myopspp_cases
    cases = discover_myopspp_cases(raw_cohort.parent)
    assert len(cases) == 80
    assert "Case1001" not in cases
    assert set(cases["Case2001"]) == {"bSSFP", "LGE", "T2w", "label"}
    (raw_cohort / "CenterB/Case2001/Case2001_T2.nii.gz").unlink()
    with pytest.raises(ValueError, match="Case2001"):
        discover_myopspp_cases(raw_cohort)


def test_patient_ids_cannot_move_between_centers(raw_cohort):
    from preprocessing.myopspp import discover_myopspp_cases
    (raw_cohort / "CenterB/Case2001").rename(raw_cohort / "CenterC/Case2001")
    with pytest.raises(ValueError, match="Case2001"):
        discover_myopspp_cases(raw_cohort)


def test_packaging_refuses_existing_cache_and_raw_tree(packaged, raw_cohort):
    from preprocessing.myopspp import preprocess_myopspp
    cache, _ = packaged
    before = lock_benchmark_data(cache, cache / "lists", "canonical", dataset_id="myopspp_bc80")
    with pytest.raises(FileExistsError):
        preprocess_myopspp(raw_cohort, cache, show_progress=False)
    with pytest.raises(ValueError):
        preprocess_myopspp(raw_cohort, raw_cohort / "cache", show_progress=False)
    assert not (raw_cohort / "cache").exists()
    after = lock_benchmark_data(cache, cache / "lists", "canonical", dataset_id="myopspp_bc80")
    assert before == after


def test_packaging_native_geometry_patient_splits_and_small_input(packaged):
    cache, info = packaged
    assert info["dataset_id"] == "myopspp_bc80"
    patients = info["patients"]
    assert [len(patients[k]) for k in ("train", "val", "test_vol")] == [51, 13, 16]
    assert not set(patients["train"]) & set(patients["val"])
    assert not set(patients["test_vol"]) & set(patients["train"] + patients["val"])
    assert [sum(n.startswith("Case2") for n in patients[k]) for k in ("train", "val", "test_vol")] == [22, 6, 7]
    assert [sum(n.startswith("Case3") for n in patients[k]) for k in ("train", "val", "test_vol")] == [29, 7, 9]
    roots = [cache / m / "test_vol_h5" for m in ("bSSFP", "LGE", "T2w")]
    sample = MyopsDataset(*roots, cache / "lists", "test_vol", label_order="canonical")[0]
    assert sample["has_geometry"]
    np.testing.assert_allclose(sample["spacing"], [1.5, 1.5, 7])
    assert sample["label"].shape == (32, 32, 1)
    assert np.count_nonzero(sample["label"] == 2) == 16
    assert np.count_nonzero(sample["label"] == 3) == 16
    assert np.isfinite(sample["image"]).all()
    assert sample["image"].min() >= 0 and sample["image"].max() <= 1
    train_roots = [cache / m / "train_npz" for m in ("bSSFP", "LGE", "T2w")]
    resized = MyopsDataset(*train_roots, cache / "lists", "train",
                           transform=RandomGenerator([128, 128]), label_order="canonical")[0]
    assert resized["image"].shape == (1, 128, 128)


def test_new_lock_rejects_wrong_profile_and_detects_cache_change(packaged):
    cache, _ = packaged
    before = lock_benchmark_data(cache, cache / "lists", "canonical", dataset_id="myopspp_bc80")
    assert before["dataset_id"] == "myopspp_bc80"
    with pytest.raises(ValueError, match="MyoPS380"):
        lock_benchmark_data(cache, cache / "lists", "canonical")
    with pytest.raises(ValueError, match="canonical"):
        lock_benchmark_data(cache, cache / "lists", "legacy", dataset_id="myopspp_bc80")
    path = next((cache / "LGE/test_vol_h5").glob("*.h5"))
    with h5py.File(path, "r+") as f:
        f["image"][0, 0, 0] = 0.5
    after = lock_benchmark_data(cache, cache / "lists", "canonical", dataset_id="myopspp_bc80")
    assert after["cache_sha256"] != before["cache_sha256"]


def test_new_lock_rejects_patient_split_changes(packaged):
    cache, _ = packaged
    path = cache / "lists/test_vol.txt"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError):
        lock_benchmark_data(cache, cache / "lists", "canonical", dataset_id="myopspp_bc80")


@pytest.mark.parametrize("problem", ["unknown_unit", "shear", "unaligned"])
def test_packager_rejects_unusable_geometry_without_publishing_cache(raw_cohort, tmp_path, problem):
    from preprocessing.myopspp import preprocess_myopspp
    affine = np.diag([1.5, 1.5, 7.0, 1.0])
    if problem == "shear":
        affine[0, 1] = 0.5
    write_case(raw_cohort, "Case2001", "CenterB", affine=affine,
               unit="unknown" if problem == "unknown_unit" else "mm")
    if problem == "unaligned":
        path = raw_cohort / "CenterB/Case2001/Case2001_LGE.nii.gz"
        image = nib.load(path)
        affine[0, 3] = 12
        replacement = nib.Nifti1Image(image.get_fdata(), affine, image.header)
        nib.save(replacement, path)
    cache = tmp_path / "bad_cache"
    with pytest.raises(ValueError):
        preprocess_myopspp(raw_cohort, cache, show_progress=False)
    assert not cache.exists()


def test_mm_metric_dispatch_keeps_voxel_benchmark_unchanged():
    from training.metrics.surface_distance import dataset_rows, summarize_dataset_rows
    target = np.zeros((5, 5, 3), dtype=np.uint8)
    prediction = target.copy()
    target[2, 2, 0] = 3
    prediction[2, 2, 1] = 3
    mm = dataset_rows(prediction, target, "case", dataset_id="myopspp_bc80", spacing=[1.5, 1.5, 7])
    scar = next(r for r in mm if r["region"] == "scar")
    assert scar["hd95_mm"] == pytest.approx(7)
    assert scar["asd_mm"] == pytest.approx(7)
    assert "official_dice" not in scar and "hd95_voxel" not in scar
    summary = summarize_dataset_rows(mm, dataset_id="myopspp_bc80")
    assert summary["scar"]["mean_hd95_mm"] == pytest.approx(7)
    old = dataset_rows(prediction, target, "case")
    assert next(r for r in old if r["region"] == "scar")["hd95_voxel"] == pytest.approx(1)
    with pytest.raises(ValueError):
        dataset_rows(prediction, target, "case", dataset_id="myopspp_bc80")


def test_training_profile_changes_dataset_but_retains_default_380():
    from training.train import parse_args
    defaults, _ = parse_args([])
    assert defaults.dataset_id == "myops380"
    args, _ = parse_args(["--dataset", "myopspp_bc80", "--config", "testing"])
    assert args.dataset_id == "myopspp_bc80"
    assert args.label_order == "canonical"
    assert Path(args.data_root).name == "cache"
    assert "myopspp_bc80" in args.data_root
    assert "myopspp_bc80" in args.list_dir
    assert "myopspp_bc80" in args.run_root


def test_real_training_resume_and_evaluation_use_separate_profile(packaged, tmp_path):
    import torch
    from training.train import main as train
    from training.evaluate import main as evaluate
    from training.trainer.trainer import load_checkpoint
    cache, _ = packaged
    run = tmp_path / "run"
    argv = ["--dataset", "myopspp_bc80", "--config", "testing", "--data-root", str(cache),
            "--list-dir", str(cache / "lists"), "--output-dir", str(run),
            "--epochs", "2", "--epochs-per-run", "1", "--batch-size", "64", "--accum-steps", "1",
            "--num-workers", "0", "--cpu-threads", "2", "--device", "cpu", "--amp", "none", "--no-tensorboard"]
    first = train(argv)
    last = run / "checkpoints/last.pt"
    checkpoint = load_checkpoint(last)
    assert checkpoint["epoch"] == 0
    assert checkpoint["benchmark_protocol"]["distance_unit"] == "mm"
    assert checkpoint["benchmark_data"]["dataset_id"] == "myopspp_bc80"
    # An existing M2-Plus checkpoint can carry the previous factory identity and
    # disabled anatomy-loss fields without changing its architecture or objective.
    checkpoint["model_config"]["architecture"] = "cmspa_net"
    checkpoint["model_config"]["ablation"] = "M2-Plus"
    checkpoint["args"].update(wall_weight=0.0, inclusion_weight=0.0)
    torch.save(checkpoint, last)
    del checkpoint
    second = train(argv + ["--resume", str(last)])
    assert second["global_step"] > first["global_step"]
    assert load_checkpoint(last)["epoch"] == 1
    best = run / "checkpoints/best.pt"
    with pytest.raises(ValueError, match="dataset"):
        evaluate(["--checkpoint", str(best), "--data-root", str(cache), "--dataset", "myops380"])
    summary = evaluate(["--checkpoint", str(best), "--data-root", str(cache), "--device", "cpu",
                        "--amp", "none", "--cpu-threads", "2", "--no-save-predictions"])
    assert summary["case_count"] == 16
    assert summary["dataset_id"] == "myopspp_bc80"
    assert "mean_hd95_mm" in summary["scar"]
    assert "mean_hd95_voxel" not in summary["scar"]


def test_run_all_supports_new_dataset_profile():
    from run_all import build_parser
    args = build_parser().parse_args(["--dataset", "myopspp_bc80", "--skip-cache"])
    assert args.dataset == "myopspp_bc80"


def test_m2_plus_weighted_loss_train_resume_evaluate_myopspp(packaged, tmp_path):
    from training.train import main as train
    from training.evaluate import main as evaluate
    from training.trainer.trainer import load_checkpoint
    cache, _ = packaged
    run = tmp_path / "weighted_run"
    argv = ["--dataset", "myopspp_bc80", "--config", "testing", "--ablation", "M2-PLUS",
            "--data-root", str(cache), "--list-dir", str(cache / "lists"), "--output-dir", str(run),
            "--epochs", "2", "--epochs-per-run", "1", "--batch-size", "64", "--accum-steps", "1",
            "--aar-weight", "0.2",
            "--scar-weight", "0.2", "--ce-class-weights", "0.2", "1", "2", "2",
            "--num-workers", "0", "--cpu-threads", "2", "--device", "cpu", "--amp", "none", "--no-tensorboard"]
    first = train(argv)
    last = run / "checkpoints/last.pt"
    checkpoint = load_checkpoint(last)
    assert checkpoint["loss_protocol_version"] == 2
    assert checkpoint["benchmark_data"]["dataset_id"] == "myopspp_bc80"
    # A changed objective must fail before another optimizer update.
    with pytest.raises(ValueError, match="scar_weight"):
        train(argv + ["--resume", str(last), "--scar-weight", "0.3"])
    second = train(argv + ["--resume", str(last)])
    assert second["global_step"] > first["global_step"]
    assert load_checkpoint(last)["epoch"] == 1
    summary = evaluate(["--checkpoint", str(run / "checkpoints/best.pt"), "--data-root", str(cache),
                        "--device", "cpu", "--amp", "none", "--cpu-threads", "2", "--no-save-predictions"])
    assert summary["case_count"] == 16
    assert summary["dataset_id"] == "myopspp_bc80"
    assert "mean_hd95_mm" in summary["scar"]


def test_run_all_trains_and_evaluates_profile_with_model_alias(packaged, tmp_path):
    from run_all import main
    cache, _ = packaged
    run = main(["--dataset", "myopspp_bc80", "--config", "testing", "--skip-cache",
                "--data-root", str(cache), "--run-root", str(tmp_path / "runs"), "--run-id", "smoke",
                "--epochs", "1", "--batch-size", "64", "--accum-steps", "1", "--device", "cpu",
                "--amp", "none", "--num-workers", "0", "--cpu-threads", "2", "--no-tensorboard"])
    assert (run / "checkpoints/best.pt").is_file()
    from training.run_layout import RunLayout
    metrics_path = RunLayout.from_checkpoint(run / "checkpoints/best.pt").evaluation("test_vol") / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    assert metrics["dataset_id"] == "myopspp_bc80"
    assert metrics["case_count"] == 16


@pytest.mark.parametrize("inclusive_improves", [False, True])
def test_best_and_early_stopping_follow_inclusive_not_exclusive_dice(packaged, tmp_path, inclusive_improves):
    from unittest.mock import patch
    from training.train import main as train
    from training.trainer.trainer import load_checkpoint
    from training.metrics.surface_distance import dataset_rows, summarize_dataset_rows
    cache, _ = packaged
    run = tmp_path / "inclusive_run"
    argv = ["--dataset", "myopspp_bc80", "--config", "testing", "--data-root", str(cache),
            "--list-dir", str(cache / "lists"), "--output-dir", str(run),
            "--epochs", "3", "--epochs-per-run", "1", "--patience", "1", "--batch-size", "64", "--accum-steps", "1",
            "--num-workers", "0", "--cpu-threads", "2", "--device", "cpu", "--amp", "none", "--no-tensorboard"]
    truth = np.zeros((32, 32, 1), dtype=np.uint8)
    truth[10:14, 10:14] = 2
    truth[16:20, 16:20] = 3
    # A: whole lesion predicted as scar: inclusive=5/6, exclusive=1/3.
    # B: half of each class missed: inclusive=2/3, exclusive=2/3.
    # The two metrics rank the epochs in opposite orders.
    merged = truth.copy()
    merged[merged == 2] = 3
    partial = truth.copy()
    partial[10:12, 10:14] = 0
    partial[16:18, 16:20] = 0
    predictions = (partial, merged) if inclusive_improves else (merged, partial)
    metrics = [summarize_dataset_rows(dataset_rows(pred, truth, "validation", compute_distance=False,
                                                  dataset_id="myopspp_bc80", spacing=[1.5, 1.5, 7]),
                                      dataset_id="myopspp_bc80") for pred in predictions]
    with patch("training.trainer.trainer.validate_volumes", return_value=metrics[0]):
        train(argv)
    best = run / "checkpoints/best.pt"
    assert load_checkpoint(best)["epoch"] == 0
    last = run / "checkpoints/last.pt"
    before = best.read_bytes()
    with patch("training.trainer.trainer.validate_volumes", return_value=metrics[1]):
        train(argv + ["--resume", str(last)])
    expected_epoch = 1 if inclusive_improves else 0
    assert load_checkpoint(best)["epoch"] == expected_epoch
    saved = load_checkpoint(last)
    assert saved["best_score"] == pytest.approx(5 / 6)
    assert saved["best_epoch"] == expected_epoch
    assert saved["bad_epochs"] == (0 if inclusive_improves else 1)
    assert saved["benchmark_protocol"]["selection_metric"] == "avg_pathology_inclusive_dice"
    assert {p.name for p in best.parent.iterdir()} == {"best.pt", "last.pt"}
    assert "best_inclusive_score" not in saved
    if not inclusive_improves:
        assert best.read_bytes() == before
    from training.run_layout import RunLayout
    records = [json.loads(line) for line in (RunLayout.from_checkpoint(last).logs / "metrics.jsonl").read_text().splitlines()]
    assert records[-1]["early_stop"] is (not inclusive_improves)
    assert records[-1]["val/avg_pathology_dice"] == pytest.approx(metrics[1]["avg_pathology_dice"])
    assert records[-1]["val/avg_pathology_inclusive_dice"] == pytest.approx(metrics[1]["avg_pathology_inclusive_dice"])


def test_dataset_identity_in_yaml_selects_complete_profile_and_cli_can_override(tmp_path):
    import yaml
    from training.train import parse_args
    config = tmp_path / "model.yaml"
    config.write_text(yaml.safe_dump({"data": {"dataset_id": "myopspp_bc80"}}))
    args, _ = parse_args(["--config", str(config)])
    assert args.dataset_id == "myopspp_bc80"
    assert "myopspp_bc80" in args.data_root
    assert args.label_order == "canonical"
    explicit, merged = parse_args(["--config", str(config), "--dataset", "myops380"])
    assert explicit.dataset_id == merged["data"]["dataset_id"] == "myops380"
    assert "MyoPS380" in explicit.data_root
    assert explicit.label_order == "legacy"


@pytest.mark.parametrize("output_key", ["dir", "output_dir"])
def test_run_all_honors_yaml_output_directory(packaged, tmp_path, output_key):
    import yaml
    from run_all import main
    cache, _ = packaged
    config = yaml.safe_load(Path("training/config/models/testing.yaml").read_text())
    run = tmp_path / "explicit_output"
    manifests = tmp_path / "configured_lists"
    manifests.mkdir()
    for path in (cache / "lists").glob("*.txt"):
        (manifests / path.name).write_bytes(path.read_bytes())
    config["data"].update(dataset_id="myopspp_bc80", data_root=str(cache), list_dir=str(manifests))
    config["outputs"] = {output_key: str(run)}
    config_path = tmp_path / "configured.yaml"
    config_path.write_text(yaml.safe_dump(config))
    actual = main(["--config", str(config_path), "--skip-cache",
                   "--run-root", str(tmp_path / "generated_runs"),
                   "--epochs", "1", "--batch-size", "64", "--accum-steps", "1", "--device", "cpu",
                   "--amp", "none", "--num-workers", "0", "--cpu-threads", "2", "--no-tensorboard"])
    assert actual == run
    assert (run / "checkpoints/best.pt").is_file()
    from training.run_layout import RunLayout
    logged = json.loads((RunLayout.from_checkpoint(run / "checkpoints/best.pt").logs / "config.json").read_text())
    assert Path(logged["args"]["list_dir"]) == manifests


def test_yaml_cache_relocation_preserves_explicit_lists(tmp_path):
    import yaml
    from training.train import parse_args
    config = tmp_path / "model.yaml"
    custom_cache = tmp_path / "custom_cache"
    custom_lists = tmp_path / "external_lists"
    document = {"data": {"dataset_id": "myopspp_bc80", "data_root": str(custom_cache)}}
    config.write_text(yaml.safe_dump(document))
    implicit, _ = parse_args(["--config", str(config)])
    assert Path(implicit.list_dir) == custom_cache / "lists"
    document["data"]["list_dir"] = str(custom_lists)
    config.write_text(yaml.safe_dump(document))
    explicit, _ = parse_args(["--config", str(config)])
    assert Path(explicit.list_dir) == custom_lists


def test_resume_of_early_stopped_checkpoint_performs_no_more_updates(packaged, tmp_path):
    from unittest.mock import patch
    from training.train import main as train
    from training.trainer.trainer import load_checkpoint
    from training.metrics.surface_distance import dataset_rows, summarize_dataset_rows
    cache, _ = packaged
    run = tmp_path / "early_stop_run"
    argv = ["--dataset", "myopspp_bc80", "--config", "testing", "--data-root", str(cache),
            "--list-dir", str(cache / "lists"), "--output-dir", str(run), "--epochs", "3", "--patience", "1",
            "--batch-size", "64", "--accum-steps", "1", "--num-workers", "0", "--cpu-threads", "2",
            "--device", "cpu", "--amp", "none", "--no-tensorboard"]
    truth = np.zeros((32, 32, 1), dtype=np.uint8)
    truth[10:14, 10:14] = 2
    truth[16:20, 16:20] = 3
    metrics = summarize_dataset_rows(dataset_rows(truth, truth, "validation", compute_distance=False,
                                                 dataset_id="myopspp_bc80", spacing=[1.5, 1.5, 7]),
                                    dataset_id="myopspp_bc80")
    with patch("training.trainer.trainer.validate_volumes", return_value=metrics):
        first = train(argv)
    last = run / "checkpoints/last.pt"
    before = last.read_bytes()
    assert load_checkpoint(last)["bad_epochs"] == 1
    second = train(argv + ["--resume", str(last)])
    assert second["global_step"] == first["global_step"]
    assert last.read_bytes() == before
