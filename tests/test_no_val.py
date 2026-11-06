"""The paper protocol must fail closed on cohort, budget and resume mistakes."""
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch
import yaml

from training.dataset.no_val import REFERENCE_DIR, read_no_val_splits, validate_no_val_splits
from training.dataset.data_contract import lock_benchmark_data, ensure_patient_splits
from training.protocol import validate_training_checkpoint
from training.train import main as train, parse_args
from training.evaluate import main as evaluate
from training.trainer.trainer import load_checkpoint
from training.models import get_testing
from training.dataset.data_contract import CLASS_NAMES
from training.run_layout import RunLayout


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("myops380_no_val")
    lists = root / "lists"
    shutil.copytree(REFERENCE_DIR, lists)
    names = read_no_val_splits(lists)
    image = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    label = np.zeros((8, 8), np.uint8)
    label[2:6, 2:6], label[3:4, 3:5], label[5:6, 3:5] = 1, 2, 3
    for modality in ("bSSFP", "LGE", "T2w"):
        slices, volumes = root / modality / "train_npz", root / modality / "test_vol_h5"
        slices.mkdir(parents=True)
        volumes.mkdir()
        for name in names["train"]:
            np.savez(slices / f"{name}.npz", image=image, label=label)
        for name in names["test_vol"]:
            with h5py.File(volumes / f"{name}.npy.h5", "w") as stream:
                stream["image"], stream["label"] = image[..., None], label[..., None]
    return root


class TinyModel(torch.nn.Module):
    """Exercise the real trainer/optimizer/RNG/evaluator cheaply on the full cohort."""
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.net = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 1), torch.nn.Dropout2d(0.1),
                                       torch.nn.Conv2d(8, 4, 1))

    def forward(self, a, b, c):
        return self.net(torch.cat((a, b, c), dim=1))


def argv(cache, run, *extra):
    return ["--config", "testing", "--no-val", "--data-root", str(cache),
            "--list-dir", str(cache / "lists"), "--output-dir", str(run),
            "--epochs", "2", "--batch-size", "127", "--accum-steps", "3",
            "--label-order", "canonical", "--device", "cpu", "--amp", "none",
            "--num-workers", "0", "--cpu-threads", "2", "--log-every", "100", "--no-tensorboard", *extra]


def test_config_and_conflicts(tmp_path):
    assert parse_args([])[0].no_val is True
    config = tmp_path / "config.yaml"
    config.write_text("train:\n  no_val: true\n")
    assert parse_args(["--config", str(config)])[0].no_val is True
    assert parse_args(["--no-val", "--val-fraction", "0"])[0].no_val is True
    for flags in (["--dataset", "myopspp_bc80"], ["--patience", "3"], ["--min-delta", "0.1"]):
        with pytest.raises(SystemExit):
            parse_args(["--no-val", *flags])


@pytest.mark.parametrize("source", ["pretrained", "init_weights"])
def test_resume_ignores_yaml_initialization_but_rejects_explicit_cli_sources(tmp_path, source):
    config = tmp_path / "initialize.yaml"
    config.write_text(yaml.safe_dump({"train": {source: "missing_initialization_file"}}))
    args, merged = parse_args(["--config", str(config), "--resume", "last.pt"])
    assert args.resume == "last.pt"
    assert args.pretrained is args.init_weights is None
    assert merged["train"][source] == "missing_initialization_file"
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config), "--resume", "last.pt",
                    "--" + source.replace("_", "-"), "explicit.pt"])


def test_checkpoint_budget_is_derived_from_locked_data_and_accumulation():
    from training.protocol import training_protocol
    checkpoint = {
        "args": {"no_val": True, "dataset_id": "myops380", "max_epochs": 300,
                 "batch_size": 16, "accum_steps": 1},
        "training_protocol": training_protocol(True), "epoch": 299,
        "planned_epochs": 300, "completed_epochs": 300, "training_complete": True,
        "total_updates": 300, "global_step": 300, "amp_skipped_updates": 0,
        "scheduler": {"last_epoch": 300},
    }
    with pytest.raises(ValueError, match="optimizer updates"):
        validate_training_checkpoint(checkpoint, require_complete=True)
    checkpoint.update(total_updates=37200, global_step=37198, amp_skipped_updates=2,
                      scheduler={"last_epoch": 37198})
    assert validate_training_checkpoint(checkpoint, require_complete=True)


def test_split_is_fixed_complete_and_seed_independent(cache, tmp_path):
    from preprocessing.build_splits import build_splits
    from preprocessing.verify import verify_dataset
    for seed in (0, 1234):
        dest = tmp_path / str(seed)
        result = build_splits(cache, dest, seed=seed, no_val=True)
        assert result["patient_counts"] == {"train": 304, "test_vol": 76}
        assert result["sample_counts"] == {"train": 1972, "test_vol": 76}
        assert read_no_val_splits(dest) == read_no_val_splits(REFERENCE_DIR)
    report = verify_dataset(cache, cache / "lists", "canonical", no_val=True)
    assert report["sample_counts"] == {"train": 1972, "test_vol": 76}
    assert "selection_metric" not in report["benchmark_protocol"]


@pytest.mark.parametrize("problem", ["missing_patient", "missing_slice", "duplicate", "test_swap", "overlap"])
def test_bad_cohort_is_rejected(problem):
    names = read_no_val_splits(REFERENCE_DIR)
    if problem == "missing_patient":
        case = names["train"][0].split("_slice")[0]
        names["train"] = [n for n in names["train"] if not n.startswith(case + "_")]
    elif problem == "missing_slice":
        names["train"].pop(1)
    elif problem == "duplicate":
        names["train"].append(names["train"][0])
    elif problem == "test_swap":
        names["test_vol"][0] = "case9999"
    else:
        names["train"].append(names["test_vol"][0] + "_slice000")
    with pytest.raises(ValueError):
        validate_no_val_splits(names)


def test_saved_manifests_cannot_silently_change(cache, tmp_path):
    saved = tmp_path / "saved"
    ensure_patient_splits(cache / "lists", output_dir=saved, no_val=True)
    with (saved / "train.txt").open("a") as stream:
        stream.write("case0002_slice000\n")
    with pytest.raises(ValueError):
        ensure_patient_splits(cache / "lists", output_dir=saved, no_val=True)
    shutil.copytree(REFERENCE_DIR, tmp_path / "with_val")
    (tmp_path / "with_val/val.txt").touch()
    with pytest.raises(ValueError, match="without val"):
        read_no_val_splits(tmp_path / "with_val")


def test_cache_missing_modality_and_mutation(cache):
    before = lock_benchmark_data(cache, cache / "lists", "canonical", no_val=True)
    name = read_no_val_splits(REFERENCE_DIR)["train"][0]
    path = cache / "LGE/train_npz" / f"{name}.npz"
    original = path.read_bytes()
    try:
        path.rename(path.with_suffix(".hidden"))
        with pytest.raises(ValueError, match="inventory"):
            lock_benchmark_data(cache, cache / "lists", "canonical", no_val=True)
        path.with_suffix(".hidden").rename(path)
        with np.load(path) as data:
            image, label = data["image"].copy(), data["label"].copy()
        image[0, 0] = 0.5
        np.savez(path, image=image, label=label)
        assert lock_benchmark_data(cache, cache / "lists", "canonical", no_val=True) != before
    finally:
        if path.with_suffix(".hidden").exists():
            path.with_suffix(".hidden").rename(path)
        path.write_bytes(original)


@pytest.mark.parametrize("recipe", ["per_image_accumulation", "im_mseg_batch"])
def test_train_resume_matches_uninterrupted_and_report_is_complete(cache, tmp_path, recipe):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    # Exercise the recommended same-YAML resume path after the initialization
    # source file has disappeared (e.g. a new Colab session).
    initial = tmp_path / "initial.pt"
    initial_model = TinyModel(get_testing())
    torch.save(dict(format_version=1, class_names=CLASS_NAMES,
                    model_config=initial_model.config.to_dict(), model=initial_model.state_dict()), initial)
    initial_sha = hashlib.sha256(initial.read_bytes()).hexdigest()
    config = yaml.safe_load(Path("training/config/models/testing.yaml").read_text())
    config["train"]["init_weights"] = str(initial)
    config_path = tmp_path / "initialized.yaml"
    config_path.write_text(yaml.safe_dump(config))
    original_argv = argv
    def initialized_argv(*args, **kwargs):
        flags = original_argv(*args, **kwargs) + ["--config", str(config_path)]
        if recipe == "im_mseg_batch":
            flags += ["--negative-slice-handling", "batch", "--accum-steps", "1"]
        return flags
    with patch("training.train.model_from_config", side_effect=TinyModel), \
         patch("training.trainer.trainer.validate_volumes", side_effect=AssertionError("validation called")), \
         patch("training.evaluate.model_from_config", side_effect=TinyModel):
        train(initialized_argv(cache, full))
        train(initialized_argv(cache, resumed, "--epochs-per-run", "1"))
        last = resumed / "checkpoints/testingsmall_seed1234.pt"
        incomplete = load_checkpoint(last)
        assert incomplete["initialization"]["kind"] == "init_weights"
        assert incomplete["initialization"]["sha256"] == initial_sha
        initial.unlink()
        assert incomplete["training_complete"] is False
        with pytest.raises(ValueError, match="Incomplete"):
            evaluate(["--checkpoint", str(last), "--data-root", str(cache)])
        # Protocol and planned-budget mismatches fail before training another epoch.
        with pytest.raises(ValueError, match="no_val"):
            train([a for a in initialized_argv(cache, resumed, "--resume", str(last)) if a != "--no-val"])
        with pytest.raises(ValueError, match="max_epochs"):
            train(initialized_argv(cache, resumed, "--resume", str(last), "--epochs", "3"))
        changed = deepcopy(incomplete)
        changed["code_provenance"]["files_sha256"]["training/train.py"] = "changed"
        torch.save(changed, last)
        with pytest.raises(ValueError, match="source code"):
            train(initialized_argv(cache, resumed, "--resume", str(last)))
        torch.save(incomplete, last)
        train(initialized_argv(cache, resumed, "--resume", str(last)))
        actual, expected = load_checkpoint(last), load_checkpoint(full / "checkpoints/testingsmall_seed1234.pt")
        assert actual["training_complete"] is True
        assert actual["initialization"] == expected["initialization"] == incomplete["initialization"]
        assert actual["completed_epochs"] == 2
        expected_updates = 32 if recipe == "im_mseg_batch" else 12
        assert actual["global_step"] == expected["global_step"] == expected_updates
        assert actual["scheduler"] == expected["scheduler"]
        for key in actual["model"]:
            assert torch.equal(actual["model"][key], expected["model"][key]), key
        assert {p.name for p in last.parent.iterdir()} == {"testingsmall_seed1234.pt"}
        assert not any(k in actual for k in ("best_score", "best_epoch", "validation_volume_metrics"))
        saved_lists = RunLayout.from_checkpoint(last).logs / "splits"
        assert {p.name for p in saved_lists.iterdir()} == {"train.txt", "test_vol.txt"}
        with pytest.raises(ValueError, match="test_vol"):
            evaluate(["--checkpoint", str(last), "--data-root", str(cache), "--split", "val_vol"])
        for key, value in (("completed_epochs", 1), ("training_complete", False),
                           ("amp_skipped_updates", 1), ("training_protocol", {})):
            bad = deepcopy(actual)
            bad[key] = value
            with pytest.raises(ValueError):
                validate_training_checkpoint(bad, require_complete=True)
        summary = evaluate(["--checkpoint", str(last), "--data-root", str(cache), "--device", "cpu",
                            "--amp", "none", "--no-save-predictions"])
        assert summary["case_count"] == 76
        assert summary["checkpoint_epoch"] == summary["planned_epochs"] == 2
        assert summary["training_complete"] is True
        assert summary["checkpoint_selection_metric"] is None
        assert summary["training_protocol"]["checkpoint_selection"] == "last_completed_epoch"
        assert summary["checkpoint_sha256"] == hashlib.sha256(last.read_bytes()).hexdigest()
        assert "mean_hd95_voxel" in summary["scar"]
        # Locked manifest and cache mutations must still be rejected by the evaluator.
        manifest = saved_lists / "test_vol.txt"
        original = manifest.read_bytes()
        try:
            manifest.write_bytes(original + b"case9999\n")
            with pytest.raises(ValueError, match="modified"):
                evaluate(["--checkpoint", str(last), "--data-root", str(cache)])
        finally:
            manifest.write_bytes(original)
        # Even a valid-looking image change invalidates the byte-locked test report.
        image_path = cache / "T2w/train_npz" / (read_no_val_splits(REFERENCE_DIR)["train"][0] + ".npz")
        image_bytes = image_path.read_bytes()
        try:
            with np.load(image_path) as sample:
                image, label = sample["image"].copy(), sample["label"].copy()
            image[0, 0] = 0.25
            np.savez(image_path, image=image, label=label)
            with pytest.raises(ValueError, match="Cache bytes"):
                evaluate(["--checkpoint", str(last), "--data-root", str(cache)])
        finally:
            image_path.write_bytes(image_bytes)


def test_run_all_propagates_no_val_and_selects_last(cache, tmp_path):
    import run_all
    commands = []
    run = tmp_path / "pipeline"
    def command(args, description):
        commands.append(args)
        if args[1] == "training/train.py":
            layout = RunLayout.create(run, "M2-Plus", 1234)
            (layout.checkpoints / "m2plus_seed1234.pt").touch()
    with patch("run_all.run_command", side_effect=command):
        run_all.main(["--no-val", "--skip-cache", "--data-root", str(cache),
                      "--list-dir", str(cache / "lists"), "--run-root", str(tmp_path), "--run-id", "pipeline"])
    assert all("--no-val" in c for c in commands[:-1])
    assert commands[-1][commands[-1].index("--checkpoint") + 1] == str(run / "checkpoints/m2plus_seed1234.pt")
    with pytest.raises(SystemExit):
        run_all.main(["--no-val", "--eval-split", "val_vol"])


def test_raw_split_never_uses_seed_for_no_val():
    from preprocessing.preprocessing import create_patient_splits
    cases = [f"case{i:04d}" for i in range(1, 381)]
    tests = read_no_val_splits(REFERENCE_DIR)["test_vol"]
    first = create_patient_splits(cases, seed=1, test_ids=tests, no_val=True)
    assert first == create_patient_splits(cases, seed=999, test_ids=tests, no_val=True)
    assert {k: len(v) for k, v in first.items()} == {"train": 304, "test_vol": 76}
    with pytest.raises(ValueError):
        create_patient_splits(cases, no_val=True)


def test_raw_packaging_creates_only_train_and_test_and_maps_legacy_labels(tmp_path):
    import nibabel as nib
    from collections import Counter
    from preprocessing.process_and_save import preprocess_dataset
    from preprocessing.verify import verify_dataset
    from training.dataset.myops_dataset import MyopsDataset
    names = read_no_val_splits(REFERENCE_DIR)
    depths = Counter(n.split("_slice")[0] for n in names["train"])
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    raw.mkdir()
    for case in sorted(set(depths) | set(names["test_vol"])):
        shape = (8, 8, depths.get(case, 1))
        image = np.ones(shape, np.float32) * 128
        label = np.zeros(shape, np.uint8)
        label[2:4, 2:4], label[4:6, 4:6] = 2, 3  # legacy scar, edema
        for modality, array in (("bSSFP", image), ("LGE", image), ("T2w", image), ("label", label)):
            volume = nib.Nifti1Image(array, np.eye(4))
            volume.header.set_xyzt_units("mm")
            directory = raw / modality
            directory.mkdir(exist_ok=True)
            nib.save(volume, directory / f"{case}.nii.gz")
    # This is the paper config's real layout: manifests already exist outside
    # the new cache directory. They must be reused without modifying any bytes.
    lists = tmp_path / "versioned_lists"
    shutil.copytree(REFERENCE_DIR, lists)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in lists.iterdir()}
    result = preprocess_dataset(raw, cache, list_dir=lists, test_list=REFERENCE_DIR / "test_vol.txt",
                                no_val=True, show_progress=False)
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in lists.iterdir()}
    assert result["sample_counts"] == {"train": 1972, "test_vol": 76}
    assert not any(cache.glob("*/val_vol_h5"))
    report = verify_dataset(cache, lists, "canonical", no_val=True)
    assert report["patient_counts"] == {"train": 304, "test_vol": 76}
    dataset = MyopsDataset(*[cache / m / "train_npz" for m in ("bSSFP", "LGE", "T2w")],
                          lists, "train", label_order="canonical")
    assert dataset[0]["label"][2, 2] == 3 and dataset[0]["label"][4, 4] == 2
