"""Reproducible, prediction-localized MyoPS++ ROI exports, independent of SCAR training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, label as components
import torch
from torch import nn
import torch.nn.functional as F

SOURCE = Path(r"E:\STUDY\DATASET\Myo_train\MyoPS_train")
SPLITS = Path(r"D:\NCKH\MyoPSpp_preprocessed_2026-10-03\roi128mm_80cases\splits.json")
RUN = Path(r"D:\NCKH\MyoPSpp_UNet2D_localizer_2026-10-27")
DESTINATIONS = {160: Path(r"D:\NCKH\MyoPSpp_UNet2D_ROI128_2026-10-27"),
                128: Path(r"D:\NCKH\MyoPSpp_UNet2D_ROI128_1mm_2026-10-27")}
RAW_CODES = (0, 200, 500, 600, 1220, 2221)
TARGETS = {"all_labels": (200, 500, 600, 1220, 2221), "myocardium": (200, 1220, 2221),
           "pathology": (1220, 2221), "scar": (2221,), "edema_exclusive": (1220,)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def normalize(array):
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite MRI intensity")
    positive = array[array > 0]
    if not positive.size:
        return np.zeros_like(array), {"p1": None, "p99": None, "constant": True}
    low, high = map(float, np.percentile(positive, [1, 99]))
    if high <= low:
        return np.zeros_like(array), {"p1": low, "p99": high, "constant": True}
    return np.clip((array - low) / (high - low), 0, 1).astype(np.float32), {"p1": low, "p99": high, "constant": False}


def target_grid(image):
    spacing = nib.affines.voxel_sizes(image.affine)
    direction = image.affine[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise ValueError("Expected orthogonal MRI grid")
    aff = image.affine.copy()
    aff[:3, 0] /= spacing[0]
    aff[:3, 1] /= spacing[1]
    shape = tuple(int(v) for v in np.ceil((np.array(image.shape[:2]) - 1) * spacing[:2]).astype(int) + 1) + (int(image.shape[2]),)
    return shape, aff


def resample_xy(image, shape, affine, order):
    if image.shape[2] != shape[2]:
        raise ValueError("Z slice count cannot change")
    mapping = np.linalg.solve(image.affine, affine)
    mapping[np.abs(mapping) < 1e-10] = 0
    if not np.allclose(mapping[2], [0, 0, 1, 0], atol=1e-4):
        raise ValueError("Input is not on matching native Z planes")
    source = np.asarray(image.dataobj)
    out = np.empty(shape, dtype=np.float32 if order else np.uint16)
    for z in range(shape[2]):
        affine_transform(source[:, :, z], mapping[:2, :2], mapping[:2, 3] + mapping[:2, 2] * z,
                         output_shape=shape[:2], output=out[:, :, z], order=order,
                         mode="constant", cval=0, prefilter=False)
    return out


def letterbox(array, side=256, is_label=False):
    """HWD -> D1SS, with exact inverse transform metadata."""
    h, w, _ = array.shape
    scale = side / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (side - nh) // 2, (side - nw) // 2
    x = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32))[:, None]
    if is_label:
        x = F.interpolate(x, size=(nh, nw), mode="nearest-exact")
    else:
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    x = F.pad(x, (left, side - nw - left, top, side - nh - top))
    return x.numpy(), {"source_hw": [h, w], "resized_hw": [nh, nw], "pad_top_left": [top, left], "model_side": side}


def unletterbox(probabilities, transform):
    nh, nw = transform["resized_hw"]
    top, left = transform["pad_top_left"]
    x = probabilities[:, :, top:top + nh, left:left + nw]
    x = F.interpolate(x, size=transform["source_hw"], mode="bilinear", align_corners=False)
    return x[:, 0].cpu().numpy().transpose(1, 2, 0)


def predicted_center(mask):
    """Accepts predicted foreground only. Never consumes a reference annotation."""
    clean = np.zeros(mask.shape, dtype=np.uint8)
    for z in range(mask.shape[2]):
        cc, n = components(mask[:, :, z], structure=np.ones((3, 3)))
        if n:
            sizes = np.bincount(cc.ravel())
            sizes[0] = 0
            clean[:, :, z] = cc == sizes.argmax()
    pos = np.argwhere(clean)
    if not pos.size:
        center = np.round((np.array(mask.shape[:2]) - 1) / 2).astype(int)
        return center, clean, {"empty_prediction": True, "fallback": "image_center_same_fixed_FOV", "bbox_hw": None}
    lo, hi = pos[:, :2].min(0), pos[:, :2].max(0)
    center = np.round((lo + hi) / 2).astype(int)
    return center, clean, {"empty_prediction": False, "fallback": None,
                            "bbox_start_hw": lo.tolist(), "bbox_stop_exclusive_hw": (hi + 1).tolist(),
                            "bbox_hw": (hi - lo + 1).tolist(), "foreground_voxels": int(pos.shape[0])}


def crop_fixed(array, affine, center, side):
    start = np.asarray(center, dtype=int) - side // 2
    stop = start + side
    lo = np.maximum(start, 0)
    hi = np.minimum(stop, array.shape[:2])
    out = np.zeros((side, side, array.shape[2]), dtype=array.dtype)
    if np.all(hi > lo):
        dst = lo - start
        out[dst[0]:dst[0] + hi[0] - lo[0], dst[1]:dst[1] + hi[1] - lo[1], :] = array[lo[0]:hi[0], lo[1]:hi[1], :]
    translation = np.eye(4)
    translation[:2, 3] = start
    return out, affine @ translation, {"start_hw": start.tolist(), "stop_exclusive_hw": stop.tolist(),
                                       "pad_before_hw": np.maximum(-start, 0).tolist(),
                                       "pad_after_hw": np.maximum(stop - array.shape[:2], 0).tolist()}


def resize_roi(array, affine, is_label=False):
    if array.shape[:2] == (128, 128):
        return array, affine.copy()
    side = array.shape[0]
    assert array.shape[1] == side
    x = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32))[:, None]
    if is_label:
        x = F.interpolate(x, size=(128, 128), mode="nearest-exact")
    else:
        x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
    result = x[:, 0].numpy().transpose(1, 2, 0).astype(array.dtype)
    scale = side / 128
    mapping = np.diag([scale, scale, 1., 1.])
    mapping[:2, 3] = (scale - 1) / 2
    return result, affine @ mapping


def save_nii(path, array, affine, is_label=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(array.astype(np.uint16 if is_label else np.float32), affine)
    image.header.set_xyzt_units("mm", "unknown")
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, str(path))


def label_counts(array):
    return {str(k): int(np.count_nonzero(array == k)) for k in RAW_CODES[1:]}


class ConvBlock(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.GroupNorm(min(8, b), b), nn.ReLU(inplace=True),
                         nn.Conv2d(b, b, 3, padding=1, bias=False), nn.GroupNorm(min(8, b), b), nn.ReLU(inplace=True))


class SmallUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.down = nn.ModuleList([ConvBlock(1, base), ConvBlock(base, base * 2),
                                   ConvBlock(base * 2, base * 4), ConvBlock(base * 4, base * 8)])
        self.bottom = ConvBlock(base * 8, base * 16)
        self.up = nn.ModuleList([ConvBlock(base * 24, base * 8), ConvBlock(base * 12, base * 4),
                                 ConvBlock(base * 6, base * 2), ConvBlock(base * 3, base)])
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottom(x)
        for block, skip in zip(self.up, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], 1))
        return self.head(x)


def prepare(args):
    if (args.run / "protocol.json").exists():
        print("Preparation already complete; reusing locked protocol.", flush=True)
        return
    args.run.mkdir(parents=True, exist_ok=True)
    splitdata = json.loads(args.splits.read_text(encoding="utf-8"))
    patients = splitdata["patients"]
    assert {k: len(v) for k, v in patients.items()} == {"train": 51, "val": 13, "test_vol": 16}
    assert len(set(sum(patients.values(), []))) == 80
    assignment = {case: split for split, cases in patients.items() for case in cases}
    rng = np.random.default_rng(1234)
    held_out = [[], [], []]
    for prefix in ("Case2", "Case3"):
        cases = sorted(c for c in patients["train"] if c.startswith(prefix))
        rng.shuffle(cases)
        for i, c in enumerate(cases):
            held_out[i % 3].append(c)
    models = {f"fold{i}": {"train": sorted(set(patients["train"]) - set(cases)),
                            "held_out_train": sorted(cases), "val": patients["val"]} for i, cases in enumerate(held_out)}
    models["final"] = {"train": patients["train"], "held_out_train": [], "val": patients["val"]}
    protocol = {"schema": 1, "seed": 1234, "source": str(args.source), "split_file": str(args.splits),
                "split_sha256": digest(args.splits), "patients": patients, "models": models,
                "model": "small_unet2d_groupnorm_base16", "input": "C0 full-FOV, 1mm XY grid, positive-volume p1/p99, letterbox256",
                "foreground_labels": [200, 1220, 2221], "threshold": .5,
                "postprocessing": "largest 8-connected component per native Z slice; no GT correction",
                "center": "np.round(midpoint of predicted myocardium bbox across all Z), common for volume",
                "crop_widths_mm": [160, 128], "model_output_hw": [128, 128],
                "empty_mask_policy": "image center; retain fixed FOV; record failure; no exclusions",
                "crop_clipping_policy": "retain all cases; no enlargement or GT-guided correction",
                "epochs": 100, "patience": 25, "batch_size": 4, "learning_rate": .001,
                "weight_decay": .0001, "loss": "0.5 BCE + 0.5 sample-mean soft Dice",
                "augmentation": "paired XY flip, rotation +/-20deg, scale[0.9,1.1], shift +/-0.08 normalized; intensity scale[0.9,1.1], shift +/-0.05",
                "checkpoint_selection": "mean patient Dice on 13 validation patients, threshold0.5 on letterbox grid; no test selection",
                "raw_label_mapping_for_downstream": {"0": 0, "500": 0, "600": 0, "200": 1, "1220": 2, "2221": 3},
                "audit_disclosure": "Full-cohort GT coverage was explored before this experiment; this test cohort is not claimed untouched by exploratory audits."}
    casesmeta, images, masks, sample_cases = {}, [], [], []
    for case in sorted(assignment):
        center = "CenterB" if case.startswith("Case2") else "CenterC"
        folder = args.source / center / case
        paths = {m: folder / f"{case}_{m}.nii.gz" for m in ("C0", "LGE", "T2", "gd")}
        source = {m: nib.load(str(p)) for m, p in paths.items()}
        ref = source["C0"]
        for m, image in source.items():
            if len(image.shape) != 3 or image.shape != ref.shape or not np.allclose(image.affine, ref.affine, atol=1e-4, rtol=1e-4):
                raise ValueError(f"{case} {m}: mismatched grid")
            if image.header.get_xyzt_units()[0] != "mm":
                raise ValueError(f"{case} {m}: missing physical units")
        rawgt = np.asarray(source["gd"].dataobj)
        if not set(np.unique(rawgt)).issubset(RAW_CODES):
            raise ValueError(f"{case}: unexpected labels")
        shape, affine = target_grid(ref)
        c0, norm = normalize(resample_xy(ref, shape, affine, 1))
        gt = resample_xy(source["gd"], shape, affine, 0)
        x, box = letterbox(c0)
        y, ybox = letterbox(np.isin(gt, (200, 1220, 2221)).astype(np.uint8), is_label=True)
        assert box == ybox
        lo = len(sample_cases)
        images.extend(x)
        masks.extend(y.astype(np.uint8))
        sample_cases.extend([case] * shape[2])
        casesmeta[case] = {"center": center, "split": assignment[case], "paths": {m: str(p) for m, p in paths.items()},
                           "source_sha256": {m: digest(p) for m, p in paths.items()},
                           "native_shape": list(ref.shape), "native_affine": ref.affine.tolist(),
                           "native_spacing_mm": nib.affines.voxel_sizes(ref.affine).tolist(),
                           "grid_1mm_shape": list(shape), "grid_1mm_affine": affine.tolist(),
                           "letterbox": box, "input_normalization": norm, "sample_range": [lo, len(sample_cases)]}
    write_json(args.run / "cases.json", casesmeta)
    np.savez_compressed(args.run / "localizer_samples.npz", image=np.stack(images), label=np.stack(masks), case=np.array(sample_cases))
    write_json(args.run / "environment.json", {"python": sys.version, "executable": sys.executable,
               "torch": torch.__version__, "numpy": np.__version__, "nibabel": nib.__version__, "platform": platform.platform(),
               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    write_json(args.run / "protocol.json", protocol)
    print(f"Prepared {len(casesmeta)} patients / {len(images)} slices; OOF sizes {[len(x) for x in held_out]}", flush=True)


def configure(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def augment(x, y):
    n = len(x)
    angle = (torch.rand(n, device=x.device) * 2 - 1) * math.pi / 9
    scale = .9 + torch.rand(n, device=x.device) * .2
    flip = torch.where(torch.rand(n, device=x.device) < .5, -1., 1.)
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = angle.cos() * scale * flip
    theta[:, 0, 1] = -angle.sin() * scale
    theta[:, 1, 0] = angle.sin() * scale * flip
    theta[:, 1, 1] = angle.cos() * scale
    theta[:, :, 2] = (torch.rand(n, 2, device=x.device) * 2 - 1) * .08
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", align_corners=False)
    y = F.grid_sample(y, grid, mode="nearest", align_corners=False)
    x = (x * (.9 + .2 * torch.rand(n, 1, 1, 1, device=x.device)) +
         (.1 * torch.rand(n, 1, 1, 1, device=x.device) - .05)).clamp_(0, 1)
    return x, y


@torch.inference_mode()
def validation(model, x, y, sample_cases, indices, device, batch_size):
    model.eval()
    totals = {}
    for off in range(0, len(indices), batch_size):
        idx = indices[off:off + batch_size]
        inp = torch.from_numpy(x[idx]).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred = model(inp).float().sigmoid() >= .5
        gt = torch.from_numpy(y[idx]).to(device).bool()
        inter = (pred & gt).sum((1, 2, 3)).cpu().numpy()
        denom = (pred.sum((1, 2, 3)) + gt.sum((1, 2, 3))).cpu().numpy()
        for c, a, b in zip(sample_cases[idx], inter, denom):
            prev = totals.setdefault(str(c), [0, 0])
            prev[0] += int(a)
            prev[1] += int(b)
    scores = {c: 2 * a / b if b else 1. for c, (a, b) in totals.items()}
    return float(np.mean(list(scores.values()))), scores


def train(args):
    protocol = json.loads((args.run / "protocol.json").read_text(encoding="utf-8"))
    configure(protocol["seed"] + list(protocol["models"]).index(args.model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = np.load(args.run / "localizer_samples.npz")
    x, y, cases = cache["image"], cache["label"], cache["case"]
    membership = protocol["models"][args.model]
    assert not set(membership["train"]) & set(membership["val"])
    assert not set(membership["train"]) & set(membership["held_out_train"])
    ti = np.flatnonzero(np.isin(cases, membership["train"]))
    vi = np.flatnonzero(np.isin(cases, membership["val"]))
    folder = args.run / "models" / args.model
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "complete.json").exists() and not args.pilot:
        print(f"{args.model}: already completed", flush=True)
        return
    model = SmallUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    bs = protocol["batch_size"]
    best, best_epoch, stale, start = -1., 0, 0, 1
    lastpath = folder / "last.pt"
    if lastpath.exists() and not args.pilot:
        state = torch.load(lastpath, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        best, best_epoch, stale, start = state["best"], state["best_epoch"], state["stale"], state["epoch"] + 1
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_state"])
        torch.set_rng_state(state["torch_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_states"]])
    total_start = time.time()
    max_epochs = 1 if args.pilot else protocol["epochs"]
    for epoch in range(start, max_epochs + 1):
        model.train()
        order = np.random.permutation(ti)
        losses = []
        epoch_start = time.time()
        for off in range(0, len(order), bs):
            idx = order[off:off + bs]
            inp, gt = torch.from_numpy(x[idx]).to(device), torch.from_numpy(y[idx]).float().to(device)
            inp, gt = augment(inp, gt)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(inp)
                bce = F.binary_cross_entropy_with_logits(logits, gt)
                prob = logits.float().sigmoid()
                dice = (2 * (prob * gt).sum((1, 2, 3)) + 1.) / (prob.sum((1, 2, 3)) + gt.sum((1, 2, 3)) + 1.)
                loss = .5 * bce + .5 * (1 - dice.mean())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
            if args.pilot and off >= bs * 9:
                break
        score, percase = validation(model, x, y, cases, vi, device, bs)
        elapsed = time.time() - epoch_start
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val_patient_dice": score,
               "seconds": elapsed, "train_slices": len(ti), "val_slices": len(vi),
               "max_cuda_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0}
        if args.pilot:
            row["pilot_train_steps"] = len(losses)
            write_json(args.run / "pilot.json", row)
            print(json.dumps(row), flush=True)
            return
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "architecture": "small_unet2d_groupnorm_base16",
                        "epoch": epoch, "val_patient_dice": score, "membership": membership,
                        "protocol_sha256": digest(args.run / "protocol.json")}, folder / "best.pt")
            write_json(folder / "best_validation.json", percase)
        else:
            stale += 1
        row.update(best=best, best_epoch=best_epoch, stale=stale)
        with (folder / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "best": best, "best_epoch": best_epoch, "stale": stale, "epoch": epoch,
                    "random_state": random.getstate(), "numpy_state": np.random.get_state(),
                    "torch_state": torch.get_rng_state(), "cuda_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}, lastpath)
        write_json(folder / "progress.json", row)
        print(f"{args.model} epoch {epoch:03d}: loss={row['loss']:.4f} valDice={score:.4f} best={best:.4f}@{best_epoch} {elapsed:.1f}s", flush=True)
        if stale >= protocol["patience"]:
            break
    write_json(folder / "complete.json", {"best_epoch": best_epoch, "best_val_patient_dice": best,
               "epochs_completed": epoch, "seconds_this_invocation": time.time() - total_start,
               "best_sha256": digest(folder / "best.pt"), "parameters": sum(p.numel() for p in model.parameters()),
               "membership": membership})


@torch.inference_mode()
def predict(args):
    protocol = json.loads((args.run / "protocol.json").read_text(encoding="utf-8"))
    cases = json.loads((args.run / "cases.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure(1234)
    models, modelhash = {}, {}
    for name in protocol["models"]:
        folder = args.run / "models" / name
        if not (folder / "complete.json").exists():
            raise RuntimeError(f"Training incomplete: {name}")
        state = torch.load(folder / "best.pt", map_location=device, weights_only=False)
        model = SmallUNet().to(device).eval()
        model.load_state_dict(state["model"])
        models[name] = model
        modelhash[name] = digest(folder / "best.pt")
    predictions = {}
    for case, meta in cases.items():
        name = next((n for n, m in protocol["models"].items() if case in m["held_out_train"]), "final")
        assert case not in protocol["models"][name]["train"]
        # Prediction reads C0 and geometry only. Reference labels are not loaded here.
        image = nib.load(meta["paths"]["C0"])
        arr = resample_xy(image, tuple(meta["grid_1mm_shape"]), np.array(meta["grid_1mm_affine"]), 1)
        arr, _ = normalize(arr)
        inp, box = letterbox(arr)
        assert box == meta["letterbox"]
        batches = []
        for off in range(0, len(inp), 4):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                p = models[name](torch.from_numpy(inp[off:off + 4]).to(device)).float().sigmoid()
            batches.append(p.cpu())
        probabilities = unletterbox(torch.cat(batches), box)
        center, mask, status = predicted_center(probabilities >= protocol["threshold"])
        folder = args.run / "predictions" / meta["center"] / case
        save_nii(folder / f"{case}_predicted_myocardium.nii.gz", mask, np.array(meta["grid_1mm_affine"]), True)
        predictions[case] = {"case": case, "center_hw_1mm": center.tolist(), "model": name,
                            "model_sha256": modelhash[name], "training_patients": protocol["models"][name]["train"],
                            "threshold": protocol["threshold"], **status}
        write_json(folder / "localization.json", predictions[case])
        print(f"Localized {case} using {name}: center={center.tolist()} empty={status['empty_prediction']}", flush=True)
    write_json(args.run / "predicted_centers.json", predictions)
    write_json(args.run / "prediction_lock.json", {"centers_sha256": digest(args.run / "predicted_centers.json"),
               "protocol_sha256": digest(args.run / "protocol.json"), "models_sha256": modelhash,
               "prediction_uses": "C0 MRI and geometric metadata only; reference labels are not consumed"})


def export(args):
    protocol = json.loads((args.run / "protocol.json").read_text(encoding="utf-8"))
    cases = json.loads((args.run / "cases.json").read_text(encoding="utf-8"))
    centers = json.loads((args.run / "predicted_centers.json").read_text(encoding="utf-8"))
    lock = json.loads((args.run / "prediction_lock.json").read_text(encoding="utf-8"))
    assert digest(args.run / "predicted_centers.json") == lock["centers_sha256"]
    for dest in DESTINATIONS.values():
        if dest.exists():
            raise FileExistsError(f"Refusing to overwrite {dest}")
    for side, dest in DESTINATIONS.items():
        dest.mkdir(parents=True)
        write_json(dest / "status.json", {"status": "building", "crop_mm": side})
        write_json(dest / "splits.json", {"dataset_id": dest.name, "patients": protocol["patients"], "source_split_sha256": protocol["split_sha256"]})
    records = {side: [] for side in DESTINATIONS}
    for case, meta in cases.items():
        grid_shape, grid_aff = tuple(meta["grid_1mm_shape"]), np.array(meta["grid_1mm_affine"])
        prediction = centers[case]
        assert case not in prediction["training_patients"]
        refgt = nib.load(meta["paths"]["gd"])
        gt = resample_xy(refgt, grid_shape, grid_aff, 0)
        native_counts = label_counts(np.asarray(refgt.dataobj))
        raw_before = label_counts(gt)
        modalities = {m: resample_xy(nib.load(meta["paths"][m]), grid_shape, grid_aff, 1) for m in ("C0", "LGE", "T2")}
        for side, dest in DESTINATIONS.items():
            folder = dest / "cases" / meta["center"] / case
            folder.mkdir(parents=True)
            cropped, aff, cropinfo = crop_fixed(gt, grid_aff, prediction["center_hw_1mm"], side)
            crop_counts = label_counts(cropped)
            output_gt, out_aff = resize_roi(cropped, aff, True)
            save_nii(folder / f"{case}_gd.nii.gz", output_gt, out_aff, True)
            norms = {}
            for m, array in modalities.items():
                roi, image_aff, _ = crop_fixed(array, grid_aff, prediction["center_hw_1mm"], side)
                roi, image_aff = resize_roi(roi, image_aff)
                assert np.allclose(image_aff, out_aff)
                roi, norms[m] = normalize(roi)
                save_nii(folder / f"{case}_{m}.nii.gz", roi, out_aff)
            native = dest / "reference_native" / meta["center"] / case / f"{case}_gd.nii.gz"
            native.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(meta["paths"]["gd"], native)
            localization_dir = dest / "localization" / meta["center"] / case
            localization_dir.mkdir(parents=True, exist_ok=True)
            for name in (f"{case}_predicted_myocardium.nii.gz", "localization.json"):
                shutil.copy2(args.run / "predictions" / meta["center"] / case / name, localization_dir / name)
            lost = {k: raw_before[k] - crop_counts[k] for k in raw_before}
            assert all(n >= 0 for n in lost.values())
            metadata = {**meta, "case": case, "localization": "predicted_myocardium_unet2d_oof_train_final_val_test",
                        "localizer": prediction, "prediction_lock_sha256": digest(args.run / "prediction_lock.json"),
                        "crop_mm": side, "crop": cropinfo, "crop_affine": aff.tolist(),
                        "output_shape": list(output_gt.shape), "output_affine": out_aff.tolist(),
                        "output_spacing_mm": nib.affines.voxel_sizes(out_aff).tolist(),
                        "normalization": norms, "normalization_method": "positive-voxel p1/p99 per modality and whole output volume after crop/resize",
                        "labels": "raw_0_200_500_600_1220_2221", "counts_before_crop_1mm": raw_before,
                        "native_label_counts": native_counts,
                        "classes_disappeared_during_resampling": [k for k, n in native_counts.items() if n and not raw_before[k]],
                        "counts_after_crop_before_resize": crop_counts, "crop_loss_voxels_1mm": lost,
                        "counts_after_resize": label_counts(output_gt), "classes_disappeared_due_to_resize":
                        [k for k, n in crop_counts.items() if n and not np.any(output_gt == int(k))],
                        "clipping_policy": "fixed window; all clipped cases retained; no GT correction"}
            write_json(folder / "metadata.json", metadata)
            records[side].append({"case": case, "center": meta["center"], "split": meta["split"],
                                  "localizer_model": prediction["model"], "empty_prediction": prediction["empty_prediction"],
                                  "crop_loss_voxels": lost, "before": raw_before, "after_crop": crop_counts})
        print(f"Exported both fixed ROIs: {case}", flush=True)
    for side, dest in DESTINATIONS.items():
        repro = dest / "reproducibility"
        repro.mkdir()
        shutil.copy2(__file__, repro / Path(__file__).name)
        for fn in ("protocol.json", "environment.json", "cases.json", "predicted_centers.json", "prediction_lock.json"):
            shutil.copy2(args.run / fn, repro / fn)
        for name in protocol["models"]:
            folder = repro / "models" / name
            folder.mkdir(parents=True)
            for fn in ("best.pt", "complete.json", "history.jsonl", "best_validation.json"):
                shutil.copy2(args.run / "models" / name / fn, folder / fn)
        manifest = {"dataset_id": dest.name, "patient_count": 80, "patients": sorted(cases), "crop_mm": side,
                    "model_hw": [128, 128], "output_inplane_spacing_mm": side / 128,
                    "localization": "predicted U-Net2D; train 3-fold OOF; val/test final model trained on 51 train patients",
                    "source": str(args.source), "shared_localizer_run": str(args.run), "excluded_cases": [],
                    "geometry": "XY resample1mm, fixed physical crop, output128, native Z retained",
                    "threshold": .5, "labels": list(RAW_CODES), "cases": records[side],
                    "protocol_sha256": digest(args.run / "protocol.json"), "prediction_lock": lock,
                    "normalization": "p1/p99 after crop/resize; [0,1]",
                    "evaluation": "For end-to-end evaluation, restore probabilities to native grid before argmax; outside ROI background."}
        write_json(dest / "dataset_manifest.json", manifest)
        write_json(dest / "status.json", {"status": "exported_pending_verification", "crop_mm": side})
        text = f"""# MyoPS++ predicted U-Net2D ROI: {side} mm -> 128 pixels

80 cases (B35/C45); fixed patient split 51 train / 13 validation / 16 test.
Localization uses C0 only, U-Net2D: out-of-fold models for train, final train-only model for val/test.
GT never determines prediction center. Fixed threshold 0.5; largest component per slice; common volume bbox midpoint.
Empty predicted mask falls back to image center with the SAME fixed crop size and is reported.
All clipped cases are retained. No GT-guided corrections or variable FOV.

XY spacing: {side / 128} x {side / 128} mm; native Z spacing retained.
Images float32 [0,1], per-modality whole-output positive-voxel p1/p99 normalization.
Labels uint16 raw: 0 background; 200 normal myocardium; 500 LV; 600 RV; 1220 edema; 2221 scar.
SCAR mapping: 0/500/600 -> 0; 200 -> 1; 1220 -> 2; 2221 -> 3.

cases/: three MRI modalities, raw GT and complete crop geometry/provenance.
reference_native/: original GT for full-image physical-distance evaluation.
localization/: predicted myocardium masks on the 1mm grid and locked center provenance.
reproducibility/: source, model checkpoints, protocol, folds, locked predicted centers and logs.
This is a new prediction-localized export, not the older oracle SCAR ROI profile. Do not relabel it as an oracle cache.
Full-cohort GT geometry was explored before this run; no claim of an untouched exploratory test cohort.
Coverage measures crop loss on the 1mm grid, separately from label changes introduced by resampling/resize.
"""
        (dest / "README.md").write_text(text, encoding="utf-8")


def verify(args):
    cases = json.loads((args.run / "cases.json").read_text(encoding="utf-8"))
    source_ok = 0
    for meta in cases.values():
        for m, p in meta["paths"].items():
            if digest(p) != meta["source_sha256"][m]:
                raise ValueError(f"Source changed: {p}")
            source_ok += 1
    for side, dest in DESTINATIONS.items():
        checked, shape_errors = 0, []
        summary = {sp: {"n": 0, **{g + "_clipped": 0 for g in TARGETS}, "empty_predictions": 0} for sp in ("train", "val", "test_vol", "all")}
        details = []
        for case, source in cases.items():
            folder = dest / "cases" / source["center"] / case
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            expected_aff = np.array(meta["output_affine"])
            expected_spacing = (side / 128, side / 128, source["native_spacing_mm"][2])
            for m in ("C0", "LGE", "T2", "gd"):
                image = nib.load(str(folder / f"{case}_{m}.nii.gz"))
                array = np.asarray(image.dataobj)
                assert image.shape == (128, 128, source["native_shape"][2])
                assert np.allclose(image.affine, expected_aff, atol=1e-4, rtol=1e-5)
                assert np.allclose(nib.affines.voxel_sizes(image.affine), expected_spacing, atol=1e-4)
                assert image.header.get_xyzt_units()[0] == "mm"
                assert np.isfinite(array).all()
                if m == "gd":
                    assert array.dtype == np.uint16 and set(np.unique(array)).issubset(RAW_CODES)
                else:
                    assert array.dtype == np.float32 and array.min() >= 0 and array.max() <= 1
                checked += 1
            native = dest / "reference_native" / source["center"] / case / f"{case}_gd.nii.gz"
            assert digest(native) == source["source_sha256"]["gd"]
            pred = meta["localizer"]
            assert case not in pred["training_patients"]
            checkpoint = dest / "reproducibility" / "models" / pred["model"] / "best.pt"
            checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert case not in checkpoint_state["membership"]["train"]
            assert digest(checkpoint) == pred["model_sha256"]
            localization_path = dest / "localization" / source["center"] / case / f"{case}_predicted_myocardium.nii.gz"
            predicted_mask = np.asarray(nib.load(str(localization_path)).dataobj) > 0
            independently_centered, _, _ = predicted_center(predicted_mask)
            assert independently_centered.tolist() == pred["center_hw_1mm"]
            grid_aff = np.array(meta["grid_1mm_affine"])
            source_gt = resample_xy(nib.load(source["paths"]["gd"]), tuple(meta["grid_1mm_shape"]), grid_aff, 0)
            # Independent crop bounds/counts; do not trust exported coverage metadata.
            start = np.array(pred["center_hw_1mm"]) - side // 2
            stop = start + side
            lo, hi = np.maximum(start, 0), np.minimum(stop, source_gt.shape[:2])
            independently_cropped = np.zeros((side, side, source_gt.shape[2]), dtype=np.uint16)
            if np.all(hi > lo):
                offset = lo - start
                independently_cropped[offset[0]:offset[0] + hi[0] - lo[0], offset[1]:offset[1] + hi[1] - lo[1]] = source_gt[lo[0]:hi[0], lo[1]:hi[1]]
            assert label_counts(source_gt) == meta["counts_before_crop_1mm"]
            assert label_counts(independently_cropped) == meta["counts_after_crop_before_resize"]
            # nearest-exact chooses floor((j+.5)*scale), checked without PyTorch.
            sample_idx = np.floor((np.arange(128) + .5) * side / 128).astype(int)
            expected_label = independently_cropped[sample_idx[:, None], sample_idx[None, :], :]
            exported_gt = np.asarray(nib.load(str(folder / f"{case}_gd.nii.gz")).dataobj)
            assert np.array_equal(expected_label, exported_gt)
            gt_myo = np.isin(source_gt, (200, 1220, 2221))
            overlap = np.count_nonzero(gt_myo & predicted_mask)
            predicted_dice = 2 * overlap / (np.count_nonzero(gt_myo) + np.count_nonzero(predicted_mask))
            gtpos = np.argwhere(gt_myo)
            oracle_center = (gtpos[:, :2].min(0) + gtpos[:, :2].max(0)) / 2
            center_error = float(np.linalg.norm(np.array(pred["center_hw_1mm"]) - oracle_center))
            before, lost = meta["counts_before_crop_1mm"], meta["crop_loss_voxels_1mm"]
            row = {"case": case, "center": source["center"], "split": source["split"], "crop_mm": side,
                   "localizer_model": pred["model"], "empty_prediction": pred["empty_prediction"],
                   "center_h": pred["center_hw_1mm"][0], "center_w": pred["center_hw_1mm"][1],
                   "localizer_dice_on_1mm_grid": float(predicted_dice), "center_error_to_gt_bbox_mm": center_error,
                   "resampling_disappeared_classes": ",".join(meta["classes_disappeared_during_resampling"]),
                   "resize_disappeared_classes": ",".join(meta["classes_disappeared_due_to_resize"])}
            for g, labels in TARGETS.items():
                n = sum(before[str(k)] for k in labels)
                count = sum(lost[str(k)] for k in labels)
                row[g + "_voxels_before"] = n
                row[g + "_lost_voxels"] = count
                row[g + "_lost_percent"] = 100 * count / n if n else 0.
            for split in (source["split"], "all"):
                summary[split]["n"] += 1
                summary[split]["empty_predictions"] += pred["empty_prediction"]
                for g in TARGETS:
                    summary[split][g + "_clipped"] += row[g + "_lost_voxels"] > 0
            details.append(row)
        with (dest / "coverage_per_case.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(details[0]))
            w.writeheader()
            w.writerows(details)
        result = {"status": "verified", "patient_count": len(cases), "checked_output_nifti": checked,
                  "verified_native_gt_copies": 80, "verified_unchanged_source_files": source_ok,
                  "split_summary": summary, "affected_cases": {g: [r["case"] for r in details if r[g + "_lost_voxels"] > 0] for g in TARGETS},
                  "worst_pathology_loss_percent": max(r["pathology_lost_percent"] for r in details),
                  "localizer_patient_mean_dice_1mm_by_split": {sp: float(np.mean([r["localizer_dice_on_1mm_grid"] for r in details if sp == "all" or r["split"] == sp])) for sp in ("train", "val", "test_vol", "all")},
                  "center_error_mean_mm": float(np.mean([r["center_error_to_gt_bbox_mm"] for r in details])),
                  "resampling_disappeared_cases": [r["case"] for r in details if r["resampling_disappeared_classes"]],
                  "resize_disappeared_cases": [r["case"] for r in details if r["resize_disappeared_classes"]]}
        write_json(dest / "verification_report.json", result)
        write_json(dest / "status.json", {"status": "verified", "crop_mm": side})
        with (dest / "checksums.sha256").open("w", encoding="utf-8") as f:
            for p in sorted(dest.rglob("*")):
                if p.is_file() and p.name != "checksums.sha256":
                    f.write(f"{digest(p)}  {p.relative_to(dest).as_posix()}\n")
        print(f"VERIFIED {dest}: {json.dumps(summary['all'])}", flush=True)
    make_report(args)


def make_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = json.loads((args.run / "cases.json").read_text(encoding="utf-8"))
    centers = json.loads((args.run / "predicted_centers.json").read_text(encoding="utf-8"))
    reports = {side: json.loads((dest / "verification_report.json").read_text(encoding="utf-8")) for side, dest in DESTINATIONS.items()}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("fold0", "fold1", "fold2", "final"):
        history = [json.loads(line) for line in (args.run / "models" / name / "history.jsonl").read_text(encoding="utf-8").splitlines()]
        axes[0].plot([r["epoch"] for r in history], [r["loss"] for r in history], label=name, linewidth=1)
        axes[1].plot([r["epoch"] for r in history], [r["val_patient_dice"] for r in history], label=name, linewidth=1)
    for ax, ylabel in zip(axes, ("Training BCE/Dice loss", "Validation patient-mean Dice")):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.run / "learning_curves.png", dpi=150)
    plt.close(fig)
    chosen = ["Case2013", "Case2017", "Case2018", "Case2031"]
    for case in reports[128]["affected_cases"]["pathology"]:
        if case not in chosen:
            chosen.append(case)
    chosen = chosen[:8]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(10, 3 * len(chosen)), squeeze=False)
    for row, case in enumerate(chosen):
        m = cases[case]
        shape, aff = tuple(m["grid_1mm_shape"]), np.array(m["grid_1mm_affine"])
        c0, _ = normalize(resample_xy(nib.load(m["paths"]["C0"]), shape, aff, 1))
        gt = resample_xy(nib.load(m["paths"]["gd"]), shape, aff, 0)
        pathology = np.isin(gt, (1220, 2221))
        z = int(pathology.sum((0, 1)).argmax())
        pred_path = args.run / "predictions" / m["center"] / case / f"{case}_predicted_myocardium.nii.gz"
        pmask = np.asarray(nib.load(str(pred_path)).dataobj)
        ax = axes[row, 0]
        ax.imshow(c0[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1)
        for mask, color in [(np.isin(gt[:, :, z], (200, 1220, 2221)), "lime"), (pmask[:, :, z] > 0, "cyan")]:
            if mask.any() and not mask.all():
                ax.contour(mask.T, levels=[.5], colors=[color], linewidths=.8)
        center = np.array(centers[case]["center_hw_1mm"])
        for side, color in [(160, "yellow"), (128, "red")]:
            start = center - side // 2
            ax.add_patch(plt.Rectangle(start, side, side, fill=False, edgecolor=color, linewidth=1))
        ax.set_title(f"{case} | slice {z}\nGT green / U-Net cyan; A yellow / B red", fontsize=9)
        for col, side in [(1, 160), (2, 128)]:
            folder = DESTINATIONS[side] / "cases" / m["center"] / case
            img = np.asarray(nib.load(str(folder / f"{case}_C0.nii.gz")).dataobj)
            lab = np.asarray(nib.load(str(folder / f"{case}_gd.nii.gz")).dataobj)
            axes[row, col].imshow(img[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1)
            mask = np.isin(lab[:, :, z], (200, 1220, 2221))
            if mask.any() and not mask.all():
                axes[row, col].contour(mask.T, levels=[.5], colors=["lime"], linewidths=.8)
            axes[row, col].set_title(f"{side} mm -> 128 pixels\nXY spacing {side / 128:g} mm", fontsize=10)
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.run / "crop_comparison.png", dpi=130)
    plt.close(fig)
    lines = ["# MyoPS++ prediction-localized ROI export report", "", "Both fixed windows were executed without GT-guided recentering, enlargement or case exclusions.", "",
             "| Variant | XY spacing | Any label clipped | Myocardium clipped | Pathology clipped | Empty masks |", "|---|---:|---:|---:|---:|---:|"]
    for side in (160, 128):
        s = reports[side]["split_summary"]["all"]
        lines.append(f"| {side}mm -> 128px | {side/128:g}mm | {s['all_labels_clipped']}/80 | {s['myocardium_clipped']}/80 | {s['pathology_clipped']}/80 | {s['empty_predictions']} |")
    lines += ["", "## Localizer models", "", "| Model | Training patients | Best epoch | Validation patient Dice | Epochs completed |", "|---|---:|---:|---:|---:|"]
    for name in ("fold0", "fold1", "fold2", "final"):
        s = json.loads((args.run / "models" / name / "complete.json").read_text(encoding="utf-8"))
        lines.append(f"| {name} | {len(s['membership']['train'])} | {s['best_epoch']} | {s['best_val_patient_dice']:.4f} | {s['epochs_completed']} |")
    lines += ["", "## Split-specific coverage", ""]
    for side in (160, 128):
        lines += [f"### {side} mm", "", "| Split | N | Myocardium clipped | Pathology clipped |", "|---|---:|---:|---:|"]
        for sp in ("train", "val", "test_vol"):
            s = reports[side]["split_summary"][sp]
            lines.append(f"| {sp} | {s['n']} | {s['myocardium_clipped']} | {s['pathology_clipped']} |")
        lines += ["", "Pathology-clipped cases: " + (", ".join(reports[side]["affected_cases"]["pathology"]) or "none"),
                  f"Worst pathology crop loss: {reports[side]['worst_pathology_loss_percent']:.4f}%.", ""]
    lines += ["## Method and scope", "", "- C0-only binary myocardium localizer, raw labels {200,1220,2221}. Train ROIs use three-fold patient-level OOF predictions. Validation/test use a final localizer fitted on the 51 training patients.",
              "- Fixed threshold 0.5; largest 8-connected component per slice; one predicted bbox midpoint per volume. An empty prediction uses the image center with unchanged crop dimensions.",
              "- All 80 cases retained, including clipping/failures. No GT is consumed by the prediction stage; predicted centers are hashed and locked before exporting labels.",
              "- 1mm in-plane resampling only; no interpolation across Z. Labels use nearest-neighbor. 160->128 uses nearest-exact for labels and bilinear for images with a consistent pixel-center affine.",
              "- Crop coverage is measured before resize on the same 1mm label grid. It does not measure interpolation loss relative to native GT, and is not downstream lesion-segmentation accuracy.",
              "- Normalize output images with per-modality whole-output positive-voxel p1/p99 to [0,1]. Keep raw GT codes and native references.",
              "- Whole-cohort GT geometry was explored in earlier audits. This report does not claim an untouched exploratory test cohort or establish external clinical generalization.",
              "- New exports have not been integrated into existing SCAR oracle profiles. Full-image evaluation should restore probabilities to native geometry before argmax and use native physical spacing.", "",
              "## Deliverables", "", *[f"- {p}" for p in DESTINATIONS.values()], f"- Shared localizer run: {args.run}", "",
              "![Prediction contours and both fixed crops](crop_comparison.png)", "",
              "![Localizer training and validation curves](learning_curves.png)", ""]
    (args.run / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(args.run / "completion.json", {"status": "complete", "exports": {str(s): str(p) for s, p in DESTINATIONS.items()},
               "source_files_unchanged": 320, "reports": reports})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "predict", "export", "verify", "report"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--model", choices=("fold0", "fold1", "fold2", "final"), default="final")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "predict": predict, "export": export, "verify": verify, "report": make_report}[args.stage](args)


if __name__ == "__main__":
    main()
