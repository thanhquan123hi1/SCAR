"""Batched 3D volume inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F


@torch.inference_mode()
def predict_volume(model, images, img_size=128, batch_size=8, device=None, amp_dtype=None):
    """Aligned H,W,D volumes -> H,W,D labels, batching adjacent slices.

    Resize images bilinearly to model grid; resize logits back before argmax.
    Each batch uses one forward pass; output masks are never filtered.
    """
    arrays = [np.asarray(value, dtype=np.float32) for value in images]
    if len(arrays) != 3 or any(a.ndim != 3 for a in arrays) or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Expected three aligned H,W,D arrays.")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Non-finite inference inputs.")
    if batch_size < 1 or img_size < 32 or img_size % 16 or any(v < 1 for v in arrays[0].shape):
        raise ValueError("Use nonempty volumes, positive batch_size and img_size >=32 divisible by 16.")
    device = torch.device(device or next(model.parameters()).device)
    height, width, depth = arrays[0].shape
    prediction = np.empty((height, width, depth), dtype=np.uint8)
    was_training = model.training
    model.eval()
    try:
        for start in range(0, depth, batch_size):
            stop = min(start + batch_size, depth)
            inputs = [
                torch.from_numpy(np.ascontiguousarray(a[:, :, start:stop].transpose(2, 0, 1))).unsqueeze(1).to(device)
                for a in arrays
            ]
            inputs = [
                F.interpolate(x, (img_size, img_size), mode="bilinear", align_corners=False)
                if x.shape[-2:] != (img_size, img_size)
                else x
                for x in inputs
            ]
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(*inputs)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite inference logits.")
            if logits.shape[-2:] != (height, width):
                logits = F.interpolate(logits.float(), (height, width), mode="bilinear", align_corners=False)
            prediction[:, :, start:stop] = logits.argmax(1).cpu().numpy().transpose(1, 2, 0)
    finally:
        model.train(was_training)
    return prediction


def main(argv=None):
    """Predict on raw, aligned NIfTI and preserve its native reference grid."""
    import nibabel as nib
    from ml_collections import ConfigDict
    from preprocessing.preprocessing import MODALITIES, load_aligned_images
    from training.models import model_from_config
    from training.trainer.trainer import load_checkpoint, resolve_device, resolve_amp, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cine", required=True, help="Aligned bSSFP NIfTI")
    parser.add_argument("--lge", required=True)
    parser.add_argument("--t2w", required=True)
    parser.add_argument("--output", required=True, help="New prediction .nii.gz path")
    parser.add_argument("--normalization", required=True, choices=("unit255", "unit", "percentile"),
                        help="Use the normalization chosen when packaging training data")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("auto", "none", "fp16", "bf16"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch-size and cpu-threads must be positive")
    output = Path(args.output).resolve()
    if not (output.name.endswith(".nii.gz") or output.suffix == ".nii"):
        parser.error("output must end in .nii or .nii.gz")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    paths = dict(zip(MODALITIES, (args.cine, args.lge, args.t2w)))
    images, spacing, affine, unit = load_aligned_images(paths, args.normalization)
    checkpoint = load_checkpoint(args.checkpoint)
    recorded = checkpoint.get("data_provenance", {}).get("metadata", {}).get("normalization")
    if recorded and recorded != args.normalization:
        raise ValueError(f"Normalization differs from training: {recorded}")
    torch.set_num_threads(args.cpu_threads)
    device = resolve_device(args.device)
    amp = resolve_amp(args.amp, device)
    model = model_from_config(ConfigDict(checkpoint["model_config"]), img_size=checkpoint["args"]["img_size"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    prediction = predict_volume(model, [images[m] for m in MODALITIES],
                                checkpoint["args"]["img_size"], args.batch_size, device, amp)
    reference = nib.load(args.cine)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_intent("label")
    header["descrip"] = b"SCAR canonical: 0 background, 1 normal, 2 edema, 3 scar"
    result = nib.Nifti1Image(prediction, affine, header)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, str(output))
    write_json(output.with_suffix(output.suffix + ".json"), {
        "checkpoint": str(Path(args.checkpoint).resolve()), "normalization": args.normalization,
        "class_names": list(checkpoint["class_names"]), "shape": list(prediction.shape),
        "spatial_unit": unit, "native_spacing": spacing.tolist(),
        "affine": affine.tolist(), "inputs": {k: str(Path(v).resolve()) for k, v in paths.items()},
    })
    print(f"Saved {output} | shape={prediction.shape} | native unit={unit}")
    return output


if __name__ == "__main__":
    main()
