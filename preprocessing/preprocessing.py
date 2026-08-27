"""Invertible array preprocessing for LGE cardiac MRI segmentation.

Adapted from cmr-multi-cinema (sinaamirrajab/cmr-multi-cinema).
Original: src/cmr_multi/data/preprocessing.py

Dependencies: numpy, scipy only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.ndimage import generate_binary_structure, label, zoom


# ---------------------------------------------------------------------------
# Transform metadata (for invertible pipeline)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CenterTransform:
    """Center crop/pad metadata for exact index-space inversion."""

    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    crop_start: tuple[int, ...]
    crop_stop: tuple[int, ...]
    pad_lower: tuple[int, ...]
    pad_upper: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpatialTransform:
    """Resize and center-crop transform metadata."""

    original_shape: tuple[int, ...]
    resized_shape: tuple[int, ...]
    source_spacing: tuple[float, ...]
    target_spacing: tuple[float, ...]
    center: CenterTransform

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------

def shape_for_spacing(
    shape: tuple[int, ...],
    source_spacing: tuple[float, ...],
    target_spacing: tuple[float, ...],
) -> tuple[int, ...]:
    """Calculate the rounded output shape for spacing-preserving resampling."""

    if not (len(shape) == len(source_spacing) == len(target_spacing)):
        raise ValueError("Shape and spacing dimensions must agree.")
    return tuple(
        max(1, int(round(size * source / target)))
        for size, source, target in zip(shape, source_spacing, target_spacing, strict=True)
    )


def resize_to_shape(
    array: np.ndarray,
    target_shape: tuple[int, ...],
    *,
    order: int,
) -> np.ndarray:
    """Resize an array to the requested shape.

    Args:
        array: Input array.
        target_shape: Desired output shape.
        order: Interpolation order. Use 1 (bilinear) for images, 0 (nearest) for masks.
    """

    source = np.asarray(array)
    if source.ndim != len(target_shape):
        raise ValueError(
            f"Cannot resize {source.ndim}D array to {len(target_shape)}D shape."
        )
    if any(value <= 0 for value in target_shape):
        raise ValueError(f"Target shape must be positive, found {target_shape}.")
    if source.shape == target_shape:
        return source.copy()
    factors = tuple(
        target / current
        for current, target in zip(source.shape, target_shape, strict=True)
    )
    resized = zoom(source, zoom=factors, order=order, mode="nearest", prefilter=order > 1)
    if resized.shape == target_shape:
        return resized
    corrected, _transform = center_crop_or_pad(resized, target_shape)
    return corrected


def center_crop_or_pad(
    array: np.ndarray,
    target_shape: tuple[int, ...],
    *,
    value: float | int = 0,
) -> tuple[np.ndarray, CenterTransform]:
    """Center crop and/or pad to target_shape without interpolation."""

    source = np.asarray(array)
    if source.ndim != len(target_shape):
        raise ValueError(
            f"Array shape {source.shape} is incompatible with target {target_shape}."
        )
    crop_start = tuple(
        max((current - target) // 2, 0)
        for current, target in zip(source.shape, target_shape, strict=True)
    )
    crop_stop = tuple(
        start + min(current, target)
        for start, current, target in zip(
            crop_start, source.shape, target_shape, strict=True
        )
    )
    slices = tuple(slice(start, stop) for start, stop in zip(crop_start, crop_stop, strict=True))
    cropped = source[slices]
    pad_lower = tuple(
        max((target - current) // 2, 0)
        for current, target in zip(cropped.shape, target_shape, strict=True)
    )
    pad_upper = tuple(
        target - current - lower
        for current, target, lower in zip(
            cropped.shape, target_shape, pad_lower, strict=True
        )
    )
    padded = np.pad(
        cropped,
        tuple(zip(pad_lower, pad_upper, strict=True)),
        mode="constant",
        constant_values=value,
    )
    transform = CenterTransform(
        source_shape=tuple(int(item) for item in source.shape),
        target_shape=tuple(int(item) for item in target_shape),
        crop_start=crop_start,
        crop_stop=crop_stop,
        pad_lower=pad_lower,
        pad_upper=pad_upper,
    )
    return padded, transform


def invert_center_crop_or_pad(
    array: np.ndarray,
    transform: CenterTransform,
    *,
    value: float | int = 0,
) -> np.ndarray:
    """Undo center crop/pad, filling cropped-away regions with ``value``."""

    current = np.asarray(array)
    if current.shape != transform.target_shape:
        raise ValueError(
            f"Expected transformed shape {transform.target_shape}, found {current.shape}."
        )
    unpad_slices = tuple(
        slice(lower, current.shape[index] - upper if upper else None)
        for index, (lower, upper) in enumerate(
            zip(transform.pad_lower, transform.pad_upper, strict=True)
        )
    )
    unpadded = current[unpad_slices]
    restored = np.full(transform.source_shape, value, dtype=current.dtype)
    restore_slices = tuple(
        slice(start, stop)
        for start, stop in zip(transform.crop_start, transform.crop_stop, strict=True)
    )
    restored[restore_slices] = unpadded
    return restored


# ---------------------------------------------------------------------------
# Intensity normalization
# ---------------------------------------------------------------------------

def extract_tissue_foreground(array: np.ndarray) -> np.ndarray:
    """Extract tissue foreground voxels by filtering out low-intensity MRI air background noise.
    
    Uses Otsu-based adaptive thresholding to separate low-intensity air background from
    valid tissue, protecting dark nulled myocardium while cleanly cutting air noise (fixes W2).
    """
    source = np.asarray(array, dtype=np.float32)
    if source.size == 0:
        return source
    s_min, s_max = float(np.min(source)), float(np.max(source))
    if s_max <= s_min + 1e-6:
        return source

    # Otsu thresholding over 256 bins
    scaled = np.clip((source - s_min) / (s_max - s_min) * 255.0, 0, 255).astype(np.uint8)
    hist = np.bincount(scaled.ravel(), minlength=256).astype(np.float64)
    total = float(scaled.size)

    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0.0
    between_vars = np.zeros(256, dtype=np.float64)

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between_vars[t] = w_b * w_f * ((m_b - m_f) ** 2)

    max_val = float(np.max(between_vars))
    if max_val > 0:
        # If there is a flat plateau between clusters, select the midpoint
        plateau = np.where(np.isclose(between_vars, max_val, rtol=1e-3))[0]
        threshold = int(np.mean(plateau))
    else:
        threshold = 32

    # Convert threshold back to original scale (with conservative 0.5 factor to preserve dark myocardium)
    otsu_val = s_min + (threshold / 255.0) * (s_max - s_min)
    noise_floor = max(s_min, s_min + (otsu_val - s_min) * 0.5)

    fg = source[source > noise_floor]
    return fg if fg.size > 100 else source


def percentile_minmax(
    array: np.ndarray,
    *,
    lower: float = 0.5,
    upper: float = 99.5,
) -> np.ndarray:
    """Percentile clipping on tissue foreground followed by [0, 1] min-max scaling."""

    source = np.asarray(array, dtype=np.float32)
    fg = extract_tissue_foreground(source)
    low, high = np.percentile(fg, (lower, upper))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(source, dtype=np.float32)
    clipped = np.clip(source, low, high)
    return ((clipped - low) / (high - low)).astype(np.float32, copy=False)


def minmax(array: np.ndarray) -> np.ndarray:
    """Scale an array to [0, 1] with a constant-image guard."""

    source = np.asarray(array, dtype=np.float32)
    low = float(np.min(source))
    high = float(np.max(source))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(source, dtype=np.float32)
    return ((source - low) / (high - low)).astype(np.float32, copy=False)


def zscore(
    array: np.ndarray,
    *,
    clip_percentiles: tuple[float, float] | None = (0.5, 99.5),
) -> np.ndarray:
    """Z-score normalization (zero mean, unit variance) on tissue foreground.
    
    Standardized according to nnUNet protocol for non-CT medical modalities (CMR).
    """
    source = np.asarray(array, dtype=np.float32)
    fg = extract_tissue_foreground(source)
    if clip_percentiles is not None:
        low, high = np.percentile(fg, clip_percentiles)
        if np.isfinite(low) and np.isfinite(high) and high > low:
            source = np.clip(source, low, high)
            fg = np.clip(fg, low, high)

    mean_val = float(np.mean(fg))
    std_val = float(np.std(fg))
    if std_val < 1e-8:
        return np.zeros_like(source, dtype=np.float32)
    return ((source - mean_val) / std_val).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Combined spatial preprocessing (main entry point)
# ---------------------------------------------------------------------------

def preprocess_spatial(
    array: np.ndarray,
    *,
    source_spacing: tuple[float, ...],
    target_spacing: tuple[float, ...],
    target_shape: tuple[int, ...],
    interpolation_order: int = 1,
    intensity_percentiles: tuple[float, float] | None = None,
    precomputed_intensity_bounds: tuple[float, float] | None = None,
    mode: str = "percentile_minmax",
) -> tuple[np.ndarray, SpatialTransform]:
    """Normalize → resample → center crop/pad an image.

    Args:
        array: Input numpy array (image or mask).
        source_spacing: Pixel/voxel spacing from NIfTI header (mm).
        target_spacing: Desired isotropic spacing (mm).
        target_shape: Output spatial shape after crop/pad.
        interpolation_order: 1 for images (bilinear), 0 for masks (nearest).
        intensity_percentiles: If given, use percentile clipping; else plain minmax.
        precomputed_intensity_bounds: If given as (low, high), use these values
            directly instead of computing percentiles on this array. Essential
            for per-slice processing to maintain consistent normalization.
        mode: 'percentile_minmax' | 'zscore' | 'minmax'.

    Returns:
        (processed_array, SpatialTransform) — transform is needed to invert predictions.
    """

    original_shape = tuple(int(item) for item in np.asarray(array).shape)
    source = np.asarray(array, dtype=np.float32)

    # Normalize BEFORE resample to avoid interpolation of outlier values
    if precomputed_intensity_bounds is not None:
        low, high = precomputed_intensity_bounds
        if np.isfinite(low) and np.isfinite(high) and high > low:
            clipped = np.clip(source, low, high)
            normalized = ((clipped - low) / (high - low)).astype(np.float32, copy=False)
        else:
            normalized = np.zeros_like(source, dtype=np.float32)
    elif mode == "zscore":
        normalized = zscore(source, clip_percentiles=intensity_percentiles)
    elif intensity_percentiles is not None:
        normalized = percentile_minmax(
            source,
            lower=float(intensity_percentiles[0]),
            upper=float(intensity_percentiles[1]),
        )
    else:
        normalized = minmax(source)

    # Resample AFTER normalization
    resized_shape = shape_for_spacing(original_shape, source_spacing, target_spacing)
    resized = resize_to_shape(normalized, resized_shape, order=interpolation_order)

    transformed, center = center_crop_or_pad(resized, target_shape)
    return transformed.astype(np.float32, copy=False), SpatialTransform(
        original_shape=original_shape,
        resized_shape=resized_shape,
        source_spacing=source_spacing,
        target_spacing=target_spacing,
        center=center,
    )


def preprocess_mask(
    mask: np.ndarray,
    *,
    source_spacing: tuple[float, ...],
    target_spacing: tuple[float, ...],
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """Resample + crop/pad a discrete label mask using one-hot continuous resampling and argmax decoding.

    Preserves exact ground truth morphology, maintains class mutual exclusivity,
    eliminates dilation artifacts, and preserves tiny lesion centroids.

    Args:
        mask: Integer label array.
        source_spacing: Voxel spacing from NIfTI header (mm).
        target_spacing: Desired spacing (mm).
        target_shape: Output spatial shape.

    Returns:
        Integer mask as int64, with label values preserved.
    """
    raw_mask = np.rint(mask).astype(np.int16)
    original_shape = tuple(int(item) for item in raw_mask.shape)
    resized_shape = shape_for_spacing(original_shape, source_spacing, target_spacing)

    unique_classes = np.unique(raw_mask)
    if len(unique_classes) <= 1:
        resized = np.full(
            resized_shape,
            unique_classes[0] if unique_classes.size > 0 else 0,
            dtype=np.int16,
        )
    elif raw_mask.shape == resized_shape:
        resized = raw_mask.copy()
    else:
        c_list = sorted(unique_classes.tolist())
        # Construct one-hot continuous representation across all present classes
        one_hot = np.stack([(raw_mask == c).astype(np.float32) for c in c_list], axis=0)

        # Resample each channel continuously (order=1: bilinear/trilinear)
        resampled_channels = [
            resize_to_shape(one_hot[i], resized_shape, order=1)
            for i in range(len(c_list))
        ]
        res_stack = np.stack(resampled_channels, axis=0)  # (K, *resized_shape)

        # Channel-wise argmax decoding: preserves 0.50 natural boundary and mutual exclusivity
        argmax_idx = np.argmax(res_stack, axis=0)
        c_arr = np.array(c_list, dtype=np.int16)
        resized = c_arr[argmax_idx]

        # Connected-component-aware micro-structure peak preservation guard
        structure = generate_binary_structure(raw_mask.ndim, raw_mask.ndim)
        for c in sorted(set(c_list) - {0}):
            c_bin = (raw_mask == c)
            labeled_comp, num_features = label(c_bin, structure=structure)
            for feat_idx in range(1, num_features + 1):
                comp_mask = (labeled_comp == feat_idx).astype(np.float32)
                res_comp = resize_to_shape(comp_mask, resized_shape, order=1)
                if not np.any((resized == c) & (res_comp > 0.0)):
                    max_val = float(np.max(res_comp))
                    if max_val > 0.0:
                        coords = np.argwhere(res_comp == max_val)
                        for coord in coords:
                            resized[tuple(coord)] = c
                    else:
                        orig_coords = np.argwhere(labeled_comp == feat_idx)
                        if orig_coords.size > 0:
                            orig_center = orig_coords.mean(axis=0)
                            target_coord = tuple(
                                int(
                                    np.clip(
                                        np.round(
                                            orig_center[d]
                                            * (resized_shape[d] / original_shape[d])
                                        ),
                                        0,
                                        resized_shape[d] - 1,
                                    )
                                )
                                for d in range(len(resized_shape))
                            )
                            resized[target_coord] = c

    transformed, _ = center_crop_or_pad(resized, target_shape, value=0)
    return np.rint(transformed).astype(np.int64, copy=False)


# ---------------------------------------------------------------------------
# Inverse transform (for restoring predictions to original NIfTI geometry)
# ---------------------------------------------------------------------------

def invert_spatial_mask(
    mask: np.ndarray,
    transform: SpatialTransform,
) -> np.ndarray:
    """Restore a discrete prediction mask to the original NIfTI array shape.

    Uses one-hot continuous resampling and argmax decoding to prevent small lesion
    drop-out and eliminate aliasing artifacts.

    Args:
        mask: Discrete prediction mask from model output.
        transform: SpatialTransform metadata recorded during preprocessing.

    Returns:
        Integer mask restored to transform.original_shape as int16.
    """
    uncropped = invert_center_crop_or_pad(mask, transform.center, value=0)
    if uncropped.shape == transform.original_shape:
        return uncropped.astype(np.int16, copy=False)

    unique_classes = np.unique(uncropped)
    if len(unique_classes) <= 1:
        return np.full(
            transform.original_shape,
            unique_classes[0] if unique_classes.size > 0 else 0,
            dtype=np.int16,
        )

    c_list = sorted(unique_classes.tolist())
    one_hot = np.stack([(uncropped == c).astype(np.float32) for c in c_list], axis=0)

    resampled_channels = [
        resize_to_shape(one_hot[i], transform.original_shape, order=1)
        for i in range(len(c_list))
    ]
    res_stack = np.stack(resampled_channels, axis=0)
    argmax_idx = np.argmax(res_stack, axis=0)
    c_arr = np.array(c_list, dtype=np.int16)
    restored = c_arr[argmax_idx]

    return restored.astype(np.int16, copy=False)
