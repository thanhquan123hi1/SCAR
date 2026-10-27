"""Geometry and subject isolation checks for prediction-localized ROI preparation."""
import numpy as np
import nibabel as nib
import torch

from preprocessing.myopspp_unet2d import (crop_fixed, letterbox, unletterbox, predicted_center,
                                         resample_xy, resize_roi, target_grid, SmallUNet)


def test_crop_affine_maps_roi_voxels_to_original_world_coordinates():
    image = np.arange(20 * 18 * 2).reshape(20, 18, 2)
    affine = np.array([[-1., 0, 0, 100], [0, 1, 0, -20], [0, 0, 12, 5], [0, 0, 0, 1]])
    roi, output_affine, info = crop_fixed(image, affine, [2, 15], 8)
    assert info["pad_before_hw"] == [2, 0]
    assert info["pad_after_hw"] == [0, 1]
    assert np.array_equal(roi[2:8, :7], image[:6, 11:18])
    assert np.array_equal(roi[:2], np.zeros((2, 8, 2)))
    assert np.allclose(output_affine @ [2, 0, 1, 1], affine @ [0, 11, 1, 1])


def test_inplane_resampling_preserves_native_z_and_matches_world_coordinates():
    a = np.zeros((9, 11, 3), np.uint16)
    a[3, 4, 1] = 2221
    affine = np.diag([2., -2., 14., 1.])
    image = nib.Nifti1Image(a, affine)
    shape, grid = target_grid(image)
    out = resample_xy(image, shape, grid, 0)
    assert shape == (17, 21, 3)
    assert out[6, 8, 1] == 2221
    assert np.allclose(grid @ [6, 8, 1, 1], affine @ [3, 4, 1, 1])
    assert np.array_equal(out[:, :, 0], np.zeros((17, 21)))


def test_letterbox_inverse_does_not_shift_center_or_include_padding():
    a = np.zeros((40, 80, 2), np.float32)
    a[14:26, 32:48, :] = 1
    boxed, metadata = letterbox(a, side=160)
    assert metadata["resized_hw"] == [80, 160]
    restored = unletterbox(torch.from_numpy(boxed), metadata)
    center, _, status = predicted_center(restored >= .5)
    assert center.tolist() == [20, 40]
    assert not status["empty_prediction"]


def test_prediction_bbox_ignores_smaller_disconnected_noise():
    mask = np.zeros((60, 70, 3), bool)
    mask[20:30, 25:45, :] = True
    mask[2:4, 2:4, :] = True
    center, clean, status = predicted_center(mask)
    assert center.tolist() == [24, 34]
    assert status["bbox_hw"] == [10, 20]
    assert not clean[2, 2, 0]


def test_empty_prediction_fallback_keeps_center_independent_of_any_gt():
    center, clean, status = predicted_center(np.zeros((100, 110, 4), bool))
    assert center.tolist() == [50, 54]
    assert status["empty_prediction"] and not clean.any()


def test_resize160_to128_updates_spacing_and_pixel_center_origin():
    arr = np.arange(160 * 160, dtype=np.float32).reshape(160, 160, 1)
    affine = np.eye(4)
    out, aff = resize_roi(arr, affine)
    assert out.shape == (128, 128, 1)
    assert np.allclose(nib.affines.voxel_sizes(aff), [1.25, 1.25, 1.])
    assert np.allclose(aff @ [0, 0, 0, 1], [0.125, 0.125, 0, 1])
    assert np.isclose(out[0, 0, 0], .125 * 160 + .125)


def test_resize_labels_never_introduces_unknown_classes():
    a = np.zeros((160, 160, 2), np.uint16)
    a[30:90, 40:100] = 2221
    out, _ = resize_roi(a, np.eye(4), True)
    assert set(np.unique(out)) == {0, 2221}


def test_localizer_restores_input_shape_with_single_binary_logit():
    torch.set_num_threads(2)
    model = SmallUNet(base=8)
    with torch.no_grad():
        logits = model(torch.zeros(1, 1, 64, 64))
    assert logits.shape == (1, 1, 64, 64)
