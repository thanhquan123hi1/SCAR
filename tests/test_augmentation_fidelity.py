"""Compare to frozen external code, including random-number consumption."""
from itertools import product
from pathlib import Path
import random
import runpy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from training.dataset.myops_dataset import RandomGenerator


REFERENCE = runpy.run_path(str(Path(__file__).parent / "fixtures/im_mseg_augmentation.py"))
KEYS = ("image", "image1", "image2", "label")


@pytest.fixture(autouse=True)
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)


def make_sample(shape, label_dtype=np.int64):
    rng = np.random.default_rng(84)
    sample = {key: rng.random(shape, dtype=np.float32) for key in KEYS[:3]}
    # Discontinuities and border structures expose wraparound/crop/interpolation.
    for image in sample.values():
        image[:3, :5] = 1
        image[-4:, -2:] = 0
    sample["label"] = rng.integers(0, 4, shape).astype(label_dtype)
    sample.update(case_name="case0002_slice0", patient_id="case0002",
                  spacing=np.array([1.5, 1.5, 7.0]), has_geometry=True)
    return sample


def run(transform, sample, seed):
    random.seed(seed)
    np.random.seed(seed)
    output = transform(sample)
    return output, random.getstate(), np.random.get_state()


def assert_same(actual, expected, sample):
    output, py_state, np_state = actual
    reference, ref_py_state, ref_np_state = expected
    for key in KEYS:
        # Exact equality, not a tolerance-based comparison.
        assert output[key].dtype == reference[key].dtype
        assert torch.equal(output[key], reference[key]), key
        assert torch.isfinite(output[key]).all(), key
    assert py_state == ref_py_state
    assert np_state[0] == ref_np_state[0]
    np.testing.assert_array_equal(np_state[1], ref_np_state[1])
    assert np_state[2:] == ref_np_state[2:]
    for key in sample.keys() - set(KEYS):
        assert output[key] is sample[key]


@pytest.mark.parametrize("shape", [(128, 128), (160, 192), (48, 64)])
@pytest.mark.parametrize("label_dtype", [np.int64, np.float32])
def test_matches_im_mseg_for_seeded_samples(shape, label_dtype):
    sample = make_sample(shape, label_dtype)
    original = {key: sample[key].copy() for key in KEYS}
    reference = REFERENCE["RandomGenerator"]([128, 128])
    actual = RandomGenerator([128, 128])
    for seed in range(64):
        assert_same(run(actual, sample, seed), run(reference, sample, seed), sample)
    for key in KEYS:
        np.testing.assert_array_equal(sample[key], original[key])


BRANCHES = {
    "rot_flip": [0.75],
    "rotate": [0.25, 0.75],
    "noise": [0.25, 0.25, 0.75],
    "none": [0.25, 0.25, 0.25],
}


@pytest.mark.parametrize("branch", BRANCHES)
@pytest.mark.parametrize("flags", list(product([False, True], repeat=4)))
def test_every_branch_and_combination_matches_im_mseg(branch, flags):
    # flags control shift, brightness, gamma and crop, in that exact order.
    decisions = BRANCHES[branch] + [0.75 if enabled else 0.25 for enabled in flags]
    sample = make_sample((137, 149))
    outputs = []
    for transform in (RandomGenerator([128, 128]), REFERENCE["RandomGenerator"]([128, 128])):
        with patch.object(random, "random", side_effect=decisions) as choose:
            outputs.append(run(transform, sample, seed=1234))
            assert choose.call_count == len(decisions)
    assert_same(*outputs, sample)
