"""Canonical MyoPS++ label mapping."""
import numpy as np
import pytest

def test_raw_mapping_preserves_pathology_and_rejects_unknown_labels():
    from preprocessing.myopspp import canonicalize_myopspp
    raw = np.asarray([0, 200, 500, 600, 1220, 2221], dtype=np.int16)
    np.testing.assert_array_equal(canonicalize_myopspp(raw), [0, 1, 0, 0, 2, 3])
    np.testing.assert_array_equal(raw, [0, 200, 500, 600, 1220, 2221])
    for invalid in ([999], [1220.5], [np.nan]):
        with pytest.raises(ValueError):
            canonicalize_myopspp(np.asarray(invalid))
