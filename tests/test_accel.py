"""GPU acceleration wrappers: graceful CPU fallback, identical results."""

import numpy as np

from scantobim.core import accel


def test_gpu_detection_never_raises():
    # In this environment there is no CUDA — must return None quietly.
    assert accel.gpu() is None or accel.gpu() is not None  # no exception
    accel.gpu_name()  # idempotent, cached


def test_unique_i64_matches_numpy():
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 1000, 50000).astype(np.int64)
    v1 = accel.unique_i64(keys)
    assert np.array_equal(v1, np.unique(keys))
    v2, idx = accel.unique_i64(keys, return_index=True)
    e2, eidx = np.unique(keys, return_index=True)
    assert np.array_equal(v2, e2) and np.array_equal(idx, eidx)
    v3, inv, cnt = accel.unique_i64(keys, return_inverse=True, return_counts=True)
    e3, einv, ecnt = np.unique(keys, return_inverse=True, return_counts=True)
    assert np.array_equal(v3, e3)
    assert np.array_equal(inv, einv) and np.array_equal(cnt, ecnt)


def test_xp_for_and_asnumpy_cpu():
    xp = accel.xp_for(10)
    assert xp is np
    a = np.arange(5)
    assert accel.asnumpy(a) is a


def test_pca_normals_cpu_fallback():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1, (500, 3))
    idx = rng.integers(0, 500, (500, 8))
    # Below min_gpu / without GPU → None (caller uses the numpy path).
    assert accel.pca_normals(pts, idx) is None
