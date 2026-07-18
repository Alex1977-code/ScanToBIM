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


def test_first_meaningful_line_digs_out_cupy_cause():
    """CuPy's banner-style ImportError yields the real cause, not ''."""
    banner = ImportError(
        "\n================================\n"
        "Failed to import CuPy.\n\n"
        "Original error:\n"
        "  ImportError: DLL load failed while importing runtime: "
        "Das angegebene Modul wurde nicht gefunden.\n"
        "================================\n"
    )
    line = accel._first_meaningful_line(banner)
    assert line.startswith("ImportError: DLL load failed")

    assert accel._first_meaningful_line(ValueError("kaputt")) == "kaputt"
    assert accel._first_meaningful_line(ValueError("")) == ""


def test_discovery_info_reports_dir_count():
    info = accel._discovery_info()
    assert "CUDA-Laufzeit-Ordner gefunden:" in info


def test_smallest_eigvec_matches_eigh():
    """Closed-form smallest eigenvector ≡ np.linalg.eigh (up to sign)."""
    rng = np.random.default_rng(3)
    m = rng.normal(size=(500, 3, 3))
    cov = m @ m.transpose(0, 2, 1)  # random SPD, mixed scales
    cov *= 10.0 ** rng.integers(-6, 6, size=(500, 1, 1))
    got = accel.smallest_eigvec_sym33(
        np,
        cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
        cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2],
    )
    _, vecs = np.linalg.eigh(cov)
    want = vecs[:, :, 0]
    agree = np.abs(np.einsum("ij,ij->i", got, want))
    assert np.all(np.linalg.norm(got, axis=1) > 0.999)
    assert np.median(agree) > 0.9999
    assert np.all(agree > 0.99)


def test_smallest_eigvec_degenerate_isotropic():
    """Isotropic and rank-deficient matrices return a unit vector, no NaN."""
    eye = np.tile(np.eye(3), (4, 1, 1))
    eye[1] *= 0.0  # zero matrix
    eye[2, 2, 2] = 5.0  # repeated smallest eigenvalue (1, 1, 5)
    got = accel.smallest_eigvec_sym33(
        np,
        eye[:, 0, 0], eye[:, 0, 1], eye[:, 0, 2],
        eye[:, 1, 1], eye[:, 1, 2], eye[:, 2, 2],
    )
    assert np.all(np.isfinite(got))
    assert np.allclose(np.linalg.norm(got, axis=1), 1.0)
    # Repeated smallest eigenvalue: any vector ⊥ the large axis is valid.
    assert abs(got[2] @ np.array([0, 0, 1.0])) < 1e-6


def test_pca_normals_cpu_fallback():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1, (500, 3))
    idx = rng.integers(0, 500, (500, 8))
    # Below min_gpu / without GPU → None (caller uses the numpy path).
    assert accel.pca_normals(pts, idx) is None
