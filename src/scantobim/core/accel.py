"""Optional GPU acceleration (NVIDIA CUDA via CuPy).

The pipeline stays CPU-first: every helper here degrades gracefully — no
CuPy installed, no NVIDIA device, out-of-memory or any other GPU error →
transparent numpy fallback with identical results. The GPU build of the
Windows exe ships CuPy; the standard build simply never finds it.

Accelerated hot spots (all dense, transfer-amortized math):

* 64-bit key sorting/deduplication (``unique_i64``) — the heart of the
  complete-mesh voxel pipeline and the streaming thinner,
* batched PCA normals (``pca_normals``) — covariance + eigenvectors for
  millions of neighborhoods,
* generic array-module dispatch (``xp_for``/``asnumpy``) for code written
  against the shared numpy/cupy API (photo projection, smoothing).

fp32 is used on the GPU where precision allows (normals, smoothing,
projections) — consumer cards run fp64 at 1/32 rate.
"""

from __future__ import annotations

import numpy as np

_gpu = None
_checked = False
_name: str | None = None


def gpu():
    """The ``cupy`` module when a working CUDA device exists, else None."""
    global _gpu, _checked, _name
    if _checked:
        return _gpu
    _checked = True
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            props = cupy.cuda.runtime.getDeviceProperties(0)
            raw = props.get("name", b"CUDA-GPU")
            _name = raw.decode() if isinstance(raw, bytes) else str(raw)
            float(cupy.arange(8).sum())  # end-to-end sanity check
            _gpu = cupy
    except Exception:
        _gpu = None
        _name = None
    return _gpu


def gpu_name() -> str | None:
    gpu()
    return _name


def xp_for(n_elements: int, min_gpu: int = 1_000_000):
    """Array module for a workload of ``n_elements``: cupy or numpy."""
    cp = gpu()
    if cp is not None and n_elements >= min_gpu:
        return cp
    return np


def asnumpy(a):
    cp = gpu()
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return a


def unique_i64(
    keys: np.ndarray,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    min_gpu: int = 2_000_000,
):
    """``np.unique`` for int64 keys, on the GPU when it pays off."""
    cp = gpu()
    if cp is not None and len(keys) >= min_gpu:
        try:
            res = cp.unique(
                cp.asarray(keys),
                return_index=return_index,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )
            if isinstance(res, tuple):
                return tuple(cp.asnumpy(r) for r in res)
            return cp.asnumpy(res)
        except Exception:
            pass
    return np.unique(
        keys,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )


def pca_normals(points: np.ndarray, idx: np.ndarray, min_gpu: int = 1_000_000):
    """Batched neighborhood-PCA normals on the GPU; None → use the CPU path.

    ``idx``: (N, k) neighbor indices. Chunked so GPU memory stays bounded.
    """
    cp = gpu()
    n, k = idx.shape
    if cp is None or n < min_gpu:
        return None
    try:
        out = np.empty((n, 3), dtype=np.float64)
        pts_g = cp.asarray(points, dtype=cp.float32)
        chunk = 1_500_000
        for s in range(0, n, chunk):
            nb = pts_g[cp.asarray(idx[s:s + chunk])]  # (c, k, 3) fp32
            nb = nb - nb.mean(axis=1, keepdims=True)
            cov = cp.einsum("nki,nkj->nij", nb, nb)
            _, vecs = cp.linalg.eigh(cov)
            normals = vecs[:, :, 0]
            norms = cp.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / cp.maximum(norms, 1e-12)
            out[s:s + chunk] = cp.asnumpy(normals).astype(np.float64)
        return out
    except Exception:
        return None
