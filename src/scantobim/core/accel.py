"""Optional GPU acceleration (NVIDIA CUDA via CuPy).

The pipeline stays CPU-first: every helper here degrades gracefully — no
CuPy installed, no NVIDIA device, out-of-memory or any other GPU error →
transparent numpy fallback with identical results. The GPU build of the
Windows exe ships CuPy; the standard build simply never finds it.

The GPU build also bundles the CUDA runtime and NVRTC as pip wheels
(``nvidia-cuda-runtime-cu12`` / ``nvidia-cuda-nvrtc-cu12``) so users need
nothing but the NVIDIA driver — no CUDA-Toolkit installation. Their DLL
directories are registered before CuPy is imported. Everything here sticks
to CuPy's elementwise/reduction/sort kernels (compiled via NVRTC at
runtime); cuBLAS/cuSOLVER are deliberately avoided, which is why the PCA
eigenvectors are computed in closed form instead of ``linalg.eigh``.

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

import os
import sys
from pathlib import Path

import numpy as np

_gpu = None
_checked = False
_name: str | None = None
_error: str | None = None


def _cuda_library_dirs() -> list[Path]:
    """Directories holding bundled / pip-installed CUDA runtime libraries.

    The ``nvidia-*-cu12`` wheels install ``nvidia/<lib>/{bin,lib}`` into
    site-packages; the PyInstaller exe bundles the same tree next to its
    extraction root. CuPy's wheels expect those libraries from a CUDA
    Toolkit installation — registering the wheel directories instead lets
    the GPU build run on machines with nothing but the NVIDIA driver.
    """
    roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)  # PyInstaller onefile
    if meipass:
        roots.append(Path(meipass))
    for entry in sys.path:
        if not entry:
            continue
        try:
            p = Path(entry)
            if (p / "nvidia").is_dir():
                roots.append(p)
        except OSError:
            continue
    dirs: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        nv = root / "nvidia"
        if not nv.is_dir():
            continue
        try:
            subs = sorted(nv.iterdir())
        except OSError:
            continue
        for sub in subs:
            for leaf in ("bin", "lib"):
                d = sub / leaf
                if d.is_dir() and d not in seen:
                    seen.add(d)
                    dirs.append(d)
    return dirs


_dirs_registered = False


def _register_cuda_dirs() -> None:
    global _dirs_registered
    if _dirs_registered:
        return
    dirs = _cuda_library_dirs()
    if not dirs:
        return
    _dirs_registered = True
    runtime = next(
        (d.parent for d in dirs if d.parent.name == "cuda_runtime"), None
    )
    if runtime is not None:
        # Force, don't setdefault: a stale CUDA_PATH from an old/uninstalled
        # toolkit on the user's machine must not shadow the bundled runtime.
        os.environ["CUDA_PATH"] = str(runtime)
    else:
        os.environ.setdefault("CUDA_PATH", str(dirs[0].parent))
    for d in dirs:
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(d))
            except OSError:
                pass


def _first_meaningful_line(exc: Exception) -> str:
    """Compact one-line summary of an exception message.

    CuPy's import failure is a multi-line banner starting with a blank line
    and '=====' rules; the actual cause hides behind 'Original error:'.
    """
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        if line.lower().startswith("original error"):
            return lines[i + 1] if i + 1 < len(lines) else line
    for line in lines:
        if set(line) - {"=", "-", "*"}:
            return line
    return lines[0] if lines else ""


def _is_cupy_missing(msg: str) -> bool:
    """True only for 'CuPy is not installed at all' (the CPU build).

    A missing SUBmodule ('No module named cupy._core') is a broken GPU
    build and must be reported, so the match is exact.
    """
    m = msg.strip()
    return m in ("No module named 'cupy'", "No module named cupy")


def _discovery_info() -> str:
    """Where the bundled CUDA runtime was (not) found — for the log."""
    n = len(_cuda_library_dirs())
    meipass = getattr(sys, "_MEIPASS", None)
    root = f", Paket: {meipass}" if meipass else ""
    return f"CUDA-Laufzeit-Ordner gefunden: {n}{root}"


class _DummyDllDirectory:
    """Stand-in for os.add_dll_directory's handle (close/context protocol)."""

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _import_cupy():
    """``import cupy``, robust against CuPy's CUDA-path guessing.

    Newer CuPy derives the CUDA installation from where it finds cudart.
    In the onefile exe cudart sits in the extracted bundle ROOT, so CuPy
    takes that folder for ``bin``, its parent (%TEMP%) for the CUDA path —
    and crashes on ``add_dll_directory('%TEMP%\\bin')`` which doesn't
    exist. Our directories are registered already; a failing
    add_dll_directory inside CuPy's startup must not kill the import.
    """
    orig = getattr(os, "add_dll_directory", None)
    if orig is None:
        import cupy

        return cupy

    def _forgiving(path):
        try:
            return orig(path)
        except OSError:
            return _DummyDllDirectory()

    os.add_dll_directory = _forgiving
    try:
        import cupy

        return cupy
    finally:
        os.add_dll_directory = orig


def gpu():
    """The ``cupy`` module when a working CUDA device exists, else None."""
    global _gpu, _checked, _name, _error
    if _checked:
        return _gpu
    _checked = True
    try:
        _register_cuda_dirs()
        cupy = _import_cupy()

        if cupy.cuda.runtime.getDeviceCount() > 0:
            props = cupy.cuda.runtime.getDeviceProperties(0)
            raw = props.get("name", b"CUDA-GPU")
            _name = raw.decode() if isinstance(raw, bytes) else str(raw)
            float(cupy.arange(8).sum())  # end-to-end sanity check
            _gpu = cupy
        else:
            _error = "kein CUDA-Gerät gefunden (NVIDIA-Treiber installiert?)"
    except ImportError as exc:
        # CPU build without CuPy → nothing to report. EVERY other import
        # failure (DLL load, missing cupy SUBmodule, …) must reach the
        # log, with the real cause dug out of CuPy's multi-line banner.
        msg = _first_meaningful_line(exc)
        if _is_cupy_missing(msg):
            _error = None
        else:
            _error = f"ImportError: {msg[:200]} [{_discovery_info()}]"
    except Exception as exc:
        _gpu = None
        _name = None
        msg = _first_meaningful_line(exc)
        _error = f"{type(exc).__name__}: {msg[:200]} [{_discovery_info()}]"
    return _gpu


def gpu_error() -> str | None:
    """Why CUDA is unavailable (None: no CuPy installed / GPU works)."""
    gpu()
    return _error


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


def smallest_eigvec_sym33(xp, a00, a01, a02, a11, a12, a22):
    """Unit eigenvector of the smallest eigenvalue, batched symmetric 3x3.

    Closed form — trigonometric eigenvalues (Cardano) plus row cross
    products — so the GPU path needs only elementwise kernels, no cuSOLVER.
    Works with numpy or cupy as ``xp``; matrices are scale-normalized first,
    making the degeneracy thresholds dimensionless.
    """
    scale = xp.abs(a00)
    for a in (a01, a02, a11, a12, a22):
        scale = xp.maximum(scale, xp.abs(a))
    s = xp.maximum(scale, 1e-30)
    a00, a01, a02 = a00 / s, a01 / s, a02 / s
    a11, a12, a22 = a11 / s, a12 / s, a22 / s

    q = (a00 + a11 + a22) / 3.0
    b00, b11, b22 = a00 - q, a11 - q, a22 - q
    p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * (
        a01 * a01 + a02 * a02 + a12 * a12
    )
    p = xp.sqrt(xp.maximum(p2 / 6.0, 0.0))
    ps = xp.maximum(p, 1e-12)
    c00, c11, c22 = b00 / ps, b11 / ps, b22 / ps
    c01, c02, c12 = a01 / ps, a02 / ps, a12 / ps
    det_b = (
        c00 * (c11 * c22 - c12 * c12)
        - c01 * (c01 * c22 - c12 * c02)
        + c02 * (c01 * c12 - c11 * c02)
    )
    r = xp.clip(det_b / 2.0, -1.0, 1.0)
    phi = xp.arccos(r) / 3.0
    lam = q + 2.0 * p * xp.cos(phi + 2.0 * np.pi / 3.0)  # smallest eigenvalue

    # Eigenvector ⊥ two independent rows of (A − λI): best cross product.
    r0 = xp.stack([a00 - lam, a01, a02], axis=-1)
    r1 = xp.stack([a01, a11 - lam, a12], axis=-1)
    r2 = xp.stack([a02, a12, a22 - lam], axis=-1)

    def _cross(u, v):
        return xp.stack(
            [
                u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1],
                u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2],
                u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0],
            ],
            axis=-1,
        )

    best = _cross(r0, r1)
    best_n = (best * best).sum(axis=1)
    for u, v in ((r0, r2), (r1, r2)):
        c = _cross(u, v)
        n = (c * c).sum(axis=1)
        take = n > best_n
        best = xp.where(take[:, None], c, best)
        best_n = xp.where(take, n, best_n)

    # Repeated smallest eigenvalue → all crosses vanish; any direction
    # orthogonal to the strongest remaining row is a valid eigenvector.
    rows = xp.stack([r0, r1, r2], axis=1)  # (n, 3, 3)
    row_n = (rows * rows).sum(axis=2)
    strongest = rows[
        xp.arange(len(row_n)), xp.argmax(row_n, axis=1)
    ]
    helper = xp.zeros_like(strongest)
    use_x = xp.abs(strongest[:, 0]) < 0.9 * xp.sqrt(
        xp.maximum((strongest * strongest).sum(axis=1), 1e-30)
    )
    helper[:, 0] = xp.where(use_x, 1.0, 0.0)
    helper[:, 1] = xp.where(use_x, 0.0, 1.0)
    fallback = _cross(strongest, helper)
    fb_n = (fallback * fallback).sum(axis=1)
    degenerate = best_n < 1e-12
    best = xp.where(degenerate[:, None], fallback, best)
    best_n = xp.where(degenerate, fb_n, best_n)
    # Fully isotropic (sphere-like) — direction is arbitrary: use +Z.
    zaxis = xp.zeros_like(best)
    zaxis[:, 2] = 1.0
    still = best_n < 1e-12
    best = xp.where(still[:, None], zaxis, best)

    norm = xp.sqrt(xp.maximum((best * best).sum(axis=1), 1e-30))
    return best / norm[:, None]


# Register the bundled CUDA directories as early as possible — even if
# something imports CuPy before the first gpu() call, the DLLs resolve.
try:
    _register_cuda_dirs()
except Exception:  # noqa: BLE001 — never break import over this
    pass


def pca_normals(points: np.ndarray, idx: np.ndarray, min_gpu: int = 1_000_000):
    """Batched neighborhood-PCA normals on the GPU; None → use the CPU path.

    ``idx``: (N, k) neighbor indices. Chunked so GPU memory stays bounded.
    Covariances and eigenvectors are computed with elementwise kernels only
    (see :func:`smallest_eigvec_sym33`) — no cuBLAS/cuSOLVER required.
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
            x, y, z = nb[..., 0], nb[..., 1], nb[..., 2]
            normals = smallest_eigvec_sym33(
                cp,
                (x * x).sum(axis=1),
                (x * y).sum(axis=1),
                (x * z).sum(axis=1),
                (y * y).sum(axis=1),
                (y * z).sum(axis=1),
                (z * z).sum(axis=1),
            )
            out[s:s + chunk] = cp.asnumpy(normals).astype(np.float64)
        return out
    except Exception:
        return None
