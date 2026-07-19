"""Photometric pose refinement (Posen-Feinschliff).

The SLAM poses land within ~5 cm of the cloud — good enough for
texturing, too coarse for multi-view triangulation. Each camera pose is
refined photometrically: the colorized LiDAR cloud is projected into the
photo and a small SE(3) correction (camera-frame rotation + translation)
is optimized so the photo luminance agrees with the point luminance.
The LiDAR geometry is the fixed anchor — this is bundle adjustment with
the structure held by the scanner, which keeps the metric scale exact.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_MAX_SHIFT_M = 0.2       # never move a camera further than this
_MAX_ROT_DEG = 2.5       # never rotate further than this
_MIN_POINTS = 400        # minimum projected cloud points to attempt a fit


def _rodrigues(w: np.ndarray) -> np.ndarray:
    """Rotation matrix for a small axis-angle vector."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    kx = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)


def _bilinear(gray: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    x = np.clip(x, 0.0, w - 1.001)
    y = np.clip(y, 0.0, h - 1.001)
    x0 = x.astype(np.int64)
    y0 = y.astype(np.int64)
    fx = x - x0
    fy = y - y0
    c00 = gray[y0, x0]
    c10 = gray[y0, x0 + 1]
    c01 = gray[y0 + 1, x0]
    c11 = gray[y0 + 1, x0 + 1]
    return (c00 * (1 - fx) + c10 * fx) * (1 - fy) + (
        c01 * (1 - fx) + c11 * fx
    ) * fy


def _normalized(v: np.ndarray) -> np.ndarray:
    s = float(v.std())
    return (v - float(v.mean())) / max(s, 1e-6)


def _refine_one(cam, gray, scale, pts, lum):
    """Refine one camera in place; returns (shift_m, rot_deg) or None."""
    from scipy.optimize import least_squares

    r0, t0 = cam.rotation.copy(), cam.translation.copy()
    pc = pts @ r0.T + t0
    front = pc[:, 2] > 0.3
    if int(front.sum()) < _MIN_POINTS:
        return None
    px, py = cam.project(pc[front])
    inside = (
        (px >= 2) & (px <= cam.width - 3)
        & (py >= 2) & (py <= cam.height - 3)
    )
    idx = np.flatnonzero(front)[inside]
    if len(idx) < _MIN_POINTS:
        return None
    if len(idx) > 4000:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, 4000, replace=False)
    p_sel = pts[idx]
    l_norm = _normalized(lum[idx])

    def residual(delta):
        r_d = _rodrigues(delta[:3])
        pc = p_sel @ (r_d @ r0).T + (r_d @ t0 + delta[3:])
        bad = pc[:, 2] <= 0.1
        pc[bad, 2] = 0.1
        qx, qy = cam.project(pc)
        vals = _bilinear(gray, qx * scale, qy * scale)
        res = _normalized(vals) - l_norm
        res[bad] = 3.0
        return res

    cost0 = float((residual(np.zeros(6)) ** 2).mean())
    try:
        fit = least_squares(
            residual, np.zeros(6), method="trf", loss="soft_l1",
            f_scale=1.0, max_nfev=120,
            diff_step=[2e-3, 2e-3, 2e-3, 4e-3, 4e-3, 4e-3],
        )
    except Exception:  # noqa: BLE001 — one bad camera must not stop the rest
        return None
    delta = fit.x
    rot_deg = float(np.rad2deg(np.linalg.norm(delta[:3])))
    r_d = _rodrigues(delta[:3])
    t_new = r_d @ t0 + delta[3:]
    c_old = -r0.T @ t0
    c_new = -(r_d @ r0).T @ t_new
    shift = float(np.linalg.norm(c_new - c_old))
    cost1 = float((residual(delta) ** 2).mean())
    if (
        cost1 > cost0 * 0.995
        or shift > _MAX_SHIFT_M
        or rot_deg > _MAX_ROT_DEG
    ):
        return None
    cam.rotation = r_d @ r0
    cam.translation = t_new
    return shift, rot_deg, cost0, cost1


def refine_camera_poses(
    cameras,
    images_dir,
    cloud_points: np.ndarray,
    cloud_colors: np.ndarray,
    image_map: dict | None = None,
    scale: float = 0.3,
    stats_out: dict | None = None,
) -> int:
    """Photometrically refine all camera poses in place.

    ``cloud_points``/``cloud_colors``: a colorized sample of the LiDAR
    cloud (world frame). Returns the number of refined cameras.
    """
    try:
        from PIL import Image
    except ImportError:
        return 0
    from scantobim.core.texture import _index_images

    if not cameras or cloud_colors is None or not len(cloud_points):
        return 0
    index = dict(image_map) if image_map else _index_images(Path(images_dir))
    pts = np.asarray(cloud_points, dtype=np.float64)
    lum = (
        0.299 * cloud_colors[:, 0].astype(np.float64)
        + 0.587 * cloud_colors[:, 1].astype(np.float64)
        + 0.114 * cloud_colors[:, 2].astype(np.float64)
    )
    if len(pts) > 150_000:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(pts), 150_000, replace=False)
        pts, lum = pts[keep], lum[keep]

    def _work(cam):
        path = index.get(cam.name) or index.get(Path(cam.name).name)
        if path is None or not path.exists():
            return None
        try:
            img = Image.open(path).convert("L")
            w = max(2, int(round(img.width * scale)))
            h = max(2, int(round(img.height * scale)))
            gray = np.asarray(
                img.resize((w, h), Image.BILINEAR), dtype=np.float32
            )
            real_scale = w / cam.width
            return _refine_one(cam, gray, real_scale, pts, lum)
        except Exception:  # noqa: BLE001
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_work, cameras))
    done = [r for r in results if r is not None]
    if stats_out is not None:
        shifts = [r[0] * 1000.0 for r in done]
        rots = [r[1] for r in done]
        stats_out["posen_feinschliff"] = {
            "kameras_geprueft": len(cameras),
            "kameras_nachgefuehrt": len(done),
            "median_korrektur_mm": (
                round(float(np.median(shifts)), 1) if shifts else 0.0
            ),
            "median_korrektur_grad": (
                round(float(np.median(rots)), 3) if rots else 0.0
            ),
            "residuum_vorher": (
                round(float(np.mean([r[2] for r in done])), 4)
                if done else None
            ),
            "residuum_nachher": (
                round(float(np.mean([r[3] for r in done])), 4)
                if done else None
            ),
        }
    return len(done)
