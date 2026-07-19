"""LiDAR-guided multi-view stereo (Foto-Geometrie).

PatchMatch-style narrow-band depth search, natively implemented (CUDA
via CuPy when available, NumPy otherwise) instead of shipping external
COLMAP/OpenMVS binaries. The LiDAR surface provides a metric depth prior
per pixel, so the photo-consistency search only has to scan a ±12 cm
band instead of the whole frustum — faster AND more robust than blind
PatchMatch. Works directly on the fisheye images via the exact
``CameraPose.unproject`` rays (no lossy pinhole resampling).

Gate rule (LiDAR-Abgleich): a photo depth that lands within 5 cm of the
LiDAR prior is always trusted; further away it must be strongly
photo-consistent (NCC ≥ 0.75) — that is the edge band where the photos
genuinely out-resolve the LiDAR (profiles, bars, overhang rims).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_GRAD_MIN = 18.0     # gray-level gradient to consider a pixel (edge band)
_NCC_MIN = 0.60      # minimum photo-consistency to accept a depth
_NCC_STRONG = 0.75   # NCC at which MVS may override the LiDAR gate
_LIDAR_GATE = 0.05   # m — max deviation from the prior without strong NCC


def _gray(img: np.ndarray) -> np.ndarray:
    return (
        0.299 * img[:, :, 0].astype(np.float32)
        + 0.587 * img[:, :, 1].astype(np.float32)
        + 0.114 * img[:, :, 2].astype(np.float32)
    )


def _bilinear_xp(xp, gray, x, y):
    h, w = gray.shape
    x = xp.clip(x, 0.0, w - 1.001)
    y = xp.clip(y, 0.0, h - 1.001)
    x0 = x.astype(xp.int64)
    y0 = y.astype(xp.int64)
    fx = (x - x0).astype(xp.float32)
    fy = (y - y0).astype(xp.float32)
    c00 = gray[y0, x0]
    c10 = gray[y0, x0 + 1]
    c01 = gray[y0 + 1, x0]
    c11 = gray[y0 + 1, x0 + 1]
    return (c00 * (1 - fx) + c10 * fx) * (1 - fy) + (
        c01 * (1 - fx) + c11 * fx
    ) * fy


def _stereo_partner_name(name: str) -> str | None:
    low = name.lower()
    if "left" in low:
        return name.lower().replace("left", "right")
    if "right" in low:
        return name.lower().replace("right", "left")
    return None


def _pick_neighbors(ci: int, cameras, centers) -> list[int]:
    """Stereo partner first (metric baseline), then nearest other view."""
    cam = cameras[ci]
    out = []
    partner = _stereo_partner_name(cam.name)
    if partner is not None:
        for cj, other in enumerate(cameras):
            if cj != ci and other.name.lower() == partner:
                out.append(cj)
                break
    d2 = np.einsum("ij,ij->i", centers - centers[ci], centers - centers[ci])
    d2[ci] = np.inf
    for cj in out:
        d2[cj] = np.inf
    order = np.argsort(d2)
    for cj in order[:2]:
        if np.isfinite(d2[cj]) and 0.005 < d2[cj] < 4.0:
            out.append(int(cj))
        if len(out) >= 2:
            break
    return out


def _depth_prior(cam, pts_world, shape, scale):
    """Min-splat LiDAR depth into the (scaled) reference view + dilation."""
    from scipy.ndimage import minimum_filter

    pc = pts_world @ cam.rotation.T + cam.translation
    front = pc[:, 2] > 0.3
    if not front.any():
        return None
    px, py = cam.project(pc[front])
    h, w = shape
    gx = np.round(px * scale).astype(np.int64)
    gy = np.round(py * scale).astype(np.int64)
    ok = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
    if not ok.any():
        return None
    prior = np.full(h * w, np.inf, dtype=np.float32)
    np.minimum.at(prior, gy[ok] * w + gx[ok], pc[front][ok][:, 2])
    prior = prior.reshape(h, w)
    raw = np.where(np.isfinite(prior), prior, np.nan)
    # Dilate priors into small gaps (roof holes, frayed rims): the search
    # band around the NEIGHBORING surface depth is exactly what closes
    # them with real photo measurements. The RAW prior is returned too:
    # deviation statistics against the dilated prior are dominated by
    # depth edges (background pixels inherit the foreground depth) and
    # made the reported "Abweichung zu LiDAR" look 10x worse than the
    # actual surface agreement.
    return minimum_filter(prior, size=9, mode="nearest"), raw


def _view_points(
    ci, cameras, grays, rgbs, prior, scale, band, steps, patch, rng,
    max_px, dense=False, stride=3, prior_raw=None,
):
    """MVS points for one reference view. Returns (world_pts, colors, ncc,
    dev) or None.

    ``dense=False``: edge-band pixels only (gradient > _GRAD_MIN) — the
    classic detail booster. ``dense=True``: EVERY ``stride``-th pixel with
    a prior becomes a depth estimate — the photos become a surface source
    over the whole frame (RealityScan principle); low-texture pixels fail
    the NCC gate by themselves and cost only compute.
    """
    from scantobim.core import accel

    cam = cameras[ci]
    gray = grays[ci]
    h, w = gray.shape
    gy_img, gx_img = np.gradient(gray)
    gradmag = np.abs(gx_img) + np.abs(gy_img)
    m = patch + 2
    if dense:
        sel = np.zeros_like(gray, dtype=bool)
        sel[::stride, ::stride] = True
        # Edge pixels always participate at full density.
        sel |= gradmag > _GRAD_MIN
    else:
        sel = gradmag > _GRAD_MIN
    sel[:m, :] = sel[-m:, :] = False
    sel[:, :m] = sel[:, -m:] = False
    sel &= np.isfinite(prior)
    iy, ix = np.nonzero(sel)
    if len(ix) < 50:
        return None
    if len(ix) > max_px:
        keep = rng.choice(len(ix), max_px, replace=False)
        iy, ix = iy[keep], ix[keep]

    neighbors = _pick_neighbors(
        ci, cameras,
        np.array([-c.rotation.T @ c.translation for c in cameras]),
    )
    neighbors = [cj for cj in neighbors if grays.get(cj) is not None]
    if not neighbors:
        return None

    # Patch offsets around each selected pixel (full-res ray geometry).
    offs = np.arange(-patch, patch + 1)
    dx, dy = np.meshgrid(offs, offs)
    dx, dy = dx.ravel(), dy.ravel()  # (P,)
    n, p = len(ix), len(dx)
    px_full = (ix[:, None] + dx[None, :]) / scale  # (n, P)
    py_full = (iy[:, None] + dy[None, :]) / scale
    rays = cam.unproject(px_full.ravel(), py_full.ravel()).reshape(n, p, 3)
    rz = np.maximum(rays[:, :, 2], 1e-6)

    ref_patch = gray[iy[:, None] + dy[None, :], ix[:, None] + dx[None, :]]
    ref_patch = ref_patch.astype(np.float32)
    ref_n = ref_patch - ref_patch.mean(axis=1, keepdims=True)
    ref_std = np.sqrt((ref_n**2).mean(axis=1)) + 1e-4

    d0 = prior[iy, ix].astype(np.float64)  # (n,)
    depths = d0[:, None] + np.linspace(-band, band, steps)[None, :]
    depths = np.maximum(depths, 0.3)  # (n, steps)

    xp = accel.gpu() or np
    rays_x = xp.asarray(rays, dtype=xp.float32)
    rz_x = xp.asarray(rz, dtype=xp.float32)
    ref_n_x = xp.asarray(ref_n, dtype=xp.float32)
    ref_std_x = xp.asarray(ref_std, dtype=xp.float32)
    depths_x = xp.asarray(depths, dtype=xp.float32)
    r_ref_t = xp.asarray(cam.rotation.T, dtype=xp.float32)
    t_ref = xp.asarray(cam.translation, dtype=xp.float32)

    ncc_sum = xp.zeros((n, steps), dtype=xp.float32)
    ncc_cnt = xp.zeros((n, steps), dtype=xp.float32)
    for cj in neighbors:
        nb = cameras[cj]
        gray_n = xp.asarray(grays[cj], dtype=xp.float32)
        r_n = xp.asarray(nb.rotation, dtype=xp.float32)
        t_n = xp.asarray(nb.translation, dtype=xp.float32)
        rel_r = r_n @ r_ref_t
        rel_t = t_n - rel_r @ t_ref
        for k in range(steps):
            z = depths_x[:, k][:, None] / rz_x  # (n, P) scale along ray
            pc_ref = rays_x * z[:, :, None]
            pc_n = pc_ref @ rel_r.T + rel_t
            flat = pc_n.reshape(-1, 3)
            if xp is np:
                qx, qy = nb.project(flat)
            else:
                qx, qy = _project_xp(xp, nb, flat)
            vals = _bilinear_xp(
                xp, gray_n,
                (qx * scale).reshape(n, p), (qy * scale).reshape(n, p),
            )
            behind = (flat[:, 2] <= 0.1).reshape(n, p)
            vals = xp.where(behind, xp.float32(np.nan), vals)
            v_n = vals - xp.nanmean(vals, axis=1, keepdims=True)
            v_n = xp.where(xp.isnan(v_n), xp.float32(0.0), v_n)
            v_std = xp.sqrt((v_n**2).mean(axis=1)) + 1e-4
            ncc = (ref_n_x * v_n).mean(axis=1) / (ref_std_x * v_std)
            good = xp.isfinite(ncc)
            ncc_sum[:, k] += xp.where(good, ncc, 0.0)
            ncc_cnt[:, k] += good.astype(xp.float32)

    ncc = ncc_sum / xp.maximum(ncc_cnt, 1.0)
    kbest = xp.argmax(ncc, axis=1)
    nb_idx = xp.arange(n)
    best = ncc[nb_idx, kbest]
    # Parabolic subpixel refinement over the depth axis.
    km = xp.clip(kbest, 1, steps - 2)
    g0 = ncc[nb_idx, km - 1]
    g1 = ncc[nb_idx, km]
    g2 = ncc[nb_idx, km + 1]
    # Peak parabola: curvature is negative at a maximum — clamp on the
    # negative side so the subpixel offset keeps its sign.
    den = g0 - 2 * g1 + g2
    den = xp.where(den > -1e-6, xp.float32(-1e-6), den)
    sub = xp.clip(0.5 * (g0 - g2) / den, -1.0, 1.0)
    step_m = 2.0 * band / (steps - 1)
    d_best = (
        depths_x[nb_idx, km] + sub.astype(xp.float32) * xp.float32(step_m)
    )
    from scantobim.core.accel import asnumpy

    best = asnumpy(best)
    d_best = asnumpy(d_best).astype(np.float64)
    dev = np.abs(d_best - d0)
    accept = (best >= _NCC_MIN) & (
        (dev <= _LIDAR_GATE) | (best >= _NCC_STRONG)
    )
    if not accept.any():
        return None
    # Statistics against the RAW (un-dilated) prior where one exists —
    # the dilated prior mis-measures every depth edge by construction.
    if prior_raw is not None:
        raw_d = prior_raw[iy, ix].astype(np.float64)
        dev = np.where(np.isfinite(raw_d), np.abs(d_best - raw_d), dev)
    center_ray = cam.unproject(ix[accept] / scale, iy[accept] / scale)
    z = d_best[accept] / np.maximum(center_ray[:, 2], 1e-6)
    pc = center_ray * z[:, None]
    world = (pc - cam.translation) @ cam.rotation
    rgb = rgbs.get(ci)
    colors = (
        rgb[iy[accept], ix[accept]]
        if rgb is not None
        else np.full((int(accept.sum()), 3), 170, dtype=np.uint8)
    )
    return world, colors, best[accept], dev[accept]


def _project_xp(xp, cam, pts):
    """CameraPose.project on the GPU (POLYFISHEYE + pinhole)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    if cam.model == "POLYFISHEYE" and cam.poly:
        rho = xp.sqrt(x * x + y * y)
        theta = xp.arctan2(rho, z)
        r = theta.copy()
        tp = theta.copy()
        for k in cam.poly:
            tp = tp * theta
            r = r + xp.float32(k) * tp
        ux = xp.where(rho > 1e-12, x / rho, 0.0)
        uy = xp.where(rho > 1e-12, y / rho, 0.0)
        mx, my = r * ux, r * uy
        return (
            xp.float32(cam.fx) * mx + xp.float32(cam.a12) * my
            + xp.float32(cam.cx),
            xp.float32(cam.fy) * my + xp.float32(cam.cy),
        )
    zz = xp.where(xp.abs(z) > 1e-9, z, 1e-9)
    xn, yn = x / zz, y / zz
    if cam.k1:
        f = 1.0 + xp.float32(cam.k1) * (xn * xn + yn * yn)
        xn, yn = xn * f, yn * f
    return (
        xp.float32(cam.fx) * xn + xp.float32(cam.cx),
        xp.float32(cam.fy) * yn + xp.float32(cam.cy),
    )


def mvs_points(
    cameras,
    images_dir,
    cloud_points: np.ndarray,
    image_map: dict | None = None,
    stats_out: dict | None = None,
    max_views: int = 140,
    scale: float = 0.3,
    band: float = 0.12,
    steps: int = 25,
    patch: int = 2,
    max_px_per_view: int = 20_000,
    max_points: int = 3_000_000,
    dense: bool = False,
    stride: int = 3,
):
    """Photo-triangulated 3D points, LiDAR-gated.

    Returns ``(points (N,3), colors (N,3) uint8)`` or ``None``. The
    points are meant to be FUSED into the detail cloud before meshing.
    ``dense=False``: edge band only (profiles, bars, hole rims).
    ``dense=True``: full-frame surface points every ``stride`` pixels —
    the photos carry the geometry wherever they are sharp enough, which
    is what lifts brick relief and terrain off the LiDAR voxel floor.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    from scantobim.core.texture import _index_images

    if not cameras or not len(cloud_points):
        return None
    index = dict(image_map) if image_map else _index_images(Path(images_dir))
    pts_world = np.asarray(cloud_points, dtype=np.float64)
    if len(pts_world) > 500_000:
        rng0 = np.random.default_rng(0)
        pts_world = pts_world[
            rng0.choice(len(pts_world), 500_000, replace=False)
        ]

    # Reference views spread evenly over the trajectory.
    n_cam = len(cameras)
    ref_idx = list(range(n_cam))
    if n_cam > max_views:
        ref_idx = list(np.linspace(0, n_cam - 1, max_views).astype(int))
    needed = set(ref_idx)
    centers = np.array([-c.rotation.T @ c.translation for c in cameras])
    for ci in list(needed):
        for cj in _pick_neighbors(ci, cameras, centers.copy()):
            needed.add(cj)

    grays: dict[int, np.ndarray] = {}
    rgbs: dict[int, np.ndarray] = {}

    def _load(ci):
        cam = cameras[ci]
        path = index.get(cam.name) or index.get(Path(cam.name).name)
        if path is None or not path.exists():
            return
        try:
            img = Image.open(path).convert("RGB")
            w = max(2, int(round(cam.width * scale)))
            h = max(2, int(round(cam.height * scale)))
            small = np.asarray(img.resize((w, h), Image.BILINEAR))
            grays[ci] = _gray(small)
            rgbs[ci] = small
        except Exception:  # noqa: BLE001
            pass

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(_load, sorted(needed)))

    rng = np.random.default_rng(1)
    all_pts, all_col, all_ncc, all_dev = [], [], [], []
    views_used = 0
    for ci in ref_idx:
        if ci not in grays:
            continue
        view_scale = grays[ci].shape[1] / cameras[ci].width
        got_prior = _depth_prior(
            cameras[ci], pts_world, grays[ci].shape, view_scale
        )
        if got_prior is None:
            continue
        prior, prior_raw = got_prior
        got = _view_points(
            ci, cameras, grays, rgbs, prior, view_scale,
            band, steps, patch, rng, max_px_per_view,
            dense=dense, stride=stride, prior_raw=prior_raw,
        )
        if got is None:
            continue
        w_pts, w_col, w_ncc, w_dev = got
        all_pts.append(w_pts)
        all_col.append(w_col)
        all_ncc.append(w_ncc)
        all_dev.append(w_dev)
        views_used += 1
        if sum(len(a) for a in all_pts) >= max_points:
            break

    if not all_pts:
        if stats_out is not None:
            stats_out["mvs"] = {"ansichten": 0, "punkte": 0}
        return None
    pts = np.vstack(all_pts)
    cols = np.vstack(all_col)
    ncc = np.concatenate(all_ncc)
    dev = np.concatenate(all_dev)
    if stats_out is not None:
        stats_out["mvs"] = {
            "ansichten": int(views_used),
            "punkte": int(len(pts)),
            "ncc_median": round(float(np.median(ncc)), 3),
            "abweichung_zu_lidar_mm_median": round(
                float(np.median(dev)) * 1000.0, 1
            ),
            "kantenband_anteil": round(
                float((dev > _LIDAR_GATE).mean()), 3
            ),
        }
    return pts, cols
