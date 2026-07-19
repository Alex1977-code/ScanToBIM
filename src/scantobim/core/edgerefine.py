"""Photo-based edge refinement (Kanten-Fotoabgleich).

The photos resolve 3-5 mm per pixel — several times finer than the LiDAR.
Candidate edges (intersection lines of adjacent structure planes) are
projected into the best-viewing validated photos, the image gradient is
searched perpendicular to the projected edge with subpixel precision, and
the 3D edge is shifted onto the photographic edge (robust median over
samples and cameras, capped at 2.5 cm). Detail-mesh vertices in a band
around the original edge follow the shift with a linear falloff.

This intentionally refines the RENDERED geometry only — the measured
plane parameters (dimensions, areas, reports) stay untouched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_MAX_SHIFT = 0.035  # m — never move an edge further than this
_SEARCH_PX = 12     # gradient search half-window in pixels
_GRAD_MIN = 10.0    # minimum gradient magnitude (gray levels) to accept


def _candidate_edges(surfaces, min_angle_deg: float = 18.0):
    """Edge segments from pairs of adjacent non-terrain surfaces."""
    edges = []
    usable = [
        s for s in surfaces
        if getattr(s, "surface_class", "") != "terrain" and len(s.outer) >= 3
    ]
    cos_max = np.cos(np.deg2rad(min_angle_deg))
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            si, sj = usable[i], usable[j]
            ni = np.asarray(si.normal, dtype=np.float64)
            nj = np.asarray(sj.normal, dtype=np.float64)
            if abs(float(ni @ nj)) > cos_max:
                continue
            direction = np.cross(ni, nj)
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            direction /= norm
            # Point on the intersection line of both planes.
            d_i = float(ni @ si.outer[0])
            d_j = float(nj @ sj.outer[0])
            a = np.array([ni, nj, direction])
            b = np.array([d_i, d_j, 0.0])
            try:
                point = np.linalg.solve(a, b)
            except np.linalg.LinAlgError:
                continue
            # Both boundaries must come close to the line, and the shared
            # extent along the line must be a real edge.
            ext = []
            near_ok = True
            for s in (si, sj):
                rel = s.outer - point
                t = rel @ direction
                lateral = np.linalg.norm(
                    rel - np.outer(t, direction), axis=1
                )
                near = lateral < 0.5
                if int(near.sum()) < 2:
                    near_ok = False
                    break
                ext.append((float(t[near].min()), float(t[near].max())))
            if not near_ok:
                continue
            t0 = max(ext[0][0], ext[1][0])
            t1 = min(ext[0][1], ext[1][1])
            if t1 - t0 < 0.25:
                continue
            edges.append({
                "point": point, "direction": direction,
                "t0": t0, "t1": t1,
                "normals": (ni, nj),
            })
    return edges


def _gray(photo: np.ndarray) -> np.ndarray:
    return (
        0.299 * photo[:, :, 0].astype(np.float32)
        + 0.587 * photo[:, :, 1].astype(np.float32)
        + 0.114 * photo[:, :, 2].astype(np.float32)
    )


def _bilinear_gray(gray: np.ndarray, px: np.ndarray, py: np.ndarray):
    h, w = gray.shape
    x = np.clip(px, 0.0, w - 1.001)
    y = np.clip(py, 0.0, h - 1.001)
    x0 = x.astype(np.int64)
    y0 = y.astype(np.int64)
    fx = x - x0
    fy = y - y0
    c00 = gray[y0, x0]
    c10 = gray[y0, x0 + 1]
    c01 = gray[y0 + 1, x0]
    c11 = gray[y0 + 1, x0 + 1]
    return (c00 * (1 - fx) + c10 * fx) * (1 - fy) + (c01 * (1 - fx) + c11 * fx) * fy


def _edge_offset_for_camera(edge, samples, cam, gray) -> np.ndarray | None:
    """Median 3D offset vector for one edge in one camera, or None."""
    pc = samples @ cam.rotation.T + cam.translation
    if not (pc[:, 2] > 0.3).all():
        return None
    px, py = cam.project(pc)
    inside = (
        (px >= _SEARCH_PX + 2) & (px <= cam.width - _SEARCH_PX - 2)
        & (py >= _SEARCH_PX + 2) & (py <= cam.height - _SEARCH_PX - 2)
    )
    if inside.mean() < 0.5:
        return None
    center = -cam.rotation.T @ cam.translation
    view = samples.mean(axis=0) - center
    view /= max(np.linalg.norm(view), 1e-9)
    w3 = np.cross(edge["direction"], view)
    n3 = np.linalg.norm(w3)
    if n3 < 0.2:  # edge nearly parallel to the view ray — unusable
        return None
    w3 /= n3
    # Image direction + scale of a 5 mm step along w3.
    step = 0.005
    pc2 = (samples + step * w3) @ cam.rotation.T + cam.translation
    px2, py2 = cam.project(pc2)
    ex = px2 - px
    ey = py2 - py
    e_len = np.sqrt(ex**2 + ey**2)
    good = inside & (e_len > 0.15)
    if good.mean() < 0.5:
        return None
    ex = ex[good] / e_len[good]
    ey = ey[good] / e_len[good]
    bx = px[good]
    by = py[good]
    scale = step / e_len[good]  # meters per pixel along the search line

    # Gradient search: sample gray at offsets -S..S along (ex, ey).
    offsets = np.arange(-_SEARCH_PX, _SEARCH_PX + 1, dtype=np.float64)
    vals = np.empty((len(bx), len(offsets)), dtype=np.float32)
    for k, s in enumerate(offsets):
        vals[:, k] = _bilinear_gray(gray, bx + s * ex, by + s * ey)
    grad = np.abs(vals[:, 2:] - vals[:, :-2])  # centered differences
    kbest = np.argmax(grad, axis=1)
    gmax = grad[np.arange(len(bx)), kbest]
    accept = gmax >= _GRAD_MIN
    if accept.mean() < 0.3:
        return None
    # Parabolic subpixel peak around kbest.
    k = kbest[accept]
    g = grad[accept]
    km = np.clip(k, 1, grad.shape[1] - 2)
    g0 = g[np.arange(len(km)), km - 1]
    g1 = g[np.arange(len(km)), km]
    g2 = g[np.arange(len(km)), km + 1]
    # Peak parabola: at a maximum the curvature g0-2g1+g2 is NEGATIVE —
    # clamp on the negative side (a positive clamp flips the direction).
    den = g0 - 2 * g1 + g2
    den = np.where(den > -1e-6, -1e-6, den)
    sub = np.clip(0.5 * (g0 - g2) / den, -1.0, 1.0)
    s_px = (km.astype(np.float64) + 1 + sub) - _SEARCH_PX  # offset in px
    shift_m = np.median(s_px * scale[accept])
    if abs(shift_m) > _MAX_SHIFT:
        return None
    return shift_m * w3


def refine_detail_edges(
    detail_mesh,
    surfaces,
    cameras,
    images_dir,
    image_map: dict | None = None,
    stats_out: dict | None = None,
    max_photos: int = 40,
) -> int:
    """Refine detail-mesh edges against the photos; returns edges moved."""
    try:
        from PIL import Image
    except ImportError:
        return 0
    from scantobim.core.texture import _index_images, _resolve_cameras

    cameras = _resolve_cameras(cameras)
    if not cameras or detail_mesh is None or not len(detail_mesh.faces):
        return 0
    index = dict(image_map) if image_map else _index_images(Path(images_dir))
    edges = _candidate_edges(surfaces)
    if not edges:
        return 0

    centers = np.array([-c.rotation.T @ c.translation for c in cameras])
    photo_cache: dict[str, np.ndarray] = {}

    def _gray_for(ci: int) -> np.ndarray | None:
        cam = cameras[ci]
        if cam.name in photo_cache:
            return photo_cache[cam.name]
        path = index.get(cam.name) or index.get(Path(cam.name).name)
        if path is None or len(photo_cache) >= max_photos:
            return None
        try:
            photo_cache[cam.name] = _gray(
                np.asarray(Image.open(path).convert("RGB"))
            )
        except Exception:  # noqa: BLE001
            return None
        return photo_cache[cam.name]

    verts = detail_mesh.vertices
    moved = 0
    shifts_mm = []
    for edge in edges:
        n_s = max(6, min(120, int((edge["t1"] - edge["t0"]) / 0.05)))
        ts = np.linspace(edge["t0"], edge["t1"], n_s)
        samples = edge["point"] + ts[:, None] * edge["direction"]
        mid = samples[len(samples) // 2]
        d2 = np.einsum("ij,ij->i", centers - mid, centers - mid)
        order = np.argsort(d2)[:10]
        offsets = []
        for ci in order:
            gray = _gray_for(int(ci))
            if gray is None:
                continue
            off = _edge_offset_for_camera(edge, samples, cameras[ci], gray)
            if off is not None:
                offsets.append(off)
            if len(offsets) >= 3:
                break
        if len(offsets) < 2:
            continue
        offset = np.median(np.array(offsets), axis=0)
        mag = float(np.linalg.norm(offset))
        if mag < 0.001 or mag > _MAX_SHIFT:
            continue
        # Pull detail vertices in a band around the edge segment.
        band = 0.08
        rel = verts - edge["point"]
        t = rel @ edge["direction"]
        seg = (t > edge["t0"] - 0.1) & (t < edge["t1"] + 0.1)
        if not seg.any():
            continue
        lateral = rel[seg] - np.outer(t[seg], edge["direction"])
        dist = np.linalg.norm(lateral, axis=1)
        in_band = dist < band
        if not in_band.any():
            continue
        idx = np.flatnonzero(seg)[in_band]
        falloff = 1.0 - dist[in_band] / band
        verts[idx] += offset[None, :] * falloff[:, None]
        moved += 1
        shifts_mm.append(mag * 1000.0)
    if stats_out is not None:
        stats_out["kanten_kandidaten"] = len(edges)
        stats_out["kanten_nachjustiert"] = moved
        stats_out["median_verschiebung_mm"] = (
            round(float(np.median(shifts_mm)), 1) if shifts_mm else 0.0
        )
    return moved
