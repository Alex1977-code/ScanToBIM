"""Render feedback loop (Render-Regelkreis).

The finished textured model is rendered from the validated camera poses
and compared against the original photos — SSIM plus gradient distance,
tile by tile. Tiles where the render disagrees with the photo mark
frayed or inaccurate geometry; the discrepancy is accumulated per face
into a heatmap the pipeline uses to re-reconstruct exactly those
regions. The loop closes automatically: rebuild, re-render, report the
SSIM distribution before/after.

The render is a z-buffered face-centroid splat at photo-thumbnail
resolution — at ~7 faces per pixel that is a faithful preview and fast
enough for hundreds of views (GPU via CuPy when available).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_C1 = (0.01 * 255.0) ** 2
_C2 = (0.03 * 255.0) ** 2


def _gray(img: np.ndarray) -> np.ndarray:
    return (
        0.299 * img[:, :, 0].astype(np.float32)
        + 0.587 * img[:, :, 1].astype(np.float32)
        + 0.114 * img[:, :, 2].astype(np.float32)
    )


def face_colors_from_atlas(mesh) -> np.ndarray:
    """Mean texture color per face, sampled at the UV centroid."""
    n_f = len(mesh.faces)
    out = np.full((n_f, 3), 170, dtype=np.uint8)
    if mesh.vertex_colors is not None:
        out = (
            mesh.vertex_colors[mesh.faces].mean(axis=1).astype(np.uint8)
        )
    pages = getattr(mesh, "textures", None) or (
        [mesh.texture] if mesh.texture is not None else []
    )
    if not pages or mesh.uvs is None:
        return out
    uv_c = mesh.uvs[mesh.faces].mean(axis=1)  # (F, 2) in [0, 1]
    face_page = (
        mesh.face_page
        if getattr(mesh, "face_page", None) is not None
        else np.zeros(n_f, dtype=np.int32)
    )
    for p, atlas in enumerate(pages):
        m = face_page == p
        if not m.any() or atlas is None:
            continue
        h, w = atlas.shape[:2]
        ix = np.clip((uv_c[m, 0] * w).astype(np.int64), 0, w - 1)
        # UV v axis points up, image rows point down.
        iy = np.clip(((1.0 - uv_c[m, 1]) * h).astype(np.int64), 0, h - 1)
        out[m] = atlas[iy, ix]
    return out


def render_view_splat(centroids_world, face_colors, cam, scale):
    """Z-buffered splat render; returns (rgb, mask, pix_face)."""
    w = max(8, int(round(cam.width * scale)))
    h = max(8, int(round(cam.height * scale)))
    pc = centroids_world @ cam.rotation.T + cam.translation
    front = pc[:, 2] > 0.2
    if not front.any():
        return None
    px, py = cam.project(pc[front])
    gx = np.round(px * scale).astype(np.int64)
    gy = np.round(py * scale).astype(np.int64)
    ok = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
    if not ok.any():
        return None
    fi = np.flatnonzero(front)[ok]
    lin = gy[ok] * w + gx[ok]
    z = pc[front][ok][:, 2]
    zbuf = np.full(h * w, np.inf, dtype=np.float64)
    np.minimum.at(zbuf, lin, z)
    win = z <= zbuf[lin] * 1.001 + 1e-6
    rgb = np.zeros((h * w, 3), dtype=np.uint8)
    pix_face = np.full(h * w, -1, dtype=np.int64)
    rgb[lin[win]] = face_colors[fi[win]]
    pix_face[lin[win]] = fi[win]
    mask = np.isfinite(zbuf)
    return (
        rgb.reshape(h, w, 3),
        mask.reshape(h, w),
        pix_face.reshape(h, w),
    )


def _ssim_map(a: np.ndarray, b: np.ndarray, win: int = 7) -> np.ndarray:
    from scipy.ndimage import uniform_filter

    mu_a = uniform_filter(a, win)
    mu_b = uniform_filter(b, win)
    s_aa = uniform_filter(a * a, win) - mu_a * mu_a
    s_bb = uniform_filter(b * b, win) - mu_b * mu_b
    s_ab = uniform_filter(a * b, win) - mu_a * mu_b
    return ((2 * mu_a * mu_b + _C1) * (2 * s_ab + _C2)) / np.maximum(
        (mu_a**2 + mu_b**2 + _C1) * (s_aa + s_bb + _C2), 1e-9
    )


def _tile_reduce(img: np.ndarray, tile: int) -> np.ndarray:
    h, w = img.shape
    th, tw = h // tile, w // tile
    if th == 0 or tw == 0:
        return img.mean()[None, None]
    return (
        img[: th * tile, : tw * tile]
        .reshape(th, tile, tw, tile)
        .mean(axis=(1, 3))
    )


def render_feedback(
    mesh,
    cameras,
    images_dir,
    image_map: dict | None = None,
    stats_out: dict | None = None,
    max_views: int = 240,
    scale: float = 0.12,
    tile: int = 32,
    min_coverage: float = 0.6,
    max_faces: int = 2_000_000,
):
    """Render the textured mesh from the camera poses and score it.

    Returns ``(face_heat, view_scores)``: per-face discrepancy in [0, 1]
    (NaN where unobserved) and the per-view mean SSIM list.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    from scantobim.core.texture import _index_images

    if mesh is None or not len(mesh.faces) or not cameras:
        return None
    index = dict(image_map) if image_map else _index_images(Path(images_dir))
    n_f = len(mesh.faces)
    # Face subsample: at thumbnail resolution 2M splats already cover
    # every pixel several times — a stride subset renders identically
    # and bounds the projection cost on 8M-face meshes.
    sub_idx = np.arange(n_f)
    if n_f > max_faces:
        sub_idx = np.linspace(0, n_f - 1, max_faces).astype(np.int64)
    centroids = mesh.vertices[mesh.faces[sub_idx]].mean(axis=1)
    colors = face_colors_from_atlas(mesh)[sub_idx]
    heat_sum = np.zeros(n_f, dtype=np.float64)
    heat_cnt = np.zeros(n_f, dtype=np.int64)
    view_scores = []

    sel = (
        list(np.linspace(0, len(cameras) - 1, max_views).astype(int))
        if len(cameras) > max_views else range(len(cameras))
    )
    for ci in sel:
        cam = cameras[ci]
        path = index.get(cam.name) or index.get(Path(cam.name).name)
        if path is None or not path.exists():
            continue
        rendered = render_view_splat(centroids, colors, cam, scale)
        if rendered is None:
            continue
        rgb, mask, pix_face = rendered
        h, w = mask.shape
        try:
            photo = np.asarray(
                Image.open(path).convert("RGB").resize(
                    (w, h), Image.BILINEAR
                )
            )
        except Exception:  # noqa: BLE001
            continue
        g_r = _gray(rgb)
        g_p = _gray(photo)
        ssim = _ssim_map(g_r, g_p)
        gy_r, gx_r = np.gradient(g_r)
        gy_p, gx_p = np.gradient(g_p)
        gdiff = np.abs(np.abs(gx_r) + np.abs(gy_r)
                       - np.abs(gx_p) - np.abs(gy_p)) / 255.0
        cover_t = _tile_reduce(mask.astype(np.float32), tile)
        ssim_t = _tile_reduce(ssim, tile)
        gdiff_t = _tile_reduce(gdiff, tile)
        valid_t = cover_t >= min_coverage
        if not valid_t.any():
            continue
        disc_t = np.clip(
            0.7 * (1.0 - ssim_t) + 0.3 * np.clip(gdiff_t * 4.0, 0, 1),
            0.0, 1.0,
        )
        view_scores.append(float(ssim_t[valid_t].mean()))
        # Accumulate tile discrepancy onto the faces that won pixels.
        th, tw = disc_t.shape
        py_t = np.minimum(
            np.arange(h) // tile, th - 1
        )
        px_t = np.minimum(np.arange(w) // tile, tw - 1)
        tile_of_pix = disc_t[py_t[:, None], px_t[None, :]]
        valid_pix = valid_t[py_t[:, None], px_t[None, :]]
        fsel = pix_face >= 0
        use = fsel & valid_pix
        if not use.any():
            continue
        full_ids = sub_idx[pix_face[use]]
        np.add.at(heat_sum, full_ids, tile_of_pix[use])
        np.add.at(heat_cnt, full_ids, 1)

    if not view_scores:
        return None
    heat = np.full(n_f, np.nan)
    seen = heat_cnt >= 2
    heat[seen] = heat_sum[seen] / heat_cnt[seen]
    if stats_out is not None:
        vs = np.array(view_scores)
        stats_out.update({
            "ansichten": int(len(view_scores)),
            "ssim_median": round(float(np.median(vs)), 3),
            "ssim_p10": round(float(np.percentile(vs, 10)), 3),
            "ssim_min": round(float(vs.min()), 3),
        })
    return heat, view_scores


def hot_face_boxes(mesh, face_heat, heat_min=0.45, cell=0.6):
    """Cluster high-discrepancy faces into world-space rebuild regions.

    Returns raw ``(lo, hi, weight)`` boxes on a coarse grid — the caller
    merges/dilates them (same machinery as the Bauteil-Nachbau).
    """
    hot = np.flatnonzero(
        np.nan_to_num(face_heat, nan=0.0) >= heat_min
    )
    if not len(hot):
        return []
    cent = mesh.vertices[mesh.faces[hot]].mean(axis=1)
    key = np.floor(cent / cell).astype(np.int64)
    _, inv, counts = np.unique(
        key, axis=0, return_inverse=True, return_counts=True
    )
    boxes = []
    for g in range(len(counts)):
        if counts[g] < 12:  # a couple of stray faces are not a region
            continue
        c = cent[inv == g]
        boxes.append((c.min(axis=0), c.max(axis=0), int(counts[g])))
    return boxes
