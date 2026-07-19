"""Photo-realistic texture atlas for the free-form complete mesh.

Vertex colors (one RGB per mesh vertex) blur the photos down to the mesh
resolution — a couple of centimeters. This module instead bakes a real
texture atlas at photo resolution onto the complete mesh, the technique
behind photogrammetry tools' "textured mesh" output (view-dependent
texture mapping):

1. **Charts** — faces are grouped by dominant normal direction into
   connected components and unwrapped by orthographic projection (the mesh
   comes from a voxel surface, so those charts are near-isometric), then
   shelf-packed into one atlas.
2. **Best view per face** — every camera scores each face (facing angle /
   distance²) and visibility is checked against a per-camera depth buffer
   rendered from the mesh itself (z-buffer occlusion, no ray casting).
3. **Sampling** — every atlas texel is rasterized (barycentric), projected
   into its face's winning photo and sampled at full photo resolution;
   texels no photo covers keep the interpolated vertex colors, so the
   atlas is always complete.

The result is a NEW mesh (vertices are duplicated along chart seams) with
``uvs`` + ``texture`` set — GLB, OBJ and the HTML viewer render it as a
photo-textured model.
"""

from __future__ import annotations

import numpy as np

from scantobim.core.mesh import Mesh

_GUTTER = 2  # texels between charts (bilinear-bleed guard)
_ZBUF_W = 200  # per-camera depth-buffer width (visibility test)


# ------------------------------------------------------------------ charts

def _build_charts(mesh: Mesh, smooth_iters: int = 3, min_chart_faces: int = 24):
    """Split faces into orthographic charts (dominant axis + connectivity).

    Returns ``(chart_of_face, chart_axes)`` where ``chart_axes[c]`` is the
    dominant-axis bin (0..5: ±x ±y ±z) of chart ``c``.

    Voxel meshes of real scans are bumpy — raw face normals flip bins every
    few faces and shatter the surface into hundreds of thousands of tiny
    charts whose per-chart padding then eats the whole atlas (observed:
    461k charts → forced 6 cm texels). Two counter-measures: the normals
    are SMOOTHED over the face adjacency before binning, and remaining
    mini-charts are MERGED into their largest neighbour until every chart
    carries a sensible face count.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    faces = mesh.faces
    n_f = len(faces)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edge_face = np.tile(np.arange(n_f), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    es, ef = edges[order], edge_face[order]
    same = np.all(es[1:] == es[:-1], axis=1)
    af1, af2 = ef[:-1][same], ef[1:][same]  # all adjacent face pairs

    fn = mesh.face_normals().copy()
    for _ in range(max(0, smooth_iters)):
        acc = fn.copy()
        np.add.at(acc, af1, fn[af2])
        np.add.at(acc, af2, fn[af1])
        norm = np.linalg.norm(acc, axis=1, keepdims=True)
        fn = np.divide(acc, norm, out=fn, where=norm > 1e-12)
    axis = np.argmax(np.abs(fn), axis=1)
    sign = np.take_along_axis(fn, axis[:, None], axis=1)[:, 0] < 0
    bins = axis * 2 + sign.astype(int)

    def _components(bins_arr):
        ok = bins_arr[af1] == bins_arr[af2]
        graph = sparse.coo_matrix(
            (np.ones(int(ok.sum()), dtype=np.int8), (af1[ok], af2[ok])),
            shape=(n_f, n_f),
        )
        return connected_components(graph, directed=False)

    n_charts, chart_of_face = _components(bins)
    for _ in range(4):
        sizes = np.bincount(chart_of_face, minlength=n_charts)
        tiny = sizes[chart_of_face] < min_chart_faces
        if not tiny.any() or tiny.all():
            break
        # Every face in a tiny chart adopts the bin of an adjacent face
        # that lives in a big chart (if any) — then recompute components.
        new_bins = bins.copy()
        for a, b in ((af1, af2), (af2, af1)):
            src_big = ~tiny[b]
            take = tiny[a] & src_big
            new_bins[a[take]] = bins[b[take]]
        if np.array_equal(new_bins, bins):
            break
        bins = new_bins
        n_charts, chart_of_face = _components(bins)

    chart_axes = np.zeros(n_charts, dtype=np.int64)
    chart_axes[chart_of_face] = bins  # any member face defines the bin
    return chart_of_face, chart_axes


_UV_AXES = {0: (1, 2), 1: (0, 2), 2: (0, 1)}  # dominant axis → (u, v) axes


def _chart_uv(points: np.ndarray, bin_id: int) -> np.ndarray:
    """Orthographic 2D coordinates (meters) for a chart's points."""
    ua, va = _UV_AXES[bin_id // 2]
    u = points[:, ua]
    if bin_id % 2:  # flip u for back-facing charts → consistent orientation
        u = -u
    return np.column_stack([u, points[:, va]])


def _pack_charts_pages(
    extents: np.ndarray, texel: float, page_size: int, max_pages: int
):
    """Shelf-pack padded charts into SQUARE pages at a FIXED texel size.

    ``extents``: (C, 2) chart sizes in meters. Returns
    ``(positions (C,2), page_of_chart (C,), n_pages)`` or ``None`` when more
    than ``max_pages`` pages would be needed at this texel size. The texel
    is chosen by the CALLER from the photo ground-sample distance — packing
    only decides placement, never resolution.
    """
    w = np.ceil(extents[:, 0] / texel).astype(np.int64) + 1
    h = np.ceil(extents[:, 1] / texel).astype(np.int64) + 1
    w = np.clip(w, 1, page_size - 2 * _GUTTER)
    h = np.clip(h, 1, page_size - 2 * _GUTTER)
    order = np.argsort(-h)
    pos = np.zeros((len(extents), 2), dtype=np.int64)
    page_of = np.zeros(len(extents), dtype=np.int64)
    page = 0
    x = y = shelf = 0
    for i in order:
        cw, ch = int(w[i]) + 2 * _GUTTER, int(h[i]) + 2 * _GUTTER
        if x + cw > page_size:
            y += shelf
            x = 0
            shelf = 0
        if y + ch > page_size:
            page += 1
            x = y = shelf = 0
            if page >= max_pages:
                return None
        pos[i] = (x + _GUTTER, y + _GUTTER)
        page_of[i] = page
        x += cw
        shelf = max(shelf, ch)
    return pos, page_of, page + 1


def _target_texel(v_scan: np.ndarray, cam_centers: np.ndarray, med_fx: float):
    """Photo ground-sample distance → texel target.

    GSD is what the NEAREST camera resolves on the surface: median over a
    vertex sample of (distance to closest camera) / fx. The former
    area-based heuristic let mesh resolution and packing pressure dictate
    the texel (observed: 6 cm texels on 4 mm photos) — the texel now comes
    from the photos alone, clamped to [3 mm, 25 mm].
    """
    from scipy.spatial import cKDTree

    if not len(cam_centers):
        return 0.01, {}
    vs = v_scan
    if len(vs) > 20_000:
        rng = np.random.default_rng(0)
        vs = vs[rng.choice(len(vs), 20_000, replace=False)]
    d, _ = cKDTree(np.asarray(cam_centers)).query(vs, k=1, workers=-1)
    # 25th percentile: robust when part of the camera path is far away.
    d_ref = float(np.percentile(d, 25))
    gsd = d_ref / max(med_fx, 1.0)
    texel = float(min(max(0.75 * gsd, 0.003), 0.025))
    diag = {
        "kamera_abstand_p25_m": round(d_ref, 2),
        "kamera_abstand_median_m": round(float(np.median(d)), 2),
        "gsd_mm": round(gsd * 1000.0, 1),
    }
    return texel, diag


# ----------------------------------------------------------- camera choice

def _rigid(xp, pts, rotation, translation):
    """``pts @ R.T + t`` with elementwise ops only (no cuBLAS on the GPU)."""
    r = xp.asarray(rotation)
    t = xp.asarray(translation)
    return xp.stack(
        [
            pts[:, 0] * r[0, 0] + pts[:, 1] * r[0, 1] + pts[:, 2] * r[0, 2] + t[0],
            pts[:, 0] * r[1, 0] + pts[:, 1] * r[1, 1] + pts[:, 2] * r[1, 2] + t[1],
            pts[:, 0] * r[2, 0] + pts[:, 1] * r[2, 1] + pts[:, 2] * r[2, 2] + t[2],
        ],
        axis=1,
    )


def _scatter_min(xp, target, index, values):
    if xp is np:
        np.minimum.at(target, index, values)
    else:  # pragma: no cover — GPU only
        import cupyx

        cupyx.scatter_min(target, index, values)


def _assign_cameras(
    centers: np.ndarray,
    normals: np.ndarray,
    verts: np.ndarray,
    cameras,
    image_index,
    min_facing: float,
    max_used_cameras: int,
    signed_facing: bool = False,
):
    """Best photo per face: facing/d² score + z-buffer visibility.

    ``centers``/``normals``: per face, ``verts``: mesh vertices — all in the
    scan/camera frame. Returns ``best_cam`` (F,) int32, −1 = no photo.
    Runs on the GPU when available (this camera sweep dominates the bake).
    """
    from scantobim.core import accel

    if len(verts) > 200_000:  # the depth buffer needs coverage, not density
        rng = np.random.default_rng(0)
        verts = verts[rng.choice(len(verts), 200_000, replace=False)]

    cp = accel.gpu()
    if cp is not None and len(centers) * max(len(cameras), 1) >= 2_000_000:
        try:
            return _assign_cameras_xp(
                cp, centers, normals, verts, cameras, image_index,
                min_facing, max_used_cameras, signed_facing,
            )
        except Exception:  # noqa: BLE001 — any GPU hiccup → CPU fallback
            pass
    return _assign_cameras_xp(
        np, centers, normals, verts, cameras, image_index,
        min_facing, max_used_cameras, signed_facing,
    )


def _assign_cameras_xp(
    xp, centers, normals, verts, cameras, image_index,
    min_facing, max_used_cameras, signed_facing=False,
):
    from pathlib import Path

    from scantobim.core.accel import asnumpy

    n_f = len(centers)
    dtype = xp.float32 if xp is not np else np.float64
    centers = xp.asarray(centers, dtype=dtype)
    normals = xp.asarray(normals, dtype=dtype)
    verts = xp.asarray(verts, dtype=dtype)
    best_score = xp.zeros(n_f, dtype=xp.float32)
    best_cam = xp.full(n_f, -1, dtype=xp.int32)

    for ci, cam in enumerate(cameras):
        if cam.name not in image_index and Path(cam.name).name not in image_index:
            continue
        zw = _ZBUF_W
        zh = max(2, int(round(zw * cam.height / max(cam.width, 1))))

        # Depth buffer from the mesh vertices (the mesh is dense enough
        # that vertex splats close the surface at this resolution).
        pv = _rigid(xp, verts, cam.rotation, cam.translation)
        vin = pv[:, 2] > 0.05
        if not bool(vin.any()):
            continue
        pvv = pv[vin]
        vx, vy = cam.project(pvv)
        gx = xp.clip((vx / cam.width * zw).astype(xp.int64), 0, zw - 1)
        gy = xp.clip((vy / cam.height * zh).astype(xp.int64), 0, zh - 1)
        inside = (vx >= 0) & (vx < cam.width) & (vy >= 0) & (vy < cam.height)
        zbuf = xp.full(zw * zh, np.inf, dtype=dtype)
        _scatter_min(
            xp, zbuf, (gy[inside] * zw + gx[inside]), pvv[inside][:, 2]
        )

        pc = _rigid(xp, centers, cam.rotation, cam.translation)
        in_front = pc[:, 2] > 0.05
        px, py = cam.project(pc)
        cam_center = -cam.rotation.T @ cam.translation
        view = xp.asarray(cam_center, dtype=dtype)[None, :] - centers
        d2 = (view * view).sum(axis=1)
        facing = (normals * view).sum(axis=1) / xp.sqrt(
            xp.maximum(d2, 1e-12)
        )
        if not signed_facing:
            # Geschlossene Meshes: beide Seiten zulassen, Z-Buffer regelt.
            # Dickenlose Strukturflaechen brauchen das VORZEICHEN, sonst
            # texturiert eine Kamera von hinten durch die Wand.
            facing = xp.abs(facing)
        ok = (
            in_front
            & (px >= 0) & (px <= cam.width - 1)
            & (py >= 0) & (py <= cam.height - 1)
            & (facing > min_facing)
        )
        if not bool(ok.any()):
            continue
        cx = xp.clip((px / cam.width * zw).astype(xp.int64), 0, zw - 1)
        cy = xp.clip((py / cam.height * zh).astype(xp.int64), 0, zh - 1)
        znear = zbuf[cy * zw + cx]
        margin = xp.maximum(0.12, 0.05 * pc[:, 2])
        visible = ok & (pc[:, 2] <= znear + margin)
        if not bool(visible.any()):
            continue
        score = (facing / xp.maximum(d2, 1e-6)).astype(xp.float32)
        better = visible & (score > best_score)
        best_score = xp.where(better, score, best_score)
        best_cam = xp.where(better, xp.int32(ci), best_cam)

    best_cam = asnumpy(best_cam)
    used = np.unique(best_cam[best_cam >= 0])
    if len(used) > max_used_cameras:
        wins = np.bincount(best_cam[best_cam >= 0], minlength=len(cameras))
        keep = np.argsort(wins)[::-1][:max_used_cameras]
        keep_mask = np.zeros(len(cameras), dtype=bool)
        keep_mask[keep] = True
        best_cam[(best_cam >= 0) & ~keep_mask[np.maximum(best_cam, 0)]] = -1
    return best_cam


def _smooth_camera_assignment(faces, best_cam, rounds: int = 2) -> int:
    """Majority-vote smoothing of the per-face camera choice (in place).

    A face is reassigned when at least 2 of its (up to 3) edge neighbors
    agree on a DIFFERENT camera. Only faces that already have a camera
    are touched — coverage never shrinks. Returns reassigned-face count.
    """
    n_f = len(faces)
    if n_f == 0:
        return 0
    f = np.asarray(faces, dtype=np.int64)
    n_v = int(f.max()) + 1
    # Edge codes (min*n_v+max) → adjacent face pairs via sort.
    ea = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    eb = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    codes = np.minimum(ea, eb) * n_v + np.maximum(ea, eb)
    face_of = np.tile(np.arange(n_f, dtype=np.int64), 3)
    order = np.argsort(codes, kind="stable")
    codes_s, face_s = codes[order], face_of[order]
    same = codes_s[1:] == codes_s[:-1]
    fa, fb = face_s[:-1][same], face_s[1:][same]
    if not len(fa):
        return 0
    # Neighbor table (F, 3), −1 = none — vectorized grouped fill.
    pf = np.concatenate([fa, fb])
    pn = np.concatenate([fb, fa])
    po = np.argsort(pf, kind="stable")
    pf, pn = pf[po], pn[po]
    starts = np.zeros(len(pf), dtype=np.int64)
    new_grp = np.flatnonzero(pf[1:] != pf[:-1]) + 1
    starts[new_grp] = new_grp
    np.maximum.accumulate(starts, out=starts)
    rank = np.arange(len(pf), dtype=np.int64) - starts
    ok3 = rank < 3
    nbr = np.full((n_f, 3), -1, dtype=np.int64)
    nbr[pf[ok3], rank[ok3]] = pn[ok3]
    total = 0
    for _ in range(max(0, int(rounds))):
        has = nbr >= 0
        ncam = np.where(has, best_cam[np.maximum(nbr, 0)], -1)
        n0, n1, n2 = ncam[:, 0], ncam[:, 1], ncam[:, 2]
        maj = np.full(n_f, -1, dtype=np.int32)
        m12 = (n1 == n2) & (n1 >= 0)
        maj[m12] = n1[m12]
        m02 = (n0 == n2) & (n0 >= 0)
        maj[m02] = n0[m02]
        m01 = (n0 == n1) & (n0 >= 0)
        maj[m01] = n0[m01]
        flip = (best_cam >= 0) & (maj >= 0) & (maj != best_cam)
        n_flip = int(flip.sum())
        if n_flip == 0:
            break
        best_cam[flip] = maj[flip]
        total += n_flip
    return total


# ------------------------------------------------------------- rasterizing

def _raster_batch(tu, tv, atlas_w, atlas_h):
    """Rasterize UV triangles (texel coords): all covered texel centers.

    ``tu``/``tv``: (B, 3) texel coordinates. Returns
    ``(face_idx, ix, iy, b0, b1, b2)`` with clipped, renormalized
    barycentrics — texel centers slightly outside a triangle are kept
    (eps 0.35) so neighbouring faces overlap and no cracks remain.
    """
    x0 = np.clip(np.floor(tu.min(axis=1)).astype(np.int64), 0, atlas_w - 1)
    x1 = np.clip(np.ceil(tu.max(axis=1)).astype(np.int64), 0, atlas_w - 1)
    y0 = np.clip(np.floor(tv.min(axis=1)).astype(np.int64), 0, atlas_h - 1)
    y1 = np.clip(np.ceil(tv.max(axis=1)).astype(np.int64), 0, atlas_h - 1)
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    counts = w * h
    total = int(counts.sum())
    if total == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, empty, empty, empty

    fidx = np.repeat(np.arange(len(tu)), counts)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    local = np.arange(total) - starts[fidx]
    ix = x0[fidx] + local % w[fidx]
    iy = y0[fidx] + local // w[fidx]
    px = ix.astype(np.float64) + 0.5
    py = iy.astype(np.float64) + 0.5

    au, av = tu[fidx, 0], tv[fidx, 0]
    e1u, e1v = tu[fidx, 1] - au, tv[fidx, 1] - av
    e2u, e2v = tu[fidx, 2] - au, tv[fidx, 2] - av
    den = e1u * e2v - e1v * e2u
    good = np.abs(den) > 1e-12
    den = np.where(good, den, 1.0)
    du, dv = px - au, py - av
    b1 = (du * e2v - dv * e2u) / den
    b2 = (e1u * dv - e1v * du) / den
    b0 = 1.0 - b1 - b2
    eps = 0.35
    inside = good & (b0 >= -eps) & (b1 >= -eps) & (b2 >= -eps)

    fidx, ix, iy = fidx[inside], ix[inside], iy[inside]
    b0, b1, b2 = b0[inside], b1[inside], b2[inside]
    b0, b1, b2 = np.clip(b0, 0, 1), np.clip(b1, 0, 1), np.clip(b2, 0, 1)
    s = np.maximum(b0 + b1 + b2, 1e-12)
    return fidx, ix, iy, b0 / s, b1 / s, b2 / s


def _bilinear(photo: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Bilinear photo sampling at float pixel coordinates → (N, 3) float."""
    h, w = photo.shape[:2]
    x = np.clip(px, 0.0, w - 1.001)
    y = np.clip(py, 0.0, h - 1.001)
    x0 = x.astype(np.int64)
    y0 = y.astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    p = photo.astype(np.float32)
    c00 = p[y0, x0]
    c10 = p[y0, x0 + 1]
    c01 = p[y0 + 1, x0]
    c11 = p[y0 + 1, x0 + 1]
    return (c00 * (1 - fx) + c10 * fx) * (1 - fy) + (c01 * (1 - fx) + c11 * fx) * fy


def _camera_gain_raw(
    photo, cam, tri_scan, col_f, fsel, sx, sy, max_samples=1200
):
    """Median per-channel ratio LiDAR-color / photo-color for one camera.

    ``sx``/``sy`` map full-resolution pixel coordinates to the (possibly
    draft-decoded) ``photo``. Returns None with too few valid samples.
    """
    take = fsel
    if len(take) > max_samples:
        take = take[
            np.linspace(0, len(take) - 1, max_samples).astype(np.int64)
        ]
    cent = tri_scan[take].mean(axis=1)
    pc = cent @ cam.rotation.T + cam.translation
    ok = pc[:, 2] > 0.05
    px, py = cam.project(pc)
    ok &= (
        (px >= 0) & (px <= cam.width - 1)
        & (py >= 0) & (py <= cam.height - 1)
    )
    if int(ok.sum()) < 40:
        return None
    sampled = _bilinear(photo, px[ok] * sx, py[ok] * sy)
    ref = col_f[take][ok].mean(axis=1)
    ratio = (
        (ref.astype(np.float64) + 8.0) / (sampled.astype(np.float64) + 8.0)
    )
    return np.median(ratio, axis=0)


def _solve_camera_gains(cameras, image_index, tri_scan, col_f, cam_groups):
    """RELATIVE per-camera color gains against the camera-median.

    Every photo carries its own auto-exposure and white balance — sampled
    side by side they produce the per-chart stripe pattern ("streifige"
    Textur). The colorized scan is ONE globally consistent reference, so
    the raw LiDAR/photo ratio captures each camera's exposure — but the
    LiDAR colors must not dictate the ABSOLUTE look (they are softer and
    darker than the photos). Dividing every raw gain by the median gain
    over all cameras cancels the reference out: inter-camera exposure
    jumps vanish, the photos' own color level stays. Photos are decoded
    at 1/8 draft resolution — plenty for a median color ratio.
    """
    from pathlib import Path

    from PIL import Image

    raw = {}
    for ci, fsel in cam_groups:
        if ci < 0 or not len(fsel):
            continue
        cam = cameras[ci]
        path = (
            image_index.get(cam.name)
            or image_index.get(Path(cam.name).name)
        )
        if path is None or not path.exists():
            continue
        try:
            im = Image.open(path)
            im.draft("RGB", (max(8, cam.width // 8), max(8, cam.height // 8)))
            photo = np.asarray(im.convert("RGB"))
        except Exception:  # noqa: BLE001 — one bad file must not kill the bake
            continue
        sy = photo.shape[0] / max(cam.height, 1)
        sx = photo.shape[1] / max(cam.width, 1)
        g = _camera_gain_raw(photo, cam, tri_scan, col_f, fsel, sx, sy)
        if g is not None:
            raw[ci] = g
    if len(raw) < 2:  # nothing to harmonize against
        return {}
    ref = np.median(np.array(list(raw.values())), axis=0)
    ref = np.maximum(ref, 1e-6)
    return {
        ci: np.clip(g / ref, 0.6, 1.6).astype(np.float32)
        for ci, g in raw.items()
    }


# ------------------------------------------------------------------- main

def bake_photo_atlas(
    mesh: Mesh,
    model_or_cameras,
    images_dir,
    transform: np.ndarray | None = None,
    max_atlas: int = 8192,
    min_facing: float = 0.2,
    max_used_cameras: int = 600,
    stats_out: dict | None = None,
    depth_points: np.ndarray | None = None,
    max_pages: int = 1,
    image_map: dict | None = None,
    signed_facing: bool = False,
) -> Mesh | None:
    """Bake a full-resolution photo texture atlas onto ``mesh``.

    Returns a new textured mesh (chart-seam vertices duplicated) or None
    when no usable camera/photo pair exists. ``mesh`` itself is untouched.

    ``depth_points`` (scan-frame, optional) densify the per-camera depth
    buffer used for the visibility test. The complete mesh is dense enough
    on its own — but a coarse structure mesh (a handful of polygon corners)
    is not, so callers pass a point-cloud sample to represent the real
    scene depth (anything the photos could not see through).
    """
    from pathlib import Path

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Foto-Textur benötigt Pillow: pip install scantobim[photos]"
        ) from exc
    from scantobim.core.texture import _index_images, _resolve_cameras

    cameras = _resolve_cameras(model_or_cameras)
    if not cameras or not len(mesh.faces):
        return None
    # A validated per-camera map (Kamera-Selbstprüfung) beats the ambiguous
    # basename index — stereo exports reuse filenames across left/right.
    image_index = (
        dict(image_map) if image_map else _index_images(Path(images_dir))
    )
    if not any(
        cam.name in image_index or Path(cam.name).name in image_index
        for cam in cameras
    ):
        return None

    rot_w = transform[:3, :3] if transform is not None else np.eye(3)
    t_w = transform[:3, 3] if transform is not None else np.zeros(3)
    v_scan = (mesh.vertices - t_w) @ rot_w  # model frame → scan/camera frame

    chart_of_face, chart_axes = _build_charts(mesh)
    n_charts = len(chart_axes)

    # --- duplicated-vertex mesh + chart-local UVs (meters) ---------------
    faces = mesh.faces
    order = np.argsort(chart_of_face, kind="stable")
    has_color_ref = mesh.vertex_colors is not None
    src_colors = (
        mesh.vertex_colors
        if has_color_ref
        else np.full((len(mesh.vertices), 3), 150, dtype=np.uint8)
    )

    verts_out, colors_out, uv_m = [], [], []
    face_chart_sorted = chart_of_face[order]
    bounds = np.searchsorted(face_chart_sorted, np.arange(n_charts + 1))
    extents = np.zeros((n_charts, 2))
    base = 0
    new_faces = np.empty_like(faces)
    remap = np.empty(len(mesh.vertices), dtype=np.int64)
    for c in range(n_charts):
        fsel = order[bounds[c]:bounds[c + 1]]
        vids = np.unique(faces[fsel])
        remap[vids] = np.arange(len(vids)) + base
        new_faces[fsel] = remap[faces[fsel]]
        verts_out.append(mesh.vertices[vids])
        colors_out.append(src_colors[vids])
        uv = _chart_uv(v_scan[vids], int(chart_axes[c]))
        lo = uv.min(axis=0)
        extents[c] = uv.max(axis=0) - lo
        uv_m.append(uv - lo)
        base += len(vids)

    # Texel size from the photo ground-sample distance (what the nearest
    # camera can actually resolve) — packing then spreads the charts over
    # as many square pages as allowed; only when even that is not enough
    # does the texel grow.
    from pathlib import Path as _Path

    cam_centers = np.array([
        -cam.rotation.T @ cam.translation
        for cam in cameras
        if cam.name in image_index or _Path(cam.name).name in image_index
    ])
    med_fx = float(np.median([cam.fx for cam in cameras]))
    texel, gsd_diag = _target_texel(v_scan, cam_centers, med_fx)
    packed = None
    for _ in range(8):
        packed = _pack_charts_pages(extents, texel, max_atlas, max_pages)
        if packed is not None:
            break
        texel *= 1.3
    if packed is None:
        return None
    pos, page_of_chart, n_pages = packed
    # Crop every page to its content extent (next power of two) — a small
    # scene or the last page must not ship as a mostly-empty full square.
    chart_w_tex = np.ceil(extents[:, 0] / texel).astype(np.int64) + 1
    chart_h_tex = np.ceil(extents[:, 1] / texel).astype(np.int64) + 1
    page_w = np.zeros(n_pages, dtype=np.int64)
    page_h = np.zeros(n_pages, dtype=np.int64)
    for c in range(len(extents)):
        p = page_of_chart[c]
        page_w[p] = max(page_w[p], pos[c, 0] + chart_w_tex[c] + _GUTTER)
        page_h[p] = max(page_h[p], pos[c, 1] + chart_h_tex[c] + _GUTTER)
    for p in range(n_pages):
        for arr in (page_w, page_h):
            v2 = 1
            while v2 < arr[p]:
                v2 *= 2
            arr[p] = min(v2, max_atlas)
    atlas_w = int(page_w.max())
    atlas_h = int(page_h.max())

    vertices = np.vstack(verts_out)
    colors = np.vstack(colors_out)
    uv_texel = np.vstack(uv_m) / texel
    chart_of_vertex = np.repeat(
        np.arange(n_charts), [len(v) for v in verts_out]
    )
    uv_texel += pos[chart_of_vertex]
    # Per-vertex normalization by the OWN page's cropped extent.
    vert_page = page_of_chart[chart_of_vertex]
    uvs = np.column_stack([
        uv_texel[:, 0] / page_w[vert_page].astype(np.float64),
        uv_texel[:, 1] / page_h[vert_page].astype(np.float64),
    ])
    face_page_arr = page_of_chart[chart_of_face]

    textured = Mesh(
        vertices=vertices,
        faces=new_faces,
        vertex_colors=colors,
        face_groups=mesh.face_groups,
        group_names=mesh.group_names,
        uvs=uvs.astype(np.float64),
        texture=None,
        face_page=face_page_arr if n_pages > 1 else None,
    )

    # --- best camera per face --------------------------------------------
    tri_scan = v_scan[faces]
    f_centers = tri_scan.mean(axis=1)
    f_normals = mesh.face_normals() @ rot_w
    depth_verts = v_scan
    if depth_points is not None and len(depth_points):
        dp_scan = (np.asarray(depth_points, dtype=np.float64) - t_w) @ rot_w
        depth_verts = np.vstack([v_scan, dp_scan])
    best_cam = _assign_cameras(
        f_centers, f_normals, depth_verts, cameras, image_index,
        min_facing, max_used_cameras, signed_facing=signed_facing,
    )
    # Anti-Schraffur: faces flickering between two near-equal cameras
    # sample the photos at slightly different exposure/parallax — that is
    # the per-face stripe pattern ("Schraffur") on walls and windows.
    # Majority smoothing snaps a face to the camera its neighbors agree
    # on, so seams collapse into few long borders instead of stripes.
    n_smoothed = _smooth_camera_assignment(faces, best_cam, rounds=2)

    # --- rasterize + sample ----------------------------------------------
    atlas_pages = [
        np.full((int(page_h[p]), int(page_w[p]), 3), 190, dtype=np.uint8)
        for p in range(n_pages)
    ]
    filled_pages = [
        np.zeros((int(page_h[p]), int(page_w[p])), dtype=bool)
        for p in range(n_pages)
    ]
    photo_pages = [
        np.zeros((int(page_h[p]), int(page_w[p])), dtype=bool)
        for p in range(n_pages)
    ]

    uvf = uv_texel[new_faces]  # (F, 3, 2) texel coordinates per face corner
    tu, tv = uvf[:, :, 0], uvf[:, :, 1]
    col_f = colors[new_faces].astype(np.float32)  # (F, 3 corners, 3 rgb)

    # Texel budget per raster batch — bbox areas bound the memory upfront.
    face_texels = (
        (np.ceil(tu.max(axis=1)) - np.floor(tu.min(axis=1)) + 1)
        * (np.ceil(tv.max(axis=1)) - np.floor(tv.min(axis=1)) + 1)
    ).astype(np.int64)

    cam_groups = [(-1, np.flatnonzero(best_cam < 0))]
    for ci in np.unique(best_cam[best_cam >= 0]):
        cam_groups.append((int(ci), np.flatnonzero(best_cam == ci)))

    def _load_photo(ci: int):
        cam = cameras[ci]
        path = image_index.get(cam.name) or image_index.get(Path(cam.name).name)
        if path is None or not path.exists():
            return None
        return np.asarray(Image.open(path).convert("RGB"))

    from concurrent.futures import ThreadPoolExecutor

    # Relative color harmonization: thumbnail pre-pass over all used
    # cameras, gains normalized to their median (see _solve_camera_gains).
    gains = (
        _solve_camera_gains(cameras, image_index, tri_scan, col_f, cam_groups)
        if has_color_ref
        else {}
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        photo_cis = [ci for ci, _ in cam_groups if ci >= 0]
        for ci in photo_cis[:4]:
            futures[ci] = pool.submit(_load_photo, ci)
        next_i = 4

        for ci, fsel in cam_groups:
            photo = None
            if ci >= 0:
                photo = futures.pop(ci).result()
                if next_i < len(photo_cis):
                    nc = photo_cis[next_i]
                    futures[nc] = pool.submit(_load_photo, nc)
                    next_i += 1
            cam = cameras[ci] if ci >= 0 else None
            gain = gains.get(ci)

            # Split into batches of ≤4M raster texels.
            cum = np.cumsum(face_texels[fsel])
            splits = np.searchsorted(cum, np.arange(1, cum[-1] // 4_000_000 + 1) * 4_000_000) if len(cum) else []
            for fs in np.array_split(fsel, splits) if len(fsel) else []:
                if not len(fs):
                    continue
                fidx, ix, iy, b0, b1, b2 = _raster_batch(
                    tu[fs], tv[fs], atlas_w, atlas_h
                )
                if not len(fidx):
                    continue
                g = fs[fidx]
                bar = np.stack([b0, b1, b2], axis=1)[:, :, None]
                color = (col_f[g] * bar).sum(axis=1)
                ok = np.zeros(len(g), dtype=bool)
                if photo is not None:
                    p3 = (tri_scan[g] * bar).sum(axis=1)
                    pc = p3 @ cam.rotation.T + cam.translation
                    ok = pc[:, 2] > 0.05
                    px, py = cam.project(pc)
                    ok &= (
                        (px >= 0) & (px <= cam.width - 1)
                        & (py >= 0) & (py <= cam.height - 1)
                    )
                    if ok.any():
                        sampled = _bilinear(photo, px[ok], py[ok])
                        if gain is not None:
                            sampled = sampled * gain[None, :]
                        color[ok] = sampled
                color8 = np.clip(color, 0, 255).astype(np.uint8)
                pg = face_page_arr[g]
                for p in np.unique(pg):
                    m = pg == p
                    iy_p = np.minimum(iy[m], atlas_pages[p].shape[0] - 1)
                    ix_p = np.minimum(ix[m], atlas_pages[p].shape[1] - 1)
                    atlas_pages[p][iy_p, ix_p] = color8[m]
                    filled_pages[p][iy_p, ix_p] = True
                    mo = m & ok
                    if mo.any():
                        iy_o = np.minimum(
                            iy[mo], atlas_pages[p].shape[0] - 1
                        )
                        ix_o = np.minimum(
                            ix[mo], atlas_pages[p].shape[1] - 1
                        )
                        photo_pages[p][iy_o, ix_o] = True

    # --- gutter: dilate filled colors so bilinear lookups never bleed grey
    for atlas, filled in zip(atlas_pages, filled_pages):
        for _ in range(_GUTTER):
            empty = ~filled
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                shifted = np.roll(filled, (dy, dx), axis=(0, 1))
                src_img = np.roll(atlas, (dy, dx), axis=(0, 1))
                take = empty & shifted
                atlas[take] = src_img[take]
                filled |= take
                empty = ~filled

    textured.textures = atlas_pages
    textured.texture = atlas_pages[0]
    n_filled = sum(int(f.sum()) for f in filled_pages)
    n_photo = sum(int(p.sum()) for p in photo_pages)
    if stats_out is not None:
        stats_out["atlas"] = [int(atlas_w), int(atlas_h)]
        stats_out["pages"] = int(n_pages)
        stats_out["texel_cm"] = round(texel * 100.0, 2)
        stats_out["charts"] = int(n_charts)
        stats_out["coverage"] = round(
            n_filled / max(sum(f.size for f in filled_pages), 1), 3
        )
        stats_out["photo_fraction"] = round(n_photo / max(n_filled, 1), 3)
        stats_out["cameras_used"] = int(len([c for c, _ in cam_groups if c >= 0]))
        stats_out["faces_with_photo"] = round(
            float((best_cam >= 0).mean()), 3
        )
        stats_out["kamera_glaettung"] = int(n_smoothed)
        if gains:
            g_lum = np.array(list(gains.values())).mean(axis=1)
            stats_out["farbabgleich"] = {
                "kameras": int(len(g_lum)),
                "gain_median": round(float(np.median(g_lum)), 3),
                "gain_p5": round(float(np.percentile(g_lum, 5)), 3),
                "gain_p95": round(float(np.percentile(g_lum, 95)), 3),
            }
        stats_out.update(gsd_diag)
    return textured
