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


def _pack_charts(extents: np.ndarray, max_atlas: int, min_texel: float = 5e-4):
    """Choose a texel size and shelf-pack all charts into one atlas.

    ``extents``: (C, 2) chart sizes in meters. ``min_texel`` floors the
    resolution (finer than the photos can resolve is wasted atlas). Returns
    ``(texel, positions (C,2), atlas_w, atlas_h)``.
    """
    total_area = float(np.prod(extents + 1e-6, axis=1).sum())
    texel = max(
        np.sqrt(total_area / (0.70 * max_atlas * max_atlas)), min_texel, 5e-4
    )

    for _ in range(10):
        w = np.ceil(extents[:, 0] / texel).astype(np.int64) + 1
        h = np.ceil(extents[:, 1] / texel).astype(np.int64) + 1
        w = np.clip(w, 1, max_atlas - 2 * _GUTTER)
        h = np.clip(h, 1, max_atlas - 2 * _GUTTER)
        order = np.argsort(-h)
        atlas_w = max_atlas
        x = y = shelf = 0
        pos = np.zeros((len(extents), 2), dtype=np.int64)
        for i in order:
            cw, ch = int(w[i]) + 2 * _GUTTER, int(h[i]) + 2 * _GUTTER
            if x + cw > atlas_w:
                y += shelf
                x = 0
                shelf = 0
            pos[i] = (x + _GUTTER, y + _GUTTER)
            x += cw
            shelf = max(shelf, ch)
        height = y + shelf
        if height <= max_atlas:
            atlas_h = 1
            while atlas_h < height:
                atlas_h *= 2
            atlas_h = min(atlas_h, max_atlas)
            return texel, pos, atlas_w, atlas_h
        texel *= 1.18
    raise ValueError("Textur-Atlas passt nicht — max_atlas erhöhen")


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
                min_facing, max_used_cameras,
            )
        except Exception:  # noqa: BLE001 — any GPU hiccup → CPU fallback
            pass
    return _assign_cameras_xp(
        np, centers, normals, verts, cameras, image_index,
        min_facing, max_used_cameras,
    )


def _assign_cameras_xp(
    xp, centers, normals, verts, cameras, image_index,
    min_facing, max_used_cameras,
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
        facing = xp.abs((normals * view).sum(axis=1)) / xp.sqrt(
            xp.maximum(d2, 1e-12)
        )
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
    image_index = _index_images(Path(images_dir))
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
    src_colors = (
        mesh.vertex_colors
        if mesh.vertex_colors is not None
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

    # Resolution floor from the photos' ground sample distance: half a
    # photo pixel at the median shooting distance.
    from pathlib import Path as _Path

    cam_centers = np.array([
        -cam.rotation.T @ cam.translation
        for cam in cameras
        if cam.name in image_index or _Path(cam.name).name in image_index
    ])
    scene_center = v_scan.mean(axis=0)
    med_dist = float(np.median(np.linalg.norm(cam_centers - scene_center, axis=1)))
    med_fx = float(np.median([cam.fx for cam in cameras]))
    gsd = med_dist / max(med_fx, 1.0)
    texel, pos, atlas_w, atlas_h = _pack_charts(
        extents, max_atlas, min_texel=0.5 * gsd
    )
    vertices = np.vstack(verts_out)
    colors = np.vstack(colors_out)
    uv_texel = np.vstack(uv_m) / texel
    chart_of_vertex = np.repeat(
        np.arange(n_charts), [len(v) for v in verts_out]
    )
    uv_texel += pos[chart_of_vertex]
    uvs = uv_texel / np.array([atlas_w, atlas_h])

    textured = Mesh(
        vertices=vertices,
        faces=new_faces,
        vertex_colors=colors,
        face_groups=mesh.face_groups,
        group_names=mesh.group_names,
        uvs=uvs.astype(np.float64),
        texture=None,
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
        min_facing, max_used_cameras,
    )

    # --- rasterize + sample ----------------------------------------------
    atlas = np.full((atlas_h, atlas_w, 3), 190, dtype=np.uint8)
    filled = np.zeros((atlas_h, atlas_w), dtype=bool)
    photo_mask = np.zeros((atlas_h, atlas_w), dtype=bool)

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
                        color[ok] = _bilinear(photo, px[ok], py[ok])
                        photo_mask[iy[ok], ix[ok]] = True
                atlas[iy, ix] = np.clip(color, 0, 255).astype(np.uint8)
                filled[iy, ix] = True

    # --- gutter: dilate filled colors so bilinear lookups never bleed grey
    for _ in range(_GUTTER):
        empty = ~filled
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = np.roll(filled, (dy, dx), axis=(0, 1))
            src_img = np.roll(atlas, (dy, dx), axis=(0, 1))
            take = empty & shifted
            atlas[take] = src_img[take]
            filled |= take
            empty = ~filled

    textured.texture = atlas
    if stats_out is not None:
        stats_out["atlas"] = [int(atlas_w), int(atlas_h)]
        stats_out["texel_cm"] = round(texel * 100.0, 2)
        stats_out["charts"] = int(n_charts)
        stats_out["coverage"] = round(float(filled.mean()), 3)
        stats_out["photo_fraction"] = round(
            int(photo_mask.sum()) / max(int(filled.sum()), 1), 3
        )
        stats_out["cameras_used"] = int(len([c for c, _ in cam_groups if c >= 0]))
        stats_out["faces_with_photo"] = round(
            float((best_cam >= 0).mean()), 3
        )
    return textured
