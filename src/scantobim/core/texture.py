"""Photo-realistic texturing of reconstructed models.

Two texture sources, both producing one packed texture atlas plus per-vertex
UV coordinates on a display mesh:

* :func:`bake_texture_from_cloud` — splats the RGB of the (photogrammetry or
  scanner) point cloud into per-surface texel grids. Works with any colored
  cloud; resolution follows the point density.
* :func:`bake_texture_from_photos` — projects the *original photos* onto the
  surfaces using the COLMAP camera poses: for every texel the best-viewing,
  non-occluded photo is selected (occlusion is ray-tested against the
  reconstructed mesh itself). This yields textures at full photo resolution,
  far sharper than the point cloud allows.

The returned mesh duplicates vertices along surface seams (a UV atlas
requires it); geometry formats (STEP/IFC) are unaffected — texturing only
feeds the visual exports (GLB, HTML, OBJ).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from scantobim.core.cloud import PointCloud
from scantobim.core.mesh import Mesh

MAX_ATLAS = 4096
GUTTER = 2  # texel padding between atlas charts


# ------------------------------------------------------------------ plane frame

def plane_frame(normal: np.ndarray, origin: np.ndarray):
    """Deterministic in-plane basis (same convention as Plane.make_basis)."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return np.asarray(origin, dtype=np.float64), u, v


@dataclass
class _Chart:
    surface_index: int
    umin: float
    vmin: float
    texel: float
    width: int  # texels
    height: int
    x0: int = 0  # atlas position
    y0: int = 0
    image: np.ndarray | None = None  # (height, width, 3) uint8
    filled: np.ndarray | None = None


# ------------------------------------------------------------ cloud color bake

def bake_texture_from_cloud(
    result,
    cloud: PointCloud,
    texel_size: float | None = None,
    transform: np.ndarray | None = None,
) -> Mesh:
    """Bake the cloud's RGB into a texture atlas; returns the textured mesh.

    ``transform`` (4x4) maps cloud coordinates into model coordinates — pass
    the alignment matrix from the report when ``--align`` was used.
    """
    if cloud.colors is None:
        raise ValueError(
            "the input cloud carries no colors — capture with RGB (photogrammetry, "
            "RGB scanner) or use bake_texture_from_photos"
        )
    from scantobim.core.preprocess import estimate_point_spacing

    points = cloud.points
    if transform is not None:
        points = points @ transform[:3, :3].T + transform[:3, 3]
    colors = cloud.colors.astype(np.float64)

    spacing = estimate_point_spacing(cloud)
    texel = texel_size if texel_size else 2.0 * spacing
    if texel <= 0:
        raise ValueError("texel size must be positive")

    charts = _make_charts(result.surfaces, texel)
    tree_needed = False
    for chart in charts:
        surf = result.surfaces[chart.surface_index]
        origin, u, v = plane_frame(surf.normal, surf.outer[0])
        rel = points - origin
        dist = rel @ (np.cross(u, v))
        near = np.abs(dist) < max(4.0 * spacing, 1.5 * texel)
        if not near.any():
            chart.image = np.full((chart.height, chart.width, 3), 200, np.uint8)
            chart.filled = np.zeros((chart.height, chart.width), bool)
            tree_needed = True
            continue
        pu = rel[near] @ u
        pv = rel[near] @ v
        ix = ((pu - chart.umin) / chart.texel).astype(np.int64)
        iy = ((pv - chart.vmin) / chart.texel).astype(np.int64)
        ok = (ix >= 0) & (ix < chart.width) & (iy >= 0) & (iy < chart.height)
        flat = iy[ok] * chart.width + ix[ok]
        sums = np.zeros((chart.width * chart.height, 3))
        counts = np.zeros(chart.width * chart.height)
        np.add.at(sums, flat, colors[near][ok])
        np.add.at(counts, flat, 1.0)
        filled = counts > 0
        img = np.full((chart.width * chart.height, 3), 200.0)
        img[filled] = sums[filled] / counts[filled][:, None]
        chart.image = img.reshape(chart.height, chart.width, 3).astype(np.uint8)
        chart.filled = filled.reshape(chart.height, chart.width)
    _fill_holes(charts)
    atlas = _pack_atlas(charts)
    return _build_textured_mesh(result, charts, atlas)


# ------------------------------------------------------------------ photo bake

def bake_texture_from_photos(
    result,
    model_dir,
    images_dir,
    texel_size: float | None = None,
    transform: np.ndarray | None = None,
    max_cameras_per_surface: int = 8,
) -> Mesh:
    """Project the original photos onto the model via COLMAP camera poses.

    ``model_dir`` must contain the COLMAP model as text (``cameras.txt``,
    ``images.txt`` — export with ``colmap model_converter --output_type TXT``;
    ``scantobim photos`` does this automatically). Occlusion is handled by
    ray-testing every texel against the reconstructed mesh. Requires Pillow
    for reading the photos (``pip install scantobim[photos]``).
    """
    from pathlib import Path

    from scantobim.photogrammetry.colmap import read_colmap_model

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "photo texturing requires Pillow: pip install scantobim[photos]"
        ) from exc

    cameras = read_colmap_model(model_dir)
    if not cameras:
        raise ValueError(f"{model_dir}: no registered images in COLMAP model")
    images_dir = Path(images_dir)

    # Model-space camera data (apply alignment transform if given).
    rot_w = transform[:3, :3] if transform is not None else np.eye(3)
    t_w = transform[:3, 3] if transform is not None else np.zeros(3)
    cam_centers = []
    for cam in cameras:
        c = -cam.rotation.T @ cam.translation  # camera center, world coords
        cam_centers.append(rot_w @ c + t_w)

    if texel_size is None:
        # Derive from photo resolution: aim for ~1 texel per 2 photo pixels
        # at the median camera-to-model distance.
        center = np.vstack([s.outer for s in result.surfaces]).mean(axis=0)
        dists = [np.linalg.norm(c - center) for c in cam_centers]
        cam0 = cameras[0]
        texel_size = 2.0 * float(np.median(dists)) / cam0.fx
    texel = float(texel_size)

    charts = _make_charts(result.surfaces, texel)
    tri_all = result.mesh.vertices[result.mesh.faces]
    tri_groups = (
        result.mesh.face_groups
        if result.mesh.face_groups is not None
        else np.zeros(len(tri_all), dtype=int)
    )
    image_cache: dict[str, np.ndarray] = {}

    for chart in charts:
        surf = result.surfaces[chart.surface_index]
        origin, u, v = plane_frame(surf.normal, surf.outer[0])
        gx, gy = np.meshgrid(
            chart.umin + (np.arange(chart.width) + 0.5) * texel,
            chart.vmin + (np.arange(chart.height) + 0.5) * texel,
        )
        texels = origin + gx.reshape(-1, 1) * u + gy.reshape(-1, 1) * v
        n_tex = len(texels)
        img = np.full((n_tex, 3), 200.0)
        filled = np.zeros(n_tex, bool)

        # Rank cameras for this surface: frontal + close first. The facing
        # test is sign-agnostic (surface normal orientation is a convention);
        # actual visibility is decided by the occlusion ray test below.
        surf_center = surf.outer.mean(axis=0)
        scores = []
        for ci, cam in enumerate(cameras):
            view = cam_centers[ci] - surf_center
            d = np.linalg.norm(view)
            if d < 1e-9:
                scores.append(-np.inf)
                continue
            scores.append(abs(float(surf.normal @ (view / d))) / max(d, 1e-9))
        order = np.argsort(scores)[::-1][:max_cameras_per_surface]

        occluders = tri_all[tri_groups != surf.plane_index]
        for ci in order:
            if filled.all():
                break
            if not np.isfinite(scores[ci]) or scores[ci] <= 1e-12:
                continue
            cam = cameras[ci]
            todo = ~filled
            pts = texels[todo]
            # World → camera (undo alignment first).
            pts_scan = (pts - t_w) @ rot_w
            pc = pts_scan @ cam.rotation.T + cam.translation
            in_front = pc[:, 2] > 1e-6
            px, py = cam.project(pc)
            in_img = (
                in_front
                & (px >= 0) & (px <= cam.width - 1)
                & (py >= 0) & (py <= cam.height - 1)
            )
            if not in_img.any():
                continue
            visible = in_img & ~_occluded(
                pts, cam_centers[ci], occluders
            )
            if not visible.any():
                continue
            name = cam.name
            if name not in image_cache:
                path = images_dir / name
                if not path.exists():
                    continue
                image_cache[name] = np.asarray(Image.open(path).convert("RGB"))
            photo = image_cache[name]
            sy = np.clip(py[visible].round().astype(int), 0, photo.shape[0] - 1)
            sx = np.clip(px[visible].round().astype(int), 0, photo.shape[1] - 1)
            idx_todo = np.flatnonzero(todo)
            img[idx_todo[visible]] = photo[sy, sx]
            filled[idx_todo[visible]] = True

        chart.image = img.reshape(chart.height, chart.width, 3).astype(np.uint8)
        chart.filled = filled.reshape(chart.height, chart.width)
    _fill_holes(charts)
    atlas = _pack_atlas(charts)
    return _build_textured_mesh(result, charts, atlas)


def _occluded(points: np.ndarray, cam_center: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Möller–Trumbore ray test: is the segment camera→point blocked?"""
    n = len(points)
    blocked = np.zeros(n, bool)
    if len(triangles) == 0:
        return blocked
    d = points - cam_center  # ray directions, t in (0, 1) hits before the texel
    for tri in triangles:
        e1 = tri[1] - tri[0]
        e2 = tri[2] - tri[0]
        pvec = np.cross(d, e2)
        det = pvec @ e1
        good = np.abs(det) > 1e-14
        if not good.any():
            continue
        inv = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
        tvec = cam_center - tri[0]
        uu = (pvec @ tvec) * inv
        qvec = np.cross(tvec, e1)
        vv = np.einsum("ij,j->i", d, qvec) * inv
        tt = (qvec @ e2) * inv
        hit = good & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (tt > 1e-4) & (tt < 1 - 1e-4)
        blocked |= hit
        if blocked.all():
            break
    return blocked


# ---------------------------------------------------------------- chart layout

def _make_charts(surfaces, texel: float) -> list[_Chart]:
    if not surfaces:
        raise ValueError("no surfaces to texture")
    charts = []
    # Global budget: shrink resolution if the summed area explodes.
    total_texels = 0
    boxes = []
    for si, surf in enumerate(surfaces):
        origin, u, v = plane_frame(surf.normal, surf.outer[0])
        rel = surf.outer - origin
        pu, pv = rel @ u, rel @ v
        umin, vmin = float(pu.min()) - texel, float(pv.min()) - texel
        umax, vmax = float(pu.max()) + texel, float(pv.max()) + texel
        boxes.append((umin, vmin, umax, vmax))
        total_texels += ((umax - umin) / texel) * ((vmax - vmin) / texel)
    budget = 0.6 * MAX_ATLAS * MAX_ATLAS
    scale = max(1.0, np.sqrt(total_texels / budget))
    texel_eff = texel * scale
    for si, surf in enumerate(surfaces):
        umin, vmin, umax, vmax = boxes[si]
        w = max(2, int(np.ceil((umax - umin) / texel_eff)))
        h = max(2, int(np.ceil((vmax - vmin) / texel_eff)))
        w, h = min(w, MAX_ATLAS - 2 * GUTTER), min(h, MAX_ATLAS - 2 * GUTTER)
        charts.append(
            _Chart(surface_index=si, umin=umin, vmin=vmin, texel=texel_eff, width=w, height=h)
        )
    return charts


def _fill_holes(charts: list[_Chart]) -> None:
    """Nearest-neighbour fill of unpainted texels (occlusions, sparse spots)."""
    for chart in charts:
        filled = chart.filled
        if filled is None or filled.all() or not filled.any():
            continue
        fy, fx = np.nonzero(filled)
        ey, ex = np.nonzero(~filled)
        tree = cKDTree(np.column_stack([fx, fy]))
        _, nearest = tree.query(np.column_stack([ex, ey]), k=1, workers=-1)
        chart.image[ey, ex] = chart.image[fy[nearest], fx[nearest]]
        chart.filled[:] = True


def _pack_atlas(charts: list[_Chart]) -> np.ndarray:
    """Shelf-pack all charts into one atlas (sorted by height, power-of-two)."""
    order = sorted(range(len(charts)), key=lambda i: -charts[i].height)
    total_area = sum((c.width + 2 * GUTTER) * (c.height + 2 * GUTTER) for c in charts)
    width = int(min(MAX_ATLAS, _pow2(int(np.sqrt(total_area) * 1.2) + 1)))
    width = max(width, max(c.width + 2 * GUTTER for c in charts))
    width = int(min(MAX_ATLAS, _pow2(width)))

    x = y = shelf_h = 0
    for i in order:
        c = charts[i]
        w, h = c.width + 2 * GUTTER, c.height + 2 * GUTTER
        if x + w > width:
            y += shelf_h
            x = 0
            shelf_h = 0
        c.x0, c.y0 = x + GUTTER, y + GUTTER
        x += w
        shelf_h = max(shelf_h, h)
    height = int(_pow2(y + shelf_h))
    if height > MAX_ATLAS:
        raise ValueError(
            "texture atlas exceeds 4096px — increase texel size (--texel)"
        )

    atlas = np.full((height, width, 3), 190, np.uint8)
    for c in charts:
        atlas[c.y0 : c.y0 + c.height, c.x0 : c.x0 + c.width] = c.image
        # Gutter: replicate border pixels to avoid bleeding at chart seams.
        atlas[c.y0 - 1, c.x0 : c.x0 + c.width] = c.image[0]
        atlas[c.y0 + c.height, c.x0 : c.x0 + c.width] = c.image[-1]
        atlas[c.y0 : c.y0 + c.height, c.x0 - 1] = c.image[:, 0]
        atlas[c.y0 : c.y0 + c.height, c.x0 + c.width] = c.image[:, -1]
    return atlas


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _build_textured_mesh(result, charts: list[_Chart], atlas: np.ndarray) -> Mesh:
    """Duplicate vertices per surface and attach atlas UVs (glTF convention)."""
    mesh = result.mesh
    chart_of = {result.surfaces[c.surface_index].plane_index: c for c in charts}
    groups = (
        mesh.face_groups
        if mesh.face_groups is not None
        else np.zeros(len(mesh.faces), dtype=int)
    )
    ah, aw = atlas.shape[:2]

    verts_out, uvs_out, faces_out, groups_out = [], [], [], []
    base = 0
    for g in sorted(set(int(x) for x in groups)):
        fsel = groups == g
        vids = np.unique(mesh.faces[fsel])
        remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
        remap[vids] = np.arange(len(vids)) + base
        verts = mesh.vertices[vids]

        chart = chart_of.get(g)
        if chart is not None:
            surf = result.surfaces[chart.surface_index]
            origin, u, v = plane_frame(surf.normal, surf.outer[0])
            rel = verts - origin
            px = chart.x0 + (rel @ u - chart.umin) / chart.texel
            py = chart.y0 + (rel @ v - chart.vmin) / chart.texel
            uv = np.column_stack([px / aw, py / ah])
        else:
            uv = np.zeros((len(verts), 2))
        verts_out.append(verts)
        uvs_out.append(uv)
        faces_out.append(remap[mesh.faces[fsel]])
        groups_out.append(np.full(fsel.sum(), g, dtype=np.int64))
        base += len(vids)

    textured = Mesh(
        vertices=np.vstack(verts_out),
        faces=np.vstack(faces_out),
        face_groups=np.concatenate(groups_out),
        group_names=mesh.group_names,
        uvs=np.vstack(uvs_out),
        texture=atlas,
    )
    return textured
