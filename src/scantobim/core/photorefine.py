"""Photoconsistent mesh refinement (photokonsistentes Feintuning).

The RefineMesh principle, natively implemented: vertices of the detail
mesh are moved along their normal so that small surface patches
reproject CONSISTENTLY into all observing photos (pairwise NCC).
Where the LiDAR smears an edge or a profile by a centimeter, the photos
agree only at the true surface — the vertex slides there.

Only high-curvature vertices are refined (edges, profiles, rims — flat
walls are already plane-projected and must not pick up photo noise),
the search is capped at ±2 cm, and the shift field is smoothed over the
mesh neighborhood so no vertex spikes out of its surroundings.
Processing is tiled to bound memory on large meshes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_SEARCH_M = 0.02      # ± along the vertex normal
_STEPS = 9
_PATCH_M = 0.012      # tangent patch half-extent (3x3 samples)
_NCC_ACCEPT = 0.45    # patch consistency needed to trust a move
_MIN_IMPROVE = 0.04   # score gain over the current position


def _bilinear(gray, x, y):
    h, w = gray.shape
    x = np.clip(x, 0.0, w - 1.001)
    y = np.clip(y, 0.0, h - 1.001)
    x0 = x.astype(np.int64)
    y0 = y.astype(np.int64)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    return (
        gray[y0, x0] * (1 - fx) + gray[y0, x0 + 1] * fx
    ) * (1 - fy) + (
        gray[y0 + 1, x0] * (1 - fx) + gray[y0 + 1, x0 + 1] * fx
    ) * fy


def _vertex_normals_and_curvature(mesh):
    v, f = mesh.vertices, mesh.faces
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.maximum(ln, 1e-12)
    vn = np.zeros_like(v)
    cnt = np.zeros(len(v))
    sq = np.zeros(len(v))
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
        np.add.at(cnt, f[:, k], 1.0)
        np.add.at(sq, f[:, k], (fn**2).sum(axis=1))
    vn_len = np.linalg.norm(vn, axis=1)
    # Resultant length ≪ count ⇒ adjacent face normals disagree ⇒ edge.
    curvature = 1.0 - vn_len / np.maximum(cnt, 1.0)
    normals = vn / np.maximum(vn_len, 1e-12)[:, None]
    return normals, curvature


def refine_mesh_photoconsistent(
    mesh,
    cameras,
    images_dir,
    image_map: dict | None = None,
    stats_out: dict | None = None,
    max_vertices: int = 200_000,
    scale: float = 0.3,
    tile_size: int = 30_000,
    max_photos: int = 48,
) -> int:
    """Move edge vertices onto the photo-consistent surface (in place).

    Returns the number of vertices adjusted.
    """
    try:
        from PIL import Image
    except ImportError:
        return 0
    from scantobim.core.texture import _index_images, _resolve_cameras

    cameras = _resolve_cameras(cameras)
    if not cameras or mesh is None or len(mesh.faces) == 0:
        return 0
    index = dict(image_map) if image_map else _index_images(Path(images_dir))
    v = mesh.vertices
    normals, curvature = _vertex_normals_and_curvature(mesh)
    cand = np.flatnonzero(curvature > 0.05)
    if not len(cand):
        return 0
    if len(cand) > max_vertices:
        order = np.argsort(curvature[cand])[::-1]
        cand = cand[order[:max_vertices]]
    cand = np.sort(cand)

    centers = np.array([-c.rotation.T @ c.translation for c in cameras])
    photo_cache: dict[int, np.ndarray | None] = {}

    def _gray_for(ci: int):
        if ci in photo_cache:
            return photo_cache[ci]
        if sum(1 for g in photo_cache.values() if g is not None) >= max_photos:
            return None
        cam = cameras[ci]
        path = index.get(cam.name) or index.get(Path(cam.name).name)
        val = None
        if path is not None and path.exists():
            try:
                img = Image.open(path).convert("L")
                w = max(2, int(round(img.width * scale)))
                h = max(2, int(round(img.height * scale)))
                val = np.asarray(
                    img.resize((w, h), Image.BILINEAR), dtype=np.float32
                )
            except Exception:  # noqa: BLE001
                val = None
        photo_cache[ci] = val
        return val

    deltas = np.linspace(-_SEARCH_M, _SEARCH_M, _STEPS)
    # Tangent 3x3 patch offsets.
    grid = np.array([-1.0, 0.0, 1.0]) * _PATCH_M
    gu, gv = np.meshgrid(grid, grid)
    gu, gv = gu.ravel(), gv.ravel()  # (P,)
    p_n = len(gu)

    shift = np.zeros(len(v))
    adjusted = np.zeros(len(v), dtype=bool)

    # Spatial tiling: process candidates in world-sorted chunks so each
    # tile sees few cameras and memory stays bounded.
    cell = np.floor(v[cand] / 3.0).astype(np.int64)
    order = np.lexsort((cell[:, 2], cell[:, 1], cell[:, 0]))
    cand = cand[order]

    for start in range(0, len(cand), tile_size):
        tile = cand[start:start + tile_size]
        tv = v[tile]
        tn = normals[tile]
        # Tangent basis per vertex.
        helper = np.where(
            np.abs(tn[:, 2:3]) < 0.9,
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        tu = np.cross(tn, helper)
        tu /= np.maximum(np.linalg.norm(tu, axis=1, keepdims=True), 1e-12)
        tw = np.cross(tn, tu)
        mid = tv.mean(axis=0)
        d2 = np.einsum("ij,ij->i", centers - mid, centers - mid)
        cam_order = np.argsort(d2)[:6]
        # Patch world points: (n, steps, P, 3)
        base = (
            tv[:, None, None, :]
            + tn[:, None, None, :] * deltas[None, :, None, None]
        )
        patch = (
            base
            + tu[:, None, None, :] * gu[None, None, :, None]
            + tw[:, None, None, :] * gv[None, None, :, None]
        )
        flat = patch.reshape(-1, 3)
        vals_per_cam = []
        for ci in cam_order:
            gray = _gray_for(int(ci))
            if gray is None:
                continue
            cam = cameras[ci]
            pc = flat @ cam.rotation.T + cam.translation
            ok = pc[:, 2] > 0.3
            if ok.mean() < 0.5:
                continue
            px, py = cam.project(np.maximum(pc, [-1e9, -1e9, 0.05]))
            real_scale = gray.shape[1] / cam.width
            vals = _bilinear(gray, px * real_scale, py * real_scale)
            vals[~ok] = np.nan
            vals_per_cam.append(vals.reshape(len(tile), _STEPS, p_n))
            if len(vals_per_cam) >= 3:
                break
        if len(vals_per_cam) < 2:
            continue
        # Pairwise NCC over the patch axis, averaged over pairs.
        score = np.zeros((len(tile), _STEPS))
        pairs = 0
        for a in range(len(vals_per_cam)):
            for b in range(a + 1, len(vals_per_cam)):
                va, vb = vals_per_cam[a], vals_per_cam[b]
                va = va - np.nanmean(va, axis=2, keepdims=True)
                vb = vb - np.nanmean(vb, axis=2, keepdims=True)
                va = np.nan_to_num(va)
                vb = np.nan_to_num(vb)
                sa = np.sqrt((va**2).mean(axis=2)) + 1e-4
                sb = np.sqrt((vb**2).mean(axis=2)) + 1e-4
                score += (va * vb).mean(axis=2) / (sa * sb)
                pairs += 1
        score /= max(pairs, 1)
        kbest = np.argmax(score, axis=1)
        rows = np.arange(len(tile))
        best = score[rows, kbest]
        at_zero = score[:, _STEPS // 2]
        move = (
            (best >= _NCC_ACCEPT)
            & (best - at_zero >= _MIN_IMPROVE)
            & (kbest != _STEPS // 2)
        )
        if not move.any():
            continue
        km = np.clip(kbest, 1, _STEPS - 2)
        g0 = score[rows, km - 1]
        g1 = score[rows, km]
        g2 = score[rows, km + 1]
        den = g0 - 2 * g1 + g2
        den = np.where(den > -1e-6, -1e-6, den)
        sub = np.clip(0.5 * (g0 - g2) / den, -1.0, 1.0)
        step_m = deltas[1] - deltas[0]
        d_move = deltas[km] + sub * step_m
        shift[tile[move]] = np.clip(d_move[move], -_SEARCH_M, _SEARCH_M)
        adjusted[tile[move]] = True

    n_adj = int(adjusted.sum())
    if n_adj == 0:
        if stats_out is not None:
            stats_out["photo_feintuning"] = {"vertices": 0}
        return 0
    # Diffuse the shift field over the mesh edges (2 rounds) so refined
    # vertices never spike out of an unrefined neighborhood.
    f = mesh.faces
    ea = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    eb = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    for _ in range(2):
        nb_sum = np.zeros(len(v))
        nb_cnt = np.zeros(len(v))
        np.add.at(nb_sum, ea, shift[eb])
        np.add.at(nb_cnt, ea, 1.0)
        nb_mean = nb_sum / np.maximum(nb_cnt, 1.0)
        shift = np.where(adjusted, 0.7 * shift + 0.3 * nb_mean,
                         0.5 * nb_mean)
    v += normals * shift[:, None]
    if stats_out is not None:
        moved_mm = np.abs(shift[adjusted]) * 1000.0
        stats_out["photo_feintuning"] = {
            "vertices": n_adj,
            "median_verschiebung_mm": round(float(np.median(moved_mm)), 1),
        }
    return n_adj
