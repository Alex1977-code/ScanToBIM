"""Automatic scene segmentation (Objekt- und Geländeerkennung).

Outdoor SLAM scans are not one object but a SCENE: terrain, one or more
buildings, vegetation, parked vehicles, masts, railings/fences, signs.
Meshing them all with one voxel surface makes terrain holey (scan
shadows), trees confetti and vehicles permanent. This module labels the
POINT CLOUD before any meshing so every class gets the treatment it
needs:

* ``gelaende``   — digital terrain model (DTM) from per-cell low
  quantiles; meshed as a CLOSED 2.5D grid instead of a holey voxel
  surface.
* ``fahrzeug``   — compact clusters sitting on the ground at vehicle
  height: temporary objects, removed from the model and listed in the
  report.
* ``vegetation`` — green (excess-green index) or amorphous clusters:
  excluded from the building mesh (they only produce fray).
* ``stabwerk``   — thin linear/vertical clusters (masts, railings,
  fences): kept, but flagged for parametric rebuild.
* ``rest``       — everything else (buildings and unknowns) stays and
  takes the normal reconstruction path.

The classification is grid/cluster-based (no per-point kNN): a 0.5 m
DTM raster, height above ground, connected voxel clusters and per-
cluster shape/color features. 20M points classify in seconds.
"""

from __future__ import annotations

import numpy as np

from scantobim.core.mesh import Mesh

# Class ids (uint8 labels).
REST = 0
GELAENDE = 1
FAHRZEUG = 2
VEGETATION = 3
STABWERK = 4

CLASS_NAMES = {
    REST: "rest",
    GELAENDE: "gelaende",
    FAHRZEUG: "fahrzeug",
    VEGETATION: "vegetation",
    STABWERK: "stabwerk",
}

_DTM_CELL = 0.5       # DTM raster (m)
_GROUND_BAND = 0.20   # points this close above the DTM are terrain (m)
_CLUSTER_VOXEL = 0.3  # occupancy voxel for clustering non-ground points


def build_dtm(points: np.ndarray, cell: float = _DTM_CELL):
    """Digital terrain model: per-cell low quantile, outlier-robust.

    Returns ``(dtm, lo, cell)``: a 2D height grid (NaN-free — empty
    cells are filled from their nearest carrier via EDT) and the grid
    origin. The 5th percentile per cell ignores points on roofs only
    where the ground was seen at all; pure-roof cells are then clamped
    against their neighborhood so buildings do not become terrain hills:
    a cell whose height jumps more than ``cell`` (45 degrees) above the
    3x3 neighborhood minimum takes that neighborhood floor instead.
    """
    from scipy import ndimage

    lo = points[:, :2].min(axis=0)
    ij = ((points[:, :2] - lo) / cell).astype(np.int64)
    nx, ny = ij.max(axis=0) + 1
    lin = ij[:, 0] * ny + ij[:, 1]
    order = np.argsort(lin, kind="stable")
    lin_s = lin[order]
    z_s = points[order, 2]
    starts = np.flatnonzero(np.concatenate([[True], lin_s[1:] != lin_s[:-1]]))
    counts = np.diff(np.concatenate([starts, [len(lin_s)]]))
    grid = np.full(nx * ny, np.nan)
    # 5th percentile per occupied cell (vectorized via sorted segments).
    z_sorted = np.empty_like(z_s)
    for s, c in zip(starts, counts):
        seg = np.sort(z_s[s:s + c])
        z_sorted[s:s + c] = seg
    q_idx = starts + np.minimum((counts * 0.05).astype(np.int64), counts - 1)
    grid[lin_s[starts]] = z_sorted[q_idx]
    dtm = grid.reshape(nx, ny)

    filled = np.isfinite(dtm)
    if not filled.any():
        return None
    # Fill empty cells from the nearest occupied cell.
    _, (fi, fj) = ndimage.distance_transform_edt(
        ~filled, return_indices=True
    )
    dtm = dtm[fi, fj]
    # Clamp roof-only cells against the neighborhood floor (slope limit).
    for _ in range(6):
        floor = ndimage.minimum_filter(dtm, size=3)
        high = dtm > floor + cell  # > 45 degrees against a neighbor
        if not high.any():
            break
        dtm[high] = floor[high] + cell
    dtm = ndimage.median_filter(dtm, size=3)
    return dtm, lo, cell


def height_above_ground(points: np.ndarray, dtm_pack) -> np.ndarray:
    dtm, lo, cell = dtm_pack
    ij = ((points[:, :2] - lo) / cell).astype(np.int64)
    i = np.clip(ij[:, 0], 0, dtm.shape[0] - 1)
    j = np.clip(ij[:, 1], 0, dtm.shape[1] - 1)
    return points[:, 2] - dtm[i, j]


def _cluster_labels(points: np.ndarray, voxel: float):
    """Connected components (26-neighborhood) on an occupancy grid."""
    from scipy import ndimage

    lo = points.min(axis=0)
    ijk = ((points - lo) / voxel).astype(np.int64)
    dims = ijk.max(axis=0) + 1
    if int(np.prod(dims)) > 400_000_000:  # scene too large -> coarser grid
        voxel *= 2.0
        ijk = ((points - lo) / voxel).astype(np.int64)
        dims = ijk.max(axis=0) + 1
    occ = np.zeros(dims, dtype=bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3), dtype=np.int8))
    return lab[ijk[:, 0], ijk[:, 1], ijk[:, 2]], n


def classify_scene(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    stats_out: dict | None = None,
) -> np.ndarray | None:
    """Label every point: gelaende/fahrzeug/vegetation/stabwerk/rest.

    Pure geometry + color — needs NO structure analysis, so it can run
    before the parallel full-mesh/structure stages. Returns uint8 labels
    (see CLASS_NAMES) or None when no terrain is identifiable.
    """
    dtm_pack = build_dtm(points)
    if dtm_pack is None:
        return None
    hag = height_above_ground(points, dtm_pack)
    labels = np.zeros(len(points), dtype=np.uint8)
    labels[hag < _GROUND_BAND] = GELAENDE

    above = np.flatnonzero(labels != GELAENDE)
    if not len(above):
        return labels
    cl, n_cl = _cluster_labels(points[above], _CLUSTER_VOXEL)
    green = None
    if colors is not None:
        c = colors.astype(np.float32)
        green = (2 * c[:, 1] - c[:, 0] - c[:, 2]) > 24.0  # excess green

    order = np.argsort(cl, kind="stable")
    cl_s = cl[order]
    starts = np.flatnonzero(np.concatenate([[True], cl_s[1:] != cl_s[:-1]]))
    bounds = np.concatenate([starts, [len(cl_s)]])
    n_veh = n_veg = n_stab = 0
    for b in range(len(starts)):
        idx = above[order[bounds[b]:bounds[b + 1]]]
        if len(idx) < 40:
            continue
        p = points[idx]
        ext = p.max(axis=0) - p.min(axis=0)
        h_lo = float(hag[idx].min())
        height = float(ext[2])
        foot = float(np.hypot(ext[0], ext[1]))
        # PCA shape features on a sample of the cluster.
        sample = p if len(p) <= 4000 else p[:: len(p) // 4000]
        centered = sample - sample.mean(axis=0)
        ev = np.linalg.eigvalsh(centered.T @ centered / len(sample))
        ev = np.sqrt(np.maximum(ev, 1e-12))[::-1]  # sd along principal axes
        linearity = (ev[0] - ev[1]) / ev[0]
        sphericity = ev[2] / ev[0]
        green_frac = float(green[idx].mean()) if green is not None else 0.0

        # A green cluster is vegetation at ANY size — buildings are not
        # green. Everything else that is building-sized stays REST.
        if green_frac > 0.30:
            labels[idx] = VEGETATION
            n_veg += 1
            continue
        if foot > 12.0 or height > 6.0 or len(idx) > 600_000:
            continue
        if sphericity > 0.35 and height > 0.8 and green_frac > 0.12:
            labels[idx] = VEGETATION
            n_veg += 1
        elif (h_lo < 0.6 and 1.0 <= height <= 3.2
              and 1.4 <= foot <= 8.0 and linearity < 0.9
              and len(idx) > 1500):
            labels[idx] = FAHRZEUG
            n_veh += 1
        elif linearity > 0.85 or min(ext[0], ext[1]) < 0.35:
            labels[idx] = STABWERK
            n_stab += 1

    if stats_out is not None:
        cnt = np.bincount(labels, minlength=len(CLASS_NAMES))
        stats_out["szene"] = {
            "punkte": {
                CLASS_NAMES[k]: int(cnt[k]) for k in CLASS_NAMES
                if cnt[k]
            },
            "cluster": {
                "fahrzeuge": n_veh,
                "vegetation": n_veg,
                "stabwerk": n_stab,
            },
            "dtm_zellen": int(np.isfinite(dtm_pack[0]).sum()),
        }
    return labels


def terrain_mesh(
    points: np.ndarray,
    labels: np.ndarray,
    colors: np.ndarray | None = None,
    cell: float = 0.25,
) -> Mesh | None:
    """CLOSED 2.5D terrain grid mesh from the ground-labeled points.

    Only cells the scan actually observed (directly or within one cell)
    are meshed — the DTM fill values elsewhere would invent terrain.
    Vertex colors come from the ground points' median color per cell.
    """
    from scipy import ndimage

    g = labels == GELAENDE
    if int(g.sum()) < 1000:
        return None
    gp = points[g]
    dtm_pack = build_dtm(gp, cell=cell)
    if dtm_pack is None:
        return None
    dtm, lo, cell = dtm_pack
    nx, ny = dtm.shape
    ij = ((gp[:, :2] - lo) / cell).astype(np.int64)
    seen = np.zeros((nx, ny), dtype=bool)
    seen[np.clip(ij[:, 0], 0, nx - 1), np.clip(ij[:, 1], 0, ny - 1)] = True
    seen = ndimage.binary_closing(seen, iterations=2)

    vid = np.full((nx + 1, ny + 1), -1, dtype=np.int64)
    # A grid VERTEX is used when any adjacent cell is observed.
    used = np.zeros((nx + 1, ny + 1), dtype=bool)
    used[:-1, :-1] |= seen
    used[1:, :-1] |= seen
    used[:-1, 1:] |= seen
    used[1:, 1:] |= seen
    vi, vj = np.nonzero(used)
    vid[vi, vj] = np.arange(len(vi))
    # Vertex height: mean of adjacent observed cells' DTM.
    acc = np.zeros((nx + 1, ny + 1))
    cnt = np.zeros((nx + 1, ny + 1))
    for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
        sl = (slice(di, nx + di), slice(dj, ny + dj))
        acc[sl] += np.where(seen, dtm, 0.0)
        cnt[sl] += seen
    z = acc[vi, vj] / np.maximum(cnt[vi, vj], 1)
    verts = np.column_stack([
        lo[0] + vi * cell, lo[1] + vj * cell, z
    ])
    ci, cj = np.nonzero(seen)
    v00 = vid[ci, cj]
    v10 = vid[ci + 1, cj]
    v01 = vid[ci, cj + 1]
    v11 = vid[ci + 1, cj + 1]
    faces = np.concatenate([
        np.column_stack([v00, v10, v11]),
        np.column_stack([v00, v11, v01]),
    ])
    vcol = None
    if colors is not None:
        gc = colors[g]
        vcol = np.full((len(verts), 3), 128, dtype=np.uint8)
        lin = (
            np.clip(ij[:, 0], 0, nx - 1) * ny + np.clip(ij[:, 1], 0, ny - 1)
        )
        med = np.zeros((nx * ny, 3), dtype=np.float64)
        npts = np.zeros(nx * ny, dtype=np.int64)
        np.add.at(med, lin, gc.astype(np.float64))
        np.add.at(npts, lin, 1)
        cell_col = np.full((nx * ny, 3), 128.0)
        has = npts > 0
        cell_col[has] = med[has] / npts[has, None]
        cc = cell_col.reshape(nx, ny, 3)
        # Vertex color: nearest adjacent observed cell's mean color.
        src = np.minimum(vi, nx - 1), np.minimum(vj, ny - 1)
        vcol = cc[src].astype(np.uint8)
    return Mesh(vertices=verts, faces=faces, vertex_colors=vcol)
