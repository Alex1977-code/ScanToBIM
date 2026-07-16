"""Freeform surface reconstruction of the plane-residual points (hybrid model).

The structured pipeline explains walls, roofs, cylinders — but curved steel
gates, railings, machinery and vegetation stay behind as *residual points*.
This module wraps them in a real triangle mesh so the final model is a true
**hybrid**: crisp parametric surfaces where structure exists, a faithful
free-form skin everywhere else.

Method (robust on noisy SLAM data, pure numpy/scipy):

1. **Sparse voxel occupancy** — the residual is voxelized on an adaptive
   grid (packed 21-bit integer keys, no dense allocation); cells with fewer
   than 2 points are treated as noise.
2. **Component filter** — connected components (6-neighbourhood, via
   ``scipy.sparse.csgraph``) below a point-count threshold are dropped, so
   stray speckles do not become floating shards.
3. **Boundary extraction** — every voxel face between occupied and empty
   space becomes a quad (outward winding), welded into a closed shell.
4. **Smoothing + data snap** — Laplacian relaxation removes the voxel
   staircase, then every vertex is pulled back towards its nearest measured
   point, so the skin follows the scan rather than the grid.
5. **Coloring** — vertex colors come from the nearest residual point, i.e.
   the same photo-derived point colors as the rest of the model.

The face budget is met by growing the voxel size, never by dropping
geometry arbitrarily.
"""

from __future__ import annotations

import numpy as np

from scantobim.core.cloud import PointCloud
from scantobim.core.mesh import Mesh

FREEFORM_GROUP = 20000

# Corner offsets (in grid corner coordinates) of the quad emitted for a
# boundary face in direction d, ordered so the two triangles wind CCW seen
# from OUTSIDE (from the empty neighbour).
_QUAD_CORNERS = {
    (1, 0, 0): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    (-1, 0, 0): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
    (0, 1, 0): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
    (0, -1, 0): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    (0, 0, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    (0, 0, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
}

_FIELD_BITS = 21
_FIELD_MASK = (1 << _FIELD_BITS) - 1


def _pack(idx: np.ndarray) -> np.ndarray:
    return (
        (idx[:, 0].astype(np.int64) << (2 * _FIELD_BITS))
        | (idx[:, 1].astype(np.int64) << _FIELD_BITS)
        | idx[:, 2].astype(np.int64)
    )


def _member(keys: np.ndarray, sorted_set: np.ndarray) -> np.ndarray:
    """Vectorized membership test against a sorted unique key array."""
    pos = np.searchsorted(sorted_set, keys)
    pos = np.minimum(pos, len(sorted_set) - 1)
    return sorted_set[pos] == keys


def freeform_mesh_from_points(
    cloud: PointCloud,
    spacing: float | None = None,
    voxel: float | None = None,
    min_component_points: int = 300,
    max_faces: int = 800_000,
) -> Mesh | None:
    """Watertight-ish free-form skin around unstructured scan points.

    Returns ``None`` when there is not enough coherent geometry.
    """
    pts = np.asarray(cloud.points, dtype=np.float64)
    if len(pts) < min_component_points:
        return None
    if spacing is None:
        from scantobim.core.preprocess import estimate_point_spacing

        spacing = estimate_point_spacing(cloud)
    if voxel is None:
        # Fine start: railings and steel members are only a few cm thick.
        # The budget loop below grows the voxel as far as necessary.
        voxel = max(2.5 * spacing, 1e-6)

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)

    for _ in range(48):
        extent = np.maximum((hi - lo) / voxel, 1.0)
        if float(extent.max()) > _FIELD_MASK - 4:
            voxel *= 1.5
            continue
        origin = lo - 1.5 * voxel  # 1-cell empty margin, no index ever 0
        idx = np.floor((pts - origin) / voxel).astype(np.int64)
        keys = _pack(idx)
        from scantobim.core.accel import unique_i64

        occ, inverse, counts = unique_i64(
            keys, return_inverse=True, return_counts=True
        )
        solid = counts >= 2
        if not solid.any():
            solid = counts >= 1
        # A voxel finer than the point density supports fragments the shell
        # (half the cells hold a single point and fall to the noise gate) —
        # grow until most occupied cells are multi-point.
        if solid.sum() < 0.6 * len(occ) and voxel < 20.0 * spacing:
            voxel *= 1.35
            continue
        occ_solid = occ[solid]  # sorted (np.unique)
        occ_points = counts[solid]

        # Estimate boundary quads; grow the voxel if over budget.
        n_quads = 0
        offsets = {
            d: (d[0] << (2 * _FIELD_BITS)) | (d[1] << _FIELD_BITS) | d[2]
            for d in _QUAD_CORNERS
        }
        boundary_masks = {}
        for d, off in offsets.items():
            m = ~_member(occ_solid + off, occ_solid)
            boundary_masks[d] = m
            n_quads += int(m.sum())
        if 2 * n_quads <= max_faces:
            break
        voxel *= 1.35
    else:
        return None
    if n_quads == 0:
        return None

    # ---- connected components over solid voxels (6-neighbourhood) ----------
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n_vox = len(occ_solid)
    rows, cols = [], []
    for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        off = offsets[d]
        pos = np.searchsorted(occ_solid, occ_solid + off)
        pos = np.minimum(pos, n_vox - 1)
        hit = occ_solid[pos] == occ_solid + off
        rows.append(np.flatnonzero(hit))
        cols.append(pos[hit])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    adj = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_vox, n_vox)
    )
    n_comp, labels = connected_components(adj, directed=False)
    comp_points = np.bincount(labels, weights=occ_points, minlength=n_comp)
    keep_comp = comp_points >= min_component_points
    keep_vox = keep_comp[labels]
    if not keep_vox.any():
        return None
    kept_fraction = float(occ_points[keep_vox].sum() / max(len(pts), 1))

    # ---- boundary quads of the kept voxels ----------------------------------
    kept_keys = occ_solid[keep_vox]  # still sorted
    ix = (kept_keys >> (2 * _FIELD_BITS)) & _FIELD_MASK
    iy = (kept_keys >> _FIELD_BITS) & _FIELD_MASK
    iz = kept_keys & _FIELD_MASK
    vox_idx = np.column_stack([ix, iy, iz])

    corner_keys_parts = []
    tri_parts = []
    corner_count = 0
    for d, corners in _QUAD_CORNERS.items():
        off = offsets[d]
        outside = ~_member(kept_keys + off, kept_keys)
        base = vox_idx[outside]
        if len(base) == 0:
            continue
        # (Q, 4, 3) corner grid coordinates → packed keys
        quad = base[:, None, :] + np.asarray(corners, dtype=np.int64)[None, :, :]
        qk = _pack(quad.reshape(-1, 3))
        n_q = len(base)
        i0 = corner_count + np.arange(n_q) * 4
        tri_parts.append(
            np.column_stack([i0, i0 + 1, i0 + 2, i0, i0 + 2, i0 + 3]).reshape(-1, 3)
        )
        corner_keys_parts.append(qk)
        corner_count += n_q * 4

    corner_keys = np.concatenate(corner_keys_parts)
    tris = np.vstack(tri_parts)
    from scantobim.core.accel import unique_i64 as _uniq

    uniq_corners, corner_map = _uniq(corner_keys, return_inverse=True)
    faces = corner_map[tris]

    cx = (uniq_corners >> (2 * _FIELD_BITS)) & _FIELD_MASK
    cy = (uniq_corners >> _FIELD_BITS) & _FIELD_MASK
    cz = uniq_corners & _FIELD_MASK
    vertices = np.column_stack([cx, cy, cz]).astype(np.float64) * voxel + origin

    # ---- smooth the staircase, then pull the skin onto the scan -------------
    # Taubin lambda/mu schedule: smooths the voxel steps WITHOUT the global
    # shrinking/melting of plain Laplacian passes.
    vertices = _taubin(vertices, faces, iterations=3)
    from scipy.spatial import cKDTree

    sample = pts
    if len(sample) > 2_000_000:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(sample), 2_000_000, replace=False)
        sample = sample[sel]
        sample_colors = None if cloud.colors is None else cloud.colors[sel]
    else:
        sample_colors = cloud.colors
    tree = cKDTree(sample)
    dist, nearest = tree.query(vertices, k=1, workers=-1)
    pull = sample[nearest] - vertices
    step = np.linalg.norm(pull, axis=1, keepdims=True)
    cap = 0.75 * voxel
    scale = np.where(step > cap, cap / np.maximum(step, 1e-12), 1.0)
    vertices = vertices + 0.8 * pull * scale
    vertices = _taubin(vertices, faces, iterations=1)

    colors = (
        sample_colors[nearest].astype(np.uint8)
        if sample_colors is not None
        else np.full((len(vertices), 3), 150, dtype=np.uint8)
    )

    mesh = Mesh(
        vertices=vertices,
        faces=faces.astype(np.int64),
        vertex_colors=colors,
        face_groups=np.full(len(faces), FREEFORM_GROUP, dtype=np.int64),
        group_names={FREEFORM_GROUP: "freiform"},
    )
    mesh.freeform_stats = {  # type: ignore[attr-defined]
        "triangles": int(len(faces)),
        "vertices": int(len(vertices)),
        "components": int(keep_comp.sum()),
        "voxel": round(float(voxel), 4),
        "points_covered": round(kept_fraction, 4),
    }
    return mesh


def _taubin(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int,
    lam: float = 0.5,
    mu: float = -0.53,
) -> np.ndarray:
    """Taubin smoothing: a shrink step (λ) followed by an inflate step (μ).

    Unlike plain Laplacian smoothing this keeps the enclosed volume — thin
    members and sharp corners stop "melting" away.
    """
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    ii = np.concatenate([edges[:, 0], edges[:, 1]])
    jj = np.concatenate([edges[:, 1], edges[:, 0]])
    from scantobim.core.accel import asnumpy, xp_for

    xp = xp_for(len(vertices), min_gpu=500_000)
    dtype = xp.float32 if xp is not np else np.float64
    ii_x = xp.asarray(ii)
    jj_x = xp.asarray(jj)
    deg = xp.bincount(ii_x, minlength=len(vertices)).astype(dtype)
    deg = xp.maximum(deg, 1.0)
    v = xp.asarray(vertices, dtype=dtype)

    def step(v, factor):
        acc = xp.zeros_like(v)
        for c in range(3):
            acc[:, c] = xp.bincount(
                ii_x, weights=v[jj_x, c], minlength=len(v)
            )
        mean = acc / deg[:, None]
        return v + factor * (mean - v)

    try:
        for _ in range(iterations):
            v = step(v, lam)
            v = step(v, mu)
        return np.asarray(asnumpy(v), dtype=np.float64)
    except Exception:
        # GPU hiccup → plain numpy retry.
        v = vertices.copy()
        deg_c = np.maximum(
            np.bincount(ii, minlength=len(vertices)).astype(np.float64), 1.0
        )
        for _ in range(iterations):
            for factor in (lam, mu):
                acc = np.zeros_like(v)
                for c in range(3):
                    acc[:, c] = np.bincount(
                        ii, weights=v[jj, c], minlength=len(v)
                    )
                v = v + factor * (acc / deg_c[:, None] - v)
        return v


def sharpen_mesh_with_planes(
    mesh: Mesh, surfaces, voxel: float, max_angle_deg: float = 32.0
) -> float:
    """Contour the free-form skin with the detected structure planes.

    Every mesh vertex that verifiably lies ON a detected plane (close in
    distance, normal-aligned, inside the plane's footprint) is projected
    exactly onto it — walls become dead flat. Vertices claimed by TWO
    non-parallel planes are pulled onto their intersection line — edges
    become crisp instead of molten. Returns the sharpened vertex fraction.
    """
    if not surfaces or not len(mesh.vertices):
        return 0.0
    v = mesh.vertices
    vn = mesh.vertex_normals()
    tol = 0.8 * voxel
    cos_tol = np.cos(np.deg2rad(max_angle_deg))

    # Up to two plane constraints per vertex. Membership is geometric
    # (distance + footprint); the normal gate is applied only for the
    # single-plane snap — edge vertices where two planes meet carry
    # diagonal normals and must still be pulled onto the crease line.
    n1 = np.zeros((len(v), 3))
    d1 = np.zeros(len(v))
    a1 = np.zeros(len(v), dtype=bool)  # normal-aligned with plane 1
    n2 = np.zeros((len(v), 3))
    d2 = np.zeros(len(v))
    count = np.zeros(len(v), dtype=np.int8)

    for geo in surfaces:
        n = np.asarray(geo.normal, dtype=np.float64)
        norm = float(np.linalg.norm(n))
        if norm < 1e-12 or len(geo.outer) < 3:
            continue
        n = n / norm
        d = -float(np.mean(geo.outer @ n))
        lo = geo.outer.min(axis=0) - 2.0 * voxel
        hi = geo.outer.max(axis=0) + 2.0 * voxel
        dist = v @ n + d
        ok = (np.abs(dist) < tol) & np.all((v >= lo) & (v <= hi), axis=1)
        if not ok.any():
            continue
        aligned = np.abs(vn @ n) > cos_tol
        first = ok & (count == 0)
        n1[first] = n
        d1[first] = d
        a1[first] = aligned[first]
        # Second constraint only if clearly non-parallel to the first.
        second = ok & (count == 1) & (np.abs(n1 @ n) < 0.94)
        n2[second] = n
        d2[second] = d
        count[first] += 1
        count[second] += 1

    single = (count == 1) & a1
    if single.any():
        dist = np.einsum("ij,ij->i", v[single], n1[single]) + d1[single]
        v[single] -= dist[:, None] * n1[single]

    double = count >= 2
    if double.any():
        # Project onto the intersection line of the two planes: solve the
        # 2x2 normal equations of the constraint pair per vertex.
        a = n1[double]
        b = n2[double]
        ra = np.einsum("ij,ij->i", v[double], a) + d1[double]
        rb = np.einsum("ij,ij->i", v[double], b) + d2[double]
        ab = np.einsum("ij,ij->i", a, b)
        det = 1.0 - ab**2
        det = np.where(np.abs(det) < 1e-9, 1e-9, det)
        la = (ra - ab * rb) / det
        lb = (rb - ab * ra) / det
        v[double] -= la[:, None] * a + lb[:, None] * b

    mesh.vertices = v
    return float((single | (count >= 2)).mean())


def sharpen_mesh_with_cylinders(
    mesh: Mesh, cylinders, voxel: float
) -> float:
    """Contour the skin with detected cylinders (pipes, columns, arches).

    Vertices whose radial distance to a detected cylinder axis matches the
    cylinder radius (within the voxel regime) are projected radially onto
    the exact cylinder surface — round members become truly round.
    ``cylinders`` are report entries with center/axis/radius/length.
    Returns the sharpened vertex fraction.
    """
    if not cylinders or not len(mesh.vertices):
        return 0.0
    v = mesh.vertices
    tol = 0.8 * voxel
    done = np.zeros(len(v), dtype=bool)
    for cyl in cylinders:
        axis = np.asarray(cyl["axis"], dtype=np.float64)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        center = np.asarray(cyl["center"], dtype=np.float64)
        radius = float(cyl["radius"])
        half = 0.5 * float(cyl["length"]) + voxel
        rel = v - center
        t = rel @ axis
        radial = rel - t[:, None] * axis
        r = np.linalg.norm(radial, axis=1)
        ok = (
            (~done)
            & (np.abs(r - radius) < tol)
            & (np.abs(t) <= half)
            & (r > 1e-9)
        )
        if not ok.any():
            continue
        v[ok] = center + t[ok, None] * axis + radial[ok] * (radius / r[ok])[:, None]
        done |= ok
    mesh.vertices = v
    return float(done.mean())
