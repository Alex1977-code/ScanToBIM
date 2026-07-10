"""Plane regularization: parallel / orthogonal / coplanar snapping.

Man-made structures are dominated by a handful of directions (Manhattan-world
assumption). Measured planes deviate from them by sensor noise and
registration error; snapping them back is what turns "almost straight" into
*clean* edges:

1. Cluster plane normals into direction groups (sign-invariant).
2. Snap groups that are nearly orthogonal to each other to exactly 90°,
   starting from the strongest group (most supporting points).
3. Refit each plane's offset with its normal fixed to the group direction.
4. Merge planes that end up coplanar (same direction, offset within
   tolerance) — they are one physical surface split by occlusion.

This is the regularization strategy used by state-of-the-art urban / indoor
reconstruction systems (cf. GlobFit, PolyFit, kinetic space partitioning).
"""

from __future__ import annotations

import numpy as np

from scantobim.core.planes import Plane, fit_plane_tls


def regularize_planes(
    planes: list[Plane],
    points: np.ndarray,
    parallel_tol_deg: float = 8.0,
    ortho_tol_deg: float = 8.0,
    merge_offset_tol: float = 0.05,
    ghost_offset_tol: float = 0.0,
) -> list[Plane]:
    """Regularize plane orientations in place and merge coplanar duplicates.

    ``ghost_offset_tol`` additionally merges *registration ghosts*: the same
    physical surface scanned from two poorly registered stations appears as
    two parallel planes a few millimetres apart with strongly overlapping
    footprints. Set it to the expected registration error (e.g. 0.01 m);
    0 disables ghost merging. Ghosts closer than the normal-estimation
    radius (~4x point spacing) never appear as two planes in the first
    place — detection absorbs them into one slightly thicker plane whose
    RMS then reflects the registration error. Coplanar merging always requires overlapping
    footprints so that separate surfaces on the same infinite plane (e.g.
    window sills on one wall line) stay separate surfaces.
    """
    if not planes:
        return planes

    # ---- 1. direction clustering (sign-invariant) -------------------------
    weights = np.array([len(p.inliers) for p in planes], dtype=np.float64)
    normals = np.array([p.normal for p in planes])
    cos_parallel = np.cos(np.deg2rad(parallel_tol_deg))

    group_of = np.full(len(planes), -1, dtype=int)
    group_dirs: list[np.ndarray] = []
    for idx in np.argsort(-weights):  # strongest planes define directions
        n = normals[idx]
        assigned = False
        for g, gdir in enumerate(group_dirs):
            if abs(float(n @ gdir)) > cos_parallel:
                group_of[idx] = g
                assigned = True
                break
        if not assigned:
            group_of[idx] = len(group_dirs)
            group_dirs.append(n.copy())

    # Weighted mean direction per group (sign-aligned to the seed).
    for g in range(len(group_dirs)):
        members = np.flatnonzero(group_of == g)
        acc = np.zeros(3)
        for m in members:
            n = normals[m]
            if float(n @ group_dirs[g]) < 0:
                n = -n
            acc += weights[m] * n
        group_dirs[g] = acc / np.linalg.norm(acc)

    # ---- 2. orthogonality snapping ----------------------------------------
    group_weight = np.array(
        [weights[group_of == g].sum() for g in range(len(group_dirs))]
    )
    order = np.argsort(-group_weight)
    cos_ortho_lo = np.cos(np.deg2rad(90.0 + ortho_tol_deg))
    cos_ortho_hi = np.cos(np.deg2rad(90.0 - ortho_tol_deg))
    fixed: list[int] = []
    for g in order:
        d = group_dirs[g]
        # Project out every already-fixed direction we are nearly orthogonal to.
        for f in fixed:
            c = float(d @ group_dirs[f])
            if cos_ortho_lo < c < cos_ortho_hi:
                d = d - (d @ group_dirs[f]) * group_dirs[f]
                norm = np.linalg.norm(d)
                if norm < 1e-6:
                    d = group_dirs[g]
                    break
                d = d / norm
        group_dirs[g] = d
        fixed.append(g)

    # ---- 3. refit offsets with fixed normals ------------------------------
    for i, plane in enumerate(planes):
        gdir = group_dirs[group_of[i]]
        if float(plane.normal @ gdir) < 0:
            gdir = -gdir
        plane.normal = gdir.copy()
        support = points[plane.inliers]
        plane.d = -float(np.mean(support @ plane.normal))
        residuals = support @ plane.normal + plane.d
        plane.rms = float(np.sqrt(np.mean(residuals**2)))
        plane.make_basis()

    # ---- 4. merge coplanar duplicates and registration ghosts -------------
    effective_tol = max(merge_offset_tol, ghost_offset_tol)
    merged: list[Plane] = []
    used = np.zeros(len(planes), dtype=bool)
    for i in range(len(planes)):
        if used[i]:
            continue
        group = [i]
        for j in range(i + 1, len(planes)):
            if used[j] or group_of[j] != group_of[i]:
                continue
            same_side = float(planes[i].normal @ planes[j].normal) > 0
            dj = planes[j].d if same_side else -planes[j].d
            offset = abs(planes[i].d - dj)
            if offset >= effective_tol:
                continue
            # Beyond the base tolerance only ghost pairs merge, and ghosts
            # (like all coplanar duplicates worth merging) must overlap.
            if not _footprints_overlap(planes[i], planes[j], points):
                continue
            group.append(j)
        if len(group) == 1:
            merged.append(planes[i])
            used[i] = True
            continue
        inliers = np.concatenate([planes[j].inliers for j in group])
        for j in group:
            used[j] = True
        normal = planes[i].normal.copy()
        d = -float(np.mean(points[inliers] @ normal))
        residuals = points[inliers] @ normal + d
        plane = Plane(
            normal=normal,
            d=d,
            inliers=inliers,
            rms=float(np.sqrt(np.mean(residuals**2))),
        )
        plane.make_basis()
        merged.append(plane)

    return merged


def _footprints_overlap(a: Plane, b: Plane, points: np.ndarray, min_ratio: float = 0.3) -> bool:
    """Do the two planes' supports overlap when projected onto plane a?"""
    if a.basis is None:
        a.make_basis()
    ua = a.project_to_2d(points[a.inliers])
    ub = a.project_to_2d(points[b.inliers])
    lo = np.maximum(ua.min(axis=0), ub.min(axis=0))
    hi = np.minimum(ua.max(axis=0), ub.max(axis=0))
    if np.any(hi <= lo):
        return False
    inter = float(np.prod(hi - lo))
    area_a = float(np.prod(ua.max(axis=0) - ua.min(axis=0)))
    area_b = float(np.prod(ub.max(axis=0) - ub.min(axis=0)))
    return inter >= min_ratio * max(min(area_a, area_b), 1e-12)


def snapped_angles_report(planes: list[Plane]) -> dict:
    """Statistics over pairwise plane angles — for the quality report."""
    n = len(planes)
    angles = []
    for i in range(n):
        for j in range(i + 1, n):
            c = abs(float(planes[i].normal @ planes[j].normal))
            angles.append(float(np.rad2deg(np.arccos(np.clip(c, -1, 1)))))
    if not angles:
        return {"pairs": 0}
    arr = np.array(angles)
    return {
        "pairs": len(arr),
        "exactly_parallel": int(np.sum(arr < 0.01)),
        "exactly_orthogonal": int(np.sum(np.abs(arr - 90.0) < 0.01)),
        "other": int(np.sum((arr >= 0.01) & (np.abs(arr - 90.0) >= 0.01))),
    }
