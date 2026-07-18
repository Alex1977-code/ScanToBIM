"""Per-plane boundary polygon extraction and straightening.

Pipeline per plane:

1. Project the plane's inliers into its 2D frame.
2. Alpha shape (Delaunay triangles with circumradius < alpha) → outer
   boundary loop of the supported region.
3. Douglas–Peucker simplification removes noise-driven micro-vertices.
4. Dominant-direction straightening: boundary segments are clustered by
   orientation; each cluster snaps to its weighted mean direction, and the
   polygon is rebuilt by re-intersecting consecutive segment lines. This is
   the "O-Snap"-style regularization that turns jagged scan outlines into
   straight architectural edges.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay


def alpha_shape_boundary(uv: np.ndarray, alpha: float) -> np.ndarray | None:
    """Outer boundary polygon (2D, CCW) of the alpha shape of ``uv`` points.

    ``alpha`` is the circumradius limit — use ~3-4x the point spacing.
    Returns the largest closed boundary loop as an ``(M, 2)`` array, or
    ``None`` if no valid loop exists.
    """
    outer, _holes = alpha_shape_loops(uv, alpha)
    return outer


def alpha_shape_loops(
    uv: np.ndarray, alpha: float
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """Outer boundary (CCW) plus inner boundaries (holes, CCW) of the alpha shape.

    Holes are regions inside the outer loop that contain no supporting points —
    physical openings such as windows and door cutouts in a scanned wall.
    Returns ``(outer, [hole, ...])``; holes are filtered to loops that lie
    strictly inside the outer boundary.
    """
    if len(uv) < 4:
        return None, []
    try:
        tri = Delaunay(uv)
    except Exception:
        return None, []

    pts = uv
    simplices = tri.simplices
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    # Circumradius R = (|ab| |bc| |ca|) / (4 * area)
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    area2 = np.abs(cross)
    with np.errstate(divide="ignore", invalid="ignore"):
        circum_r = (ab * bc * ca) / (2.0 * area2)
    keep = simplices[(area2 > 1e-14) & (circum_r < alpha)]
    if len(keep) == 0:
        return None, []

    # Boundary edges appear exactly once among kept triangles. Encode each
    # undirected edge as one int64 — np.unique on scalars is many times
    # faster than the row-wise (axis=0) variant on million-edge arrays.
    edges = np.vstack([keep[:, [0, 1]], keep[:, [1, 2]], keep[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    codes = edges_sorted[:, 0].astype(np.int64) * len(pts) + edges_sorted[:, 1]
    _, first_idx, counts = np.unique(
        codes, return_index=True, return_counts=True
    )
    boundary_edges = edges[first_idx[counts == 1]]
    if len(boundary_edges) < 3:
        return None, []

    loops = _assemble_loops(boundary_edges)
    if not loops:
        return None, []
    # Outer boundary = loop with the largest enclosed area; the rest are
    # hole candidates.
    loops.sort(key=lambda lp: -abs(_signed_area(pts[lp])))
    outer = pts[loops[0]]
    if _signed_area(outer) < 0:
        outer = outer[::-1]

    holes: list[np.ndarray] = []
    for lp in loops[1:]:
        hole = pts[lp]
        if len(hole) < 3:
            continue
        rep = hole.mean(axis=0)
        if not point_in_polygon(rep, outer):
            continue
        if _signed_area(hole) < 0:
            hole = hole[::-1]
        holes.append(hole)
    return outer, holes


def point_in_polygon(pt: np.ndarray, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test (boundary counts as outside-ish)."""
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_int:
                inside = not inside
    return inside


def _assemble_loops(edges: np.ndarray) -> list[list[int]]:
    """Assemble undirected edges into closed vertex loops."""
    from collections import defaultdict

    neighbors: dict[int, list[int]] = defaultdict(list)
    for u, v in edges:
        neighbors[int(u)].append(int(v))
        neighbors[int(v)].append(int(u))

    # Each undirected boundary edge belongs to exactly one loop — mark both
    # directions as visited so a loop is traced once, not once per direction
    # (the reversed duplicate would later masquerade as a giant hole).
    visited_edges: set[tuple[int, int]] = set()
    loops: list[list[int]] = []
    for start in list(neighbors):
        for nxt in neighbors[start]:
            if (start, nxt) in visited_edges:
                continue
            loop = [start]
            prev, cur = start, nxt
            visited_edges.add((start, nxt))
            visited_edges.add((nxt, start))
            ok = True
            while cur != start:
                loop.append(cur)
                candidates = [w for w in neighbors[cur] if w != prev]
                # Prefer an unvisited edge; at junctions just take the first.
                step = None
                for w in candidates:
                    if (cur, w) not in visited_edges:
                        step = w
                        break
                if step is None:
                    ok = False
                    break
                visited_edges.add((cur, step))
                visited_edges.add((step, cur))
                prev, cur = cur, step
                if len(loop) > len(edges) + 1:
                    ok = False
                    break
            if ok and len(loop) >= 3:
                loops.append(loop)
    return loops


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def simplify_polygon(poly: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas–Peucker simplification of a closed polygon."""
    if len(poly) <= 4 or tolerance <= 0:
        return poly
    # Split at the two most distant vertices so DP works on open chains.
    d = np.linalg.norm(poly - poly.mean(axis=0), axis=1)
    i0 = int(d.argmax())
    rolled = np.roll(poly, -i0, axis=0)
    d2 = np.linalg.norm(rolled - rolled[0], axis=1)
    i1 = int(d2.argmax())
    chain_a = _dp(rolled[: i1 + 1], tolerance)
    chain_b = _dp(np.vstack([rolled[i1:], rolled[:1]]), tolerance)
    out = np.vstack([chain_a[:-1], chain_b[:-1]])
    return out if len(out) >= 3 else poly


def _dp(chain: np.ndarray, tol: float) -> np.ndarray:
    if len(chain) <= 2:
        return chain
    start, end = chain[0], chain[-1]
    seg = end - start
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-12:
        dists = np.linalg.norm(chain - start, axis=1)
    else:
        d = seg / seg_len
        rel = chain - start
        dists = np.abs(d[0] * rel[:, 1] - d[1] * rel[:, 0])
    idx = int(dists.argmax())
    if dists[idx] <= tol:
        return np.vstack([start, end])
    left = _dp(chain[: idx + 1], tol)
    right = _dp(chain[idx:], tol)
    return np.vstack([left[:-1], right])


def straighten_polygon(
    poly: np.ndarray,
    angle_tol_deg: float = 12.0,
    weak_cluster_ratio: float = 0.08,
) -> np.ndarray:
    """Snap polygon segments to dominant directions and re-intersect them.

    Segment orientations (mod 180°) are clustered; every cluster within
    ``angle_tol_deg`` of a stronger cluster snaps to it, and clusters near 90°
    to the dominant one become exactly perpendicular. Segments belonging to
    weak clusters (total length below ``weak_cluster_ratio`` x the strongest
    cluster) are staircase noise and are dropped entirely — their neighbours
    close the gap when consecutive snapped segment lines are re-intersected
    to produce the straightened polygon.
    """
    n = len(poly)
    if n < 4:
        return poly

    segs = np.roll(poly, -1, axis=0) - poly
    lengths = np.linalg.norm(segs, axis=1)
    valid = lengths > 1e-12
    if valid.sum() < 4:
        return poly
    angles = np.mod(np.arctan2(segs[:, 1], segs[:, 0]), np.pi)

    # Cluster directions (mod pi), weighted by segment length.
    tol = np.deg2rad(angle_tol_deg)
    order = np.argsort(-lengths)
    cluster_dirs: list[float] = []
    cluster_weights: list[float] = []
    seg_cluster = np.full(n, -1, dtype=int)
    for idx in order:
        if not valid[idx]:
            continue
        ang = angles[idx]
        assigned = False
        for ci, cdir in enumerate(cluster_dirs):
            diff = _angdiff_pi(ang, cdir)
            if diff < tol:
                # Weighted circular update (mod pi via doubling).
                w = cluster_weights[ci]
                z = w * np.exp(2j * cdir) + lengths[idx] * np.exp(2j * ang)
                cluster_dirs[ci] = float(np.mod(np.angle(z) / 2.0, np.pi))
                cluster_weights[ci] += lengths[idx]
                seg_cluster[idx] = ci
                assigned = True
                break
        if not assigned:
            seg_cluster[idx] = len(cluster_dirs)
            cluster_dirs.append(float(ang))
            cluster_weights.append(float(lengths[idx]))

    if not cluster_dirs:
        return poly

    # Consolidate cluster directions, strongest first: weaker clusters adopt a
    # stronger cluster's direction when nearly parallel, or become exactly
    # perpendicular to it when nearly orthogonal. Weak clusters (< 50% of the
    # strongest) conform within a doubled tolerance — they are noise-driven
    # and must not found their own direction in the lexicon.
    order2 = np.argsort(-np.asarray(cluster_weights))
    strongest = float(cluster_weights[order2[0]])
    fixed_clusters: list[int] = []
    for g in order2:
        w = cluster_weights[g]
        tol_eff = tol if w >= 0.5 * strongest else 2.0 * tol
        for f in fixed_clusters:
            base = cluster_dirs[f]
            diff = _angdiff_pi(cluster_dirs[g], base)
            if diff < tol_eff:
                cluster_dirs[g] = base
                break
            if abs(diff - np.pi / 2) < tol_eff:
                cluster_dirs[g] = float(np.mod(base + np.pi / 2, np.pi))
                break
        fixed_clusters.append(g)

    # Principal directions = final directions with substantial total support.
    # Every segment snaps to the nearest principal direction (within 2x
    # tolerance); segments too far from every principal direction are
    # staircase noise and are dropped — their neighbours close the gap on
    # re-intersection.
    dir_weight: dict[float, float] = {}
    for ci in range(len(cluster_dirs)):
        dir_weight[cluster_dirs[ci]] = dir_weight.get(cluster_dirs[ci], 0.0) + cluster_weights[ci]
    max_dir_weight = max(dir_weight.values())
    principal_dirs = [
        d for d, w in dir_weight.items() if w >= weak_cluster_ratio * max_dir_weight
    ]
    snap_tol = 2.0 * tol
    seg_list: list[tuple[np.ndarray, np.ndarray]] = []  # (point_on_line, unit_dir)
    for i in range(n):
        if not valid[i] or seg_cluster[i] < 0:
            continue
        diffs = [(_angdiff_pi(angles[i], d), d) for d in principal_dirs]
        best_diff, best_dir = min(diffs)
        if best_diff > snap_tol:
            continue  # staircase noise between dominant directions
        ang = best_dir
        direction = np.array([np.cos(ang), np.sin(ang)])
        midpoint = (poly[i] + poly[(i + 1) % n]) / 2.0
        if seg_list:
            prev_pt, prev_dir = seg_list[-1]
            if abs(prev_dir[0] * direction[1] - prev_dir[1] * direction[0]) < 1e-9:
                # Same direction: merge by weighted midpoint (keeps line stable).
                seg_list[-1] = ((prev_pt + midpoint) / 2.0, prev_dir)
                continue
        seg_list.append((midpoint, direction))

    # Wrap-around merge.
    if len(seg_list) >= 2:
        (p0, d0), (pl, dl) = seg_list[0], seg_list[-1]
        if abs(dl[0] * d0[1] - dl[1] * d0[0]) < 1e-9:
            seg_list[0] = ((p0 + pl) / 2.0, d0)
            seg_list.pop()

    if len(seg_list) < 3:
        return poly

    new_poly = []
    m = len(seg_list)
    for i in range(m):
        p1, d1 = seg_list[i]
        p2, d2 = seg_list[(i + 1) % m]
        pt = _line_intersect_2d(p1, d1, p2, d2)
        if pt is None:
            pt = (p1 + p2) / 2.0
        new_poly.append(pt)
    out = np.array(new_poly)

    # Reject pathological results (extreme area change means the
    # regularization broke the shape — keep the unstraightened polygon).
    a_old, a_new = abs(_signed_area(poly)), abs(_signed_area(out))
    if a_old > 0 and not (0.5 <= a_new / a_old <= 2.0):
        return poly
    if _signed_area(out) < 0:
        out = out[::-1]
    return out


def _angdiff_pi(a: float, b: float) -> float:
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)


def _line_intersect_2d(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray
) -> np.ndarray | None:
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / cross
    return p1 + t * d1


def polygon_area(poly: np.ndarray) -> float:
    return abs(_signed_area(poly))


def remove_collinear(poly: np.ndarray, tolerance: float) -> np.ndarray:
    """Drop vertices lying within ``tolerance`` of the line through their
    neighbours — cleans up intermediate vertices left on straight edges after
    line snapping, so a rectangular face ends up with exactly 4 vertices."""
    changed = True
    while changed and len(poly) > 3:
        changed = False
        n = len(poly)
        for i in range(n):
            a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
            ac = c - a
            length = np.hypot(ac[0], ac[1])
            if length < 1e-12:
                continue
            dist = abs(ac[0] * (b[1] - a[1]) - ac[1] * (b[0] - a[0])) / length
            if dist < tolerance:
                poly = np.delete(poly, i, axis=0)
                changed = True
                break
    return poly


def dedupe_polygon(poly: np.ndarray, min_dist: float) -> np.ndarray:
    """Drop consecutive vertices closer than ``min_dist`` (after snapping)."""
    if len(poly) < 3:
        return poly
    keep = [0]
    for i in range(1, len(poly)):
        if np.linalg.norm(poly[i] - poly[keep[-1]]) >= min_dist:
            keep.append(i)
    # Last vs first
    while len(keep) > 3 and np.linalg.norm(poly[keep[-1]] - poly[keep[0]]) < min_dist:
        keep.pop()
    return poly[keep]
