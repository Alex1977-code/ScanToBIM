"""Exact edge geometry from plane intersections.

Instead of tracing edges through noisy points, edges are *derived*: two
adjacent regularized planes intersect in a mathematically exact line, three
mutually adjacent planes meet in an exact corner point. Boundary polygons are
later snapped onto these lines/corners, which is what produces razor-sharp,
straight creases in the final mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scantobim.core.planes import Plane


@dataclass
class IntersectionLine:
    """Line of intersection between planes ``i`` and ``j``: ``p = point + t * direction``."""

    plane_i: int
    plane_j: int
    point: np.ndarray  # (3,)
    direction: np.ndarray  # unit (3,)


@dataclass
class Corner:
    """Exact corner where three planes meet."""

    planes: tuple[int, int, int]
    position: np.ndarray  # (3,)


def intersect_planes(a: Plane, b: Plane) -> tuple[np.ndarray, np.ndarray] | None:
    """Intersection line of two planes, or ``None`` if (near-)parallel.

    Returns ``(point, direction)`` where ``point`` is the point on the line
    closest to the origin.
    """
    direction = np.cross(a.normal, b.normal)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return None
    direction = direction / norm
    # Solve for the line point closest to origin:
    #   [na; nb; dir] p = [-da; -db; 0]
    m = np.vstack([a.normal, b.normal, direction])
    rhs = np.array([-a.d, -b.d, 0.0])
    point = np.linalg.solve(m, rhs)
    return point, direction


def intersect_three_planes(a: Plane, b: Plane, c: Plane) -> np.ndarray | None:
    """Exact intersection point of three planes, or ``None`` if degenerate."""
    m = np.vstack([a.normal, b.normal, c.normal])
    if abs(np.linalg.det(m)) < 1e-6:
        return None
    return np.linalg.solve(m, -np.array([a.d, b.d, c.d]))


def build_intersection_lines(
    planes: list[Plane], adjacency: set[tuple[int, int]]
) -> list[IntersectionLine]:
    lines: list[IntersectionLine] = []
    for i, j in sorted(adjacency):
        res = intersect_planes(planes[i], planes[j])
        if res is None:
            continue
        point, direction = res
        lines.append(IntersectionLine(plane_i=i, plane_j=j, point=point, direction=direction))
    return lines


def build_corners(
    planes: list[Plane],
    adjacency: set[tuple[int, int]],
    max_distance_from_support: float,
    points: np.ndarray,
) -> list[Corner]:
    """Corners from all mutually adjacent plane triples.

    A triple only yields a corner if the intersection point lies close to at
    least one inlier of each participating plane — this rejects phantom
    corners produced by planes whose infinite extensions meet far away from
    the actual scanned geometry.
    """
    from scipy.spatial import cKDTree

    n = len(planes)
    adj = {tuple(sorted(p)) for p in adjacency}
    trees = [cKDTree(points[p.inliers]) for p in planes]
    corners: list[Corner] = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in adj:
                continue
            for k in range(j + 1, n):
                if (i, k) not in adj or (j, k) not in adj:
                    continue
                pos = intersect_three_planes(planes[i], planes[j], planes[k])
                if pos is None:
                    continue
                if all(
                    trees[t].query(pos)[0] <= max_distance_from_support
                    for t in (i, j, k)
                ):
                    corners.append(Corner(planes=(i, j, k), position=pos))
    return corners


def snap_points_to_line(
    points_3d: np.ndarray,
    line_point: np.ndarray,
    line_dir: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project every point within ``max_distance`` of the line onto the line.

    Returns ``(snapped_points, snapped_mask)``.
    """
    rel = points_3d - line_point
    t = rel @ line_dir
    foot = line_point + np.outer(t, line_dir)
    dist = np.linalg.norm(points_3d - foot, axis=1)
    mask = dist <= max_distance
    out = points_3d.copy()
    out[mask] = foot[mask]
    return out, mask


def snap_points_to_corners(
    points_3d: np.ndarray,
    corners: list[Corner],
    max_distance: float,
) -> np.ndarray:
    """Snap points to the nearest exact corner within ``max_distance``."""
    if not corners:
        return points_3d
    corner_pos = np.array([c.position for c in corners])
    out = points_3d.copy()
    for idx in range(len(points_3d)):
        d = np.linalg.norm(corner_pos - points_3d[idx], axis=1)
        best = int(d.argmin())
        if d[best] <= max_distance:
            out[idx] = corner_pos[best]
    return out
