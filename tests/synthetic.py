"""Synthetic scan generators shared by the tests and examples."""

from __future__ import annotations

import numpy as np

from scantobim.core.cloud import PointCloud


def sample_rect(
    origin, u_axis, v_axis, u_len, v_len, density, noise, rng
) -> np.ndarray:
    """Sample a noisy rectangle patch: origin + s*u + t*v, s∈[0,u_len], t∈[0,v_len]."""
    origin = np.asarray(origin, dtype=np.float64)
    u_axis = np.asarray(u_axis, dtype=np.float64)
    v_axis = np.asarray(v_axis, dtype=np.float64)
    normal = np.cross(u_axis, v_axis)
    normal = normal / np.linalg.norm(normal)
    n = max(16, int(u_len * v_len * density))
    s = rng.uniform(0, u_len, n)
    t = rng.uniform(0, v_len, n)
    pts = origin + np.outer(s, u_axis) + np.outer(t, v_axis)
    pts += np.outer(rng.normal(0, noise, n), normal)
    return pts


def make_box_scan(
    size=(4.0, 3.0, 2.5),
    density: float = 900.0,
    noise: float = 0.004,
    seed: int = 42,
    open_top: bool = False,
) -> PointCloud:
    """A box scanned from outside (walls + floor + ceiling), like a small building.

    ``density`` is points per square meter, ``noise`` the Gaussian sigma along
    the surface normal (4 mm ≈ terrestrial scanner at mid range).
    """
    rng = np.random.default_rng(seed)
    sx, sy, sz = size
    x, y, z = np.eye(3)
    faces = [
        ((0, 0, 0), x, y, sx, sy),  # floor
        ((0, 0, 0), x, z, sx, sz),  # wall y=0
        ((0, sy, 0), x, z, sx, sz),  # wall y=sy
        ((0, 0, 0), y, z, sy, sz),  # wall x=0
        ((sx, 0, 0), y, z, sy, sz),  # wall x=sx
    ]
    if not open_top:
        faces.append(((0, 0, sz), x, y, sx, sy))  # ceiling
    parts = [
        sample_rect(o, u, v, ul, vl, density, noise, rng) for o, u, v, ul, vl in faces
    ]
    pts = np.vstack(parts)
    return PointCloud(points=pts, source="synthetic box")


def make_l_room_scan(
    density: float = 900.0, noise: float = 0.004, seed: int = 3
) -> PointCloud:
    """An L-shaped room footprint extruded to 2.5 m — six walls + floor + ceiling."""
    rng = np.random.default_rng(seed)
    h = 2.5
    # L footprint: (0,0) (5,0) (5,2.5) (2.5,2.5) (2.5,4.5) (0,4.5)
    corners = [(0, 0), (5, 0), (5, 2.5), (2.5, 2.5), (2.5, 4.5), (0, 4.5)]
    parts = []
    z = np.array([0.0, 0.0, 1.0])
    for i in range(len(corners)):
        a = np.array([*corners[i], 0.0])
        b = np.array([*corners[(i + 1) % len(corners)], 0.0])
        u = b - a
        length = np.linalg.norm(u)
        u = u / length
        parts.append(sample_rect(a, u, z, length, h, density, noise, rng))
    # floor + ceiling: rejection-sample the L polygon
    for z0 in (0.0, h):
        n = int(5 * 4.5 * density)
        xy = rng.uniform([0, 0], [5, 4.5], size=(n, 2))
        inside = ~((xy[:, 0] > 2.5) & (xy[:, 1] > 2.5))
        xy = xy[inside]
        zs = z0 + rng.normal(0, noise, len(xy))
        parts.append(np.column_stack([xy, zs]))
    return PointCloud(points=np.vstack(parts), source="synthetic L-room")


def add_outliers(cloud: PointCloud, fraction: float = 0.01, seed: int = 9) -> PointCloud:
    """Scatter uniform noise points around the cloud's bounding box."""
    rng = np.random.default_rng(seed)
    lo, hi = cloud.aabb
    span = hi - lo
    n = int(len(cloud) * fraction)
    junk = rng.uniform(lo - 0.2 * span, hi + 0.2 * span, size=(n, 3))
    return PointCloud(points=np.vstack([cloud.points, junk]), source=cloud.source)
