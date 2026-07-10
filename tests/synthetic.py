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
    density: float = 900.0, noise: float = 0.004, seed: int = 3, window: bool = False
) -> PointCloud:
    """An L-shaped room footprint extruded to 2.5 m — six walls + floor + ceiling.

    With ``window=True`` the long south wall gets a 1.4 x 1.0 m window opening.
    """
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
        wall = sample_rect(a, u, z, length, h, density, noise, rng)
        if window and i == 0:  # south wall y=0: window x in [1.5, 2.9], z in [0.9, 1.9]
            inside = (
                (wall[:, 0] > 1.5) & (wall[:, 0] < 2.9)
                & (wall[:, 2] > 0.9) & (wall[:, 2] < 1.9)
            )
            wall = wall[~inside]
        parts.append(wall)
    # floor + ceiling: rejection-sample the L polygon
    for z0 in (0.0, h):
        n = int(5 * 4.5 * density)
        xy = rng.uniform([0, 0], [5, 4.5], size=(n, 2))
        inside = ~((xy[:, 0] > 2.5) & (xy[:, 1] > 2.5))
        xy = xy[inside]
        zs = z0 + rng.normal(0, noise, len(xy))
        parts.append(np.column_stack([xy, zs]))
    return PointCloud(points=np.vstack(parts), source="synthetic L-room")


def sample_cylinder(
    center, axis, radius, length, density, noise, rng, caps: bool = False
) -> np.ndarray:
    """Sample the lateral surface of a cylinder (optionally with end caps)."""
    center = np.asarray(center, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1, 0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    n = max(64, int(2 * np.pi * radius * length * density))
    theta = rng.uniform(0, 2 * np.pi, n)
    t = rng.uniform(-length / 2, length / 2, n)
    r = radius + rng.normal(0, noise, n)
    pts = (
        center
        + np.outer(t, axis)
        + np.outer(r * np.cos(theta), u)
        + np.outer(r * np.sin(theta), v)
    )
    if caps:
        for sign in (-1.0, 1.0):
            m = max(32, int(np.pi * radius**2 * density))
            rr = radius * np.sqrt(rng.uniform(0, 1, m))
            th = rng.uniform(0, 2 * np.pi, m)
            cap = (
                center
                + (sign * length / 2 + rng.normal(0, noise, m))[:, None] * axis
                + np.outer(rr * np.cos(th), u)
                + np.outer(rr * np.sin(th), v)
            )
            pts = np.vstack([pts, cap])
    return pts


def make_stepped_shaft_scan(
    steps=((0.030, 0.08), (0.050, 0.12), (0.040, 0.10)),
    axis=(0, 0, 1.0),
    origin=(0, 0, 0),
    density: float = 400000.0,
    noise: float = 0.0003,
    seed: int = 11,
) -> PointCloud:
    """A stepped shaft: coaxial cylinder segments ``(radius, length)`` in a row."""
    rng = np.random.default_rng(seed)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    origin = np.asarray(origin, dtype=np.float64)
    parts = []
    pos = 0.0
    for radius, length in steps:
        center = origin + (pos + length / 2) * axis
        parts.append(sample_cylinder(center, axis, radius, length, density, noise, rng))
        pos += length
    return PointCloud(points=np.vstack(parts), source="synthetic shaft")


def make_gear_scan(
    teeth: int = 24,
    module: float = 0.004,
    width: float = 0.030,
    center=(0, 0, 0),
    axis=(0, 0, 1.0),
    density: float = 400000.0,
    noise: float = 0.0002,
    seed: int = 13,
) -> PointCloud:
    """A spur gear approximated with trapezoidal teeth (DIN-style dimensions).

    Tip diameter da = m*(z+2), root df = m*(z-2.5); the tooth flanks are
    linear. Samples the toothed lateral surface plus both annular end faces.
    """
    rng = np.random.default_rng(seed)
    center = np.asarray(center, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1, 0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    ra = module * (teeth + 2) / 2.0
    rf = module * (teeth - 2.5) / 2.0

    def radius_at(theta):
        # Periodic trapezoid: tip land 40%, root land 30%, flanks 15% each.
        phase = np.mod(theta * teeth / (2 * np.pi), 1.0)
        r = np.empty_like(phase)
        tip = phase < 0.40
        flank1 = (phase >= 0.40) & (phase < 0.55)
        root = (phase >= 0.55) & (phase < 0.85)
        flank2 = phase >= 0.85
        r[tip] = ra
        r[flank1] = ra + (rf - ra) * (phase[flank1] - 0.40) / 0.15
        r[root] = rf
        r[flank2] = rf + (ra - rf) * (phase[flank2] - 0.85) / 0.15
        return r

    n = max(2000, int(2 * np.pi * ra * width * density))
    theta = rng.uniform(0, 2 * np.pi, n)
    t = rng.uniform(-width / 2, width / 2, n)
    r = radius_at(theta) + rng.normal(0, noise, n)
    pts = (
        center + np.outer(t, axis)
        + np.outer(r * np.cos(theta), u) + np.outer(r * np.sin(theta), v)
    )
    # End faces (annulus from 0.5*rf to tooth surface).
    for sign in (-1.0, 1.0):
        m = n // 3
        th = rng.uniform(0, 2 * np.pi, m)
        rr = np.sqrt(rng.uniform((0.5 * rf) ** 2, radius_at(th) ** 2))
        face = (
            center + (sign * width / 2 + rng.normal(0, noise, m))[:, None] * axis
            + np.outer(rr * np.cos(th), u) + np.outer(rr * np.sin(th), v)
        )
        pts = np.vstack([pts, face])
    return PointCloud(points=pts, source=f"synthetic gear z={teeth}")


def make_ipe_beam_scan(
    h: float = 0.200,
    b: float = 0.100,
    tf: float = 0.0085,
    tw: float = 0.0056,
    length: float = 3.0,
    density: float = 40000.0,
    noise: float = 0.001,
    seed: int = 17,
) -> PointCloud:
    """An I-profile steel beam along +X: two flanges + web, outer surfaces."""
    rng = np.random.default_rng(seed)
    x = np.array([1.0, 0, 0])
    y = np.array([0.0, 1, 0])
    z = np.array([0.0, 0, 1])
    parts = [
        # Bottom flange: top + bottom surface
        sample_rect((0, -b / 2, 0), x, y, length, b, density, noise, rng),
        sample_rect((0, -b / 2, tf), x, y, length, b, density, noise, rng),
        # Top flange
        sample_rect((0, -b / 2, h - tf), x, y, length, b, density, noise, rng),
        sample_rect((0, -b / 2, h), x, y, length, b, density, noise, rng),
        # Web: both sides
        sample_rect((0, -tw / 2, tf), x, z, length, h - 2 * tf, density, noise, rng),
        sample_rect((0, tw / 2, tf), x, z, length, h - 2 * tf, density, noise, rng),
    ]
    return PointCloud(points=np.vstack(parts), source="synthetic IPE beam")


def add_outliers(cloud: PointCloud, fraction: float = 0.01, seed: int = 9) -> PointCloud:
    """Scatter uniform noise points around the cloud's bounding box."""
    rng = np.random.default_rng(seed)
    lo, hi = cloud.aabb
    span = hi - lo
    n = int(len(cloud) * fraction)
    junk = rng.uniform(lo - 0.2 * span, hi + 0.2 * span, size=(n, 3))
    return PointCloud(points=np.vstack([cloud.points, junk]), source=cloud.source)
