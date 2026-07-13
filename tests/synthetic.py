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


def make_welded_tank_scan(
    length: float = 0.6,
    width: float = 0.4,
    height: float = 0.3,
    t_bottom: float = 0.008,
    t_wall: float = 0.006,
    density: float = 500000.0,
    noise: float = 0.0001,
    seed: int = 21,
) -> PointCloud:
    """An open-top welded tank: bottom plate + 4 wall plates, both faces of
    every plate sampled (as scanned inside and outside)."""
    rng = np.random.default_rng(seed)
    x, y, z = np.eye(3)
    parts = []
    # Bottom plate: inner face z=0, outer face z=-t_bottom.
    parts.append(sample_rect((0, 0, 0), x, y, length, width, density, noise, rng))
    parts.append(sample_rect((0, 0, -t_bottom), x, y, length, width, density, noise, rng))
    # Walls (plate mid-planes at the tank contour), inner + outer faces.
    walls = [
        ((0, 0, 0), x, length, y),   # y = 0 wall
        ((0, width, 0), x, length, y),  # y = width wall
        ((0, 0, 0), y, width, x),    # x = 0 wall
        ((length, 0, 0), y, width, x),  # x = length wall
    ]
    for origin, u_axis, u_len, n_axis in walls:
        origin = np.asarray(origin, dtype=np.float64)
        sign = 1.0 if np.allclose(origin[:2], 0) or origin @ n_axis == 0 else -1.0
        inner = origin
        outer = origin - sign * t_wall * n_axis
        parts.append(sample_rect(inner, u_axis, z, u_len, height, density, noise, rng))
        parts.append(sample_rect(outer, u_axis, z, u_len, height, density, noise, rng))
    return PointCloud(points=np.vstack(parts), source="synthetic welded tank")


def make_cone_scan(
    apex=(0, 0, 0),
    axis=(0, 0, 1.0),
    half_angle_deg: float = 25.0,
    t_min: float = 0.2,
    t_max: float = 0.8,
    density: float = 30000.0,
    noise: float = 0.0005,
    seed: int = 23,
) -> PointCloud:
    """A truncated cone shell (hopper) between axial distances t_min..t_max."""
    rng = np.random.default_rng(seed)
    apex = np.asarray(apex, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1, 0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    tan_half = np.tan(np.deg2rad(half_angle_deg))

    area = np.pi * (t_min + t_max) * tan_half * (t_max - t_min)
    n = max(2000, int(area * density))
    # Uniform on the cone: t ~ sqrt-distributed for constant surface density.
    t = np.sqrt(rng.uniform(t_min**2, t_max**2, n))
    theta = rng.uniform(0, 2 * np.pi, n)
    r = t * tan_half + rng.normal(0, noise, n)
    pts = (
        apex
        + np.outer(t, axis)
        + np.outer(r * np.cos(theta), u)
        + np.outer(r * np.sin(theta), v)
    )
    return PointCloud(points=pts, source="synthetic cone")


def make_shs_beam_scan(
    size: float = 0.100,
    length: float = 2.0,
    density: float = 40000.0,
    noise: float = 0.001,
    seed: int = 27,
) -> PointCloud:
    """A square hollow section along +X: the four outer faces."""
    rng = np.random.default_rng(seed)
    x, y, z = np.eye(3)
    s = size
    parts = [
        sample_rect((0, -s / 2, 0), x, y, length, s, density, noise, rng),      # bottom
        sample_rect((0, -s / 2, s), x, y, length, s, density, noise, rng),      # top
        sample_rect((0, -s / 2, 0), x, z, length, s, density, noise, rng),      # left
        sample_rect((0, s / 2, 0), x, z, length, s, density, noise, rng),       # right
    ]
    return PointCloud(points=np.vstack(parts), source="synthetic SHS")


def make_l_angle_scan(
    leg: float = 0.080,
    length: float = 2.0,
    density: float = 40000.0,
    noise: float = 0.001,
    seed: int = 29,
) -> PointCloud:
    """An equal angle profile along +X: two perpendicular legs."""
    rng = np.random.default_rng(seed)
    x, y, z = np.eye(3)
    parts = [
        sample_rect((0, 0, 0), x, y, length, leg, density, noise, rng),  # horizontal leg
        sample_rect((0, 0, 0), x, z, length, leg, density, noise, rng),  # vertical leg
    ]
    return PointCloud(points=np.vstack(parts), source="synthetic L angle")


def make_two_storey_scan(
    size=(4.0, 3.0), storey_height: float = 2.6,
    density: float = 700.0, noise: float = 0.004, seed: int = 31,
) -> PointCloud:
    """Two stacked rooms: slabs at 0 / h / 2h plus full-height walls."""
    rng = np.random.default_rng(seed)
    sx, sy = size
    h = storey_height
    x, y, z = np.eye(3)
    parts = []
    for z0 in (0.0, h, 2 * h):
        parts.append(sample_rect((0, 0, z0), x, y, sx, sy, density, noise, rng))
    walls = [
        ((0, 0, 0), x, sx), ((0, sy, 0), x, sx),
        ((0, 0, 0), y, sy), ((sx, 0, 0), y, sy),
    ]
    for origin, u_axis, u_len in walls:
        parts.append(sample_rect(origin, u_axis, z, u_len, 2 * h, density, noise, rng))
    return PointCloud(points=np.vstack(parts), source="synthetic two-storey")


def _bridge_deck(length, width, deck_z, thickness, density, noise, rng):
    x, y = np.eye(3)[0], np.eye(3)[1]
    top = sample_rect((-length / 2, -width / 2, deck_z), x, y, length, width, density, noise, rng)
    bottom = sample_rect(
        (-length / 2, -width / 2, deck_z - thickness), x, y, length, width, density, noise, rng
    )
    return [top, bottom]


def make_beam_bridge_scan(
    length: float = 30.0,
    width: float = 6.0,
    deck_z: float = 8.0,
    density: float = 300.0,
    noise: float = 0.01,
    seed: int = 41,
) -> PointCloud:
    """Two-pier beam bridge: deck slab, 2 round piers, 2 abutment walls."""
    rng = np.random.default_rng(seed)
    parts = _bridge_deck(length, width, deck_z, 0.5, density, noise, rng)
    for station in (-length / 6, length / 6):
        parts.append(
            sample_cylinder(
                (station, 0, (deck_z - 0.5) / 2), (0, 0, 1), 0.6, deck_z - 0.5,
                density * 4, noise, rng,
            )
        )
    y, z = np.eye(3)[1], np.eye(3)[2]
    for station in (-length / 2, length / 2):
        parts.append(
            sample_rect((station, -width / 2, 0), y, z, width, deck_z - 0.5, density * 2, noise, rng)
        )
    return PointCloud(points=np.vstack(parts), source="synthetic beam bridge")


def make_arch_bridge_scan(
    length: float = 24.0,
    width: float = 5.0,
    deck_z: float = 8.0,
    radius: float = 12.0,
    density: float = 300.0,
    noise: float = 0.01,
    seed: int = 43,
) -> PointCloud:
    """Arch bridge: deck slab + barrel arch (partial cylinder, axis across)."""
    rng = np.random.default_rng(seed)
    parts = _bridge_deck(length, width, deck_z, 0.5, density, noise, rng)
    # Arch: cylinder axis along y, center below the deck, crown near deck.
    center = np.array([0.0, 0.0, deck_z - 1.0 - radius])
    n = int(2 * radius * 1.6 * width * density)
    theta = rng.uniform(np.pi / 2 - 0.9, np.pi / 2 + 0.9, n)  # around the crown
    yy = rng.uniform(-width / 2, width / 2, n)
    r = radius + rng.normal(0, noise, n)
    arch = np.column_stack(
        [r * np.cos(theta), yy, center[2] + r * np.sin(theta)]
    )
    parts.append(arch)
    return PointCloud(points=np.vstack(parts), source="synthetic arch bridge")


def make_cable_stayed_bridge_scan(
    length: float = 40.0,
    width: float = 7.0,
    deck_z: float = 10.0,
    pylon_height: float = 18.0,
    density: float = 250.0,
    noise: float = 0.01,
    seed: int = 47,
) -> PointCloud:
    """Cable-stayed bridge: deck, central pylon, 8 diagonal stay cables."""
    rng = np.random.default_rng(seed)
    parts = _bridge_deck(length, width, deck_z, 0.5, density, noise, rng)
    # Pylon above (and a pier below) the deck at midspan.
    parts.append(
        sample_cylinder(
            (0, 0, deck_z + pylon_height / 2), (0, 0, 1), 0.7, pylon_height,
            density * 6, noise, rng,
        )
    )
    parts.append(
        sample_cylinder(
            (0, 0, (deck_z - 0.5) / 2), (0, 0, 1), 0.7, deck_z - 0.5,
            density * 4, noise, rng,
        )
    )
    top = np.array([0.0, 0.0, deck_z + pylon_height * 0.95])
    for station in (-16.0, -12.0, -8.0, -4.5, 4.5, 8.0, 12.0, 16.0):
        anchor = np.array([station, 0.0, deck_z + 0.1])
        axis = anchor - top
        cable_len = np.linalg.norm(axis)
        mid = (anchor + top) / 2.0
        parts.append(
            sample_cylinder(mid, axis / cable_len, 0.05, cable_len * 0.9,
                            density * 3, noise * 0.5, rng)
        )
    return PointCloud(points=np.vstack(parts), source="synthetic cable-stayed bridge")


def make_bent_plate_scan(
    base_size=(0.3, 0.2),
    flange_height: float = 0.15,
    second_flange: bool = False,
    thickness: float = 0.003,
    density: float = 800000.0,
    noise: float = 0.0001,
    seed: int = 51,
) -> PointCloud:
    """A bent sheet metal part (Kantteil): base plate + flange(s) at 90°.

    Base mid-plane at z=0 (faces z=±t/2); flange bent up at y=b (mid-plane
    y=b, faces y=b±t/2). ``second_flange`` adds one at y=0 (U-channel).
    """
    rng = np.random.default_rng(seed)
    a, b = base_size
    t = thickness
    x, y, z = np.eye(3)
    parts = [
        sample_rect((0, 0, -t / 2), x, y, a, b, density, noise, rng),
        sample_rect((0, 0, t / 2), x, y, a, b, density, noise, rng),
        sample_rect((0, b - t / 2, 0), x, z, a, flange_height, density, noise, rng),
        sample_rect((0, b + t / 2, 0), x, z, a, flange_height, density, noise, rng),
    ]
    if second_flange:
        parts.append(sample_rect((0, -t / 2, 0), x, z, a, flange_height, density, noise, rng))
        parts.append(sample_rect((0, t / 2, 0), x, z, a, flange_height, density, noise, rng))
    return PointCloud(points=np.vstack(parts), source="synthetic bent part")


def add_registration_ghost(
    cloud: PointCloud, offset=(0.008, 0.0, 0.0), seed: int = 53
) -> PointCloud:
    """Simulate a poorly registered second station: the full cloud again,
    shifted by ``offset`` (planes ⊥ offset become double walls)."""
    shifted = cloud.points + np.asarray(offset, dtype=np.float64)
    return PointCloud(points=np.vstack([cloud.points, shifted]), source=cloud.source)


def add_mixed_pixel_strings(
    cloud: PointCloud, n_strings: int = 6, points_per_string: int 	= 40, seed: int = 57
) -> PointCloud:
    """Simulate mixed-pixel edge artifacts: sparse strings of points hanging
    off silhouette edges into space."""
    rng = np.random.default_rng(seed)
    lo, hi = cloud.aabb
    span = float(np.linalg.norm(hi - lo))
    strings = []
    for _ in range(n_strings):
        anchor = cloud.points[rng.integers(len(cloud.points))]
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        length = rng.uniform(0.1, 0.25) * span
        ts = np.sort(rng.uniform(0, length, points_per_string))
        pts = anchor + np.outer(ts, direction)
        pts += rng.normal(0, 0.001 * span, pts.shape)
        strings.append(pts)
    return PointCloud(
        points=np.vstack([cloud.points, *strings]), source=cloud.source
    )


def add_outliers(cloud: PointCloud, fraction: float = 0.01, seed: int = 9) -> PointCloud:
    """Scatter uniform noise points around the cloud's bounding box."""
    rng = np.random.default_rng(seed)
    lo, hi = cloud.aabb
    span = hi - lo
    n = int(len(cloud) * fraction)
    junk = rng.uniform(lo - 0.2 * span, hi + 0.2 * span, size=(n, 3))
    return PointCloud(points=np.vstack([cloud.points, junk]), source=cloud.source)


def make_hall_with_column_scan(
    size=(6.0, 5.0, 3.0),
    column_radius: float = 0.25,
    column_center=(2.0, 2.0),
    density: float = 900.0,
    noise: float = 0.004,
    seed: int = 33,
) -> PointCloud:
    """A box hall with one round column from floor to ceiling."""
    rng = np.random.default_rng(seed)
    box = make_box_scan(size=size, density=density, noise=noise, seed=seed)
    # Column lateral surface.
    n = int(density * 2 * np.pi * column_radius * size[2])
    theta = rng.uniform(0, 2 * np.pi, n)
    zc = rng.uniform(0, size[2], n)
    r = column_radius + rng.normal(0, noise, n)
    pts = np.column_stack(
        [
            column_center[0] + r * np.cos(theta),
            column_center[1] + r * np.sin(theta),
            zc,
        ]
    )
    return PointCloud(
        points=np.vstack([box.points, pts]), source="synthetic hall with column"
    )
