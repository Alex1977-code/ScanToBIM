"""Parametric solid geometry for detected objects.

Turns measured parameters back into true-to-shape 3D solids:

* **Extrusion** of arbitrary 2D cross sections (with holes) along any axis —
  the workhorse for steel profiles, gear bodies and bridge members.
* **Involute spur gears** (DIN 867): tooth flanks as true involutes of the
  base circle, addendum ``m``, dedendum ``1.25 m``, pressure angle 20° —
  so a measured gear (z, module, width) becomes a gear with correct tooth
  geometry instead of a plain cylinder.
* **Steel profile cross sections**: I/H sections with web and flange
  thickness from the DIN 1025 catalog, U channels, equal angles and hollow
  sections — extruded along the detected member axis in the detected
  orientation.

All dimensions in the unit of the point cloud (meters recommended).
"""

from __future__ import annotations

import numpy as np

from scantobim.core.mesh import Mesh, triangulate_with_holes, weld_vertices


def frame_from_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right-handed orthonormal basis ``(u, v)`` perpendicular to ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(axis[0]) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def _polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly if _polygon_area(poly) >= 0 else poly[::-1]


def extrude_polygon(
    outline: np.ndarray,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    length: float,
    holes: list[np.ndarray] | None = None,
    color: tuple[int, int, int] = (170, 190, 220),
    group: int = 0,
) -> Mesh:
    """Extrude a 2D cross section into a prismatic solid.

    ``outline``/``holes`` are 2D loops in the (u, v) plane (any winding —
    normalized internally); the solid spans ``origin ± w·length/2``. Cap
    normals face ±w, wall normals face outward.
    """
    outline = _ensure_ccw(np.asarray(outline, dtype=np.float64))
    holes = [_ensure_ccw(np.asarray(h, dtype=np.float64)) for h in (holes or [])]
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    w = w / np.linalg.norm(w)
    origin = np.asarray(origin, dtype=np.float64)

    def lift(pts2d: np.ndarray, s: float) -> np.ndarray:
        return origin + np.outer(pts2d[:, 0], u) + np.outer(pts2d[:, 1], v) + s * w

    cap2d, cap_tris = triangulate_with_holes(outline, holes)
    half = length / 2.0
    verts: list[np.ndarray] = [lift(cap2d, half), lift(cap2d, -half)]
    n_cap = len(cap2d)
    faces: list[np.ndarray] = [cap_tris, cap_tris[:, ::-1] + n_cap]
    offset = 2 * n_cap

    # Walls per loop: outer CCW → outward normals; holes reversed → normals
    # face the cavity.
    for loop, reverse in [(outline, False)] + [(h, True) for h in holes]:
        ring = loop[::-1] if reverse else loop
        top = lift(ring, half)
        bot = lift(ring, -half)
        n = len(ring)
        verts.extend([bot, top])
        i = np.arange(n)
        j = (i + 1) % n
        b_i, b_j = offset + i, offset + j
        t_i, t_j = offset + n + i, offset + n + j
        faces.append(np.column_stack([b_i, b_j, t_j]))
        faces.append(np.column_stack([b_i, t_j, t_i]))
        offset += 2 * n

    all_verts = np.vstack(verts)
    all_faces = np.vstack(faces)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (len(all_verts), 1))
    mesh = Mesh(
        vertices=all_verts,
        faces=all_faces,
        vertex_colors=colors,
        face_groups=np.full(len(all_faces), group, dtype=np.int64),
    )
    # Caps and walls share loop coordinates — weld them into a closed solid.
    scale = float(np.abs(outline).max()) + abs(length)
    return weld_vertices(mesh, 1e-7 * scale)


def box_mesh(
    center: np.ndarray,
    x_axis: np.ndarray,
    size: tuple[float, float, float],
    color: tuple[int, int, int] = (180, 180, 180),
    group: int = 0,
) -> Mesh:
    """Axis-oriented box: ``x_axis`` horizontal length direction, z stays up."""
    x = np.asarray(x_axis, dtype=np.float64)
    x = x - x[2] * np.array([0.0, 0.0, 1.0])  # project to horizontal
    n = np.linalg.norm(x)
    x = x / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    y = np.cross([0.0, 0.0, 1.0], x)
    rect = np.array(
        [
            [-size[0] / 2, -size[1] / 2],
            [size[0] / 2, -size[1] / 2],
            [size[0] / 2, size[1] / 2],
            [-size[0] / 2, size[1] / 2],
        ]
    )
    return extrude_polygon(
        rect, center, x, y, np.array([0.0, 0.0, 1.0]), size[2],
        color=color, group=group,
    )


# ------------------------------------------------------------------ gears

def involute_gear_outline(
    teeth: int,
    module: float,
    pressure_angle_deg: float = 20.0,
    flank_points: int = 8,
    tip_points: int = 4,
    root_points: int = 4,
) -> np.ndarray:
    """CCW outline of a DIN 867 spur gear in the plane of rotation.

    Standard proportions: addendum ``m``, dedendum ``1.25 m``; flanks are
    exact involutes of the base circle, the root is joined with a radial
    segment below the base circle (common approximation of the trochoid).
    """
    z = int(teeth)
    m = float(module)
    alpha = np.deg2rad(pressure_angle_deg)
    rp = m * z / 2.0  # pitch
    rb = rp * np.cos(alpha)  # base
    ra = rp + m  # tip
    rf = max(rp - 1.25 * m, 0.15 * rp)  # root

    def involute_angle(r: float) -> float:
        """Polar angle swept by the involute from the base circle to r."""
        t = np.sqrt(max((r / rb) ** 2 - 1.0, 0.0))
        return t - np.arctan(t)

    half_pitch = np.pi / (2.0 * z)  # angular half tooth thickness at rp
    inv_alpha = np.tan(alpha) - alpha

    def flank_offset(r: float) -> float:
        """Angle from tooth center line to the flank at radius r."""
        return half_pitch + inv_alpha - involute_angle(r)

    r_start = max(rf, rb)
    radii = np.linspace(r_start, ra, flank_points)
    tooth: list[tuple[float, float]] = []  # (angle, radius) right→tip→left

    if rf < rb:  # radial segment root → base circle
        tooth.append((-flank_offset(rb), rf))
    for r in radii:  # right flank rising to the tip
        tooth.append((-flank_offset(r), r))
    beta_tip = flank_offset(ra)
    for a in np.linspace(-beta_tip, beta_tip, tip_points + 2)[1:-1]:  # tip arc
        tooth.append((a, ra))
    for r in radii[::-1]:  # left flank back down
        tooth.append((flank_offset(r), r))
    if rf < rb:
        tooth.append((flank_offset(rb), rf))
    # Root arc to the start of the next tooth.
    beta_root = flank_offset(rb) if rf < rb else flank_offset(r_start)
    pitch = 2.0 * np.pi / z
    for a in np.linspace(beta_root, pitch - beta_root, root_points + 2)[1:-1]:
        tooth.append((a, rf))

    angles, rads = np.array(tooth).T
    pts = []
    for k in range(z):
        rot = angles + k * pitch
        pts.append(np.column_stack([rads * np.cos(rot), rads * np.sin(rot)]))
    return np.vstack(pts)


def involute_gear_mesh(
    center: np.ndarray,
    axis: np.ndarray,
    teeth: int,
    module: float,
    width: float,
    bore_radius: float = 0.0,
    color: tuple[int, int, int] = (214, 170, 110),
    group: int = 0,
) -> Mesh:
    """Solid spur gear with true involute tooth geometry."""
    outline = involute_gear_outline(teeth, module)
    holes = []
    if bore_radius > 0:
        theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        holes.append(
            np.column_stack([bore_radius * np.cos(theta), bore_radius * np.sin(theta)])
        )
    u, v = frame_from_axis(axis)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return extrude_polygon(
        outline, np.asarray(center, dtype=np.float64), u, v, axis, width,
        holes=holes, color=color, group=group,
    )


def cone_mesh(
    apex: np.ndarray,
    axis: np.ndarray,
    r_min: float,
    r_max: float,
    height: float,
    offset: float = 0.0,
    color: tuple[int, int, int] = (170, 190, 220),
    segments: int = 48,
    group: int = 0,
) -> Mesh:
    """Frustum between two circular sections along ``axis`` (from the apex)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    apex = np.asarray(apex, dtype=np.float64)
    u, v = frame_from_axis(axis)
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v)
    near = apex + axis * offset + r_min * ring
    far = apex + axis * (offset + height) + r_max * ring
    verts = np.vstack([near, far, [apex + axis * offset], [apex + axis * (offset + height)]])
    faces = []
    cn, cf = 2 * segments, 2 * segments + 1
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, segments + i])
        faces.append([j, segments + j, segments + i])
        faces.append([cn, j, i])
        faces.append([cf, segments + i, segments + j])
    colors = np.tile(np.asarray(color, dtype=np.uint8), (len(verts), 1))
    return Mesh(
        vertices=verts,
        faces=np.array(faces),
        vertex_colors=colors,
        face_groups=np.full(len(faces), group, dtype=np.int64),
    )


# ------------------------------------------------------------ steel profiles

# (web thickness tw, flange thickness tf) per designation, meters —
# DIN 1025-2/3/5 and DIN 1026-1.
PROFILE_THICKNESS: dict[str, tuple[float, float]] = {
    "IPE 80": (0.0038, 0.0052), "IPE 100": (0.0041, 0.0057),
    "IPE 120": (0.0044, 0.0063), "IPE 140": (0.0047, 0.0069),
    "IPE 160": (0.0050, 0.0074), "IPE 180": (0.0053, 0.0080),
    "IPE 200": (0.0056, 0.0085), "IPE 220": (0.0059, 0.0092),
    "IPE 240": (0.0062, 0.0098), "IPE 270": (0.0066, 0.0102),
    "IPE 300": (0.0071, 0.0107), "IPE 330": (0.0075, 0.0115),
    "IPE 360": (0.0080, 0.0127), "IPE 400": (0.0086, 0.0135),
    "IPE 450": (0.0094, 0.0146), "IPE 500": (0.0102, 0.0160),
    "IPE 550": (0.0111, 0.0172), "IPE 600": (0.0120, 0.0190),
    "HEA 100": (0.0050, 0.0080), "HEA 120": (0.0050, 0.0080),
    "HEA 140": (0.0055, 0.0085), "HEA 160": (0.0060, 0.0090),
    "HEA 180": (0.0060, 0.0095), "HEA 200": (0.0065, 0.0100),
    "HEA 220": (0.0070, 0.0110), "HEA 240": (0.0075, 0.0120),
    "HEA 260": (0.0075, 0.0125), "HEA 280": (0.0080, 0.0130),
    "HEA 300": (0.0085, 0.0140), "HEA 320": (0.0090, 0.0155),
    "HEA 340": (0.0095, 0.0165), "HEA 360": (0.0100, 0.0175),
    "HEA 400": (0.0110, 0.0190), "HEA 450": (0.0115, 0.0210),
    "HEA 500": (0.0120, 0.0230), "HEA 600": (0.0130, 0.0250),
    "HEB 100": (0.0060, 0.0100), "HEB 120": (0.0065, 0.0110),
    "HEB 140": (0.0070, 0.0120), "HEB 160": (0.0080, 0.0130),
    "HEB 180": (0.0085, 0.0140), "HEB 200": (0.0090, 0.0150),
    "HEB 220": (0.0095, 0.0160), "HEB 240": (0.0100, 0.0170),
    "HEB 260": (0.0100, 0.0175), "HEB 280": (0.0105, 0.0180),
    "HEB 300": (0.0110, 0.0190), "HEB 320": (0.0115, 0.0205),
    "HEB 340": (0.0120, 0.0215), "HEB 360": (0.0125, 0.0225),
    "HEB 400": (0.0135, 0.0240), "HEB 450": (0.0140, 0.0260),
    "HEB 500": (0.0145, 0.0280), "HEB 600": (0.0155, 0.0300),
    "UPN 80": (0.0060, 0.0080), "UPN 100": (0.0060, 0.0085),
    "UPN 120": (0.0070, 0.0090), "UPN 140": (0.0070, 0.0100),
    "UPN 160": (0.0075, 0.0105), "UPN 180": (0.0080, 0.0110),
    "UPN 200": (0.0085, 0.0115), "UPN 220": (0.0090, 0.0125),
    "UPN 240": (0.0095, 0.0130), "UPN 260": (0.0100, 0.0140),
    "UPN 280": (0.0100, 0.0150), "UPN 300": (0.0100, 0.0160),
}


def _section_thickness(profile: str, family: str, height: float) -> tuple[float, float]:
    """(tw, tf); catalog values where known, EN-typical heuristics otherwise."""
    if profile in PROFILE_THICKNESS:
        return PROFILE_THICKNESS[profile]
    if family == "hollow":  # EN 10219 typical wall
        t = float(np.clip(round(height / 20.0 / 0.0005) * 0.0005, 0.003, 0.0125))
        return t, t
    if family == "L":  # EN 10056: t ≈ h/10
        t = float(np.clip(height / 10.0, 0.004, 0.016))
        return t, t
    t = max(0.005, 0.05 * height)
    return t, 1.6 * t


def steel_profile_outline(
    profile: str, family: str, height: float, width: float
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Canonical cross section ``(outer, holes)`` centered on the bbox center.

    u = width axis, v = height axis. Canonical orientation: U web on the
    -u side, L legs on the -u and -v sides (detection flips the member's
    section frame so this matches reality).
    """
    h2, b2 = height / 2.0, width / 2.0
    tw, tf = _section_thickness(profile, family, height)

    if family == "I/H":
        w2 = tw / 2.0
        outer = np.array(
            [
                (-b2, -h2), (b2, -h2), (b2, -h2 + tf), (w2, -h2 + tf),
                (w2, h2 - tf), (b2, h2 - tf), (b2, h2), (-b2, h2),
                (-b2, h2 - tf), (-w2, h2 - tf), (-w2, -h2 + tf), (-b2, -h2 + tf),
            ]
        )
        return outer, []
    if family == "U":
        outer = np.array(
            [
                (-b2, -h2), (b2, -h2), (b2, -h2 + tf), (-b2 + tw, -h2 + tf),
                (-b2 + tw, h2 - tf), (b2, h2 - tf), (b2, h2), (-b2, h2),
            ]
        )
        return outer, []
    if family == "L":
        outer = np.array(
            [
                (-b2, -h2), (b2, -h2), (b2, -h2 + tw), (-b2 + tw, -h2 + tw),
                (-b2 + tw, h2), (-b2, h2),
            ]
        )
        return outer, []
    if family == "hollow":
        outer = np.array([(-b2, -h2), (b2, -h2), (b2, h2), (-b2, h2)])
        hole = np.array(
            [
                (-b2 + tw, -h2 + tw), (b2 - tw, -h2 + tw),
                (b2 - tw, h2 - tw), (-b2 + tw, h2 - tw),
            ]
        )
        return outer, [hole]
    # Unknown family: solid rectangle with the measured envelope.
    return np.array([(-b2, -h2), (b2, -h2), (b2, h2), (-b2, h2)]), []


def steel_member_mesh(
    member,
    color: tuple[int, int, int] = (120, 140, 170),
    group: int = 0,
) -> Mesh:
    """Extrude a detected steel member with its catalog cross section."""
    outer, holes = steel_profile_outline(
        member.profile, member.family, member.height, member.width
    )
    if member.section_u is not None and member.section_v is not None:
        u, v = member.section_u, member.section_v
    else:
        u, v = frame_from_axis(member.axis)
    origin = member.section_center if member.section_center is not None else member.centroid
    return extrude_polygon(
        outer, origin, u, v, member.axis, member.length,
        holes=holes, color=color, group=group,
    )
