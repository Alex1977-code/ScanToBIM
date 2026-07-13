"""Parametric solids: involute gears, steel profile extrusion, bridge parts."""

import numpy as np

from scantobim.core.geometry3d import (
    box_mesh,
    cone_mesh,
    extrude_polygon,
    involute_gear_mesh,
    involute_gear_outline,
    steel_member_mesh,
    steel_profile_outline,
)
from scantobim.core.semantics import signed_volume


def _outline_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ------------------------------------------------------------------ gears

def test_involute_outline_geometry():
    z, m = 24, 0.004
    outline = involute_gear_outline(z, m)
    r = np.linalg.norm(outline, axis=1)
    rp = m * z / 2
    ra, rf = rp + m, rp - 1.25 * m
    assert abs(r.min() - rf) < 1e-9  # root circle
    assert abs(r.max() - ra) < 1e-9  # tip circle
    # Tooth area lies strictly between root and tip discs.
    area = _outline_area(outline)
    assert np.pi * rf**2 < area < np.pi * ra**2
    # z-fold symmetry: rotating by one pitch maps the outline onto itself.
    pitch = 2 * np.pi / z
    c, s = np.cos(pitch), np.sin(pitch)
    rotated = outline @ np.array([[c, s], [-s, c]])
    n_tooth = len(outline) // z
    assert np.allclose(np.roll(outline, -n_tooth, axis=0), rotated, atol=1e-12)


def test_involute_flank_is_involute():
    """Flank points must satisfy the involute equation of the base circle."""
    z, m = 30, 0.003
    alpha = np.deg2rad(20.0)
    rb = m * z / 2 * np.cos(alpha)
    outline = involute_gear_outline(z, m, flank_points=12)
    r = np.linalg.norm(outline, axis=1)
    # Points strictly between base and tip circle belong to flanks.
    flank = outline[(r > rb * 1.001) & (r < (m * z / 2 + m) * 0.999)]
    phi = np.arctan2(flank[:, 1], flank[:, 0])
    rr = np.linalg.norm(flank, axis=1)
    t = np.sqrt((rr / rb) ** 2 - 1)
    inv = t - np.arctan(t)  # involute function at that radius
    # Tooth center lines sit at k*pitch; the flank's angular distance from
    # its center line must equal half_pitch + inv(alpha) - inv(r).
    pitch = 2 * np.pi / z
    expected = np.pi / (2 * z) + (np.tan(alpha) - alpha) - inv
    center_dist = np.abs((phi + pitch / 2) % pitch - pitch / 2)
    assert np.allclose(center_dist, np.abs(expected), atol=1e-9)


def test_gear_mesh_solid():
    gear = involute_gear_mesh(
        center=[0.1, -0.2, 0.05], axis=[0, 1, 0], teeth=20, module=0.005,
        width=0.03, bore_radius=0.01,
    )
    stats = gear.edge_stats()
    assert stats["boundary_edges"] == 0
    assert stats["non_manifold_edges"] == 0
    v = signed_volume(gear)
    outline_area = _outline_area(involute_gear_outline(20, 0.005))
    expected = (outline_area - np.pi * 0.01**2) * 0.03
    assert abs(v - expected) / expected < 0.01  # bore is a 32-gon


# ------------------------------------------------------------- extrusion

def test_extrusion_watertight_and_oriented():
    outer = np.array([(-1.0, -0.5), (1.0, -0.5), (1.0, 0.5), (-1.0, 0.5)])
    hole = np.array([(-0.5, -0.2), (0.5, -0.2), (0.5, 0.2), (-0.5, 0.2)])
    solid = extrude_polygon(
        outer, [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 2.0, holes=[hole]
    )
    stats = solid.edge_stats()
    assert stats["boundary_edges"] == 0 and stats["non_manifold_edges"] == 0
    assert abs(signed_volume(solid) - (2 * 1 - 1 * 0.4) * 2.0) < 1e-9


def test_box_mesh():
    box = box_mesh([1.0, 2.0, 3.0], [0, 1, 0], (4.0, 2.0, 1.0))
    assert abs(signed_volume(box) - 8.0) < 1e-9
    lo, hi = box.vertices.min(axis=0), box.vertices.max(axis=0)
    assert abs(hi[2] - 3.5) < 1e-9 and abs(lo[2] - 2.5) < 1e-9


def test_cone_mesh_solid():
    cone = cone_mesh([0, 0, 0], [0, 0, 1], 0.05, 0.1, 0.2, offset=0.2)
    stats = cone.edge_stats()
    assert stats["boundary_edges"] == 0
    # Frustum volume h*pi/3*(R^2+Rr+r^2), 48-gon slightly smaller.
    v_exact = 0.2 * np.pi / 3 * (0.1**2 + 0.1 * 0.05 + 0.05**2)
    assert 0.98 * v_exact < signed_volume(cone) < v_exact


# --------------------------------------------------------- steel profiles

def test_profile_outlines_match_envelope():
    cases = [
        ("IPE 200", "I/H", 0.2, 0.1), ("HEB 300", "I/H", 0.3, 0.3),
        ("UPN 200", "U", 0.2, 0.075), ("L 80x80", "L", 0.08, 0.08),
        ("SHS 100", "hollow", 0.1, 0.1), ("RHS 120x60", "hollow", 0.12, 0.06),
        ("unbekannt", "?", 0.15, 0.1),
    ]
    for profile, family, h, b in cases:
        outer, holes = steel_profile_outline(profile, family, h, b)
        ext = outer.max(axis=0) - outer.min(axis=0)
        assert abs(ext[0] - b) < 1e-9 and abs(ext[1] - h) < 1e-9, profile
        assert (len(holes) == 1) == (family == "hollow")


def test_ipe_section_area():
    """Extruded IPE 200 must have the catalog cross-section area (~28.5 cm²)."""
    outer, holes = steel_profile_outline("IPE 200", "I/H", 0.2, 0.1)
    solid = extrude_polygon(
        outer, [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 1.0, holes=holes
    )
    area = signed_volume(solid)  # length 1 → volume == section area
    assert abs(area - 28.5e-4) < 1.5e-4


def _outline_distance(q: np.ndarray, outline: np.ndarray) -> np.ndarray:
    """Distance of 2D points to the closed polyline ``outline``."""
    d_min = np.full(len(q), np.inf)
    for i in range(len(outline)):
        a, b = outline[i], outline[(i + 1) % len(outline)]
        ab = b - a
        tt = np.clip((q - a) @ ab / max(ab @ ab, 1e-18), 0.0, 1.0)
        proj = a + np.outer(tt, ab)
        d_min = np.minimum(d_min, np.linalg.norm(q - proj, axis=1))
    return d_min


def test_member_mesh_matches_scan():
    """The extruded L angle must lie ON the scanned surfaces (orientation!)."""
    from scantobim.core.steel import detect_steel_members
    from tests.synthetic import make_l_angle_scan

    cloud = make_l_angle_scan()
    members = detect_steel_members(cloud)
    assert len(members) == 1
    m = members[0]
    assert m.profile == "L 80x80"
    assert m.section_u is not None

    # Project the scan into the detected section frame and measure the
    # distance to the canonical outline — wrong flips put legs on the
    # wrong side and blow this up to centimeters.
    outer, _ = steel_profile_outline(m.profile, m.family, m.height, m.width)
    rel = cloud.points - m.section_center
    q = np.column_stack([rel @ m.section_u, rel @ m.section_v])
    dist = _outline_distance(q, outer)
    assert np.median(dist) < 0.004  # noise 1 mm + leg thickness 8 mm / 2

    mesh = steel_member_mesh(m)
    assert mesh.edge_stats()["boundary_edges"] == 0
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    assert abs((hi[0] - lo[0]) - m.length) < 0.02


def test_ipe_member_mesh_envelope():
    from scantobim.core.steel import detect_steel_members
    from tests.synthetic import make_ipe_beam_scan

    members = detect_steel_members(make_ipe_beam_scan())
    assert len(members) == 1 and members[0].profile == "IPE 200"
    mesh = steel_member_mesh(members[0])
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    # Envelope: length ~3 m along x, 0.1 m wide, 0.2 m tall.
    assert abs((hi[0] - lo[0]) - 3.0) < 0.05
    assert abs((hi[1] - lo[1]) - 0.1) < 0.01
    assert abs((hi[2] - lo[2]) - 0.2) < 0.01
    # True profile, not a solid box: volume ≈ A(IPE 200)·L = 28.5 cm² · 3 m.
    v = signed_volume(mesh)
    assert abs(v - 28.5e-4 * 3.0) / (28.5e-4 * 3.0) < 0.08
