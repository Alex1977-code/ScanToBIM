"""Openings (windows/doors) as holes: alpha-shape loops + hole triangulation
+ end-to-end wall-with-window reconstruction."""

import numpy as np
import pytest

from scantobim import PipelineConfig, reconstruct
from scantobim.core.cloud import PointCloud
from scantobim.core.mesh import triangulate_with_holes
from scantobim.core.polygons import alpha_shape_loops, point_in_polygon, polygon_area
from tests.synthetic import sample_rect


def _tri_area_sum(verts, tris):
    total = 0.0
    for a, b, c in tris:
        v1, v2 = verts[b] - verts[a], verts[c] - verts[a]
        total += 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    return total


def test_alpha_loops_finds_hole():
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 0], [6, 3], size=(14000, 2))
    # Punch a 1.5 x 1.0 window at (2, 1)
    hole_mask = (pts[:, 0] > 2) & (pts[:, 0] < 3.5) & (pts[:, 1] > 1) & (pts[:, 1] < 2)
    pts = pts[~hole_mask]
    outer, holes = alpha_shape_loops(pts, alpha=0.25)
    assert outer is not None
    assert len(holes) == 1
    assert abs(polygon_area(holes[0]) - 1.5) < 0.3
    # hole centroid inside outer
    assert point_in_polygon(holes[0].mean(axis=0), outer)


def test_point_in_polygon():
    square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    assert point_in_polygon(np.array([1.0, 1.0]), square)
    assert not point_in_polygon(np.array([3.0, 1.0]), square)
    assert not point_in_polygon(np.array([-0.1, 1.0]), square)


def test_triangulate_with_single_hole():
    outer = np.array([[0, 0], [6, 0], [6, 3], [0, 3]], dtype=float)
    hole = np.array([[2, 1], [3.5, 1], [3.5, 2], [2, 2]], dtype=float)
    verts, tris = triangulate_with_holes(outer, [hole])
    assert len(tris) >= 8
    assert abs(_tri_area_sum(verts, tris) - (18.0 - 1.5)) < 1e-9
    # No triangle centroid may lie inside the hole.
    for a, b, c in tris:
        centroid = (verts[a] + verts[b] + verts[c]) / 3.0
        inside_hole = (2 < centroid[0] < 3.5) and (1 < centroid[1] < 2)
        assert not inside_hole


def test_triangulate_with_two_holes():
    outer = np.array([[0, 0], [10, 0], [10, 4], [0, 4]], dtype=float)
    h1 = np.array([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=float)
    h2 = np.array([[6, 1], [8, 1], [8, 3], [6, 3]], dtype=float)
    verts, tris = triangulate_with_holes(outer, [h1, h2])
    assert abs(_tri_area_sum(verts, tris) - (40.0 - 4.0 - 4.0)) < 1e-9


def test_triangulate_no_holes_passthrough():
    outer = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
    verts, tris = triangulate_with_holes(outer, [])
    assert len(verts) == 4 and len(tris) == 2


@pytest.fixture(scope="module")
def wall_with_window():
    """Free-standing wall (6 x 3 m) with a 1.5 x 1.0 m window, plus a floor
    slab so plane detection has context."""
    rng = np.random.default_rng(7)
    x, y, z = np.eye(3)
    wall = sample_rect((0, 0, 0), x, z, 6.0, 3.0, 1400, 0.004, rng)
    inside = (
        (wall[:, 0] > 2.0) & (wall[:, 0] < 3.5) & (wall[:, 2] > 1.0) & (wall[:, 2] < 2.0)
    )
    wall = wall[~inside]
    floor = sample_rect((0, -2, 0), x, y, 6.0, 2.0, 1400, 0.004, rng)
    return PointCloud(points=np.vstack([wall, floor]), source="wall+window")


def test_pipeline_reconstructs_window(wall_with_window):
    cfg = PipelineConfig.preset("building")
    result = reconstruct(wall_with_window, cfg)
    walls = [s for s in result.report["surfaces"] if s.get("openings", 0) > 0]
    assert len(walls) == 1, "exactly one surface must carry the window"
    wall = walls[0]
    assert wall["openings"] == 1
    assert abs(wall["opening_areas"][0] - 1.5) < 0.35
    # Net wall area = 18 - 1.5
    assert abs(wall["area"] - 16.5) < 0.8


def test_pipeline_openings_can_be_disabled(wall_with_window):
    cfg = PipelineConfig.preset("building")
    cfg.detect_openings = False
    result = reconstruct(wall_with_window, cfg)
    assert all(s.get("openings", 0) == 0 for s in result.report["surfaces"])


def test_window_classified_with_sill_height():
    from scantobim import PipelineConfig, reconstruct
    from tests.synthetic import make_l_room_scan

    result = reconstruct(
        make_l_room_scan(density=900, noise=0.004, window=True),
        PipelineConfig.preset("indoor"),
    )
    rep = result.report
    details = rep["opening_details"]
    assert len(details) == 1
    win = details[0]
    assert win["type"] == "fenster"
    # Alpha-shape hole boundaries overshoot by ~one resolution cell.
    assert abs(win["width"] - 1.4) < 0.15
    assert abs(win["height"] - 1.0) < 0.15
    assert abs(win["sill_height"] - 0.9) < 0.1  # Brüstungshöhe
    assert rep["quantities"]["windows"] == 1


def test_door_classified():
    """A floor-to-lintel opening (5 cm threshold) is a door, not a window."""
    import numpy as np

    from scantobim import PipelineConfig, reconstruct
    from scantobim.core.cloud import PointCloud
    from tests.synthetic import make_box_scan

    cloud = make_box_scan(density=900, noise=0.004)
    pts = cloud.points
    door = (
        (np.abs(pts[:, 1]) < 0.05)
        & (pts[:, 0] > 1.5) & (pts[:, 0] < 2.4)
        & (pts[:, 2] > 0.2) & (pts[:, 2] < 2.2)
    )
    result = reconstruct(
        PointCloud(points=pts[~door]), PipelineConfig.preset("building")
    )
    details = result.report["opening_details"]
    doors = [d for d in details if d["type"] == "tuer"]
    assert len(doors) == 1
    assert abs(doors[0]["height"] - 2.0) < 0.12
    assert abs(doors[0]["width"] - 0.9) < 0.1
    assert result.report["quantities"]["doors"] == 1
