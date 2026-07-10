import numpy as np

from scantobim.core.planes import Plane
from scantobim.core.edges import (
    intersect_planes,
    intersect_three_planes,
    snap_points_to_line,
)


def _plane(normal, d):
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    p = Plane(normal=normal, d=float(d), inliers=np.zeros(0, dtype=int))
    p.make_basis()
    return p


def test_intersect_two_planes():
    # x=1 wall and y=2 wall meet in the vertical line (1, 2, t)
    a = _plane([1, 0, 0], -1.0)
    b = _plane([0, 1, 0], -2.0)
    point, direction = intersect_planes(a, b)
    assert abs(abs(direction[2]) - 1.0) < 1e-12
    assert abs(point[0] - 1.0) < 1e-12 and abs(point[1] - 2.0) < 1e-12


def test_intersect_parallel_returns_none():
    a = _plane([0, 0, 1], 0.0)
    b = _plane([0, 0, 1], -3.0)
    assert intersect_planes(a, b) is None


def test_three_plane_corner():
    a = _plane([1, 0, 0], -1.0)
    b = _plane([0, 1, 0], -2.0)
    c = _plane([0, 0, 1], -3.0)
    corner = intersect_three_planes(a, b, c)
    np.testing.assert_allclose(corner, [1, 2, 3], atol=1e-12)


def test_three_plane_degenerate():
    a = _plane([1, 0, 0], 0.0)
    b = _plane([0, 1, 0], 0.0)
    c = _plane([1, 1, 0], -1.0)  # parallel to the z axis pencil
    assert intersect_three_planes(a, b, c) is None


def test_snap_points_to_line():
    line_pt = np.array([0.0, 0.0, 0.0])
    line_dir = np.array([1.0, 0.0, 0.0])
    pts = np.array(
        [
            [2.0, 0.05, 0.0],  # near → snapped to (2,0,0)
            [5.0, 3.0, 0.0],  # far → untouched
        ]
    )
    out, mask = snap_points_to_line(pts, line_pt, line_dir, max_distance=0.1)
    assert mask[0] and not mask[1]
    np.testing.assert_allclose(out[0], [2, 0, 0], atol=1e-12)
    np.testing.assert_allclose(out[1], pts[1])
