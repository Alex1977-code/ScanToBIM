import numpy as np

from scantobim.core.mesh import triangulate_polygon
from scantobim.core.polygons import (
    alpha_shape_boundary,
    dedupe_polygon,
    polygon_area,
    simplify_polygon,
    straighten_polygon,
)


def _noisy_rect_points(w=4.0, h=2.0, n=4000, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform([0, 0], [w, h], size=(n, 2))
    return pts + rng.normal(0, noise, (n, 2))


def test_alpha_shape_rectangle():
    pts = _noisy_rect_points()
    poly = alpha_shape_boundary(pts, alpha=0.25)
    assert poly is not None
    area = polygon_area(poly)
    assert abs(area - 8.0) < 0.5  # ~ w*h
    # boundary stays near the rectangle edges
    assert poly[:, 0].min() > -0.15 and poly[:, 0].max() < 4.15


def test_alpha_shape_l_shape():
    rng = np.random.default_rng(1)
    pts = rng.uniform([0, 0], [4, 4], size=(8000, 2))
    keep = ~((pts[:, 0] > 2) & (pts[:, 1] > 2))
    poly = alpha_shape_boundary(pts[keep], alpha=0.3)
    assert poly is not None
    assert abs(polygon_area(poly) - 12.0) < 0.8  # 16 - 4


def test_simplify_collinear():
    # A square sampled with many collinear points on each edge.
    t = np.linspace(0, 1, 25)[:-1]
    edges = []
    corners = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        edges.append(a + np.outer(t, b - a))
    poly = np.vstack(edges)
    out = simplify_polygon(poly, tolerance=0.01)
    assert len(out) <= 6  # essentially the 4 corners


def test_straighten_squares_jagged_rectangle():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 1, 30)[:-1]
    corners = np.array([[0, 0], [5, 0], [5, 3], [0, 3]], dtype=float)
    edges = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        seg = a + np.outer(t, b - a)
        seg += rng.normal(0, 0.03, seg.shape)  # jitter
        edges.append(seg)
    poly = np.vstack(edges)
    poly = simplify_polygon(poly, tolerance=0.06)
    out = straighten_polygon(poly, angle_tol_deg=15)
    # Every segment must be parallel or exactly orthogonal to the first one
    # (a small global tilt from the data is fine — right angles are not).
    segs = np.roll(out, -1, axis=0) - out
    angles = np.mod(np.arctan2(segs[:, 1], segs[:, 0]), np.pi)
    rel = np.mod(angles - angles[0], np.pi / 2)
    dev = np.minimum(rel, np.pi / 2 - rel)
    assert np.all(dev < 1e-9)
    assert abs(polygon_area(out) - 15.0) < 1.2


def test_straighten_keeps_valid_shape_on_diagonal():
    # A 45° ramp polygon must survive straightening (no right-angle forcing).
    poly = np.array([[0, 0], [4, 0], [4, 2], [2, 4], [0, 4]], dtype=float)
    out = straighten_polygon(poly, angle_tol_deg=10)
    assert abs(polygon_area(out) - polygon_area(poly)) < 0.5


def test_dedupe():
    poly = np.array([[0, 0], [0.001, 0.001], [1, 0], [1, 1], [0, 1], [0.0005, 0.0]])
    out = dedupe_polygon(poly, min_dist=0.01)
    assert len(out) == 4


def test_triangulate_convex():
    poly = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
    tris = triangulate_polygon(poly)
    assert len(tris) == 2
    area = 0.0
    for a, b, c in tris:
        v1, v2 = poly[b] - poly[a], poly[c] - poly[a]
        area += 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    assert abs(area - 2.0) < 1e-9


def test_triangulate_concave():
    # L-shape: 6 vertices → 4 triangles, total area 12
    poly = np.array([[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]], dtype=float)
    tris = triangulate_polygon(poly)
    assert len(tris) == 4
    area = 0.0
    for a, b, c in tris:
        v1, v2 = poly[b] - poly[a], poly[c] - poly[a]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        assert cross > 0  # orientation preserved (CCW)
        area += 0.5 * cross
    assert abs(area - 12.0) < 1e-9


def test_triangulate_with_collinear_vertices():
    poly = np.array([[0, 0], [1, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    tris = triangulate_polygon(poly)
    area = 0.0
    for a, b, c in tris:
        v1, v2 = poly[b] - poly[a], poly[c] - poly[a]
        area += 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    assert abs(area - 4.0) < 1e-9
