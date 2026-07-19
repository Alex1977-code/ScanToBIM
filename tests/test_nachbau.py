"""Bauteil-Nachbau: verworfene Detail-Regionen einzeln neu vernetzen,
Fensterscheiben aus Öffnungen und Verschmelzen zu EINEM Modell."""

from types import SimpleNamespace

import numpy as np

from scantobim.cli import (
    _build_window_panes,
    _merge_plain,
    _merge_regions,
    _rebuild_elements,
    _strip_detail_clutter,
)
from scantobim.core.cloud import PointCloud
from scantobim.core.mesh import Mesh


def _grid(n, half, offset=(0.0, 0.0, 0.0)):
    g = np.linspace(-half, half, n)
    xx, yy = np.meshgrid(g, g)
    verts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    verts = verts + np.asarray(offset, dtype=np.float64)
    idx = lambda i, j: i * n + j  # noqa: E731
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            faces.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
            faces.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])
    return verts, np.asarray(faces, dtype=np.int64)


def test_strip_reports_dropped_regions():
    # Main component: 30x30 quads = 1800 faces (>= 1500 keeps it).
    v1, f1 = _grid(31, 2.0)
    # Small component far away: 6x6 quads = 72 faces (>= 50 -> region).
    v2, f2 = _grid(7, 0.5, offset=(10.0, 0.0, 0.0))
    mesh = Mesh(
        vertices=np.vstack([v1, v2]),
        faces=np.vstack([f1, f2 + len(v1)]),
    )
    mesh.freeform_stats = {"voxel": 0.02}
    stripped, boxes = _strip_detail_clutter(mesh)
    assert len(stripped.faces) == len(f1)
    assert len(boxes) == 1
    lo, hi = boxes[0]
    assert np.all(lo <= np.array([10.0, 0.0, 0.0]))
    assert np.all(hi >= np.array([10.0, 0.0, 0.0]))


def test_merge_regions_unions_overlaps_and_caps():
    z = np.zeros(3)
    a = (z, z + 1.0, 100)
    b = (z + 1.2, z + 2.0, 50)   # overlaps a after 0.25 dilation
    c = (z + 50.0, z + 51.0, 10)  # far away
    merged = _merge_regions([a, b, c], dilate=0.25, cap=40)
    assert len(merged) == 2
    lo0, hi0 = merged[0]  # largest weight first: a+b
    assert np.all(hi0 >= 2.0) and np.all(lo0 <= 0.0)
    capped = _merge_regions([a, b, c], dilate=0.25, cap=1)
    assert len(capped) == 1


def test_rebuild_elements_meshes_each_region():
    rng = np.random.default_rng(0)
    g = np.linspace(-0.25, 0.25, 60)
    xx, yy = np.meshgrid(g, g)
    pts = np.column_stack([
        xx.ravel() + 5.0, yy.ravel() + 5.0,
        rng.normal(0.0, 0.002, xx.size),
    ])
    cloud = PointCloud(points=pts)
    box_ok = (np.array([4.5, 4.5, -0.5]), np.array([5.5, 5.5, 0.5]))
    box_empty = (np.array([90.0, 90.0, 0.0]), np.array([91.0, 91.0, 1.0]))
    out = _rebuild_elements(cloud, [box_ok, box_empty], raster=0.02)
    assert len(out) == 1
    m = out[0]
    assert len(m.faces) >= 150
    assert np.all(m.vertices.min(axis=0) >= box_ok[0] - 0.1)
    assert np.all(m.vertices.max(axis=0) <= box_ok[1] + 0.1)


def test_window_panes_from_surface_holes():
    ring = np.array(
        [[1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [2.0, 0.0, 2.0], [1.0, 0.0, 2.0]]
    )
    tiny = ring * 0.05  # 0.0025 m² -> below area filter
    wall = SimpleNamespace(
        surface_class="wall",
        normal=np.array([0.0, 1.0, 0.0]),
        outer=np.array(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 0.0, 4.0], [0.0, 0.0, 4.0]]
        ),
        holes=[ring, tiny],
    )
    terrain = SimpleNamespace(
        surface_class="terrain",
        normal=np.array([0.0, 0.0, 1.0]),
        outer=wall.outer,
        holes=[ring],
    )
    panes = _build_window_panes(SimpleNamespace(surfaces=[wall, terrain]))
    assert len(panes) == 1
    pane = panes[0]
    assert len(pane.faces) == 4  # centroid fan over the 4-point ring
    assert np.allclose(pane.vertices[:, 1], -0.04)  # inset along -normal
    assert pane.vertex_colors is not None


def test_merge_plain_concatenates_and_fills_colors():
    v1, f1 = _grid(3, 1.0)
    v2, f2 = _grid(3, 1.0, offset=(5.0, 0.0, 0.0))
    m1 = Mesh(
        vertices=v1, faces=f1,
        vertex_colors=np.full((len(v1), 3), 90, dtype=np.uint8),
    )
    m1.freeform_stats = {"voxel": 0.02}
    m2 = Mesh(vertices=v2, faces=f2)
    merged = _merge_plain([m1, m2])
    assert len(merged.vertices) == len(v1) + len(v2)
    assert len(merged.faces) == len(f1) + len(f2)
    assert int(merged.faces[len(f1):].min()) == len(v1)
    assert merged.vertex_colors is not None
    assert np.all(merged.vertex_colors[: len(v1)] == 90)
    assert np.all(merged.vertex_colors[len(v1):] == 180)
    assert merged.freeform_stats["triangles"] == len(merged.faces)


def test_close_planar_holes_on_roof():
    n = 15
    v, f = _grid(n, 1.0)
    center = (n // 2) * n + n // 2
    keep = ~np.any(f == center, axis=1)
    mesh = Mesh(vertices=v, faces=f[keep])
    mesh.freeform_stats = {"voxel": 0.02}
    roof = SimpleNamespace(
        surface_class="roof",
        normal=np.array([0.0, 0.0, 1.0]),
        outer=np.array(
            [[-1.2, -1.2, 0.0], [1.2, -1.2, 0.0],
             [1.2, 1.2, 0.0], [-1.2, 1.2, 0.0]]
        ),
        holes=[],
    )
    from scantobim.cli import _close_planar_holes

    before = len(mesh.faces)
    closed = _close_planar_holes(
        mesh, SimpleNamespace(surfaces=[roof])
    )
    assert closed == 1
    assert len(mesh.faces) > before  # fan filled the hole


def test_strip_floaters_removes_far_fragments():
    v1, f1 = _grid(21, 1.0)  # main: 800 faces
    far = np.array([[5.0, 5.0, 5.0], [5.1, 5.0, 5.0], [5.0, 5.1, 5.0]])
    near = np.array([[1.2, 0.0, 0.0], [1.3, 0.0, 0.0], [1.2, 0.1, 0.0]])
    tri = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = Mesh(
        vertices=np.vstack([v1, far, near]),
        faces=np.vstack([f1, tri + len(v1), tri + len(v1) + 3]),
    )
    mesh.freeform_stats = {"voxel": 0.02}
    from scantobim.cli import _strip_floaters

    out, n_removed = _strip_floaters(mesh)
    assert n_removed == 1  # far fragment gone, near one keeps its place
    assert len(out.faces) == len(f1) + 1


def test_replace_regions_rebuilds_corrupt_patch():
    v, f = _grid(31, 1.0)
    corrupt = (np.abs(v[:, 0]) < 0.4) & (np.abs(v[:, 1]) < 0.4)
    v = v.copy()
    v[corrupt, 2] = 0.3  # frayed zone bulges 30 cm out of the plane
    mesh = Mesh(vertices=v, faces=f)
    mesh.freeform_stats = {"voxel": 0.02}
    rng = np.random.default_rng(0)
    g = np.linspace(-0.5, 0.5, 71)
    xx, yy = np.meshgrid(g, g)
    sub = PointCloud(points=np.column_stack(
        [xx.ravel(), yy.ravel(), rng.normal(0.0, 0.002, xx.size)]
    ))
    boxes = [(np.array([-0.5, -0.5, -0.3]), np.array([0.5, 0.5, 0.4]))]
    from scantobim.cli import _replace_regions

    merged, n_rep = _replace_regions(mesh, boxes, sub, 0.02)
    assert n_rep == 1
    cent = merged.vertices[merged.faces].mean(axis=1)
    inside = np.all(
        (cent >= boxes[0][0]) & (cent <= boxes[0][1]), axis=1
    )
    assert inside.any()
    assert float(np.abs(cent[inside][:, 2]).max()) < 0.1  # bulge replaced
