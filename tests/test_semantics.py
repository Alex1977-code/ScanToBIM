"""Surface classification, quantity takeoff and axis alignment."""

import numpy as np

from scantobim import PipelineConfig, reconstruct
from scantobim.core.cloud import PointCloud
from scantobim.core.mesh import Mesh
from scantobim.core.semantics import classify_surface, signed_volume
from tests.synthetic import make_box_scan

BOX_SIZE = (4.0, 3.0, 2.5)


def test_classify_surface_basics():
    z = np.array([0.0, 0.0, 1.0])
    x = np.array([1.0, 0.0, 0.0])
    slope = np.array([0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    assert classify_surface(z, 0.0, 0.0, 2.5) == "floor"
    assert classify_surface(z, 2.5, 0.0, 2.5) == "ceiling"
    assert classify_surface(z, 1.2, 0.0, 2.5) == "slab"
    assert classify_surface(x, 1.0, 0.0, 2.5) == "wall"
    assert classify_surface(slope, 2.0, 0.0, 2.5) == "sloped"


def test_signed_volume_unit_cube():
    # Unit cube, outward orientation.
    v = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float
    )
    f = np.array(
        [[0, 2, 1], [0, 3, 2],  # bottom (normal -z)
         [4, 5, 6], [4, 6, 7],  # top (+z)
         [0, 1, 5], [0, 5, 4],  # y=0
         [2, 3, 7], [2, 7, 6],  # y=1
         [1, 2, 6], [1, 6, 5],  # x=1
         [3, 0, 4], [3, 4, 7]]  # x=0
    )
    mesh = Mesh(vertices=v, faces=f)
    assert abs(signed_volume(mesh) - 1.0) < 1e-12


def test_box_quantities():
    cloud = make_box_scan(size=BOX_SIZE, density=900, noise=0.004)
    result = reconstruct(cloud, PipelineConfig.preset("building"))
    q = result.report["quantities"]
    assert q["watertight"]
    sx, sy, sz = BOX_SIZE
    assert abs(q["volume"] - sx * sy * sz) / (sx * sy * sz) < 0.05
    assert q["orientation"] == "outward"
    counts = q["surface_count_by_class"]
    assert counts == {"floor": 1, "ceiling": 1, "wall": 4}
    assert abs(q["area_by_class"]["wall"] - 2 * (sx + sy) * sz) < 2.0
    assert abs(q["footprint"]["size_x"] - sx) < 0.05
    assert abs(q["footprint"]["height"] - sz) < 0.05
    # semantic OBJ group names present
    names = set(result.mesh.group_names.values())
    assert any(n.startswith("wall_") for n in names)
    assert any(n.startswith("floor_") for n in names)


def test_alignment_of_rotated_scan():
    cloud = make_box_scan(size=BOX_SIZE, density=700, noise=0.004)
    # Rotate the scan by 30° about Z and 3° about X, then shift.
    a = np.deg2rad(30)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    b = np.deg2rad(3)
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    rot = rx @ rz
    moved = PointCloud(points=cloud.points @ rot.T + np.array([12.0, -4.0, 7.0]))

    cfg = PipelineConfig.preset("building")
    cfg.align_axes = True
    result = reconstruct(moved, cfg)

    # After alignment: floor at Z=0, walls axis-parallel.
    verts = result.mesh.vertices
    assert abs(verts[:, 2].min()) < 1e-6
    assert abs(verts[:, 2].max() - BOX_SIZE[2]) < 0.05
    normals = result.mesh.face_normals()
    axis_dev = np.abs(normals).max(axis=1)
    assert np.all(axis_dev > 0.9999), "all faces must be axis-aligned after --align"
    assert "alignment" in result.report
    t = np.array(result.report["alignment"])
    assert t.shape == (4, 4)


def test_alignment_identity_without_vertical():
    # A single wall (no floor): no reliable vertical → identity rotation.
    from scantobim.core.planes import Plane
    from scantobim.core.transform import compute_alignment

    p = Plane(normal=np.array([1.0, 0.0, 0.0]), d=0.0, inliers=np.arange(100))
    t = compute_alignment([p])
    np.testing.assert_allclose(t, np.eye(4))
