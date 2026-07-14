"""Round-trip tests for readers and writers."""

import json
import struct

import numpy as np
import pytest

from scantobim.core.cloud import PointCloud
from scantobim.core.mesh import Mesh
from scantobim.io.readers import read_point_cloud
from scantobim.io.writers import write_mesh, write_point_cloud


@pytest.fixture
def cloud():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-5, 5, (500, 3))
    colors = rng.integers(0, 256, (500, 3), dtype=np.uint8)
    normals = np.tile([0.0, 0.0, 1.0], (500, 1))
    return PointCloud(points=pts, colors=colors, normals=normals)


@pytest.fixture
def mesh():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    colors = np.array([[255, 0, 0]] * 4, dtype=np.uint8)
    return Mesh(vertices=vertices, faces=faces, vertex_colors=colors,
                face_groups=np.array([0, 0]))


def test_ply_cloud_roundtrip(tmp_path, cloud):
    path = write_point_cloud(cloud, tmp_path / "cloud.ply")
    back = read_point_cloud(path)
    assert len(back) == len(cloud)
    np.testing.assert_allclose(back.points, cloud.points, atol=1e-4)
    np.testing.assert_array_equal(back.colors, cloud.colors)
    np.testing.assert_allclose(back.normals, cloud.normals, atol=1e-4)


def test_ply_ascii_read(tmp_path):
    path = tmp_path / "ascii.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
        "0 0 0 255 0 0\n1 0 0 0 255 0\n0 1 2.5 0 0 255\n"
    )
    cloud = read_point_cloud(path)
    assert len(cloud) == 3
    np.testing.assert_allclose(cloud.points[2], [0, 1, 2.5])
    np.testing.assert_array_equal(cloud.colors[0], [255, 0, 0])


def test_xyz_and_pts(tmp_path):
    path = tmp_path / "cloud.xyz"
    path.write_text("0 0 0\n1 2 3\n4 5 6\n")
    assert len(read_point_cloud(path)) == 3

    # Leica-style .pts with count header and intensity+RGB
    pts = tmp_path / "cloud.pts"
    pts.write_text("2\n0 0 0 -1024 10 20 30\n1 1 1 0 40 50 60\n")
    cloud = read_point_cloud(pts)
    assert len(cloud) == 2
    assert cloud.colors is not None
    np.testing.assert_array_equal(cloud.colors[1], [40, 50, 60])
    assert cloud.intensity is not None


def test_csv_with_header(tmp_path):
    path = tmp_path / "cloud.csv"
    path.write_text("x,y,z\n0,0,0\n1,2,3\n")
    cloud = read_point_cloud(path)
    assert len(cloud) == 2
    np.testing.assert_allclose(cloud.points[1], [1, 2, 3])


def test_las_roundtrip(tmp_path, cloud):
    import laspy

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x, las.y, las.z = cloud.points.T
    las.red = cloud.colors[:, 0].astype(np.uint16) * 257
    las.green = cloud.colors[:, 1].astype(np.uint16) * 257
    las.blue = cloud.colors[:, 2].astype(np.uint16) * 257
    path = tmp_path / "cloud.las"
    las.write(str(path))

    back = read_point_cloud(path)
    assert len(back) == len(cloud)
    np.testing.assert_allclose(back.points, cloud.points, atol=0.002)
    np.testing.assert_array_equal(back.colors, cloud.colors)


def test_laz_roundtrip(tmp_path, cloud):
    pytest.importorskip("lazrs")
    import laspy

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x, las.y, las.z = cloud.points.T
    path = tmp_path / "cloud.laz"
    las.write(str(path))
    back = read_point_cloud(path)
    np.testing.assert_allclose(back.points, cloud.points, atol=0.002)


def test_pcd_binary(tmp_path):
    pts = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]], dtype=np.float32)
    header = (
        "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
        "COUNT 1 1 1\nWIDTH 3\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS 3\nDATA binary\n"
    )
    path = tmp_path / "cloud.pcd"
    with open(path, "wb") as fh:
        fh.write(header.encode())
        fh.write(pts.tobytes())
    cloud = read_point_cloud(path)
    assert len(cloud) == 3
    np.testing.assert_allclose(cloud.points[1], [1, 2, 3], atol=1e-5)


def test_obj_write(tmp_path, mesh):
    path = write_mesh(mesh, tmp_path / "model.obj")
    text = path.read_text()
    assert "v 0.000000 0.000000 0.000000" in text
    assert "f 1//1 2//2 3//3" in text
    assert "g surface_000" in text


def test_stl_write(tmp_path, mesh):
    path = write_mesh(mesh, tmp_path / "model.stl")
    raw = path.read_bytes()
    n_tri = struct.unpack("<I", raw[80:84])[0]
    assert n_tri == 2
    assert len(raw) == 84 + 50 * 2


def test_ply_mesh_write(tmp_path, mesh):
    path = write_mesh(mesh, tmp_path / "model.ply")
    raw = path.read_bytes()
    assert raw.startswith(b"ply")
    assert b"element vertex 4" in raw
    assert b"element face 2" in raw


def test_glb_write(tmp_path, mesh):
    path = write_mesh(mesh, tmp_path / "model.glb")
    raw = path.read_bytes()
    magic, version, total = struct.unpack("<III", raw[:12])
    assert magic == 0x46546C67 and version == 2 and total == len(raw)
    json_len, json_type = struct.unpack("<II", raw[12:20])
    assert json_type == 0x4E4F534A
    gltf = json.loads(raw[20 : 20 + json_len])
    assert gltf["asset"]["version"] == "2.0"
    acc = gltf["accessors"]
    assert acc[0]["count"] == 4  # positions
    assert acc[-1]["count"] == 6  # indices
    assert "COLOR_0" in gltf["meshes"][0]["primitives"][0]["attributes"]


def test_gltf_write(tmp_path, mesh):
    path = write_mesh(mesh, tmp_path / "model.gltf")
    gltf = json.loads(path.read_text())
    assert gltf["buffers"][0]["uri"].startswith("data:application/octet-stream;base64,")


def test_unknown_format(tmp_path, mesh):
    with pytest.raises(ValueError, match="Unsupported"):
        write_mesh(mesh, tmp_path / "model.xyz")
    p = tmp_path / "cloud.abc"
    p.write_text("")
    with pytest.raises(ValueError, match="Unsupported"):
        read_point_cloud(p)


def test_e57_roundtrip_with_colors(tmp_path):
    """Colorized scanner E57 (like real exports) reads with points + RGB."""
    pye57 = pytest.importorskip("pye57")
    import numpy as np

    from tests.synthetic import make_box_scan

    cloud = make_box_scan(density=300, noise=0.004)
    rng = np.random.default_rng(1)
    colors = rng.integers(40, 220, (len(cloud.points), 3))
    e57 = pye57.E57(str(tmp_path / "scan.e57"), mode="w")
    e57.write_scan_raw(
        {
            "cartesianX": cloud.points[:, 0],
            "cartesianY": cloud.points[:, 1],
            "cartesianZ": cloud.points[:, 2],
            "colorRed": colors[:, 0].astype(np.uint8),
            "colorGreen": colors[:, 1].astype(np.uint8),
            "colorBlue": colors[:, 2].astype(np.uint8),
        }
    )
    e57.close()

    got = read_point_cloud(tmp_path / "scan.e57")
    assert len(got) == len(cloud.points)
    assert got.colors is not None and got.colors.dtype.kind == "u"
    assert got.colors.min() >= 40 and got.colors.max() <= 220


def test_streaming_thinner_surface_site_not_crushed():
    """Regression: a surface scan of a sprawling site must fill the budget.

    The grow-only voxel search started at diagonal/1000 — a 230 m site was
    crushed to 22 cm spacing (45.7M points → 157k) instead of the budget.
    """
    from scantobim.io.readers import StreamingThinner

    # Dense planar scan (2.5 cm grid) + one distant stray point that
    # stretches the diagonal to ~870 m.
    g = np.linspace(0, 20, 800)
    xx, yy = np.meshgrid(g, g)
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    pts = np.vstack([pts, [[500.0, 500.0, 500.0]]])
    thinner = StreamingThinner(100_000)
    for start in range(0, len(pts), 150_000):
        thinner.add(pts[start:start + 150_000], None, None)
    cloud = thinner.finish()
    assert len(cloud) <= 100_000
    assert len(cloud) >= 70_000  # near the budget — not crushed to diag/1000


def test_streaming_thinner_bounds_and_coverage():
    """Huge input → bounded output that still covers the whole extent."""
    from scantobim.io.readers import StreamingThinner

    rng = np.random.default_rng(0)
    thinner = StreamingThinner(20_000)
    for _ in range(10):  # 500k points in blocks of 50k
        pts = rng.uniform(0, 10, (50_000, 3))
        colors = rng.integers(0, 255, (50_000, 3)).astype("uint8")
        thinner.add(pts, colors, None)
    cloud = thinner.finish()
    assert len(cloud) <= 20_000
    assert cloud.colors is not None and len(cloud.colors) == len(cloud)
    lo, hi = cloud.points.min(axis=0), cloud.points.max(axis=0)
    assert np.all(lo < 0.5) and np.all(hi > 9.5)  # full extent preserved


def test_read_las_streamed_with_max_points(tmp_path):
    """LAS beyond the budget is read block-wise and thinned."""
    import laspy

    from tests.synthetic import make_box_scan

    cloud = make_box_scan(density=2000, noise=0.004)  # ~92k points
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.0001, 0.0001, 0.0001])
    las = laspy.LasData(header)
    las.x, las.y, las.z = cloud.points.T
    src = tmp_path / "big.las"
    las.write(str(src))
    got = read_point_cloud(src, max_points=30_000)
    assert 5_000 < len(got) <= 30_000
    lo, hi = got.aabb
    assert np.all(hi - lo > np.array([3.9, 2.9, 2.4]))  # extent intact

    full = read_point_cloud(src)  # without budget: everything
    assert len(full) == len(cloud.points)


def test_read_ply_thinned(tmp_path):
    from tests.synthetic import make_box_scan

    cloud = make_box_scan(density=2000, noise=0.004)
    src = write_point_cloud(cloud, tmp_path / "big.ply")
    got = read_point_cloud(src, max_points=25_000)
    assert len(got) <= 25_000


def test_e57_16bit_colors_normalized(tmp_path):
    """S20-style 16-bit E57 colors must not clip to a white model."""
    pye57 = pytest.importorskip("pye57")
    import numpy as np

    from tests.synthetic import make_box_scan

    cloud = make_box_scan(density=200, noise=0.004)
    rng = np.random.default_rng(3)
    rgb16 = rng.integers(5000, 60000, (len(cloud.points), 3)).astype(np.uint16)
    e57 = pye57.E57(str(tmp_path / "s16.e57"), mode="w")
    e57.write_scan_raw({
        "cartesianX": cloud.points[:, 0],
        "cartesianY": cloud.points[:, 1],
        "cartesianZ": cloud.points[:, 2],
        "colorRed": rgb16[:, 0],
        "colorGreen": rgb16[:, 1],
        "colorBlue": rgb16[:, 2],
    })
    e57.close()
    got = read_point_cloud(tmp_path / "s16.e57")
    assert got.colors is not None
    # Real variation preserved instead of everything clipping to 255.
    assert got.colors.max() <= 255 and got.colors.min() >= 0
    assert got.colors.std() > 20
    assert (got.colors == 255).mean() < 0.1
