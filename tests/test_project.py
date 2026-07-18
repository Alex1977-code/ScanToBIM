"""SLAM scanner project import: folder detection, trajectory, binary COLMAP."""

import json
import struct

import numpy as np
import pytest

from scantobim.io.project import read_trajectory, scan_project_dir
from scantobim.io.writers import write_point_cloud
from tests.synthetic import make_box_scan


# ------------------------------------------------------------- trajectory

def test_read_trajectory_plain_xyz(tmp_path):
    p = tmp_path / "trajectory.txt"
    p.write_text("0 0 1.7\n1 0 1.7\n2 0 1.7\n")
    traj = read_trajectory(p)
    assert traj.shape == (3, 3)
    assert np.allclose(traj[:, 2], 1.7)


def test_read_trajectory_timestamped_and_tum(tmp_path):
    p = tmp_path / "traj_time.txt"
    p.write_text("# time x y z\n100.0 5 6 7\n100.1 5 6 8\n100.2 5 6 9\n")
    traj = read_trajectory(p)
    assert traj.shape == (3, 4)  # positions + recognized time column
    assert np.allclose(traj[0, :3], [5, 6, 7]) and traj[0, 3] == 100.0

    tum = tmp_path / "traj_tum.txt"
    tum.write_text(
        "100.0 1 2 3 0 0 0 1\n100.1 1 2 4 0 0 0 1\n100.2 1 2 5 0 0 0 1\n"
    )
    traj = read_trajectory(tum)
    assert traj.shape == (3, 4)
    assert np.allclose(traj[:, 0], 1) and np.allclose(traj[-1, :3], [1, 2, 5])
    assert np.allclose(traj[:, 3], [100.0, 100.1, 100.2])


def test_read_trajectory_csv(tmp_path):
    p = tmp_path / "path.csv"
    p.write_text("x,y,z\n1,2,3\n4,5,6\n")
    assert np.allclose(read_trajectory(p), [[1, 2, 3], [4, 5, 6]])


def test_trajectory_orients_normals_inward():
    """An indoor scan with the scanner path inside: normals face the path."""
    from scantobim.core.preprocess import (
        estimate_normals,
        orient_normals_along_trajectory,
    )

    cloud = make_box_scan(size=(4.0, 3.0, 2.5), density=400, noise=0.003)
    estimate_normals(cloud, k_neighbors=12)
    traj = np.column_stack(
        [np.linspace(1, 3, 20), np.full(20, 1.5), np.full(20, 1.6)]
    )
    orient_normals_along_trajectory(cloud, traj)
    to_path = traj[len(traj) // 2] - cloud.points
    aligned = np.einsum("ij,ij->i", cloud.normals, to_path) > 0
    assert aligned.mean() > 0.95


def test_reconstruct_with_trajectory(tmp_path):
    from scantobim import PipelineConfig, reconstruct

    cloud = make_box_scan(density=700, noise=0.004)
    traj = np.array([[2.0, 1.5, 1.2], [2.5, 1.5, 1.2]])
    result = reconstruct(cloud, PipelineConfig.preset("building"), trajectory=traj)
    assert result.report["planes"] == 6
    assert result.report["trajectory_positions"] == 2


# ----------------------------------------------------------- binary COLMAP

def _write_binary_model(model_dir, poses_spec):
    """Minimal cameras.bin/images.bin writer (PINHOLE) for tests."""
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "cameras.bin", "wb") as fh:
        fh.write(struct.pack("<Q", 1))
        fh.write(struct.pack("<iiQQ", 1, 1, 400, 300))  # PINHOLE 400x300
        fh.write(struct.pack("<4d", 350.0, 350.0, 200.0, 150.0))
    with open(model_dir / "images.bin", "wb") as fh:
        fh.write(struct.pack("<Q", len(poses_spec)))
        for i, (name, q, t) in enumerate(poses_spec):
            fh.write(struct.pack("<idddddddi", i + 1, *q, *t, 1))
            fh.write(name.encode() + b"\x00")
            fh.write(struct.pack("<Q", 2))  # two 2D points to skip
            fh.write(struct.pack("<ddQ", 1.0, 2.0, 7))
            fh.write(struct.pack("<ddQ", 3.0, 4.0, 18446744073709551615))


def test_read_colmap_binary_model(tmp_path):
    from scantobim.photogrammetry.colmap import read_colmap_model

    _write_binary_model(
        tmp_path / "sparse" / "0",
        [
            ("img_001.jpg", (1.0, 0.0, 0.0, 0.0), (0.5, -0.5, 2.0)),
            ("img_002.jpg", (0.0, 1.0, 0.0, 0.0), (1.5, 0.5, 3.0)),
        ],
    )
    poses = read_colmap_model(tmp_path)  # finds sparse/0 automatically
    assert [p.name for p in poses] == ["img_001.jpg", "img_002.jpg"]
    p = poses[0]
    assert (p.width, p.height, p.fx, p.cx) == (400, 300, 350.0, 200.0)
    assert np.allclose(p.rotation, np.eye(3))
    assert np.allclose(p.translation, [0.5, -0.5, 2.0])
    # Second pose: quaternion (0,1,0,0) = 180° about x.
    assert np.allclose(poses[1].rotation, np.diag([1.0, -1.0, -1.0]))


# ------------------------------------------------------- folder detection

def _build_project(tmp_path, with_photos=True):
    root = tmp_path / "s20_export"
    (root / "lidar").mkdir(parents=True)
    cloud = make_box_scan(density=500, noise=0.004)
    rng = np.random.default_rng(2)
    from scantobim.core.cloud import PointCloud

    colored = PointCloud(
        points=cloud.points,
        colors=rng.integers(60, 200, (len(cloud.points), 3), dtype=np.uint8),
    )
    write_point_cloud(colored, root / "lidar" / "cloud.ply")
    (root / "trajectory.txt").write_text(
        "\n".join(f"{i * 0.1:.2f} {2 + i * 0.01} 1.5 1.6" for i in range(50))
    )
    (root / "log.json").write_text("{}")
    (root / "raw.bag").write_bytes(b"ROSBAG-DUMMY")
    if with_photos:
        img_dir = root / "camera" / "undistorted"
        img_dir.mkdir(parents=True)
        for i in range(6):
            (img_dir / f"frame_{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xdb dummy")
        _write_binary_model(
            root / "camera" / "sparse" / "0",
            [("frame_0000.jpg", (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 5.0))],
        )
    return root


def test_scan_project_dir(tmp_path):
    root = _build_project(tmp_path)
    project = scan_project_dir(root)
    assert project.cloud is not None and project.cloud.name == "cloud.ply"
    assert project.trajectory is not None
    assert project.images_dir is not None and project.image_count == 6
    assert "undistorted" in str(project.images_dir)
    assert project.colmap_model is not None
    assert len(project.bags) == 1
    lines = "\n".join(project.describe())
    assert "Punktwolke" in lines and "Trajektorie" in lines and ".bag" in lines


# ---------------------------------------------- colorized cloud selection

def _write_ply_cloud(path, n_points, colored):
    from scantobim.core.cloud import PointCloud

    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 5, (n_points, 3))
    colors = (
        rng.integers(40, 220, (n_points, 3), dtype=np.uint8) if colored else None
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_point_cloud(PointCloud(points=pts, colors=colors), path)


def test_cloud_has_colors_ply_and_las(tmp_path):
    from scantobim.io.project import cloud_has_colors

    _write_ply_cloud(tmp_path / "rgb.ply", 100, colored=True)
    _write_ply_cloud(tmp_path / "plain.ply", 100, colored=False)
    assert cloud_has_colors(tmp_path / "rgb.ply") is True
    assert cloud_has_colors(tmp_path / "plain.ply") is False

    import laspy

    for fmt, expected in ((3, True), (0, False)):
        header = laspy.LasHeader(point_format=fmt, version="1.2")
        las = laspy.LasData(header)
        las.x, las.y, las.z = [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]
        las.write(str(tmp_path / f"pf{fmt}.las"))
        assert cloud_has_colors(tmp_path / f"pf{fmt}.las") is expected

    # Unknown / unreadable → None, never an exception.
    (tmp_path / "kaputt.ply").write_bytes(b"not a ply")
    assert cloud_has_colors(tmp_path / "kaputt.ply") in (False, None)
    (tmp_path / "plain.xyz").write_text("0 0 0\n")
    assert cloud_has_colors(tmp_path / "plain.xyz") is None


def test_cloud_has_colors_e57(tmp_path):
    pye57 = pytest.importorskip("pye57")
    pts = np.random.default_rng(1).uniform(0, 3, (200, 3))
    base = {
        "cartesianX": pts[:, 0],
        "cartesianY": pts[:, 1],
        "cartesianZ": pts[:, 2],
    }
    e57 = pye57.E57(str(tmp_path / "plain.e57"), mode="w")
    e57.write_scan_raw(dict(base))
    e57.close()
    rgb = np.random.default_rng(2).integers(0, 255, (200, 3)).astype(np.uint8)
    e57 = pye57.E57(str(tmp_path / "rgb.e57"), mode="w")
    e57.write_scan_raw(
        dict(base, colorRed=rgb[:, 0], colorGreen=rgb[:, 1], colorBlue=rgb[:, 2])
    )
    e57.close()

    from scantobim.io.project import cloud_has_colors

    assert cloud_has_colors(tmp_path / "rgb.e57") is True
    assert cloud_has_colors(tmp_path / "plain.e57") is False


def test_scan_project_dir_prefers_colorized_by_name(tmp_path):
    """Colorized + LARGER uncolorized: geometry wins, colors transfer."""
    root = tmp_path / "export"
    _write_ply_cloud(root / "haus_uncolorized_segmented.ply", 4000, colored=False)
    _write_ply_cloud(root / "haus_colorized_segmented.ply", 1500, colored=True)
    project = scan_project_dir(root)
    assert project.cloud.name == "haus_uncolorized_segmented.ply"
    assert project.color_source is not None
    assert project.color_source.name == "haus_colorized_segmented.ply"
    assert "Farbquelle" in "\n".join(project.describe())


def test_scan_project_dir_prefers_colorized_by_probe(tmp_path):
    """Colored wins only when it is practically the same size (≥95%)."""
    root = tmp_path / "export"
    _write_ply_cloud(root / "scan_a.ply", 1500, colored=False)
    _write_ply_cloud(root / "scan_b.ply", 1480, colored=True)  # ~99% as dense
    project = scan_project_dir(root)
    assert project.cloud.name == "scan_b.ply"
    assert project.cloud_colored is True
    assert project.color_source is None  # chosen cloud carries the colors
    assert "mit Farben" in "\n".join(project.describe())


def test_scan_project_dir_uncolored_single(tmp_path):
    root = tmp_path / "export"
    _write_ply_cloud(root / "scan.ply", 800, colored=False)
    project = scan_project_dir(root)
    assert project.cloud.name == "scan.ply"
    assert project.cloud_colored is False
    assert "ohne Farben" in "\n".join(project.describe())


def test_cloud_header_info_counts(tmp_path):
    from scantobim.io.project import cloud_header_info

    _write_ply_cloud(tmp_path / "rgb.ply", 123, colored=True)
    assert cloud_header_info(tmp_path / "rgb.ply") == (123, True)

    import laspy

    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x, las.y, las.z = [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]
    las.write(str(tmp_path / "c.las"))
    assert cloud_header_info(tmp_path / "c.las") == (3, True)


def test_scan_project_dir_dense_beats_thin_colored_preview(tmp_path):
    """A ~100×-thinner colored quicklook must not displace the real scan —
    instead it becomes the color source for a nearest-neighbour transfer."""
    root = tmp_path / "export"
    _write_ply_cloud(root / "scan_full.ply", 30000, colored=False)
    _write_ply_cloud(root / "scan_preview.ply", 250, colored=True)
    project = scan_project_dir(root)
    assert project.cloud.name == "scan_full.ply"
    assert project.cloud_colored is False
    assert project.cloud_points == 30000
    assert project.color_source is not None
    assert project.color_source.name == "scan_preview.ply"
    text = "\n".join(project.describe())
    assert "Farbquelle" in text and "scan_preview.ply" in text


def test_transfer_colors():
    from scantobim.core.cloud import PointCloud
    from scantobim.core.preprocess import transfer_colors

    rng = np.random.default_rng(0)
    src_pts = rng.uniform(0, 10, (2000, 3))
    src_colors = rng.integers(0, 255, (2000, 3)).astype(np.uint8)
    source = PointCloud(points=src_pts, colors=src_colors)
    dst_pts = np.vstack(
        [src_pts + rng.normal(0, 0.001, src_pts.shape), [[1e3, 1e3, 1e3]]]
    )
    cloud = PointCloud(points=dst_pts)
    fraction = transfer_colors(cloud, source)
    assert cloud.colors is not None and fraction > 0.99
    assert np.array_equal(cloud.colors[:-1], src_colors)  # nearest = own point
    assert np.array_equal(cloud.colors[-1], [128, 128, 128])  # out of reach


def test_project_cli_color_transfer(tmp_path, capsys):
    """Dense uncolorized + sparse colorized sibling → colored dense model."""
    from scantobim.cli import main
    from scantobim.core.cloud import PointCloud

    root = tmp_path / "export"
    root.mkdir()
    cloud = make_box_scan(density=500, noise=0.004)
    write_point_cloud(PointCloud(points=cloud.points), root / "voll.ply")
    rng = np.random.default_rng(5)
    sub = cloud.points[::6]
    write_point_cloud(
        PointCloud(
            points=sub,
            colors=rng.integers(60, 200, (len(sub), 3), dtype=np.uint8),
        ),
        root / "vorschau.ply",
    )
    out = tmp_path / "modell.html"
    assert main(["project", str(root), "-o", str(out), "--seed", "1"]) == 0
    log = capsys.readouterr().out
    assert "Farben übertragen von: vorschau.ply" in log
    assert "der Punkte eingefärbt" in log
    report = json.loads((tmp_path / "modell_bericht.json").read_text())
    assert report["texture"]["source"].startswith("punktfarben")


def test_project_cli_end_to_end(tmp_path, capsys):
    """`scantobim project <dir>`: detect, reconstruct, texture, report."""
    from scantobim.cli import main

    root = _build_project(tmp_path, with_photos=False)
    out = tmp_path / "modell.html"
    code = main(["project", str(root), "-o", str(out), "--seed", "1"])
    assert code == 0
    assert out.stat().st_size > 5000
    report = json.loads((tmp_path / "modell_bericht.json").read_text())
    assert report["planes"] == 6
    assert report["trajectory_positions"] == 50
    log = capsys.readouterr().out
    assert "SLAM-Projekt" in log and "Trajektorie" in log


def test_project_cli_missing_cloud(tmp_path, capsys):
    from scantobim.cli import main

    empty = tmp_path / "leer"
    empty.mkdir()
    assert main(["project", str(empty)]) == 1
    assert "keine Punktwolke" in capsys.readouterr().err


def test_project_cli_with_photo_projection(tmp_path, capsys):
    """Full S20-style project: cloud + COLMAP poses + photo → projected texture."""
    PIL = pytest.importorskip("PIL.Image")
    from scantobim.cli import main
    from tests.synthetic import sample_rect
    from scantobim.core.cloud import PointCloud

    root = tmp_path / "s20_photo"
    root.mkdir()
    rng = np.random.default_rng(0)
    x, y, z = np.eye(3)
    floor = sample_rect((0, 0, 0), x, y, 4.0, 4.0, 2500, 0.002, rng)
    wall = sample_rect((0, 0, 0), x, z, 4.0, 1.0, 2500, 0.002, rng)
    write_point_cloud(PointCloud(points=np.vstack([floor, wall])), root / "cloud.ply")

    # One overhead camera, checkerboard photo (COLMAP text model).
    w = h = 400
    f = 300.0
    rotation = np.diag([1.0, -1.0, -1.0])
    translation = -rotation @ np.array([2.0, 2.0, 5.0])
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    board = (((xx // 40) + (yy // 40)) % 2).astype(np.uint8)
    img = np.stack([board * 255, board * 255, np.full_like(board, 60)], axis=-1)
    (root / "photos").mkdir()
    PIL.fromarray(img).save(root / "photos" / "cam0.png")
    model = root / "model_txt"
    model.mkdir()
    (model / "cameras.txt").write_text(f"1 PINHOLE {w} {h} {f} {f} {w/2} {h/2}\n")
    (model / "images.txt").write_text(
        f"1 1 0 0 0 {translation[0]} {translation[1]} {translation[2]} 1 cam0.png\n\n"
    )

    out = root / "modell.html"
    code = main(["project", str(root), "-o", str(out), "--texel", "0.02", "--seed", "1"])
    assert code == 0
    log = capsys.readouterr().out
    assert "projecting 1 photos" in log and "texture atlas" in log
    # Atlas texture embedded in the viewer (JPEG for big atlases, PNG small).
    assert b"data:image/" in out.read_bytes()


def test_project_photo_fallback_wrong_frame(tmp_path, capsys):
    """Poses in a different coordinate frame → automatic cloud-color fallback."""
    from scantobim.cli import main

    root = _build_project(tmp_path, with_photos=False)
    img_dir = root / "camera" / "undistorted"
    img_dir.mkdir(parents=True)
    for i in range(6):
        (img_dir / f"frame_{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xdb dummy")
    # Camera centers ~1.4 km away from the scanned box: wrong frame.
    _write_binary_model(
        root / "camera" / "sparse" / "0",
        [("frame_0000.jpg", (1.0, 0.0, 0.0, 0.0), (-1000.0, -1000.0, -50.0))],
    )
    out = tmp_path / "modell.html"
    assert main(["project", str(root), "-o", str(out), "--seed", "1"]) == 0
    log = capsys.readouterr().out
    assert "anderes Koordinatensystem" in log
    assert "Textur: punktfarben" in log
    report = json.loads((tmp_path / "modell_bericht.json").read_text())
    assert report["texture"]["source"] == "punktfarben"
    assert b"data:image/" in out.read_bytes()  # textured after fallback


def test_project_photo_fallback_unreadable_images(tmp_path, capsys):
    """Photos that never project/open must not leave a grey model behind."""
    from scantobim.cli import main

    root = _build_project(tmp_path, with_photos=True)  # dummy jpgs, cam at z=5
    out = tmp_path / "modell.html"
    assert main(["project", str(root), "-o", str(out), "--seed", "1"]) == 0
    log = capsys.readouterr().out
    assert "Textur: punktfarben" in log  # fell back instead of grey texture


def test_stereo_image_dirs_use_common_parent(tmp_path):
    """S20 stereo rig: left/ and right/ image folders → parent is the root."""
    root = tmp_path / "s20_stereo"
    for side in ("left", "right"):
        d = root / "camera" / "undistorted" / side
        d.mkdir(parents=True)
        for i in range(4):
            (d / f"frame_{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xdb x")
    write_point_cloud(make_box_scan(density=300), root / "cloud.ply")
    project = scan_project_dir(root)
    assert project.images_dir == root / "camera" / "undistorted"
    assert project.image_count == 8  # both cameras counted


def test_image_index_resolves_stereo_names(tmp_path):
    """COLMAP names like 'left/frame.jpg' resolve from any chosen folder."""
    from scantobim.core.texture import _index_images

    base = tmp_path / "undistorted"
    for side in ("left", "right"):
        (base / side).mkdir(parents=True)
        (base / side / "frame_0001.jpg").write_bytes(b"x")

    # Chosen dir = parent → subdir-prefixed names resolve directly.
    idx = _index_images(base)
    assert idx["left/frame_0001.jpg"] == base / "left" / "frame_0001.jpg"
    assert idx["right/frame_0001.jpg"] == base / "right" / "frame_0001.jpg"

    # Chosen dir = left only → sibling right/ is still found via the index.
    idx = _index_images(base / "left")
    assert idx["right/frame_0001.jpg"] == base / "right" / "frame_0001.jpg"
    assert "frame_0001.jpg" in idx  # basename fallback


def test_project_photo_projection_stereo(tmp_path, capsys):
    """Stereo project: both cameras project, halves of the floor get each photo."""
    PIL = pytest.importorskip("PIL.Image")
    from scantobim.cli import main
    from scantobim.core.cloud import PointCloud
    from tests.synthetic import sample_rect

    root = tmp_path / "s20_stereo_proj"
    root.mkdir()
    rng = np.random.default_rng(0)
    x, y, z = np.eye(3)
    floor = sample_rect((0, 0, 0), x, y, 4.0, 4.0, 2500, 0.002, rng)
    wall = sample_rect((0, 0, 0), x, z, 4.0, 1.0, 2500, 0.002, rng)
    write_point_cloud(PointCloud(points=np.vstack([floor, wall])), root / "cloud.ply")

    # Two cameras (a stereo pair side by side, both looking down), photos in
    # left/ and right/ subfolders, COLMAP names carry the subfolder prefix.
    w = h = 400
    f = 300.0
    rotation = np.diag([1.0, -1.0, -1.0])
    for side, cx, color in (("left", 1.0, (255, 0, 0)), ("right", 3.0, (0, 0, 255))):
        d = root / "photos" / side
        d.mkdir(parents=True)
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = color
        PIL.fromarray(img).save(d / "cam.png")
    model = root / "model_txt"
    model.mkdir()
    (model / "cameras.txt").write_text(f"1 PINHOLE {w} {h} {f} {f} {w/2} {h/2}\n")
    lines = []
    for i, (side, cx) in enumerate([("left", 1.0), ("right", 3.0)], start=1):
        t = -rotation @ np.array([cx, 2.0, 5.0])
        lines.append(f"{i} 1 0 0 0 {t[0]} {t[1]} {t[2]} 1 {side}/cam.png\n\n")
    (model / "images.txt").write_text("".join(lines))

    out = root / "modell.html"
    code = main(["project", str(root), "-o", str(out), "--texel", "0.05", "--seed", "1"])
    assert code == 0
    log = capsys.readouterr().out
    assert "Fotos gefunden: 2 von 2" in log
    assert "Textur: foto-projektion" in log


def test_scan_project_dir_finds_deep_colmap(tmp_path):
    """S20 layout: sparse/0 lives at depth 5 — must still be found."""
    root = tmp_path / "projekt"
    deep = root / "steuerhaus" / "output" / "colmap" / "sparse" / "0"
    _write_binary_model(deep, [("frame_0000.jpg", (1.0, 0, 0, 0), (0, 0, 5.0))])
    img_dir = root / "steuerhaus" / "output" / "undistort"
    img_dir.mkdir(parents=True)
    (img_dir / "frame_0000.jpg").write_bytes(b"\xff\xd8\xff\xdb x")
    _write_ply_cloud(root / "wolke.ply", 800, colored=True)
    project = scan_project_dir(root)
    assert project.colmap_model is not None
    assert project.colmap_model.name == "0"
    assert "Kameraposen:" in "\n".join(project.describe())


def test_describe_warns_when_poses_missing(tmp_path):
    root = tmp_path / "projekt"
    img_dir = root / "fotos"
    img_dir.mkdir(parents=True)
    (img_dir / "a.jpg").write_bytes(b"\xff\xd8\xff\xdb x")
    _write_ply_cloud(root / "wolke.ply", 800, colored=True)
    project = scan_project_dir(root)
    text = "\n".join(project.describe())
    assert "NICHT gefunden" in text and "Foto-Projektion" in text


def test_photo_colors_for_mesh(tmp_path):
    """Vertices seen by a camera get the photo's pixel colors."""
    PIL = pytest.importorskip("PIL.Image")
    from scantobim.core.mesh import Mesh
    from scantobim.core.texture import photo_colors_for_mesh

    # Floor patch around the origin; camera at (0,0,-5) looking +z (COLMAP
    # identity pose with t=(0,0,5)) sees it from below.
    g = np.linspace(-1, 1, 30)
    xx, yy = np.meshgrid(g, g)
    verts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    idx = lambda i, j: i * 30 + j
    faces = []
    for i in range(29):
        for j in range(29):
            faces.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
            faces.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])
    mesh = Mesh(vertices=verts, faces=np.array(faces))

    model = tmp_path / "sparse"
    _write_binary_model(model, [("foto.jpg", (1.0, 0, 0, 0), (0.0, 0.0, 5.0))])
    img_dir = tmp_path / "bilder"
    img_dir.mkdir()
    PIL.new("RGB", (400, 300), (200, 40, 90)).save(img_dir / "foto.jpg")

    stats = {}
    frac = photo_colors_for_mesh(mesh, model, img_dir, stats_out=stats)
    assert frac > 0.9
    assert stats["cameras_used"] == 1
    med = np.median(mesh.vertex_colors, axis=0)
    assert np.all(np.abs(med - [200, 40, 90]) < 12)  # JPEG-kompression


def test_read_xyzopk_variants(tmp_path):
    from scantobim.photogrammetry.xyzopk import read_xyzopk

    # Header, name first, comma-separated, name last.
    p = tmp_path / "xyzopk.txt"
    p.write_text(
        "# name x y z omega phi kappa\n"
        "foto_001.jpg 1.0 2.0 3.0 180.0 0.0 10.0\n"
        "foto_002.jpg, 4.0, 5.0, 6.0, -90.0, 5.0, 0.0\n"
        "7 8 9 10 20 30 foto_003.jpg\n"
    )
    entries = read_xyzopk(p)
    assert [e[0] for e in entries] == ["foto_001.jpg", "foto_002.jpg", "foto_003.jpg"]
    assert np.allclose(entries[0][1], [1, 2, 3])
    assert np.allclose(entries[0][2], [180, 0, 10])
    assert np.allclose(entries[2][1], [7, 8, 9])

    # All-radian angles are converted to degrees.
    q = tmp_path / "rad.txt"
    q.write_text("a.jpg 0 0 0 3.14159265 0 0.5\n")
    (name, xyz, opk) = read_xyzopk(q)[0]
    assert abs(opk[0] - 180.0) < 0.01


def test_scan_project_dir_detects_xyzopk(tmp_path):
    root = tmp_path / "projekt"
    img_dir = root / "output" / "undistort"
    img_dir.mkdir(parents=True)
    (img_dir / "a.jpg").write_bytes(b"\xff\xd8\xff\xdb x")
    (img_dir / "xyzopk.txt").write_text("a.jpg 0 0 -5 180 0 0\n")
    _write_ply_cloud(root / "wolke.ply", 800, colored=True)
    project = scan_project_dir(root)
    assert project.xyzopk is not None and project.xyzopk.name == "xyzopk.txt"
    text = "\n".join(project.describe())
    assert "xyzopk" in text and "selbstkalibriert" in text
    assert "NICHT gefunden" not in text


def test_cameras_from_xyzopk_selfcalibration(tmp_path):
    """Convention + focal length recovered from the colorized cloud."""
    PIL = pytest.importorskip("PIL.Image")
    from scantobim.core.cloud import PointCloud
    from scantobim.core.mesh import Mesh
    from scantobim.core.texture import photo_colors_for_mesh
    from scantobim.photogrammetry.xyzopk import cameras_from_xyzopk

    fx_true, W, H = 350.0, 400, 300

    # Smooth color field on the floor plane z=0.
    def field(x, y):
        r = 50 + 100 * (x + 1) / 2
        b = 120 + 100 * (y + 1) / 2
        return np.stack([r, np.full_like(r, 80.0), b], axis=-1)

    # Photo from camera at (0,0,-5), looking +z (COLMAP identity):
    # pixel (px,py) sees world (x,y) with x=(px-cx)*5/fx, y=(py-cy)*5/fx.
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    photo = field((px - W / 2) * 5.0 / fx_true, (py - H / 2) * 5.0 / fx_true)
    img_dir = tmp_path / "bilder"
    img_dir.mkdir()
    PIL.fromarray(photo.astype(np.uint8)).save(img_dir / "foto.png")

    rng = np.random.default_rng(0)
    pts = np.column_stack([
        rng.uniform(-1, 1, 20000), rng.uniform(-0.7, 0.7, 20000),
        np.zeros(20000),
    ])
    cloud = PointCloud(
        points=pts, colors=field(pts[:, 0], pts[:, 1]).astype(np.uint8)
    )

    # Convention A ("opk/cam2world"): R_colmap = I needs omega=180.
    opk = tmp_path / "xyzopk.txt"
    opk.write_text("foto.png 0.0 0.0 -5.0 180.0 0.0 0.0\n")

    stats = {}
    cams = cameras_from_xyzopk(opk, img_dir, cloud, stats_out=stats)
    assert len(cams) == 1
    assert 290 < stats["fx"] < 440  # true 350, recovered on the sweep grid
    assert stats["score"] > 0.85
    # The calibrated camera must reproduce the field on a floor mesh.
    g = np.linspace(-0.8, 0.8, 20)
    xx, yy = np.meshgrid(g, g)
    verts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    faces = []
    for i in range(19):
        for j in range(19):
            a = i * 20 + j
            faces.append([a, a + 20, a + 21])
            faces.append([a, a + 21, a + 1])
    mesh = Mesh(vertices=verts, faces=np.array(faces))
    frac = photo_colors_for_mesh(mesh, cams, img_dir)
    assert frac > 0.9
    expected = field(verts[:, 0], verts[:, 1])
    err = np.abs(mesh.vertex_colors.astype(float) - expected).mean()
    assert err < 25  # fx grid quantization + rounding


def test_unused_report_lists_files_with_reasons(tmp_path):
    """Every file the import does not use is listed with a reason."""
    root = tmp_path / "projekt"
    (root / "output" / "undistort").mkdir(parents=True)
    _write_ply_cloud(root / "wolke_colorized.ply", 3000, colored=True)
    _write_ply_cloud(root / "wolke_uncolorized.ply", 4000, colored=False)
    for i in range(3):
        (root / "output" / "undistort" / f"f{i}.jpg").write_bytes(b"\xff\xd8\xff\xdb x")
    (root / "output" / "undistort" / "xyzopk.txt").write_text("f0.jpg 0 0 -5 180 0 0\n")
    (root / "preview" / "thumbs").mkdir(parents=True)
    (root / "preview" / "thumbs" / "t1.jpg").write_bytes(b"\xff\xd8\xff\xdb x")
    (root / "trajectory.txt").write_text("0 0 1.7\n1 0 1.7\n")
    (root / "aufnahme.bag").write_bytes(b"B" * 5000)
    (root / "scanner.log").write_text("log")
    (root / "meta.json").write_text("{}")
    (root / "kalib.xml").write_text("<x/>")

    project = scan_project_dir(root)
    # Densest cloud is the geometry source, the colored one the color donor.
    assert project.cloud.name == "wolke_uncolorized.ply"
    assert project.color_source is not None
    report = "\n".join(project.unused_report())
    assert "aufnahme.bag" in report and "Rohaufnahme" in report
    assert "scanner.log" in report or "*.log" in report
    assert "Vorschau" in report or "Foto-Ordners" in report  # thumbs/t1.jpg
    assert ".xml" in report or "kalib.xml" in report
    # Used files must NOT be listed as unused (the chosen cloud's name may
    # appear inside a REASON text, so check line starts):
    lines = project.unused_report()
    assert not any(l.startswith("wolke_colorized.ply") for l in lines)
    assert not any(l.startswith("wolke_uncolorized.ply") for l in lines)
    assert "xyzopk.txt" not in report
    assert "trajectory.txt" not in report
    assert "f0.jpg" not in report and "*.jpg (3" not in report


def test_project_cli_prints_inventory(tmp_path, capsys):
    from scantobim.cli import main

    root = _build_project(tmp_path, with_photos=False)
    out = tmp_path / "modell.html"
    assert main(["project", str(root), "-o", str(out), "--seed", "1"]) == 0
    log = capsys.readouterr().out
    assert "Datei-Inventar" in log
    assert "raw.bag" in log and "Rohaufnahme" in log


# ------------------------------------------------- camera calibration file

def test_read_camera_calibration_styles(tmp_path):
    from scantobim.photogrammetry.calibration import read_camera_calibration

    flat = tmp_path / "calib_flat.yaml"
    flat.write_text(
        "camera:\n  fx: 1234.5\n  fy: 1236.1\n  cx: 2027.3\n  cy: 1519.8\n"
        "  k1: -0.041\nimage_width: 4056\nimage_height: 3040\n"
    )
    cams = read_camera_calibration(flat)
    assert cams is not None
    d = cams[0]
    assert d["fx"] == 1234.5 and d["fy"] == 1236.1
    assert d["cx"] == 2027.3 and d["k1"] == -0.041
    assert d["width"] == 4056 and d["height"] == 3040

    ocv = tmp_path / "calibration.yaml"
    ocv.write_text(
        "camera_matrix:\n  rows: 3\n  cols: 3\n"
        "  data: [1234.5, 0., 2027.3, 0., 1236.1, 1519.8, 0., 0., 1.]\n"
        "distortion_coefficients:\n  rows: 1\n  cols: 5\n"
        "  data: [-0.041, 0.012, 0., 0., 0.]\n"
    )
    cams = read_camera_calibration(ocv)
    assert cams is not None
    d = cams[0]
    assert d["fx"] == 1234.5 and d["cy"] == 1519.8 and d["k1"] == -0.041

    junk = tmp_path / "leer.yaml"
    junk.write_text("sensor: lidar\nversion: 3\n")
    assert read_camera_calibration(junk) is None


def test_cameras_from_xyzopk_with_calibration(tmp_path):
    """Factory intrinsics narrow the sweep and land closer to the truth."""
    PIL = pytest.importorskip("PIL.Image")
    from scantobim.core.cloud import PointCloud
    from scantobim.photogrammetry.xyzopk import cameras_from_xyzopk

    fx_true, W, H = 350.0, 400, 300

    def field(x, y):
        r = 50 + 100 * (x + 1) / 2
        b = 120 + 100 * (y + 1) / 2
        return np.stack([r, np.full_like(r, 80.0), b], axis=-1)

    px, py = np.meshgrid(np.arange(W), np.arange(H))
    photo = field((px - W / 2) * 5.0 / fx_true, (py - H / 2) * 5.0 / fx_true)
    img_dir = tmp_path / "bilder"
    img_dir.mkdir()
    PIL.fromarray(photo.astype(np.uint8)).save(img_dir / "foto.png")

    rng = np.random.default_rng(0)
    pts = np.column_stack([
        rng.uniform(-1, 1, 20000), rng.uniform(-0.7, 0.7, 20000),
        np.zeros(20000),
    ])
    cloud = PointCloud(
        points=pts, colors=field(pts[:, 0], pts[:, 1]).astype(np.uint8)
    )
    opk = tmp_path / "xyzopk.txt"
    opk.write_text("foto.png 0.0 0.0 -5.0 180.0 0.0 0.0\n")

    calib = {"fx": 350.0, "fy": 350.0, "cx": W / 2.0, "cy": H / 2.0,
             "k1": None, "width": W, "height": H}
    stats = {}
    cams = cameras_from_xyzopk(
        opk, img_dir, cloud, stats_out=stats, calibration=calib
    )
    assert stats["intrinsics_quelle"] == "calibration.yaml"
    assert 340 < stats["fx"] < 372  # narrow band around the factory value
    assert stats["score"] > 0.9
    assert cams[0].cx == W / 2.0


def test_scan_project_dir_detects_calibration(tmp_path):
    root = _build_project(tmp_path, with_photos=True)
    info = root / "info"
    info.mkdir()
    (info / "calibration.yaml").write_text(
        "camera_matrix:\n  data: [1234.5, 0., 200.0, 0., 1236.1, 150.0, 0., 0., 1.]\n"
    )
    project = scan_project_dir(root)
    assert project.calibration is not None
    assert project.calibration_data[0]["fx"] == 1234.5
    lines = "\n".join(project.describe())
    assert "Kalibrierung" in lines and "1234" in lines
    used, reason = project._classify(info / "calibration.yaml")
    assert used and "Kalibrierung" in reason


# ------------------------------------- best-model sources (Auftrag Stufe 1)

def test_trajectory_s20_time_last_layout(tmp_path):
    """x y z roll pitch yaw qx qy qz qw TIME → positions + time column."""
    p = tmp_path / "trajectory.txt"
    rows = []
    for i in range(20):
        t = 1741600000.0 + i * 0.1
        rows.append(f"{1 + i * 0.5} {2.0} {1.5} 0 0 0 0 0 0 1 {t}")
    p.write_text("\n".join(rows))
    traj = read_trajectory(p)
    assert traj.shape == (20, 4)
    assert np.allclose(traj[0, :3], [1, 2, 1.5])
    assert traj[0, 3] == 1741600000.0
    assert np.all(np.diff(traj[:, 3]) > 0)


def test_time_based_normal_orientation():
    """Each point faces the scanner position at its capture time."""
    from scantobim.core.cloud import PointCloud
    from scantobim.core.preprocess import orient_normals_along_trajectory

    # Scanner moves along +x above the floor; floor points sampled at the
    # matching times. Sensor is ALWAYS overhead at that moment.
    n = 200
    t = np.linspace(0.0, 10.0, n)
    pts = np.column_stack([t * 2.0, np.zeros(n), np.zeros(n)])
    cloud = PointCloud(points=pts, times=1741600000.0 + t)
    cloud.normals = np.tile([0.0, 0.0, -1.0], (n, 1))  # all pointing DOWN

    traj = np.column_stack([
        t * 2.0, np.zeros(n), np.full(n, 3.0), 1741600000.0 + t
    ])
    stats = {}
    orient_normals_along_trajectory(cloud, traj, stats_out=stats)
    assert stats["mode"] == "zeitbasiert"
    assert np.all(cloud.normals[:, 2] > 0)  # flipped up toward the sensor


def test_time_orientation_clock_offset_alignment():
    """Constant clock offset between LAS gps_time and trajectory is bridged."""
    from scantobim.core.cloud import PointCloud
    from scantobim.core.preprocess import orient_normals_along_trajectory

    n = 100
    t = np.linspace(0.0, 10.0, n)
    pts = np.column_stack([t, np.zeros(n), np.zeros(n)])
    cloud = PointCloud(points=pts, times=5000.0 + t)  # different clock
    cloud.normals = np.tile([0.0, 0.0, -1.0], (n, 1))
    traj = np.column_stack([t, np.zeros(n), np.full(n, 2.0), 999000.0 + t])
    stats = {}
    orient_normals_along_trajectory(cloud, traj, stats_out=stats)
    assert stats["mode"] == "zeitbasiert"
    assert np.all(cloud.normals[:, 2] > 0)


def test_densest_cloud_wins_with_color_transfer(tmp_path):
    """20M-uncolorized beats 18M-colorized: geometry first, colors follow."""
    root = tmp_path / "proj"
    (root / "output").mkdir(parents=True)
    _write_ply_cloud(root / "output" / "steuerhaus_uncolorized.ply", 20000, False)
    _write_ply_cloud(root / "output" / "steuerhaus_colorized.ply", 18000, True)
    project = scan_project_dir(root)
    assert project.cloud.name == "steuerhaus_uncolorized.ply"
    assert project.color_source is not None
    assert project.color_source.name == "steuerhaus_colorized.ply"
    assert "Geometriequelle" in (project.cloud_note or "")


def test_read_imgpose_layouts(tmp_path):
    from scantobim.photogrammetry.xyzopk import read_imgpose

    p = tmp_path / "ImgPose.txt"
    p.write_text(
        "# name x y z qx qy qz qw time\n"
        "foto_001.jpg 10.0 20.0 3.0 0.0 0.0 0.0 1.0 1741600000.5\n"
        "1741600001.5 11.0 21.0 3.1 0.0 0.0 0.0 1.0 foto_002.jpg\n"
    )
    entries = read_imgpose(p)
    assert len(entries) == 2
    name, xyz, quat, t = entries[0]
    assert name == "foto_001.jpg"
    assert np.allclose(xyz, [10, 20, 3])
    assert np.allclose(quat, [0, 0, 0, 1])
    assert t == 1741600000.5
    name2, xyz2, _q2, t2 = entries[1]
    assert name2 == "foto_002.jpg"
    assert np.allclose(xyz2, [11, 21, 3.1])
    assert t2 == 1741600001.5


def test_cameras_from_imgpose_quaternion(tmp_path):
    """Quaternion poses recover the projection like the xyzopk path."""
    PIL = pytest.importorskip("PIL.Image")
    from scantobim.core.cloud import PointCloud
    from scantobim.photogrammetry.xyzopk import cameras_from_imgpose

    fx_true, W, H = 350.0, 400, 300

    def field(x, y):
        r = 50 + 100 * (x + 1) / 2
        b = 120 + 100 * (y + 1) / 2
        return np.stack([r, np.full_like(r, 80.0), b], axis=-1)

    px, py = np.meshgrid(np.arange(W), np.arange(H))
    photo = field((px - W / 2) * 5.0 / fx_true, (py - H / 2) * 5.0 / fx_true)
    img_dir = tmp_path / "bilder"
    img_dir.mkdir()
    PIL.fromarray(photo.astype(np.uint8)).save(img_dir / "foto.png")

    rng = np.random.default_rng(0)
    pts = np.column_stack([
        rng.uniform(-1, 1, 20000), rng.uniform(-0.7, 0.7, 20000),
        np.zeros(20000),
    ])
    cloud = PointCloud(
        points=pts, colors=field(pts[:, 0], pts[:, 1]).astype(np.uint8)
    )
    # Camera at (0,0,-5) looking +z: R_colmap = I → identity quaternion.
    ip = tmp_path / "ImgPose.txt"
    ip.write_text("foto.png 0.0 0.0 -5.0 0.0 0.0 0.0 1.0 1741600000.0\n")

    stats = {}
    cams = cameras_from_imgpose(ip, img_dir, cloud, stats_out=stats)
    assert len(cams) == 1
    assert stats["score"] > 0.85
    assert 290 < stats["fx"] < 440
    assert stats["convention"].startswith("quaternion")


def test_scan_project_dir_detects_imgpose(tmp_path):
    root = _build_project(tmp_path, with_photos=True)
    (root / "camera" / "ImgPose.txt").write_text(
        "frame_0000.jpg 0 0 0 0 0 0 1 1.0\n"
    )
    project = scan_project_dir(root)
    assert project.imgpose is not None
    used, reason = project._classify(root / "camera" / "ImgPose.txt")
    assert used and "Quaternionen" in reason


# --------------------------------------- v3.11: verified-data corrections

def test_grey_rgb_cloud_is_not_colored(tmp_path):
    """R=G=B intensity ramps (uncolorized.las) must not count as colored."""
    import laspy

    from scantobim.io.project import _colors_grey_probe

    def _write_las(path, grey):
        header = laspy.LasHeader(point_format=3, version="1.2")
        las = laspy.LasData(header)
        n = 500
        rng = np.random.default_rng(1)
        las.x = rng.uniform(0, 5, n)
        las.y = rng.uniform(0, 5, n)
        las.z = rng.uniform(0, 3, n)
        if grey:
            v = rng.integers(0, 65535, n)
            las.red = v
            las.green = v
            las.blue = v
        else:
            las.red = rng.integers(0, 65535, n)
            las.green = rng.integers(0, 65535, n)
            las.blue = rng.integers(0, 65535, n)
        las.write(str(path))

    _write_las(tmp_path / "grau.las", grey=True)
    _write_las(tmp_path / "bunt.las", grey=False)
    assert _colors_grey_probe(tmp_path / "grau.las") is True
    assert _colors_grey_probe(tmp_path / "bunt.las") is False

    # Selection: grey-RGB densest cloud + truly colored sibling → densest
    # wins as geometry, colored sibling becomes the color source.
    root = tmp_path / "proj"
    root.mkdir()
    _write_las(root / "scan_uncolorized.las", grey=True)
    _write_ply_cloud(root / "scan_col.ply", 400, colored=True)
    project = scan_project_dir(root)
    assert project.cloud.name == "scan_uncolorized.las"
    assert project.cloud_colored is False  # grey ≠ colored
    assert project.color_source is not None


def test_colors_are_grey_helper():
    from scantobim.cli import _colors_are_grey
    from scantobim.core.cloud import PointCloud

    n = 300
    rng = np.random.default_rng(0)
    v = rng.integers(0, 255, n).astype(np.uint8)
    grey = PointCloud(
        points=rng.uniform(0, 1, (n, 3)),
        colors=np.column_stack([v, v, v]),
    )
    assert _colors_are_grey(grey) is True
    colored = PointCloud(
        points=rng.uniform(0, 1, (n, 3)),
        colors=rng.integers(0, 255, (n, 3)).astype(np.uint8),
    )
    assert _colors_are_grey(colored) is False


def test_pick_calibration_matches_photo_size(tmp_path):
    """The 640×480 navigation camera must never calibrate the photo rig."""
    from scantobim.photogrammetry.calibration import (
        pick_calibration,
        read_camera_calibration,
    )

    yaml = tmp_path / "calibration.yaml"
    yaml.write_text(
        "fisheye_middle:\n"
        "  image_width: 640\n"
        "  image_height: 480\n"
        "  fx: 548.0\n  fy: 548.0\n  cx: 320.0\n  cy: 240.0\n"
        "fisheye_left:\n"
        "  camera_model: POLYFISHEYE\n"
        "  image_width: 3504\n"
        "  image_height: 4672\n"
        "  A11: 1480.2\n  A22: 1481.0\n  u0: 1752.3\n  v0: 2336.1\n"
        "  k2: -0.003\n  k3: 0.0006\n  k4: 0.0\n"
    )
    cams = read_camera_calibration(yaml)
    assert cams is not None and len(cams) >= 2
    names = [c.get("name") for c in cams]
    assert "fisheye_middle" in names and "fisheye_left" in names

    # Photo size 3504×4672 → the POLYFISHEYE camera, A11 as fx.
    cam = pick_calibration(cams, 3504, 4672)
    assert cam is not None
    assert cam["name"] == "fisheye_left"
    assert abs(cam["fx"] - 1480.2) < 0.01
    assert cam["cx"] == 1752.3
    assert cam["model"] == "POLYFISHEYE"
    assert cam["poly"] == [-0.003, 0.0006, 0.0]

    # Landscape/portrait tolerant.
    assert pick_calibration(cams, 4672, 3504)["name"] == "fisheye_left"
    # Nav camera resolution picks the nav camera.
    assert pick_calibration(cams, 640, 480)["name"] == "fisheye_middle"
    # Unknown size → NO calibration (self-calibration instead of wrong cam).
    assert pick_calibration(cams, 2000, 1500) is None


# ------------------------------------------- POLYFISHEYE (echte S20-Kalib)

_S20_YAML = "tests/data_s20_calibration.yaml"


def test_real_s20_calibration_parsed():
    """The genuine S20 calibration.yaml: all cameras, exact parameters."""
    from pathlib import Path

    from scantobim.photogrammetry.calibration import (
        calibration_for_name,
        pick_calibration,
        pick_calibrations,
        read_camera_calibration,
    )

    cams = read_camera_calibration(Path(_S20_YAML))
    assert cams is not None
    by_name = {c["name"]: c for c in cams}
    left = by_name["fisheye_left"]
    assert left["model"] == "POLYFISHEYE"
    assert abs(left["fx"] - 1480.0265217354813) < 1e-6
    assert abs(left["cx"] - 1749.5565059398484) < 1e-6
    assert abs(left["a12"] - 1.0749520834239590) < 1e-9
    assert left["max_theta_deg"] == 120.0
    assert len(left["poly"]) == 6
    assert abs(left["poly"][0] - (-1.6738521467337498e-02)) < 1e-12

    # Photo size 3504×4672 → left/right, never the 640×480 nav camera.
    pick = pick_calibration(cams, 3504, 4672)
    assert pick["name"] == "fisheye_left"
    sized = pick_calibrations(cams, 3504, 4672)
    assert {c["name"] for c in sized} == {"fisheye_left", "fisheye_right"}
    assert calibration_for_name(sized, "right/frame_01.jpg")["name"] == "fisheye_right"
    assert calibration_for_name(sized, "left/frame_01.jpg")["name"] == "fisheye_left"
    assert pick_calibration(cams, 640, 480)["name"] == "fisheye_middle"


def test_polyfisheye_projection_geometry():
    """Center ray → principal point; r(θ) monotonic; 60° near the edge."""
    from pathlib import Path

    from scantobim.photogrammetry.calibration import read_camera_calibration
    from scantobim.photogrammetry.colmap import polyfisheye_px

    left = {c["name"]: c for c in read_camera_calibration(Path(_S20_YAML))}[
        "fisheye_left"
    ]
    poly = tuple(left["poly"])

    center = np.array([[0.0, 0.0, 1.0]])
    px, py = polyfisheye_px(center, left["fx"], left["fy"], left["a12"],
                            left["cx"], left["cy"], poly)
    assert abs(px[0] - left["cx"]) < 1e-6 and abs(py[0] - left["cy"]) < 1e-6

    # Rays at increasing incidence along +x → strictly growing radius.
    thetas = np.deg2rad(np.arange(5, 121, 5))
    pts = np.column_stack([np.sin(thetas), np.zeros_like(thetas), np.cos(thetas)])
    px, py = polyfisheye_px(pts, left["fx"], left["fy"], left["a12"],
                            left["cx"], left["cy"], poly)
    radii = px - left["cx"]
    assert np.all(np.diff(radii) > 0)
    # 60° incidence lands near the horizontal image border (~1578 px).
    r60 = radii[np.argmin(np.abs(thetas - np.deg2rad(60)))]
    assert 1400 < r60 < 1800


def test_cameras_from_imgpose_polyfisheye_end_to_end(tmp_path):
    """A photo RENDERED with the real fisheye model validates at high score."""
    PIL = pytest.importorskip("PIL.Image")
    from pathlib import Path

    from scantobim.core.cloud import PointCloud
    from scantobim.photogrammetry.calibration import read_camera_calibration
    from scantobim.photogrammetry.colmap import polyfisheye_px
    from scantobim.photogrammetry.xyzopk import cameras_from_imgpose

    cams = read_camera_calibration(Path(_S20_YAML))
    left = {c["name"]: c for c in cams}["fisheye_left"]
    # Shrink to a small test sensor, SAME polynomial: scale A/centers by 1/8.
    scale = 8.0
    calib = dict(left)
    calib["fx"] = left["fx"] / scale
    calib["fy"] = left["fy"] / scale
    calib["a12"] = left["a12"] / scale
    calib["cx"] = left["cx"] / scale
    calib["cy"] = left["cy"] / scale
    calib["width"] = int(3504 / scale)
    calib["height"] = int(4672 / scale)

    def field(x, y):
        r = 50 + 100 * (x + 3) / 6
        b = 120 + 100 * (y + 2) / 4
        return np.stack([r, np.full_like(r, 80.0), b], axis=-1)

    # Floor grid, camera at (0,0,-5) looking +z (identity quaternion).
    gx, gy = np.meshgrid(np.linspace(-3, 3, 700), np.linspace(-2, 2, 500))
    gpts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    pc = gpts - np.array([0.0, 0.0, -5.0])
    px, py = polyfisheye_px(pc, calib["fx"], calib["fy"], calib["a12"],
                            calib["cx"], calib["cy"], tuple(calib["poly"]))
    W, H = calib["width"], calib["height"]
    photo = np.full((H, W, 3), 90, dtype=np.float64)
    cols = field(gpts[:, 0], gpts[:, 1])
    ix = np.clip(px.round().astype(int), 1, W - 2)
    iy = np.clip(py.round().astype(int), 1, H - 2)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            photo[iy + dy, ix + dx] = cols
    img_dir = tmp_path / "left"
    img_dir.mkdir()
    PIL.fromarray(photo.astype(np.uint8)).save(img_dir / "foto.png")

    rng = np.random.default_rng(0)
    pts = np.column_stack([
        rng.uniform(-3, 3, 20000), rng.uniform(-2, 2, 20000), np.zeros(20000),
    ])
    cloud = PointCloud(points=pts, colors=field(pts[:, 0], pts[:, 1]).astype(np.uint8))
    ip = tmp_path / "ImgPose.txt"
    ip.write_text("foto.png 0.0 0.0 -5.0 0.0 0.0 0.0 1.0 100.0\n")

    stats = {}
    result = cameras_from_imgpose(
        ip, img_dir, cloud, stats_out=stats, calibration=[calib]
    )
    assert stats["score"] > 0.8
    assert "POLYFISHEYE" in stats["intrinsics_quelle"]
    cam = result[0]
    assert cam.model == "POLYFISHEYE"
    assert cam.poly == tuple(calib["poly"])
    assert abs(cam.fx - calib["fx"]) < 1e-9


def test_detail_mesh_region_excludes_terrain(tmp_path):
    """Detail bbox comes from BUILDING surfaces — terrain must not widen it."""
    from types import SimpleNamespace

    from scantobim.cli import _build_detail_mesh, _write_detail_mesh
    from scantobim.core.cloud import PointCloud
    from scantobim.core.pipeline import SurfaceGeometry

    rng = np.random.default_rng(0)
    # Building: 2x2x2 m box corner at origin, dense wall points.
    n = 40_000
    wall = np.column_stack([
        rng.random(n) * 2.0, np.zeros(n), rng.random(n) * 2.0
    ]) + rng.normal(0, 0.003, (n, 3))
    # Terrain: huge sparse ground far beyond the building.
    m = 40_000
    ground = np.column_stack([
        rng.random(m) * 60.0 - 30.0, rng.random(m) * 40.0 - 20.0,
        np.zeros(m),
    ]) + rng.normal(0, 0.003, (m, 3))
    cloud = PointCloud(points=np.vstack([wall, ground]))

    z = np.array([0.0, 0.0, 1.0])
    terrain_geo = SurfaceGeometry(
        plane_index=0, normal=z.copy(),
        outer=np.array([[-30.0, -20, 0], [30, -20, 0], [30, 20, 0], [-30, 20, 0]]),
        holes=[], surface_class="terrain",
    )
    wall_geo = SurfaceGeometry(
        plane_index=1, normal=np.array([0.0, 1.0, 0.0]),
        outer=np.array([[0.0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2]]),
        holes=[], surface_class="wall",
    )
    result = SimpleNamespace(surfaces=[terrain_geo, wall_geo])

    detail = _build_detail_mesh(cloud, result, raster=0.05)
    assert detail is not None
    lo = detail.vertices.min(axis=0)
    hi = detail.vertices.max(axis=0)
    # Region = wall bbox + 1 m buffer (+ raster slack) — nowhere near ±30 m.
    assert lo[0] > -1.5 and hi[0] < 3.5
    assert lo[1] > -1.5 and hi[1] < 1.5

    # Plain write path (no cameras): file + report entry.
    rep: dict = {}
    out = tmp_path / "modell.html"
    _write_detail_mesh(detail, None, None, None, rep, out)
    assert (tmp_path / "modell_detail.glb").exists()
    assert rep["detail_mesh"]["triangles"] > 0
