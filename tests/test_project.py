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
    assert np.allclose(traj[0], [5, 6, 7])

    tum = tmp_path / "traj_tum.txt"
    tum.write_text(
        "100.0 1 2 3 0 0 0 1\n100.1 1 2 4 0 0 0 1\n100.2 1 2 5 0 0 0 1\n"
    )
    traj = read_trajectory(tum)
    assert traj.shape == (3, 3)
    assert np.allclose(traj[:, 0], 1) and np.allclose(traj[-1], [1, 2, 5])


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
    assert b"data:image/png" in out.read_bytes()  # textured viewer


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
    assert b"data:image/png" in out.read_bytes()  # textured after fallback


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
