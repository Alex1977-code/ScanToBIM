"""Hybrid model: free-form skin around the plane-residual points."""

import json
import re

import numpy as np

from scantobim.core.cloud import PointCloud
from scantobim.core.freeform import freeform_mesh_from_points


def _tube_cloud(n=30000, seed=0, with_noise_blob=True):
    """Arch tube (like a railing / steel arch member) + optional far blob."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    R, r = 5.0, 0.06
    pts = np.column_stack([
        (R + r * np.cos(phi)) * np.cos(t),
        r * np.sin(phi),
        (R + r * np.cos(phi)) * np.sin(t),
    ]) + rng.normal(0, 0.003, (n, 3))
    if with_noise_blob:
        pts = np.vstack([pts, rng.uniform(40, 41, (50, 3))])
    colors = rng.integers(60, 200, (len(pts), 3)).astype(np.uint8)
    return PointCloud(points=pts, colors=colors)


def test_freeform_arch_tube():
    cloud = _tube_cloud()
    mesh = freeform_mesh_from_points(cloud)
    assert mesh is not None and len(mesh.faces) > 1000
    st = mesh.freeform_stats
    assert st["components"] == 1  # the far noise blob must be dropped
    assert st["points_covered"] > 0.9
    assert mesh.vertex_colors is not None
    assert len(mesh.vertex_colors) == len(mesh.vertices)
    assert mesh.faces.min() >= 0 and mesh.faces.max() < len(mesh.vertices)
    # The skin must follow the tube: distance of vertices to the ideal
    # torus surface stays in the voxel regime.
    v = mesh.vertices
    d_axis = np.abs(np.sqrt(v[:, 0] ** 2 + v[:, 2] ** 2) - 5.0)
    rr = np.sqrt(d_axis**2 + v[:, 1] ** 2)
    assert np.abs(rr - 0.06).mean() < 0.05


def test_freeform_face_budget():
    cloud = _tube_cloud(with_noise_blob=False)
    mesh = freeform_mesh_from_points(cloud, max_faces=8000)
    assert mesh is not None
    assert len(mesh.faces) <= 8000  # voxel grows until the budget fits


def test_freeform_too_sparse_returns_none():
    rng = np.random.default_rng(1)
    tiny = PointCloud(points=rng.uniform(0, 1, (50, 3)))
    assert freeform_mesh_from_points(tiny) is None


def test_viewer_embeds_freeform_layer(tmp_path):
    from scantobim.core.mesh import Mesh
    from scantobim.io.writers import write_mesh

    base = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
    )
    ff = freeform_mesh_from_points(_tube_cloud(n=8000, with_noise_blob=False))
    out = write_mesh(base, tmp_path / "hybrid.html", freeform=ff)
    html = out.read_text()
    meta = json.loads(re.search(r"const META = (\{.*?\});", html).group(1))
    assert meta["ff_triangles"] == len(ff.faces)
    assert meta["ff_indices"] == len(ff.faces) * 3
    assert "Freiform-Restgeometrie" in html


def test_cli_reconstruct_hybrid(tmp_path, capsys):
    """Column residual of a hall scan becomes a free-form layer + GLB."""
    from scantobim.cli import main
    from scantobim.io.writers import write_point_cloud
    from tests.synthetic import make_hall_with_column_scan

    cloud = make_hall_with_column_scan(density=420, noise=0.003)
    src = tmp_path / "halle.ply"
    write_point_cloud(cloud, src)
    out = tmp_path / "modell.html"
    code = main([
        "reconstruct", str(src), "-o", str(out), "--preset", "building",
        "--seed", "3", "--report", str(tmp_path / "bericht.json"),
    ])
    assert code == 0
    log = capsys.readouterr().out
    assert "Freiform-Mesh:" in log
    assert (tmp_path / "modell_freiform.glb").exists()
    report = json.loads((tmp_path / "bericht.json").read_text())
    assert report["freeform"]["triangles"] > 100
    html = out.read_text()
    meta = json.loads(re.search(r"const META = (\{.*?\});", html).group(1))
    assert meta["ff_indices"] > 0
    # Raw points layer is replaced by the free-form layer.
    assert meta["points"] == 0


def test_cli_reconstruct_no_freeform_flag(tmp_path, capsys):
    from scantobim.cli import main
    from scantobim.io.writers import write_point_cloud
    from tests.synthetic import make_hall_with_column_scan

    cloud = make_hall_with_column_scan(density=420, noise=0.003)
    src = tmp_path / "halle.ply"
    write_point_cloud(cloud, src)
    out = tmp_path / "modell.html"
    assert main([
        "reconstruct", str(src), "-o", str(out), "--no-freeform", "--seed", "3",
    ]) == 0
    assert not (tmp_path / "modell_freiform.glb").exists()
    log = capsys.readouterr().out
    assert "Freiform-Mesh:" not in log
