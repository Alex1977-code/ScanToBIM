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
    assert st["points_covered"] > 0.85
    # The far noise blob must be dropped: no vertex anywhere near (40,40,40).
    assert mesh.vertices.max() < 20.0
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


def test_cli_reconstruct_mesh_first(tmp_path, capsys):
    """Stage 1 builds the complete mesh, stage 2 adds the structure model."""
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
    assert "Komplett-Mesh:" in log
    assert log.index("Komplett-Mesh:") < log.index("reconstructing")  # mesh first
    assert (tmp_path / "modell_komplett.glb").exists()
    report = json.loads((tmp_path / "bericht.json").read_text())
    assert report["komplett_mesh"]["triangles"] > 100
    assert report["planes"] >= 5  # structure stage still ran
    html = out.read_text()
    meta = json.loads(re.search(r"const META = (\{.*?\});", html).group(1))
    assert meta["ff_indices"] > 0
    assert "Komplett-Mesh (Scan)" in html
    assert meta["points"] == 0


def test_cli_reconstruct_no_structure(tmp_path, capsys):
    """--no-structure: only the complete mesh, no plane search, exit 0."""
    from scantobim.cli import main
    from scantobim.io.writers import write_point_cloud
    from tests.synthetic import make_hall_with_column_scan

    cloud = make_hall_with_column_scan(density=420, noise=0.003)
    src = tmp_path / "halle.ply"
    write_point_cloud(cloud, src)
    out = tmp_path / "modell.html"
    code = main([
        "reconstruct", str(src), "-o", str(out), "--no-structure",
        "--seed", "3", "--report", str(tmp_path / "bericht.json"),
    ])
    assert code == 0
    log = capsys.readouterr().out
    assert "Strukturanalyse übersprungen" in log
    assert "reconstructing" not in log
    assert out.exists() and (tmp_path / "modell_komplett.glb").exists()
    report = json.loads((tmp_path / "bericht.json").read_text())
    assert "komplett_mesh" in report and "planes" not in report


def test_cli_structure_failure_keeps_mesh(tmp_path, capsys):
    """A scene without planes must still deliver the complete mesh."""
    from scantobim.cli import main
    from scantobim.io.writers import write_point_cloud

    # Gaussian blob + micro distance threshold: the plane stage (incl. its
    # rescue pass) finds nothing, but the mesh must still be delivered.
    rng = np.random.default_rng(2)
    cloud = PointCloud(
        points=rng.normal(0, 0.5, (60000, 3)),
        colors=rng.integers(0, 255, (60000, 3)).astype(np.uint8),
    )
    src = tmp_path / "rauschen.ply"
    write_point_cloud(cloud, src)
    out = tmp_path / "modell.html"
    code = main([
        "reconstruct", str(src), "-o", str(out), "--seed", "3",
        "--dist", "0.00001",
    ])
    assert code == 0
    log = capsys.readouterr().out
    assert "Strukturanalyse fehlgeschlagen" in log
    assert out.exists() and (tmp_path / "modell_komplett.glb").exists()


def test_cli_reconstruct_no_mesh_flag(tmp_path, capsys):
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
    assert not (tmp_path / "modell_komplett.glb").exists()
    log = capsys.readouterr().out
    assert "Komplett-Mesh:" not in log


def test_sharpen_mesh_with_planes():
    """Vertices on detected planes land exactly on them; edge vertices on
    the intersection line — walls flat, edges crisp instead of molten."""
    from scantobim.core.freeform import sharpen_mesh_with_planes
    from scantobim.core.pipeline import SurfaceGeometry

    rng = np.random.default_rng(0)
    # Two orthogonal noisy sheets meeting at the edge x=0/z=0.
    n = 40000
    floor = np.column_stack([
        rng.uniform(0, 4, n), rng.uniform(0, 4, n), rng.normal(0, 0.004, n)])
    wall = np.column_stack([
        rng.normal(0, 0.004, n), rng.uniform(0, 4, n), rng.uniform(0, 3, n)])
    cloud = PointCloud(points=np.vstack([floor, wall]))
    mesh = freeform_mesh_from_points(cloud)
    assert mesh is not None
    voxel = mesh.freeform_stats["voxel"]

    surfaces = [
        SurfaceGeometry(
            plane_index=0, normal=np.array([0.0, 0.0, 1.0]),
            outer=np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]], float),
            holes=[], surface_class="slab", name="slab_000",
        ),
        SurfaceGeometry(
            plane_index=1, normal=np.array([1.0, 0.0, 0.0]),
            outer=np.array([[0, 0, 0], [0, 4, 0], [0, 4, 3], [0, 0, 3]], float),
            holes=[], surface_class="wall", name="wall_001",
        ),
    ]
    frac = sharpen_mesh_with_planes(mesh, surfaces, voxel)
    assert frac > 0.5  # most of the skin lies on the two planes

    v = mesh.vertices
    # Flat-floor vertices (away from the edge) must sit EXACTLY on z=0.
    interior = (v[:, 0] > 1.0) & (v[:, 0] < 3.0) & (v[:, 1] > 1.0) & (v[:, 1] < 3.0)
    near_floor = interior & (np.abs(v[:, 2]) < 0.8 * voxel)
    assert near_floor.sum() > 50
    # The clear majority snaps EXACTLY onto the plane (rim vertices with
    # sideways normals are deliberately left alone).
    assert (np.abs(v[near_floor, 2]) < 1e-9).mean() > 0.6
    # Edge vertices must sit exactly on the intersection line x=0, z=0.
    edge = (np.abs(v[:, 0]) < 1e-9) & (np.abs(v[:, 2]) < 1e-9)
    assert edge.sum() > 3
