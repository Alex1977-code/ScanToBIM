"""End-to-end quality gates: noisy synthetic scans must yield clean-edged models."""

import json

import numpy as np
import pytest

from scantobim import PipelineConfig, reconstruct
from scantobim.io.writers import write_mesh
from tests.synthetic import add_outliers, make_box_scan, make_l_room_scan

BOX_SIZE = (4.0, 3.0, 2.5)


@pytest.fixture(scope="module")
def box_result():
    cloud = add_outliers(make_box_scan(size=BOX_SIZE, density=900, noise=0.004), 0.01)
    cfg = PipelineConfig.preset("building")
    return reconstruct(cloud, cfg)


def test_box_plane_count(box_result):
    assert box_result.report["planes"] == 6


def test_box_angles_exact(box_result):
    angles = box_result.report["plane_angles"]
    # 15 plane pairs in a box: 3 parallel + 12 orthogonal, nothing else.
    assert angles["pairs"] == 15
    assert angles["exactly_parallel"] == 3
    assert angles["exactly_orthogonal"] == 12
    assert angles["other"] == 0


def test_box_corners_found(box_result):
    assert box_result.report["exact_corners"] == 8


def test_box_corner_accuracy(box_result):
    """Mesh corners must sit on the true box corners within millimeters."""
    sx, sy, sz = BOX_SIZE
    true_corners = np.array(
        [[x, y, z] for x in (0, sx) for y in (0, sy) for z in (0, sz)]
    )
    verts = box_result.mesh.vertices
    for corner in true_corners:
        d = np.linalg.norm(verts - corner, axis=1).min()
        assert d < 0.03, f"corner {corner} missed by {d:.3f} m"


def test_box_edges_are_straight(box_result):
    """Boundary vertices of each face must lie on exact plane intersections.

    We verify via face count: a perfectly straightened box face needs exactly
    2 triangles (4 boundary vertices). Allow minor slack for snapping order.
    """
    mesh = box_result.mesh
    # 6 faces x 2 triangles = 12 triangles for a perfect box.
    assert len(mesh.faces) <= 24
    # Surface area close to the analytic value.
    sx, sy, sz = BOX_SIZE
    true_area = 2 * (sx * sy + sx * sz + sy * sz)
    assert abs(mesh.area() - true_area) / true_area < 0.05


def test_box_watertight(box_result):
    """After corner snapping + welding the box must close up watertight."""
    stats = box_result.mesh.edge_stats()
    assert stats["boundary_edges"] == 0
    assert stats["non_manifold_edges"] == 0


def test_box_outward_normals(box_result):
    mesh = box_result.mesh
    center = np.array(BOX_SIZE) / 2.0
    tri_centers = mesh.vertices[mesh.faces].mean(axis=1)
    outward = np.einsum("ij,ij->i", mesh.face_normals(), tri_centers - center)
    assert np.all(outward > 0)


def test_box_residual_small(box_result):
    rep = box_result.report
    assert rep["residual_points"] / rep["preprocessed_points"] < 0.08


def test_l_room():
    cloud = make_l_room_scan(density=900, noise=0.004)
    cfg = PipelineConfig.preset("indoor")
    result = reconstruct(cloud, cfg)
    # 6 walls + floor + ceiling
    assert result.report["planes"] == 8
    angles = result.report["plane_angles"]
    assert angles["other"] == 0  # everything parallel or orthogonal
    # inward normals: faces point towards the room interior
    mesh = result.mesh
    interior = np.array([1.25, 1.25, 1.25])  # inside the L
    tri_centers = mesh.vertices[mesh.faces].mean(axis=1)
    to_interior = interior - tri_centers
    inward = np.einsum("ij,ij->i", mesh.face_normals(), to_interior)
    assert (inward > 0).mean() > 0.95


def test_open_box_keeps_boundary():
    """A box without ceiling must NOT be forced watertight."""
    cloud = make_box_scan(density=900, noise=0.004, open_top=True)
    result = reconstruct(cloud, PipelineConfig.preset("building"))
    assert result.report["planes"] == 5
    stats = result.mesh.edge_stats()
    assert stats["boundary_edges"] > 0  # the open rim stays open


def test_deterministic_with_seed():
    cloud = make_box_scan(density=400, noise=0.004)
    r1 = reconstruct(cloud, PipelineConfig(seed=11))
    r2 = reconstruct(cloud, PipelineConfig(seed=11))
    np.testing.assert_array_equal(r1.mesh.faces, r2.mesh.faces)
    np.testing.assert_allclose(r1.mesh.vertices, r2.mesh.vertices)


def test_report_is_json_serializable(box_result, tmp_path):
    text = json.dumps(box_result.report, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o.item())
    assert "planes" in json.loads(text)


def test_export_all_formats(box_result, tmp_path):
    for ext in (".obj", ".ply", ".stl", ".glb"):
        out = write_mesh(box_result.mesh, tmp_path / f"model{ext}")
        assert out.stat().st_size > 0


def test_rejects_pure_noise():
    # A Gaussian ball has no planar structure at all (a uniform cube would
    # legitimately expose its six boundary faces as planes).
    rng = np.random.default_rng(0)
    from scantobim.core.cloud import PointCloud

    cloud = PointCloud(points=rng.normal(0, 3.0, (3000, 3)))
    with pytest.raises(ValueError):
        reconstruct(cloud, PipelineConfig())


def test_detail_preset_reconstructs_column():
    """Preset 'detail': a round column becomes a true cylinder in the model."""
    from tests.synthetic import make_hall_with_column_scan

    cloud = make_hall_with_column_scan()
    result = reconstruct(cloud, PipelineConfig.preset("detail"))
    rep = result.report
    assert rep["planes"] >= 6
    assert len(rep.get("cylinders", [])) == 1
    cyl = rep["cylinders"][0]
    assert abs(cyl["radius"] - 0.25) < 0.01
    assert abs(cyl["length"] - 3.0) < 0.2
    assert abs(abs(cyl["axis"][2]) - 1.0) < 0.02  # vertical
    # The column's points must no longer sit in the residual.
    assert any(n.startswith("zylinder") for n in result.mesh.group_names.values())


def test_cylinder_detection_off_by_default():
    from tests.synthetic import make_hall_with_column_scan

    cloud = make_hall_with_column_scan(density=700)
    result = reconstruct(cloud, PipelineConfig.preset("building"))
    assert "cylinders" not in result.report


def test_detail_recovery_finds_small_surfaces():
    """A 0.5 m cabinet is below the main size gate but must be recovered."""
    from tests.synthetic import make_hall_with_cabinet_scan

    cloud = make_hall_with_cabinet_scan()

    plain = reconstruct(cloud, PipelineConfig.preset("building"))
    assert plain.report["planes"] == 6  # main pass skips the cabinet

    cfg = PipelineConfig.preset("building")
    cfg.detail_recovery = True
    result = reconstruct(cloud, cfg)
    ok = [s for s in result.report["surfaces"] if s.get("status") == "ok"]
    cabinet_faces = [s for s in ok if s["plane"] >= 6]
    # All five cabinet faces polygonize at their true 0.25 m² size;
    # noise slivers are rejected during polygonization and never surface.
    assert len(cabinet_faces) == 5
    for s in cabinet_faces:
        assert abs(s["area"] - 0.25) < 0.02
    assert result.report["residual_points"] < 0.4 * plain.report["residual_points"]


def test_auto_reconstruct_picks_best():
    """Auto-tuning must score all candidates and return a clean model."""
    from scantobim.core.autotune import auto_reconstruct
    from tests.synthetic import make_hall_with_cabinet_scan

    logs = []
    result = auto_reconstruct(
        make_hall_with_cabinet_scan(density=700), seed=1, log=logs.append
    )
    board = result.report["auto_tuning"]
    assert len([e for e in board if "score" in e]) >= 3
    assert sum(1 for e in board if e.get("selected")) == 1
    winner = next(e for e in board if e.get("selected"))
    # The winner must explain the scene well.
    assert winner["unexplained"] < 0.1
    assert result.report["planes"] >= 6
    assert any("gewinnt" in line for line in logs)
    # The winning configuration is exported in profile form (scale-free).
    wc = result.report["auto_tuning_winner_config"]
    assert wc["candidate"] == winner["candidate"]
    assert wc["preset"] in ("building", "detail")
    assert wc["advanced"]["distance_factor"] > 0
    assert 0 < wc["advanced"]["min_inlier_ratio"] < 1


def test_source_profiles():
    """Sensor profiles layer over scene presets and reach the pipeline."""
    from scantobim.core.pipeline import SOURCE_PROFILES, apply_source_profile

    cfg = PipelineConfig.preset("building")
    apply_source_profile(cfg, "slam")
    assert cfg.distance_factor == 3.5
    assert cfg.ghost_offset_tol == 0.03  # SLAM ghost walls merged by default

    cfg2 = PipelineConfig.preset("indoor")
    apply_source_profile(cfg2, "tls")
    assert cfg2.distance_factor == 2.5
    assert cfg2.orient == "inward"  # preset survives the profile layer
    assert set(SOURCE_PROFILES) == {"standard", "slam", "tls", "drohne", "iphone"}


def test_cli_source_profile(tmp_path):
    import json as _json

    from scantobim.cli import main
    from scantobim.io.writers import write_point_cloud

    src = write_point_cloud(make_box_scan(density=700, noise=0.004), tmp_path / "s.ply")
    rep_path = tmp_path / "r.json"
    code = main(
        ["reconstruct", str(src), "-o", str(tmp_path / "m.glb"),
         "--source", "slam", "--report", str(rep_path), "--seed", "1"]
    )
    assert code == 0
    config = _json.loads(rep_path.read_text())["config"]
    assert config["distance_factor"] == 3.5
    assert config["ghost_offset_tol"] == 0.03


def test_rescue_on_slam_double_walls():
    """Un-merged SLAM registration ghosts (double sheets inside the
    tolerance band) used to kill plane detection entirely — the rescue
    pass must recover and flag itself in the report."""
    rng = np.random.default_rng(4)

    def double_sheet(u_axis, v_axis, n_axis, extent, n=40000):
        u = rng.uniform(0, extent, n)[:, None] * u_axis
        v = rng.uniform(0, extent, n)[:, None] * v_axis
        base = u + v + rng.normal(0, 0.0015, (n, 3))
        ghost = base + 0.015 * np.asarray(n_axis)  # 1.5 cm registration ghost
        return np.vstack([base, ghost])

    ex, ey, ez = np.eye(3)
    pts = np.vstack([
        double_sheet(ex, ey, ez, 2.0),  # doubled floor
        double_sheet(ex, ez, ey, 2.0),  # doubled wall y=0
        double_sheet(ey, ez, ex, 2.0),  # doubled wall x=0
    ])
    from scantobim.core.cloud import PointCloud

    cloud = PointCloud(points=pts)
    cfg = PipelineConfig.preset("building")
    cfg.voxel_size = 0.006          # keep both sheets (no coarse merge)
    cfg.distance_threshold = 0.020  # > ghost gap → mid-plane fit, gates reject
    cfg.detect_openings = False
    result = reconstruct(cloud, cfg)
    assert result.report["planes"] >= 2
    assert "rescue" in result.report
    assert "gelockerten toleranzen" in result.report["rescue"]
