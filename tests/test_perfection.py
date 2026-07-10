"""v1.6 quality upgrades: real-scan hardening, uncertainties, watertight
optimization, sheet metal unfolding."""

import numpy as np
import pytest

from scantobim import PipelineConfig, reconstruct
from scantobim.core.cloud import PointCloud
from scantobim.core.preprocess import remove_edge_artifacts
from tests.synthetic import (
    add_mixed_pixel_strings,
    add_registration_ghost,
    make_bent_plate_scan,
    make_box_scan,
    make_l_room_scan,
)

BOX_SIZE = (4.0, 3.0, 2.5)


# ------------------------------------------------- 1. real-scan hardening

def test_edge_artifact_filter():
    cloud = make_box_scan(density=700, noise=0.004)
    n_surface = len(cloud.points)
    dirty = add_mixed_pixel_strings(cloud, n_strings=8, points_per_string=40)
    cleaned, keep = remove_edge_artifacts(dirty, require_sparse=False)
    # Most artifact points removed, almost no surface points lost.
    artifact_kept = keep[n_surface:].sum()
    surface_kept = keep[:n_surface].sum()
    assert artifact_kept < 0.3 * (len(dirty) - n_surface)
    assert surface_kept > 0.995 * n_surface


def test_pipeline_survives_mixed_pixels():
    # Dense strings: sparse ones are already caught by the outlier filter,
    # dense ones only by the linearity criterion.
    cloud = add_mixed_pixel_strings(
        make_box_scan(density=900, noise=0.004), n_strings=6, points_per_string=400
    )
    result = reconstruct(cloud, PipelineConfig.preset("building"))
    assert result.report["planes"] == 6
    assert result.report["plane_angles"]["other"] == 0
    assert result.report.get("edge_artifacts_removed", 0) > 0


def test_ghost_merge():
    # Separating both ghost sheets requires the k-NN normal radius
    # sqrt(k/(pi*density)) to stay below the ghost offset — TLS-realistic
    # density and a small k achieve that; thinning stays off.
    cloud = make_box_scan(size=(1.2, 1.0, 0.8), density=30000, noise=0.001)
    ghosted = add_registration_ghost(cloud, offset=(0.02, 0.0, 0.0))

    def config(ghost_tol):
        cfg = PipelineConfig.preset("building")
        cfg.voxel_size = 0
        cfg.normal_neighbors = 10
        cfg.distance_threshold = 0.004
        cfg.ghost_offset_tol = ghost_tol
        return cfg

    r_off = reconstruct(ghosted, config(0.0))
    r_on = reconstruct(ghosted, config(0.03))

    # Without ghost merging the x-walls double up; with it the box is clean.
    assert r_off.report["planes"] > 6
    assert r_on.report["planes"] == 6
    assert r_on.report["plane_angles"]["other"] == 0


def test_coplanar_patches_stay_separate():
    """The overlap requirement must keep separate coplanar surfaces apart."""
    from scantobim.core.planes import detect_planes
    from scantobim.core.preprocess import estimate_normals
    from scantobim.core.regularize import regularize_planes

    rng = np.random.default_rng(5)

    def patch(x0):
        n = 800
        return np.column_stack(
            [rng.uniform(x0, x0 + 1, n), rng.uniform(0, 1, n), rng.normal(0, 0.002, n)]
        )

    cloud = PointCloud(points=np.vstack([patch(0.0), patch(5.0)]))
    estimate_normals(cloud, k_neighbors=10)
    planes, _ = detect_planes(
        cloud.points, cloud.normals, distance_threshold=0.008, min_inliers=200, seed=3
    )
    merged = regularize_planes(planes, cloud.points, merge_offset_tol=0.05)
    assert len(merged) == 2  # 4 m apart, no overlap → no merge


# ------------------------------------------------------ 2. uncertainties

def test_surface_sigmas_present_and_plausible():
    cloud = make_box_scan(size=BOX_SIZE, density=900, noise=0.004)
    result = reconstruct(cloud, PipelineConfig.preset("building"))
    for s in result.report["surfaces"]:
        if s.get("status") != "ok":
            continue
        assert 0 < s["sigma_offset"] < 0.001  # sub-mm plane position
        assert 0 < s["area_sigma"] < 1.0
        assert 0 < s["dimension_sigma"] < 0.1
    q = result.report["quantities"]
    assert "volume_sigma" in q
    # Truth must lie within 5 sigma (plus the documented sampling bias bound).
    truth = BOX_SIZE[0] * BOX_SIZE[1] * BOX_SIZE[2]
    tolerance = 5 * q["volume_sigma"] + 0.5  # sampling bias on 6 faces
    assert abs(q["volume"] - truth) < tolerance


def test_sheetmetal_thickness_sigma():
    from scantobim.core.sheetmetal import analyze_sheet_metal
    from tests.synthetic import make_welded_tank_scan

    report = analyze_sheet_metal(make_welded_tank_scan())
    for p in report["plates"]:
        assert 0 < p["thickness_sigma"] < 0.0005
        # measured thickness within 5 sigma + half a point of noise
        assert abs(p["thickness_measured"] - p["thickness_catalog"]) < max(
            5 * p["thickness_sigma"], 0.0005
        )


def test_machinery_sigmas():
    from scantobim.core.machinery import analyze_machinery
    from tests.synthetic import make_stepped_shaft_scan

    report = analyze_machinery(make_stepped_shaft_scan(), seed=1)
    for c in report["cylinders"]:
        assert 0 < c["diameter_sigma"] < 0.001


# --------------------------------------------- 3. watertight optimization

def test_watertight_closes_scan_shadow():
    cloud = make_box_scan(size=BOX_SIZE, density=900, noise=0.004)
    pts = cloud.points
    occluded = (np.abs(pts[:, 2] - BOX_SIZE[2]) < 0.03) & (pts[:, 0] > 2.0)
    shadowed = PointCloud(points=pts[~occluded])

    greedy = reconstruct(shadowed, PipelineConfig.preset("building"))
    assert greedy.mesh.edge_stats()["boundary_edges"] > 0  # hole in the ceiling

    cfg = PipelineConfig.preset("building")
    cfg.watertight = True
    result = reconstruct(shadowed, cfg)
    stats = result.mesh.edge_stats()
    assert stats["boundary_edges"] == 0
    assert stats["non_manifold_edges"] == 0
    q = result.report["quantities"]
    assert q["watertight"]
    truth = BOX_SIZE[0] * BOX_SIZE[1] * BOX_SIZE[2]
    assert abs(q["volume"] - truth) / truth < 0.01
    assert result.report["watertight_optimization"]["selected_faces"] == 6


def test_watertight_l_room():
    cfg = PipelineConfig.preset("indoor")
    cfg.watertight = True
    result = reconstruct(make_l_room_scan(density=900, noise=0.004), cfg)
    assert result.mesh.edge_stats()["boundary_edges"] == 0
    truth = 5 * 4.5 * 2.5 - 2.5 * 2 * 2.5
    assert abs(result.report["quantities"]["volume"] - truth) / truth < 0.01


def test_watertight_step_export_is_solid(tmp_path):
    from scantobim.io.step import write_step

    cfg = PipelineConfig.preset("building")
    cfg.watertight = True
    result = reconstruct(make_box_scan(density=700, noise=0.004), cfg)
    text = write_step(result.surfaces, tmp_path / "solid.stp").read_text()
    assert "CLOSED_SHELL" in text and "MANIFOLD_SOLID_BREP" in text


# ------------------------------------------------------- 4. unfolding

def _bent_part_analysis(second_flange=False):
    from scantobim.core.sheetmetal import analyze_sheet_metal

    cloud = make_bent_plate_scan(second_flange=second_flange)
    return analyze_sheet_metal(cloud)


def test_junction_classification():
    from scantobim.core.unfold import classify_junctions
    from tests.synthetic import make_welded_tank_scan
    from scantobim.core.sheetmetal import analyze_sheet_metal

    report = _bent_part_analysis()
    objs = report["_objects"]
    kinds = classify_junctions(objs["plates"], objs["seams"])
    assert kinds == ["corner"]  # edge-to-edge junction


def test_unfold_l_part():
    from scantobim.core.unfold import unfold_parts

    report = _bent_part_analysis()
    objs = report["_objects"]
    parts = unfold_parts(objs["plates"], objs["seams"], k_factor=0.44)
    assert len(parts) == 1
    part = parts[0]
    assert len(part.bend_lines) == 1
    assert abs(part.bend_angles_deg[0] - 90.0) < 3.0
    # Flat length: 0.2 (base) + BA + 0.15 (flange), BA = (π/2)(t + 0.44 t)
    t = 0.003
    ba = np.pi / 2 * (t + 0.44 * t)
    expected = 0.2 + ba + 0.15
    long_side, short_side = part.size
    assert abs(long_side - expected) < 0.01
    assert abs(short_side - 0.3) < 0.01


def test_unfold_u_channel():
    from scantobim.core.unfold import unfold_parts

    report = _bent_part_analysis(second_flange=True)
    objs = report["_objects"]
    parts = unfold_parts(objs["plates"], objs["seams"])
    assert len(parts) == 1
    assert len(parts[0].bend_lines) == 2


def test_flat_pattern_dxf(tmp_path):
    from scantobim.core.unfold import unfold_parts
    from scantobim.io.dxf import write_flat_pattern_dxf

    report = _bent_part_analysis()
    objs = report["_objects"]
    parts = unfold_parts(objs["plates"], objs["seams"])
    path = write_flat_pattern_dxf(parts, tmp_path / "abwicklung.dxf")
    text = path.read_text()
    assert "BIEGELINIE" in text and "ZUSCHNITT" in text
    assert "Kantung" in text
    assert text.rstrip().endswith("EOF")
