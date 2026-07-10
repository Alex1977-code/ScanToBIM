"""Industrial analysis: cylinders, stepped shafts, gears, gear stages."""

import numpy as np
import pytest

from scantobim.core.cloud import PointCloud
from scantobim.core.machinery import (
    analyze_machinery,
    detect_cylinders,
    find_gear_stages,
    group_shafts,
    measure_gear,
)
from scantobim.core.preprocess import estimate_normals
from tests.synthetic import make_gear_scan, make_stepped_shaft_scan, sample_cylinder


def test_detect_single_cylinder():
    rng = np.random.default_rng(1)
    pts = sample_cylinder((0, 0, 0), (0, 0, 1), 0.040, 0.2, 300000, 0.0003, rng)
    cloud = PointCloud(points=pts)
    estimate_normals(cloud, k_neighbors=12)
    cylinders, unassigned = detect_cylinders(
        cloud.points, cloud.normals, distance_threshold=0.0012, min_inliers=500, seed=2
    )
    assert len(cylinders) == 1
    c = cylinders[0]
    assert abs(c.radius - 0.040) < 0.001
    assert abs(abs(float(c.axis @ [0, 0, 1])) - 1.0) < 1e-3
    assert abs(c.length - 0.2) < 0.01
    assert unassigned.mean() < 0.05


def test_stepped_shaft():
    cloud = make_stepped_shaft_scan(
        steps=((0.030, 0.08), (0.050, 0.12), (0.040, 0.10))
    )
    estimate_normals(cloud, k_neighbors=12)
    cylinders, _ = detect_cylinders(
        cloud.points, cloud.normals, distance_threshold=0.0012, min_inliers=400, seed=1
    )
    assert len(cylinders) == 3
    shafts = group_shafts(cylinders, cloud.points)
    assert len(shafts) == 1
    steps = shafts[0].steps
    assert [round(s.diameter, 3) for s in steps] == [0.060, 0.100, 0.080]
    assert abs(shafts[0].total_length - 0.30) < 0.01
    # Steps are ordered and contiguous.
    for a, b in zip(steps, steps[1:]):
        assert b.start >= a.start
        assert abs(b.start - a.end) < 0.02


@pytest.mark.parametrize("teeth", [17, 24, 48])
def test_gear_tooth_count(teeth):
    cloud = make_gear_scan(teeth=teeth, module=0.004, width=0.030)
    gear = measure_gear(cloud.points)
    assert gear is not None
    assert gear.teeth == teeth
    ra_expected = 0.004 * (teeth + 2) / 2
    assert abs(gear.tip_diameter / 2 - ra_expected) < 0.0008
    assert abs(gear.module - 0.004) < 0.0004
    assert abs(gear.width - 0.030) < 0.004


def test_smooth_cylinder_is_not_a_gear():
    rng = np.random.default_rng(3)
    pts = sample_cylinder((0, 0, 0), (0, 0, 1), 0.05, 0.03, 400000, 0.0002, rng, caps=True)
    assert measure_gear(pts) is None


def test_gear_stage_from_meshing_pair():
    m = 0.004
    z1, z2 = 20, 45
    a = m * (z1 + z2) / 2  # center distance of meshing gears
    g1 = make_gear_scan(teeth=z1, module=m, width=0.03, center=(0, 0, 0), seed=1)
    g2 = make_gear_scan(teeth=z2, module=m, width=0.03, center=(a, 0, 0), seed=2)
    gear1 = measure_gear(g1.points)
    gear2 = measure_gear(g2.points)
    assert gear1 is not None and gear2 is not None
    stages = find_gear_stages([gear1, gear2])
    assert len(stages) == 1
    assert abs(stages[0].ratio - z2 / z1) < 0.02
    assert abs(stages[0].center_distance - a) < 0.002


def test_analyze_machinery_end_to_end():
    """Shaft + gear in one scene: full report."""
    shaft = make_stepped_shaft_scan(
        steps=((0.020, 0.10), (0.035, 0.15)), origin=(0.3, 0.3, 0), seed=4
    )
    gear = make_gear_scan(teeth=30, module=0.005, width=0.04, center=(0, 0, 0), seed=5)
    cloud = PointCloud(points=np.vstack([shaft.points, gear.points]))
    report = analyze_machinery(cloud, seed=3)
    assert len(report["shafts"]) >= 1
    assert any(len(s["steps"]) == 2 for s in report["shafts"])
    assert any(g["teeth"] == 30 for g in report["gears"])
    assert "_objects" in report
