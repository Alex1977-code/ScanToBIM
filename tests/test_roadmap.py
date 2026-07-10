"""Roadmap features: cones, hollow/angle steel profiles, multi-storey."""

import numpy as np
import pytest

from scantobim import PipelineConfig, reconstruct
from scantobim.core.machinery import analyze_machinery, detect_cones
from scantobim.core.preprocess import estimate_normals
from scantobim.core.semantics import detect_storeys
from scantobim.core.steel import measure_member
from scantobim.io.ifc import write_ifc
from tests.synthetic import (
    make_cone_scan,
    make_l_angle_scan,
    make_shs_beam_scan,
    make_two_storey_scan,
)


# ----------------------------------------------------------------------- cones

def test_detect_cone():
    cloud = make_cone_scan(half_angle_deg=25.0, t_min=0.2, t_max=0.8)
    estimate_normals(cloud, k_neighbors=12)
    cones, unassigned = detect_cones(
        cloud.points, cloud.normals, distance_threshold=0.002, min_inliers=500, seed=1
    )
    assert len(cones) == 1
    c = cones[0]
    assert abs(c.half_angle_deg - 25.0) < 1.0
    assert np.linalg.norm(c.apex) < 0.02  # apex at origin
    assert abs(abs(float(c.axis @ [0, 0, 1])) - 1.0) < 1e-3
    assert abs(c.height - 0.6) < 0.03
    assert unassigned.mean() < 0.05


def test_cone_in_analyze():
    cloud = make_cone_scan(
        apex=(1, 1, 0), half_angle_deg=30.0, t_min=0.15, t_max=0.5, seed=3
    )
    report = analyze_machinery(cloud, seed=2)
    assert len(report["cones"]) == 1
    assert abs(report["cones"][0]["half_angle_deg"] - 30.0) < 1.5


def test_cylinder_not_detected_as_cone():
    from tests.synthetic import sample_cylinder

    rng = np.random.default_rng(5)
    pts = sample_cylinder((0, 0, 0), (0, 0, 1), 0.05, 0.4, 300000, 0.0003, rng)
    from scantobim.core.cloud import PointCloud

    cloud = PointCloud(points=pts)
    estimate_normals(cloud, k_neighbors=12)
    cones, _ = detect_cones(
        cloud.points, cloud.normals, distance_threshold=0.0012, min_inliers=500, seed=1
    )
    # A cylinder has parallel tangent planes — no stable apex exists.
    assert len(cones) == 0


# ------------------------------------------------------------------ steel v2

def test_shs_profile():
    cloud = make_shs_beam_scan(size=0.100, length=2.0)
    member = measure_member(cloud.points)
    assert member is not None
    assert member.family == "hollow"
    assert member.profile == "SHS 100"
    assert abs(member.length - 2.0) < 0.05


def test_l_angle_profile():
    cloud = make_l_angle_scan(leg=0.080, length=2.0)
    member = measure_member(cloud.points)
    assert member is not None
    assert member.family == "L"
    assert member.profile == "L 80x80"


def test_ipe_still_works():
    from tests.synthetic import make_ipe_beam_scan

    member = measure_member(make_ipe_beam_scan(h=0.200, b=0.100, length=3.0).points)
    assert member is not None and member.profile == "IPE 200"


# -------------------------------------------------------------- multi-storey

def test_detect_storeys_unit():
    assert detect_storeys([]) == []
    one = detect_storeys([0.0, 2.5])
    assert len(one) == 1 and one[0]["height"] == 2.5
    two = detect_storeys([0.0, 0.02, 2.6, 2.62, 5.2])
    assert len(two) == 2
    assert abs(two[0]["height"] - 2.59) < 0.05
    assert abs(two[1]["elevation"] - 2.61) < 0.05
    # a shelf 40 cm above the floor is no storey boundary
    assert len(detect_storeys([0.0, 0.4, 2.6])) == 1


@pytest.fixture(scope="module")
def two_storey_result():
    cloud = make_two_storey_scan()
    return reconstruct(cloud, PipelineConfig.preset("indoor"))


def test_two_storey_report(two_storey_result):
    storeys = two_storey_result.report["storeys"]
    assert len(storeys) == 2
    assert abs(storeys[0]["elevation"] - 0.0) < 0.1
    assert abs(storeys[1]["elevation"] - 2.6) < 0.1
    assert abs(storeys[0]["height"] - 2.6) < 0.1


def test_two_storey_ifc(two_storey_result, tmp_path):
    path = write_ifc(
        two_storey_result.surfaces,
        tmp_path / "haus.ifc",
        storeys=two_storey_result.report["storeys"],
    )
    text = path.read_text()
    assert text.count("IFCBUILDINGSTOREY(") == 2
    assert "Geschoss 0" in text and "Geschoss 1" in text
    assert text.count("IFCRELCONTAINEDINSPATIALSTRUCTURE(") == 2
