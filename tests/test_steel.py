"""Steel member recognition and profile catalog matching."""

import numpy as np

from scantobim.core.cloud import PointCloud
from scantobim.core.steel import (
    detect_steel_members,
    match_profile,
    measure_member,
)
from tests.synthetic import make_ipe_beam_scan


def test_match_profile_exact():
    name, dev = match_profile(0.200, 0.100, "I/H")
    assert name == "IPE 200"
    assert dev < 0.01


def test_match_profile_with_measurement_noise():
    name, _ = match_profile(0.2035, 0.0982, "I/H")
    assert name == "IPE 200"
    name, _ = match_profile(0.298, 0.302, "I/H")
    assert name in ("HEB 300", "HEA 300")


def test_match_profile_rejects_far_off():
    name, dev = match_profile(0.523, 0.417, "I/H")
    assert name == "unbekannt"


def test_measure_ipe_beam():
    cloud = make_ipe_beam_scan(h=0.200, b=0.100, length=3.0)
    member = measure_member(cloud.points)
    assert member is not None
    assert member.family == "I/H"
    assert member.profile == "IPE 200"
    assert abs(member.length - 3.0) < 0.05
    assert abs(member.height - 0.200) < 0.01
    assert abs(member.width - 0.100) < 0.01
    # Member axis along X
    assert abs(abs(float(member.axis @ [1, 0, 0])) - 1.0) < 1e-2


def test_detect_two_members():
    beam1 = make_ipe_beam_scan(h=0.200, b=0.100, length=3.0, seed=1)
    beam2 = make_ipe_beam_scan(h=0.300, b=0.150, length=2.5, seed=2)
    pts2 = beam2.points + np.array([0.0, 2.0, 0.0])  # well separated
    cloud = PointCloud(points=np.vstack([beam1.points, pts2]))
    members = detect_steel_members(cloud)
    profiles = sorted(m.profile for m in members)
    assert profiles == ["IPE 200", "IPE 300"]


def test_compact_cluster_is_not_a_member():
    rng = np.random.default_rng(0)
    blob = rng.normal(0, 0.1, (5000, 3))
    assert measure_member(blob) is None
