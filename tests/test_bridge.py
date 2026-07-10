"""Bridge structure recognition + connection details."""

import numpy as np
import pytest

from scantobim.core.bridge import analyze_bridge
from scantobim.core.connections import connections_report, detect_connections
from scantobim.core.steel import measure_member
from tests.synthetic import (
    make_arch_bridge_scan,
    make_beam_bridge_scan,
    make_cable_stayed_bridge_scan,
    make_ipe_beam_scan,
)


def test_beam_bridge():
    cloud = make_beam_bridge_scan()
    report = analyze_bridge(cloud)
    assert report["bridge_type"] == "Balkenbruecke"
    deck = report["deck"]
    assert abs(deck["length"] - 30.0) < 1.5
    assert abs(deck["width"] - 6.0) < 0.5
    assert abs(deck["elevation"] - 8.0) < 0.2
    assert len(report["piers"]) == 2
    stations = sorted(p["station"] for p in report["piers"])
    assert abs(stations[0] + 5.0) < 1.0 and abs(stations[1] - 5.0) < 1.0
    # 3 spans: 10 + 10 + 10
    assert len(report["spans"]) == 3
    assert all(abs(s - 10.0) < 1.5 for s in report["spans"])
    assert report["abutments"] == 2
    # Bearing points at the pier tops.
    for p in report["piers"]:
        assert abs(p["bearing_point"][2] - 7.5) < 0.3


def test_arch_bridge():
    cloud = make_arch_bridge_scan()
    report = analyze_bridge(cloud)
    assert report["bridge_type"] == "Bogenbruecke"
    assert report["arch"] is not None
    assert abs(report["arch"]["radius"] - 12.0) < 0.8


def test_cable_stayed_bridge():
    cloud = make_cable_stayed_bridge_scan()
    report = analyze_bridge(cloud)
    assert report["bridge_type"] == "Schraegseilbruecke"
    assert report["pylons"] == 1
    assert len(report["cables"]) == 8
    # The stay fan: outer cables flat (~43°), inner ones steep (~15°).
    inclinations = sorted(c["inclination_deg"] for c in report["cables"])
    assert inclinations[0] > 10 and inclinations[-1] < 55
    assert sum(1 for c in report["cables"] if c["inclination_deg"] > 25) >= 4
    assert len(report["piers"]) == 1  # the pier under the pylon
    # Cable diameters near the true 100 mm... 0.1 m nominal (2 x r=0.05).
    for c in report["cables"]:
        assert 0.02 < c["diameter"] < 0.12


def test_no_deck_raises():
    from scantobim.core.cloud import PointCloud

    rng = np.random.default_rng(0)
    cloud = PointCloud(points=rng.normal(0, 3, (3000, 3)))
    with pytest.raises(ValueError, match="no bridge deck"):
        analyze_bridge(cloud)


# ----------------------------------------------------------- connection details

def _member_at(transform_rot, offset, h=0.2, b=0.1, length=3.0, seed=1):
    cloud = make_ipe_beam_scan(h=h, b=b, length=length, seed=seed)
    pts = cloud.points @ transform_rot.T + offset
    return measure_member(pts)


def test_t_connection():
    eye = np.eye(3)
    # Beam A along +X, beam B along +Y ending at A's midpoint: T joint.
    rot_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    a = _member_at(eye, np.array([0.0, 0.0, 0.0]), seed=1)
    b = _member_at(rot_z, np.array([1.5, 0.15, 0.0]), seed=2)
    assert a is not None and b is not None
    nodes = detect_connections([a, b])
    assert len(nodes) == 1
    node = nodes[0]
    assert node.members == [0, 1]
    assert abs(node.angles_deg[0] - 90.0) < 3.0
    assert abs(node.position[0] - 1.5) < 0.3
    rep = connections_report(nodes, [a, b])
    assert rep[0]["members"][0]["profile"] == "IPE 200"


def test_diagonal_connection_angle():
    eye = np.eye(3)
    ang = np.deg2rad(45)
    rot45 = np.array(
        [[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]
    )
    a = _member_at(eye, np.zeros(3), seed=3)
    b = _member_at(rot45, np.array([0.0, 0.05, 0.0]), seed=4)
    nodes = detect_connections([a, b])
    assert len(nodes) == 1
    assert abs(nodes[0].angles_deg[0] - 45.0) < 4.0


def test_far_members_no_connection():
    eye = np.eye(3)
    a = _member_at(eye, np.zeros(3), seed=5)
    b = _member_at(eye, np.array([0.0, 5.0, 0.0]), seed=6)
    assert detect_connections([a, b]) == []


def test_three_member_node():
    eye = np.eye(3)
    rot_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    rot_zx = rot_z @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    a = _member_at(eye, np.zeros(3), seed=7)
    b = _member_at(rot_z, np.array([1.5, 0.15, 0.0]), seed=8)
    c = _member_at(rot_zx, np.array([1.5, 0.1, 0.15]), seed=9)
    nodes = detect_connections([a, b, c])
    assert len(nodes) == 1
    assert len(nodes[0].members) == 3
    assert len(nodes[0].angles_deg) == 3
