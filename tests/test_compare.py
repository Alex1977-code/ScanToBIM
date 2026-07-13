"""Epoch comparison: registration + signed displacement monitoring."""

import json

import numpy as np

from scantobim.core.cloud import PointCloud
from scantobim.core.compare import compare_epochs
from scantobim.io.writers import write_point_cloud
from tests.synthetic import make_box_scan


def _epochs(bulge_mm: float = 30.0):
    """Epoch 1: pristine box. Epoch 2: rigid offset + a bulged wall patch."""
    ref = make_box_scan(density=700, noise=0.002, seed=1)
    cur = make_box_scan(density=700, noise=0.002, seed=2)
    pts = cur.points.copy()
    bulge = (
        (np.abs(pts[:, 0]) < 0.02) & (pts[:, 1] > 1.0) & (pts[:, 1] < 2.0)
        & (pts[:, 2] > 0.8) & (pts[:, 2] < 1.8)
    )
    pts[bulge, 0] -= bulge_mm / 1000.0  # outward on the x=0 wall
    pts = pts + np.array([0.015, -0.008, 0.004])  # rigid survey offset
    return ref, PointCloud(points=pts)


def test_compare_detects_bulge_after_registration():
    ref, cur = _epochs()
    stats, cloud, transform = compare_epochs(ref, cur, tolerance=0.005)
    # The rigid offset (~17.5 mm) must be removed by ICP …
    assert stats["registration_shift"] > 0.010
    # Two independently sampled epochs at 2 mm noise → ~5 mm NN floor.
    assert stats["rms"] < 0.007  # the bulk of the structure is stable
    # … while the 30 mm bulge survives as real deformation.
    assert 0.02 < stats["max"] < 0.08
    assert stats["within_tolerance"] > 0.85
    assert cloud.colors is not None and len(cloud) == stats["points"]


def test_compare_without_registration_sees_offset():
    ref, cur = _epochs(bulge_mm=0.0)
    stats, _, _ = compare_epochs(ref, cur, register=False, tolerance=0.005)
    assert stats["rms"] > 0.008  # the rigid offset dominates


def test_cli_compare(tmp_path, capsys):
    from scantobim.cli import main

    ref, cur = _epochs()
    a = write_point_cloud(ref, tmp_path / "epoche1.ply")
    b = write_point_cloud(cur, tmp_path / "epoche2.ply")
    rep_path = tmp_path / "verformung.json"
    heat = tmp_path / "verformung.ply"
    code = main(
        ["compare", str(a), str(b), "-o", str(rep_path), "--heatmap", str(heat)]
    )
    assert code == 0
    rep = json.loads(rep_path.read_text())
    assert rep["max"] > 0.02
    assert heat.stat().st_size > 0
    assert "Verformung" in capsys.readouterr().out
