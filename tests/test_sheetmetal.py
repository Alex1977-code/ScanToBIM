"""Welded sheet metal: plates, thickness, weld seams, cutting DXF."""

import re

import numpy as np
import pytest

from scantobim.core.sheetmetal import analyze_sheet_metal, match_thickness
from scantobim.io.dxf import write_cutting_dxf
from tests.synthetic import make_welded_tank_scan


def test_match_thickness():
    assert match_thickness(0.0079) == 0.008
    assert match_thickness(0.0061) == 0.006
    assert match_thickness(0.045) is None  # far from any stock size


@pytest.fixture(scope="module")
def tank_report():
    cloud = make_welded_tank_scan()
    return analyze_sheet_metal(cloud)


def test_tank_plates(tank_report):
    plates = tank_report["plates"]
    assert len(plates) == 5  # bottom + 4 walls
    thicknesses = sorted(p["thickness_catalog"] for p in plates)
    assert thicknesses == [0.006, 0.006, 0.006, 0.006, 0.008]
    # Bottom plate: 0.6 x 0.4
    bottom = max(plates, key=lambda p: p["area"])
    assert bottom["thickness_catalog"] == 0.008
    assert abs(bottom["size"][0] - 0.6) < 0.02
    assert abs(bottom["size"][1] - 0.4) < 0.02
    # Thickness measured within 0.5 mm of nominal.
    for p in plates:
        assert abs(p["thickness_measured"] - p["thickness_catalog"]) < 0.0005


def test_tank_weights(tank_report):
    # Bottom: 0.6*0.4*0.008*7850 ≈ 15.1 kg
    bottom = max(tank_report["plates"], key=lambda p: p["area"])
    assert abs(bottom["weight_kg"] - 0.6 * 0.4 * 0.008 * 7850) < 1.5
    assert tank_report["totals"]["weight_kg"] > 0


def test_tank_weld_seams(tank_report):
    seams = tank_report["weld_seams"]
    # 4 bottom-wall seams + 4 vertical wall-wall seams
    assert len(seams) == 8
    lengths = sorted(s["length"] for s in seams)
    # 4 vertical seams ≈ 0.3, then 2 x ≈0.4 and 2 x ≈0.6
    assert all(abs(v - 0.3) < 0.06 for v in lengths[:4])
    assert all(s["angle_deg"] > 80 for s in seams)  # all right-angle fillet welds
    total = tank_report["totals"]["weld_length"]
    expected = 4 * 0.3 + 2 * 0.4 + 2 * 0.6
    assert abs(total - expected) / expected < 0.15


def test_cutting_dxf(tank_report, tmp_path):
    plates = tank_report["_objects"]["plates"]
    path = write_cutting_dxf(plates, tmp_path / "zuschnitt.dxf")
    text = path.read_text()
    assert text.count("POLYLINE") == len(plates)
    assert text.count("SEQEND") == len(plates)
    assert "ZUSCHNITT" in text and "BESCHRIFTUNG" in text
    # Every plate labelled with its catalog thickness.
    labels = re.findall(r"^Blech \d+ t=(\d+)mm$", text, flags=re.M)
    assert sorted(labels) == ["6", "6", "6", "6", "8"]
    assert text.rstrip().endswith("EOF")


def test_dxf_outlines_true_to_scale(tank_report, tmp_path):
    plates = tank_report["_objects"]["plates"]
    path = write_cutting_dxf(plates, tmp_path / "zuschnitt.dxf")
    text = path.read_text()
    # Sum of X extents of all outlines ≈ sum of plate lengths (+gaps).
    xs = [float(v) for v in re.findall(r"^10\n([-\d.]+)$", text, flags=re.M)]
    total_span = max(xs) - min(xs)
    plate_lengths = sum(p.size[0] for p in plates)
    assert total_span >= plate_lengths * 0.95


def test_rejects_single_sided_scan():
    from scantobim.core.cloud import PointCloud
    from tests.synthetic import sample_rect

    rng = np.random.default_rng(0)
    x, y = np.eye(3)[0], np.eye(3)[1]
    single = sample_rect((0, 0, 0), x, y, 1.0, 1.0, 200000, 0.0002, rng)
    wall = sample_rect((0, 0, 0), x, np.eye(3)[2], 1.0, 0.5, 200000, 0.0002, rng)
    cloud = PointCloud(points=np.vstack([single, wall]))
    with pytest.raises(ValueError, match="no sheet metal plates"):
        analyze_sheet_metal(cloud)
