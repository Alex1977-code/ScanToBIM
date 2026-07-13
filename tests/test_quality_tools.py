"""QA tooling: Soll-Ist deviation analysis, orthographic views, dimensions."""

import json

import numpy as np
import pytest

from scantobim import PipelineConfig, reconstruct
from scantobim.cli import main
from scantobim.io.writers import write_point_cloud
from tests.synthetic import make_box_scan

BOX_SIZE = (4.0, 3.0, 2.5)


# ------------------------------------------------------------ deviation

def test_point_triangle_distance_regions():
    from scantobim.core.deviation import _point_triangle_distance

    a = np.array([[0.0, 0, 0]] * 5)
    b = np.array([[2.0, 0, 0]] * 5)
    c = np.array([[0.0, 2, 0]] * 5)
    p = np.array(
        [
            [0.5, 0.5, 1.0],  # above the interior → d = 1
            [-1.0, -1.0, 0.0],  # vertex A region → sqrt(2)
            [3.0, 0.0, 0.0],  # vertex B region → 1
            [1.0, -2.0, 0.0],  # edge AB region → 2
            [2.0, 2.0, 0.0],  # edge BC region → sqrt(2)
        ]
    )
    d, closest = _point_triangle_distance(p, a, b, c)
    assert np.allclose(d, [1.0, np.sqrt(2), 1.0, 2.0, np.sqrt(2)], atol=1e-12)
    assert np.allclose(closest[0], [0.5, 0.5, 0.0])


def test_signed_distance_to_mesh():
    from scantobim.core.deviation import signed_distance_to_mesh
    from scantobim.core.geometry3d import box_mesh

    box = box_mesh([0, 0, 0], [1, 0, 0], (2.0, 2.0, 2.0))  # walls at ±1
    pts = np.array([[0.0, 0.0, 1.3], [0.0, 0.0, 0.6], [1.2, 0.0, 0.0]])
    d = signed_distance_to_mesh(pts, box)
    assert np.allclose(np.abs(d), [0.3, 0.4, 0.2], atol=1e-9)
    assert d[0] > 0  # outside → in front of the outward-facing top
    assert d[1] < 0  # inside → behind the surface
    assert d[2] > 0


def test_deviation_analysis_statistics():
    from scantobim.core.deviation import deviation_analysis

    noise = 0.004
    cloud = make_box_scan(size=BOX_SIZE, density=700, noise=noise)
    result = reconstruct(cloud, PipelineConfig.preset("building"))
    stats, dev_cloud = deviation_analysis(result.mesh, cloud, tolerance=3 * noise)
    # Gaussian noise: RMS ≈ σ, ~99.7% within 3σ.
    assert 0.5 * noise < stats["rms"] < 2.0 * noise
    assert stats["within_tolerance"] > 0.97
    assert stats["p95"] < 3 * noise
    assert dev_cloud.colors is not None and len(dev_cloud) == stats["points"]
    # Near-zero deviation → colors close to white.
    mean_color = dev_cloud.colors.mean()
    assert mean_color > 180


def test_deviation_detects_bulge():
    """A bulged wall region must blow up the statistics and turn red/blue."""
    from scantobim.core.deviation import deviation_analysis

    cloud = make_box_scan(size=BOX_SIZE, density=700, noise=0.003)
    result = reconstruct(cloud, PipelineConfig.preset("building"))
    pts = cloud.points.copy()
    bulge = (pts[:, 0] < 0.01) & (pts[:, 1] > 1.0) & (pts[:, 1] < 2.0)
    pts[bulge, 0] -= 0.03  # 30 mm outward bulge on the x=0 wall
    from scantobim.core.cloud import PointCloud

    stats, dev_cloud = deviation_analysis(
        result.mesh, PointCloud(points=pts), tolerance=0.005
    )
    assert stats["max"] > 0.02
    assert stats["within_tolerance"] < 0.995
    hist = stats["histogram_mm"]
    assert sum(v for k, v in hist.items() if k.startswith("20-")) > 0


def test_cli_reconstruct_deviation(tmp_path, capsys):
    src = write_point_cloud(make_box_scan(density=700, noise=0.004), tmp_path / "scan.ply")
    dev = tmp_path / "abweichung.ply"
    rep = tmp_path / "bericht.json"
    code = main(
        ["reconstruct", str(src), "-o", str(tmp_path / "m.glb"),
         "--deviation", str(dev), "--tolerance", "0.012",
         "--report", str(rep), "--seed", "1"]
    )
    assert code == 0
    assert dev.stat().st_size > 0
    report = json.loads(rep.read_text())
    # 3σ tolerance on σ=4 mm noise → ~99.7% within.
    assert report["deviation"]["within_tolerance"] > 0.98
    assert "Soll-Ist" in capsys.readouterr().out


# ---------------------------------------------------------------- views

def test_ortho_views_written(tmp_path):
    from scantobim.io.views import write_ortho_views

    cloud = make_box_scan(size=BOX_SIZE, density=700, noise=0.004)
    result = reconstruct(cloud, PipelineConfig.preset("building"))
    files = write_ortho_views(result.mesh, tmp_path / "ansichten", max_size=400)
    names = {f.name for f in files}
    assert names == {
        "draufsicht.png", "ansicht_nord.png", "ansicht_sued.png",
        "ansicht_ost.png", "ansicht_west.png",
    }
    for f in files:
        assert f.stat().st_size > 500
        assert (f.with_suffix(".pgw")).exists()


def test_ortho_view_scale_and_content(tmp_path):
    """The rendered footprint must match the true scale from the world file."""
    from scantobim.io.png import encode_png  # noqa: F401 (sanity import)
    from scantobim.io.views import render_ortho_view

    cloud = make_box_scan(size=BOX_SIZE, density=700, noise=0.004)
    result = reconstruct(cloud, PipelineConfig.preset("building"))
    ppu = 50.0  # 50 px per meter
    img, _ = render_ortho_view(
        result.mesh,
        np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]),
        pixels_per_unit=ppu,
    )
    # Non-background pixels form the roof footprint: 4 m × 3 m → 200 × 150 px.
    filled = np.any(img != 248, axis=2)
    cols = np.flatnonzero(filled.any(axis=0))
    rows = np.flatnonzero(filled.any(axis=1))
    assert abs((cols[-1] - cols[0] + 1) - BOX_SIZE[0] * ppu) < 4
    assert abs((rows[-1] - rows[0] + 1) - BOX_SIZE[1] * ppu) < 4


def test_ortho_views_use_texture(tmp_path):
    """Textured models render their photo texture into the views."""
    from scantobim.core.cloud import PointCloud
    from scantobim.core.texture import bake_texture_from_cloud
    from scantobim.io.views import render_ortho_view

    cloud = make_box_scan(density=700, noise=0.004)
    rng = np.random.default_rng(0)
    colored = PointCloud(
        points=cloud.points,
        colors=np.tile(np.array([200, 40, 40], dtype=np.uint8), (len(cloud.points), 1)),
    )
    result = reconstruct(colored, PipelineConfig.preset("building"))
    mesh = bake_texture_from_cloud(result, colored)
    img, _ = render_ortho_view(
        mesh, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
        np.array([0, 0, 1.0]), pixels_per_unit=40.0,
    )
    filled = np.any(img != 248, axis=2)
    reds = img[filled]
    assert reds[:, 0].mean() > 150 and reds[:, 1].mean() < 100  # red texture


def test_cli_views_option(tmp_path, capsys):
    src = write_point_cloud(make_box_scan(density=700, noise=0.004), tmp_path / "scan.ply")
    views = tmp_path / "ansichten"
    code = main(
        ["reconstruct", str(src), "-o", str(tmp_path / "m.glb"),
         "--views", str(views), "--seed", "1"]
    )
    assert code == 0
    assert len(list(views.glob("*.png"))) == 5
    assert "Ansichten" in capsys.readouterr().out


# ----------------------------------------------------- floorplan dimensions

def test_floorplan_dimensions(tmp_path):
    from scantobim.io.dxf import write_floorplan_dxf

    cloud = make_box_scan(size=BOX_SIZE, density=700, noise=0.004)
    result = reconstruct(cloud, PipelineConfig.preset("building"))
    plan = write_floorplan_dxf(result.mesh, tmp_path / "plan.dxf")
    text = plan.read_text()
    assert "BEMASSUNG" in text
    # The 4 m and 3 m walls must be dimensioned within a centimeter.
    import re

    dims = [float(m) for m in re.findall(r'\n1\n(\d+\.\d{3})\n', text)]
    assert any(abs(v - 4.0) < 0.02 for v in dims), dims
    assert any(abs(v - 3.0) < 0.02 for v in dims), dims
