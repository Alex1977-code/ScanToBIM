"""CLI smoke tests (info + reconstruct, exercised through main())."""

import json

import numpy as np

from scantobim.cli import main
from scantobim.io.readers import read_point_cloud
from scantobim.io.writers import write_point_cloud
from tests.synthetic import make_box_scan


def test_cli_info(tmp_path, capsys):
    cloud = make_box_scan(density=200, noise=0.003)
    src = write_point_cloud(cloud, tmp_path / "scan.ply")
    assert main(["info", str(src)]) == 0
    out = capsys.readouterr().out
    assert "points:" in out and "spacing:" in out


def test_cli_reconstruct(tmp_path, capsys):
    cloud = make_box_scan(density=700, noise=0.004)
    src = write_point_cloud(cloud, tmp_path / "scan.ply")
    out_mesh = tmp_path / "model.glb"
    out_report = tmp_path / "report.json"
    out_residual = tmp_path / "residual.ply"
    code = main(
        [
            "reconstruct", str(src),
            "-o", str(out_mesh),
            "--preset", "building",
            "--report", str(out_report),
            "--residual-out", str(out_residual),
            "--seed", "1",
        ]
    )
    assert code == 0
    assert out_mesh.stat().st_size > 0
    report = json.loads(out_report.read_text())
    assert report["planes"] == 6
    if out_residual.exists():
        residual = read_point_cloud(out_residual)
        assert len(residual) == report["residual_points"]


def test_cli_reconstruct_with_floorplan_align_html(tmp_path, capsys):
    cloud = make_box_scan(density=700, noise=0.004)
    src = write_point_cloud(cloud, tmp_path / "scan.ply")
    out_mesh = tmp_path / "model.html"
    plan = tmp_path / "plan.dxf"
    code = main(
        [
            "reconstruct", str(src),
            "-o", str(out_mesh),
            "--align",
            "--floorplan", str(plan),
            "--seed", "2",
        ]
    )
    assert code == 0
    assert out_mesh.stat().st_size > 5000
    assert plan.read_text().rstrip().endswith("EOF")
    out = capsys.readouterr().out
    assert "classes:" in out and "volume:" in out


def test_cli_register(tmp_path, capsys):
    ref = write_point_cloud(make_box_scan(density=400, seed=1), tmp_path / "a.ply")
    other = write_point_cloud(make_box_scan(density=400, seed=2), tmp_path / "b.ply")
    merged = tmp_path / "merged.ply"
    transforms = tmp_path / "t.json"
    code = main(
        ["register", str(ref), str(other), "-o", str(merged), "--transforms", str(transforms)]
    )
    assert code == 0
    got = read_point_cloud(merged)
    assert len(got) > 0
    data = json.loads(transforms.read_text())
    assert len(data) == 2
    assert np.array(data[str(other)]).shape == (4, 4)


def test_cli_reconstruct_multiple_inputs(tmp_path, capsys):
    """Two half-scans of the same box merge into one model."""
    cloud = make_box_scan(density=700, noise=0.004)
    half = len(cloud.points) // 2
    from scantobim.core.cloud import PointCloud

    a = write_point_cloud(PointCloud(points=cloud.points[:half]), tmp_path / "a.ply")
    b = write_point_cloud(PointCloud(points=cloud.points[half:]), tmp_path / "b.ply")
    out = tmp_path / "model.glb"
    report = tmp_path / "report.json"
    code = main(
        ["reconstruct", str(a), str(b), "-o", str(out), "--report", str(report), "--seed", "1"]
    )
    assert code == 0
    assert json.loads(report.read_text())["planes"] == 6
    assert "merged 2 clouds" in capsys.readouterr().out


def test_cli_reconstruct_step_output(tmp_path, capsys):
    cloud = make_box_scan(density=700, noise=0.004)
    src = write_point_cloud(cloud, tmp_path / "scan.ply")
    out = tmp_path / "model.stp"
    assert main(["reconstruct", str(src), "-o", str(out), "--seed", "1"]) == 0
    text = out.read_text()
    assert text.startswith("ISO-10303-21;")
    assert "MANIFOLD_SOLID_BREP" in text


def test_cli_reconstruct_ifc_output(tmp_path, capsys):
    cloud = make_box_scan(density=700, noise=0.004)
    src = write_point_cloud(cloud, tmp_path / "scan.ply")
    out = tmp_path / "model.ifc"
    assert main(["reconstruct", str(src), "-o", str(out), "--seed", "1"]) == 0
    assert "IFC4" in out.read_text()


def test_cli_analyze(tmp_path, capsys):
    from tests.synthetic import make_stepped_shaft_scan

    shaft = make_stepped_shaft_scan(steps=((0.030, 0.08), (0.050, 0.12)))
    src = write_point_cloud(shaft, tmp_path / "shaft.ply")
    report_path = tmp_path / "analysis.json"
    mesh_path = tmp_path / "primitives.glb"
    code = main(
        ["analyze", str(src), "-o", str(report_path), "--mesh", str(mesh_path), "--no-steel"]
    )
    assert code == 0
    report = json.loads(report_path.read_text())
    assert len(report["shafts"]) == 1
    assert len(report["shafts"][0]["steps"]) == 2
    assert mesh_path.stat().st_size > 0
    assert "shaft:" in capsys.readouterr().out


def test_cli_reconstruct_with_texture(tmp_path, capsys):
    cloud = make_box_scan(density=700, noise=0.004)
    rng = np.random.default_rng(0)
    from scantobim.core.cloud import PointCloud

    colored = PointCloud(
        points=cloud.points,
        colors=rng.integers(60, 200, (len(cloud.points), 3), dtype=np.uint8),
    )
    src = write_point_cloud(colored, tmp_path / "scan.ply")
    out = tmp_path / "model.glb"
    code = main(["reconstruct", str(src), "-o", str(out), "--texture", "--seed", "1"])
    assert code == 0
    assert "texture atlas:" in capsys.readouterr().out
    raw = out.read_bytes()
    assert b"image/png" in raw


def test_cli_error_on_missing_file(tmp_path, capsys):
    assert main(["info", str(tmp_path / "nope.ply")]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_version(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
