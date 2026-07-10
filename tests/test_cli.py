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


def test_cli_error_on_missing_file(tmp_path, capsys):
    assert main(["info", str(tmp_path / "nope.ply")]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_version(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
