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


def test_cli_analyze_gear_mesh(tmp_path, capsys):
    """Gears in the analysis mesh carry true involute tooth geometry."""
    from tests.synthetic import make_gear_scan

    src = write_point_cloud(make_gear_scan(), tmp_path / "gear.ply")
    mesh_path = tmp_path / "gear.glb"
    code = main(
        ["analyze", str(src), "-o", str(tmp_path / "g.json"),
         "--mesh", str(mesh_path), "--no-steel"]
    )
    assert code == 0
    # An involute gear body has far more geometry than the old stand-in tube.
    assert mesh_path.stat().st_size > 20000
    report = json.loads((tmp_path / "g.json").read_text())
    assert report["gears"][0]["teeth"] == 24


def test_cli_analyze_steel_mesh(tmp_path, capsys):
    """Steel members are exported as extruded catalog profiles."""
    from tests.synthetic import make_ipe_beam_scan

    src = write_point_cloud(make_ipe_beam_scan(), tmp_path / "beam.ply")
    mesh_path = tmp_path / "beam.glb"
    code = main(["analyze", str(src), "-o", str(tmp_path / "s.json"), "--mesh", str(mesh_path)])
    assert code == 0
    assert mesh_path.stat().st_size > 0
    assert "IPE 200" in capsys.readouterr().out


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


def test_cli_sheetmetal(tmp_path, capsys):
    from tests.synthetic import make_welded_tank_scan

    src = write_point_cloud(make_welded_tank_scan(), tmp_path / "tank.ply")
    report_path = tmp_path / "bleche.json"
    dxf_path = tmp_path / "zuschnitt.dxf"
    code = main(
        ["sheetmetal", str(src), "-o", str(report_path), "--dxf", str(dxf_path)]
    )
    assert code == 0
    report = json.loads(report_path.read_text())
    assert report["totals"]["plate_count"] == 5
    assert report["totals"]["weld_length"] > 0
    assert dxf_path.read_text().count("POLYLINE") == 5
    out = capsys.readouterr().out
    assert "plate 0:" in out and "totals:" in out


def test_cli_error_on_missing_file(tmp_path, capsys):
    assert main(["info", str(tmp_path / "nope.ply")]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_version(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_cli_texture_without_colors_skipped(tmp_path, capsys):
    """--texture on a colorless cloud writes the model untextured + a note."""
    src = write_point_cloud(make_box_scan(density=700, noise=0.004), tmp_path / "scan.ply")
    out = tmp_path / "model.glb"
    assert main(["reconstruct", str(src), "-o", str(out), "--texture", "--seed", "1"]) == 0
    assert out.stat().st_size > 0
    assert "no RGB colors" in capsys.readouterr().out


def test_cli_no_args_prints_help(capsys):
    """Plain `scantobim` (not frozen) prints the help instead of a stack trace."""
    assert main([]) == 2
    out = capsys.readouterr().out
    assert "reconstruct" in out and "analyze" in out


def _feed_input(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


def test_cli_wizard_with_files(tmp_path, capsys, monkeypatch):
    """`scantobim wizard scan.ply` runs the guided console reconstruction."""
    src = write_point_cloud(make_box_scan(density=700, noise=0.004), tmp_path / "scan.ply")
    _feed_input(monkeypatch, ["", ""])  # format: HTML (default), scene: default
    assert main(["wizard", str(src)]) == 0
    html = tmp_path / "scan_modell.html"
    assert html.stat().st_size > 5000
    report = json.loads((tmp_path / "scan_bericht.json").read_text())
    assert report["planes"] == 6
    assert "Fertig!" in capsys.readouterr().out


def test_cli_wizard_prompts_for_file(tmp_path, capsys, monkeypatch):
    """Wizard without arguments: welcome screen, prompt for a file."""
    src = write_point_cloud(make_box_scan(density=700, noise=0.004), tmp_path / "scan.ply")
    _feed_input(
        monkeypatch,
        [f'"{src}"',  # dragged into the window (Windows quotes paths)
         "",          # no more files
         "2",         # STEP output
         "1"],        # scene: building
    )
    assert main(["wizard"]) == 0
    stp = tmp_path / "scan_modell.stp"
    assert stp.read_text().startswith("ISO-10303-21;")
    assert "Konsolen-Modus" in capsys.readouterr().out


def test_cli_wizard_rejects_bad_paths(tmp_path, capsys, monkeypatch):
    """Nonexistent paths and unknown formats re-prompt instead of crashing."""
    src = write_point_cloud(make_box_scan(density=700, noise=0.004), tmp_path / "scan.ply")
    bogus = tmp_path / "notes.docx"
    bogus.write_text("x")
    _feed_input(
        monkeypatch,
        [str(tmp_path / "missing.ply"), str(bogus), str(src), "", "4", ""],
    )
    assert main(["wizard"]) == 0
    assert (tmp_path / "scan_modell.glb").stat().st_size > 0
    out = capsys.readouterr().out
    assert "nicht gefunden" in out and "kein unterstuetztes" in out


def test_cli_frozen_double_click_opens_gui(tmp_path, monkeypatch):
    """No arguments + frozen exe → the GUI is launched."""
    import sys as _sys

    import scantobim.cli as cli

    calls = {}

    def fake_gui(files, port=8317, open_browser=True):
        calls["files"] = files
        return 0

    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    monkeypatch.setattr(cli, "_launch_gui", fake_gui)
    assert main([]) == 0
    assert calls["files"] == []


def test_cli_file_args_open_gui_preloaded(tmp_path, monkeypatch):
    """Dragging clouds onto the exe preloads them into the GUI."""
    import scantobim.cli as cli

    src = write_point_cloud(make_box_scan(density=200), tmp_path / "scan.ply")
    calls = {}
    monkeypatch.setattr(
        cli, "_launch_gui", lambda files, **kw: calls.setdefault("files", files) and 0 or 0
    )
    assert main([str(src)]) == 0
    assert calls["files"] == [src]
