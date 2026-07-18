"""GPU deep diagnosis: composed sections, file writing, CLI command."""

from pathlib import Path

from scantobim.core.gpudiag import build_gpu_diagnosis, write_gpu_diagnosis


def test_build_diagnosis_contains_all_sections():
    text = build_gpu_diagnosis()
    assert "ScanToBIM GPU-Diagnose" in text
    assert "1. Umgebungsvariablen" in text
    assert "2. Gebündelte CUDA-Laufzeit" in text
    assert "3. DLL-Ladetests" in text
    assert "4. CuPy-Import Schritt für Schritt" in text
    assert "5. nvidia-smi" in text
    assert "Ende der Diagnose." in text
    # In this CUDA-less environment the CuPy step must report its failure
    # (or, with CuPy installed, the device query outcome) — never crash.
    assert "cupy" in text.lower()


def test_write_diagnosis_to_directory(tmp_path):
    path = write_gpu_diagnosis(tmp_path)
    assert path == tmp_path / "gpu_diagnose.txt"
    assert "GPU-Diagnose" in path.read_text(encoding="utf-8")


def test_cli_gpu_command(tmp_path, capsys):
    from scantobim.cli import main

    code = main(["gpu", "-o", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "GPU-Diagnose" in out
    assert (tmp_path / "gpu_diagnose.txt").exists()


def test_gpu_failure_writes_diagnosis_next_to_output(tmp_path, monkeypatch):
    """A failing GPU build drops gpu_diagnose.txt beside the results."""
    from scantobim import cli
    from scantobim.core import accel

    monkeypatch.setattr(accel, "_name", None)
    monkeypatch.setattr(accel, "_error", "ImportError: DLL load failed [Test]")
    monkeypatch.setattr(accel, "_checked", True)
    rep: dict = {}
    cli._print_gpu_status(rep, diag_dir=tmp_path)
    assert (tmp_path / "gpu_diagnose.txt").exists()
    assert rep["gpu_diagnose"] == str(tmp_path / "gpu_diagnose.txt")
    assert Path(rep["gpu_diagnose"]).read_text(encoding="utf-8").startswith(
        "ScanToBIM GPU-Diagnose"
    )
