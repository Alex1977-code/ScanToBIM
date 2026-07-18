"""Deep GPU/CUDA startup diagnosis.

When the GPU build cannot start CUDA, this module writes a
``gpu_diagnose.txt`` that tests every link in the chain separately —
environment variables, the bundled runtime files, individual DLL loads
(including the NVIDIA driver itself), the CuPy import with full traceback,
and ``nvidia-smi``. One look at the file pinpoints the failing link
instead of guessing from a one-line error.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path


def _section(lines: list[str], title: str) -> None:
    lines += ["", f"[{title}]", "-" * (len(title) + 4)]


def build_gpu_diagnosis() -> str:
    """Compose the full diagnosis text (never raises)."""
    from scantobim import __version__
    from scantobim.core import accel

    lines = ["ScanToBIM GPU-Diagnose", "=" * 40]
    lines.append(f"Zeit:          {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Version:       ScanToBIM {__version__}")
    frozen = "PyInstaller-Exe" if getattr(sys, "frozen", False) else "Quellcode"
    lines.append(f"Python:        {sys.version.split()[0]} ({frozen})")
    lines.append(f"Programm:      {sys.executable}")
    meipass = getattr(sys, "_MEIPASS", None)
    lines.append(f"Paketordner:   {meipass or '—'}")
    try:
        lines.append(f"Arbeitsordner: {os.getcwd()}")
    except OSError:
        pass

    _section(lines, "1. Umgebungsvariablen")
    found = False
    for var in sorted(os.environ):
        if "CUDA" in var.upper():
            lines.append(f"{var} = {os.environ[var]}")
            found = True
    if not found:
        lines.append("(keine CUDA_*-Variablen gesetzt)")
    path_hits = [
        p for p in os.environ.get("PATH", "").split(os.pathsep)
        if "cuda" in p.lower() or "nvidia" in p.lower() or "_mei" in p.lower()
    ]
    lines.append("PATH-Einträge mit cuda/nvidia/_MEI:")
    lines += [f"  {p}" for p in path_hits] or ["  (keine)"]

    _section(lines, "2. Gebündelte CUDA-Laufzeit")
    try:
        dirs = accel._cuda_library_dirs()
        lines.append(f"Laufzeit-Ordner gefunden: {len(dirs)}")
        for d in dirs:
            try:
                dlls = sorted(
                    f"{f.name} ({f.stat().st_size // 1024} KB)"
                    for f in d.iterdir() if f.suffix.lower() == ".dll"
                )
            except OSError as exc:
                dlls = [f"<Lesefehler: {exc}>"]
            lines.append(f"  {d}")
            lines += [f"    {x}" for x in dlls] or ["    (keine DLLs)"]
    except Exception:
        lines.append(traceback.format_exc())
    roots = []
    if meipass:
        roots.append(("Paketordner", Path(meipass)))
    try:
        roots.append(("Programmordner", Path(sys.executable).resolve().parent))
    except OSError:
        pass
    for label, root in roots:
        try:
            hits = sorted(
                f.name for f in root.glob("*.dll")
                if any(k in f.name.lower() for k in ("cudart", "nvrtc"))
            )
            lines.append(
                f"CUDA-DLLs direkt im {label} ({root}): "
                f"{', '.join(hits) or 'KEINE'}"
            )
        except OSError as exc:
            lines.append(f"{label} nicht lesbar: {exc}")

    _section(lines, "3. DLL-Ladetests")
    if os.name == "nt":
        import ctypes

        def _try_load(target) -> str:
            try:
                ctypes.WinDLL(str(target))
                return "OK"
            except OSError as exc:
                return f"FEHLER — {exc}"

        res = _try_load("nvcuda.dll")
        lines.append(f"nvcuda.dll (NVIDIA-TREIBER): {res}")
        if res == "OK":
            try:
                drv = ctypes.WinDLL("nvcuda.dll")
                ver = ctypes.c_int(0)
                drv.cuDriverGetVersion(ctypes.byref(ver))
                lines.append(
                    "  Treiber unterstützt CUDA bis: "
                    f"{ver.value // 1000}.{ver.value % 1000 // 10}"
                )
            except Exception as exc:  # noqa: BLE001
                lines.append(f"  Versionsabfrage fehlgeschlagen: {exc}")
        else:
            lines.append(
                "  → OHNE nvcuda.dll gibt es keinen NVIDIA-Treiber in diesem "
                "Windows — Treiber installieren/aktualisieren."
            )
        for dll in ("cudart64_12.dll", "nvrtc64_120_0.dll"):
            lines.append(f"{dll} (Suche über PATH/DLL-Pfade): {_try_load(dll)}")
        try:
            from scantobim.core.accel import _cuda_library_dirs

            for d in _cuda_library_dirs():
                for f in sorted(d.glob("*.dll")):
                    lines.append(f"{f} (direkter Pfad): {_try_load(f)}")
        except Exception:
            lines.append(traceback.format_exc())
    else:
        lines.append("(DLL-Ladetests nur unter Windows)")

    _section(lines, "4. CuPy-Import Schritt für Schritt")
    try:
        accel._register_cuda_dirs()
        lines.append(
            f"CUDA_PATH nach Registrierung: {os.environ.get('CUDA_PATH', '—')}"
        )
    except Exception:
        lines.append(traceback.format_exc())
    try:
        import cupy

        lines.append(f"import cupy: OK (CuPy {cupy.__version__})")
        try:
            n = cupy.cuda.runtime.getDeviceCount()
            lines.append(f"getDeviceCount(): {n}")
            for i in range(n):
                props = cupy.cuda.runtime.getDeviceProperties(i)
                raw = props.get("name", b"?")
                name = raw.decode() if isinstance(raw, bytes) else str(raw)
                mem = props.get("totalGlobalMem", 0) / 1e9
                lines.append(f"  GPU {i}: {name} ({mem:.1f} GB VRAM)")
            if n:
                lines.append(
                    f"Rechentest arange(8).sum(): {float(cupy.arange(8).sum())} "
                    "(erwartet: 28.0)"
                )
        except Exception:
            lines.append("Geräteabfrage FEHLGESCHLAGEN:")
            lines.append(traceback.format_exc())
    except BaseException:  # noqa: BLE001 — auch ein harter Ladefehler soll ins Log
        lines.append("import cupy FEHLGESCHLAGEN:")
        lines.append(traceback.format_exc())

    _section(lines, "5. nvidia-smi (Treiber-Werkzeug)")
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=15
        )
        lines.append(out.stdout.strip() or out.stderr.strip() or "(keine Ausgabe)")
    except FileNotFoundError:
        lines.append(
            "nvidia-smi nicht gefunden — spricht für fehlenden NVIDIA-Treiber."
        )
    except Exception as exc:  # noqa: BLE001
        lines.append(f"nvidia-smi Fehler: {exc}")

    lines += ["", "Ende der Diagnose.", ""]
    return "\n".join(lines)


def write_gpu_diagnosis(directory: Path | str | None = None) -> Path | None:
    """Write ``gpu_diagnose.txt``; returns its path (None if nowhere writable)."""
    try:
        text = build_gpu_diagnosis()
    except Exception:  # noqa: BLE001 — diagnosis must never crash the run
        text = "GPU-Diagnose selbst fehlgeschlagen:\n" + traceback.format_exc()
    candidates = []
    if directory is not None:
        candidates.append(Path(directory))
    try:
        candidates.append(Path.cwd())
    except OSError:
        pass
    candidates.append(Path(tempfile.gettempdir()))
    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            path = cand / "gpu_diagnose.txt"
            path.write_text(text, encoding="utf-8")
            return path
        except OSError:
            continue
    return None
