"""HTTP backend of the local GUI.

Endpoints (all bound to 127.0.0.1 only):

* ``GET  /``                    — the app page
* ``GET  /api/meta``            — version, preloaded files (drag & drop onto exe)
* ``POST /api/upload``          — raw file body + X-Filename header → server path
* ``POST /api/addpath``         — register an existing local path (big scans
                                  need no copy — the server runs on this machine)
* ``POST /api/run``             — start a job: {mode, files, options}
* ``GET  /api/status?job=ID``   — state, log lines, outputs, summary
* ``GET  /api/view?job=ID``     — the job's 3D viewer HTML (iframe)
* ``GET  /api/output?job=ID&name=F`` — download a result file
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import urllib.parse
import webbrowser
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scantobim import __version__

_CLOUD_EXTS = {".las", ".laz", ".ply", ".pcd", ".e57", ".xyz", ".pts", ".txt", ".csv", ".asc"}

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".json": "application/json",
    ".glb": "model/gltf-binary",
    ".obj": "text/plain; charset=utf-8",
    ".stl": "model/stl",
    ".stp": "application/step",
    ".ifc": "application/x-step",
    ".dxf": "image/vnd.dxf",
    ".ply": "application/octet-stream",
}


# User profiles persist across sessions in the home directory.
_PROFILE_FILE = Path.home() / ".scantobim" / "profiles.json"

# Advanced parameters the GUI may override (whitelist with converters).
_ADVANCED_FIELDS = {
    "voxel_size": float,
    "distance_factor": float,
    "min_inlier_ratio": float,
    "max_planes": int,
    "ortho_tol_deg": float,
    "parallel_tol_deg": float,
    "ghost_offset_tol": float,
    "min_opening_factor": float,
}


def _load_profiles() -> dict:
    try:
        return json.loads(_PROFILE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_profiles(profiles: dict) -> None:
    _PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PROFILE_FILE.write_text(
        json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _apply_advanced(cfg, advanced: dict | None) -> dict:
    """Apply whitelisted advanced overrides to the config; returns them typed."""
    applied = {}
    for key, value in (advanced or {}).items():
        conv = _ADVANCED_FIELDS.get(key)
        if conv is None or value in (None, ""):
            continue
        try:
            typed = conv(value)
        except (TypeError, ValueError):
            continue
        setattr(cfg, key, typed)
        applied[key] = typed
    return applied


class _LogWriter(io.TextIOBase):
    """Line-buffered stdout replacement feeding the job log."""

    def __init__(self, job: dict):
        self.job = job
        self.buf = ""

    def write(self, s: str) -> int:  # noqa: D102
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.job["log"].append(line.rstrip())
        return len(s)


class GuiState:
    """Shared state: uploads, jobs, preloaded files."""

    def __init__(self, initial_files: list[Path] | None = None):
        self.base = Path(tempfile.mkdtemp(prefix="scantobim-gui-"))
        (self.base / "uploads").mkdir()
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.counter = 0
        self.busy = False
        self.initial_files = [Path(f).resolve() for f in (initial_files or [])]

    def new_job(self, mode: str) -> dict:
        with self.lock:
            self.counter += 1
            job_id = f"job{self.counter}"
            job_dir = self.base / job_id
            job_dir.mkdir()
            job = {
                "id": job_id,
                "dir": str(job_dir),
                "mode": mode,
                "state": "running",
                "log": [],
                "summary": None,
                "error": None,
            }
            self.jobs[job_id] = job
            return job


# --------------------------------------------------------------- job runners

def _run_reconstruct(files: list[Path], opts: dict, outdir: Path) -> dict:
    import numpy as np

    from scantobim.cli import _json_default, _read_inputs
    from scantobim.core.pipeline import PipelineConfig, reconstruct
    from scantobim.io.writers import write_mesh

    # A SLAM project folder takes the dedicated import path (photo
    # projection + trajectory) and shares the CLI implementation.
    if len(files) == 1 and files[0].is_dir():
        from scantobim.cli import _cmd_project

        advanced = {}
        from scantobim.core.pipeline import PipelineConfig as _Cfg

        advanced = _apply_advanced(_Cfg(), opts.get("advanced"))  # type-check only
        ns = Namespace(
            directory=files[0],
            output=outdir / "modell.html",
            preset=opts.get("preset", "building"),
            source=opts.get("source", "slam"),
            advanced=advanced,
            watertight=bool(opts.get("watertight")),
            align=bool(opts.get("align")),
            no_photos=False,
            no_texture=not opts.get("texture", True),
            texel=None,
            report=outdir / "bericht.json",
            deviation=(outdir / "abweichung.ply") if opts.get("deviation") else None,
            tolerance=float(opts.get("tolerance") or 0.005),
            views=(outdir / "ansichten") if opts.get("views") else None,
            report_html=(outdir / "pruefbericht.html")
            if opts.get("report_html") else None,
            max_points=int(opts.get("max_points") or 40_000_000),
            seed=None,
        )
        code = _cmd_project(ns)
        if code != 0:
            raise RuntimeError("Projekt-Import fehlgeschlagen")
        rep = json.loads((outdir / "bericht.json").read_text())
        summary = {
            "Flächen": rep["planes"],
            "Exakte Ecken": rep.get("exact_corners", 0),
            "Restpunkte": rep["residual_points"],
            "Rechenzeit": f"{rep['runtime_seconds']} s",
        }
        if "volume" in rep.get("quantities", {}):
            summary["Volumen"] = f"{rep['quantities']['volume']:.3f} m³"
        if rep.get("trajectory_positions"):
            summary["Trajektorie"] = f"{rep['trajectory_positions']} Positionen"
        tex = rep.get("texture", {})
        if tex:
            label = tex.get("source", "keine")
            if tex.get("coverage") is not None:
                label += f" ({tex['coverage'] * 100:.0f}% Abdeckung)"
            summary["Textur"] = label
        dev = rep.get("deviation")
        if dev:
            fid = dev.get("fidelity") or dev
            summary["Modelltreue RMS"] = f"{fid['rms'] * 1000:.1f} mm"
            if dev.get("coverage") is not None:
                summary["Modellabdeckung"] = f"{dev['coverage'] * 100:.1f}% des Scans"
        return summary

    # Memory guard for the packaged app: huge scans are thinned block-wise.
    max_points = int(opts.get("max_points") or 40_000_000)
    cloud = _read_inputs(files, bool(opts.get("register")), max_points=max_points)
    preset = opts.get("preset", "building")
    source = opts.get("source", "standard")
    cfg = PipelineConfig.preset(preset if preset != "auto" else "building")
    from scantobim.core.pipeline import SOURCE_PROFILES, apply_source_profile

    apply_source_profile(cfg, source)
    advanced = _apply_advanced(cfg, opts.get("advanced"))
    if opts.get("watertight"):
        cfg.watertight = True
    if opts.get("cylinders"):
        cfg.cylinder_detection = True
    if opts.get("align"):
        cfg.align_axes = True

    print(f"Rekonstruktion läuft (Preset: {preset}, Quelle: {source}) …")
    if preset == "auto":
        from scantobim.core.autotune import auto_reconstruct

        result = auto_reconstruct(
            cloud,
            overrides={
                **SOURCE_PROFILES.get(source, {}),
                **advanced,
                "watertight": cfg.watertight,
                "align_axes": cfg.align_axes,
                "cylinder_detection": cfg.cylinder_detection,
            },
        )
    else:
        result = reconstruct(cloud, cfg)
    rep = result.report
    q = rep["quantities"]

    output_mesh = result.mesh
    texture_source = "keine"
    if opts.get("texture"):
        if cloud.colors is None:
            print("Hinweis: Punktwolke ohne Farbwerte — Textur übersprungen")
        else:
            from scantobim.core.texture import bake_texture_from_cloud

            print("Backe Fototextur aus den Punktfarben …")
            transform = None
            if "alignment" in rep:
                transform = np.array(rep["alignment"])
            output_mesh = bake_texture_from_cloud(result, cloud, transform=transform)
            texture_source = "punktfarben"

    write_mesh(output_mesh, outdir / "modell.html")
    print("geschrieben: modell.html")
    for fmt in opts.get("formats", []):
        if fmt == "step":
            from scantobim.io.step import write_step

            write_step(result.surfaces, outdir / "modell.stp")
            print("geschrieben: modell.stp (STEP AP214)")
        elif fmt == "ifc":
            from scantobim.io.ifc import write_ifc

            write_ifc(result.surfaces, outdir / "modell.ifc", storeys=rep.get("storeys"))
            print("geschrieben: modell.ifc (IFC4)")
        elif fmt == "glb":
            write_mesh(output_mesh, outdir / "modell.glb")
            print("geschrieben: modell.glb")
        elif fmt == "obj":
            write_mesh(output_mesh, outdir / "modell.obj")
            print("geschrieben: modell.obj")
        elif fmt == "dxf":
            from scantobim.io.dxf import write_floorplan_dxf

            write_floorplan_dxf(result.mesh, outdir / "grundriss.dxf")
            print("geschrieben: grundriss.dxf")
    if opts.get("deviation"):
        from scantobim.cli import _write_deviation

        _write_deviation(
            result, cloud, outdir / "abweichung.ply",
            float(opts.get("tolerance") or 0.005),
        )
    if opts.get("views"):
        from scantobim.cli import _write_views

        _write_views(output_mesh, outdir / "ansichten")
    if opts.get("report_html"):
        from scantobim.io.report_html import render_report_html

        render_report_html(
            rep, outdir / "pruefbericht.html",
            title=f"Prüfbericht — {files[0].stem}",
            views_dir=(outdir / "ansichten") if opts.get("views") else None,
        )
        print("geschrieben: pruefbericht.html")
    (outdir / "bericht.json").write_text(
        json.dumps(rep, indent=2, default=_json_default)
    )

    openings = sum(s.get("openings", 0) for s in rep["surfaces"])
    summary = {
        "Flächen": rep["planes"],
        "Exakte Ecken": rep["exact_corners"],
        "Öffnungen": openings,
        "Exakt rechte Winkel": rep["plane_angles"].get("exactly_orthogonal", 0),
    }
    if rep.get("cylinders"):
        summary["Zylinder (Stützen/Rohre)"] = len(rep["cylinders"])
    if "volume" in q:
        sigma = q.get("volume_sigma")
        summary["Volumen"] = f"{q['volume']:.3f} m³" + (
            f" ± {sigma:.3f}" if sigma else ""
        )
    for cls, n in q.get("surface_count_by_class", {}).items():
        summary[f"Bauteile: {cls}"] = n
    summary["Textur"] = texture_source
    dev = rep.get("deviation")
    if dev:
        fid = dev.get("fidelity") or dev
        summary["Modelltreue RMS"] = f"{fid['rms'] * 1000:.1f} mm"
        summary["Modelltreue P95"] = f"{fid['p95'] * 1000:.1f} mm"
        if dev.get("coverage") is not None:
            summary["Modellabdeckung"] = f"{dev['coverage'] * 100:.1f}% des Scans"
        summary["Innerhalb Toleranz (modellnah)"] = (
            f"{fid.get('within_tolerance', dev['within_tolerance']) * 100:.1f}% "
            f"(±{dev['tolerance'] * 1000:.0f} mm)"
        )
    summary["Dreiecke"] = rep["mesh"]["triangles"]
    summary["Restpunkte"] = rep["residual_points"]
    summary["Rechenzeit"] = f"{rep['runtime_seconds']} s"
    return summary


def _run_analyze(files: list[Path], opts: dict, outdir: Path) -> dict:
    from scantobim.cli import _cmd_analyze

    ns = Namespace(
        input=files,
        output=outdir / "analyse.json",
        mesh=outdir / "modell.html",
        dist=None,
        no_steel=False,
    )
    code = _cmd_analyze(ns)
    if code != 0:
        raise RuntimeError("Analyse fehlgeschlagen")
    rep = json.loads((outdir / "analyse.json").read_text())
    summary = {
        "Zylinder": len(rep.get("cylinders", [])),
        "Wellen": len(rep.get("shafts", [])),
        "Zahnräder": len(rep.get("gears", [])),
        "Getriebestufen": len(rep.get("gear_stages", [])),
        "Kegel": len(rep.get("cones", [])),
        "Stahlprofile": len(rep.get("steel_members", [])),
        "Anschlüsse": len(rep.get("connections", [])),
    }
    for g in rep.get("gears", []):
        summary[f"Zahnrad z={g['teeth']}"] = (
            f"Modul {g['module'] * 1000:.2f} mm, da {g['tip_diameter'] * 1000:.1f} mm"
        )
    for m in rep.get("steel_members", []):
        summary[f"Profil {m['profile']}"] = f"L = {m['length']:.2f} m"
    return summary


def _run_sheetmetal(files: list[Path], opts: dict, outdir: Path) -> dict:
    from scantobim.cli import _cmd_sheetmetal

    unfold = bool(opts.get("unfold"))
    ns = Namespace(
        input=files,
        output=outdir / "bleche.json",
        dxf=outdir / ("abwicklung.dxf" if unfold else "zuschnitt.dxf"),
        dist=None,
        max_thickness=0.05,
        unfold=unfold,
        k_factor=0.44,
    )
    code = _cmd_sheetmetal(ns)
    if code != 0:
        raise RuntimeError("Blechanalyse fehlgeschlagen")
    rep = json.loads((outdir / "bleche.json").read_text())
    tot = rep["totals"]
    summary = {
        "Bleche": tot["plate_count"],
        "Gewicht": f"{tot['weight_kg']:.1f} kg",
        "Schweißnahtlänge": f"{tot['weld_length'] * 1000:.0f} mm",
    }
    if rep.get("flat_parts"):
        summary["Abwicklungen"] = len(rep["flat_parts"])
    for p in rep["plates"]:
        t = p["thickness_catalog"] or p["thickness_measured"]
        summary[f"Blech {p['id']}"] = (
            f"t={t * 1000:.1f} mm, "
            f"{p['size'][0] * 1000:.0f} × {p['size'][1] * 1000:.0f} mm"
        )
    return summary


def _run_compare(files: list[Path], opts: dict, outdir: Path) -> dict:
    from scantobim.cli import _cmd_compare

    if len(files) != 2:
        raise ValueError(
            "Epochen-Vergleich braucht genau zwei Scans (alt, dann neu)"
        )
    ns = Namespace(
        reference=files[0],
        current=files[1],
        output=outdir / "verformung.json",
        heatmap=outdir / "verformung.ply",
        no_register=False,
        tolerance=0.005,
        max_points=40_000_000,
    )
    code = _cmd_compare(ns)
    if code != 0:
        raise RuntimeError("Epochenvergleich fehlgeschlagen")
    rep = json.loads((outdir / "verformung.json").read_text())
    return {
        "Registrierung": f"{rep['registration_shift'] * 1000:.1f} mm Versatz entfernt",
        "Verformung RMS": f"{rep['rms'] * 1000:.1f} mm",
        "Verformung P95": f"{rep['p95'] * 1000:.1f} mm",
        "Maximum": f"{rep['max'] * 1000:.1f} mm",
        "Innerhalb Toleranz": (
            f"{rep['within_tolerance'] * 100:.1f}% (±{rep['tolerance'] * 1000:.0f} mm)"
        ),
        "Punkte": rep["points"],
    }


def _run_bridge(files: list[Path], opts: dict, outdir: Path) -> dict:
    from scantobim.cli import _cmd_bridge

    ns = Namespace(input=files, output=outdir / "bauwerk.json", mesh=outdir / "modell.html")
    code = _cmd_bridge(ns)
    if code != 0:
        raise RuntimeError("Brückenanalyse fehlgeschlagen")
    rep = json.loads((outdir / "bauwerk.json").read_text())
    deck = rep["deck"]
    summary = {
        "Brückentyp": rep["bridge_type"],
        "Überbau": f"{deck['length']:.1f} × {deck['width']:.1f} m",
        "OK Fahrbahn": f"{deck['elevation']:.2f} m",
        "Felder": " + ".join(f"{s:.1f}" for s in rep["spans"]) + " m",
        "Pfeiler": len(rep["piers"]),
        "Widerlager": rep["abutments"],
        "Pylone": rep["pylons"],
        "Seile": len(rep["cables"]),
    }
    if rep.get("arch"):
        summary["Bogen"] = (
            f"R = {rep['arch']['radius']:.1f} m, Stich {rep['arch']['rise']:.1f} m"
        )
    return summary


_RUNNERS = {
    "reconstruct": _run_reconstruct,
    "analyze": _run_analyze,
    "sheetmetal": _run_sheetmetal,
    "bridge": _run_bridge,
    "compare": _run_compare,
}


def _execute_job(state: GuiState, job: dict, files: list[Path], opts: dict) -> None:
    writer = _LogWriter(job)
    try:
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            summary = _RUNNERS[job["mode"]](files, opts, Path(job["dir"]))
        job["summary"] = summary
        job["state"] = "done"
    except Exception as exc:  # noqa: BLE001 — surfaced in the GUI
        job["log"].append(f"FEHLER: {exc}")
        job["error"] = str(exc)
        job["state"] = "error"
    finally:
        state.busy = False


# ------------------------------------------------------------------ handler

def _make_handler(state: GuiState):
    from scantobim.gui.page import render_page

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence request logging
            pass

        # ---- helpers ----
        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _query(self) -> dict:
            q = urllib.parse.urlparse(self.path).query
            return {k: v[0] for k, v in urllib.parse.parse_qs(q).items()}

        def _job_outputs(self, job: dict) -> list[dict]:
            base = Path(job["dir"])
            out = []
            for f in sorted(base.rglob("*")):
                if f.is_file():
                    out.append(
                        {"name": str(f.relative_to(base)), "size": f.stat().st_size}
                    )
            return out

        # ---- GET ----
        def do_GET(self):  # noqa: N802
            route = urllib.parse.urlparse(self.path).path
            if route == "/":
                self._send(200, render_page().encode(), "text/html; charset=utf-8")
            elif route == "/api/meta":
                files = [
                    {"path": str(f), "name": f.name, "size": f.stat().st_size}
                    for f in state.initial_files
                    if f.exists()
                ]
                self._json({"version": __version__, "initial_files": files})
            elif route == "/api/profiles":
                self._json({"profiles": _load_profiles()})
            elif route == "/api/status":
                job = state.jobs.get(self._query().get("job", ""))
                if job is None:
                    self._json({"error": "unbekannter Job"}, 404)
                    return
                auto_winner = None
                bericht = Path(job["dir"]) / "bericht.json"
                if job["state"] == "done" and bericht.exists():
                    try:
                        auto_winner = json.loads(bericht.read_text()).get(
                            "auto_tuning_winner_config"
                        )
                    except (OSError, ValueError):
                        pass
                self._json(
                    {
                        "state": job["state"],
                        "log": job["log"],
                        "summary": job["summary"],
                        "error": job["error"],
                        "outputs": self._job_outputs(job),
                        "has_viewer": (Path(job["dir"]) / "modell.html").exists(),
                        "auto_winner": auto_winner,
                    }
                )
            elif route == "/api/view":
                job = state.jobs.get(self._query().get("job", ""))
                viewer = Path(job["dir"]) / "modell.html" if job else None
                if viewer is None or not viewer.exists():
                    self._send(404, b"kein Modell", "text/plain")
                    return
                self._send(200, viewer.read_bytes(), "text/html; charset=utf-8")
            elif route == "/api/output":
                q = self._query()
                job = state.jobs.get(q.get("job", ""))
                target = None
                if job is not None:
                    base = Path(job["dir"]).resolve()
                    candidate = (base / q.get("name", "")).resolve()
                    if candidate.is_file() and candidate.is_relative_to(base):
                        target = candidate
                if target is None:
                    self._send(404, b"nicht gefunden", "text/plain")
                    return
                ctype = _CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
                self._send(
                    200, target.read_bytes(), ctype,
                    {"Content-Disposition": f'attachment; filename="{target.name}"'},
                )
            else:
                self._send(404, b"not found", "text/plain")

        # ---- POST ----
        def do_POST(self):  # noqa: N802
            route = urllib.parse.urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""

            if route == "/api/upload":
                raw_name = self.headers.get("X-Filename", "scan.ply")
                name = Path(urllib.parse.unquote(raw_name)).name
                if Path(name).suffix.lower() not in _CLOUD_EXTS:
                    self._json({"error": f"kein Punktwolken-Format: {name}"}, 400)
                    return
                target = state.base / "uploads" / name
                stem, suffix, k = target.stem, target.suffix, 1
                while target.exists():
                    target = target.with_name(f"{stem}_{k}{suffix}")
                    k += 1
                target.write_bytes(body)
                self._json({"path": str(target), "name": target.name, "size": len(body)})
            elif route == "/api/profiles":
                try:
                    req = json.loads(body.decode())
                    name = str(req["name"]).strip()[:60]
                    settings = req["settings"]
                    assert name and isinstance(settings, dict)
                except Exception:
                    self._json({"error": "ungültige Anfrage"}, 400)
                    return
                profiles = _load_profiles()
                profiles[name] = settings
                try:
                    _save_profiles(profiles)
                except OSError as exc:
                    self._json({"error": f"Speichern fehlgeschlagen: {exc}"}, 500)
                    return
                self._json({"profiles": profiles})
            elif route == "/api/profiles/delete":
                try:
                    name = str(json.loads(body.decode())["name"])
                except Exception:
                    self._json({"error": "ungültige Anfrage"}, 400)
                    return
                profiles = _load_profiles()
                profiles.pop(name, None)
                try:
                    _save_profiles(profiles)
                except OSError as exc:
                    self._json({"error": f"Speichern fehlgeschlagen: {exc}"}, 500)
                    return
                self._json({"profiles": profiles})
            elif route == "/api/addpath":
                try:
                    p = Path(json.loads(body.decode())["path"].strip().strip('"'))
                except Exception:
                    self._json({"error": "ungültige Anfrage"}, 400)
                    return
                if p.is_dir():
                    # SLAM project folder (SHARE SLAM S20 & Co.).
                    from scantobim.io.project import scan_project_dir

                    project = scan_project_dir(p)
                    if project.cloud is None:
                        self._json(
                            {"error": f"keine Punktwolke im Ordner gefunden: {p}"}, 404
                        )
                        return
                    detail = [project.cloud.name]
                    if project.cloud_points:
                        mio = project.cloud_points / 1e6
                        detail.append(
                            f"{mio:.1f} Mio Punkte" if mio >= 1
                            else f"{project.cloud_points:,} Punkte"
                        )
                    if project.cloud_colored is True:
                        detail.append("mit Farben")
                    elif project.color_source is not None:
                        detail.append(f"Farben aus {project.color_source.name}")
                    elif project.cloud_colored is False:
                        detail.append("ohne Farben")
                    if project.images_dir is not None:
                        detail.append(f"{project.image_count} Fotos")
                    if project.colmap_model is not None:
                        detail.append("Kameraposen")
                    if project.trajectory is not None:
                        detail.append("Trajektorie")
                    self._json(
                        {
                            "path": str(p),
                            "name": f"📂 {p.name} ({', '.join(detail)})",
                            "size": project.cloud.stat().st_size,
                            "kind": "project",
                        }
                    )
                    return
                if not p.is_file():
                    self._json({"error": f"Datei nicht gefunden: {p}"}, 404)
                    return
                if p.suffix.lower() not in _CLOUD_EXTS:
                    self._json({"error": f"kein Punktwolken-Format: {p.suffix}"}, 400)
                    return
                self._json({"path": str(p), "name": p.name, "size": p.stat().st_size})
            elif route == "/api/run":
                try:
                    req = json.loads(body.decode())
                    mode = req["mode"]
                    files = [Path(f) for f in req["files"]]
                    opts = req.get("options", {})
                except Exception:
                    self._json({"error": "ungültige Anfrage"}, 400)
                    return
                if mode not in _RUNNERS:
                    self._json({"error": f"unbekannter Modus: {mode}"}, 400)
                    return
                if not files or not all(f.exists() for f in files):
                    self._json({"error": "Eingabedatei fehlt"}, 400)
                    return
                # Project FOLDERS take a dedicated import path — they cannot
                # be mixed with loose cloud files in one run (the folder
                # would be misread as a cloud file).
                if any(f.is_dir() for f in files):
                    if mode != "reconstruct":
                        self._json(
                            {
                                "error": "Projektordner werden nur im Modus "
                                "„Rekonstruktion“ unterstützt."
                            },
                            400,
                        )
                        return
                    if len(files) > 1:
                        self._json(
                            {
                                "error": "Bitte entweder Punktwolken-Dateien ODER "
                                "genau EINEN Projektordner starten — nicht beides "
                                "zusammen. Nicht benötigte Einträge mit ✕ aus der "
                                "Liste entfernen (der Projektordner enthält seine "
                                "Punktwolke bereits)."
                            },
                            400,
                        )
                        return
                if state.busy:
                    self._json({"error": "es läuft bereits ein Auftrag"}, 409)
                    return
                state.busy = True
                job = state.new_job(mode)
                threading.Thread(
                    target=_execute_job, args=(state, job, files, opts), daemon=True
                ).start()
                self._json({"job": job["id"]})
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


# ------------------------------------------------------------------- launch

def create_server(
    port: int = 8317, initial_files: list[Path] | None = None
) -> tuple[ThreadingHTTPServer, GuiState]:
    """Bind the GUI server on 127.0.0.1 (fallback ports if taken)."""
    state = GuiState(initial_files)
    handler = _make_handler(state)
    last_error: Exception | None = None
    for candidate in [port] + [port + i for i in range(1, 20)] + [0]:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), handler)
            server.daemon_threads = True
            return server, state
        except OSError as exc:
            last_error = exc
    raise last_error  # pragma: no cover


def run_gui(
    port: int = 8317,
    initial_files: list[Path] | None = None,
    open_browser: bool = True,
) -> int:
    """Start the GUI and block until Ctrl+C."""
    server, _state = create_server(port, initial_files)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print("=" * 62)
    print(f" ScanToBIM {__version__} — grafische Oberfläche")
    print("=" * 62)
    print(f"\n  Läuft auf:  {url}")
    print("  Dieses Fenster ist das Protokoll — einfach offen lassen.")
    print("  Beenden: Strg+C oder Fenster schließen.\n")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # pragma: no cover
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        server.server_close()
    return 0
