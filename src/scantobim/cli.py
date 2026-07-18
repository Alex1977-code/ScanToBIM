"""Command line interface.

Examples
--------
Inspect a scan::

    scantobim info scan.las

Reconstruct a clean-edged model::

    scantobim reconstruct scan.laz -o model.glb --preset indoor
    scantobim reconstruct scan.ply -o model.obj --voxel 0.02 --report report.json

Photos → point cloud (requires COLMAP)::

    scantobim photos ./fotos -o cloud.ply
    scantobim reconstruct cloud.ply -o model.glb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scantobim import __version__
from scantobim.core.pipeline import PipelineConfig, reconstruct
from scantobim.io.readers import read_point_cloud
from scantobim.io.writers import write_mesh, write_point_cloud

_CLOUD_EXTS = {".las", ".laz", ".ply", ".pcd", ".e57", ".xyz", ".pts", ".txt", ".csv", ".asc"}


def _is_frozen() -> bool:
    """True inside the packaged Windows executable (PyInstaller)."""
    return bool(getattr(sys, "frozen", False))


def _pause() -> None:
    """Keep the console window open — a double-clicked exe closes it instantly."""
    try:
        input("\nEnter druecken zum Beenden ... ")
    except (EOFError, KeyboardInterrupt):
        pass


def _launch_gui(files: list[Path], port: int = 8317, open_browser: bool = True) -> int:
    """Start the local GUI; on failure keep the error readable in the exe."""
    try:
        from scantobim.gui import run_gui

        return run_gui(port=port, initial_files=files, open_browser=open_browser)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # noqa: BLE001 — window must stay readable
        print(f"error: {exc}", file=sys.stderr)
        if _is_frozen():
            _pause()
        return 1


def main(argv: list[str] | None = None) -> int:
    # Windows consoles/pipes often use cp1252 — characters like '→' or '✓'
    # in the protokoll must degrade gracefully instead of crashing a print.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001 — cosmetic only
            pass
    if argv is None:
        argv = sys.argv[1:]

    # Double-clicked exe (no arguments): open the graphical interface.
    if not argv:
        if _is_frozen():
            return _launch_gui([])

    # Files dragged onto the exe (or `scantobim scan.laz` without a command):
    # open the GUI with those clouds preloaded.
    elif all(
        Path(a).suffix.lower() in _CLOUD_EXTS and Path(a).exists() for a in argv
    ):
        return _launch_gui([Path(a) for a in argv])

    parser = _build_parser()
    if not argv:
        parser.print_help()
        return 2
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        if _is_frozen():
            _pause()  # keep argparse errors readable in the double-click case
        raise
    return _dispatch(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scantobim",
        description=(
            "Reconstruct clean-edged 3D models from LiDAR point clouds "
            "(.las/.laz/.ply/.pcd/.e57/.xyz/.pts) and photos."
        ),
    )
    parser.add_argument("--version", action="version", version=f"scantobim {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="print statistics about a point cloud")
    p_info.add_argument("input", type=Path)
    p_info.add_argument("--max-points", type=int, default=None,
                        help="thin huge scans to at most this many points while "
                        "reading (bounded memory)")

    p_rec = sub.add_parser("reconstruct", help="point cloud(s) → clean-edged 3D model")
    p_rec.add_argument(
        "input", type=Path, nargs="+",
        help="input point cloud(s); multiple files are merged into one model",
    )
    p_rec.add_argument(
        "-o", "--output", type=Path, required=True,
        help="output model: .obj/.ply/.stl/.glb/.gltf (mesh), .html "
        "(interactive standalone viewer), .stp/.step (CAD B-rep, also the "
        "exchange route into HiCAD), .ifc (BIM building elements)",
    )
    p_rec.add_argument(
        "--register-inputs", action="store_true",
        help="ICP-register additional input clouds onto the first one before "
        "merging (default: assume shared coordinates)",
    )
    p_rec.add_argument(
        "--preset", default="building",
        choices=["auto", "building", "indoor", "object", "detail", "fast"],
        help="parameter preset (default: building; 'detail' keeps small "
        "structures; 'auto' tries several parameter sets and keeps the "
        "objectively best result — slower but self-optimizing)",
    )
    p_rec.add_argument(
        "--source", default="standard",
        choices=["standard", "slam", "tls", "drohne", "iphone"],
        help="sensor profile layered over the preset: slam (handheld, e.g. "
        "SHARE S20), tls (tripod laser scanner), drohne (photogrammetry/"
        "aerial), iphone (mobile LiDAR)",
    )
    p_rec.add_argument("--cylinders", action="store_true",
                       help="also reconstruct cylindrical members (columns, "
                       "pipes) from the residual (included in --preset detail)")
    p_rec.add_argument("--voxel", type=float, default=None,
                       help="voxel size for thinning in input units (default: auto, 0 = off)")
    p_rec.add_argument("--dist", type=float, default=None,
                       help="RANSAC plane distance threshold (default: auto from point spacing)")
    p_rec.add_argument("--max-planes", type=int, default=None, help="maximum number of planes")
    p_rec.add_argument("--no-regularize", action="store_true",
                       help="disable parallel/orthogonal angle snapping")
    p_rec.add_argument("--no-straighten", action="store_true",
                       help="disable boundary straightening")
    p_rec.add_argument("--no-color", action="store_true",
                       help="do not color surfaces in the output mesh")
    p_rec.add_argument("--texture", action="store_true",
                       help="bake a photo-realistic texture from the cloud's "
                       "RGB colors (photogrammetry / RGB scanner)")
    p_rec.add_argument("--texture-photos", type=Path, default=None, metavar="IMAGE_DIR",
                       help="project the original photos onto the model "
                       "(sharpest result; needs --colmap-model and Pillow)")
    p_rec.add_argument("--colmap-model", type=Path, default=None, metavar="MODEL_DIR",
                       help="COLMAP text model with camera poses "
                       "(cameras.txt/images.txt; `scantobim photos` writes it "
                       "to <work_dir>/model_txt)")
    p_rec.add_argument("--texel", type=float, default=None,
                       help="texture resolution: texel edge length in input "
                       "units (default: auto)")
    p_rec.add_argument("--watertight", action="store_true",
                       help="globally optimized watertight model (PolyFit): "
                       "closes scan shadows with the geometrically exact faces")
    p_rec.add_argument("--no-freeform", "--no-mesh", dest="freeform",
                       action="store_false",
                       help="kein Komplett-Mesh des Scans (Standard: an)")
    p_rec.add_argument("--no-structure", dest="structure", action="store_false",
                       help="nur Komplett-Mesh — keine Ebenen/Linien-Suche, "
                       "keine BIM-Auswertung")
    p_rec.add_argument("--ghost-tol", type=float, default=None, metavar="M",
                       help="merge registration ghosts (double walls) within "
                       "this offset in input units")
    p_rec.add_argument("--no-openings", action="store_true",
                       help="do not reconstruct window/door openings as holes")
    p_rec.add_argument("--align", action="store_true",
                       help="rotate dominant directions onto the X/Y/Z axes and "
                       "put the floor at Z=0 (transform is stored in the report)")
    p_rec.add_argument("--floorplan", type=Path, default=None,
                       help="additionally write a 2D floor plan as DXF")
    p_rec.add_argument("--floorplan-height", type=float, default=1.0,
                       help="slice height above the model base for --floorplan "
                       "(default: 1.0)")
    p_rec.add_argument("--report", type=Path, default=None,
                       help="write the quality report to this JSON file")
    p_rec.add_argument("--residual-out", type=Path, default=None,
                       help="write points not explained by any surface to this .ply")
    p_rec.add_argument("--trajectory", type=Path, default=None, metavar="TXT",
                       help="scanner trajectory (SLAM path, e.g. trajectory.txt): "
                       "orients normals towards the scanner for robust detection")
    p_rec.add_argument("--deviation", type=Path, default=None, metavar="PLY",
                       help="as-built QA: write the scan colored by signed "
                       "deviation from the model (blue-white-red) plus "
                       "statistics in the report")
    p_rec.add_argument("--tolerance", type=float, default=0.005,
                       help="deviation tolerance in input units for the QA "
                       "quote (default: 0.005)")
    p_rec.add_argument("--views", type=Path, default=None, metavar="DIR",
                       help="write true-to-scale orthographic views (N/E/S/W/top) "
                       "as PNG + world file into this directory")
    p_rec.add_argument("--max-points", type=int, default=None,
                       help="thin huge scans to at most this many points while "
                       "reading (block-wise, bounded memory)")
    p_rec.add_argument("--report-html", type=Path, default=None, metavar="HTML",
                       help="print-ready inspection report (A4): all measured "
                       "values, deviation statistics, views")
    p_rec.add_argument("--seed", type=int, default=None, help="RANSAC random seed")

    p_proj = sub.add_parser(
        "project",
        help="import a SLAM scanner project folder (SHARE SLAM S20, GeoSLAM, "
        "…): auto-detects point cloud, undistorted photos, COLMAP poses and "
        "trajectory, then reconstructs a photo-textured model",
    )
    p_proj.add_argument("directory", type=Path, help="the exported project folder")
    p_proj.add_argument("-o", "--output", type=Path, default=None,
                        help="output model (default: <folder>/scantobim_modell.html)")
    p_proj.add_argument("--preset", default="building",
                        choices=["auto", "building", "indoor", "object", "detail", "fast"])
    p_proj.add_argument("--source", default="slam",
                        choices=["standard", "slam", "tls", "drohne", "iphone"],
                        help="sensor profile (default: slam — project folders "
                        "come from SLAM scanners)")
    p_proj.add_argument("--watertight", action="store_true",
                        help="globally optimized watertight model")
    p_proj.add_argument("--no-freeform", "--no-mesh", dest="freeform",
                        action="store_false",
                        help="kein Komplett-Mesh des Scans (Standard: an)")
    p_proj.add_argument("--no-structure", dest="structure", action="store_false",
                        help="nur Komplett-Mesh — keine Ebenen/Linien-Suche, "
                        "keine BIM-Auswertung")
    p_proj.add_argument("--align", action="store_true",
                        help="axis-align the model, floor at Z=0")
    p_proj.add_argument("--no-photos", action="store_true",
                        help="skip photo projection (use point cloud colors)")
    p_proj.add_argument("--no-texture", action="store_true",
                        help="no texturing at all (plain surface colors)")
    p_proj.add_argument("--texel", type=float, default=None,
                        help="texture resolution in input units (default: auto)")
    p_proj.add_argument("--report", type=Path, default=None,
                        help="quality report JSON (default: next to the output)")
    p_proj.add_argument("--deviation", type=Path, default=None, metavar="PLY",
                        help="as-built QA: deviation-colored scan + statistics")
    p_proj.add_argument("--tolerance", type=float, default=0.005)
    p_proj.add_argument("--views", type=Path, default=None, metavar="DIR",
                        help="true-to-scale orthographic views as PNG")
    p_proj.add_argument("--detail-raster", dest="detail_raster", type=float,
                        default=0.02, metavar="M",
                        help="Raster des Detail-Mesh der Gebäuderegion in m "
                             "(Standard 0.02 = 2 cm; 0 = aus)")
    p_proj.add_argument("--max-points", type=int, default=None,
                        help="thin huge scans to at most this many points")
    p_proj.add_argument("--report-html", type=Path, default=None, metavar="HTML",
                        help="print-ready inspection report (A4)")
    p_proj.add_argument("--seed", type=int, default=None)

    p_reg = sub.add_parser(
        "register",
        help="register roughly pre-aligned scans onto the first one (ICP) "
        "and merge them into a single cloud",
    )
    p_reg.add_argument("reference", type=Path, help="reference scan (stays fixed)")
    p_reg.add_argument("others", type=Path, nargs="+", help="scans to align onto it")
    p_reg.add_argument("-o", "--output", type=Path, required=True,
                       help="merged output cloud (.ply)")
    p_reg.add_argument("--voxel", type=float, default=0.0,
                       help="voxel size for thinning the merged cloud (0 = keep all)")
    p_reg.add_argument("--transforms", type=Path, default=None,
                       help="write the 4x4 transforms as JSON")

    p_an = sub.add_parser(
        "analyze",
        help="industrial analysis: cylinders, stepped shafts, gears, gear "
        "stages and steel profiles (Wellen, Zahnräder, Getriebe, Stahlbau)",
    )
    p_an.add_argument("input", type=Path, nargs="+", help="point cloud(s) to analyze")
    p_an.add_argument("-o", "--output", type=Path, default=None,
                      help="write the analysis report to this JSON file")
    p_an.add_argument("--mesh", type=Path, default=None,
                      help="write detected primitives as a mesh (.glb/.html/.obj)")
    p_an.add_argument("--dist", type=float, default=None,
                      help="cylinder RANSAC distance threshold (default: auto)")
    p_an.add_argument("--no-steel", action="store_true",
                      help="skip steel profile matching")

    p_br = sub.add_parser(
        "bridge",
        help="bridge structure analysis: type, deck, spans, piers, bearings, "
        "arch, pylons, cables (Brückenbauwerke aller Art)",
    )
    p_br.add_argument("input", type=Path, nargs="+", help="point cloud(s) of the bridge")
    p_br.add_argument("-o", "--output", type=Path, default=None,
                      help="write the analysis report to this JSON file")
    p_br.add_argument("--mesh", type=Path, default=None,
                      help="write a solid 3D model of the bridge (deck, piers, "
                      "arch, pylons, cables) to .glb/.html/.obj/.stl")

    p_sm = sub.add_parser(
        "sheetmetal",
        help="welded sheet metal analysis: plates with thickness, weld seam "
        "lengths, weights, cutting outlines (geschweißte Blechkonstruktionen)",
    )
    p_sm.add_argument("input", type=Path, nargs="+", help="point cloud(s), both "
                      "plate faces must be scanned")
    p_sm.add_argument("-o", "--output", type=Path, default=None,
                      help="write the analysis report to this JSON file")
    p_sm.add_argument("--dxf", type=Path, default=None,
                      help="write 1:1 cutting outlines of all plates to this DXF")
    p_sm.add_argument("--dist", type=float, default=None,
                      help="plane RANSAC distance threshold (default: auto)")
    p_sm.add_argument("--max-thickness", type=float, default=0.05,
                      help="maximum plate thickness in input units (default: 0.05)")
    p_sm.add_argument("--unfold", action="store_true",
                      help="treat edge-to-edge corner junctions as bends and "
                      "export unfolded flat patterns with bend lines")
    p_sm.add_argument("--k-factor", type=float, default=0.44,
                      help="neutral axis k-factor for the bend allowance "
                      "(default: 0.44)")

    p_cmp = sub.add_parser(
        "compare",
        help="compare two scan epochs of the same structure: fine-register "
        "(ICP), then signed displacement along the local normals — "
        "settlement/deformation monitoring (Verformungsmessung)",
    )
    p_cmp.add_argument("reference", type=Path, help="older epoch (stays fixed)")
    p_cmp.add_argument("current", type=Path, help="newer epoch")
    p_cmp.add_argument("-o", "--output", type=Path, default=None,
                       help="write the comparison report to this JSON file")
    p_cmp.add_argument("--heatmap", type=Path, default=None, metavar="PLY",
                       help="write the newer epoch colored by displacement "
                       "(blue = settlement, red = bulging)")
    p_cmp.add_argument("--no-register", action="store_true",
                       help="scans share exact coordinates — skip ICP")
    p_cmp.add_argument("--tolerance", type=float, default=0.005,
                       help="displacement tolerance in input units (default 0.005)")
    p_cmp.add_argument("--max-points", type=int, default=None,
                       help="thin huge scans while reading")

    p_gui = sub.add_parser(
        "gui", help="start the graphical interface (local web app in the browser)"
    )
    p_gui.add_argument("files", type=Path, nargs="*",
                       help="point cloud(s) to preload")
    p_gui.add_argument("--port", type=int, default=8317,
                       help="port on 127.0.0.1 (default: 8317)")
    p_gui.add_argument("--no-browser", action="store_true",
                       help="do not open the browser automatically")

    p_wiz = sub.add_parser(
        "wizard", help="guided console mode (no browser needed)"
    )
    p_wiz.add_argument("files", type=Path, nargs="*",
                       help="point cloud(s) to reconstruct")

    p_gpu = sub.add_parser(
        "gpu", help="GPU/CUDA-Diagnose: prüft jede Stufe und schreibt gpu_diagnose.txt"
    )
    p_gpu.add_argument("-o", "--output", type=Path, default=None,
                       help="Zielordner für gpu_diagnose.txt (Standard: aktueller Ordner)")

    p_photos = sub.add_parser(
        "photos", help="photos → dense point cloud via COLMAP (must be installed)"
    )
    p_photos.add_argument("image_dir", type=Path, help="directory with overlapping photos")
    p_photos.add_argument("-o", "--output", type=Path, required=True,
                          help="output point cloud (.ply)")
    p_photos.add_argument("--work-dir", type=Path, default=None,
                          help="COLMAP scratch directory (default: <output>_colmap)")
    p_photos.add_argument("--quality", default="high", choices=["low", "medium", "high"])
    p_photos.add_argument("--cpu", action="store_true", help="disable GPU feature extraction")
    p_photos.add_argument("-v", "--verbose", action="store_true", help="stream COLMAP output")

    return parser


def _dispatch(args) -> int:
    try:
        if args.command == "info":
            return _cmd_info(args)
        if args.command == "reconstruct":
            return _cmd_reconstruct(args)
        if args.command == "project":
            return _cmd_project(args)
        if args.command == "register":
            return _cmd_register(args)
        if args.command == "analyze":
            return _cmd_analyze(args)
        if args.command == "sheetmetal":
            return _cmd_sheetmetal(args)
        if args.command == "bridge":
            return _cmd_bridge(args)
        if args.command == "compare":
            return _cmd_compare(args)
        if args.command == "gui":
            return _launch_gui(
                list(args.files), port=args.port, open_browser=not args.no_browser
            )
        if args.command == "wizard":
            code = _interactive(list(args.files) or None)
            if _is_frozen():
                _pause()
            return code
        if args.command == "gpu":
            return _cmd_gpu(args)
        if args.command == "photos":
            return _cmd_photos(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


_WELCOME = f"""
=====================================================================
 ScanToBIM {__version__} - Punktwolken & Fotos -> 3D-Modelle mit sauberen Kanten
=====================================================================

Grafische Oberflaeche:  scantobim.exe gui   (oeffnet den Browser)
Kommandozeile, z. B.:

  scantobim.exe reconstruct scan.laz -o modell.html --texture
  scantobim.exe analyze getriebe.e57 -o analyse.json
  scantobim.exe --help

Dies ist der gefuehrte Konsolen-Modus: Punktwolke einfach mit der
Maus in dieses Fenster ziehen (oder den Pfad eintippen) und Enter
druecken.
"""

_FORMATS = [
    ("HTML", ".html", "interaktiver 3D-Viewer - einfach im Browser oeffnen"),
    ("STEP", ".stp", "CAD (HiCAD, Inventor, SolidWorks, ...)"),
    ("IFC", ".ifc", "BIM (Revit, ArchiCAD, ...)"),
    ("GLB", ".glb", "3D-Austauschformat (Blender, Windows 3D-Viewer)"),
]

_SCENES = [
    ("Gebaeude aussen", "building"),
    ("Innenraum", "indoor"),
    ("Einzelobjekt / Bauteil", "object"),
]


def _clean_path(raw: str) -> Path:
    """Normalize a path typed or dragged into the console (strips quotes)."""
    return Path(raw.strip().strip('"').strip("'"))


def _ask_choice(title: str, options: list[tuple], describe) -> int:
    """Numbered menu; empty input selects the first entry. Returns the index."""
    print(f"\n{title}")
    for i, opt in enumerate(options):
        default = "  (Standard)" if i == 0 else ""
        print(f"  [{i + 1}] {describe(opt)}{default}")
    while True:
        try:
            raw = input(f"Auswahl [1-{len(options)}, Enter = 1] > ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if not raw:
            return 0
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"  bitte 1-{len(options)} eingeben")


def _interactive(files: list[Path] | None = None) -> int:
    """Guided mode for the double-clicked exe / files dragged onto it."""
    print(_WELCOME)
    files = list(files or [])
    if files:
        for f in files:
            print(f"  Eingabe: {f}")
    else:
        while True:
            prompt = "Datei> " if not files else "weitere Datei (Enter = fertig)> "
            try:
                raw = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                return 0
            if not raw:
                if files:
                    break
                return 0
            path = _clean_path(raw)
            if not path.exists():
                print(f"  Datei nicht gefunden: {path}")
                continue
            if path.suffix.lower() not in _CLOUD_EXTS:
                print(f"  kein unterstuetztes Punktwolken-Format: {path.suffix}\n"
                      f"  unterstuetzt: {' '.join(sorted(_CLOUD_EXTS))}")
                continue
            files.append(path)

    fmt = _FORMATS[_ask_choice(
        "Ausgabeformat:", _FORMATS, lambda o: f"{o[0]:5} - {o[2]}"
    )]
    preset = _SCENES[_ask_choice(
        "Was wurde gescannt?", _SCENES, lambda o: o[0]
    )][1]

    out = files[0].with_name(files[0].stem + "_modell" + fmt[1])
    report = files[0].with_name(files[0].stem + "_bericht.json")
    cmd = ["reconstruct", *(str(f) for f in files),
           "-o", str(out), "--preset", preset, "--report", str(report)]
    if fmt[1] in (".html", ".glb"):
        cmd.append("--texture")
    if len(files) > 1:
        registered = _ask_choice(
            "Mehrere Scans - liegen sie schon im gleichen Koordinatensystem?",
            [("Ja - direkt verschmelzen",), ("Nein - automatisch ausrichten (ICP)",)],
            lambda o: o[0],
        )
        if registered == 1:
            cmd.append("--register-inputs")

    print()
    try:
        code = main(cmd)
    except Exception:  # noqa: BLE001 — the window must stay readable
        import traceback

        traceback.print_exc()
        return 1
    if code == 0:
        print(f"\nFertig! Modell:  {out}")
        print(f"        Bericht: {report}")
        if fmt[1] == ".html":
            print("Die HTML-Datei einfach doppelklicken - sie oeffnet im Browser.")
    return code


def _cmd_info(args) -> int:
    cloud = read_point_cloud(args.input, max_points=args.max_points)
    lo, hi = cloud.aabb
    size = hi - lo
    print(f"file:        {args.input}")
    print(f"points:      {len(cloud):,}")
    print(f"extent:      {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f}")
    print(f"min:         {lo[0]:.3f} {lo[1]:.3f} {lo[2]:.3f}")
    print(f"max:         {hi[0]:.3f} {hi[1]:.3f} {hi[2]:.3f}")
    print(f"colors:      {'yes' if cloud.colors is not None else 'no'}")
    print(f"normals:     {'yes' if cloud.normals is not None else 'no'}")
    print(f"intensity:   {'yes' if cloud.intensity is not None else 'no'}")
    from scantobim.core.preprocess import estimate_point_spacing

    spacing = estimate_point_spacing(cloud)
    print(f"spacing:     {spacing:.5f} (median nearest-neighbor distance)")
    return 0


def _read_inputs(
    paths: list[Path], register: bool, max_points: int | None = None
) -> "object":
    """Read one or more clouds; optionally ICP-register onto the first."""
    clouds = []
    for p in paths:
        print(f"reading {p} …")
        cloud = read_point_cloud(p, max_points=max_points)
        print(f"  {len(cloud):,} points")
        clouds.append(cloud)
    if len(clouds) == 1:
        return clouds[0]
    if register:
        from scantobim.core.registration import apply_transform, register_point_to_plane

        for i in range(1, len(clouds)):
            res = register_point_to_plane(clouds[i], clouds[0])
            clouds[i] = apply_transform(clouds[i], res.transform)
            print(f"  registered {paths[i].name}: rmse={res.rmse:.5f}")
    from scantobim.core.registration import merge_clouds

    merged = merge_clouds(clouds)
    print(f"merged {len(clouds)} clouds → {len(merged):,} points")
    return merged


def _cmd_reconstruct(args) -> int:
    cloud = _read_inputs(args.input, args.register_inputs, max_points=args.max_points)

    cfg = PipelineConfig.preset(
        args.preset if args.preset != "auto" else "building"
    )
    from scantobim.core.pipeline import SOURCE_PROFILES, apply_source_profile

    apply_source_profile(cfg, getattr(args, "source", "standard"))
    if args.voxel is not None:
        cfg.voxel_size = args.voxel
    if args.dist is not None:
        cfg.distance_threshold = args.dist
    if args.max_planes is not None:
        cfg.max_planes = args.max_planes
    if args.no_regularize:
        cfg.regularize = False
    if args.no_straighten:
        cfg.straighten = False
    if args.no_color:
        cfg.color_surfaces = False
    if args.no_openings:
        cfg.detect_openings = False
    if args.watertight:
        cfg.watertight = True
    if args.ghost_tol is not None:
        cfg.ghost_offset_tol = args.ghost_tol
    if args.cylinders:
        cfg.cylinder_detection = True
    if args.align:
        cfg.align_axes = True
    if args.seed is not None:
        cfg.seed = args.seed

    trajectory = None
    if args.trajectory is not None:
        from scantobim.io.project import read_trajectory

        trajectory = read_trajectory(args.trajectory)
        print(f"trajectory: {len(trajectory)} scanner positions "
              "(normals oriented towards the path)")

    # ---- Stufe 1: Komplett-Mesh aus dem gesamten Scan ----------------------
    report_extra: dict = {}
    _print_gpu_status(report_extra, diag_dir=args.output.parent)
    if args.input:
        _copy_diagnosis(report_extra, Path(args.input[0]).parent)
    from concurrent.futures import ThreadPoolExecutor as _TPE

    full_mesh = full_viewer = None
    _ff_pool = None
    _ff_future = None
    if (
        getattr(args, "freeform", True)
        and args.output.suffix.lower() not in (".stp", ".step", ".ifc")
    ):
        # Komplett-Mesh parallel zur Strukturanalyse (unabhängige Stufen).
        _ff_pool = _TPE(max_workers=1)
        _ff_future = _ff_pool.submit(_build_full_mesh, cloud, report_extra)

    def _join_ff():
        nonlocal full_mesh, full_viewer
        if _ff_future is not None:
            full_mesh, full_viewer = _ff_future.result()
            _ff_pool.shutdown(wait=False)

    # ---- Stufe 2 (Option): Ebenen & Linien suchen ---------------------------
    if not getattr(args, "structure", True):
        print("Strukturanalyse übersprungen (--no-structure).")
        _join_ff()
        return _write_full_only(
            args.output, full_mesh, full_viewer,
            {"input_points": len(cloud), **report_extra},
            getattr(args, "report", None),
        )
    print(f"reconstructing (preset: {args.preset}) …")
    try:
        if args.preset == "auto":
            from scantobim.core.autotune import auto_reconstruct

            result = auto_reconstruct(
                cloud,
                trajectory=trajectory,
                seed=args.seed,
                overrides={
                    **SOURCE_PROFILES.get(getattr(args, "source", "standard"), {}),
                    "watertight": cfg.watertight,
                    "align_axes": cfg.align_axes,
                    "ghost_offset_tol": cfg.ghost_offset_tol,
                    "cylinder_detection": cfg.cylinder_detection,
                    "detect_openings": cfg.detect_openings,
                },
            )
        else:
            result = reconstruct(cloud, cfg, trajectory=trajectory)
    except ValueError as exc:
        _join_ff()
        if full_mesh is None:
            raise
        print(f"Strukturanalyse fehlgeschlagen ({exc})")
        print("Komplett-Mesh bleibt als Ergebnis erhalten.")
        return _write_full_only(
            args.output, full_mesh, full_viewer,
            {"input_points": len(cloud), **report_extra},
            getattr(args, "report", None),
        )
    _join_ff()
    rep = result.report
    rep.update(report_extra)
    _contour_full_mesh(full_mesh, full_viewer, result, rep)
    openings = sum(s.get("openings", 0) for s in rep["surfaces"])
    q = rep["quantities"]
    print(f"  planes:    {rep['planes']}")
    print(f"  angles:    {rep['plane_angles']}")
    print(f"  corners:   {rep['exact_corners']}")
    print(f"  openings:  {openings}")
    if rep.get("cylinders"):
        dias = ", ".join(f"⌀{c['radius'] * 2:.3f}" for c in rep["cylinders"])
        print(f"  cylinders: {len(rep['cylinders'])} [{dias}]")
    print(f"  classes:   {q['surface_count_by_class']}")
    if "volume" in q:
        print(f"  volume:    {q['volume']} (watertight, {q['orientation']} normals)")
    print(f"  mesh:      {rep['mesh']['vertices']} vertices, {rep['mesh']['triangles']} triangles")
    print(f"  residual:  {rep['residual_points']} unexplained points")
    print(f"  runtime:   {rep['runtime_seconds']} s")

    ext = args.output.suffix.lower()
    output_mesh = result.mesh
    wants_texture = args.texture or args.texture_photos is not None
    if wants_texture:
        if ext in (".stp", ".step", ".ifc", ".ply", ".stl"):
            print(f"note: {ext} carries no texture — texturing skipped "
                  "(use .glb/.html/.obj)")
        elif args.texture_photos is None and cloud.colors is None:
            print("note: the input cloud carries no RGB colors — texturing "
                  "skipped (capture with RGB or use --texture-photos)")
        else:
            transform = None
            if "alignment" in result.report:
                transform = np.array(result.report["alignment"])
            if args.texture_photos is not None:
                if args.colmap_model is None:
                    raise ValueError("--texture-photos requires --colmap-model")
                from scantobim.core.texture import bake_texture_from_photos

                print("projecting photos onto the model …")
                output_mesh = bake_texture_from_photos(
                    result, args.colmap_model, args.texture_photos,
                    texel_size=args.texel, transform=transform,
                )
            else:
                from scantobim.core.texture import bake_texture_from_cloud

                print("baking texture from cloud colors …")
                output_mesh = bake_texture_from_cloud(
                    result, cloud, texel_size=args.texel, transform=transform
                )
            th, tw = output_mesh.texture.shape[:2]
            print(f"  texture atlas: {tw} x {th} px")

    if (
        full_viewer is not None
        and args.texture_photos is not None
        and args.colmap_model is not None
        and ext not in (".stp", ".step", ".ifc")
    ):
        transform = None
        if "alignment" in result.report:
            transform = np.array(result.report["alignment"])
        full_viewer = _bake_full_photo_atlas(
            full_viewer, args.colmap_model, args.texture_photos,
            transform, rep, args.output,
        )

    if ext in (".stp", ".step"):
        from scantobim.io.step import write_step

        out = write_step(result.surfaces, args.output)
        print(f"wrote {out} (STEP AP214 — in HiCAD über Datei > Import > STEP laden)")
    elif ext == ".ifc":
        from scantobim.io.ifc import write_ifc

        out = write_ifc(
            result.surfaces, args.output, storeys=result.report.get("storeys")
        )
        n_storeys = max(1, len(result.report.get("storeys", [])))
        print(f"wrote {out} (IFC4, {n_storeys} Geschoss(e))")
    else:
        out = write_mesh(
            output_mesh, args.output,
            residual=None if full_viewer is not None else result.residual,
            freeform=full_viewer,
            freeform_label="Komplett-Mesh (Scan)",
        )
        print(f"wrote {out}")
        if full_mesh is not None:
            glb = args.output.with_name(args.output.stem + "_komplett.glb")
            write_mesh(full_mesh, glb)
            print(f"wrote {glb} (Komplett-Mesh, volle Auflösung)")
        elif args.output.suffix.lower() in (".html", ".htm") and len(result.residual):
            print(
                f"  Viewer: {min(len(result.residual), 800_000):,} Scan-Restpunkte "
                "als schaltbare Ebene eingebettet"
            )

    if args.floorplan is not None:
        from scantobim.io.dxf import write_floorplan_dxf

        plan = write_floorplan_dxf(
            result.mesh, args.floorplan, height_above_floor=args.floorplan_height
        )
        print(f"wrote {plan} (slice at base + {args.floorplan_height})")

    if args.deviation is not None:
        _write_deviation(result, cloud, args.deviation, args.tolerance)
    if args.views is not None:
        _write_views(output_mesh, args.views)
    if args.report_html is not None:
        from scantobim.io.report_html import render_report_html

        out = render_report_html(
            rep, args.report_html,
            title=f"Prüfbericht — {args.input[0].stem}",
            views_dir=args.views,
        )
        print(f"wrote {out} (druckfertiger Prüfbericht)")

    _print_gpu_usage(rep)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, indent=2, default=_json_default))
        print(f"wrote {args.report}")
    if args.residual_out is not None and len(result.residual):
        write_point_cloud(result.residual, args.residual_out)
        print(f"wrote {args.residual_out}")
    return 0


def _cmd_project(args) -> int:
    """SLAM project folder → photo-textured clean-edged model."""
    from scantobim.core.progress import report as _prog
    from scantobim.io.project import read_trajectory, scan_project_dir

    _prog(0.01, "Projektordner analysieren")
    project = scan_project_dir(args.directory)
    if project.cloud is None:
        raise ValueError(
            f"{args.directory}: keine Punktwolke im Projektordner gefunden "
            f"(gesucht: {' '.join(CLOUD_EXTS_SORTED)})"
        )
    print(f"SLAM-Projekt: {args.directory}")
    for line in project.describe():
        print(f"  {line}")
    unused = project.unused_report()
    if unused:
        print("  Datei-Inventar — im Ordner gefunden, aber NICHT verwendet:")
        for line in unused:
            print(f"    • {line}")
    else:
        print("  Datei-Inventar: alle erkannten Dateien werden verwendet.")

    _prog(0.03, "Punktwolke laden")
    cloud = read_point_cloud(project.cloud, max_points=args.max_points)
    _prog(0.10, "Punktwolke geladen")
    print(f"  geladen: {len(cloud):,} Punkte"
          + (", mit Farben" if cloud.colors is not None else ""))
    if (
        args.max_points
        and project.cloud_points
        and project.cloud_points > args.max_points
    ):
        print(
            f"  HINWEIS: Wolke hat {project.cloud_points:,} Punkte — auf das "
            f"Limit von {args.max_points / 1e6:.0f} Mio ausgedünnt. Für "
            "maximales Detail das Limit erhöhen (GUI: Erweiterte "
            "Einstellungen → Max. Punkte, CLI: --max-points)."
        )

    # Dense-but-uncolorized cloud + colored sibling → graft the colors onto
    # the dense geometry. IMPORTANT: R=G=B intensity greyscales (e.g.
    # uncolorized.las) count as UNcolored — grey must never become the
    # texture when real colors are available.
    grey_rgb = _colors_are_grey(cloud)
    if grey_rgb:
        print("  Hinweis: RGB der Wolke ist nur Intensitäts-Grau (R=G=B) — "
              "wird nicht als Farbtextur verwendet")
    if (cloud.colors is None or grey_rgb) and project.color_source is not None:
        from scantobim.core.preprocess import transfer_colors

        print(f"  Farben übertragen von: {project.color_source.name} …")
        color_cloud = read_point_cloud(
            project.color_source, max_points=args.max_points
        )
        fraction = transfer_colors(cloud, color_cloud)
        print(f"  → {fraction:.0%} der Punkte eingefärbt")
    elif grey_rgb:
        cloud.colors = None  # grau lieber neutral als falsch

    trajectory = None
    if project.trajectory is not None:
        trajectory = read_trajectory(project.trajectory)
        if trajectory.shape[1] >= 4 and cloud.times is not None:
            print(f"  Trajektorie: {len(trajectory)} Positionen mit Zeitstempeln "
                  "— Normalen werden ZEIT-exakt zur Scannerposition orientiert "
                  "(gps_time der Punkte ↔ Trajektorie)")
        else:
            print(f"  Trajektorie: {len(trajectory)} Positionen "
              "(Normalen werden zum Scanner orientiert)")

    cfg = PipelineConfig.preset(
        args.preset if args.preset != "auto" else "building"
    )
    from scantobim.core.pipeline import SOURCE_PROFILES, apply_source_profile

    source = getattr(args, "source", "slam")
    # Real structures carry round members (pipes, columns, arches) — detect
    # them by default so curved parts become regular geometry, not blobs.
    cfg.cylinder_detection = True
    if source != "slam" and project.trajectory is not None:
        print(
            f"  ACHTUNG: Quelle '{source}' bei einem SLAM-Projekt — empfohlen "
            "ist 'SLAM-Handscanner' (verschmilzt Registrierungs-Doppelwände, "
            "sonst drohen null erkannte Ebenen)."
        )
    apply_source_profile(cfg, source)
    for key, value in (getattr(args, "advanced", None) or {}).items():
        setattr(cfg, key, value)
    if args.watertight:
        cfg.watertight = True
    if args.align:
        cfg.align_axes = True
    if args.seed is not None:
        cfg.seed = args.seed

    # ---- Stufe 1: Komplett-Mesh aus dem gesamten Scan ----------------------
    output = args.output or (args.directory / "scantobim_modell.html")
    report_extra: dict = {}
    _print_gpu_status(report_extra, diag_dir=output.parent)
    _copy_diagnosis(report_extra, args.directory)
    # Stufe 1 und Stufe 2 sind unabhängig — sie laufen PARALLEL (numpy/
    # scipy geben das GIL bei großen Operationen frei, die Kerne addieren
    # sich statt zu warten).
    from concurrent.futures import ThreadPoolExecutor

    full_mesh = full_viewer = None
    _full_pool = None
    _full_future = None
    if getattr(args, "freeform", True):
        _prog(0.14, "Komplett-Mesh ∥ Strukturanalyse")
        _full_pool = ThreadPoolExecutor(max_workers=1)
        _full_future = _full_pool.submit(_build_full_mesh, cloud, report_extra)
    _prog(0.34, "Strukturanalyse")

    def _join_full_mesh():
        nonlocal full_mesh, full_viewer
        if _full_future is not None:
            full_mesh, full_viewer = _full_future.result()
            _full_pool.shutdown(wait=False)

    def _fallback_report():
        _join_full_mesh()
        return {
            "input_points": len(cloud),
            "trajectory_positions": 0 if trajectory is None else int(len(trajectory)),
            **report_extra,
        }
    fallback_report_path = args.report or output.with_name(
        output.stem + "_bericht.json"
    )

    # ---- Stufe 2 (Option): Ebenen & Linien suchen ---------------------------
    if not getattr(args, "structure", True):
        print("Strukturanalyse übersprungen (--no-structure).")
        fb = _fallback_report()
        full_viewer = _fallback_photo_atlas(
            project, args, cloud, full_viewer, fb, output
        )
        return _write_full_only(
            output, full_mesh, full_viewer, fb, fallback_report_path
        )
    print(f"reconstructing (preset: {args.preset}, Quelle: {source}) …")
    try:
        if args.preset == "auto":
            from scantobim.core.autotune import auto_reconstruct

            result = auto_reconstruct(
                cloud, trajectory=trajectory, seed=args.seed,
                overrides={
                    **SOURCE_PROFILES.get(source, {}),
                    **(getattr(args, "advanced", None) or {}),
                    "watertight": cfg.watertight,
                    "align_axes": cfg.align_axes,
                    "cylinder_detection": True,
                },
            )
        else:
            result = reconstruct(cloud, cfg, trajectory=trajectory)
    except ValueError as exc:
        _join_full_mesh()
        if full_mesh is None:
            raise
        print(f"Strukturanalyse fehlgeschlagen ({exc})")
        print("Komplett-Mesh bleibt als Ergebnis erhalten.")
        fb = _fallback_report()
        full_viewer = _fallback_photo_atlas(
            project, args, cloud, full_viewer, fb, output
        )
        return _write_full_only(
            output, full_mesh, full_viewer,
            fb, fallback_report_path,
        )
    _join_full_mesh()
    rep = result.report
    rep.update(report_extra)
    _prog(0.55, "Kontur-Schärfung")
    _contour_full_mesh(full_mesh, full_viewer, result, rep)
    _prog(0.60, "Kameraposen")
    print(f"  planes: {rep['planes']}  angles: {rep['plane_angles']}  "
          f"residual: {rep['residual_points']}")
    if rep.get("point_spacing", 0) > 0.05:
        print(
            f"  HINWEIS: mittlerer Punktabstand "
            f"{rep['point_spacing'] * 100:.0f} cm — die Wolke ist stark "
            "ausgedünnt. Für ein detailtreues Modell die hochauflösende "
            "Export-Datei des Scanners verwenden."
        )

    # Texture: photo projection when poses + photos exist, else cloud colors.
    output_mesh = result.mesh
    texture_info: dict = {"source": "keine"}
    camera_source = project.colmap_model
    use_photos = (
        not args.no_photos
        and project.images_dir is not None
        and (
            camera_source is not None
            or project.xyzopk is not None
            or project.imgpose is not None
        )
    )
    transform = None
    if "alignment" in rep:
        transform = np.array(rep["alignment"])

    if use_photos and camera_source is None and project.imgpose is not None:
        # ImgPose carries QUATERNIONS — unambiguous rotations. Preferred
        # over the angle-based xyzopk when both exist.
        try:
            from scantobim.photogrammetry.xyzopk import cameras_from_imgpose

            print("Kameraposen (ImgPose): Quaternion-Posen werden am Scan "
                  "validiert …")
            ip_stats: dict = {}
            cams = cameras_from_imgpose(
                project.imgpose, project.images_dir, cloud,
                stats_out=ip_stats, calibration=project.calibration_data,
            )
            if ip_stats.get("score", 0.0) >= 0.5:
                camera_source = cams
                rep["kameraposen"] = {"quelle": "imgpose", **ip_stats}
                print(
                    f"  → {ip_stats['convention']}, Brennweite "
                    f"{ip_stats['fx']:.0f} px "
                    f"({ip_stats.get('intrinsics_quelle', 'selbstkalibriert')}), "
                    f"Übereinstimmung {ip_stats['score'] * 100:.0f}%"
                )
            else:
                print(f"  ImgPose-Übereinstimmung zu gering "
                      f"({ip_stats.get('score', 0.0) * 100:.0f}%) — "
                      "versuche xyzopk")
        except Exception as exc:  # noqa: BLE001 — photos are best-effort
            print(f"  ImgPose übersprungen ({exc})")

    if use_photos and camera_source is None and project.xyzopk is not None:
        # xyzopk poses carry neither intrinsics nor a rotation convention —
        # both are self-calibrated against the (photo-)colorized cloud.
        try:
            from scantobim.photogrammetry.xyzopk import cameras_from_xyzopk

            print("Kameraposen (xyzopk): Selbstkalibrierung von Ausrichtung "
                  "und Brennweite am Scan …")
            xy_stats: dict = {}
            camera_source = cameras_from_xyzopk(
                project.xyzopk, project.images_dir, cloud, stats_out=xy_stats,
                calibration=project.calibration_data,
            )
            rep["kameraposen"] = {"quelle": "xyzopk", **xy_stats}
            print(
                f"  → Konvention {xy_stats['convention']}, Brennweite "
                f"{xy_stats['fx']:.0f} px "
                f"({xy_stats.get('intrinsics_quelle', 'selbstkalibriert')}), "
                f"Übereinstimmung {xy_stats['score'] * 100:.0f}%"
            )
        except Exception as exc:  # noqa: BLE001 — photos are best-effort
            print(f"  Selbstkalibrierung fehlgeschlagen ({exc}) — "
                  "Foto-Projektion übersprungen")
            camera_source = None

    if use_photos and camera_source is None:
        print("Hinweis: keine nutzbaren Kameraposen — Foto-Projektion "
              "übersprungen")
        use_photos = False

    if use_photos:
        # Sanity check: the camera path must live in the same coordinate
        # frame as the cloud — SLAM exports sometimes keep poses in a local
        # session frame while the cloud is georeferenced.
        from scantobim.core.texture import _resolve_cameras

        cams = _resolve_cameras(camera_source)
        centers = np.array([-c.rotation.T @ c.translation for c in cams])
        lo, hi = cloud.aabb
        diag = float(np.linalg.norm(hi - lo))
        dist = np.linalg.norm(centers - (lo + hi) / 2.0, axis=1)
        if float(np.median(dist)) > 2.0 * diag:
            print("Hinweis: Kameraposen liegen weit außerhalb der Punktwolke "
                  "(anderes Koordinatensystem?) — Foto-Projektion übersprungen")
            use_photos = False

    # From here on the camera poses are FIXED: the structure-model texture,
    # the complete-mesh photo colors, the photo atlas and the detail mesh
    # each use them with their OWN quality gates. One stage falling back
    # (e.g. little photo coverage on huge terrain planes) must never kill
    # the photorealistic atlas of the building meshes.
    photo_cams = camera_source if use_photos else None
    image_map = None
    if photo_cams is not None and cloud.colors is not None:
        # EVERY camera is color-checked against the cloud individually —
        # a 92% pose score can hide single cameras sampling the wrong
        # file (stereo exports reuse basenames across left/ and right/).
        try:
            from scantobim.photogrammetry.camcheck import validate_cameras

            ck: dict = {}
            kept_cams, image_map = validate_cameras(
                _resolve_cameras(photo_cams), project.images_dir, cloud,
                stats_out=ck,
            )
            print(
                f"  Kamera-Selbstprüfung: {ck.get('validiert', 0)} von "
                f"{ck.get('gesamt', 0)} Kameras farb-validiert "
                f"(Median-Score {ck.get('median_score', 0.0):.2f}"
                + (
                    f", {ck['mehrdeutige_namen']} mehrdeutige Dateinamen "
                    "aufgelöst"
                    if ck.get("mehrdeutige_namen") else ""
                )
                + ")"
            )
            rep["kamera_pruefung"] = ck
            if image_map is not None:
                photo_cams = kept_cams
        except Exception as exc:  # noqa: BLE001 — check is best-effort
            print(f"  Kamera-Selbstprüfung übersprungen ({exc})")
    depth_sample = None
    if photo_cams is not None:
        # Dense scene depth for the visibility test on the coarse structure
        # mesh — the cloud IS the scene, photos cannot see through it.
        depth_sample = cloud.points
        if len(depth_sample) > 400_000:
            rng = np.random.default_rng(0)
            depth_sample = depth_sample[
                rng.choice(len(depth_sample), 400_000, replace=False)
            ]

    no_texture = getattr(args, "no_texture", False)
    if photo_cams is not None and not no_texture:
        try:
            from pathlib import Path as _P

            from scantobim.core.phototex import bake_photo_atlas
            from scantobim.core.texture import _index_images

            print(f"projecting {project.image_count} photos onto the model …")
            _cams_l = _resolve_cameras(photo_cams)
            _img_idx = _index_images(project.images_dir)
            _found = sum(
                1 for c in _cams_l
                if c.name in _img_idx or _P(c.name).name in _img_idx
            )
            print(f"  Fotos gefunden: {_found} von {len(_cams_l)} "
                  "registrierten Kameras")
            base = result.mesh
            if base.vertex_colors is None and cloud.colors is not None:
                # Nearest cloud color per vertex — base layer for texels
                # no photo reaches (atlas texels fall back to vertex colors).
                from scipy.spatial import cKDTree

                rng = np.random.default_rng(0)
                idx = rng.choice(
                    len(cloud.points), min(len(cloud.points), 400_000),
                    replace=False,
                )
                pts_model = cloud.points[idx]
                if transform is not None:
                    pts_model = pts_model @ transform[:3, :3].T + transform[:3, 3]
                _, nn = cKDTree(pts_model).query(base.vertices, k=1, workers=-1)
                base.vertex_colors = cloud.colors[idx][nn]
            st_stats: dict = {}
            textured = bake_photo_atlas(
                base, photo_cams, project.images_dir,
                transform=transform, stats_out=st_stats,
                depth_points=depth_sample, image_map=image_map,
            )
            frac = (
                st_stats.get("photo_fraction", 0.0)
                if textured is not None else 0.0
            )
            print(f"  Foto-Anteil: {frac * 100:.0f}% der Modellfläche "
                  f"({st_stats.get('cameras_used', 0)} Kameras)")
            if textured is not None and frac >= 0.15:
                output_mesh = textured
                aw, ah = st_stats["atlas"]
                print(f"  texture atlas: {aw} x {ah} px")
                texture_info = {
                    "source": "foto-projektion",
                    "coverage": round(frac, 3),
                    "images_used": st_stats.get("cameras_used", 0),
                }
            else:
                print("Hinweis: Foto-Anteil am Strukturmodell zu gering — "
                      "es erhält Punktwolken-Farben (der Foto-Atlas der "
                      "Netz-Modelle läuft davon unabhängig)")
        except Exception as exc:  # noqa: BLE001 — fall back, don't fail the model
            print(f"Hinweis: Foto-Projektion aufs Strukturmodell "
                  f"fehlgeschlagen ({exc}) — verwende Punktwolken-Farben")
    if texture_info["source"] == "keine" and not no_texture:
        if cloud.colors is not None:
            from scantobim.core.texture import bake_texture_from_cloud

            print("baking texture from cloud colors …")
            output_mesh = bake_texture_from_cloud(
                result, cloud, texel_size=args.texel, transform=transform
            )
            texture_info = {"source": "punktfarben"}
        else:
            print("Hinweis: Punktwolke ohne Farbwerte — Modell bleibt untexturiert")
    rep["texture"] = texture_info
    print(f"Textur: {texture_info['source']}")

    _prog(0.68, "Foto-Farben")
    # High-resolution photo colors onto the complete mesh (per vertex).
    if (
        full_mesh is not None
        and photo_cams is not None
        and project.images_dir is not None
    ):
        try:
            from scantobim.core.texture import photo_colors_for_mesh

            print("Foto-Farben werden auf das Komplett-Mesh übertragen …")
            ff_stats: dict = {}
            frac = photo_colors_for_mesh(
                full_mesh, photo_cams, project.images_dir,
                transform=transform, stats_out=ff_stats,
                image_map=image_map,
            )
            if frac > 0 and full_viewer is not None and full_viewer is not full_mesh:
                from scipy.spatial import cKDTree

                tree = cKDTree(full_mesh.vertices)
                _, nearest = tree.query(full_viewer.vertices, k=1, workers=-1)
                full_viewer.vertex_colors = full_mesh.vertex_colors[nearest]
            if "komplett_mesh" in rep:
                rep["komplett_mesh"]["foto_farben_anteil"] = round(frac, 3)
            print(
                f"  → {frac * 100:.0f}% der Netz-Ecken mit Foto-Farben "
                f"({ff_stats.get('cameras_used', 0)} Kameras verwendet)"
            )
        except Exception as exc:  # noqa: BLE001 — photo colors are best-effort
            print(f"  Foto-Farben übersprungen ({exc})")

    # Detail-Mesh und Foto-Atlas sind unabhängig → parallel.
    _detail_future = None
    _detail_pool = None
    if output.suffix.lower() not in (".stp", ".step", ".ifc"):
        _detail_pool = ThreadPoolExecutor(max_workers=1)
        _detail_future = _detail_pool.submit(
            _build_detail_mesh, cloud, result,
            getattr(args, "detail_raster", 0.02),
        )
    _prog(0.78, "Foto-Textur-Atlas ∥ Detail-Mesh")
    # Photo-realistic texture ATLAS on the complete mesh: full photo
    # resolution instead of one color per vertex.
    if (
        full_viewer is not None
        and photo_cams is not None
        and project.images_dir is not None
        and output.suffix.lower() not in (".stp", ".step", ".ifc")
    ):
        full_viewer = _bake_full_photo_atlas(
            full_viewer, photo_cams, project.images_dir,
            transform, rep, output, image_map=image_map,
        )
    detail_result = None
    if _detail_future is not None:
        detail_result = _detail_future.result()
        _detail_pool.shutdown(wait=False)
    detail_viewer = None
    if detail_result is not None:
        detail_mesh, detail_sub = detail_result
        detail_viewer = _write_detail_mesh(
            detail_mesh, detail_sub, photo_cams, project.images_dir,
            transform, rep, output, result=result, image_map=image_map,
        )
    state_out = getattr(args, "state_out", None)
    if state_out is not None:
        from scantobim.io.state import save_export_state

        save_export_state(
            state_out, result.surfaces, rep.get("storeys"),
            result.mesh, output_mesh,
        )
    _prog(0.92, "Dateien schreiben")

    ext = output.suffix.lower()
    if ext in (".stp", ".step"):
        from scantobim.io.step import write_step

        out = write_step(result.surfaces, output)
    elif ext == ".ifc":
        from scantobim.io.ifc import write_ifc

        out = write_ifc(result.surfaces, output, storeys=rep.get("storeys"))
    else:
        out = write_mesh(
            output_mesh, output,
            residual=None if full_viewer is not None else result.residual,
            freeform=full_viewer,
            freeform_label="Komplett-Mesh (ganze Szene, reduziert)",
            detail=detail_viewer,
            detail_label="Detail-Mesh Gebäude (fotorealistisch)",
        )
        if full_mesh is not None:
            glb = output.with_name(output.stem + "_komplett.glb")
            write_mesh(full_mesh, glb)
            print(f"wrote {glb} (Komplett-Mesh, volle Auflösung)")
        elif ext in (".html", ".htm") and len(result.residual):
            print(
                f"  Viewer: {min(len(result.residual), 800_000):,} Scan-Restpunkte "
                "als schaltbare Ebene eingebettet"
            )
    print(f"wrote {out}")

    if args.deviation is not None:
        _write_deviation(result, cloud, args.deviation, args.tolerance)
    if args.views is not None:
        _write_views(output_mesh, args.views)
    if getattr(args, "report_html", None) is not None:
        from scantobim.io.report_html import render_report_html

        out = render_report_html(
            rep, args.report_html,
            title=f"Prüfbericht — {args.directory.name}",
            views_dir=args.views,
        )
        print(f"wrote {out} (druckfertiger Prüfbericht)")

    _print_gpu_usage(rep)
    _prog(1.0, "fertig")
    report_path = args.report or output.with_name(output.stem + "_bericht.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(rep, indent=2, default=_json_default))
    print(f"wrote {report_path}")
    return 0


def _cmd_compare(args) -> int:
    from scantobim.core.compare import compare_epochs

    print(f"reading {args.reference} …")
    reference = read_point_cloud(args.reference, max_points=args.max_points)
    print(f"  {len(reference):,} points (Referenz-Epoche)")
    print(f"reading {args.current} …")
    current = read_point_cloud(args.current, max_points=args.max_points)
    print(f"  {len(current):,} points (neue Epoche)")

    if not args.no_register:
        print("fine-registering the epochs (trimmed ICP) …")
    stats, heat_cloud, transform = compare_epochs(
        reference, current,
        register=not args.no_register,
        tolerance=args.tolerance,
    )
    if not args.no_register:
        print(f"  Registrierung: Versatz {stats['registration_shift'] * 1000:.1f} mm entfernt")
    print(
        f"  Verformung: RMS {stats['rms'] * 1000:.1f} mm | "
        f"P95 {stats['p95'] * 1000:.1f} mm | max {stats['max'] * 1000:.1f} mm | "
        f"{stats['within_tolerance'] * 100:.1f}% innerhalb ±{args.tolerance * 1000:.1f} mm"
    )

    if args.heatmap is not None:
        write_point_cloud(heat_cloud, args.heatmap)
        print(f"wrote {args.heatmap} (blau = Setzung, rot = Ausbauchung)")
    if args.output is not None:
        report = {"epochs": [str(args.reference), str(args.current)], **stats,
                  "transform": transform.tolist()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=_json_default))
        print(f"wrote {args.output}")
    return 0


def _cmd_gpu(args) -> int:
    """Print the deep GPU/CUDA diagnosis and write gpu_diagnose.txt."""
    from scantobim.core.gpudiag import build_gpu_diagnosis, write_gpu_diagnosis

    text = build_gpu_diagnosis(refresh=True)
    print(text)
    path = write_gpu_diagnosis(args.output)
    if path is not None:
        print(f"wrote {path}")
    return 0


def _print_gpu_usage(rep: dict | None = None) -> None:
    """End-of-run proof of what the GPU actually computed."""
    from scantobim.core.accel import gpu_name, gpu_usage

    if not gpu_name():
        return
    u = gpu_usage()
    parts = []
    if u.get("normalen_punkte"):
        parts.append(f"Normalen für {u['normalen_punkte'] / 1e6:.1f} Mio Punkte")
    if u.get("sortier_laeufe"):
        parts.append(
            f"{int(u['sortier_laeufe'])} Voxel-Sortierungen "
            f"({u.get('sortier_schluessel', 0) / 1e6:.0f} Mio Schlüssel)"
        )
    if u.get("weitere_laeufe"):
        parts.append(f"{int(u['weitere_laeufe'])} weitere GPU-Läufe")
    if parts:
        print("GPU-Nutzung: " + " · ".join(parts))
    else:
        print("GPU-Nutzung: keine GPU-tauglichen Schritte in diesem Lauf "
              "(Wolke zu klein) — die CPU hat alles übernommen")
    if rep is not None:
        rep["gpu_nutzung"] = u


def _colors_are_grey(cloud, sample: int = 5000) -> bool:
    """True when the cloud's RGB is just intensity as grey (R≈G≈B)."""
    if cloud.colors is None or len(cloud) == 0:
        return False
    step = max(1, len(cloud) // sample)
    c = cloud.colors[::step].astype(np.int16)
    dev = int(np.abs(c[:, 0] - c[:, 1]).max()) if len(c) else 0
    dev = max(dev, int(np.abs(c[:, 1] - c[:, 2]).max()) if len(c) else 0)
    return dev <= 3


def _copy_diagnosis(report_extra: dict, target_dir: Path) -> None:
    """Mirror gpu_diagnose.txt into the user's own folder.

    GUI jobs run in a hidden temp directory — the user looks next to their
    scan data, so the diagnosis must land there too.
    """
    import shutil

    diag = report_extra.get("gpu_diagnose")
    if not diag:
        return
    src = Path(diag)
    try:
        target = Path(target_dir) / "gpu_diagnose.txt"
        if src.resolve() == target.resolve():
            return
        shutil.copyfile(src, target)
        print(f"  GPU-Diagnose auch hier: {target}")
        report_extra["gpu_diagnose"] = str(target)
    except OSError:
        pass


def _print_gpu_status(
    report_extra: dict | None = None, diag_dir: Path | None = None
) -> None:
    from scantobim import __version__
    from scantobim.core.accel import gpu_error, gpu_name

    print(f"ScanToBIM {__version__}")
    if report_extra is not None:
        report_extra["programm"] = f"ScanToBIM {__version__}"
    name = gpu_name()
    error = gpu_error()
    if name:
        print(f"GPU: {name} — CUDA-Beschleunigung aktiv")
        print("  (Task-Manager zeigt CUDA-Last im Diagramm 'CUDA' bzw. "
              "'Compute_0' — NICHT unter '3D')")
    elif error:
        print(f"GPU: CUDA nicht nutzbar ({error}) — CPU-Modus")
        low = error.lower()
        if "driver" in low or "treiber" in low:
            print("  → Der NVIDIA-Grafiktreiber ist zu alt für CUDA 12. "
                  "Bitte aktualisieren: https://www.nvidia.de/Download/index.aspx "
                  "(danach Neustart) — das Programm läuft bis dahin im CPU-Modus.")
        # Automatic deep diagnosis: every link of the CUDA chain tested
        # separately, written next to the results.
        from scantobim.core.gpudiag import write_gpu_diagnosis

        diag_path = write_gpu_diagnosis(diag_dir)
        if diag_path is not None:
            print(f"  GPU-Diagnose geschrieben: {diag_path} — "
                  "diese Datei zeigt die genaue Ursache (bitte mitschicken).")
            if report_extra is not None:
                report_extra["gpu_diagnose"] = str(diag_path)
    else:
        print("GPU: nicht verfügbar — CPU-Modus "
              "(NVIDIA-Karten: GPU-Version scantobim-windows-x64-gpu.zip)")
    if report_extra is not None:
        report_extra["gpu"] = name
        if error:
            report_extra["gpu_fehler"] = error


def _build_full_mesh(cloud, report=None):
    """Stage 1 of the mesh-first pipeline: the COMPLETE scan as one colored
    triangle mesh, built from all points before any structure analysis.

    Returns ``(fine, viewer)`` — full resolution for the GLB export and a
    lighter variant for the embedded HTML viewer — or ``(None, None)``.
    """
    from scantobim.core.freeform import freeform_mesh_from_points

    if len(cloud) < 300:
        return None, None
    print("Komplett-Mesh: gesamter Scan wird vernetzt …")
    fine = freeform_mesh_from_points(cloud, max_faces=4_000_000)
    if fine is None:
        print("  zu wenig zusammenhängende Geometrie — übersprungen")
        return None, None
    st = fine.freeform_stats
    if report is not None:
        report["komplett_mesh"] = st
    print(
        f"  Komplett-Mesh: {st['triangles']:,} Dreiecke, "
        f"{st['components']} Bauteile, deckt {st['points_covered'] * 100:.0f}% "
        f"des Scans ab (Raster {st['voxel'] * 100:.1f} cm)"
    )
    viewer = fine
    if len(fine.faces) > 600_000:
        lighter = freeform_mesh_from_points(cloud, max_faces=600_000)
        if lighter is not None:
            viewer = lighter
    return fine, viewer


def _contour_full_mesh(full_mesh, full_viewer, result, rep) -> None:
    """Contour stage-1's skin with stage-2's planes (walls flat, edges crisp)."""
    if full_mesh is None or not result.surfaces:
        return
    from scantobim.core.freeform import sharpen_mesh_with_planes

    frac = sharpen_mesh_with_planes(
        full_mesh, result.surfaces, full_mesh.freeform_stats["voxel"]
    )
    if full_viewer is not None and full_viewer is not full_mesh:
        sharpen_mesh_with_planes(
            full_viewer, result.surfaces, full_viewer.freeform_stats["voxel"]
        )
    if "komplett_mesh" in rep:
        rep["komplett_mesh"]["konturiert_anteil"] = round(frac, 3)
    print(
        f"  Kontur-Schärfung: {frac * 100:.0f}% der Netz-Ecken auf "
        "Strukturflächen/-kanten gezogen"
    )
    cylinders = rep.get("cylinders") or []
    if cylinders:
        from scantobim.core.freeform import sharpen_mesh_with_cylinders

        frac_c = sharpen_mesh_with_cylinders(
            full_mesh, cylinders, full_mesh.freeform_stats["voxel"]
        )
        if full_viewer is not None and full_viewer is not full_mesh:
            sharpen_mesh_with_cylinders(
                full_viewer, cylinders, full_viewer.freeform_stats["voxel"]
            )
        if "komplett_mesh" in rep:
            rep["komplett_mesh"]["zylinder_konturiert_anteil"] = round(frac_c, 3)
        print(
            f"  Zylinder-Kontur: {frac_c * 100:.1f}% der Netz-Ecken auf "
            f"{len(cylinders)} erkannte Rundbauteile gezogen"
        )


def _fallback_photo_atlas(project, args, cloud, full_viewer, report, output):
    """Photo texture for the mesh-only result (structure skipped or failed).

    The regular photo path runs after the structure stage — when that stage
    is skipped, this brings the same photorealism to the complete mesh:
    self-calibrate the xyzopk poses if needed, refresh the vertex colors
    from the photos, then bake the atlas. Best-effort throughout.
    """
    if (
        full_viewer is None
        or getattr(args, "no_photos", False)
        or project.images_dir is None
    ):
        return full_viewer
    camera_source = project.colmap_model
    if camera_source is None and project.imgpose is not None:
        try:
            from scantobim.photogrammetry.xyzopk import cameras_from_imgpose

            print("Kameraposen (ImgPose): Quaternion-Posen werden am Scan "
                  "validiert …")
            ip_stats: dict = {}
            cams = cameras_from_imgpose(
                project.imgpose, project.images_dir, cloud,
                stats_out=ip_stats, calibration=project.calibration_data,
            )
            if ip_stats.get("score", 0.0) >= 0.5:
                camera_source = cams
                report["kameraposen"] = {"quelle": "imgpose", **ip_stats}
        except Exception as exc:  # noqa: BLE001 — photos are best-effort
            print(f"  ImgPose übersprungen ({exc})")
    if camera_source is None and project.xyzopk is not None:
        try:
            from scantobim.photogrammetry.xyzopk import cameras_from_xyzopk

            print("Kameraposen (xyzopk): Selbstkalibrierung von Ausrichtung "
                  "und Brennweite am Scan …")
            xy_stats: dict = {}
            camera_source = cameras_from_xyzopk(
                project.xyzopk, project.images_dir, cloud, stats_out=xy_stats,
                calibration=project.calibration_data,
            )
            report["kameraposen"] = {"quelle": "xyzopk", **xy_stats}
        except Exception as exc:  # noqa: BLE001 — photos are best-effort
            print(f"  Selbstkalibrierung fehlgeschlagen ({exc})")
            return full_viewer
    if camera_source is None:
        return full_viewer
    try:
        from scantobim.core.texture import photo_colors_for_mesh

        photo_colors_for_mesh(full_viewer, camera_source, project.images_dir)
    except Exception:  # noqa: BLE001 — vertex colors are only the base layer
        pass
    return _bake_full_photo_atlas(
        full_viewer, camera_source, project.images_dir, None, report, output
    )


def _build_detail_mesh(cloud, result, raster: float):
    """High-detail mesh of the BUILDING region (1–2 cm raster).

    The overview mesh covers the whole 100-m-class scene at a coarse
    raster. The detail region is the bounding box of the BUILDING surfaces
    (walls, roofs, slabs, ceilings) — terrain is excluded, otherwise a
    59×41 m ground plane drags the whole scene into the box and the face
    budget forces the raster far above the requested 1–2 cm. Returns the
    mesh (written later, photo-textured when cameras exist) or ``None``.
    """
    if raster is None or raster <= 0 or result is None or not result.surfaces:
        return None
    try:
        from scipy.spatial import cKDTree

        from scantobim.core.freeform import (
            freeform_mesh_from_points,
            sharpen_mesh_with_planes,
        )

        # FREISTELLUNG: keep only points close to the BUILDING surfaces
        # (walls, roofs, slabs, ceilings). A bounding box kept ~95% of the
        # scene (walls at opposite scene ends span everything) — the
        # distance band actually cuts terrain, vegetation and street
        # clutter away, which is what makes the detail mesh clean.
        building = {
            s.plane_index for s in result.surfaces
            if getattr(s, "surface_class", "") != "terrain"
        }
        mesh = result.mesh
        sel = (
            np.isin(mesh.face_groups, list(building))
            if building and mesh.face_groups is not None
            else np.ones(len(mesh.faces), dtype=bool)
        )
        tris = mesh.vertices[mesh.faces[sel]]
        if not len(tris):
            return None
        # Sample each surface triangle at ~0.3 m so the KD-tree represents
        # the faces themselves, not just their corners.
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        n_per = np.clip(np.ceil(areas / 0.09).astype(np.int64), 1, 400)
        total = int(n_per.sum())
        if total > 400_000:
            n_per = np.maximum(1, (n_per * (400_000 / total)).astype(np.int64))
        rng = np.random.default_rng(0)
        rep = np.repeat(np.arange(len(tris)), n_per)
        r1 = np.sqrt(rng.random(len(rep)))
        r2 = rng.random(len(rep))
        samples = (
            a[rep] * (1 - r1)[:, None]
            + b[rep] * (r1 * (1 - r2))[:, None]
            + c[rep] * (r1 * r2)[:, None]
        )
        band = 0.8  # meters around the building surfaces
        dist, _ = cKDTree(samples).query(
            cloud.points, k=1, workers=-1, distance_upper_bound=band
        )
        mask = np.isfinite(dist)
        n_kept = int(mask.sum())
        if n_kept < 5_000:
            return None
        sub = cloud.select(mask)
        print(
            f"Detail-Mesh: Gebäude freigestellt — {n_kept:,} von "
            f"{len(cloud):,} Punkten (Abstand ≤ {band:.1f} m zu "
            f"Gebäudeflächen), Raster {raster * 100:.1f} cm …"
        )

        # The plane fits know these surfaces to fractions of a millimeter —
        # snapping their supporting points onto the plane BEFORE meshing
        # removes the voxel/noise ripple on walls and roofs at the root.
        # Points near TWO planes (edges, corners) stay untouched, so no
        # corner gets rounded off.
        pts = sub.points
        claims = np.zeros(len(pts), dtype=np.int8)
        offset = np.zeros(len(pts), dtype=np.float64)
        normal_of = np.zeros((len(pts), 3), dtype=np.float64)
        flat_tol = 0.035
        for s in result.surfaces:
            if getattr(s, "surface_class", "") == "terrain":
                continue
            n = np.asarray(s.normal, dtype=np.float64)
            p0 = np.asarray(s.outer[0], dtype=np.float64)
            lo_s = s.outer.min(axis=0) - 0.3
            hi_s = s.outer.max(axis=0) + 0.3
            box = np.all((pts >= lo_s) & (pts <= hi_s), axis=1)
            if not box.any():
                continue
            d = (pts[box] - p0) @ n
            near = np.abs(d) < flat_tol
            idx = np.flatnonzero(box)[near]
            claims[idx] += 1
            offset[idx] = d[near]
            normal_of[idx] = n
        single = claims == 1
        if single.any():
            pts[single] -= offset[single, None] * normal_of[single]
            print(
                f"  Ebenen-Glättung: {int(single.sum()):,} Punkte "
                f"({single.mean() * 100:.0f}%) exakt auf ihre "
                "Strukturebene projiziert (Wände/Dächer plan)"
            )

        detail = freeform_mesh_from_points(
            sub, voxel=float(raster), max_faces=12_000_000
        )
        if detail is None:
            print("  Detail-Mesh übersprungen (zu wenig zusammenhängende Geometrie)")
            return None
        got = float(detail.freeform_stats["voxel"])
        if got > float(raster) * 1.05:
            print(
                f"  ACHTUNG: Detail-Raster auf {got * 100:.1f} cm vergröbert "
                f"(Flächen-Budget 12 Mio erreicht) — kleinere Region oder "
                f"größeres Raster wählen"
            )
        else:
            print(f"  Detail-Raster gehalten: {got * 100:.1f} cm")
        sharpen_mesh_with_planes(
            detail, result.surfaces, detail.freeform_stats["voxel"]
        )
        detail = _strip_detail_clutter(detail)
        return detail, sub
    except Exception as exc:  # noqa: BLE001 — detail is a bonus layer
        print(f"  Detail-Mesh übersprungen ({exc})")
        return None


def _strip_detail_clutter(detail):
    """Drop tiny disconnected components (rail['s'], mast fragments, noise
    speckles) from the detail mesh — the fringe-makers. They stay part of
    the complete mesh; the detail mesh keeps the building plus every
    component of meaningful size."""
    try:
        from scipy import sparse
        from scipy.sparse.csgraph import connected_components

        faces = detail.faces
        n_v = len(detail.vertices)
        rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        graph = sparse.coo_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(n_v, n_v),
        )
        n_comp, labels = connected_components(graph, directed=False)
        if n_comp <= 1:
            return detail
        face_label = labels[faces[:, 0]]
        counts = np.bincount(face_label, minlength=n_comp)
        min_faces = max(1_500, int(0.003 * len(faces)))
        keep_comp = counts >= min_faces
        if keep_comp.all():
            return detail
        keep_faces = keep_comp[face_label]
        dropped = int((~keep_faces).sum())
        n_dropped_comp = int((~keep_comp).sum())
        new_faces = faces[keep_faces]
        used = np.zeros(n_v, dtype=bool)
        used[new_faces] = True
        remap = np.cumsum(used) - 1
        stripped_stats = dict(detail.freeform_stats)
        from scantobim.core.mesh import Mesh as _M

        stripped = _M(
            vertices=detail.vertices[used],
            faces=remap[new_faces],
            vertex_colors=(
                detail.vertex_colors[used]
                if detail.vertex_colors is not None else None
            ),
        )
        stripped_stats["triangles"] = int(len(stripped.faces))
        stripped_stats["vertices"] = int(len(stripped.vertices))
        stripped_stats["components"] = int(keep_comp.sum())
        stripped.freeform_stats = stripped_stats
        print(
            f"  Störer abgetrennt: {n_dropped_comp:,} Kleinst-Komponenten "
            f"({dropped:,} Dreiecke) aus dem Detail-Mesh entfernt — "
            f"{int(keep_comp.sum())} Komponenten bleiben "
            "(im Komplett-Mesh weiterhin enthalten)"
        )
        return stripped
    except Exception:  # noqa: BLE001 — clutter strip is best-effort
        return detail


def _write_detail_mesh(
    detail, sub_cloud, photo_cams, images_dir, transform, rep: dict,
    output: Path, result=None, image_map=None,
):
    """Photo-texture and write ``<name>_detail.glb``; returns the VIEWER
    layer (a lighter, equally photo-textured variant of the detail mesh —
    the default view of the HTML viewer) or ``None``."""
    viewer_layer = None
    try:
        from scantobim.core.phototex import bake_photo_atlas

        st = detail.freeform_stats
        out_mesh = detail
        if (
            photo_cams is not None
            and images_dir is not None
            and result is not None
            and result.surfaces
        ):
            # Kanten-Fotoabgleich: die Fotos lösen 3-5 mm auf — Kanten des
            # Detail-Mesh werden an den Bild-Gradienten nachjustiert,
            # BEVOR der Atlas gebacken wird. Die vermessenen Ebenen
            # (Maße, Flächen) bleiben unberührt.
            try:
                from scantobim.core.edgerefine import refine_detail_edges

                er_stats: dict = {}
                refine_detail_edges(
                    detail, result.surfaces, photo_cams, images_dir,
                    image_map=image_map, stats_out=er_stats,
                )
                if er_stats.get("kanten_kandidaten"):
                    print(
                        f"  Kanten-Fotoabgleich: "
                        f"{er_stats.get('kanten_nachjustiert', 0)} von "
                        f"{er_stats['kanten_kandidaten']} Kanten an den "
                        f"Foto-Gradienten nachjustiert (Median "
                        f"{er_stats.get('median_verschiebung_mm', 0.0):.1f} mm)"
                    )
                    st = {**st, "kanten_fotoabgleich": er_stats}
            except Exception as exc:  # noqa: BLE001 — refinement optional
                print(f"  Kanten-Fotoabgleich übersprungen ({exc})")
        if photo_cams is not None and images_dir is not None:
            try:
                print("Foto-Textur: Atlas wird auf das Detail-Mesh projiziert …")
                dx_stats: dict = {}
                textured = bake_photo_atlas(
                    detail, photo_cams, images_dir,
                    transform=transform, stats_out=dx_stats,
                    max_atlas=8192, max_pages=4, image_map=image_map,
                )
                if (
                    textured is not None
                    and dx_stats.get("photo_fraction", 0.0) >= 0.15
                ):
                    out_mesh = textured
                    aw, ah = dx_stats["atlas"]
                    print(
                        f"  → {dx_stats.get('pages', 1)} Atlas-Seite(n) à "
                        f"{aw}×{ah} px, "
                        f"{dx_stats['texel_cm'] * 10:.0f} mm/Texel, "
                        f"{dx_stats['photo_fraction'] * 100:.0f}% Foto-Anteil "
                        f"({dx_stats['cameras_used']} Kameras)"
                    )
                    st = {**st, "foto_textur": dx_stats}
                elif textured is not None:
                    print("  Foto-Anteil zu gering — Detail-Mesh behält "
                          "Vertex-Farben")
            except Exception as exc:  # noqa: BLE001 — atlas is best-effort
                print(f"  Detail-Foto-Textur übersprungen ({exc})")
        detail_glb = output.with_name(output.stem + "_detail.glb")
        write_mesh(out_mesh, detail_glb)
        print(
            f"  Detail-Mesh: {st['triangles']:,} Dreiecke "
            f"(Raster {st['voxel'] * 100:.1f} cm) → {detail_glb.name}"
        )
        rep["detail_mesh"] = st

        # Viewer layer: the browser cannot hold 5-10M embedded triangles —
        # a lighter re-mesh of the SAME region carries the full-resolution
        # photo atlas, so the default view stays photorealistic.
        if len(detail.faces) <= 1_500_000:
            viewer_layer = out_mesh if out_mesh.texture is not None else detail
        elif sub_cloud is not None:
            try:
                from scantobim.core.freeform import (
                    freeform_mesh_from_points,
                    sharpen_mesh_with_planes,
                )

                light = freeform_mesh_from_points(
                    sub_cloud, max_faces=1_200_000
                )
                if light is not None:
                    if result is not None and result.surfaces:
                        sharpen_mesh_with_planes(
                            light, result.surfaces,
                            light.freeform_stats["voxel"],
                        )
                    if photo_cams is not None and images_dir is not None:
                        # Same GSD texel as the full detail atlas — reduced
                        # GEOMETRY, full TEXTURE sharpness in the viewer.
                        lv_stats: dict = {}
                        lt = bake_photo_atlas(
                            light, photo_cams, images_dir,
                            transform=transform, stats_out=lv_stats,
                            max_atlas=8192, max_pages=3, image_map=image_map,
                        )
                        if (
                            lt is not None
                            and lv_stats.get("photo_fraction", 0.0) >= 0.15
                        ):
                            light = lt
                    viewer_layer = light
                    print(
                        f"  Viewer-Ansicht: Detail-Mesh mit "
                        f"{len(light.faces):,} Dreiecken eingebettet "
                        f"(Datei {detail_glb.name} trägt die volle Auflösung)"
                    )
            except Exception as exc:  # noqa: BLE001 — viewer layer optional
                print(f"  Viewer-Detailansicht übersprungen ({exc})")
    except Exception as exc:  # noqa: BLE001 — detail is a bonus layer
        print(f"  Detail-Mesh übersprungen ({exc})")
    return viewer_layer


def _bake_full_photo_atlas(
    full_viewer, camera_source, images_dir, transform, rep, output,
    image_map=None,
):
    """Bake the photo atlas onto the complete-mesh viewer layer.

    Returns the textured mesh (and writes ``<stem>_foto.glb``) when enough
    texels could be sampled from photos; otherwise returns ``full_viewer``
    unchanged — vertex colors stay active. Best-effort: never raises.
    """
    try:
        from scantobim.core.phototex import bake_photo_atlas

        print("Foto-Textur: Atlas in Foto-Auflösung wird auf das "
              "Komplett-Mesh projiziert …")
        px_stats: dict = {}
        textured = bake_photo_atlas(
            full_viewer, camera_source, images_dir,
            transform=transform, stats_out=px_stats, image_map=image_map,
        )
        if textured is not None and px_stats.get("photo_fraction", 0.0) >= 0.15:
            aw, ah = px_stats["atlas"]
            print(
                f"  → Atlas {aw}×{ah} px, {px_stats['texel_cm']:.1f} cm/Texel, "
                f"{px_stats['photo_fraction'] * 100:.0f}% Foto-Anteil "
                f"({px_stats['cameras_used']} Kameras)"
            )
            if "komplett_mesh" in rep:
                rep["komplett_mesh"]["foto_textur"] = px_stats
            foto_glb = output.with_name(output.stem + "_foto.glb")
            write_mesh(textured, foto_glb)
            print(f"wrote {foto_glb} (Komplett-Mesh der ganzen Szene mit "
                  "Foto-Atlas, reduzierte Vorschau-Auflösung — das "
                  "hochaufgelöste Gebäudemodell ist _detail.glb)")
            return textured
        if textured is not None:
            print("  Foto-Anteil zu gering — Vertex-Farben bleiben aktiv")
    except Exception as exc:  # noqa: BLE001 — atlas is best-effort
        print(f"  Foto-Textur übersprungen ({exc})")
    return full_viewer


def _write_full_only(output, full_mesh, full_viewer, report, report_path) -> int:
    """Deliver the complete-mesh-only result (structure skipped or failed)."""
    from scantobim.core.mesh import Mesh as _Mesh

    if full_mesh is None:
        raise ValueError(
            "weder Strukturmodell noch Komplett-Mesh möglich — Scan prüfen"
        )
    ext = output.suffix.lower()
    if ext in (".html", ".htm"):
        empty = _Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
        out = write_mesh(
            empty, output,
            freeform=full_viewer, freeform_label="Komplett-Mesh (Scan)",
        )
    else:
        out = write_mesh(full_mesh, output)
    print(f"wrote {out}")
    glb = output.with_name(output.stem + "_komplett.glb")
    write_mesh(full_mesh, glb)
    print(f"wrote {glb} (Komplett-Mesh, volle Auflösung)")
    _print_gpu_usage(report)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, default=_json_default))
        print(f"wrote {report_path}")
    return 0


def _write_deviation(result, cloud, deviation_path: Path, tolerance: float) -> None:
    """As-built QA: deviation-colored scan + statistics into the report."""
    from scantobim.core.deviation import building_only_mesh, deviation_analysis

    rep = result.report
    alignment = np.array(rep["alignment"]) if "alignment" in rep else None
    stats, dev_cloud = deviation_analysis(
        building_only_mesh(result.mesh), cloud,
        tolerance=tolerance, alignment=alignment,
    )
    write_point_cloud(dev_cloud, deviation_path)
    rep["deviation"] = stats
    fid = stats.get("fidelity")
    if fid:
        print(
            f"  Soll-Ist (Modelltreue): RMS {fid['rms'] * 1000:.1f} mm | "
            f"P95 {fid['p95'] * 1000:.1f} mm | "
            f"{fid['within_tolerance'] * 100:.1f}% innerhalb ±{tolerance * 1000:.1f} mm"
        )
    print(
        f"  Modellabdeckung: {stats['coverage'] * 100:.1f}% des Scans liegen "
        f"innerhalb ±{stats['coverage_band'] * 100:.0f} cm am Modell"
    )
    if stats.get("coverage_model_area") is not None:
        print(
            f"  … im Modellbereich (Bauwerks-Umfeld +1 m): "
            f"{stats['coverage_model_area'] * 100:.1f}%"
        )
    print(f"wrote {deviation_path} (Abweichungswolke blau-weiss-rot)")


def _write_views(mesh, views_dir: Path) -> None:
    from scantobim.io.views import write_ortho_views

    files = write_ortho_views(mesh, views_dir)
    print(f"wrote {len(files)} Ansichten + World-Files → {views_dir} "
          "(nord/ost/sued/west/draufsicht, maßstabsgetreu)")


CLOUD_EXTS_SORTED = sorted(_CLOUD_EXTS)


def _cmd_register(args) -> int:
    from scantobim.core.registration import (
        apply_transform,
        merge_clouds,
        register_point_to_plane,
    )

    reference = read_point_cloud(args.reference)
    print(f"reference: {args.reference} ({len(reference):,} points)")
    aligned = [reference]
    transforms = {str(args.reference): np.eye(4).tolist()}
    for other in args.others:
        cloud = read_point_cloud(other)
        res = register_point_to_plane(cloud, reference)
        aligned.append(apply_transform(cloud, res.transform))
        transforms[str(other)] = res.transform.tolist()
        status = "converged" if res.converged else f"stopped after {res.iterations} it."
        print(f"  {other}: rmse={res.rmse:.5f} ({status})")

    merged = merge_clouds(aligned, voxel_size=args.voxel)
    write_point_cloud(merged, args.output)
    print(f"wrote {args.output} ({len(merged):,} points)")
    if args.transforms is not None:
        args.transforms.parent.mkdir(parents=True, exist_ok=True)
        args.transforms.write_text(json.dumps(transforms, indent=2))
        print(f"wrote {args.transforms}")
    print(f"next: scantobim reconstruct {args.output} -o model.glb")
    return 0


def _cmd_analyze(args) -> int:
    from scantobim.core.machinery import analyze_machinery, cylinder_mesh
    from scantobim.core.steel import detect_steel_members, steel_report

    cloud = _read_inputs(args.input, register=False)
    print("analyzing rotational geometry (cylinders, shafts, gears) …")
    report = analyze_machinery(cloud, distance_threshold=args.dist)
    objects = report.pop("_objects")

    print(f"  cylinders:   {len(report['cylinders'])}")
    for s in report["shafts"]:
        dias = " → ".join(f"⌀{st['diameter'] * 1000:.1f}mm" for st in s["steps"])
        print(f"  shaft:       {len(s['steps'])} steps, {s['total_length'] * 1000:.1f}mm  [{dias}]")
    for g in report["gears"]:
        print(
            f"  gear:        z={g['teeth']}, m={g['module'] * 1000:.2f}mm, "
            f"da={g['tip_diameter'] * 1000:.1f}mm, b={g['width'] * 1000:.1f}mm"
        )
    for st in report["gear_stages"]:
        print(f"  gear stage:  i={st['ratio']:.3f}, a={st['center_distance'] * 1000:.1f}mm")
    for c in report["cones"]:
        print(
            f"  cone:        {c['half_angle_deg']:.1f}°, "
            f"⌀{c['diameter_small'] * 1000:.1f}→⌀{c['diameter_large'] * 1000:.1f}mm, "
            f"h={c['height'] * 1000:.1f}mm"
        )

    if not args.no_steel:
        print("matching steel profiles …")
        members = detect_steel_members(cloud)
        report["steel_members"] = steel_report(members)
        for m in report["steel_members"]:
            print(f"  steel:       {m['profile']} ({m['family']}), L={m['length']:.3f}")
        if len(members) >= 2:
            from scantobim.core.connections import connections_report, detect_connections

            nodes = detect_connections(members)
            report["connections"] = connections_report(nodes, members)
            for n in report["connections"]:
                profs = " + ".join(m["profile"] for m in n["members"])
                angles = "/".join(f"{a:.0f}°" for a in n["angles_deg"])
                print(f"  connection:  {profs} @ {angles} "
                      f"(e={n['eccentricity'] * 1000:.0f}mm)")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=_json_default))
        print(f"wrote {args.output}")

    if args.mesh is not None:
        from scantobim.core.geometry3d import (
            cone_mesh,
            involute_gear_mesh,
            steel_member_mesh,
        )
        from scantobim.core.mesh import merge_meshes

        parts = []
        names: dict[int, str] = {}
        for i, c in enumerate(objects["cylinders"]):
            parts.append(
                cylinder_mesh(c.center, c.axis, c.radius, c.length, group=i)
            )
            names[i] = f"zylinder_{i}_d{c.radius * 2000:.0f}mm"
        for j, g in enumerate(objects["gears"]):
            # A coaxial smaller cylinder (the shaft) becomes the gear's bore.
            bore = 0.0
            r_pitch = g.pitch_diameter / 2
            for c in objects["cylinders"]:
                if abs(float(c.axis @ g.axis)) < 0.98 or c.radius > 0.7 * r_pitch:
                    continue
                rel = g.center - c.center
                if np.linalg.norm(rel - (rel @ c.axis) * c.axis) < 0.2 * r_pitch:
                    bore = max(bore, c.radius)
            parts.append(
                involute_gear_mesh(
                    g.center, g.axis, g.teeth, g.module, g.width,
                    bore_radius=bore, group=1000 + j,
                )
            )
            names[1000 + j] = f"zahnrad_z{g.teeth}_m{g.module * 1000:.2f}"
        for k, cn in enumerate(objects.get("cones", [])):
            half = np.deg2rad(cn.half_angle_deg)
            offset = cn.r_min / max(np.tan(half), 1e-9)
            parts.append(
                cone_mesh(
                    cn.apex, cn.axis, cn.r_min, cn.r_max, cn.height,
                    offset=offset, group=3000 + k,
                )
            )
            names[3000 + k] = f"kegel_{cn.half_angle_deg:.0f}deg"
        if not args.no_steel:
            for k, member in enumerate(members):
                parts.append(steel_member_mesh(member, group=2000 + k))
                names[2000 + k] = member.profile.replace(" ", "_")
        if parts:
            model = merge_meshes(parts)
            model.group_names = names
            write_mesh(model, args.mesh)
            print(f"wrote {args.mesh}")
        else:
            print("no primitives detected — skipping mesh output")
    return 0


def _cmd_sheetmetal(args) -> int:
    from scantobim.core.sheetmetal import analyze_sheet_metal

    cloud = _read_inputs(args.input, register=False)
    print("detecting plates and weld seams …")
    report = analyze_sheet_metal(
        cloud,
        distance_threshold=args.dist,
        max_thickness=args.max_thickness,
    )
    objects = report.pop("_objects")

    for p in report["plates"]:
        t_cat = p["thickness_catalog"]
        t_str = f"t={t_cat * 1000:.0f}mm" if t_cat else f"t≈{p['thickness_measured'] * 1000:.1f}mm"
        print(
            f"  plate {p['id']}: {t_str}, "
            f"{p['size'][0] * 1000:.0f} x {p['size'][1] * 1000:.0f} mm, "
            f"{p['weight_kg']:.1f} kg"
        )
    for s in report["weld_seams"]:
        print(
            f"  seam {s['plates'][0]}-{s['plates'][1]}: "
            f"{s['length'] * 1000:.0f} mm, {s['angle_deg']:.0f}°"
        )
    tot = report["totals"]
    print(
        f"  totals: {tot['plate_count']} plates, {tot['weight_kg']:.1f} kg, "
        f"weld length {tot['weld_length'] * 1000:.0f} mm"
    )

    from scantobim.core.unfold import classify_junctions

    kinds = classify_junctions(objects["plates"], objects["seams"])
    for entry, kind in zip(report["weld_seams"], kinds):
        entry["junction"] = kind

    parts = None
    if args.unfold:
        from scantobim.core.unfold import unfold_parts

        parts = unfold_parts(objects["plates"], objects["seams"], k_factor=args.k_factor)
        report["flat_parts"] = [
            {
                "plates": p.plate_ids,
                "bends": len(p.bend_lines),
                "bend_angles_deg": p.bend_angles_deg,
                "size": [round(s, 4) for s in p.size],
                "thickness": p.catalog_thickness or round(p.thickness, 5),
                "flat_area": round(p.outline_area, 4),
            }
            for p in parts
        ]
        for fp in report["flat_parts"]:
            print(
                f"  part {fp['plates']}: {fp['bends']} Kantungen, "
                f"{fp['size'][0] * 1000:.0f} x {fp['size'][1] * 1000:.0f} mm flach"
            )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=_json_default))
        print(f"wrote {args.output}")
    if args.dxf is not None:
        if parts is not None:
            from scantobim.io.dxf import write_flat_pattern_dxf

            out = write_flat_pattern_dxf(parts, args.dxf)
            print(f"wrote {out} (Abwicklungen 1:1 mit Biegelinien)")
        else:
            from scantobim.io.dxf import write_cutting_dxf

            out = write_cutting_dxf(objects["plates"], args.dxf)
            print(f"wrote {out} (Zuschnittkonturen 1:1)")
    return 0


def _cmd_bridge(args) -> int:
    from scantobim.core.bridge import analyze_bridge

    cloud = _read_inputs(args.input, register=False)
    print("analyzing bridge structure …")
    report = analyze_bridge(cloud)

    if args.mesh is not None:
        from scantobim.core.bridge import bridge_model_mesh

        model = bridge_model_mesh(report)
        write_mesh(model, args.mesh)
        print(f"wrote {args.mesh} ({len(model.faces)} triangles, "
              f"{len(model.group_names or {})} Bauteile)")
    report.pop("_objects", None)

    deck = report["deck"]
    print(f"  type:      {report['bridge_type']}")
    print(f"  deck:      {deck['length']:.1f} x {deck['width']:.1f} m, "
          f"OK {deck['elevation']:.2f} m, {deck['area']:.0f} m²")
    print(f"  spans:     {' + '.join(f'{s:.1f}' for s in report['spans'])} m")
    print(f"  piers:     {len(report['piers'])}  |  abutments: {report['abutments']}"
          f"  |  pylons: {report['pylons']}  |  cables: {len(report['cables'])}")
    for p in report["piers"]:
        b = p["bearing_point"]
        print(f"    pier @ station {p['station']:+.1f} m, h={p['height']:.1f} m, "
              f"Lager ({b[0]:.1f}, {b[1]:.1f}, {b[2]:.2f})")
    if report["arch"]:
        a = report["arch"]
        print(f"  arch:      R={a['radius']:.1f} m, Stich {a['rise']:.1f} m")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=_json_default))
        print(f"wrote {args.output}")
    return 0


def _cmd_photos(args) -> int:
    from scantobim.photogrammetry.colmap import photos_to_point_cloud

    work_dir = args.work_dir or args.output.with_name(args.output.stem + "_colmap")
    print(f"running COLMAP on {args.image_dir} (quality: {args.quality}) …")
    cloud = photos_to_point_cloud(
        args.image_dir,
        work_dir,
        quality=args.quality,
        use_gpu=not args.cpu,
        verbose=args.verbose,
    )
    write_point_cloud(cloud, args.output)
    print(f"wrote {args.output} ({len(cloud):,} points)")
    print(f"next: scantobim reconstruct {args.output} -o model.glb")
    return 0


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj)}")


if __name__ == "__main__":
    raise SystemExit(main())
