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

    print(f"reconstructing (preset: {args.preset}) …")
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
    rep = result.report
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
        out = write_mesh(output_mesh, args.output)
        print(f"wrote {out}")

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
    from scantobim.io.project import read_trajectory, scan_project_dir

    project = scan_project_dir(args.directory)
    if project.cloud is None:
        raise ValueError(
            f"{args.directory}: keine Punktwolke im Projektordner gefunden "
            f"(gesucht: {' '.join(CLOUD_EXTS_SORTED)})"
        )
    print(f"SLAM-Projekt: {args.directory}")
    for line in project.describe():
        print(f"  {line}")

    cloud = read_point_cloud(project.cloud, max_points=args.max_points)
    print(f"  geladen: {len(cloud):,} Punkte"
          + (", mit Farben" if cloud.colors is not None else ""))

    trajectory = None
    if project.trajectory is not None:
        trajectory = read_trajectory(project.trajectory)
        print(f"  Trajektorie: {len(trajectory)} Positionen "
              "(Normalen werden zum Scanner orientiert)")

    cfg = PipelineConfig.preset(
        args.preset if args.preset != "auto" else "building"
    )
    from scantobim.core.pipeline import SOURCE_PROFILES, apply_source_profile

    source = getattr(args, "source", "slam")
    apply_source_profile(cfg, source)
    for key, value in (getattr(args, "advanced", None) or {}).items():
        setattr(cfg, key, value)
    if args.watertight:
        cfg.watertight = True
    if args.align:
        cfg.align_axes = True
    if args.seed is not None:
        cfg.seed = args.seed

    print(f"reconstructing (preset: {args.preset}, Quelle: {source}) …")
    if args.preset == "auto":
        from scantobim.core.autotune import auto_reconstruct

        result = auto_reconstruct(
            cloud, trajectory=trajectory, seed=args.seed,
            overrides={
                **SOURCE_PROFILES.get(source, {}),
                **(getattr(args, "advanced", None) or {}),
                "watertight": cfg.watertight,
                "align_axes": cfg.align_axes,
            },
        )
    else:
        result = reconstruct(cloud, cfg, trajectory=trajectory)
    rep = result.report
    print(f"  planes: {rep['planes']}  angles: {rep['plane_angles']}  "
          f"residual: {rep['residual_points']}")

    # Texture: photo projection when poses + photos exist, else cloud colors.
    output_mesh = result.mesh
    texture_info: dict = {"source": "keine"}
    use_photos = (
        not args.no_photos
        and project.colmap_model is not None
        and project.images_dir is not None
    )
    transform = None
    if "alignment" in rep:
        transform = np.array(rep["alignment"])

    if use_photos:
        # Sanity check: the camera path must live in the same coordinate
        # frame as the cloud — SLAM exports sometimes keep poses in a local
        # session frame while the cloud is georeferenced.
        from scantobim.photogrammetry.colmap import read_colmap_model

        cams = read_colmap_model(project.colmap_model)
        centers = np.array([-c.rotation.T @ c.translation for c in cams])
        lo, hi = cloud.aabb
        diag = float(np.linalg.norm(hi - lo))
        dist = np.linalg.norm(centers - (lo + hi) / 2.0, axis=1)
        if float(np.median(dist)) > 2.0 * diag:
            print("Hinweis: Kameraposen liegen weit außerhalb der Punktwolke "
                  "(anderes Koordinatensystem?) — Foto-Projektion übersprungen")
            use_photos = False

    if use_photos:
        try:
            from scantobim.core.texture import bake_texture_from_photos

            print(f"projecting {project.image_count} photos onto the model …")
            stats: dict = {}
            photo_mesh = bake_texture_from_photos(
                result, project.colmap_model, project.images_dir,
                texel_size=args.texel, transform=transform, stats_out=stats,
            )
            coverage = stats.get("coverage", 0.0)
            print(f"  Foto-Abdeckung: {coverage * 100:.0f}% der Flächen "
                  f"({stats.get('images_used', 0)} Fotos verwendet)")
            if coverage < 0.2:
                print("Hinweis: Foto-Abdeckung zu gering — "
                      "verwende stattdessen Punktwolken-Farben")
                use_photos = False
            else:
                output_mesh = photo_mesh
                th, tw = output_mesh.texture.shape[:2]
                print(f"  texture atlas: {tw} x {th} px")
                texture_info = {
                    "source": "foto-projektion",
                    "coverage": round(coverage, 3),
                    "images_used": stats.get("images_used", 0),
                }
        except Exception as exc:  # noqa: BLE001 — fall back, don't fail the model
            print(f"Hinweis: Foto-Projektion fehlgeschlagen ({exc}) — "
                  "verwende Punktwolken-Farben")
            use_photos = False
    no_texture = getattr(args, "no_texture", False)
    if not use_photos and not no_texture:
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

    output = args.output or (args.directory / "scantobim_modell.html")
    ext = output.suffix.lower()
    if ext in (".stp", ".step"):
        from scantobim.io.step import write_step

        out = write_step(result.surfaces, output)
    elif ext == ".ifc":
        from scantobim.io.ifc import write_ifc

        out = write_ifc(result.surfaces, output, storeys=rep.get("storeys"))
    else:
        out = write_mesh(output_mesh, output)
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
