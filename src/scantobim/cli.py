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


def main(argv: list[str] | None = None) -> int:
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
        choices=["building", "indoor", "object", "fast"],
        help="parameter preset (default: building)",
    )
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
    p_rec.add_argument("--seed", type=int, default=None, help="RANSAC random seed")

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

    args = parser.parse_args(argv)
    try:
        if args.command == "info":
            return _cmd_info(args)
        if args.command == "reconstruct":
            return _cmd_reconstruct(args)
        if args.command == "register":
            return _cmd_register(args)
        if args.command == "analyze":
            return _cmd_analyze(args)
        if args.command == "photos":
            return _cmd_photos(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_info(args) -> int:
    cloud = read_point_cloud(args.input)
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


def _read_inputs(paths: list[Path], register: bool) -> "object":
    """Read one or more clouds; optionally ICP-register onto the first."""
    clouds = []
    for p in paths:
        print(f"reading {p} …")
        cloud = read_point_cloud(p)
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
    cloud = _read_inputs(args.input, args.register_inputs)

    cfg = PipelineConfig.preset(args.preset)
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
    if args.align:
        cfg.align_axes = True
    if args.seed is not None:
        cfg.seed = args.seed

    print(f"reconstructing (preset: {args.preset}) …")
    result = reconstruct(cloud, cfg)
    rep = result.report
    openings = sum(s.get("openings", 0) for s in rep["surfaces"])
    q = rep["quantities"]
    print(f"  planes:    {rep['planes']}")
    print(f"  angles:    {rep['plane_angles']}")
    print(f"  corners:   {rep['exact_corners']}")
    print(f"  openings:  {openings}")
    print(f"  classes:   {q['surface_count_by_class']}")
    if "volume" in q:
        print(f"  volume:    {q['volume']} (watertight, {q['orientation']} normals)")
    print(f"  mesh:      {rep['mesh']['vertices']} vertices, {rep['mesh']['triangles']} triangles")
    print(f"  residual:  {rep['residual_points']} unexplained points")
    print(f"  runtime:   {rep['runtime_seconds']} s")

    ext = args.output.suffix.lower()
    if ext in (".stp", ".step"):
        from scantobim.io.step import write_step

        out = write_step(result.surfaces, args.output)
        print(f"wrote {out} (STEP AP214 — in HiCAD über Datei > Import > STEP laden)")
    elif ext == ".ifc":
        from scantobim.io.ifc import write_ifc

        out = write_ifc(result.surfaces, args.output)
        print(f"wrote {out} (IFC4)")
    else:
        out = write_mesh(result.mesh, args.output)
        print(f"wrote {out}")

    if args.floorplan is not None:
        from scantobim.io.dxf import write_floorplan_dxf

        plan = write_floorplan_dxf(
            result.mesh, args.floorplan, height_above_floor=args.floorplan_height
        )
        print(f"wrote {plan} (slice at base + {args.floorplan_height})")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, indent=2, default=_json_default))
        print(f"wrote {args.report}")
    if args.residual_out is not None and len(result.residual):
        write_point_cloud(result.residual, args.residual_out)
        print(f"wrote {args.residual_out}")
    return 0


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

    if not args.no_steel:
        print("matching steel profiles …")
        members = detect_steel_members(cloud)
        report["steel_members"] = steel_report(members)
        for m in report["steel_members"]:
            print(f"  steel:       {m['profile']} ({m['family']}), L={m['length']:.3f}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=_json_default))
        print(f"wrote {args.output}")

    if args.mesh is not None:
        from scantobim.core.mesh import merge_meshes

        parts = []
        for i, c in enumerate(objects["cylinders"]):
            parts.append(
                cylinder_mesh(c.center, c.axis, c.radius, c.length, group=i)
            )
        for j, g in enumerate(objects["gears"]):
            parts.append(
                cylinder_mesh(
                    g.center, g.axis, g.tip_diameter / 2, g.width,
                    color=(230, 180, 120), group=1000 + j,
                )
            )
        if parts:
            write_mesh(merge_meshes(parts), args.mesh)
            print(f"wrote {args.mesh}")
        else:
            print("no primitives detected — skipping mesh output")
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
