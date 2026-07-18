"""End-to-end reconstruction pipeline: point cloud → clean-edged mesh."""

from __future__ import annotations

import colorsys
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from scantobim.core.cloud import PointCloud
from scantobim.core.edges import (
    build_corners,
    build_intersection_lines,
    snap_points_to_corners,
    snap_points_to_line,
)
from scantobim.core.mesh import Mesh, merge_meshes, triangulate_with_holes, weld_vertices
from scantobim.core.planes import detect_planes, plane_adjacency
from scantobim.core.polygons import (
    alpha_shape_loops,
    dedupe_polygon,
    point_in_polygon,
    polygon_area,
    remove_collinear,
    simplify_polygon,
    straighten_polygon,
)
from scantobim.core.semantics import classify_surfaces, detect_storeys, quantity_takeoff
from scantobim.core.transform import apply_alignment, compute_alignment
from scantobim.core.preprocess import (
    estimate_normals,
    estimate_point_spacing,
    local_point_spacing,
    remove_edge_artifacts,
    remove_statistical_outliers,
    voxel_downsample,
)
from scantobim.core.regularize import regularize_planes, snapped_angles_report


# Reference size for ratio-based minimum-inlier thresholds: percentages are
# meant relative to a "normal" scan — on multi-million-point clouds they must
# not grow without bound, or small real surfaces become undetectable.
_RATIO_REF = 2_000_000


@dataclass
class PipelineConfig:
    """All tuning knobs of the reconstruction, with sane data-driven defaults.

    Every length not set explicitly is derived from the measured point
    spacing of the (downsampled) cloud, so the same preset works for a
    room scan in meters and a tabletop scan in millimeters.
    """

    # --- preprocessing ---
    voxel_size: float | None = None  # None = auto (2x raw spacing), 0 = off
    sor_neighbors: int = 16
    sor_std_ratio: float = 2.5
    normal_neighbors: int = 16

    # --- real-world scan hardening ---
    edge_artifact_filter: bool = True  # remove mixed-pixel strings at silhouettes
    ghost_offset_tol: float = 0.0  # merge registration ghosts within this offset (m)
    adaptive_density: bool = True  # per-surface tolerances from local point spacing

    # --- plane detection ---
    distance_threshold: float | None = None  # None = auto (distance_factor x spacing)
    distance_factor: float = 3.0  # used when distance_threshold is None
    normal_threshold_deg: float = 30.0
    min_inlier_ratio: float = 0.01  # fraction of cloud size
    min_inliers_abs: int = 60
    max_planes: int = 64
    ransac_iterations: int = 600
    seed: int = 7

    # --- detail geometry: cylinders (columns, pipes) in the residual ---
    cylinder_detection: bool = False
    max_cylinders: int = 16

    # --- detail recovery: second, finer plane pass on the residual ---
    detail_recovery: bool = False
    detail_min_inliers: int = 40
    max_detail_planes: int = 32

    # --- regularization ---
    regularize: bool = True
    parallel_tol_deg: float = 8.0
    ortho_tol_deg: float = 8.0
    merge_offset_factor: float = 4.0  # x distance_threshold

    # --- polygons & edges ---
    alpha_factor: float = 4.0  # x spacing
    simplify_factor: float = 2.0  # x spacing
    straighten: bool = True
    straighten_angle_tol_deg: float = 14.0
    edge_snap_factor: float = 8.0  # x spacing
    corner_snap_factor: float = 12.0  # x spacing
    contact_factor: float = 6.0  # x spacing (plane adjacency)

    # --- openings (windows / door cutouts as holes in surfaces) ---
    detect_openings: bool = True
    min_opening_factor: float = 8.0  # opening area >= (factor x spacing)^2

    # --- mesh ---
    orient: str = "outward"  # "outward" | "inward" | "none"
    weld_factor: float = 0.25  # x spacing
    color_surfaces: bool = True
    watertight: bool = False  # global PolyFit optimization (closes scan shadows)

    # --- output coordinate system ---
    align_axes: bool = False  # rotate dominant directions onto X/Y/Z, floor at Z=0

    @classmethod
    def preset(cls, name: str) -> "PipelineConfig":
        """Built-in presets: ``building``, ``indoor``, ``object``, ``fast``."""
        if name == "building":
            return cls(orient="outward")
        if name == "indoor":
            return cls(
                orient="inward",
                normal_threshold_deg=35.0,
                edge_snap_factor=10.0,
            )
        if name == "object":
            return cls(
                orient="outward",
                min_inlier_ratio=0.02,
                parallel_tol_deg=10.0,
                ortho_tol_deg=10.0,
            )
        if name == "fast":
            return cls(ransac_iterations=250, max_planes=32, sor_neighbors=8)
        if name == "detail":
            # Detail-faithful: keep small regions, more planes, finer
            # simplification, and reconstruct cylindrical members (columns,
            # pipes) in the residual as true cylinders.
            return cls(
                orient="outward",
                min_inlier_ratio=0.004,
                min_inliers_abs=40,
                max_planes=128,
                ransac_iterations=1200,
                alpha_factor=3.5,
                simplify_factor=1.5,
                min_opening_factor=6.0,
                cylinder_detection=True,
                detail_recovery=True,
            )
        raise ValueError(
            f"Unknown preset {name!r} (use building/indoor/object/detail/fast)"
        )


# Source profiles: the SCENE preset says what was scanned, the source
# profile says WHICH SENSOR captured it — layered on top of any preset.
# Values reflect the sensors' noise/drift characteristics.
SOURCE_PROFILES: dict[str, dict] = {
    # neutral default
    "standard": {},
    # Handheld SLAM (SHARE SLAM S20, GeoSLAM, LiGrip …): cm-level drift,
    # registration ghosts from loop closures, denser noise.
    "slam": {
        "distance_factor": 3.5,
        "ghost_offset_tol": 0.03,
        "sor_std_ratio": 2.0,
        "normal_neighbors": 20,
    },
    # Terrestrial laser scanner on tripod: mm noise, crisp edges.
    "tls": {
        "distance_factor": 2.5,
        "simplify_factor": 1.5,
    },
    # Drone photogrammetry / aerial LiDAR: noisier surfaces, outliers.
    "drohne": {
        "distance_factor": 4.0,
        "sor_std_ratio": 2.0,
        "min_inlier_ratio": 0.012,
    },
    # iPhone/iPad LiDAR apps: coarse depth, smoothed geometry.
    "iphone": {
        "distance_factor": 5.0,
        "min_inlier_ratio": 0.015,
        "normal_neighbors": 24,
    },
}


def apply_source_profile(cfg: PipelineConfig, source: str) -> PipelineConfig:
    """Layer a sensor profile over the config (unknown names = neutral)."""
    for key, value in SOURCE_PROFILES.get(source, {}).items():
        setattr(cfg, key, value)
    return cfg


@dataclass
class SurfaceGeometry:
    """Exact polygonal geometry of one reconstructed surface.

    This is the CAD-grade representation (planar face with outer boundary
    and hole loops) that feeds the STEP and IFC exporters — the triangle
    mesh is derived from it for visualization formats.
    """

    plane_index: int
    normal: np.ndarray  # unit normal (faces outward/inward per config)
    outer: np.ndarray  # (N, 3) CCW around the normal
    holes: list  # list of (M, 3) arrays, CCW around the normal
    surface_class: str = ""
    name: str = ""


@dataclass
class ReconstructionResult:
    mesh: Mesh
    residual: PointCloud  # points not explained by any planar surface
    surfaces: list = field(default_factory=list)  # list[SurfaceGeometry]
    report: dict = field(default_factory=dict)


def preprocess_signature(cfg: PipelineConfig) -> tuple:
    """The config fields that influence preprocessing — candidates sharing
    this signature can share one preprocessing pass (auto-tuning)."""
    return (
        cfg.voxel_size, cfg.sor_neighbors, cfg.sor_std_ratio,
        cfg.edge_artifact_filter, cfg.normal_neighbors, cfg.ghost_offset_tol,
    )


def preprocess_cloud(
    cloud: PointCloud,
    cfg: PipelineConfig,
    trajectory: np.ndarray | None = None,
) -> tuple[PointCloud, float, dict]:
    """Stage 0-1: thin, denoise, estimate + orient normals.

    Returns ``(work, spacing, report_fields)`` — reusable across candidate
    runs whose ``preprocess_signature`` matches.
    """
    pre: dict = {}
    raw_spacing = estimate_point_spacing(cloud)
    voxel = cfg.voxel_size if cfg.voxel_size is not None else 2.0 * raw_spacing
    work = voxel_downsample(cloud, voxel) if voxel and voxel > 0 else cloud
    work, _ = remove_statistical_outliers(work, cfg.sor_neighbors, cfg.sor_std_ratio)
    if cfg.edge_artifact_filter and len(work) > 100:
        n_before = len(work)
        # Buildings have no legitimate string-like geometry — filter dense
        # mixed-pixel strings too (linearity-only criterion).
        work, _ = remove_edge_artifacts(work, require_sparse=False)
        pre["edge_artifacts_removed"] = n_before - len(work)
    if len(work) < 16:
        raise ValueError(
            f"Too few points after preprocessing ({len(work)}); "
            "check input units or lower voxel_size"
        )
    work = estimate_normals(work, cfg.normal_neighbors)
    if trajectory is not None and len(trajectory):
        from scantobim.core.preprocess import orient_normals_along_trajectory

        ostats: dict = {}
        work = orient_normals_along_trajectory(work, trajectory, stats_out=ostats)
        pre["trajectory_positions"] = int(len(trajectory))
        if ostats.get("mode"):
            pre["normalen_orientierung"] = ostats["mode"]
    spacing = estimate_point_spacing(work)
    if spacing <= 0:
        raise ValueError("Degenerate point cloud (zero spacing)")
    pre["preprocessed_points"] = len(work)
    pre["point_spacing"] = round(spacing, 6)
    return work, spacing, pre


def reconstruct(
    cloud: PointCloud,
    config: PipelineConfig | None = None,
    trajectory: np.ndarray | None = None,
    preprocessed: tuple | None = None,
) -> ReconstructionResult:
    """Run the full scan-to-model pipeline on ``cloud``.

    ``trajectory``: optional (N, 3) scanner path (SLAM/mobile mapping) —
    normals are oriented towards the nearest scanner position, which makes
    detection more robust on real-world scans.

    ``preprocessed``: optional ``preprocess_cloud`` result to reuse — the
    auto-tuner preprocesses ONCE for all candidates instead of per run.
    """
    cfg = config or PipelineConfig()
    report: dict = {"input_points": len(cloud), "config": _config_dict(cfg)}
    t0 = time.perf_counter()

    # ---- 1. preprocessing (or reuse of a shared pass) -----------------------
    if preprocessed is None:
        preprocessed = preprocess_cloud(cloud, cfg, trajectory)
    work, spacing, pre = preprocessed
    report.update(pre)

    # ---- 2. plane detection ------------------------------------------------
    dist_thresh = (
        cfg.distance_threshold
        if cfg.distance_threshold is not None
        else cfg.distance_factor * spacing
    )
    # The ratio-based minimum counts against at most _RATIO_REF points: a
    # straight percentage explodes on dense scans (1% of a 10M-point scan
    # demands 100k-point ≈ 6 m² planes — window reveals, piers and small
    # roof faces would never be detected).
    min_inliers = max(
        cfg.min_inliers_abs,
        int(cfg.min_inlier_ratio * min(len(work), _RATIO_REF)),
    )
    planes, unassigned = detect_planes(
        work.points,
        work.normals,
        distance_threshold=dist_thresh,
        normal_threshold_deg=cfg.normal_threshold_deg,
        min_inliers=min_inliers,
        max_planes=cfg.max_planes,
        ransac_iterations=cfg.ransac_iterations,
        seed=cfg.seed,
    )
    if not planes:
        # Rescue pass: un-merged SLAM registration ghosts (double walls a
        # few cm apart) fill the whole tolerance band, so every candidate
        # fails the noise gates and the search dies with zero planes.
        # Retry once with widened band and relaxed gates instead of
        # aborting a multi-minute run.
        print(
            "  keine Ebenen mit Standard-Toleranzen gefunden — zweiter "
            "Versuch mit gelockerten Toleranzen (SLAM-Doppelwände?) …"
        )
        rescue_thresh = 1.6 * dist_thresh
        planes, unassigned = detect_planes(
            work.points,
            work.normals,
            distance_threshold=rescue_thresh,
            normal_threshold_deg=cfg.normal_threshold_deg,
            min_inliers=max(cfg.min_inliers_abs, min_inliers // 2),
            max_planes=cfg.max_planes,
            ransac_iterations=cfg.ransac_iterations,
            seed=cfg.seed,
            max_rms_ratio=0.7,
            core_fraction_min=0.30,
            surface_variation_max=0.02,
        )
        if planes:
            dist_thresh = rescue_thresh
            report["rescue"] = (
                "ebenen-erkennung mit gelockerten toleranzen wiederholt "
                "(tipp: bei SLAM-Scannern Quelle 'SLAM-Handscanner' wählen — "
                "verschmilzt Registrierungs-Doppelwände)"
            )
    if not planes:
        raise ValueError(
            "No planar structure found — for organic shapes use a Poisson-based "
            "tool; this pipeline targets built environments and CAD-like objects"
        )

    # ---- 3. regularization -------------------------------------------------
    if cfg.regularize:
        planes = regularize_planes(
            planes,
            work.points,
            parallel_tol_deg=cfg.parallel_tol_deg,
            ortho_tol_deg=cfg.ortho_tol_deg,
            merge_offset_tol=cfg.merge_offset_factor * dist_thresh,
            ghost_offset_tol=cfg.ghost_offset_tol,
        )
    _orient_planes(planes, work.points, cfg.orient)
    report["planes"] = len(planes)
    report["plane_angles"] = snapped_angles_report(planes)

    # ---- 4a. global watertight optimization (PolyFit) — replaces the
    # greedy per-plane polygonization when requested.
    if cfg.watertight:
        return _reconstruct_watertight(work, planes, spacing, cfg, report, t0)

    # ---- 3b. detail geometry: cylinders first (more specific than planes —
    # a column's shell would otherwise be eaten as small planar facets by
    # the finer plane pass below).
    cylinders = []
    if cfg.cylinder_detection:
        cylinders, unassigned = _detect_residual_cylinders(
            work, unassigned, dist_thresh, cfg
        )

    # ---- 3c. detail recovery: a second, finer pass over the residual picks
    # up small true surfaces (reveals, ledges, niches) that the main pass
    # skipped because of its size threshold. The noise gates inside
    # detect_planes keep diffuse clutter out.
    if cfg.detail_recovery:
        detail_planes, unassigned = _recover_detail_planes(
            work, unassigned, dist_thresh, cfg
        )
        if detail_planes:
            _orient_planes(detail_planes, work.points, cfg.orient)
            planes = planes + detail_planes
            report["detail_surfaces"] = len(detail_planes)
            report["planes"] = len(planes)

    # ---- 4. exact edge geometry ---------------------------------------------
    contact_radius = cfg.contact_factor * spacing
    adjacency = plane_adjacency(planes, work.points, contact_radius)
    lines = build_intersection_lines(planes, adjacency)
    corners = build_corners(
        planes, adjacency, max_distance_from_support=3.0 * contact_radius, points=work.points
    )
    report["adjacent_plane_pairs"] = len(lines)
    report["exact_corners"] = len(corners)

    # ---- 5. per-plane polygons, snapped to shared edges ---------------------
    lines_by_plane: dict[int, list] = {}
    for line in lines:
        lines_by_plane.setdefault(line.plane_i, []).append(line)
        lines_by_plane.setdefault(line.plane_j, []).append(line)
    corners_by_plane: dict[int, list] = {}
    for corner in corners:
        for p in corner.planes:
            corners_by_plane.setdefault(p, []).append(corner)

    # Local density map: near/far scanner resolution varies, so each surface
    # gets tolerances from ITS spacing instead of one global number.
    plane_spacing: dict[int, float] = {}
    if cfg.adaptive_density:
        local_sp = local_point_spacing(work.points)
        for pi, plane in enumerate(planes):
            plane_spacing[pi] = float(np.median(local_sp[plane.inliers]))

    palette = _make_palette(len(planes))
    parts: list[Mesh] = []
    geometries: list[SurfaceGeometry] = []
    plane_reports = []
    # Every plane's polygonization (Delaunay, boundary tracing, snapping)
    # is independent of the others — run them across the cores instead of
    # one after another; results are collected in plane order.
    from concurrent.futures import ThreadPoolExecutor

    import os as _os

    with ThreadPoolExecutor(
        max_workers=max(1, min(8, (_os.cpu_count() or 4)))
    ) as _pool:
        _futures = [
            _pool.submit(
                _reconstruct_plane,
                plane,
                pi,
                work.points,
                plane_spacing.get(pi, spacing),
                cfg,
                lines_by_plane.get(pi, []),
                corners_by_plane.get(pi, []),
                palette[pi],
            )
            for pi, plane in enumerate(planes)
        ]
        _results = [f.result() for f in _futures]
    for part, info, geometry in _results:
        plane_reports.append(info)
        if part is not None:
            parts.append(part)
        if geometry is not None:
            geometries.append(geometry)
    report["surfaces"] = plane_reports

    if not parts:
        raise ValueError("All detected planes were rejected during polygonization")

    # ---- 6. assemble --------------------------------------------------------
    mesh = merge_meshes(parts)
    mesh = weld_vertices(mesh, cfg.weld_factor * spacing)
    if not cfg.color_surfaces:
        mesh.vertex_colors = None

    residual = work.select(unassigned)

    # ---- 7. world alignment (optional) --------------------------------------
    plane_normals = {pi: planes[pi].normal.copy() for pi in range(len(planes))}
    if cfg.align_axes:
        alignment = compute_alignment(planes)
        mesh, res_pts, final_t = apply_alignment(mesh, residual.points, alignment)
        residual.points = res_pts
        plane_normals = {
            pi: alignment[:3, :3] @ n for pi, n in plane_normals.items()
        }
        rot, trans = final_t[:3, :3], final_t[:3, 3]
        for geo in geometries:
            geo.outer = geo.outer @ rot.T + trans
            geo.holes = [h @ rot.T + trans for h in geo.holes]
            geo.normal = rot @ geo.normal
        for c in cylinders:
            c.center = rot @ c.center + trans
            c.axis = rot @ c.axis
        report["alignment"] = [[round(float(v), 8) for v in row] for row in final_t]

    # ---- 8. semantics: classification + quantity takeoff --------------------
    surface_ids = sorted({int(g) for g in mesh.face_groups}) if mesh.face_groups is not None else []
    mean_z: dict[int, float] = {}
    surf_area: dict[int, float] = {}
    for gid in surface_ids:
        sel = mesh.face_groups == gid
        tri = mesh.vertices[mesh.faces[sel]]
        areas = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
        )
        surf_area[gid] = float(areas.sum())
        mean_z[gid] = float(tri[:, :, 2].mean())
    classes = classify_surfaces(
        [plane_normals[g] for g in surface_ids], [mean_z[g] for g in surface_ids]
    )
    from scantobim.core.semantics import refine_roof_classes, refine_terrain_classes

    rms_of = {
        info["plane"]: info.get("rms")
        for info in plane_reports
        if info.get("status") == "ok"
    }
    classes = refine_terrain_classes(classes, [rms_of.get(g) for g in surface_ids])
    classes = refine_roof_classes(classes, [mean_z[g] for g in surface_ids])
    class_of = dict(zip(surface_ids, classes))
    mesh.group_names = {g: f"{class_of[g]}_{g:03d}" for g in surface_ids}
    for info in plane_reports:
        if info.get("status") == "ok" and info["plane"] in class_of:
            info["class"] = class_of[info["plane"]]
    for geo in geometries:
        if geo.plane_index in class_of:
            geo.surface_class = class_of[geo.plane_index]
            geo.name = mesh.group_names[geo.plane_index]
    report["quantities"] = quantity_takeoff(mesh, class_of, surf_area)
    # Volume uncertainty: shifting face g by its σ changes V by A_g·σ_g;
    # root-sum-square over all faces (first order).
    sigma_by_plane = {
        info["plane"]: info.get("sigma_offset", 0.0)
        for info in plane_reports
        if info.get("status") == "ok"
    }
    if "volume" in report["quantities"]:
        vol_sigma = float(
            np.sqrt(
                sum(
                    (surf_area.get(g, 0.0) * sigma_by_plane.get(g, 0.0)) ** 2
                    for g in surface_ids
                )
            )
        )
        report["quantities"]["volume_sigma"] = round(vol_sigma, 5)
    report["storeys"] = detect_storeys(
        [mean_z[g] for g in surface_ids if class_of[g] in ("floor", "slab", "ceiling")]
    )
    roof_faces = [g for g in geometries if g.surface_class == "roof"]
    if roof_faces:
        from scantobim.core.semantics import roof_report

        report["roof"] = roof_report(roof_faces, surf_area)

    opening_details = _classify_openings(
        geometries, float(mesh.vertices[:, 2].min())
    )
    if opening_details:
        report["opening_details"] = opening_details
        q = report["quantities"]
        q["windows"] = sum(1 for o in opening_details if o["type"] == "fenster")
        q["doors"] = sum(1 for o in opening_details if o["type"] == "tuer")

    # ---- 8b. add detail cylinders as true solids ----------------------------
    if cylinders:
        from scantobim.core.machinery import cylinder_mesh

        cyl_parts = []
        names = dict(mesh.group_names or {})
        for k, c in enumerate(cylinders):
            gid = 10000 + k
            cyl_parts.append(
                cylinder_mesh(
                    c.center, c.axis, c.radius, c.length,
                    color=(186, 178, 168), group=gid,
                )
            )
            names[gid] = f"zylinder_{k:03d}"
        mesh = merge_meshes([mesh] + cyl_parts)
        mesh.group_names = names
        report["cylinders"] = [
            {
                "radius": round(c.radius, 4),
                "length": round(c.length, 3),
                "axis": [round(float(x), 4) for x in c.axis],
                "center": [round(float(x), 4) for x in c.center],
                "rms": round(c.rms, 5),
            }
            for c in cylinders
        ]

    report["mesh"] = {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.faces),
        "surface_area": round(mesh.area(), 4),
        **mesh.edge_stats(),
    }
    report["residual_points"] = len(residual)
    report["runtime_seconds"] = round(time.perf_counter() - t0, 3)
    return ReconstructionResult(
        mesh=mesh, residual=residual, surfaces=geometries, report=report
    )


def _classify_openings(geometries, floor_z: float) -> list[dict]:
    """Window vs. door for every opening in a wall, with build measurements.

    Doors reach (almost) down to the floor and are man-height; everything
    else is a window with its sill height (Brüstungshöhe).
    """
    details = []
    up = np.array([0.0, 0.0, 1.0])
    for geo in geometries:
        if geo.surface_class != "wall":
            continue
        n = geo.normal / max(np.linalg.norm(geo.normal), 1e-12)
        h_dir = np.cross(up, n)
        nh = float(np.linalg.norm(h_dir))
        if nh < 1e-9:
            continue
        h_dir /= nh
        for hole in geo.holes:
            z_min = float(hole[:, 2].min())
            z_max = float(hole[:, 2].max())
            height = z_max - z_min
            t = hole @ h_dir
            width = float(t.max() - t.min())
            # Occlusion shadows and noise punch small holes into walls —
            # a real window/door is at least 25 cm in both directions.
            if width < 0.25 or height < 0.25:
                continue
            sill = z_min - floor_z
            kind = "tuer" if sill < 0.3 and height > 1.6 else "fenster"
            details.append(
                {
                    "surface": geo.name,
                    "type": kind,
                    "width": round(width, 3),
                    "height": round(height, 3),
                    "sill_height": round(sill, 3),
                }
            )
    return details


def _recover_detail_planes(work, unassigned, dist_thresh, cfg):
    """Finer plane pass on the residual; returns ``(planes, new_unassigned)``."""
    idx = np.flatnonzero(unassigned)
    if len(idx) < 3 * cfg.detail_min_inliers:
        return [], unassigned
    detail, sub_unassigned = detect_planes(
        work.points[idx],
        work.normals[idx],
        distance_threshold=0.8 * dist_thresh,
        normal_threshold_deg=cfg.normal_threshold_deg,
        min_inliers=cfg.detail_min_inliers,
        max_planes=cfg.max_detail_planes,
        ransac_iterations=cfg.ransac_iterations,
        seed=cfg.seed + 1,
    )
    for p in detail:
        p.inliers = idx[p.inliers]
    new_unassigned = np.zeros(len(work.points), dtype=bool)
    new_unassigned[idx[sub_unassigned]] = True
    return detail, new_unassigned


def _detect_residual_cylinders(work, unassigned, dist_thresh, cfg):
    """Cylinders (columns, pipes, ducts) among the plane-residual points."""
    from scantobim.core.machinery import detect_cylinders

    idx = np.flatnonzero(unassigned)
    if len(idx) < 200:
        return [], unassigned
    pts = work.points[idx]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    cyls, sub_unassigned = detect_cylinders(
        pts,
        work.normals[idx],
        distance_threshold=dist_thresh,
        min_inliers=max(
            150, int(cfg.min_inlier_ratio * min(len(work.points), _RATIO_REF))
        ),
        max_cylinders=cfg.max_cylinders,
        max_radius=0.5 * float(np.max(hi - lo)),
        seed=cfg.seed,
    )
    for c in cyls:
        c.inliers = idx[c.inliers]  # back into work-cloud indexing
    new_unassigned = np.zeros(len(work.points), dtype=bool)
    new_unassigned[idx[sub_unassigned]] = True
    return cyls, new_unassigned


def _reconstruct_watertight(work, planes, spacing, cfg, report, t0):
    """Watertight branch: global cell selection instead of greedy polygons."""
    from scantobim.core.watertight import make_watertight

    mesh, cell_geometries, wt_info = make_watertight(planes, work.points, spacing)
    report["watertight_optimization"] = wt_info

    palette = _make_palette(len(planes))
    colors = np.empty((len(mesh.vertices), 3), dtype=np.uint8)
    colors[:] = (200, 200, 200)
    # Color vertices by the group of the first face using them.
    for face, group in zip(mesh.faces, mesh.face_groups):
        colors[face] = palette[int(group) % len(palette)]
    mesh.vertex_colors = colors if cfg.color_surfaces else None

    geometries = [
        SurfaceGeometry(
            plane_index=pi,
            normal=planes[pi].normal.copy(),
            outer=poly3,
            holes=[],
        )
        for pi, poly3 in cell_geometries
    ]
    selected_planes = {g.plane_index for g in geometries}
    plane_reports = [
        {
            "plane": pi,
            "points": int(len(planes[pi].inliers)),
            "rms": round(planes[pi].rms, 6),
            "sigma_offset": round(
                planes[pi].rms / max(np.sqrt(len(planes[pi].inliers)), 1.0), 7
            ),
            "normal": [round(float(x), 6) for x in planes[pi].normal],
            "status": "ok" if pi in selected_planes else "rejected: not selected",
        }
        for pi in range(len(planes))
    ]
    report["surfaces"] = plane_reports

    residual_mask = np.ones(len(work.points), dtype=bool)
    for p in planes:
        residual_mask[p.inliers] = False
    residual = work.select(residual_mask)

    # ---- world alignment + semantics (same as the greedy path) -------------
    plane_normals = {pi: planes[pi].normal.copy() for pi in range(len(planes))}
    if cfg.align_axes:
        alignment = compute_alignment(planes)
        mesh, res_pts, final_t = apply_alignment(mesh, residual.points, alignment)
        residual.points = res_pts
        plane_normals = {pi: alignment[:3, :3] @ n for pi, n in plane_normals.items()}
        rot, trans = final_t[:3, :3], final_t[:3, 3]
        for geo in geometries:
            geo.outer = geo.outer @ rot.T + trans
            geo.normal = rot @ geo.normal
        report["alignment"] = [[round(float(v), 8) for v in row] for row in final_t]

    surface_ids = sorted({int(g) for g in mesh.face_groups})
    mean_z: dict[int, float] = {}
    surf_area: dict[int, float] = {}
    for gid in surface_ids:
        sel = mesh.face_groups == gid
        tri = mesh.vertices[mesh.faces[sel]]
        areas = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
        )
        surf_area[gid] = float(areas.sum())
        mean_z[gid] = float(tri[:, :, 2].mean())
    classes = classify_surfaces(
        [plane_normals[g] for g in surface_ids], [mean_z[g] for g in surface_ids]
    )
    from scantobim.core.semantics import refine_roof_classes

    classes = refine_roof_classes(classes, [mean_z[g] for g in surface_ids])
    class_of = dict(zip(surface_ids, classes))
    mesh.group_names = {g: f"{class_of[g]}_{g:03d}" for g in surface_ids}
    for geo in geometries:
        if geo.plane_index in class_of:
            geo.surface_class = class_of[geo.plane_index]
            geo.name = mesh.group_names[geo.plane_index]
    report["quantities"] = quantity_takeoff(mesh, class_of, surf_area)
    report["storeys"] = detect_storeys(
        [mean_z[g] for g in surface_ids if class_of[g] in ("floor", "slab", "ceiling")]
    )
    if "volume" in report["quantities"]:
        sigma_by_plane = {
            info["plane"]: info.get("sigma_offset", 0.0) for info in plane_reports
        }
        report["quantities"]["volume_sigma"] = round(
            float(
                np.sqrt(
                    sum(
                        (surf_area.get(g, 0.0) * sigma_by_plane.get(g, 0.0)) ** 2
                        for g in surface_ids
                    )
                )
            ),
            5,
        )

    report["mesh"] = {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.faces),
        "surface_area": round(mesh.area(), 4),
        **mesh.edge_stats(),
    }
    report["residual_points"] = len(residual)
    report["runtime_seconds"] = round(time.perf_counter() - t0, 3)
    return ReconstructionResult(
        mesh=mesh, residual=residual, surfaces=geometries, report=report
    )


def _reconstruct_plane(
    plane,
    plane_index: int,
    points: np.ndarray,
    spacing: float,
    cfg: PipelineConfig,
    lines: list,
    corners: list,
    color: tuple[int, int, int],
) -> tuple[Mesh | None, dict, "SurfaceGeometry | None"]:
    info: dict = {
        "plane": plane_index,
        "points": int(len(plane.inliers)),
        "rms": round(plane.rms, 6),
        # Standard error of the plane position along its normal.
        "sigma_offset": round(plane.rms / max(np.sqrt(len(plane.inliers)), 1.0), 7),
        "normal": [round(float(x), 6) for x in plane.normal],
    }

    uv = plane.project_to_2d(points[plane.inliers])
    boundary, hole_loops = alpha_shape_loops(uv, alpha=cfg.alpha_factor * spacing)
    if boundary is None or len(boundary) < 3:
        info["status"] = "rejected: no boundary"
        return None, info, None

    boundary = simplify_polygon(boundary, tolerance=cfg.simplify_factor * spacing)
    if cfg.straighten:
        boundary = straighten_polygon(
            boundary, angle_tol_deg=cfg.straighten_angle_tol_deg
        )

    # Lift to 3D and snap onto exact intersection lines and corners.
    poly3d = plane.lift_to_3d(boundary)
    snap_dist = cfg.edge_snap_factor * spacing
    for line in lines:
        poly3d, _ = snap_points_to_line(poly3d, line.point, line.direction, snap_dist)
    poly3d = snap_points_to_corners(poly3d, corners, cfg.corner_snap_factor * spacing)

    # Back to 2D (still exactly in-plane) for dedupe + triangulation.
    uv_final = plane.project_to_2d(poly3d)
    uv_final = dedupe_polygon(uv_final, min_dist=max(0.5 * spacing, 1e-9))
    uv_final = remove_collinear(uv_final, tolerance=0.5 * spacing)
    if len(uv_final) < 3 or polygon_area(uv_final) < (2.0 * spacing) ** 2:
        info["status"] = "rejected: degenerate after snapping"
        return None, info, None

    # Openings (windows, door cutouts): regularize each hole loop and keep it
    # if it stays a valid polygon strictly inside the final outer boundary.
    holes: list[np.ndarray] = []
    if cfg.detect_openings:
        min_hole_area = (cfg.min_opening_factor * spacing) ** 2
        for hole in hole_loops:
            hole = simplify_polygon(hole, tolerance=cfg.simplify_factor * spacing)
            if cfg.straighten:
                hole = straighten_polygon(hole, angle_tol_deg=cfg.straighten_angle_tol_deg)
            hole = dedupe_polygon(hole, min_dist=max(0.5 * spacing, 1e-9))
            hole = remove_collinear(hole, tolerance=0.5 * spacing)
            if len(hole) < 3 or polygon_area(hole) < min_hole_area:
                continue
            # A real opening is clearly smaller than its surface.
            if polygon_area(hole) > 0.8 * polygon_area(uv_final):
                continue
            if not all(point_in_polygon(v, uv_final) for v in hole):
                continue
            holes.append(hole)

    verts2d, tris = triangulate_with_holes(uv_final, holes)
    if len(tris) == 0:
        info["status"] = "rejected: triangulation failed"
        return None, info, None

    vertices = plane.lift_to_3d(verts2d)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (len(vertices), 1))
    mesh = Mesh(
        vertices=vertices,
        faces=tris,
        vertex_colors=colors,
        face_groups=np.full(len(tris), plane_index, dtype=np.int64),
    )
    info["status"] = "ok"
    info["boundary_vertices"] = int(len(uv_final))
    info["area"] = round(
        polygon_area(uv_final) - sum(polygon_area(h) for h in holes), 4
    )
    # Area uncertainty from boundary sampling: every outline edge is known
    # to about half the local point spacing (snapped edges are better; this
    # is the conservative bound).
    perimeter = float(
        np.linalg.norm(np.roll(uv_final, -1, axis=0) - uv_final, axis=1).sum()
    )
    info["area_sigma"] = round(perimeter * 0.5 * spacing, 4)
    info["dimension_sigma"] = round(float(np.sqrt(2.0)) * 0.5 * spacing, 5)
    info["openings"] = len(holes)
    if holes:
        info["opening_areas"] = [round(polygon_area(h), 4) for h in holes]
    geometry = SurfaceGeometry(
        plane_index=plane_index,
        normal=plane.normal.copy(),
        outer=plane.lift_to_3d(uv_final),
        holes=[plane.lift_to_3d(h) for h in holes],
    )
    return mesh, info, geometry


def _orient_planes(planes, points: np.ndarray, mode: str) -> None:
    """Flip plane normals outward/inward relative to the cloud centroid."""
    if mode not in ("outward", "inward"):
        return
    centroid = points.mean(axis=0)
    for plane in planes:
        support_center = points[plane.inliers].mean(axis=0)
        outward = float(plane.normal @ (support_center - centroid))
        flip = outward < 0 if mode == "outward" else outward > 0
        if flip:
            plane.normal = -plane.normal
            plane.d = -plane.d
        plane.make_basis()


def _make_palette(n: int) -> list[tuple[int, int, int]]:
    """Visually distinct, print-friendly surface colors (golden-angle hues)."""
    colors = []
    for i in range(max(n, 1)):
        hue = (i * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.45, 0.85)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def _config_dict(cfg: PipelineConfig) -> dict:
    d = asdict(cfg)
    return {k: v for k, v in d.items() if v is not None}
