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
from scantobim.core.mesh import Mesh, merge_meshes, triangulate_polygon, weld_vertices
from scantobim.core.planes import detect_planes, plane_adjacency
from scantobim.core.polygons import (
    alpha_shape_boundary,
    dedupe_polygon,
    polygon_area,
    remove_collinear,
    simplify_polygon,
    straighten_polygon,
)
from scantobim.core.preprocess import (
    estimate_normals,
    estimate_point_spacing,
    remove_statistical_outliers,
    voxel_downsample,
)
from scantobim.core.regularize import regularize_planes, snapped_angles_report


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

    # --- plane detection ---
    distance_threshold: float | None = None  # None = auto (3x spacing)
    normal_threshold_deg: float = 30.0
    min_inlier_ratio: float = 0.01  # fraction of cloud size
    min_inliers_abs: int = 60
    max_planes: int = 64
    ransac_iterations: int = 600
    seed: int = 7

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

    # --- mesh ---
    orient: str = "outward"  # "outward" | "inward" | "none"
    weld_factor: float = 0.25  # x spacing
    color_surfaces: bool = True

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
        raise ValueError(f"Unknown preset {name!r} (use building/indoor/object/fast)")


@dataclass
class ReconstructionResult:
    mesh: Mesh
    residual: PointCloud  # points not explained by any planar surface
    report: dict = field(default_factory=dict)


def reconstruct(cloud: PointCloud, config: PipelineConfig | None = None) -> ReconstructionResult:
    """Run the full scan-to-model pipeline on ``cloud``."""
    cfg = config or PipelineConfig()
    report: dict = {"input_points": len(cloud), "config": _config_dict(cfg)}
    t0 = time.perf_counter()

    # ---- 1. preprocessing --------------------------------------------------
    raw_spacing = estimate_point_spacing(cloud)
    voxel = cfg.voxel_size if cfg.voxel_size is not None else 2.0 * raw_spacing
    work = voxel_downsample(cloud, voxel) if voxel and voxel > 0 else cloud
    work, _ = remove_statistical_outliers(work, cfg.sor_neighbors, cfg.sor_std_ratio)
    if len(work) < 16:
        raise ValueError(
            f"Too few points after preprocessing ({len(work)}); "
            "check input units or lower voxel_size"
        )
    work = estimate_normals(work, cfg.normal_neighbors)
    spacing = estimate_point_spacing(work)
    if spacing <= 0:
        raise ValueError("Degenerate point cloud (zero spacing)")
    report["preprocessed_points"] = len(work)
    report["point_spacing"] = round(spacing, 6)

    # ---- 2. plane detection ------------------------------------------------
    dist_thresh = (
        cfg.distance_threshold if cfg.distance_threshold is not None else 3.0 * spacing
    )
    min_inliers = max(cfg.min_inliers_abs, int(cfg.min_inlier_ratio * len(work)))
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
        )
    _orient_planes(planes, work.points, cfg.orient)
    report["planes"] = len(planes)
    report["plane_angles"] = snapped_angles_report(planes)

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

    palette = _make_palette(len(planes))
    parts: list[Mesh] = []
    plane_reports = []
    for pi, plane in enumerate(planes):
        part, info = _reconstruct_plane(
            plane,
            pi,
            work.points,
            spacing,
            cfg,
            lines_by_plane.get(pi, []),
            corners_by_plane.get(pi, []),
            palette[pi],
        )
        plane_reports.append(info)
        if part is not None:
            parts.append(part)
    report["surfaces"] = plane_reports

    if not parts:
        raise ValueError("All detected planes were rejected during polygonization")

    # ---- 6. assemble --------------------------------------------------------
    mesh = merge_meshes(parts)
    mesh = weld_vertices(mesh, cfg.weld_factor * spacing)
    if not cfg.color_surfaces:
        mesh.vertex_colors = None

    residual = work.select(unassigned)
    report["mesh"] = {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.faces),
        "surface_area": round(mesh.area(), 4),
        **mesh.edge_stats(),
    }
    report["residual_points"] = len(residual)
    report["runtime_seconds"] = round(time.perf_counter() - t0, 3)
    return ReconstructionResult(mesh=mesh, residual=residual, report=report)


def _reconstruct_plane(
    plane,
    plane_index: int,
    points: np.ndarray,
    spacing: float,
    cfg: PipelineConfig,
    lines: list,
    corners: list,
    color: tuple[int, int, int],
) -> tuple[Mesh | None, dict]:
    info: dict = {
        "plane": plane_index,
        "points": int(len(plane.inliers)),
        "rms": round(plane.rms, 6),
        "normal": [round(float(x), 6) for x in plane.normal],
    }

    uv = plane.project_to_2d(points[plane.inliers])
    boundary = alpha_shape_boundary(uv, alpha=cfg.alpha_factor * spacing)
    if boundary is None or len(boundary) < 3:
        info["status"] = "rejected: no boundary"
        return None, info

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
        return None, info

    tris = triangulate_polygon(uv_final)
    if len(tris) == 0:
        info["status"] = "rejected: triangulation failed"
        return None, info

    vertices = plane.lift_to_3d(uv_final)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (len(vertices), 1))
    mesh = Mesh(
        vertices=vertices,
        faces=tris,
        vertex_colors=colors,
        face_groups=np.full(len(tris), plane_index, dtype=np.int64),
    )
    info["status"] = "ok"
    info["boundary_vertices"] = int(len(uv_final))
    info["area"] = round(polygon_area(uv_final), 4)
    return mesh, info


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
