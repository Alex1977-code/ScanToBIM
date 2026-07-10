"""Welded sheet metal constructions (geschweißte Blechkonstruktionen).

From a scan of a welded assembly (tank, hopper housing, machine frame,
chute) the module recovers what a fabricator needs to re-build or cost it:

* **Plates** — pairs of parallel planes a few millimetres apart are the two
  faces of one sheet: mid-plane, measured thickness (matched against the
  standard plate series), straightened outline, dimensions, area and weight
  (steel, 7.85 kg/dm³).
* **Weld seams** — where two plates meet, their mid-planes intersect in a
  line; the covered span of that line is the seam, reported with length and
  dihedral angle. The summed length prices the welding work.
* **Cutting outlines** — every plate outline exported 1:1 into a DXF for
  laser/plasma cutting (:func:`scantobim.io.dxf.write_cutting_dxf`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scantobim.core.cloud import PointCloud
from scantobim.core.planes import Plane, detect_planes
from scantobim.core.polygons import (
    alpha_shape_boundary,
    polygon_area,
    remove_collinear,
    simplify_polygon,
    straighten_polygon,
)

# EN 10029 / common stock plate thicknesses, meters
PLATE_THICKNESSES = [
    0.002, 0.003, 0.004, 0.005, 0.006, 0.008,
    0.010, 0.012, 0.015, 0.020, 0.025, 0.030,
]
STEEL_DENSITY = 7850.0  # kg/m^3


@dataclass
class Plate:
    normal: np.ndarray  # mid-plane unit normal
    d: float  # mid-plane offset (n·x + d = 0)
    thickness: float
    thickness_sigma: float
    catalog_thickness: float | None
    outline_2d: np.ndarray  # (N, 2) in the plate frame, straightened
    origin: np.ndarray  # frame origin (on the mid-plane)
    u: np.ndarray
    v: np.ndarray
    area: float
    size: tuple[float, float]  # bbox of the outline (length, width)
    plane_indices: tuple[int, int]
    inliers: np.ndarray = field(default=None, repr=False)

    @property
    def weight(self) -> float:
        t = self.catalog_thickness or self.thickness
        return self.area * t * STEEL_DENSITY

    def outline_3d(self) -> np.ndarray:
        return self.origin + self.outline_2d[:, :1] * self.u + self.outline_2d[:, 1:] * self.v


@dataclass
class WeldSeam:
    plate_a: int
    plate_b: int
    length: float
    angle_deg: float  # dihedral angle between the plates
    start: np.ndarray
    end: np.ndarray


def detect_plates(
    cloud: PointCloud,
    distance_threshold: float | None = None,
    max_thickness: float = 0.05,
    min_inlier_ratio: float = 0.01,
    seed: int = 7,
) -> tuple[list[Plate], list[Plane], dict]:
    """Detect sheet metal plates as parallel plane pairs.

    Returns ``(plates, planes, info)`` where ``info`` carries the
    preprocessing numbers for the report.
    """
    from scantobim.core.preprocess import (
        estimate_normals,
        estimate_point_spacing,
        remove_statistical_outliers,
        voxel_downsample,
    )

    raw_spacing = estimate_point_spacing(cloud)
    work = voxel_downsample(cloud, 1.2 * raw_spacing)
    work, _ = remove_statistical_outliers(work)
    work = estimate_normals(work, k_neighbors=12)
    spacing = estimate_point_spacing(work)
    if distance_threshold is None:
        # Thickness measurement demands tight tolerances.
        distance_threshold = 1.2 * spacing
    if distance_threshold > max_thickness / 3:
        raise ValueError(
            "point spacing too coarse to resolve plate thicknesses — "
            "scan denser or raise max_thickness"
        )

    min_inliers = max(150, int(min_inlier_ratio * len(work)))
    planes, _ = detect_planes(
        work.points,
        work.normals,
        distance_threshold=distance_threshold,
        min_inliers=min_inliers,
        seed=seed,
        normal_threshold_deg=25.0,
    )

    plates: list[Plate] = []
    used = set()
    order = sorted(range(len(planes)), key=lambda i: -len(planes[i].inliers))
    for i in order:
        if i in used:
            continue
        best_j = -1
        best_thickness = np.inf
        for j in order:
            if j == i or j in used:
                continue
            a, b = planes[i], planes[j]
            if abs(float(a.normal @ b.normal)) < np.cos(np.deg2rad(3.0)):
                continue  # not parallel
            # Signed offset of b's support from a's plane.
            dists = a.distance(work.points[b.inliers])
            thickness = float(np.abs(np.median(dists)))
            if thickness < 2.5 * distance_threshold or thickness > max_thickness:
                continue
            if np.std(dists) > 3.0 * distance_threshold:
                continue  # not a uniform gap
            if not _footprints_overlap(a, b, work.points):
                continue
            if thickness < best_thickness:
                best_thickness = thickness
                best_j = j
        if best_j < 0:
            continue
        plate = _build_plate(planes[i], planes[best_j], work.points, spacing, (i, best_j))
        if plate is not None:
            plates.append(plate)
            used.add(i)
            used.add(best_j)

    info = {
        "points": len(cloud),
        "analyzed_points": len(work),
        "point_spacing": round(spacing, 6),
        "distance_threshold": round(float(distance_threshold), 6),
        "planes": len(planes),
        "unpaired_planes": len(planes) - 2 * len(plates),
    }
    return plates, planes, info


def _footprints_overlap(a: Plane, b: Plane, points: np.ndarray, min_ratio: float = 0.3) -> bool:
    """Do the two planes' supports overlap when projected onto plane a?"""
    ua = a.project_to_2d(points[a.inliers])
    ub = a.project_to_2d(points[b.inliers])
    lo = np.maximum(ua.min(axis=0), ub.min(axis=0))
    hi = np.minimum(ua.max(axis=0), ub.max(axis=0))
    if np.any(hi <= lo):
        return False
    inter = np.prod(hi - lo)
    area_a = np.prod(ua.max(axis=0) - ua.min(axis=0))
    area_b = np.prod(ub.max(axis=0) - ub.min(axis=0))
    return inter >= min_ratio * min(area_a, area_b)


def _build_plate(
    a: Plane, b: Plane, points: np.ndarray, spacing: float, indices: tuple[int, int]
) -> Plate | None:
    normal = a.normal
    dists_b = a.distance(points[b.inliers])
    thickness = float(np.abs(np.median(dists_b)))
    thickness_sigma = float(np.std(np.abs(dists_b)) / max(np.sqrt(len(dists_b)), 1.0))
    # Mid-plane between the two faces.
    d_mid = a.d - float(np.median(dists_b)) / 2.0

    inliers = np.concatenate([a.inliers, b.inliers])
    origin = -d_mid * normal
    helper = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1, 0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    rel = points[inliers] - origin
    uv = np.column_stack([rel @ u, rel @ v])
    outline = alpha_shape_boundary(uv, alpha=4.0 * spacing)
    if outline is None or len(outline) < 3:
        return None
    outline = simplify_polygon(outline, tolerance=2.0 * spacing)
    outline = straighten_polygon(outline, angle_tol_deg=14.0)
    outline = remove_collinear(outline, tolerance=0.5 * spacing)
    if len(outline) < 3:
        return None
    area = polygon_area(outline)
    if area < (4 * spacing) ** 2:
        return None
    size = (
        float(outline[:, 0].max() - outline[:, 0].min()),
        float(outline[:, 1].max() - outline[:, 1].min()),
    )
    return Plate(
        normal=normal,
        d=d_mid,
        thickness=thickness,
        thickness_sigma=thickness_sigma,
        catalog_thickness=match_thickness(thickness),
        outline_2d=outline,
        origin=origin,
        u=u,
        v=v,
        area=float(area),
        size=(max(size), min(size)),
        plane_indices=indices,
        inliers=inliers,
    )


def match_thickness(measured: float, tolerance: float = 0.15) -> float | None:
    """Nearest standard plate thickness within ``tolerance`` (relative)."""
    best = min(PLATE_THICKNESSES, key=lambda t: abs(t - measured))
    if abs(best - measured) / best <= tolerance:
        return best
    return None


def detect_weld_seams(
    plates: list[Plate],
    points: np.ndarray,
    contact_radius: float,
    min_length: float = 0.02,
) -> list[WeldSeam]:
    """Weld seams: covered spans of the mid-plane intersection lines."""
    from scipy.spatial import cKDTree

    seams: list[WeldSeam] = []
    trees = [cKDTree(points[p.inliers]) for p in plates]
    for i in range(len(plates)):
        for j in range(i + 1, len(plates)):
            a, b = plates[i], plates[j]
            cos_angle = float(a.normal @ b.normal)
            if abs(cos_angle) > 0.985:
                continue  # near parallel — no seam line
            direction = np.cross(a.normal, b.normal)
            direction = direction / np.linalg.norm(direction)
            m = np.vstack([a.normal, b.normal, direction])
            line_point = np.linalg.solve(m, np.array([-a.d, -b.d, 0.0]))

            # Points of both plates close to the intersection line.
            covered = []
            for plate, tree in ((a, trees[i]), (b, trees[j])):
                pts = points[plate.inliers]
                rel = pts - line_point
                t = rel @ direction
                dist = np.linalg.norm(rel - np.outer(t, direction), axis=1)
                near = dist < contact_radius
                covered.append(t[near])
            if len(covered[0]) < 8 or len(covered[1]) < 8:
                continue
            t_all = np.concatenate(covered)
            # Coverage-based length: histogram over the parameter range.
            t_lo, t_hi = float(t_all.min()), float(t_all.max())
            if t_hi - t_lo < min_length:
                continue
            n_bins = max(4, int((t_hi - t_lo) / max(contact_radius, 1e-9)))
            hist, _ = np.histogram(t_all, bins=n_bins, range=(t_lo, t_hi))
            length = float((hist > 0).sum() / n_bins * (t_hi - t_lo))
            if length < min_length:
                continue
            angle = float(np.rad2deg(np.arccos(np.clip(abs(cos_angle), -1, 1))))
            seams.append(
                WeldSeam(
                    plate_a=i,
                    plate_b=j,
                    length=length,
                    angle_deg=180.0 - angle if angle > 90 else angle,
                    start=line_point + t_lo * direction,
                    end=line_point + t_hi * direction,
                )
            )
    return seams


def analyze_sheet_metal(
    cloud: PointCloud,
    distance_threshold: float | None = None,
    max_thickness: float = 0.05,
    seed: int = 7,
) -> dict:
    """Full sheet metal analysis: plates + weld seams + fabrication totals."""
    from scantobim.core.preprocess import estimate_point_spacing

    plates, planes, info = detect_plates(
        cloud,
        distance_threshold=distance_threshold,
        max_thickness=max_thickness,
        seed=seed,
    )
    if not plates:
        raise ValueError(
            "no sheet metal plates found — both faces of each plate must be "
            "scanned (inside and outside) to measure thickness"
        )
    # Re-derive spacing for the contact radius (cheap).
    spacing = info["point_spacing"]
    from scantobim.core.preprocess import voxel_downsample, remove_statistical_outliers

    raw_spacing = estimate_point_spacing(cloud)
    work = voxel_downsample(cloud, 1.2 * raw_spacing)
    work, _ = remove_statistical_outliers(work)
    seams = detect_weld_seams(
        plates, work.points, contact_radius=max(6 * spacing, 2 * max(p.thickness for p in plates))
    )

    report = {
        **info,
        "plates": [
            {
                "id": k,
                "thickness_measured": round(p.thickness, 5),
                "thickness_sigma": round(p.thickness_sigma, 8),
                "thickness_catalog": p.catalog_thickness,
                "size": [round(p.size[0], 4), round(p.size[1], 4)],
                "area": round(p.area, 4),
                "weight_kg": round(p.weight, 2),
                "outline_vertices": int(len(p.outline_2d)),
                "normal": [round(float(x), 4) for x in p.normal],
            }
            for k, p in enumerate(plates)
        ],
        "weld_seams": [
            {
                "plates": [s.plate_a, s.plate_b],
                "length": round(s.length, 4),
                "angle_deg": round(s.angle_deg, 1),
            }
            for s in seams
        ],
        "totals": {
            "plate_count": len(plates),
            "plate_area": round(sum(p.area for p in plates), 4),
            "weight_kg": round(sum(p.weight for p in plates), 2),
            "weld_length": round(sum(s.length for s in seams), 4),
            "seam_count": len(seams),
        },
        "_objects": {"plates": plates, "seams": seams},
    }
    return report
