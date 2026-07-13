"""Surface classification and quantity takeoff (Mengenermittlung).

Surfaces are classified from their regularized normal and height into BIM
categories (floor / ceiling / slab / wall / sloped). From the classified,
welded mesh the module derives the quantities architects actually bill by:
areas per category, net room/building volume (for watertight models) and
footprint extents.
"""

from __future__ import annotations

import numpy as np

from scantobim.core.mesh import Mesh

UP = np.array([0.0, 0.0, 1.0])

HORIZONTAL_TILT_DEG = 30.0  # tilt from horizontal still counted as slab
VERTICAL_TILT_DEG = 30.0  # tilt from vertical still counted as wall


def classify_surface(
    normal: np.ndarray, mean_z: float, z_low: float, z_high: float
) -> str:
    """Classify one surface: ``floor``/``ceiling``/``slab``/``wall``/``sloped``.

    Horizontal surfaces near the bottom of the model are floors, near the top
    ceilings, in between slabs (e.g. stair landings, tabletops).
    """
    c = abs(float(normal @ UP))
    if c >= np.cos(np.deg2rad(HORIZONTAL_TILT_DEG)):
        span = max(z_high - z_low, 1e-9)
        rel = (mean_z - z_low) / span
        if rel <= 0.25:
            return "floor"
        if rel >= 0.75:
            return "ceiling"
        return "slab"
    if c <= np.cos(np.deg2rad(90.0 - VERTICAL_TILT_DEG)):
        return "wall"
    return "sloped"


def classify_surfaces(
    normals: list[np.ndarray], mean_zs: list[float]
) -> list[str]:
    """Classify all surfaces of a model together (shared height range)."""
    if not normals:
        return []
    z_low, z_high = min(mean_zs), max(mean_zs)
    return [
        classify_surface(n, z, z_low, z_high) for n, z in zip(normals, mean_zs)
    ]


def detect_storeys(
    horizontal_zs: list[float],
    merge_tol: float = 0.5,
    min_storey_height: float = 1.8,
) -> list[dict]:
    """Storeys from the elevations of horizontal surfaces.

    Elevations within ``merge_tol`` collapse into one slab level; every pair
    of consecutive levels at least ``min_storey_height`` apart is a storey.
    Returns ``[{"index", "elevation", "height"}, ...]`` bottom-up.
    """
    if not horizontal_zs:
        return []
    zs = sorted(horizontal_zs)
    levels: list[list[float]] = [[zs[0]]]
    for z in zs[1:]:
        if z - levels[-1][-1] <= merge_tol:
            levels[-1].append(z)
        else:
            levels.append([z])
    elevations = [float(np.mean(lv)) for lv in levels]
    storeys = []
    for lo, hi in zip(elevations, elevations[1:]):
        if hi - lo >= min_storey_height:
            storeys.append(
                {
                    "index": len(storeys),
                    "elevation": round(lo, 4),
                    "height": round(hi - lo, 4),
                }
            )
    return storeys


def signed_volume(mesh: Mesh) -> float:
    """Signed volume via the divergence theorem (valid for closed meshes).

    Positive for outward-oriented normals, negative for inward (indoor
    preset). Meaningless if the mesh has boundary edges.
    """
    tri = mesh.vertices[mesh.faces]
    return float(
        np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    )


def quantity_takeoff(
    mesh: Mesh, surface_classes: dict[int, str], surface_areas: dict[int, float]
) -> dict:
    """Aggregate BIM quantities from classified surfaces."""
    area_by_class: dict[str, float] = {}
    count_by_class: dict[str, int] = {}
    for gid, cls in surface_classes.items():
        area_by_class[cls] = area_by_class.get(cls, 0.0) + surface_areas.get(gid, 0.0)
        count_by_class[cls] = count_by_class.get(cls, 0) + 1

    lo = mesh.vertices.min(axis=0) if len(mesh.vertices) else np.zeros(3)
    hi = mesh.vertices.max(axis=0) if len(mesh.vertices) else np.zeros(3)
    stats = mesh.edge_stats()
    out = {
        "surface_count_by_class": count_by_class,
        "area_by_class": {k: round(v, 4) for k, v in area_by_class.items()},
        "footprint": {
            "size_x": round(float(hi[0] - lo[0]), 4),
            "size_y": round(float(hi[1] - lo[1]), 4),
            "height": round(float(hi[2] - lo[2]), 4),
        },
        "watertight": stats["boundary_edges"] == 0 and stats["non_manifold_edges"] == 0,
    }
    if out["watertight"]:
        vol = signed_volume(mesh)
        out["volume"] = round(abs(vol), 4)
        out["orientation"] = "outward" if vol > 0 else "inward"
    return out


def refine_roof_classes(classes: list[str], mean_zs: list[float]) -> list[str]:
    """Sloped surfaces above the walls are roof faces.

    The reference is the height of the detected walls, NOT the scene
    extent — outdoor scans contain terrain, masts and vegetation that
    stretch the z-range and would otherwise push real roofs below any
    scene-relative threshold. Sloped terrain stays "sloped".
    """
    if not classes:
        return classes
    wall_zs = [z for c, z in zip(classes, mean_zs) if c == "wall"]
    if wall_zs:
        threshold = max(wall_zs)  # roofs sit above every wall's midpoint
    else:
        z_low, z_high = min(mean_zs), max(mean_zs)
        threshold = z_low + 0.5 * max(z_high - z_low, 1e-9)
    return [
        "roof" if c == "sloped" and z > threshold else c
        for c, z in zip(classes, mean_zs)
    ]


def refine_terrain_classes(
    classes: list[str], rms_values: list[float], rms_threshold: float = 0.025
) -> list[str]:
    """Rough horizontal surfaces are terrain, not building slabs.

    Ground, gravel and lawns fit a plane only to a few centimeters —
    building floors and slabs to millimeters. Everything horizontal whose
    plane RMS exceeds ``rms_threshold`` is reclassified ``terrain`` so it
    stays out of storeys, opening schedules and slab quantities.
    """
    return [
        "terrain"
        if c in ("floor", "slab") and rms is not None and rms > rms_threshold
        else c
        for c, rms in zip(classes, rms_values)
    ]


def roof_report(geometries, areas: dict[int, float]) -> dict:
    """Roof metrics: per-face slope/azimuth/area, ridge and eaves heights.

    ``geometries`` are the SurfaceGeometry objects classified ``roof``;
    ``areas`` maps plane index → measured area.
    """
    faces = []
    ridge = -np.inf
    eaves = np.inf
    total = 0.0
    for geo in geometries:
        n = geo.normal / max(np.linalg.norm(geo.normal), 1e-12)
        slope = float(np.degrees(np.arccos(min(abs(float(n[2])), 1.0))))
        horiz = np.array([n[0], n[1]])
        azimuth = None
        if np.linalg.norm(horiz) > 1e-9:
            # Compass azimuth of the direction the face looks towards
            # (N = 0°, E = 90°).
            azimuth = float(np.degrees(np.arctan2(horiz[0], horiz[1]))) % 360.0
        z_top = float(geo.outer[:, 2].max())
        z_bottom = float(geo.outer[:, 2].min())
        area = float(areas.get(geo.plane_index, 0.0))
        ridge = max(ridge, z_top)
        eaves = min(eaves, z_bottom)
        total += area
        faces.append(
            {
                "name": geo.name or f"roof_{geo.plane_index:03d}",
                "slope_deg": round(slope, 2),
                "azimuth_deg": round(azimuth, 1) % 360.0 if azimuth is not None else None,
                "area": round(area, 3),
                "top": round(z_top, 3),
                "bottom": round(z_bottom, 3),
            }
        )
    return {
        "faces": faces,
        "ridge_height": round(float(ridge), 3),
        "eaves_height": round(float(eaves), 3),
        "total_area": round(total, 3),
    }
