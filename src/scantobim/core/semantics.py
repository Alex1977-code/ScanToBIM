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
