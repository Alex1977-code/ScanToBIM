"""2D floor plan export as DXF (AutoCAD R12 ASCII).

The reconstructed model is sliced with a horizontal plane at a configurable
height above the lowest point; the resulting wall cut lines form the floor
plan (Grundriss) — directly usable in AutoCAD, LibreCAD, QCAD, BricsCAD or
any architecture package. Written entirely without dependencies: DXF R12 is
a plain-text tag format.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scantobim.core.mesh import Mesh


def slice_mesh(mesh: Mesh, z: float) -> np.ndarray:
    """Intersect all triangles with the plane ``Z = z``.

    Returns ``(N, 2, 2)`` line segments in XY.
    """
    tri = mesh.vertices[mesh.faces]  # (F, 3, 3)
    d = tri[:, :, 2] - z  # signed distance per corner
    segments = []
    for corners, dist in zip(tri, d):
        pts = []
        for i in range(3):
            j = (i + 1) % 3
            di, dj = dist[i], dist[j]
            if di == 0.0:
                pts.append(corners[i, :2])
            if (di > 0) != (dj > 0) and di != 0.0 and dj != 0.0:
                t = di / (di - dj)
                p = corners[i] + t * (corners[j] - corners[i])
                pts.append(p[:2])
        if len(pts) >= 2:
            # Deduplicate (vertex-on-plane cases can duplicate endpoints).
            uniq: list[np.ndarray] = []
            for p in pts:
                if not any(np.linalg.norm(p - q) < 1e-12 for q in uniq):
                    uniq.append(p)
            if len(uniq) >= 2:
                segments.append([uniq[0], uniq[1]])
    return np.array(segments) if segments else np.zeros((0, 2, 2))


def write_floorplan_dxf(
    mesh: Mesh, path: str | Path, height_above_floor: float = 1.0
) -> Path:
    """Write a DXF floor plan sliced ``height_above_floor`` above the model base.

    Wall cuts land on layer ``SCHNITT``; the model's footprint outline (the
    slice through everything) is what architects expect at 1.00 m.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(mesh.vertices) == 0:
        raise ValueError("empty mesh — nothing to slice")
    z0 = float(mesh.vertices[:, 2].min())
    z = z0 + height_above_floor
    segments = slice_mesh(mesh, z)
    if len(segments) == 0:
        raise ValueError(
            f"slice at Z={z:.3f} intersects no geometry — "
            "adjust height_above_floor"
        )

    lines = ["0", "SECTION", "2", "ENTITIES"]
    for (x1, y1), (x2, y2) in segments:
        lines += [
            "0", "LINE",
            "8", "SCHNITT",
            "10", f"{x1:.6f}",
            "20", f"{y1:.6f}",
            "30", "0.0",
            "11", f"{x2:.6f}",
            "21", f"{y2:.6f}",
            "31", "0.0",
        ]
    lines += ["0", "ENDSEC", "0", "EOF"]
    path.write_text("\n".join(lines) + "\n")
    return path
