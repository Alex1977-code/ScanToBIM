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
    mesh: Mesh, path: str | Path, height_above_floor: float = 1.0,
    annotate: bool = True,
) -> Path:
    """Write a DXF floor plan sliced ``height_above_floor`` above the model base.

    Wall cuts land on layer ``SCHNITT``; with ``annotate`` every merged wall
    run carries its length in meters as a dimension text on layer
    ``BEMASSUNG`` — a checkable, print-ready plan.
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

    if annotate:
        seg = np.asarray(segments, dtype=np.float64)
        extent = float(max(np.ptp(seg[:, :, 0]), np.ptp(seg[:, :, 1]), 1e-9))
        centroid = seg.reshape(-1, 2).mean(axis=0)
        text_h = max(0.02 * extent, 0.08)
        for start, end in _merge_wall_runs(seg):
            d = end - start
            length = float(np.linalg.norm(d))
            if length < 3 * text_h:
                continue  # too short to label legibly
            d /= length
            normal = np.array([-d[1], d[0]])
            angle = float(np.degrees(np.arctan2(d[1], d[0])))
            if angle > 90.0 or angle <= -90.0:
                angle = (angle + 180.0) % 360.0  # keep text upright
            center = (start + end) / 2.0
            if normal @ (centroid - center) < 0:
                normal = -normal  # dimension text towards the plan interior
            mid = center + normal * 1.2 * text_h
            lines += [
                "0", "TEXT",
                "8", "BEMASSUNG",
                "10", f"{mid[0]:.6f}",
                "20", f"{mid[1]:.6f}",
                "30", "0.0",
                "40", f"{text_h:.4f}",
                "1", f"{length:.3f}",
                "50", f"{angle:.2f}",
                "72", "1",
                "11", f"{mid[0]:.6f}",
                "21", f"{mid[1]:.6f}",
                "31", "0.0",
            ]

    lines += ["0", "ENDSEC", "0", "EOF"]
    path.write_text("\n".join(lines) + "\n")
    return path


def _merge_wall_runs(
    segments: np.ndarray,
    offset_tol: float = 0.03,
    gap_tol: float = 0.08,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Merge collinear slice segments into wall runs for dimensioning.

    A wall crossed at 1 m yields hundreds of short per-triangle cuts; the
    dimension belongs to the merged run. Segments are grouped by direction
    and perpendicular offset, projected onto their common line and merged
    where gaps stay below ``gap_tol``.
    """
    n = len(segments)
    used = np.zeros(n, dtype=bool)
    starts, ends = segments[:, 0], segments[:, 1]
    vecs = ends - starts
    lens = np.linalg.norm(vecs, axis=1)
    ok = lens > 1e-9
    dirs = np.zeros_like(vecs)
    dirs[ok] = vecs[ok] / lens[ok, None]
    flip = (dirs[:, 0] < 0) | ((dirs[:, 0] == 0) & (dirs[:, 1] < 0))
    dirs[flip] *= -1  # canonical sign: opposite directions compare equal

    runs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n):
        if used[i] or not ok[i]:
            continue
        d = dirs[i]
        normal = np.array([-d[1], d[0]])
        offsets = starts @ normal
        group = (
            ok & ~used
            & (np.abs(dirs @ normal) < 0.03)
            & (np.abs(offsets - offsets[i]) < offset_tol)
        )
        idx = np.flatnonzero(group)
        used[idx] = True
        line_offset = float(offsets[idx].mean())
        # Merge the 1D intervals along the common direction.
        t = np.sort(np.stack([starts[idx] @ d, ends[idx] @ d], axis=1), axis=1)
        t = t[np.argsort(t[:, 0])]
        cur_lo, cur_hi = t[0]
        for lo, hi in t[1:]:
            if lo <= cur_hi + gap_tol:
                cur_hi = max(cur_hi, hi)
            else:
                runs.append(
                    (line_offset * normal + cur_lo * d, line_offset * normal + cur_hi * d)
                )
                cur_lo, cur_hi = lo, hi
        runs.append(
            (line_offset * normal + cur_lo * d, line_offset * normal + cur_hi * d)
        )
    return runs


def write_cutting_dxf(plates, path: str | Path, gap: float = 0.05) -> Path:
    """Cutting layout for laser/plasma: every plate outline 1:1, side by side.

    Each plate becomes a closed POLYLINE on layer ``ZUSCHNITT`` plus a TEXT
    label (``Blech <id> t=<mm>``) on layer ``BESCHRIFTUNG``. Outlines are
    axis-aligned via their bounding rectangle and laid out in a row with
    ``gap`` spacing — ready for nesting software or direct cutting.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not plates:
        raise ValueError("no plates to export")

    lines = ["0", "SECTION", "2", "ENTITIES"]
    cursor_x = 0.0
    for k, plate in enumerate(plates):
        outline = _axis_align_2d(plate.outline_2d)
        outline = outline - outline.min(axis=0) + np.array([cursor_x, 0.0])
        lines += ["0", "POLYLINE", "8", "ZUSCHNITT", "66", "1", "70", "1"]
        for x, y in outline:
            lines += [
                "0", "VERTEX", "8", "ZUSCHNITT",
                "10", f"{x:.6f}", "20", f"{y:.6f}", "30", "0.0",
            ]
        lines += ["0", "SEQEND"]

        t = plate.catalog_thickness or plate.thickness
        label = f"Blech {k} t={t * 1000:.0f}mm"
        height = max(0.02, 0.06 * float(outline[:, 1].max() - outline[:, 1].min()))
        lines += [
            "0", "TEXT", "8", "BESCHRIFTUNG",
            "10", f"{cursor_x:.6f}",
            "20", f"{float(outline[:, 1].max()) + height:.6f}",
            "30", "0.0",
            "40", f"{height:.4f}",
            "1", label,
        ]
        cursor_x = float(outline[:, 0].max()) + gap
    lines += ["0", "ENDSEC", "0", "EOF"]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_flat_pattern_dxf(parts, path: str | Path, gap: float = 0.05) -> Path:
    """Cutting layout of unfolded parts: outlines on ``ZUSCHNITT``, bend
    lines on ``BIEGELINIE``, one label per part."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not parts:
        raise ValueError("no parts to export")

    lines = ["0", "SECTION", "2", "ENTITIES"]
    cursor_x = 0.0
    for k, part in enumerate(parts):
        pts = np.vstack(part.loops)
        offset = np.array([cursor_x, 0.0]) - pts.min(axis=0)
        for loop in part.loops:
            poly = loop + offset
            lines += ["0", "POLYLINE", "8", "ZUSCHNITT", "66", "1", "70", "1"]
            for x, y in poly:
                lines += [
                    "0", "VERTEX", "8", "ZUSCHNITT",
                    "10", f"{x:.6f}", "20", f"{y:.6f}", "30", "0.0",
                ]
            lines += ["0", "SEQEND"]
        for a, b in part.bend_lines:
            a2, b2 = a + offset, b + offset
            lines += [
                "0", "LINE", "8", "BIEGELINIE",
                "10", f"{a2[0]:.6f}", "20", f"{a2[1]:.6f}", "30", "0.0",
                "11", f"{b2[0]:.6f}", "21", f"{b2[1]:.6f}", "31", "0.0",
            ]
        t = part.catalog_thickness or part.thickness
        n_bends = len(part.bend_lines)
        label = f"Teil {k} t={t * 1000:.0f}mm"
        if n_bends:
            label += f" ({n_bends} Kantung{'en' if n_bends > 1 else ''})"
        top = float((pts + offset).max(axis=0)[1])
        height = max(0.02, 0.05 * (top - float((pts + offset).min(axis=0)[1])))
        lines += [
            "0", "TEXT", "8", "BESCHRIFTUNG",
            "10", f"{cursor_x:.6f}", "20", f"{top + height:.6f}", "30", "0.0",
            "40", f"{height:.4f}", "1", label,
        ]
        cursor_x = float((pts + offset).max(axis=0)[0]) + gap
    lines += ["0", "ENDSEC", "0", "EOF"]
    path.write_text("\n".join(lines) + "\n")
    return path


def _axis_align_2d(poly: np.ndarray) -> np.ndarray:
    """Rotate a 2D outline so its dominant edge direction runs along +X."""
    segs = np.roll(poly, -1, axis=0) - poly
    lengths = np.hypot(segs[:, 0], segs[:, 1])
    k = int(np.argmax(lengths))
    ang = np.arctan2(segs[k, 1], segs[k, 0])
    c, s = np.cos(-ang), np.sin(-ang)
    rot = np.array([[c, -s], [s, c]])
    return poly @ rot.T
