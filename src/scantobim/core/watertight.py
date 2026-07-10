"""Globally optimized watertight reconstruction (PolyFit approach).

The greedy pipeline polygonizes each plane independently — scan shadows
(occluded ceiling patches, hidden wall strips) stay as holes. This module
instead solves the reconstruction *globally*:

1. Every detected plane is clipped to the scene box and partitioned by its
   intersection lines with all other planes into convex **candidate cells**.
2. Every cell gets a data score: how much of its area is supported by
   measured points.
3. A binary integer program (scipy/HiGHS) selects the subset of cells that
   maximizes point support, penalizes unsupported area and edge length, and
   enforces the **watertightness constraint**: along every elementary
   segment of every plane–plane intersection line, the number of selected
   incident cells must be exactly 0 or 2.

The result is a closed 2-manifold whenever the detected planes can bound
one — occluded regions are filled by the optimizer with the geometrically
exact face, not by smoothing (Nan & Wonka, *PolyFit*, ICCV 2017).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from scantobim.core.mesh import Mesh, weld_vertices

MAX_PLANES = 30


@dataclass
class _Cell:
    plane: int
    poly: np.ndarray  # (N, 2) convex, CCW, in the plane frame
    area: float = 0.0
    support: float = 0.0
    n_points: int = 0
    has_rim_edge: bool = False
    edges_on_lines: list = field(default_factory=list)  # (line_key, t0, t1)


def make_watertight(
    planes: list,
    points: np.ndarray,
    spacing: float,
    w_fit: float = 0.43,
    w_coverage: float = 0.27,
    w_complexity: float = 0.30,
    time_limit: float = 60.0,
) -> tuple[Mesh, list, dict]:
    """Select the watertight optimum over all candidate faces.

    Returns ``(mesh, cell_geometries, info)`` where each cell geometry is a
    ``(plane_index, polygon_3d)`` tuple for the CAD exporters.
    """
    n_planes = len(planes)
    if n_planes < 4:
        raise ValueError("watertight optimization needs at least 4 planes")
    if n_planes > MAX_PLANES:
        raise ValueError(
            f"watertight optimization is limited to {MAX_PLANES} planes "
            f"(got {n_planes}) — reduce max_planes or disable --watertight"
        )
    for p in planes:
        if p.basis is None:
            p.make_basis()

    lo = points.min(axis=0)
    hi = points.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    margin = 0.05 * diag
    lo = lo - margin
    hi = hi + margin
    eps = 1e-7 * diag

    box_faces = [(np.array(n_), float(c)) for n_, c in _box_halfspaces(lo, hi)]

    # ---- intersection lines (3D), keyed by the plane pair -----------------
    lines: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for i in range(n_planes):
        for j in range(i + 1, n_planes):
            direction = np.cross(planes[i].normal, planes[j].normal)
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                continue
            direction = direction / norm
            m = np.vstack([planes[i].normal, planes[j].normal, direction])
            point = np.linalg.solve(m, np.array([-planes[i].d, -planes[j].d, 0.0]))
            lines[(i, j)] = (point, direction)

    # ---- candidate cells: clip to box, split by lines ----------------------
    cells: list[_Cell] = []
    for pi, plane in enumerate(planes):
        rect = _plane_clipped_by_box(plane, box_faces, diag)
        if rect is None:
            continue
        polys = [rect]
        for (i, j), (lp, ld) in lines.items():
            if pi not in (i, j):
                continue
            # In-plane 2D line.
            p2 = plane.project_to_2d(lp[None, :])[0]
            d3 = ld - (ld @ plane.normal) * plane.normal
            n3 = np.linalg.norm(d3)
            if n3 < 1e-9:
                continue
            d2 = plane.project_to_2d((lp + ld)[None, :])[0] - p2
            d2 = d2 / np.linalg.norm(d2)
            normal2 = np.array([-d2[1], d2[0]])
            new_polys = []
            for poly in polys:
                a, b = _split_convex(poly, p2, normal2, eps)
                if a is not None:
                    new_polys.append(a)
                if b is not None:
                    new_polys.append(b)
            polys = new_polys
        for poly in polys:
            area = _poly_area(poly)
            if area < (0.5 * spacing) ** 2:
                continue
            cells.append(_Cell(plane=pi, poly=poly, area=area))

    # ---- support per cell ---------------------------------------------------
    # Covered area via an occupancy raster: counting points times the median
    # spacing squared underestimates coverage ~4x (Poisson statistics), an
    # occupied-texel count measures the actual covered area.
    texel = 2.0 * spacing
    for pi, plane in enumerate(planes):
        uv = plane.project_to_2d(points[plane.inliers])
        occupied = np.unique(np.floor(uv / texel).astype(np.int64), axis=0)
        texel_centers = (occupied + 0.5) * texel
        plane_cells = [c for c in cells if c.plane == pi]
        for cell in plane_cells:
            cell.n_points = int(_points_in_convex(uv, cell.poly, eps).sum())
            covered = _points_in_convex(texel_centers, cell.poly, eps)
            cell.support = min(float(covered.sum()) * texel**2, cell.area)

    # ---- edge classification -------------------------------------------------
    line_items = list(lines.items())
    for cell in cells:
        plane = planes[cell.plane]
        poly3 = plane.lift_to_3d(cell.poly)
        m = len(poly3)
        for k in range(m):
            a3, b3 = poly3[k], poly3[(k + 1) % m]
            if np.linalg.norm(b3 - a3) < eps:
                continue
            if _on_box_face(a3, b3, box_faces, eps):
                cell.has_rim_edge = True
                continue
            assigned = False
            for key, (lp, ld) in line_items:
                if cell.plane not in key:
                    continue
                if _dist_to_line(a3, lp, ld) < 5 * eps and _dist_to_line(b3, lp, ld) < 5 * eps:
                    t0 = float((a3 - lp) @ ld)
                    t1 = float((b3 - lp) @ ld)
                    cell.edges_on_lines.append((key, min(t0, t1), max(t0, t1)))
                    assigned = True
                    break
            if not assigned:
                # Numerical orphan — treat as rim so the cell can't leave
                # an open boundary.
                cell.has_rim_edge = True

    # ---- elementary intervals + constraints ----------------------------------
    intervals: list[tuple[float, list[int]]] = []  # (length, incident cell ids)
    for key in lines:
        entries = []
        for ci, cell in enumerate(cells):
            for k2, t0, t1 in cell.edges_on_lines:
                if k2 == key:
                    entries.append((t0, t1, ci))
        if not entries:
            continue
        breaks = sorted({t for t0, t1, _ in entries for t in (t0, t1)})
        for b0, b1 in zip(breaks, breaks[1:]):
            if b1 - b0 < 10 * eps:
                continue
            mid = (b0 + b1) / 2.0
            inc = [ci for t0, t1, ci in entries if t0 - eps <= mid <= t1 + eps]
            if inc:
                intervals.append((b1 - b0, inc))

    # ---- integer program (PolyFit energy) --------------------------------------
    # Not selecting a face leaves its measured points unexplained — that is
    # what costs, so the trivial empty solution never wins:
    #   E = -w_fit·(covered points) + w_cov·(unsupported area) + w_cmpl·(edges)
    # with each term normalized to [0, 1].
    n_cells = len(cells)
    n_edges = len(intervals)
    n_points_total = max(sum(len(p.inliers) for p in planes), 1)
    area_ref = max(sum(c.support for c in cells), 1e-12)
    len_ref = max(sum(length for length, _ in intervals), 1e-12)
    cost = np.zeros(n_cells + n_edges)
    for ci, cell in enumerate(cells):
        cost[ci] = (
            -w_fit * cell.n_points / n_points_total
            + w_coverage * max(cell.area - cell.support, 0.0) / area_ref
        )
    for ei, (length, _) in enumerate(intervals):
        cost[n_cells + ei] = w_complexity * length / len_ref

    rows, cols, vals = [], [], []
    for ei, (_, inc) in enumerate(intervals):
        for ci in inc:
            rows.append(ei)
            cols.append(ci)
            vals.append(1.0)
        rows.append(ei)
        cols.append(n_cells + ei)
        vals.append(-2.0)
    a_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_edges, n_cells + n_edges))

    ub = np.ones(n_cells + n_edges)
    for ci, cell in enumerate(cells):
        if cell.has_rim_edge:
            ub[ci] = 0.0  # a rim edge can never be closed → cell forbidden

    res = milp(
        c=cost,
        constraints=LinearConstraint(a_mat, 0.0, 0.0),
        integrality=np.ones(n_cells + n_edges),
        bounds=Bounds(np.zeros(n_cells + n_edges), ub),
        options={"time_limit": time_limit},
    )
    if res.x is None:
        raise ValueError("watertight optimization found no solution")
    selected = [ci for ci in range(n_cells) if res.x[ci] > 0.5]
    if not selected:
        raise ValueError(
            "watertight optimization selected no faces — the detected planes "
            "cannot bound a closed volume (e.g. open scan without a lid plane)"
        )

    # ---- assemble mesh --------------------------------------------------------
    verts_out, faces_out, groups_out = [], [], []
    geometries = []
    base = 0
    for ci in selected:
        cell = cells[ci]
        poly3 = planes[cell.plane].lift_to_3d(cell.poly)
        geometries.append((cell.plane, poly3))
        m = len(poly3)
        verts_out.append(poly3)
        for k in range(1, m - 1):
            faces_out.append([base, base + k, base + k + 1])
        groups_out.extend([cell.plane] * (m - 2))
        base += m
    mesh = Mesh(
        vertices=np.vstack(verts_out),
        faces=np.array(faces_out, dtype=np.int64),
        face_groups=np.array(groups_out, dtype=np.int64),
    )
    mesh = weld_vertices(mesh, tolerance=max(10 * eps, 1e-9))
    _orient_consistently(mesh)

    info = {
        "candidate_faces": n_cells,
        "selected_faces": len(selected),
        "edge_constraints": n_edges,
        "objective": round(float(res.fun), 6),
        "solver_status": int(res.status),
    }
    return mesh, geometries, info


# ------------------------------------------------------------------ geometry ops

def _box_halfspaces(lo: np.ndarray, hi: np.ndarray):
    for axis in range(3):
        n_pos = np.zeros(3)
        n_pos[axis] = 1.0
        yield n_pos.copy(), hi[axis]
        yield -n_pos, -lo[axis]


def _plane_clipped_by_box(plane, box_faces, diag: float) -> np.ndarray | None:
    """The plane's intersection with the scene box, as a 2D convex polygon."""
    half = 4.0 * diag
    poly = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    for n_, c in box_faces:
        # Half-space n·X <= c, evaluated on lifted vertices.
        lifted = plane.lift_to_3d(poly)
        signed = lifted @ n_ - c
        poly = _clip_by_signed(poly, signed)
        if poly is None or len(poly) < 3:
            return None
    return poly


def _clip_by_signed(poly: np.ndarray, signed: np.ndarray) -> np.ndarray | None:
    """Sutherland–Hodgman keep-side (signed <= 0) with precomputed values."""
    out = []
    m = len(poly)
    for k in range(m):
        a, b = poly[k], poly[(k + 1) % m]
        sa, sb = signed[k], signed[(k + 1) % m]
        if sa <= 0:
            out.append(a)
        if (sa < 0 < sb) or (sb < 0 < sa):
            t = sa / (sa - sb)
            out.append(a + t * (b - a))
    if len(out) < 3:
        return None
    return np.array(out)


def _split_convex(poly: np.ndarray, line_pt: np.ndarray, line_normal: np.ndarray, eps: float):
    """Split a convex polygon by a line; returns (negative side, positive side)."""
    signed = (poly - line_pt) @ line_normal
    if np.all(signed <= eps):
        return poly, None
    if np.all(signed >= -eps):
        return None, poly
    neg = _clip_by_signed(poly, signed)
    pos = _clip_by_signed(poly, -signed)
    return neg, pos


def _poly_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _points_in_convex(uv: np.ndarray, poly: np.ndarray, eps: float) -> np.ndarray:
    inside = np.ones(len(uv), dtype=bool)
    m = len(poly)
    # CCW assumed; allow either orientation by checking the polygon's sign.
    sign = 1.0 if _signed(poly) > 0 else -1.0
    for k in range(m):
        a, b = poly[k], poly[(k + 1) % m]
        edge = b - a
        cross = sign * ((uv[:, 0] - a[0]) * edge[1] - (uv[:, 1] - a[1]) * edge[0])
        inside &= cross <= eps
    return inside


def _signed(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _on_box_face(a: np.ndarray, b: np.ndarray, box_faces, eps: float) -> bool:
    for n_, c in box_faces:
        if abs(a @ n_ - c) < 5 * eps and abs(b @ n_ - c) < 5 * eps:
            return True
    return False


def _dist_to_line(p: np.ndarray, lp: np.ndarray, ld: np.ndarray) -> float:
    rel = p - lp
    return float(np.linalg.norm(rel - (rel @ ld) * ld))


def _orient_consistently(mesh: Mesh) -> None:
    """Flip triangle windings so the closed mesh is consistently outward."""
    faces = mesh.faces
    n_f = len(faces)
    edge_map: dict[tuple[int, int], list[int]] = {}
    for fi, f in enumerate(faces):
        for k in range(3):
            key = tuple(sorted((int(f[k]), int(f[(k + 1) % 3]))))
            edge_map.setdefault(key, []).append(fi)

    visited = np.zeros(n_f, dtype=bool)
    for start in range(n_f):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        while stack:
            fi = stack.pop()
            f = faces[fi]
            directed = {(int(f[k]), int(f[(k + 1) % 3])) for k in range(3)}
            for k in range(3):
                key = tuple(sorted((int(f[k]), int(f[(k + 1) % 3]))))
                for fj in edge_map[key]:
                    if fj == fi or visited[fj]:
                        continue
                    g = faces[fj]
                    g_dir = {(int(g[k2]), int(g[(k2 + 1) % 3])) for k2 in range(3)}
                    # Consistent orientation = the shared edge runs in
                    # opposite directions in the two faces.
                    if directed & g_dir:
                        faces[fj] = g[::-1]
                    visited[fj] = True
                    stack.append(fj)

    # Global sign: outward normals give positive volume.
    tri = mesh.vertices[mesh.faces]
    volume = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    if volume < 0:
        mesh.faces = mesh.faces[:, ::-1]
