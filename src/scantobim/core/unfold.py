"""Sheet metal unfolding (Abwicklung gekanteter Bleche).

A bent part (Kantteil) appears in the scan as several plates of the *same*
thickness meeting edge-to-edge at an angle. Unfolding rotates the chain of
plates into one plane and inserts the **bend allowance** — the neutral-axis
arc length ``BA = θ·(r + k·t)`` (k-factor default 0.44, inner radius
default = thickness) — between them. The result is the flat cutting contour
with **bend lines**, ready for laser/plasma plus press brake.

Junction classification matters: a T-joint (one plate's face meets another's
edge) can only be welded; an edge-to-edge corner may be a bend or a welded
corner seam — the scan alone cannot always tell a sharp bend from a corner
weld, so unfolding is an explicit request (``--unfold``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_K_FACTOR = 0.44


@dataclass
class FlatPart:
    plate_ids: list[int]
    loops: list[np.ndarray]  # one 2D outline per plate, flattened, 1:1
    bend_lines: list[tuple[np.ndarray, np.ndarray]]  # 2D segments
    bend_angles_deg: list[float]
    thickness: float
    catalog_thickness: float | None = None
    outline_area: float = 0.0

    @property
    def size(self) -> tuple[float, float]:
        pts = np.vstack(self.loops)
        ext = pts.max(axis=0) - pts.min(axis=0)
        return float(max(ext)), float(min(ext))


def classify_junctions(plates: list, seams: list) -> list[str]:
    """``corner`` (edge-to-edge — bend or corner weld) vs ``tee``
    (face-to-edge — weld only) for every detected seam."""
    return [
        "corner"
        if _hinge_on_boundary(plates[s.plate_a], s)
        and _hinge_on_boundary(plates[s.plate_b], s)
        else "tee"
        for s in seams
    ]


def unfold_parts(
    plates: list,
    seams: list,
    k_factor: float = DEFAULT_K_FACTOR,
    inner_radius: float | None = None,
) -> list[FlatPart]:
    """Unfold chains of corner-connected same-gauge plates into flat parts.

    Each connected component of the bend graph becomes one part, flattened
    into the plane of its largest plate along a spanning tree (cycles —
    closed boxes — are broken at the shortest hinge).
    """
    kinds = classify_junctions(plates, seams)
    bends = []
    for seam, kind in zip(seams, kinds):
        if kind != "corner":
            continue
        ta = plates[seam.plate_a].catalog_thickness or plates[seam.plate_a].thickness
        tb = plates[seam.plate_b].catalog_thickness or plates[seam.plate_b].thickness
        if abs(ta - tb) > 0.25 * max(ta, tb):
            continue  # different gauge → separate (welded) parts
        bends.append(seam)

    parent = list(range(len(plates)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s in bends:
        ra, rb = find(s.plate_a), find(s.plate_b)
        if ra != rb:
            parent[ra] = rb

    parts = []
    for root in sorted({find(i) for i in range(len(plates))}):
        members = [i for i in range(len(plates)) if find(i) == root]
        comp_bends = [s for s in bends if find(s.plate_a) == root]
        parts.append(_unfold_component(plates, members, comp_bends, k_factor, inner_radius))
    return parts


def _unfold_component(plates, members, bends, k_factor, inner_radius) -> FlatPart:
    base_id = max(members, key=lambda i: plates[i].area)
    base = plates[base_id]
    thickness = base.catalog_thickness or base.thickness
    r = inner_radius if inner_radius is not None else thickness

    # Spanning tree, longest hinges first (cycles break at short hinges).
    ranked = sorted(bends, key=lambda s: -float(np.linalg.norm(s.end - s.start)))
    joined = {base_id}
    tree = []
    changed = True
    while changed:
        changed = False
        for s in ranked:
            a, b = s.plate_a, s.plate_b
            if (a in joined) == (b in joined):
                continue
            parent_id, child_id = (a, b) if a in joined else (b, a)
            tree.append((parent_id, child_id, s))
            joined.add(child_id)
            changed = True

    # flat_map[plate] maps 3D points OF THAT PLATE'S PLANE into the flat
    # pattern; children compose through their parent's map recursively.
    def base_map(p3: np.ndarray) -> np.ndarray:
        rel = p3 - base.origin
        return np.column_stack([rel @ base.u, rel @ base.v])

    flat_map = {base_id: base_map}
    loops = {base_id: base_map(base.outline_3d())}
    bend_lines = []
    bend_angles = []

    for parent_id, child_id, seam in tree:
        child = plates[child_id]
        theta = np.deg2rad(_bend_angle_deg(plates[parent_id], child))
        allowance = theta * (r + k_factor * thickness)
        bend_angles.append(round(float(np.rad2deg(theta)), 1))

        hinge_a3, hinge_b3 = seam.start, seam.end
        hinge_dir3 = hinge_b3 - hinge_a3
        hinge_dir3 = hinge_dir3 / max(np.linalg.norm(hinge_dir3), 1e-12)

        pmap = flat_map[parent_id]
        h_a = pmap(hinge_a3[None, :])[0]
        h_b = pmap(hinge_b3[None, :])[0]
        h_dir = h_b - h_a
        h_len = float(np.linalg.norm(h_dir))
        if h_len < 1e-12:
            continue
        h_dir = h_dir / h_len
        h_normal = np.array([-h_dir[1], h_dir[0]])

        # Parent material lies on one side of the hinge; unfold the child to
        # the other side, offset by the bend allowance.
        parent_center = loops[parent_id].mean(axis=0)
        side = float(np.sign((parent_center - h_a) @ h_normal))
        side = -side if side != 0 else 1.0

        def child_map(p3, _ha3=hinge_a3, _hd3=hinge_dir3, _ha=h_a,
                      _hd=h_dir, _hn=h_normal, _side=side, _ba=allowance):
            rel = p3 - _ha3
            along = rel @ _hd3
            perp = np.linalg.norm(rel - np.outer(along, _hd3), axis=1)
            return (
                _ha[None, :]
                + np.outer(along, _hd)
                + np.outer(_side * (perp + _ba), _hn)
            )

        flat_map[child_id] = child_map
        loops[child_id] = child_map(child.outline_3d())
        mid = side * allowance / 2.0
        bend_lines.append((h_a + mid * h_normal, h_b + mid * h_normal))

    loop_list = [loops[i] for i in members if i in loops]
    return FlatPart(
        plate_ids=sorted(members),
        loops=loop_list,
        bend_lines=bend_lines,
        bend_angles_deg=bend_angles,
        thickness=thickness,
        catalog_thickness=base.catalog_thickness,
        outline_area=float(sum(_area2(lp) for lp in loop_list)),
    )


def _hinge_on_boundary(plate, seam, tol_factor: float = 3.0) -> bool:
    """Does the seam line run along the plate outline (edge) rather than
    across its interior (face)?"""
    mid = (seam.start + seam.end) / 2.0
    rel = mid - plate.origin
    uv = np.array([float(rel @ plate.u), float(rel @ plate.v)])
    outline = plate.outline_2d
    d_boundary = min(
        _point_segment_dist(uv, outline[k], outline[(k + 1) % len(outline)])
        for k in range(len(outline))
    )
    return d_boundary < tol_factor * plate.thickness


def _point_segment_dist(p, a, b) -> float:
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-18 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _bend_angle_deg(a, b) -> float:
    """Bending angle θ between two plates (0 = flat, 90 = right angle)."""
    c = abs(float(a.normal @ b.normal))
    return float(np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0))))


def _area2(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
