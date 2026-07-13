"""Bridge structure recognition (Brückenbauwerke).

From a bridge scan the module identifies the structural system and the
numbers an inspector or planner records first:

* **Deck** (Überbau) — the dominant elongated horizontal surface: axis,
  length, width, elevation, area.
* **Substructure** — piers (Pfeiler) and abutments (Widerlager) below the
  deck, with positions along the bridge axis, the resulting **span layout**
  (Feldweiten) and **bearing points** (Lagerpunkte) at the pier tops.
* **Arch** (Bogen) — a horizontal-axis cylinder segment under the deck:
  span, rise (Stich), radius.
* **Pylons and cables** — tall verticals above the deck plus thin, long
  cylinders: hangers (vertical) vs stay cables (diagonal).
* **Type classification** — Bogen-, Schrägseil-, Hänge-, Balken-/
  Plattenbrücke from the found components.

All detectors scale from footbridges to viaducts because every threshold is
relative to the measured deck size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scantobim.core.cloud import PointCloud
from scantobim.core.machinery import _cluster_points, detect_cylinders
from scantobim.core.planes import detect_planes


@dataclass
class BridgeDeck:
    center: np.ndarray
    axis: np.ndarray  # unit, along the bridge
    length: float
    width: float
    elevation: float  # top surface z
    area: float


@dataclass
class Pier:
    position: np.ndarray  # centroid
    station: float  # position along the bridge axis (0 = deck start)
    height: float
    footprint: float  # max horizontal extent
    bearing: np.ndarray  # bearing point (top center)


@dataclass
class Cable:
    start: np.ndarray
    end: np.ndarray
    length: float
    diameter: float
    inclination_deg: float  # 0 = vertical (hanger), 90 = horizontal


@dataclass
class BridgeReport:
    bridge_type: str
    deck: BridgeDeck
    piers: list[Pier] = field(default_factory=list)
    abutments: int = 0
    spans: list[float] = field(default_factory=list)
    arch: dict | None = None
    pylons: int = 0
    cables: list[Cable] = field(default_factory=list)


def analyze_bridge(cloud: PointCloud, seed: int = 7) -> dict:
    """Detect the structural system of a scanned bridge."""
    from scantobim.core.preprocess import (
        estimate_normals,
        estimate_point_spacing,
        remove_statistical_outliers,
        voxel_downsample,
    )

    raw_spacing = estimate_point_spacing(cloud)
    work = voxel_downsample(cloud, 2.0 * raw_spacing)
    work, _ = remove_statistical_outliers(work)
    work = estimate_normals(work, k_neighbors=12)
    spacing = estimate_point_spacing(work)

    deck = _find_deck(work, spacing, seed)
    if deck is None:
        raise ValueError(
            "no bridge deck found — expected a dominant elongated horizontal "
            "surface (Überbau)"
        )

    # Split the scene relative to the deck.
    rel = work.points - deck.center
    station = rel @ deck.axis
    z = work.points[:, 2]
    below_mask = (
        (z < deck.elevation - deck_thickness_guess(deck, spacing))
        & (np.abs(station) < 0.55 * deck.length)
    )
    above_mask = (z > deck.elevation + 8 * spacing) & (
        np.abs(station) < 0.55 * deck.length
    )

    piers, abutment_objs = _find_substructure(work, below_mask, deck, spacing)
    abutments = len(abutment_objs)
    arch, arch_geom = _find_arch(work, below_mask, deck, spacing, seed)
    pylon_objs, cables = _find_cables_and_pylons(work, above_mask, deck, spacing, seed)
    pylons = len(pylon_objs)

    spans = _span_layout(deck, piers)
    bridge_type = _classify(deck, piers, arch, pylons, cables)

    report = {
        "points": len(cloud),
        "analyzed_points": len(work),
        "bridge_type": bridge_type,
        "deck": {
            "length": round(deck.length, 3),
            "width": round(deck.width, 3),
            "elevation": round(deck.elevation, 3),
            "area": round(deck.area, 2),
            "axis": [round(float(x), 4) for x in deck.axis],
            "center": [round(float(x), 3) for x in deck.center],
        },
        "piers": [
            {
                "station": round(p.station, 3),
                "height": round(p.height, 3),
                "footprint": round(p.footprint, 3),
                "bearing_point": [round(float(x), 3) for x in p.bearing],
            }
            for p in piers
        ],
        "abutments": abutments,
        "spans": [round(s, 3) for s in spans],
        "arch": arch,
        "pylons": pylons,
        "cables": [
            {
                "length": round(c.length, 3),
                "diameter": round(c.diameter, 4),
                "inclination_deg": round(c.inclination_deg, 1),
            }
            for c in cables
        ],
        "_objects": {
            "deck": deck,
            "piers": piers,
            "cables": cables,
            "abutments": abutment_objs,
            "pylons": pylon_objs,
            "arch": arch_geom,
            "deck_thickness": deck_thickness_guess(deck, spacing),
        },
    }
    return report


def deck_thickness_guess(deck: BridgeDeck, spacing: float) -> float:
    return max(0.02 * deck.length, 8 * spacing)


def _find_deck(work, spacing: float, seed: int) -> BridgeDeck | None:
    planes, _ = detect_planes(
        work.points,
        work.normals,
        distance_threshold=3.0 * spacing,
        min_inliers=max(300, int(0.02 * len(work))),
        max_planes=24,
        seed=seed,
    )
    candidates: list[tuple[float, BridgeDeck]] = []
    for plane in planes:
        if abs(float(plane.normal @ [0, 0, 1])) < 0.9:
            continue  # not horizontal
        pts = work.points[plane.inliers]
        centered = pts - pts.mean(axis=0)
        cov = centered.T @ centered / len(pts)
        vals, vecs = np.linalg.eigh(cov)
        length = 4.0 * np.sqrt(vals[2])  # ~full extent of a uniform strip
        width = 4.0 * np.sqrt(vals[1])
        if width <= 0 or length / max(width, 1e-9) < 2.0:
            continue  # decks are elongated
        axis = vecs[:, 2].copy()
        axis[2] = 0.0
        n = np.linalg.norm(axis)
        if n < 1e-9:
            continue
        axis /= n
        t = centered @ vecs[:, 2]
        w = centered @ vecs[:, 1]
        deck = BridgeDeck(
            center=pts.mean(axis=0),
            axis=axis,
            length=float(np.quantile(t, 0.995) - np.quantile(t, 0.005)),
            width=float(np.quantile(w, 0.995) - np.quantile(w, 0.005)),
            elevation=float(np.quantile(pts[:, 2], 0.9)),
            area=float(length * width),
        )
        candidates.append((float(length * width), deck))
    if not candidates:
        return None
    # The top and bottom faces of a deck slab score alike — take the driving
    # surface: the highest among the near-best candidates.
    best_score = max(s for s, _ in candidates)
    top = [d for s, d in candidates if s >= 0.7 * best_score]
    return max(top, key=lambda d: d.elevation)


def _find_substructure(work, below_mask, deck: BridgeDeck, spacing: float):
    piers: list[Pier] = []
    abutments: list[dict] = []
    idx = np.flatnonzero(below_mask)
    if len(idx) < 200:
        return piers, abutments
    pts = work.points[idx]
    clusters = _cluster_points(pts, radius=6 * spacing)
    for cl in clusters:
        if len(cl) < 200:
            continue
        cpts = pts[cl]
        height = float(cpts[:, 2].max() - cpts[:, 2].min())
        if height < 0.1 * deck.length and height < 8 * spacing * 10:
            continue
        horiz = cpts[:, :2]
        footprint = float(
            np.linalg.norm(horiz.max(axis=0) - horiz.min(axis=0))
        )
        centroid = cpts.mean(axis=0)
        station = float((centroid - deck.center) @ deck.axis)
        at_end = abs(station) > 0.4 * deck.length
        if at_end and footprint > 0.5 * deck.width:
            abutments.append(
                {
                    "station": station,
                    "center_xy": centroid[:2],
                    "footprint": footprint,
                    "z_min": float(cpts[:, 2].min()),
                    "z_max": float(cpts[:, 2].max()),
                }
            )
            continue
        if footprint > 1.2 * deck.width:
            continue  # terrain / embankment, not a pier
        top = cpts[cpts[:, 2] > cpts[:, 2].max() - 4 * spacing].mean(axis=0)
        piers.append(
            Pier(
                position=centroid,
                station=station,
                height=height,
                footprint=footprint,
                bearing=top,
            )
        )
    piers.sort(key=lambda p: p.station)
    return piers, abutments


def _find_arch(work, below_mask, deck: BridgeDeck, spacing: float, seed: int):
    idx = np.flatnonzero(below_mask)
    if len(idx) < 500:
        return None, None
    cylinders, _ = detect_cylinders(
        work.points[idx],
        work.normals[idx],
        distance_threshold=4.0 * spacing,
        min_inliers=max(400, int(0.1 * len(idx))),
        max_cylinders=4,
        ransac_iterations=900,
        seed=seed,
    )
    for c in cylinders:
        horizontal = abs(float(c.axis @ [0, 0, 1])) < 0.2
        across = abs(float(c.axis @ deck.axis)) < 0.4
        big = c.radius > 0.1 * deck.length
        if not (horizontal and across and big):
            continue
        pts = work.points[idx[c.inliers]]
        span = float(
            np.quantile((pts - deck.center) @ deck.axis, 0.99)
            - np.quantile((pts - deck.center) @ deck.axis, 0.01)
        )
        rise = float(pts[:, 2].max() - pts[:, 2].min())
        # Geometry for the 3D model: angular extent of the barrel about its
        # axis (0 = crown / straight up) and its width across the bridge.
        e2 = np.array([0.0, 0.0, 1.0]) - float(c.axis[2]) * c.axis
        e2 /= max(np.linalg.norm(e2), 1e-12)
        e1 = np.cross(e2, c.axis)
        rel = pts - c.center
        theta = np.arctan2(rel @ e1, rel @ e2)
        t_axial = rel @ c.axis
        geometry = {
            "center": c.center,
            "axis": c.axis,
            "radius": c.radius,
            "theta_min": float(np.quantile(theta, 0.01)),
            "theta_max": float(np.quantile(theta, 0.99)),
            "width": float(np.quantile(t_axial, 0.99) - np.quantile(t_axial, 0.01)),
        }
        return {
            "radius": round(c.radius, 3),
            "span": round(span, 3),
            "rise": round(rise, 3),
        }, geometry
    return None, None


def _find_cables_and_pylons(work, above_mask, deck: BridgeDeck, spacing: float, seed: int):
    """Pylons via 2D density (tall, tiny footprint), cables via line RANSAC.

    Cables attach to the pylon, so connectivity clustering would merge them
    into one blob — the two detectors below are immune to that.
    """
    pylon_objs: list[dict] = []
    cables: list[Cable] = []
    idx = np.flatnonzero(above_mask)
    if len(idx) < 200:
        return pylon_objs, cables
    pts = work.points[idx]

    # ---- pylons: xy cells with a large vertical extent -----------------
    cell = max(0.05 * deck.width, 4 * spacing)
    keys = np.floor(pts[:, :2] / cell).astype(np.int64)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    z_min = np.full(len(uniq), np.inf)
    z_max = np.full(len(uniq), -np.inf)
    np.minimum.at(z_min, inverse, pts[:, 2])
    np.maximum.at(z_max, inverse, pts[:, 2])
    tall = (z_max - z_min) > 0.15 * deck.length
    pylon_mask = np.zeros(len(pts), dtype=bool)
    if tall.any():
        # A pylon fills its height CONTINUOUSLY; cables crossing above each
        # other only occupy thin z-bands of a cell. Keep tall cells whose
        # vertical occupancy is dense, then group them (a pylon is thick —
        # several connected cells; a near-vertical cable is one or two).
        continuous = np.zeros(len(uniq), dtype=bool)
        for ci in np.flatnonzero(tall):
            sel = inverse == ci
            zs = pts[sel, 2]
            span = z_max[ci] - z_min[ci]
            n_bins = max(4, int(span / cell))
            hist, _ = np.histogram(zs, bins=n_bins, range=(z_min[ci], z_max[ci]))
            if (hist > 0).mean() > 0.6:
                continuous[ci] = True
        groups = _cell_groups(uniq[continuous])
        big_groups = [g for g in groups if len(g) >= 4]
        cell_index = {tuple(c): i for i, c in enumerate(uniq)}
        for g in big_groups:
            gc = np.array(g, dtype=np.float64)
            gi = [cell_index[c] for c in g]
            pylon_objs.append(
                {
                    "center": ((gc.min(axis=0) + gc.max(axis=0) + 1.0) / 2.0 * cell),
                    "size": np.maximum((gc.max(axis=0) - gc.min(axis=0) + 1.0) * cell, 0.3),
                    "z_min": float(np.min(z_min[gi])),
                    "z_max": float(np.max(z_max[gi])),
                }
            )
        pylon_cells = {c for g in big_groups for c in g}
        cell_is_pylon = np.array(
            [tuple(c) in pylon_cells for c in uniq], dtype=bool
        )
        pylon_mask = cell_is_pylon[inverse]

    # ---- cables: sequential line RANSAC on the remaining points --------
    rest = pts[~pylon_mask]
    rng = np.random.default_rng(seed)
    tol = 3.0 * spacing
    min_line_pts = 60
    remaining = np.arange(len(rest))
    for _ in range(24):
        if len(remaining) < min_line_pts:
            break
        sub = rest[remaining]
        best_mask, best_count = None, 0
        for _ in range(500):
            i, j = rng.integers(len(sub), size=2)
            d = sub[j] - sub[i]
            norm = np.linalg.norm(d)
            if norm < 0.05 * deck.length:
                continue
            d = d / norm
            rel = sub - sub[i]
            dist = np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)
            mask = dist < tol
            count = int(mask.sum())
            if count > best_count:
                best_count, best_mask = count, mask
        if best_mask is None or best_count < min_line_pts:
            break
        cpts = sub[best_mask]
        centered = cpts - cpts.mean(axis=0)
        vals, vecs = np.linalg.eigh(centered.T @ centered / len(cpts))
        axis = vecs[:, 2]
        elong = np.sqrt(vals[2] / max(vals[1], 1e-12))
        t = centered @ axis
        length = float(np.quantile(t, 0.99) - np.quantile(t, 0.01))
        thickness = 2.0 * float(np.sqrt(max(vals[1], 0.0)))
        if elong > 8 and length > 0.05 * deck.length and thickness < 0.05 * deck.width:
            vertical = abs(float(axis @ [0, 0, 1]))
            candidate = Cable(
                start=cpts[int(np.argmin(t))],
                end=cpts[int(np.argmax(t))],
                length=length,
                diameter=thickness,
                inclination_deg=float(np.rad2deg(np.arccos(np.clip(vertical, 0, 1)))),
            )
            # Leftover fragments of an already-found cable refit as a second
            # line. Fragments are COLLINEAR with the original — their
            # midpoints may sit anywhere along the cable — so the robust
            # duplicate test is parallel direction + distance of the new
            # midpoint from the existing cable's infinite line.
            mid = (candidate.start + candidate.end) / 2.0
            cand_dir = candidate.end - candidate.start
            cand_dir = cand_dir / max(np.linalg.norm(cand_dir), 1e-12)
            duplicate = False
            for c in cables:
                d = c.end - c.start
                d = d / max(np.linalg.norm(d), 1e-12)
                if abs(float(d @ cand_dir)) < np.cos(np.deg2rad(5.0)):
                    continue
                rel = mid - c.start
                line_dist = np.linalg.norm(rel - (rel @ d) * d)
                if line_dist < 0.02 * deck.length:
                    duplicate = True
                    break
            if not duplicate:
                cables.append(candidate)
        remaining = remaining[~best_mask]
    return pylon_objs, cables


def _cell_groups(cells: np.ndarray) -> list[list[tuple]]:
    """8-connected groups among integer grid cells."""
    cell_set = {tuple(c) for c in cells}
    groups: list[list[tuple]] = []
    while cell_set:
        seed_cell = cell_set.pop()
        group = [seed_cell]
        stack = [seed_cell]
        while stack:
            cx, cy = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cx + dx, cy + dy)
                    if nb in cell_set:
                        cell_set.remove(nb)
                        group.append(nb)
                        stack.append(nb)
        groups.append(group)
    return groups


def _span_layout(deck: BridgeDeck, piers: list[Pier]) -> list[float]:
    """Field lengths between abutment — piers — abutment along the axis."""
    stations = [-deck.length / 2] + [p.station for p in piers] + [deck.length / 2]
    return [b - a for a, b in zip(stations, stations[1:])]


def _classify(deck, piers, arch, pylons, cables) -> str:
    if arch is not None:
        return "Bogenbruecke"
    if cables:
        diagonal = [c for c in cables if c.inclination_deg > 25]
        if pylons and len(diagonal) >= max(2, len(cables) // 2):
            return "Schraegseilbruecke"
        if pylons:
            return "Haengebruecke"
    if piers:
        return "Balkenbruecke"
    return "Platten-/Balkenbruecke (Einfeld)"


# ---------------------------------------------------------------- 3D model

def bridge_model_mesh(report: dict) -> "Mesh":
    """Build a solid 3D model of the bridge from the analysis result.

    Every detected element becomes a true volume: deck slab, piers,
    abutments, arch barrel (swept annular sector), pylons and cables —
    with named groups so CAD/viewers can address single members.
    """
    from scantobim.core.geometry3d import box_mesh, extrude_polygon
    from scantobim.core.machinery import cylinder_mesh
    from scantobim.core.mesh import merge_meshes

    objects = report["_objects"]
    deck: BridgeDeck = objects["deck"]
    piers: list[Pier] = objects["piers"]
    cables: list[Cable] = objects["cables"]
    thickness = objects["deck_thickness"]

    parts = []
    names: dict[int, str] = {}

    # Deck slab: top face at the measured elevation.
    deck_center = np.array(
        [deck.center[0], deck.center[1], deck.elevation - thickness / 2.0]
    )
    parts.append(
        box_mesh(
            deck_center, deck.axis, (deck.length, deck.width, thickness),
            color=(205, 205, 210), group=0,
        )
    )
    names[0] = "ueberbau"

    for i, p in enumerate(piers):
        side = max(p.footprint / np.sqrt(2.0), 0.3)
        center = np.array([p.position[0], p.position[1], p.bearing[2] - p.height / 2.0])
        parts.append(
            box_mesh(
                center, deck.axis, (side, side, p.height),
                color=(168, 166, 160), group=10 + i,
            )
        )
        names[10 + i] = f"pfeiler_{i + 1}"

    for i, a in enumerate(objects.get("abutments", [])):
        depth = max(a["footprint"] / np.sqrt(2.0), 0.5)
        height = max(a["z_max"] - a["z_min"], thickness)
        center = np.array([a["center_xy"][0], a["center_xy"][1], a["z_min"] + height / 2.0])
        parts.append(
            box_mesh(
                center, deck.axis, (depth, max(deck.width, a["footprint"] / 2), height),
                color=(168, 166, 160), group=40 + i,
            )
        )
        names[40 + i] = f"widerlager_{i + 1}"

    arch = objects.get("arch")
    if arch is not None:
        r, t = arch["radius"], max(0.04 * arch["radius"], thickness / 2)
        e2 = np.array([0.0, 0.0, 1.0]) - float(arch["axis"][2]) * arch["axis"]
        e2 /= max(np.linalg.norm(e2), 1e-12)
        e1 = np.cross(e2, arch["axis"])
        theta = np.linspace(arch["theta_min"], arch["theta_max"], 40)
        # Annular sector in the (e1, e2) plane; theta measured from vertical.
        outer_arc = np.column_stack([(r + t / 2) * np.sin(theta), (r + t / 2) * np.cos(theta)])
        inner_arc = np.column_stack([(r - t / 2) * np.sin(theta), (r - t / 2) * np.cos(theta)])
        ring = np.vstack([outer_arc, inner_arc[::-1]])
        parts.append(
            extrude_polygon(
                ring, arch["center"], e1, e2, arch["axis"], arch["width"],
                color=(176, 170, 158), group=60,
            )
        )
        names[60] = "bogen"

    for i, py in enumerate(objects.get("pylons", [])):
        height = py["z_max"] - py["z_min"]
        center = np.array([py["center"][0], py["center"][1], py["z_min"] + height / 2.0])
        parts.append(
            box_mesh(
                center, deck.axis, (py["size"][0], py["size"][1], height),
                color=(150, 155, 165), group=70 + i,
            )
        )
        names[70 + i] = f"pylon_{i + 1}"

    for i, c in enumerate(cables):
        direction = c.end - c.start
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            continue
        parts.append(
            cylinder_mesh(
                (c.start + c.end) / 2.0, direction / length,
                max(c.diameter / 2.0, 0.02), length,
                color=(90, 95, 105), segments=16, group=100 + i,
            )
        )
        names[100 + i] = f"seil_{i + 1}"

    model = merge_meshes(parts)
    model.group_names = names
    return model
