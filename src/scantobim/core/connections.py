"""Structural connection details (Anschlussdetails).

Where detected members meet, there is a connection — the detail a steel
detailer or inspector documents. From the member axes (steel module) the
module derives:

* **Nodes** (Knotenpunkte) — locations where two or more member axes come
  within a connection tolerance of each other; nearby nodes merge.
* Per node: the **connected members** with their profiles, the **angles**
  between them, and the connection **eccentricity** (axis offset — a
  quality indicator for the detail).
* **Connection plates** (Knoten-/Kopfplatten) — sheet metal plates from the
  sheet metal detector that sit within the connection zone.

The result is a machine-readable connection schedule (Anschlussliste).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConnectionNode:
    position: np.ndarray
    members: list[int]  # indices into the member list
    angles_deg: list[float]  # pairwise angles between the member axes
    eccentricity: float  # max axis-to-axis distance at the node
    plates: list[int] = field(default_factory=list)  # indices into plates


def detect_connections(
    members: list,
    tolerance: float | None = None,
    plates: list | None = None,
    plate_radius_factor: float = 3.0,
) -> list[ConnectionNode]:
    """Find connection nodes between structural members.

    ``tolerance`` defaults to 1.5x the largest profile height — member axes
    do not meet exactly (gusset offsets, eccentric bolting), so the search
    radius must cover realistic detailing offsets.
    """
    if len(members) < 2:
        return []
    if tolerance is None:
        tolerance = 1.5 * max(m.height for m in members)

    # Candidate nodes: closest points between member axis segments.
    raw: list[tuple[np.ndarray, int, int, float]] = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            p, dist = _segment_closest_point(members[i], members[j])
            if dist <= tolerance:
                raw.append((p, i, j, dist))

    # Merge candidates within tolerance into nodes.
    nodes: list[ConnectionNode] = []
    for p, i, j, dist in raw:
        merged = False
        for node in nodes:
            if np.linalg.norm(node.position - p) <= 2.0 * tolerance:
                k = len(node.members)
                node.position = (node.position * k + p) / (k + 1)
                for m in (i, j):
                    if m not in node.members:
                        node.members.append(m)
                node.eccentricity = max(node.eccentricity, dist)
                merged = True
                break
        if not merged:
            nodes.append(
                ConnectionNode(
                    position=p, members=[i, j], angles_deg=[], eccentricity=dist
                )
            )

    for node in nodes:
        node.members.sort()
        node.angles_deg = [
            round(_axis_angle(members[a], members[b]), 1)
            for ai, a in enumerate(node.members)
            for b in node.members[ai + 1 :]
        ]
        if plates:
            radius = plate_radius_factor * tolerance
            for pi, plate in enumerate(plates):
                center = plate.origin + (
                    plate.outline_2d.mean(axis=0)[0] * plate.u
                    + plate.outline_2d.mean(axis=0)[1] * plate.v
                )
                if np.linalg.norm(center - node.position) <= radius:
                    node.plates.append(pi)
    return nodes


def connections_report(nodes: list[ConnectionNode], members: list) -> list[dict]:
    return [
        {
            "position": [round(float(x), 3) for x in n.position],
            "members": [
                {"index": m, "profile": members[m].profile} for m in n.members
            ],
            "angles_deg": n.angles_deg,
            "eccentricity": round(n.eccentricity, 4),
            "connection_plates": n.plates,
        }
        for n in nodes
    ]


def _segment_closest_point(a, b) -> tuple[np.ndarray, float]:
    """Closest point between the axis segments of two members."""
    p1 = a.centroid - a.axis * a.length / 2
    d1 = a.axis * a.length
    p2 = b.centroid - b.axis * b.length / 2
    d2 = b.axis * b.length

    r = p1 - p2
    a11 = d1 @ d1
    a12 = d1 @ d2
    a22 = d2 @ d2
    b1 = d1 @ r
    b2 = d2 @ r
    denom = a11 * a22 - a12 * a12
    if abs(denom) < 1e-12 * max(a11 * a22, 1e-30):
        s = 0.0
        t = np.clip(b2 / a22 if a22 > 0 else 0.0, 0.0, 1.0)
    else:
        s = np.clip((a12 * b2 - a22 * b1) / denom, 0.0, 1.0)
        t = np.clip((a11 * b2 - a12 * b1) / denom, 0.0, 1.0)
    # One re-clamp round for the segment case.
    q1 = p1 + s * d1
    t = np.clip(((q1 - p2) @ d2) / a22 if a22 > 0 else 0.0, 0.0, 1.0)
    q2 = p2 + t * d2
    s = np.clip(((q2 - p1) @ d1) / a11 if a11 > 0 else 0.0, 0.0, 1.0)
    q1 = p1 + s * d1
    dist = float(np.linalg.norm(q1 - q2))
    return (q1 + q2) / 2.0, dist


def _axis_angle(a, b) -> float:
    c = abs(float(a.axis @ b.axis))
    return float(np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0))))
