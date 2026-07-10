"""Steel structure recognition: match scanned members against profile catalogs.

Elongated point clusters are measured (member axis, cross-section height and
width, web position) and matched against the European standard hot-rolled
profile series — IPE, HEA, HEB (I/H sections) and UPN (channels). The result
is what a structural engineer needs from an as-built survey: profile
designation, length, axis and deviation from catalog dimensions.

All catalog dimensions in meters (DIN 1025 / EN 10365).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scantobim.core.cloud import PointCloud

# (h, b) per designation, meters
CATALOG: dict[str, tuple[float, float]] = {
    # IPE — DIN 1025-5
    "IPE 80": (0.080, 0.046), "IPE 100": (0.100, 0.055), "IPE 120": (0.120, 0.064),
    "IPE 140": (0.140, 0.073), "IPE 160": (0.160, 0.082), "IPE 180": (0.180, 0.091),
    "IPE 200": (0.200, 0.100), "IPE 220": (0.220, 0.110), "IPE 240": (0.240, 0.120),
    "IPE 270": (0.270, 0.135), "IPE 300": (0.300, 0.150), "IPE 330": (0.330, 0.160),
    "IPE 360": (0.360, 0.170), "IPE 400": (0.400, 0.180), "IPE 450": (0.450, 0.190),
    "IPE 500": (0.500, 0.200), "IPE 550": (0.550, 0.210), "IPE 600": (0.600, 0.220),
    # HEA — DIN 1025-3
    "HEA 100": (0.096, 0.100), "HEA 120": (0.114, 0.120), "HEA 140": (0.133, 0.140),
    "HEA 160": (0.152, 0.160), "HEA 180": (0.171, 0.180), "HEA 200": (0.190, 0.200),
    "HEA 220": (0.210, 0.220), "HEA 240": (0.230, 0.240), "HEA 260": (0.250, 0.260),
    "HEA 280": (0.270, 0.280), "HEA 300": (0.290, 0.300), "HEA 320": (0.310, 0.300),
    "HEA 340": (0.330, 0.300), "HEA 360": (0.350, 0.300), "HEA 400": (0.390, 0.300),
    "HEA 450": (0.440, 0.300), "HEA 500": (0.490, 0.300), "HEA 600": (0.590, 0.300),
    # HEB — DIN 1025-2
    "HEB 100": (0.100, 0.100), "HEB 120": (0.120, 0.120), "HEB 140": (0.140, 0.140),
    "HEB 160": (0.160, 0.160), "HEB 180": (0.180, 0.180), "HEB 200": (0.200, 0.200),
    "HEB 220": (0.220, 0.220), "HEB 240": (0.240, 0.240), "HEB 260": (0.260, 0.260),
    "HEB 280": (0.280, 0.280), "HEB 300": (0.300, 0.300), "HEB 320": (0.320, 0.300),
    "HEB 340": (0.340, 0.300), "HEB 360": (0.360, 0.300), "HEB 400": (0.400, 0.300),
    "HEB 450": (0.450, 0.300), "HEB 500": (0.500, 0.300), "HEB 600": (0.600, 0.300),
    # UPN — DIN 1026-1
    "UPN 80": (0.080, 0.045), "UPN 100": (0.100, 0.050), "UPN 120": (0.120, 0.055),
    "UPN 140": (0.140, 0.060), "UPN 160": (0.160, 0.065), "UPN 180": (0.180, 0.070),
    "UPN 200": (0.200, 0.075), "UPN 220": (0.220, 0.080), "UPN 240": (0.240, 0.085),
    "UPN 260": (0.260, 0.090), "UPN 280": (0.280, 0.095), "UPN 300": (0.300, 0.100),
}


@dataclass
class SteelMember:
    profile: str  # catalog designation or "unbekannt"
    family: str  # "I/H" | "U" | "unknown"
    height: float
    width: float
    length: float
    centroid: np.ndarray
    axis: np.ndarray
    deviation: float  # max relative deviation from catalog h/b


def match_profile(
    height: float, width: float, family: str, tolerance: float = 0.08
) -> tuple[str, float]:
    """Nearest catalog profile for measured outer dimensions.

    Returns ``(designation, relative_deviation)`` or ``("unbekannt", inf)``
    when nothing lies within ``tolerance``.
    """
    prefixes = ("IPE", "HEA", "HEB") if family == "I/H" else ("UPN",) if family == "U" else ()
    best_name, best_dev = "unbekannt", float("inf")
    for name, (h, b) in CATALOG.items():
        if prefixes and not name.startswith(prefixes):
            continue
        dev = max(abs(height - h) / h, abs(width - b) / b)
        if dev < best_dev:
            best_name, best_dev = name, dev
    if best_dev > tolerance:
        return "unbekannt", best_dev
    return best_name, best_dev


def measure_member(points: np.ndarray) -> SteelMember | None:
    """Measure one elongated cluster: axis, cross-section, profile family."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 400:
        return None
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    vals, vecs = np.linalg.eigh(centered.T @ centered / len(pts))
    axis = vecs[:, 2]  # longest direction
    if vals[2] < 9 * vals[1]:  # elongation ratio (std) < 3 — not a member
        return None

    t = centered @ axis
    length = float(np.quantile(t, 0.995) - np.quantile(t, 0.005))

    # Cross-section in the perpendicular plane, PCA-aligned.
    cross = centered - np.outer(t, axis)
    q = np.column_stack([cross @ vecs[:, 1], cross @ vecs[:, 0]])
    cvals, cvecs = np.linalg.eigh(q.T @ q / len(q))
    q = q @ cvecs  # columns: minor, major
    height = float(np.quantile(q[:, 1], 0.995) - np.quantile(q[:, 1], 0.005))
    width = float(np.quantile(q[:, 0], 0.995) - np.quantile(q[:, 0], 0.005))

    family = _classify_family(q, height, width)
    profile, deviation = match_profile(height, width, family)
    return SteelMember(
        profile=profile,
        family=family,
        height=height,
        width=width,
        length=length,
        centroid=centroid,
        axis=axis,
        deviation=deviation,
    )


def _classify_family(q: np.ndarray, height: float, width: float) -> str:
    """I/H vs U from the web position in the normalized cross-section.

    ``q[:, 1]`` runs along the profile height, ``q[:, 0]`` along the width.
    Points in the mid-height band belong to the web; a centered web means an
    I/H section, a web at one width edge means a channel (U).
    """
    mid = np.abs(q[:, 1]) < 0.25 * height
    if mid.sum() < 20:
        return "unknown"
    web_offset = float(np.median(q[mid, 0]))
    web_spread = float(np.std(q[mid, 0]))
    if web_spread > 0.30 * width:
        return "unknown"  # no concentrated web (e.g. hollow section)
    if abs(web_offset) < 0.15 * width:
        return "I/H"
    if abs(web_offset) > 0.30 * width:
        return "U"
    return "unknown"


def detect_steel_members(cloud: PointCloud) -> list[SteelMember]:
    """Find and measure steel members in a structural scan."""
    from scantobim.core.machinery import _cluster_points
    from scantobim.core.preprocess import (
        estimate_point_spacing,
        remove_statistical_outliers,
        voxel_downsample,
    )

    raw_spacing = estimate_point_spacing(cloud)
    work = voxel_downsample(cloud, 1.5 * raw_spacing)
    work, _ = remove_statistical_outliers(work)
    spacing = estimate_point_spacing(work)

    clusters = _cluster_points(work.points, radius=5.0 * spacing)
    members = []
    for cl in clusters:
        if len(cl) < 400:
            continue
        member = measure_member(work.points[cl])
        if member is not None:
            members.append(member)
    return members


def steel_report(members: list[SteelMember]) -> list[dict]:
    return [
        {
            "profile": m.profile,
            "family": m.family,
            "height": round(m.height, 4),
            "width": round(m.width, 4),
            "length": round(m.length, 4),
            "centroid": [round(float(x), 4) for x in m.centroid],
            "axis": [round(float(x), 4) for x in m.axis],
            "catalog_deviation": round(m.deviation, 4) if np.isfinite(m.deviation) else None,
        }
        for m in members
    ]
