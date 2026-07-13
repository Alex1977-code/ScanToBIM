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
    # Square hollow sections SHS — EN 10219
    "SHS 40": (0.040, 0.040), "SHS 50": (0.050, 0.050), "SHS 60": (0.060, 0.060),
    "SHS 70": (0.070, 0.070), "SHS 80": (0.080, 0.080), "SHS 90": (0.090, 0.090),
    "SHS 100": (0.100, 0.100), "SHS 120": (0.120, 0.120), "SHS 140": (0.140, 0.140),
    "SHS 150": (0.150, 0.150), "SHS 160": (0.160, 0.160), "SHS 180": (0.180, 0.180),
    "SHS 200": (0.200, 0.200), "SHS 250": (0.250, 0.250),
    # Rectangular hollow sections RHS — EN 10219
    "RHS 50x30": (0.050, 0.030), "RHS 60x40": (0.060, 0.040),
    "RHS 80x40": (0.080, 0.040), "RHS 100x50": (0.100, 0.050),
    "RHS 100x60": (0.100, 0.060), "RHS 120x60": (0.120, 0.060),
    "RHS 120x80": (0.120, 0.080), "RHS 140x80": (0.140, 0.080),
    "RHS 160x80": (0.160, 0.080), "RHS 200x100": (0.200, 0.100),
    "RHS 200x120": (0.200, 0.120), "RHS 250x150": (0.250, 0.150),
    "RHS 300x200": (0.300, 0.200),
    # Equal angles L — EN 10056-1
    "L 40x40": (0.040, 0.040), "L 50x50": (0.050, 0.050), "L 60x60": (0.060, 0.060),
    "L 70x70": (0.070, 0.070), "L 80x80": (0.080, 0.080), "L 90x90": (0.090, 0.090),
    "L 100x100": (0.100, 0.100), "L 120x120": (0.120, 0.120),
    "L 150x150": (0.150, 0.150),
}

_FAMILY_PREFIXES = {
    "I/H": ("IPE", "HEA", "HEB"),
    "U": ("UPN",),
    "hollow": ("SHS", "RHS"),
    "L": ("L ",),
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
    section_u: np.ndarray | None = None  # 3D width axis of the cross section
    section_v: np.ndarray | None = None  # 3D height axis of the cross section
    section_center: np.ndarray | None = None  # bbox center of the section


def match_profile(
    height: float, width: float, family: str, tolerance: float = 0.08
) -> tuple[str, float]:
    """Nearest catalog profile for measured outer dimensions.

    Returns ``(designation, relative_deviation)`` or ``("unbekannt", inf)``
    when nothing lies within ``tolerance``.
    """
    prefixes = _FAMILY_PREFIXES.get(family, ())
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

    # Cross-section in the perpendicular plane. PCA orientation is degenerate
    # for square (SHS) and L sections, so align with the minimum-area
    # bounding rectangle instead (rotating-calipers style search).
    cross = centered - np.outer(t, axis)
    q_raw = np.column_stack([cross @ vecs[:, 1], cross @ vecs[:, 0]])
    q = _min_area_align(q_raw)
    # Put the larger extent on axis 1 ("height").
    ext0 = float(np.quantile(q[:, 0], 0.995) - np.quantile(q[:, 0], 0.005))
    ext1 = float(np.quantile(q[:, 1], 0.995) - np.quantile(q[:, 1], 0.005))
    if ext0 > ext1:
        q = q[:, ::-1]
        ext0, ext1 = ext1, ext0
    q = q - (np.quantile(q, 0.995, axis=0) + np.quantile(q, 0.005, axis=0)) / 2.0
    width, height = ext0, ext1

    family = _classify_family(q, height, width)
    profile, deviation = match_profile(height, width, family)

    # Recover the 3D frame of the aligned section: solve cross ≈ q·[u; v] + c.
    # All 2D steps were rotations/swaps, so the fit is exact up to noise.
    design = np.column_stack([q, np.ones(len(q))])
    basis, *_ = np.linalg.lstsq(design, cross, rcond=None)
    u3 = basis[0] / max(np.linalg.norm(basis[0]), 1e-12)
    v3 = basis[1] / max(np.linalg.norm(basis[1]), 1e-12)
    section_center = centroid + basis[2]
    # Canonical orientation for asymmetric sections (mirror-equivalent under
    # proper rotation, so only sign flips are needed): U web on -u,
    # L legs on -u/-v — the material-heavy side.
    if family in ("U", "L") and (q[:, 0] > 0).sum() > (q[:, 0] <= 0).sum():
        u3 = -u3
    if family == "L" and (q[:, 1] > 0).sum() > (q[:, 1] <= 0).sum():
        v3 = -v3
    if float(np.linalg.det(np.column_stack([u3, v3, axis]))) < 0:
        axis = -axis  # keep the frame right-handed; axis sign is free

    return SteelMember(
        profile=profile,
        family=family,
        height=height,
        width=width,
        length=length,
        centroid=centroid,
        axis=axis,
        deviation=deviation,
        section_u=u3,
        section_v=v3,
        section_center=section_center,
    )


def _min_area_align(q: np.ndarray, step_deg: float = 2.0) -> np.ndarray:
    """Rotate 2D points into the profile's natural orientation.

    Rolled sections consist of axis-parallel straight segments, so the right
    rotation is the one that puts the most material ON the bounding-box
    edges (minimum bbox area is ambiguous for L sections).
    """
    best_ang, best_score = 0.0, -np.inf
    sub = q[:: max(1, len(q) // 4000)]
    for ang in np.deg2rad(np.arange(0.0, 90.0, step_deg)):
        c, s = np.cos(ang), np.sin(ang)
        xr = sub[:, 0] * c - sub[:, 1] * s
        yr = sub[:, 0] * s + sub[:, 1] * c
        x_lo, x_hi = np.quantile(xr, 0.01), np.quantile(xr, 0.99)
        y_lo, y_hi = np.quantile(yr, 0.01), np.quantile(yr, 0.99)
        band_x = 0.08 * (x_hi - x_lo)
        band_y = 0.08 * (y_hi - y_lo)
        on_edge = (
            (xr < x_lo + band_x) | (xr > x_hi - band_x)
            | (yr < y_lo + band_y) | (yr > y_hi - band_y)
        )
        score = on_edge.mean()
        if score > best_score:
            best_score, best_ang = score, ang
    c, s = np.cos(best_ang), np.sin(best_ang)
    return q @ np.array([[c, s], [-s, c]])


def _classify_family(q: np.ndarray, height: float, width: float) -> str:
    """Cross-section family from the occupancy pattern.

    ``q[:, 1]`` runs along the profile height, ``q[:, 0]`` along the width.
    * I/H — flanges at both height edges, web centered in the width.
    * U — flanges at both height edges, web at one width edge.
    * hollow (RHS/SHS) — material along all four bounding edges, empty core.
    * L — material along exactly two adjacent edges.
    """
    band_h = 0.14 * height
    band_w = 0.14 * width
    top = q[:, 1] > 0.5 * height - band_h
    bottom = q[:, 1] < -0.5 * height + band_h
    left = q[:, 0] < -0.5 * width + band_w
    right = q[:, 0] > 0.5 * width - band_w
    n = len(q)
    f_top, f_bottom = top.mean(), bottom.mean()
    f_left, f_right = left.mean(), right.mean()

    edge_flags = [f > 0.08 for f in (f_top, f_bottom, f_left, f_right)]
    if all(edge_flags):
        # All four edges occupied: hollow section — unless a centered web
        # exists (an I-profile's flange tips also touch left/right bands).
        mid = np.abs(q[:, 1]) < 0.25 * height
        if mid.sum() >= 20:
            web = np.abs(q[mid, 0]) < 0.15 * width
            core = mid & (np.abs(q[:, 0]) < 0.5 * width - band_w)
            if web.mean() > 0.5:
                return "I/H"
            if core.sum() < 0.02 * n:
                return "hollow"
        else:
            return "hollow"

    if sum(edge_flags) == 2 and (
        (edge_flags[0] != edge_flags[1]) and (edge_flags[2] != edge_flags[3])
    ):
        return "L"  # one height edge + one width edge

    # Web-based I/H vs U decision.
    mid = np.abs(q[:, 1]) < 0.25 * height
    if mid.sum() < 20:
        return "unknown"
    web_offset = float(np.median(q[mid, 0]))
    web_spread = float(np.std(q[mid, 0]))
    if web_spread > 0.30 * width:
        return "unknown"
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
