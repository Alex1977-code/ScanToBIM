"""Epoch comparison (Verformungs-/Setzungsmessung).

Two scans of the same structure taken at different times are fine-registered
(trimmed point-to-plane ICP) and then compared point by point: every point
of the newer epoch gets its **signed displacement along the local surface
normal** of the reference epoch — settlement, bulging and deformation stand
out as coherent colored regions, while global rigid offsets are removed by
the registration.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from scantobim.core.cloud import PointCloud
from scantobim.core.deviation import _diverging_colors


def compare_epochs(
    reference: PointCloud,
    current: PointCloud,
    register: bool = True,
    tolerance: float = 0.005,
    max_points: int = 300_000,
    k_normals: int = 12,
    seed: int = 0,
) -> tuple[dict, PointCloud, np.ndarray]:
    """Compare ``current`` against ``reference``.

    Returns ``(stats, displacement_cloud, transform)`` — statistics, the
    newer epoch colored by signed displacement (blue = towards the
    reference surface / settlement, red = away / bulging) and the 4×4
    fine-registration transform that was applied.
    """
    from scantobim.core.preprocess import estimate_normals
    from scantobim.core.registration import apply_transform, register_point_to_plane

    transform = np.eye(4)
    if register:
        result = register_point_to_plane(current, reference)
        current = apply_transform(current, result.transform)
        transform = result.transform

    rng = np.random.default_rng(seed)

    def subsample(pts: np.ndarray) -> np.ndarray:
        if len(pts) <= max_points:
            return pts
        return pts[rng.choice(len(pts), size=max_points, replace=False)]

    ref_pts = subsample(np.asarray(reference.points, dtype=np.float64))
    cur_pts = subsample(np.asarray(current.points, dtype=np.float64))

    ref_cloud = PointCloud(points=ref_pts)
    estimate_normals(ref_cloud, k_neighbors=k_normals)
    tree = cKDTree(ref_pts)
    _, nearest = tree.query(cur_pts, k=1, workers=-1)
    normals = ref_cloud.normals[nearest]
    disp = np.einsum("ij,ij->i", cur_pts - ref_pts[nearest], normals)

    abs_d = np.abs(disp)
    p95 = float(np.quantile(abs_d, 0.95))
    edges = np.array([0, 1, 2, 5, 10, 20, 50, 100, np.inf]) * 1e-3
    hist, _ = np.histogram(abs_d, bins=edges)
    stats = {
        "points": int(len(cur_pts)),
        "registered": bool(register),
        "registration_shift": round(float(np.linalg.norm(transform[:3, 3])), 5),
        "tolerance": tolerance,
        "within_tolerance": round(float((abs_d <= tolerance).mean()), 4),
        "rms": round(float(np.sqrt((disp**2).mean())), 6),
        "p95": round(p95, 6),
        "max": round(float(abs_d.max()), 6),
        "histogram_mm": {
            f"{edges[i] * 1000:g}-{edges[i + 1] * 1000:g}": int(hist[i])
            for i in range(len(hist))
        },
    }
    colors = _diverging_colors(disp, scale=max(p95, tolerance))
    cloud = PointCloud(points=cur_pts, colors=colors, source="epoch comparison")
    return stats, cloud, transform
