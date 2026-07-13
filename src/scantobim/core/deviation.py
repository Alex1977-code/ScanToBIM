"""Soll-Ist-Abweichungsanalyse (as-built QA).

Measures the distance of every scan point to the reconstructed model —
the number an inspector needs to trust a model: does the clean-edged
geometry actually lie on the measured surfaces?

Output:

* **Statistics** — RMS, mean, 95 % quantile, maximum, share of points
  within a tolerance (default 5 mm), histogram — global numbers for the
  acceptance report.
* **Colored deviation cloud** — every scan point tinted on a diverging
  scale (blue = behind the surface, white = on it, red = in front), ready
  to inspect in any point cloud viewer.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from scantobim.core.cloud import PointCloud
from scantobim.core.mesh import Mesh


def _point_triangle_distance(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Distance of points ``p`` to triangles ``(a, b, c)`` (all (N, 3)).

    Returns ``(distance, closest_point)`` — Ericson's real-time collision
    detection region test, fully vectorized.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    closest = np.empty_like(p)
    done = np.zeros(len(p), dtype=bool)

    def assign(mask, pts):
        m = mask & ~done
        closest[m] = pts[m] if pts.shape == p.shape else pts
        done[m] = True

    assign((d1 <= 0) & (d2 <= 0), a)  # vertex A
    assign((d3 >= 0) & (d4 <= d3), b)  # vertex B
    assign((d6 >= 0) & (d5 <= d6), c)  # vertex C

    vc = d1 * d4 - d3 * d2
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~done  # edge AB
    v = np.divide(d1, d1 - d3, out=np.zeros_like(d1), where=(d1 - d3) != 0)
    closest[m] = a[m] + v[m, None] * ab[m]
    done |= m

    vb = d5 * d2 - d1 * d6
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~done  # edge AC
    w = np.divide(d2, d2 - d6, out=np.zeros_like(d2), where=(d2 - d6) != 0)
    closest[m] = a[m] + w[m, None] * ac[m]
    done |= m

    va = d3 * d6 - d5 * d4
    m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0) & ~done  # edge BC
    denom = (d4 - d3) + (d5 - d6)
    w = np.divide(d4 - d3, denom, out=np.zeros_like(denom), where=denom != 0)
    closest[m] = b[m] + w[m, None] * (c[m] - b[m])
    done |= m

    m = ~done  # interior
    denom = va + vb + vc
    v = np.divide(vb, denom, out=np.zeros_like(denom), where=denom != 0)
    w = np.divide(vc, denom, out=np.zeros_like(denom), where=denom != 0)
    closest[m] = a[m] + v[m, None] * ab[m] + w[m, None] * ac[m]

    return np.linalg.norm(p - closest, axis=1), closest


def signed_distance_to_mesh(
    points: np.ndarray, mesh: Mesh, k_candidates: int = 8
) -> np.ndarray:
    """Signed distance of every point to the nearest mesh triangle.

    Candidate triangles come from a centroid k-d tree; sign follows the
    winning triangle's normal (positive = in front of the surface).
    """
    tri = mesh.vertices[mesh.faces]  # (T, 3, 3)
    centroids = tri.mean(axis=1)
    tree = cKDTree(centroids)
    k = min(k_candidates, len(tri))
    _, cand = tree.query(points, k=k, workers=-1)
    if k == 1:
        cand = cand[:, None]

    best_d = np.full(len(points), np.inf)
    best_closest = np.zeros_like(points)
    best_tri = np.zeros(len(points), dtype=np.int64)
    for col in range(cand.shape[1]):
        idx = cand[:, col]
        d, closest = _point_triangle_distance(
            points, tri[idx, 0], tri[idx, 1], tri[idx, 2]
        )
        better = d < best_d
        best_d[better] = d[better]
        best_closest[better] = closest[better]
        best_tri[better] = idx[better]

    normals = mesh.face_normals()[best_tri]
    sign = np.sign(np.einsum("ij,ij->i", points - best_closest, normals))
    sign[sign == 0] = 1.0
    return best_d * sign


def _diverging_colors(dist: np.ndarray, scale: float) -> np.ndarray:
    """Blue (−scale) → white (0) → red (+scale), uint8 RGB."""
    t = np.clip(dist / max(scale, 1e-12), -1.0, 1.0)
    colors = np.full((len(dist), 3), 255.0)
    pos = t > 0  # towards red: reduce G, B
    colors[pos, 1] = 255 * (1 - t[pos])
    colors[pos, 2] = 255 * (1 - t[pos])
    neg = t < 0  # towards blue: reduce R, G
    colors[neg, 0] = 255 * (1 + t[neg])
    colors[neg, 1] = 255 * (1 + t[neg])
    return colors.astype(np.uint8)


def deviation_analysis(
    mesh: Mesh,
    cloud: PointCloud,
    tolerance: float = 0.005,
    max_points: int = 300_000,
    alignment: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[dict, PointCloud]:
    """Compare the scan against the reconstructed model.

    Returns ``(stats, deviation_cloud)`` — JSON-ready statistics plus the
    scan (subsampled to ``max_points``) colored by signed deviation.
    ``alignment`` is the 4×4 transform from the report when the model was
    axis-aligned (applied to the points before comparing).
    """
    pts = np.asarray(cloud.points, dtype=np.float64)
    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), size=max_points, replace=False)]
    if alignment is not None:
        alignment = np.asarray(alignment, dtype=np.float64)
        pts = pts @ alignment[:3, :3].T + alignment[:3, 3]

    dist = signed_distance_to_mesh(pts, mesh)
    abs_d = np.abs(dist)
    p95 = float(np.quantile(abs_d, 0.95))
    edges = np.array([0, 0.5, 1, 2, 5, 10, 20, 50, np.inf]) * 1e-3
    hist, _ = np.histogram(abs_d, bins=edges)
    stats = {
        "points": int(len(pts)),
        "tolerance": tolerance,
        "within_tolerance": round(float((abs_d <= tolerance).mean()), 4),
        "rms": round(float(np.sqrt((dist**2).mean())), 6),
        "mean_abs": round(float(abs_d.mean()), 6),
        "p95": round(p95, 6),
        "max": round(float(abs_d.max()), 6),
        "histogram_mm": {
            f"{edges[i] * 1000:g}-{edges[i + 1] * 1000:g}": int(hist[i])
            for i in range(len(hist))
        },
    }
    colors = _diverging_colors(dist, scale=max(p95, tolerance))
    return stats, PointCloud(points=pts, colors=colors, source="deviation")
