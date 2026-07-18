"""Point cloud preprocessing: thinning, denoising, normal estimation."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from scantobim.core.cloud import PointCloud


def voxel_downsample(cloud: PointCloud, voxel_size: float) -> PointCloud:
    """Thin the cloud to (at most) one representative point per voxel.

    The point closest to the voxel centroid is kept, so attributes (color,
    intensity) stay authentic instead of being averaged across surfaces.
    """
    if voxel_size <= 0:
        return cloud
    pts = cloud.points
    mins = pts.min(axis=0)
    keys = np.floor((pts - mins) / voxel_size).astype(np.int64)
    # Unique voxel id per point.
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_sorted = keys[order]
    new_voxel = np.ones(len(pts), dtype=bool)
    new_voxel[1:] = np.any(keys_sorted[1:] != keys_sorted[:-1], axis=1)
    group_ids = np.cumsum(new_voxel) - 1
    n_groups = group_ids[-1] + 1 if len(group_ids) else 0

    # Centroid per voxel, then pick the sorted point nearest to it.
    sums = np.zeros((n_groups, 3))
    np.add.at(sums, group_ids, pts[order])
    counts = np.bincount(group_ids, minlength=n_groups)
    centroids = sums / counts[:, None]
    d2 = np.einsum("ij,ij->i", pts[order] - centroids[group_ids], pts[order] - centroids[group_ids])

    # argmin of d2 within each group
    best_d = np.full(n_groups, np.inf)
    np.minimum.at(best_d, group_ids, d2)
    is_best = d2 <= best_d[group_ids]
    # first "best" point per group wins
    idx_sorted = np.arange(len(pts))
    candidates = idx_sorted[is_best]
    cand_groups = group_ids[is_best]
    first_of_group = np.ones(len(candidates), dtype=bool)
    first_of_group[1:] = cand_groups[1:] != cand_groups[:-1]
    best = candidates[first_of_group]

    keep = order[best]
    keep.sort()
    return cloud.select(keep)


def remove_statistical_outliers(
    cloud: PointCloud, k_neighbors: int = 16, std_ratio: float = 2.5
) -> tuple[PointCloud, np.ndarray]:
    """Classic statistical outlier removal (SOR).

    Points whose mean distance to their ``k`` nearest neighbours exceeds
    ``mean + std_ratio * std`` of the whole cloud are dropped.
    Returns the filtered cloud and the boolean keep-mask.
    """
    pts = cloud.points
    n = len(pts)
    if n <= k_neighbors + 1:
        return cloud, np.ones(n, dtype=bool)
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k_neighbors + 1, workers=-1)
    mean_d = dists[:, 1:].mean(axis=1)
    mu, sigma = mean_d.mean(), mean_d.std()
    keep = mean_d <= mu + std_ratio * sigma
    return cloud.select(keep), keep


def remove_edge_artifacts(
    cloud: PointCloud,
    k_neighbors: int = 12,
    linearity_threshold: float = 0.95,
    sparsity_ratio: float = 2.5,
    require_sparse: bool = True,
) -> tuple[PointCloud, np.ndarray]:
    """Remove mixed-pixel edge artifacts (Kantenartefakte).

    Laser scanners produce phantom points along silhouette edges where the
    beam hits two surfaces at once: thin, *sparse* strings of points hanging
    between foreground and background. Those strings are exactly where clean
    edges are reconstructed, so they must go.

    A point is dropped when its neighbourhood is strongly LINEAR
    (``(λ₁-λ₂)/λ₁ > linearity_threshold``) and — with ``require_sparse`` —
    also SPARSE (mean neighbour distance > ``sparsity_ratio`` x the cloud
    median). Real surface points are planar (λ₁≈λ₂), crease points see both
    faces (planar too). ``require_sparse=True`` protects dense thin
    structures such as bridge cables; for building reconstruction, where no
    legitimate string-like geometry exists, ``require_sparse=False`` also
    removes dense mixed-pixel strings that survive the outlier filter.

    Returns the filtered cloud and the boolean keep-mask.
    """
    pts = cloud.points
    n = len(pts)
    k = min(k_neighbors + 1, n)
    if n < k_neighbors * 3:
        return cloud, np.ones(n, dtype=bool)
    tree = cKDTree(pts)
    dists, idx = tree.query(pts, k=k, workers=-1)
    mean_d = dists[:, 1:].mean(axis=1)
    median_d = float(np.median(mean_d))

    if require_sparse:
        candidates = mean_d > sparsity_ratio * median_d
    else:
        candidates = np.ones(n, dtype=bool)
    keep = np.ones(n, dtype=bool)
    check = np.flatnonzero(candidates)
    if len(check):
        neigh = pts[idx[check]]
        centered = neigh - neigh.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centered, centered)
        eig = np.linalg.eigvalsh(cov)  # ascending
        lam1 = eig[:, 2]
        lam2 = eig[:, 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            linearity = np.where(lam1 > 0, (lam1 - lam2) / lam1, 0.0)
        keep[check[linearity > linearity_threshold]] = False
    return cloud.select(keep), keep


def local_point_spacing(points: np.ndarray, k: int = 8) -> np.ndarray:
    """Per-point local spacing (mean kNN distance) — the density map that
    lets downstream stages adapt to near/far scanner resolution."""
    n = len(points)
    k = min(k + 1, n)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k, workers=-1)
    return dists[:, 1:].mean(axis=1)


def estimate_normals(
    cloud: PointCloud,
    k_neighbors: int = 16,
    orient_towards: np.ndarray | None = None,
) -> PointCloud:
    """Estimate unit normals via PCA on the k-nearest-neighbour covariance.

    If ``orient_towards`` (a sensor / viewpoint position) is given, normals
    are flipped to face it. Otherwise the sign is left ambiguous — the
    plane-detection stage treats normals sign-invariantly, so this is fine
    for reconstruction.
    """
    pts = cloud.points
    n = len(pts)
    k = min(k_neighbors + 1, n)
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k, workers=-1)
    from scantobim.core.accel import pca_normals

    normals = pca_normals(pts, idx)  # GPU when available and worthwhile
    if normals is None:
        neighborhoods = pts[idx]  # (N, k, 3)
        centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        # Covariance per point: (N, 3, 3)
        cov = np.einsum("nki,nkj->nij", centered, centered) / k
        # Smallest eigenvector = normal. eigh returns ascending eigenvalues.
        _, vecs = np.linalg.eigh(cov)
        normals = vecs[:, :, 0]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(
            normals, norms, out=np.zeros_like(normals), where=norms > 0
        )

    if orient_towards is not None:
        to_sensor = np.asarray(orient_towards, dtype=np.float64) - pts
        flip = np.einsum("ij,ij->i", normals, to_sensor) < 0
        normals[flip] *= -1

    cloud.normals = normals
    return cloud


def _sensor_positions_by_time(
    point_times: np.ndarray, traj_xyz: np.ndarray, traj_t: np.ndarray
) -> np.ndarray | None:
    """Scanner position at each point's capture time (clock-aligned interp).

    LAS ``gps_time`` and the trajectory timestamps usually share a clock;
    when they differ by a constant offset (adjusted GPS time vs. system
    time), both are aligned at their start as long as the recording
    durations agree. Returns None when the clocks cannot be reconciled —
    the caller falls back to nearest-in-space orientation.
    """
    order = np.argsort(traj_t)
    traj_t = traj_t[order]
    traj_xyz = traj_xyz[order]
    span_p = float(point_times.max() - point_times.min())
    span_t = float(traj_t[-1] - traj_t[0])
    if span_t <= 0 or span_p <= 0:
        return None
    overlap = min(point_times.max(), traj_t[-1]) - max(point_times.min(), traj_t[0])
    t = point_times
    if overlap < 0.5 * min(span_p, span_t):
        # Constant clock offset: align both at their start — but only if
        # the durations roughly agree (same recording).
        if abs(span_p - span_t) > 0.25 * max(span_p, span_t):
            return None
        t = point_times - point_times.min() + traj_t[0]
    return np.column_stack([
        np.interp(t, traj_t, traj_xyz[:, 0]),
        np.interp(t, traj_t, traj_xyz[:, 1]),
        np.interp(t, traj_t, traj_xyz[:, 2]),
    ])


def orient_normals_along_trajectory(
    cloud: PointCloud, trajectory: np.ndarray, stats_out: dict | None = None
) -> PointCloud:
    """Flip normals to face the scanner position that saw each point.

    SLAM scanners (handheld/mobile mapping) record their path; every surface
    was seen FROM that path. With per-point capture times (LAS ``gps_time``)
    and a timestamped trajectory, each point faces the scanner position at
    its EXACT moment of capture — robust where the path passes a facade
    twice. Without times: nearest trajectory position in space, as before.
    """
    if cloud.normals is None:
        raise ValueError("estimate normals before orienting them")
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2:
        traj = traj.reshape(-1, 3)
    if len(traj) == 0:
        return cloud
    positions = traj[:, :3]

    sensor = None
    if traj.shape[1] >= 4 and cloud.times is not None and len(traj) > 1:
        sensor = _sensor_positions_by_time(cloud.times, positions, traj[:, 3])
    mode = "zeitbasiert" if sensor is not None else "raeumlich"
    if sensor is None:
        tree = cKDTree(positions)
        _, nearest = tree.query(cloud.points, k=1, workers=-1)
        sensor = positions[nearest]
    to_sensor = sensor - cloud.points
    flip = np.einsum("ij,ij->i", cloud.normals, to_sensor) < 0
    cloud.normals[flip] *= -1
    if stats_out is not None:
        stats_out["mode"] = mode
    return cloud


def estimate_point_spacing(cloud: PointCloud, sample: int = 2000) -> float:
    """Median nearest-neighbour distance, estimated on a random subsample.

    Used to derive sensible defaults for RANSAC thresholds, alpha-shape radius
    and edge-snapping distances from the actual scan resolution.
    """
    pts = cloud.points
    n = len(pts)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(0)
    probe = pts[rng.choice(n, size=min(sample, n), replace=False)]
    tree = cKDTree(pts)
    dists, _ = tree.query(probe, k=2, workers=-1)
    return float(np.median(dists[:, 1]))


def transfer_colors(
    cloud: PointCloud, source: PointCloud, max_dist: float | None = None
) -> float:
    """Colorize ``cloud`` in place from the nearest neighbours of ``source``.

    SLAM exports often pair a dense uncolorized cloud with a sparser
    colorized one — this grafts the RGB of the colored cloud onto the dense
    geometry. Points farther than ``max_dist`` (default: 4× the source's
    point spacing) from any source point stay neutral grey, so texture
    coverage stays honest. Returns the colored fraction (0..1).
    """
    if source.colors is None or len(source) == 0 or len(cloud) == 0:
        return 0.0
    if max_dist is None:
        max_dist = 4.0 * estimate_point_spacing(source)
        if max_dist <= 0:
            max_dist = np.inf
    tree = cKDTree(source.points)
    colors = np.full((len(cloud), 3), 128, dtype=np.uint8)
    hits = 0
    chunk = 2_000_000
    for start in range(0, len(cloud), chunk):
        pts = cloud.points[start:start + chunk]
        dist, idx = tree.query(pts, k=1, workers=-1)
        ok = dist <= max_dist
        colors[start:start + chunk][ok] = source.colors[idx[ok]]
        hits += int(ok.sum())
    cloud.colors = colors
    return hits / len(cloud)
