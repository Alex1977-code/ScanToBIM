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
    neighborhoods = pts[idx]  # (N, k, 3)
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    # Covariance per point: (N, 3, 3)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k
    # Smallest eigenvector = normal. eigh returns ascending eigenvalues.
    _, vecs = np.linalg.eigh(cov)
    normals = vecs[:, :, 0]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms > 0)

    if orient_towards is not None:
        to_sensor = np.asarray(orient_towards, dtype=np.float64) - pts
        flip = np.einsum("ij,ij->i", normals, to_sensor) < 0
        normals[flip] *= -1

    cloud.normals = normals
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
