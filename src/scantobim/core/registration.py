"""Multi-scan registration: trimmed point-to-plane ICP.

Merges several scans of the same scene (different scanner positions, drone +
terrestrial, LiDAR + photogrammetry) into one cloud. Point-to-plane ICP
converges much faster than point-to-point on the man-made, plane-dominated
scenes this package targets; trimming makes it robust against partial
overlap. Scans must be *roughly* pre-aligned (as delivered by scanner
software or shared georeferencing) — ICP is a fine-registration method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from scantobim.core.cloud import PointCloud
from scantobim.core.preprocess import estimate_normals, voxel_downsample


@dataclass
class RegistrationResult:
    transform: np.ndarray  # 4x4, maps source coordinates into target frame
    rmse: float  # point-to-plane RMSE over kept (trimmed) pairs
    iterations: int
    converged: bool


def register_point_to_plane(
    source: PointCloud,
    target: PointCloud,
    max_iterations: int = 40,
    sample: int = 8000,
    trim_ratio: float = 0.8,
    max_pair_distance: float | None = None,
    tolerance: float = 1e-7,
    seed: int = 0,
) -> RegistrationResult:
    """Estimate the rigid transform aligning ``source`` onto ``target``.

    Parameters
    ----------
    sample:
        Number of source points used per iteration (random, fixed seed).
    trim_ratio:
        Fraction of best-matching pairs kept per iteration (robustness
        against non-overlapping regions).
    max_pair_distance:
        Hard correspondence cutoff; default is 20x the target's median
        point spacing.
    """
    if target.normals is None:
        estimate_normals(target, k_neighbors=16)
    tgt_pts = target.points
    tgt_nrm = target.normals
    tree = cKDTree(tgt_pts)

    if max_pair_distance is None:
        d, _ = tree.query(
            tgt_pts[np.random.default_rng(0).choice(len(tgt_pts), min(1000, len(tgt_pts)), replace=False)],
            k=2, workers=-1,
        )
        max_pair_distance = 20.0 * float(np.median(d[:, 1]))

    rng = np.random.default_rng(seed)
    src_idx = rng.choice(len(source), size=min(sample, len(source)), replace=False)
    src = source.points[src_idx]

    transform = np.eye(4)
    rmse = np.inf
    converged = False
    it = 0
    for it in range(1, max_iterations + 1):
        moved = src @ transform[:3, :3].T + transform[:3, 3]
        dist, nn = tree.query(moved, k=1, workers=-1)
        keep = dist < max_pair_distance
        if keep.sum() < 50:
            break
        # Trim to the best pairs.
        cutoff = np.quantile(dist[keep], trim_ratio)
        keep &= dist <= cutoff

        p = moved[keep]
        q = tgt_pts[nn[keep]]
        n = tgt_nrm[nn[keep]]

        # Linearized point-to-plane system:  [p×n, n] · [ω, t] = -(p-q)·n
        a = np.hstack([np.cross(p, n), n])
        b = -np.einsum("ij,ij->i", p - q, n)
        x, *_ = np.linalg.lstsq(a, b, rcond=None)
        omega, t = x[:3], x[3:]

        delta = np.eye(4)
        delta[:3, :3] = _rodrigues(omega)
        delta[:3, 3] = t
        transform = delta @ transform

        new_rmse = float(np.sqrt(np.mean(b**2)))
        if abs(rmse - new_rmse) < tolerance and np.linalg.norm(x) < 1e-6:
            rmse = new_rmse
            converged = True
            break
        rmse = new_rmse

    return RegistrationResult(
        transform=transform, rmse=rmse, iterations=it, converged=converged
    )


def merge_clouds(clouds: list[PointCloud], voxel_size: float = 0.0) -> PointCloud:
    """Concatenate registered clouds; optional voxel thinning of the result."""
    points = np.vstack([c.points for c in clouds])
    colors = None
    if all(c.colors is not None for c in clouds):
        colors = np.vstack([c.colors for c in clouds])
    intensity = None
    if all(c.intensity is not None for c in clouds):
        intensity = np.concatenate([c.intensity for c in clouds])
    merged = PointCloud(points=points, colors=colors, intensity=intensity, source="merged")
    if voxel_size > 0:
        merged = voxel_downsample(merged, voxel_size)
    return merged


def apply_transform(cloud: PointCloud, transform: np.ndarray) -> PointCloud:
    """Return a new cloud with ``transform`` applied to the points/normals."""
    rot = transform[:3, :3]
    return PointCloud(
        points=cloud.points @ rot.T + transform[:3, 3],
        colors=cloud.colors,
        normals=None if cloud.normals is None else cloud.normals @ rot.T,
        intensity=cloud.intensity,
        source=cloud.source,
    )


def _rodrigues(omega: np.ndarray) -> np.ndarray:
    """Rotation matrix from an axis-angle vector (exponential map)."""
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3)
    k = omega / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)
