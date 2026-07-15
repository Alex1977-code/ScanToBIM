"""Robust multi-plane extraction.

The detector follows the efficient-RANSAC school (Schnabel et al. 2007) with
two refinements that matter for clean models of built environments:

* **Normal-aware scoring** — a candidate plane only collects inliers whose
  estimated surface normal agrees with the plane normal. This prevents the
  classic RANSAC failure of a "diagonal" plane slicing through two walls.
* **Connected-component filtering** — inliers must form one spatially
  connected patch. Two coplanar but distant surfaces (e.g. window sills on
  opposite walls) become two separate planes instead of one, which keeps
  boundary polygons simple and edges clean.

Each accepted plane is refit with total least squares (SVD) on its inliers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass
class Plane:
    """An oriented plane with its supporting inlier points."""

    normal: np.ndarray  # unit (3,)
    d: float  # plane eq: normal . x + d = 0
    inliers: np.ndarray  # int indices into the source cloud
    rms: float = 0.0
    basis: np.ndarray = field(default=None, repr=False)  # (2, 3) in-plane axes

    @property
    def center(self) -> np.ndarray:
        return -self.d * self.normal

    def distance(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.d

    def project_to_2d(self, points: np.ndarray) -> np.ndarray:
        """Project 3D points into the plane's 2D coordinate frame."""
        rel = points - self.origin_point()
        return rel @ self.basis.T

    def lift_to_3d(self, uv: np.ndarray) -> np.ndarray:
        """Map 2D in-plane coordinates back to 3D (on the plane)."""
        return self.origin_point() + uv @ self.basis

    def origin_point(self) -> np.ndarray:
        return -self.d * self.normal

    def make_basis(self) -> None:
        """Build a stable orthonormal in-plane basis (u, v)."""
        n = self.normal
        helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, helper)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        self.basis = np.vstack([u, v])


def fit_plane_tls(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Total-least-squares plane fit. Returns ``(normal, d, rms)``."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    # Smallest right-singular vector of the centered points.
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    d = -float(normal @ centroid)
    residuals = centered @ normal
    rms = float(np.sqrt(np.mean(residuals**2)))
    return normal, d, rms


def detect_planes(
    points: np.ndarray,
    normals: np.ndarray,
    distance_threshold: float,
    normal_threshold_deg: float = 25.0,
    min_inliers: int = 200,
    max_planes: int = 64,
    connectivity_radius: float | None = None,
    ransac_iterations: int = 600,
    seed: int = 7,
    max_rms_ratio: float = 0.5,
    core_fraction_min: float = 0.45,
    surface_variation_max: float = 0.01,
) -> tuple[list[Plane], np.ndarray]:
    """Sequentially extract planes; returns ``(planes, unassigned_mask)``.

    ``connectivity_radius`` (the neighbour radius of the connected-component
    filter) defaults to ``max(4 * distance_threshold, 3 * point spacing)`` so
    it adapts to the scan resolution.

    ``max_rms_ratio`` is a quality gate: a candidate whose inlier RMS exceeds
    ``max_rms_ratio * distance_threshold`` is a diffuse slab of noise rather
    than a surface and gets rejected (a real surface concentrates its inliers
    near the plane; uniform noise fills the whole threshold band with an RMS
    of ~0.58x the threshold). Keep ``distance_threshold`` at >= 2x the sensor
    noise sigma so genuine surfaces stay clearly below the gate.
    """
    rng = np.random.default_rng(seed)
    n_total = len(points)
    if connectivity_radius is None:
        spacing = _median_spacing(points, rng)
        connectivity_radius = max(4.0 * distance_threshold, 3.0 * spacing)
    cos_tol = np.cos(np.deg2rad(normal_threshold_deg))

    remaining = np.arange(n_total)
    planes: list[Plane] = []
    consecutive_rejects = 0

    while (
        len(remaining) >= min_inliers
        and len(planes) < max_planes
        and consecutive_rejects < 8
    ):
        pts = points[remaining]
        nrm = normals[remaining]

        best_mask: np.ndarray | None = None
        best_count = 0
        # Adaptive iteration count via the standard RANSAC stopping criterion.
        needed = ransac_iterations
        it = 0
        while it < needed:
            it += 1
            i = rng.integers(len(pts))
            # Hypothesis from one point + its normal (needs only 1 sample and
            # is far more stable on noisy data than 3-point hypotheses).
            n0 = nrm[i]
            p0 = pts[i]
            d0 = -float(n0 @ p0)
            dist = np.abs(pts @ n0 + d0)
            align = np.abs(nrm @ n0)
            mask = (dist < distance_threshold) & (align > cos_tol)
            count = int(mask.sum())
            if count > best_count:
                best_count = count
                best_mask = mask
                # Update adaptive stopping: probability of hitting an inlier.
                w = count / len(pts)
                if w > 0:
                    est = np.log(0.001) / np.log(max(1e-12, 1.0 - w))
                    needed = min(ransac_iterations, max(32, int(est) + 1))

        if best_mask is None or best_count < min_inliers:
            break

        # Refine: TLS refit, then re-collect inliers once.
        cand = remaining[best_mask]
        normal, d, _ = fit_plane_tls(points[cand])
        dist = np.abs(points[remaining] @ normal + d)
        align = np.abs(normals[remaining] @ normal)
        mask = (dist < distance_threshold) & (align > cos_tol)
        cand = remaining[mask]
        if len(cand) < min_inliers:
            remaining = remaining[~best_mask]
            continue

        # Largest connected component of the candidate inliers.
        component = _largest_component(points[cand], connectivity_radius)
        if len(component) < min_inliers:
            # Fragmented candidate: none of its pieces can support a plane of
            # this orientation — discard them all so the search progresses.
            remaining = np.setdiff1d(remaining, cand, assume_unique=True)
            continue
        cand = cand[component]

        normal, d, rms = fit_plane_tls(points[cand])
        # Three noise gates. A real surface concentrates its inliers near the
        # plane (high core fraction, low RMS) and is much thinner than it is
        # wide (low surface variation); a slab of random points fills the
        # whole threshold band (core fraction ~1/3, RMS ~0.58t) and stays
        # proportionally thick no matter the scale.
        core = np.abs(points[cand] @ normal + d) < distance_threshold / 3.0
        if (
            rms > max_rms_ratio * distance_threshold
            or core.mean() < core_fraction_min
            or _surface_variation(points[cand]) > surface_variation_max
        ):
            # Not a surface. Do not consume the points — they may still
            # support a better-oriented plane later.
            consecutive_rejects += 1
            continue
        consecutive_rejects = 0
        plane = Plane(normal=normal, d=d, inliers=cand, rms=rms)
        plane.make_basis()
        planes.append(plane)
        remaining = np.setdiff1d(remaining, cand, assume_unique=True)

    unassigned = np.zeros(n_total, dtype=bool)
    unassigned[remaining] = True
    return planes, unassigned


def _surface_variation(points: np.ndarray) -> float:
    """Pauly's surface variation: λ₃ / (λ₁ + λ₂ + λ₃) of the point scatter.

    Scale-free thickness measure — near zero for sampled surfaces (even rough
    ones), ~0.03+ for diffuse volumetric point slabs.
    """
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered / len(points)
    eig = np.linalg.eigvalsh(cov)
    total = float(eig.sum())
    return float(eig[0] / total) if total > 0 else 0.0


def _median_spacing(points: np.ndarray, rng: np.random.Generator, sample: int = 1500) -> float:
    """Median nearest-neighbour distance on a random subsample."""
    n = len(points)
    if n < 2:
        return 0.0
    probe = points[rng.choice(n, size=min(sample, n), replace=False)]
    tree = cKDTree(points)
    dists, _ = tree.query(probe, k=2, workers=-1)
    return float(np.median(dists[:, 1]))


def _largest_component(pts: np.ndarray, radius: float) -> np.ndarray:
    """Indices (into ``pts``) of the largest radius-connected component."""
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(1)  # everything isolated; keep a single point
    graph = sparse.csr_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    )
    n_comp, labels = connected_components(graph, directed=False)
    if n_comp == 1:
        return np.arange(n)
    counts = np.bincount(labels)
    return np.flatnonzero(labels == counts.argmax())


def plane_adjacency(
    planes: list[Plane],
    points: np.ndarray,
    contact_radius: float,
    min_contacts: int = 8,
) -> set[tuple[int, int]]:
    """Find pairs of planes whose inlier sets touch.

    Two planes are adjacent if at least ``min_contacts`` inlier points of one
    lie within ``contact_radius`` of the other's inliers — these pairs share a
    physical edge in the scanned scene.
    """
    trees = [cKDTree(points[p.inliers]) for p in planes]
    adjacent: set[tuple[int, int]] = set()
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            # Skip near-parallel planes — they cannot form a real edge.
            cos_angle = abs(float(planes[i].normal @ planes[j].normal))
            if cos_angle > 0.97:
                continue
            contacts = trees[i].query_ball_tree(trees[j], r=contact_radius)
            n_contact = sum(1 for lst in contacts if lst)
            if n_contact >= min_contacts:
                adjacent.add((i, j))
    return adjacent
