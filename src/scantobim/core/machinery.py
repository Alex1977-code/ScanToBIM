"""Industrial reverse engineering: cylinders, shafts, gears, gear trains.

Detects rotational machine geometry in point clouds:

* **Cylinders** via normal-based RANSAC (two oriented points define axis,
  center and radius) with least-squares refinement — pipes, shafts journals,
  bores, columns.
* **Stepped shafts** (Wellen): coaxial cylinder chains ordered along the
  axis, reported as steps with diameter/length — the numbers a machinist
  needs for re-manufacturing.
* **Spur gears** (Stirnräder): tooth count via FFT of the angular radius
  profile, tip/root circle, DIN module ``m = da / (z + 2)``, face width.
* **Gear stages** (Getriebestufen): meshing gear pairs found from parallel
  axes whose distance matches the sum of pitch radii — with transmission
  ratio, e.g. for documenting legacy Großgetriebe.

All lengths are in the unit of the input cloud (meters recommended).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from scantobim.core.cloud import PointCloud
from scantobim.core.mesh import Mesh
from scantobim.core.planes import _largest_component, _median_spacing


@dataclass
class Cylinder:
    center: np.ndarray  # point on axis (midpoint of extent)
    axis: np.ndarray  # unit vector
    radius: float
    length: float
    inliers: np.ndarray
    rms: float

    def axial_range(self, points: np.ndarray) -> tuple[float, float]:
        t = (points[self.inliers] - self.center) @ self.axis
        return float(t.min()), float(t.max())


@dataclass
class ShaftStep:
    diameter: float
    length: float
    start: float  # axial position along the shaft axis
    end: float


@dataclass
class Shaft:
    axis: np.ndarray
    origin: np.ndarray  # axis point at the first step's start
    steps: list[ShaftStep]

    @property
    def total_length(self) -> float:
        return self.steps[-1].end - self.steps[0].start if self.steps else 0.0


@dataclass
class Gear:
    center: np.ndarray
    axis: np.ndarray
    teeth: int
    module: float
    tip_diameter: float
    root_diameter: float
    pitch_diameter: float
    width: float
    inliers: np.ndarray = field(default=None, repr=False)


@dataclass
class GearStage:
    gear_a: int  # indices into the gears list
    gear_b: int
    ratio: float  # z_b / z_a
    center_distance: float


@dataclass
class Cone:
    apex: np.ndarray
    axis: np.ndarray  # unit, pointing from apex into the material
    half_angle_deg: float
    r_min: float  # radius at the near end of the scanned extent
    r_max: float  # radius at the far end
    height: float  # axial extent of the scanned surface
    inliers: np.ndarray
    rms: float


# --------------------------------------------------------------------- cylinders

def detect_cylinders(
    points: np.ndarray,
    normals: np.ndarray,
    distance_threshold: float,
    min_inliers: int = 300,
    max_cylinders: int = 32,
    max_radius: float | None = None,
    ransac_iterations: int = 1500,
    seed: int = 7,
) -> tuple[list[Cylinder], np.ndarray]:
    """Sequentially extract cylinders; returns ``(cylinders, unassigned_mask)``."""
    rng = np.random.default_rng(seed)
    n_total = len(points)
    spacing = _median_spacing(points, rng)
    connectivity_radius = max(4.0 * distance_threshold, 3.0 * spacing)
    remaining = np.arange(n_total)
    cylinders: list[Cylinder] = []
    consecutive_fails = 0

    while (
        len(remaining) >= min_inliers
        and len(cylinders) < max_cylinders
        and consecutive_fails < 6
    ):
        pts = points[remaining]
        nrm = normals[remaining]

        best_count = 0
        best_model = None
        for _ in range(ransac_iterations):
            i, j = rng.integers(len(pts), size=2)
            model = _cylinder_from_two_points(pts[i], nrm[i], pts[j], nrm[j])
            if model is None:
                continue
            center, axis, radius = model
            if radius < 2 * distance_threshold:
                continue
            if max_radius is not None and radius > max_radius:
                continue
            dist, radial_align = _cylinder_residuals(pts, nrm, center, axis, radius)
            mask = (np.abs(dist) < distance_threshold) & (radial_align > 0.85)
            count = int(mask.sum())
            if count > best_count:
                best_count = count
                best_model = (center, axis, radius, mask)

        if best_model is None or best_count < min_inliers:
            break

        center, axis, radius, mask = best_model
        # Refine on inliers, then re-collect once.
        for _ in range(3):
            center, axis, radius = _refine_cylinder(pts[mask], nrm[mask], center, axis)
            dist, radial_align = _cylinder_residuals(pts, nrm, center, axis, radius)
            mask = (np.abs(dist) < distance_threshold) & (radial_align > 0.85)
            if mask.sum() < min_inliers:
                break
        if mask.sum() < min_inliers:
            consecutive_fails += 1
            continue

        cand = remaining[mask]
        component = _largest_component(points[cand], connectivity_radius)
        if len(component) < min_inliers:
            remaining = np.setdiff1d(remaining, cand, assume_unique=True)
            continue
        cand = cand[component]

        dist, _ = _cylinder_residuals(points[cand], normals[cand], center, axis, radius)
        rms = float(np.sqrt(np.mean(dist**2)))
        if rms > 0.5 * distance_threshold:
            consecutive_fails += 1
            continue
        # Angular coverage gate: a real cylinder occupies a substantial arc.
        # A flat patch masquerading as a huge-radius cylinder covers only a
        # sliver of its circumference and is rejected here.
        u_b, v_b = _plane_basis(axis)
        rel_c = points[cand] - center
        ang = np.arctan2(rel_c @ v_b, rel_c @ u_b)
        occupied = len(np.unique(((ang + np.pi) / (2 * np.pi) * 64).astype(int) % 64))
        if occupied < 16:  # less than a quarter of the circumference
            consecutive_fails += 1
            continue
        consecutive_fails = 0

        t = (points[cand] - center) @ axis
        mid = center + axis * (t.min() + t.max()) / 2.0
        cylinders.append(
            Cylinder(
                center=mid,
                axis=axis,
                radius=float(radius),
                length=float(t.max() - t.min()),
                inliers=cand,
                rms=rms,
            )
        )
        remaining = np.setdiff1d(remaining, cand, assume_unique=True)

    unassigned = np.zeros(n_total, dtype=bool)
    unassigned[remaining] = True
    return cylinders, unassigned


def _cylinder_from_two_points(p1, n1, p2, n2):
    """Axis/center/radius hypothesis from two oriented surface points."""
    axis = np.cross(n1, n2)
    norm = np.linalg.norm(axis)
    if norm < 0.15:  # nearly parallel normals — unstable
        return None
    axis = axis / norm
    # Project into the plane perpendicular to the axis and intersect the
    # normal lines: q1 + s*m1 = q2 + t*m2 (2D).
    u, v = _plane_basis(axis)
    q1 = np.array([p1 @ u, p1 @ v])
    q2 = np.array([p2 @ u, p2 @ v])
    m1 = np.array([n1 @ u, n1 @ v])
    m2 = np.array([n2 @ u, n2 @ v])
    det = m1[0] * (-m2[1]) - m1[1] * (-m2[0])
    if abs(det) < 1e-9:
        return None
    rhs = q2 - q1
    s = (rhs[0] * (-m2[1]) - rhs[1] * (-m2[0])) / det
    c2d = q1 + s * m1
    radius = (np.linalg.norm(q1 - c2d) + np.linalg.norm(q2 - c2d)) / 2.0
    if radius < 1e-6:
        return None
    center = c2d[0] * u + c2d[1] * v  # any axis point (axial position arbitrary)
    return center, axis, radius


def _cylinder_residuals(pts, nrm, center, axis, radius):
    rel = pts - center
    radial = rel - np.outer(rel @ axis, axis)
    r = np.linalg.norm(radial, axis=1)
    dist = r - radius
    with np.errstate(invalid="ignore", divide="ignore"):
        radial_dir = radial / np.where(r[:, None] > 1e-12, r[:, None], 1.0)
    radial_align = np.abs(np.einsum("ij,ij->i", nrm, radial_dir))
    return dist, radial_align


def _refine_cylinder(pts, nrm, center, axis):
    """One refinement step: axis from normal covariance, then 2D circle fit."""
    # Cylinder normals span the plane perpendicular to the axis; the axis is
    # the eigenvector of the normal covariance with the smallest eigenvalue.
    cov = nrm.T @ nrm
    _, vecs = np.linalg.eigh(cov)
    new_axis = vecs[:, 0]
    if new_axis @ axis < 0:
        new_axis = -new_axis
    u, v = _plane_basis(new_axis)
    q = np.column_stack([pts @ u, pts @ v])
    cx, cy, radius = _fit_circle_2d(q)
    center = cx * u + cy * v
    return center, new_axis, radius


def _fit_circle_2d(q: np.ndarray) -> tuple[float, float, float]:
    """Kåsa algebraic circle fit."""
    a = np.column_stack([2 * q[:, 0], 2 * q[:, 1], np.ones(len(q))])
    b = q[:, 0] ** 2 + q[:, 1] ** 2
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy, c = sol
    radius = float(np.sqrt(max(c + cx**2 + cy**2, 1e-18)))
    return float(cx), float(cy), radius


def _plane_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1, 0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


# ------------------------------------------------------------------------- cones

def detect_cones(
    points: np.ndarray,
    normals: np.ndarray,
    distance_threshold: float,
    min_inliers: int = 400,
    max_cones: int = 8,
    ransac_iterations: int = 1200,
    seed: int = 7,
) -> tuple[list[Cone], np.ndarray]:
    """Detect truncated cones (hoppers, reducers, chutes).

    Every tangent plane of a cone passes through its apex, so the apex is the
    least-squares intersection of the inliers' tangent planes — that property
    drives both the 3-point RANSAC hypothesis and the refinement.
    """
    rng = np.random.default_rng(seed)
    n_total = len(points)
    spacing = _median_spacing(points, rng)
    connectivity_radius = max(4.0 * distance_threshold, 3.0 * spacing)
    scene = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    remaining = np.arange(n_total)
    cones: list[Cone] = []
    consecutive_fails = 0

    while (
        len(remaining) >= min_inliers
        and len(cones) < max_cones
        and consecutive_fails < 6
    ):
        pts = points[remaining]
        nrm = normals[remaining]

        best_count = 0
        best = None
        for _ in range(ransac_iterations):
            idx = rng.integers(len(pts), size=3)
            model = _cone_from_three_points(pts[idx], nrm[idx], scene)
            if model is None:
                continue
            apex, axis, half = model
            resid = _cone_residuals(pts, apex, axis, half)
            mask = np.abs(resid) < distance_threshold
            count = int(mask.sum())
            if count > best_count:
                best_count = count
                best = (apex, axis, half, mask)
        if best is None or best_count < min_inliers:
            break

        apex, axis, half, mask = best
        for _ in range(3):
            model = _refine_cone(pts[mask], nrm[mask])
            if model is None:
                break
            apex, axis, half = model
            resid = _cone_residuals(pts, apex, axis, half)
            mask = np.abs(resid) < distance_threshold
            if mask.sum() < min_inliers:
                break
        if mask.sum() < min_inliers or not (3.0 < np.rad2deg(half) < 80.0):
            consecutive_fails += 1
            continue

        cand = remaining[mask]
        component = _largest_component(points[cand], connectivity_radius)
        if len(component) < min_inliers:
            remaining = np.setdiff1d(remaining, cand, assume_unique=True)
            continue
        cand = cand[component]

        resid = _cone_residuals(points[cand], apex, axis, half)
        rms = float(np.sqrt(np.mean(resid**2)))
        if rms > 0.5 * distance_threshold or np.linalg.norm(apex) > 10 * scene:
            consecutive_fails += 1
            continue
        # Angular coverage around the axis (rejects planes posing as cones).
        u_b, v_b = _plane_basis(axis)
        rel = points[cand] - apex
        ang = np.arctan2(rel @ v_b, rel @ u_b)
        occupied = len(np.unique(((ang + np.pi) / (2 * np.pi) * 64).astype(int) % 64))
        if occupied < 16:
            consecutive_fails += 1
            continue
        consecutive_fails = 0

        t = rel @ axis
        t_lo, t_hi = float(t.min()), float(t.max())
        tan_half = np.tan(half)
        cones.append(
            Cone(
                apex=apex,
                axis=axis,
                half_angle_deg=float(np.rad2deg(half)),
                r_min=abs(t_lo) * tan_half,
                r_max=abs(t_hi) * tan_half,
                height=t_hi - t_lo,
                inliers=cand,
                rms=rms,
            )
        )
        remaining = np.setdiff1d(remaining, cand, assume_unique=True)

    unassigned = np.zeros(n_total, dtype=bool)
    unassigned[remaining] = True
    return cones, unassigned


def _cone_from_three_points(pts, nrm, scene: float):
    """Apex from three tangent planes, axis/angle from a small-circle fit."""
    m = np.vstack(nrm)
    if abs(np.linalg.det(m)) < 1e-4:
        return None
    apex = np.linalg.solve(m, np.einsum("ij,ij->i", nrm, pts))
    if np.linalg.norm(apex - pts.mean(axis=0)) > 10 * scene:
        return None
    return _axis_angle_from_apex(pts, apex)


def _refine_cone(pts, nrm):
    """Least-squares apex over all tangent planes, then small-circle refit."""
    b = np.einsum("ij,ij->i", nrm, pts)
    apex, *_ = np.linalg.lstsq(nrm, b, rcond=None)
    return _axis_angle_from_apex(pts, apex)


def _axis_angle_from_apex(pts, apex):
    """Fit axis and half-angle given the apex.

    The unit directions u_i = (p_i - apex)/|…| of cone surface points lie on
    a small circle of the unit sphere: u·axis = cos(half). Solving
    ``u_i · w = 1`` in least squares gives axis = w/|w|, cos(half) = 1/|w| —
    unbiased even for partial arcs, unlike the mean-direction estimator.
    """
    u = pts - apex
    lengths = np.linalg.norm(u, axis=1)
    good = lengths > 1e-9
    if good.sum() < 3:
        return None
    u = u[good] / lengths[good, None]
    w, *_ = np.linalg.lstsq(u, np.ones(len(u)), rcond=None)
    norm_w = float(np.linalg.norm(w))
    if norm_w <= 1.0 + 1e-9:
        return None  # cos(half) >= 1 — degenerate (plane or numerical junk)
    axis = w / norm_w
    half = float(np.arccos(1.0 / norm_w))
    if not (np.deg2rad(1.0) < half < np.deg2rad(88.0)):
        return None
    return apex, axis, half


def _cone_residuals(pts, apex, axis, half):
    rel = pts - apex
    lengths = np.linalg.norm(rel, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.clip((rel @ axis) / np.where(lengths > 1e-12, lengths, 1.0), -1, 1)
    theta = np.arccos(cosang)
    return lengths * np.sin(theta - half)


# ------------------------------------------------------------------------ shafts

def group_shafts(
    cylinders: list[Cylinder],
    points: np.ndarray,
    axis_angle_tol_deg: float = 3.0,
    radial_offset_tol: float | None = None,
) -> list[Shaft]:
    """Group coaxial cylinders into stepped shafts (ordered along the axis)."""
    if not cylinders:
        return []
    cos_tol = np.cos(np.deg2rad(axis_angle_tol_deg))
    used = [False] * len(cylinders)
    shafts: list[Shaft] = []
    order = np.argsort([-c.length for c in cylinders])
    for idx in order:
        if used[idx]:
            continue
        base = cylinders[idx]
        group = [idx]
        used[idx] = True
        offset_tol = radial_offset_tol if radial_offset_tol is not None else 0.25 * base.radius
        for jdx in range(len(cylinders)):
            if used[jdx]:
                continue
            other = cylinders[jdx]
            if abs(float(base.axis @ other.axis)) < cos_tol:
                continue
            # Radial distance of the other axis from the base axis.
            rel = other.center - base.center
            radial = rel - (rel @ base.axis) * base.axis
            if np.linalg.norm(radial) > offset_tol:
                continue
            group.append(jdx)
            used[jdx] = True

        # Order along the axis and emit steps.
        axis = base.axis
        entries = []
        for g in group:
            c = cylinders[g]
            t = (points[c.inliers] - base.center) @ axis
            entries.append((float(t.min()), float(t.max()), c))
        entries.sort(key=lambda e: e[0])
        origin = base.center + entries[0][0] * axis
        steps = [
            ShaftStep(
                diameter=2.0 * c.radius,
                length=t1 - t0,
                start=t0 - entries[0][0],
                end=t1 - entries[0][0],
            )
            for t0, t1, c in entries
        ]
        shafts.append(Shaft(axis=axis, origin=origin, steps=steps))
    return shafts


# ------------------------------------------------------------------------- gears

def measure_gear(
    points: np.ndarray,
    center_hint: np.ndarray | None = None,
    axis_hint: np.ndarray | None = None,
    min_teeth: int = 8,
    max_teeth: int = 300,
) -> Gear | None:
    """Measure a spur gear from its points (tooth count, module, circles).

    The angular radius profile (max radius per angle bin) oscillates once per
    tooth; the tooth count is the dominant FFT frequency. Returns ``None``
    when no significant periodicity exists (i.e. the part is not a gear).
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 500:
        return None
    centroid = pts.mean(axis=0) if center_hint is None else np.asarray(center_hint)

    if axis_hint is not None:
        axis = np.asarray(axis_hint, dtype=np.float64)
        candidates = [axis / np.linalg.norm(axis)]
    else:
        # Disk-like gears have their axis along the smallest extent; pinion
        # shafts (longer than wide) along the largest. Try both, keep the
        # candidate with the stronger tooth periodicity.
        centered = pts - centroid
        _, vecs = np.linalg.eigh(centered.T @ centered)
        candidates = [vecs[:, 0], vecs[:, 2]]

    # Refine each rough candidate with the flat end faces: their surface
    # normals are exactly parallel to the true rotation axis, which fixes
    # PCA tilt on gears whose diameter and width are similar.
    normals = _quick_normals(pts)
    refined = []
    for axis in candidates:
        for _ in range(2):
            dots = normals @ axis
            face = np.abs(dots) > 0.8
            if face.sum() < 50:
                break
            axis = (normals[face] * np.sign(dots[face])[:, None]).mean(axis=0)
            axis = axis / np.linalg.norm(axis)
        if not any(abs(float(axis @ r)) > 0.999 for r in refined):
            refined.append(axis)

    best: Gear | None = None
    best_score = 0.0
    for axis in refined:
        gear, score = _measure_gear_axis(pts, centroid, axis, min_teeth, max_teeth)
        if gear is not None and score > best_score:
            best, best_score = gear, score
    return best


def _quick_normals(pts: np.ndarray, k: int = 12) -> np.ndarray:
    """PCA normals on a subsampled kNN graph (sign-ambiguous, unit length)."""
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k, len(pts)), workers=-1)
    neigh = pts[idx]
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered)
    _, vecs = np.linalg.eigh(cov)
    normals = vecs[:, :, 0]
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)


def _measure_gear_axis(
    pts: np.ndarray,
    centroid: np.ndarray,
    axis: np.ndarray,
    min_teeth: int,
    max_teeth: int,
) -> tuple[Gear | None, float]:
    u, v = _plane_basis(axis)
    rel = pts - centroid
    # Recenter radially with a circle fit on the outer points.
    q = np.column_stack([rel @ u, rel @ v])
    r_all = np.linalg.norm(q, axis=1)
    outer = r_all > 0.6 * np.quantile(r_all, 0.98)
    cx, cy, _ = _fit_circle_2d(q[outer])
    q -= np.array([cx, cy])
    center = centroid + cx * u + cy * v

    r = np.linalg.norm(q, axis=1)
    theta = np.arctan2(q[:, 1], q[:, 0])

    # Angular profile: max radius per bin → tips; oscillation counts teeth.
    # Bin count adapts to the sampling density (~3 points per bin) so sparse
    # scans of small gears keep their bins populated.
    n_bins = 1 << int(np.clip(np.log2(max(len(pts) / 3, 64)), 6, 10))
    max_teeth = min(max_teeth, n_bins // 3)
    bins = ((theta + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    profile = np.full(n_bins, -np.inf)
    np.maximum.at(profile, bins, r)
    valid = np.isfinite(profile)
    if valid.mean() < 0.5:
        return None, 0.0
    # Fill gaps by interpolation over the circle.
    idx = np.arange(n_bins)
    profile[~valid] = np.interp(
        idx[~valid], idx[valid], profile[valid], period=n_bins
    )

    signal = profile - profile.mean()
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.arange(len(spectrum))
    band = (freqs >= min_teeth) & (freqs <= max_teeth)
    if not band.any():
        return None, 0.0
    peak = int(freqs[band][np.argmax(spectrum[band])])
    # Square-ish tooth profiles put a lot of power into harmonics; if half
    # the peak frequency is also strong, that is the true fundamental.
    if peak % 2 == 0 and peak // 2 >= min_teeth:
        if spectrum[peak // 2] > 0.5 * spectrum[peak]:
            peak = peak // 2
    peak_power = spectrum[peak]
    # Significance: the tooth frequency must clearly dominate the band —
    # ignoring its own harmonics, which are legitimate parts of the signal.
    non_harmonic = band.copy()
    k = peak
    while k <= max_teeth:
        non_harmonic[max(k - 1, 0) : k + 2] = False
        k += peak
    if peak_power < 4.0 * np.median(spectrum[band]) or (
        non_harmonic.any() and peak_power < 2.0 * spectrum[non_harmonic].max()
    ):
        return None, 0.0

    teeth = peak
    tip_r = float(np.quantile(profile, 0.90))
    # Root circle from the valleys of the profile.
    root_r = float(np.quantile(profile, 0.08))
    if tip_r - root_r < 1e-6 or root_r <= 0:
        return None, 0.0
    # Physical plausibility: gear teeth are shallow compared to the radius
    # (DIN: (ra-rf)/ra = 4.5/(z+2) <= 0.45). A "gear" seen down the wrong
    # axis produces huge tooth heights and fails this gate.
    if (tip_r - root_r) / tip_r > 0.5:
        return None, 0.0
    module = 2.0 * tip_r / (teeth + 2)  # DIN 780: da = m (z + 2)
    t_axial = rel @ axis
    width = float(np.quantile(t_axial, 0.99) - np.quantile(t_axial, 0.01))
    return Gear(
        center=center,
        axis=axis,
        teeth=teeth,
        module=module,
        tip_diameter=2 * tip_r,
        root_diameter=2 * root_r,
        pitch_diameter=module * teeth,
        width=width,
    ), float(peak_power / max(np.sum(spectrum[band]), 1e-12))


def find_gear_stages(
    gears: list[Gear], center_distance_tol: float = 0.08
) -> list[GearStage]:
    """Meshing gear pairs: parallel axes at pitch-circle center distance."""
    stages: list[GearStage] = []
    for i in range(len(gears)):
        for j in range(i + 1, len(gears)):
            a, b = gears[i], gears[j]
            if abs(float(a.axis @ b.axis)) < 0.99:
                continue
            # Distance between the two (parallel) axes.
            rel = b.center - a.center
            radial = rel - (rel @ a.axis) * a.axis
            dist = float(np.linalg.norm(radial))
            expected = (a.pitch_diameter + b.pitch_diameter) / 2.0
            if expected <= 0 or abs(dist - expected) / expected > center_distance_tol:
                continue
            stages.append(
                GearStage(
                    gear_a=i,
                    gear_b=j,
                    ratio=b.teeth / a.teeth,
                    center_distance=dist,
                )
            )
    return stages


# ------------------------------------------------------------------- primitives

def cylinder_mesh(
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    length: float,
    color: tuple[int, int, int] = (170, 190, 220),
    segments: int = 48,
    group: int = 0,
) -> Mesh:
    """Parametric tube mesh (with end caps) for visualizing detections."""
    axis = axis / np.linalg.norm(axis)
    u, v = _plane_basis(axis)
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v)
    bottom = center - axis * length / 2 + radius * ring
    top = center + axis * length / 2 + radius * ring
    verts = np.vstack([bottom, top, [center - axis * length / 2], [center + axis * length / 2]])
    faces = []
    cb, ct = 2 * segments, 2 * segments + 1
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, segments + i])
        faces.append([j, segments + j, segments + i])
        faces.append([cb, j, i])
        faces.append([ct, segments + i, segments + j])
    colors = np.tile(np.asarray(color, dtype=np.uint8), (len(verts), 1))
    return Mesh(
        vertices=verts,
        faces=np.array(faces),
        vertex_colors=colors,
        face_groups=np.full(len(faces), group, dtype=np.int64),
    )


# -------------------------------------------------------------------- top level

def analyze_machinery(
    cloud: PointCloud,
    distance_threshold: float | None = None,
    min_inlier_ratio: float = 0.02,
    seed: int = 7,
) -> dict:
    """Full industrial analysis: cylinders → shafts, gears, gear stages.

    Returns a JSON-ready report dict; detected primitive objects are attached
    under the ``"_objects"`` key for further processing (not serialized).
    """
    from scantobim.core.preprocess import (
        estimate_normals,
        estimate_point_spacing,
        remove_statistical_outliers,
        voxel_downsample,
    )

    raw_spacing = estimate_point_spacing(cloud)
    work = voxel_downsample(cloud, 1.5 * raw_spacing)
    work, _ = remove_statistical_outliers(work)
    work = estimate_normals(work, k_neighbors=16)
    spacing = estimate_point_spacing(work)
    if distance_threshold is None:
        # Machine parts need tighter tolerances than building scans — shaft
        # steps a few millimetres apart must stay separable.
        distance_threshold = 1.5 * spacing

    lo, hi = work.aabb
    min_inliers = max(150, int(min_inlier_ratio * len(work)))
    cylinders, unassigned = detect_cylinders(
        work.points,
        work.normals,
        distance_threshold=distance_threshold,
        min_inliers=min_inliers,
        max_radius=0.5 * float(np.max(hi - lo)),
        seed=seed,
    )

    # A gear's tip circle is detected as a cylinder — reclassify: gather all
    # points around each cylinder and test for tooth periodicity.
    gears: list[Gear] = []
    true_cylinders: list[Cylinder] = []
    consumed = np.zeros(len(work.points), dtype=bool)
    for c in cylinders:
        rel = work.points - c.center
        t = rel @ c.axis
        radial_r = np.linalg.norm(rel - np.outer(t, c.axis), axis=1)
        near = (
            (np.abs(t) < c.length / 2 + 4 * spacing)
            & (radial_r < c.radius + 2 * spacing)
            & (radial_r > 0.3 * c.radius)
        )
        gear = measure_gear(work.points[near], axis_hint=c.axis) if near.sum() > 500 else None
        if gear is not None and abs(gear.tip_diameter / 2 - c.radius) < 0.08 * c.radius:
            gear.inliers = np.flatnonzero(near)
            gears.append(gear)
            consumed |= near
        else:
            true_cylinders.append(c)
    cylinders = true_cylinders
    unassigned = unassigned & ~consumed
    shafts = group_shafts(cylinders, work.points)

    # Cones (hoppers, reducers) on the points no cylinder explained.
    cones: list[Cone] = []
    residual_idx = np.flatnonzero(unassigned)
    if len(residual_idx) > 400:
        found, cone_unassigned = detect_cones(
            work.points[residual_idx],
            work.normals[residual_idx],
            distance_threshold=distance_threshold,
            min_inliers=max(400, min_inliers),
            seed=seed,
        )
        for cone in found:
            cone.inliers = residual_idx[cone.inliers]
            cones.append(cone)
        still = np.zeros(len(work.points), dtype=bool)
        still[residual_idx[cone_unassigned]] = True
        unassigned = still

    # Gear candidates the cylinder stage missed entirely: clusters of
    # leftover points (teeth break the smooth cylinder model). Meshing gears
    # touch and form one cluster, so each cluster is decomposed recursively.
    residual_pts = work.points[unassigned]
    if len(residual_pts) > 500:
        clusters = _cluster_points(residual_pts, radius=max(6 * spacing, 2 * distance_threshold))
        for cl in clusters:
            if len(cl) < 500:
                continue
            for gear in _gears_in_cluster(residual_pts[cl], spacing):
                if not any(
                    np.linalg.norm(gear.center - g.center) < 0.5 * g.tip_diameter
                    for g in gears
                ):
                    gears.append(gear)
    stages = find_gear_stages(gears)

    report = {
        "points": len(cloud),
        "analyzed_points": len(work),
        "distance_threshold": round(float(distance_threshold), 6),
        "cylinders": [
            {
                "center": [round(float(x), 5) for x in c.center],
                "axis": [round(float(x), 5) for x in c.axis],
                "diameter": round(2 * c.radius, 5),
                "length": round(c.length, 5),
                "rms": round(c.rms, 6),
                "points": int(len(c.inliers)),
            }
            for c in cylinders
        ],
        "shafts": [
            {
                "axis": [round(float(x), 5) for x in s.axis],
                "origin": [round(float(x), 5) for x in s.origin],
                "total_length": round(s.total_length, 5),
                "steps": [
                    {
                        "diameter": round(st.diameter, 5),
                        "length": round(st.length, 5),
                        "from": round(st.start, 5),
                        "to": round(st.end, 5),
                    }
                    for st in s.steps
                ],
            }
            for s in shafts
        ],
        "gears": [
            {
                "center": [round(float(x), 5) for x in g.center],
                "axis": [round(float(x), 5) for x in g.axis],
                "teeth": g.teeth,
                "module": round(g.module, 6),
                "tip_diameter": round(g.tip_diameter, 5),
                "root_diameter": round(g.root_diameter, 5),
                "pitch_diameter": round(g.pitch_diameter, 5),
                "width": round(g.width, 5),
            }
            for g in gears
        ],
        "gear_stages": [
            {
                "gears": [st.gear_a, st.gear_b],
                "ratio": round(st.ratio, 4),
                "center_distance": round(st.center_distance, 5),
            }
            for st in stages
        ],
        "cones": [
            {
                "apex": [round(float(x), 5) for x in c.apex],
                "axis": [round(float(x), 5) for x in c.axis],
                "half_angle_deg": round(c.half_angle_deg, 2),
                "diameter_small": round(2 * c.r_min, 5),
                "diameter_large": round(2 * c.r_max, 5),
                "height": round(c.height, 5),
                "rms": round(c.rms, 6),
            }
            for c in cones
        ],
        "_objects": {
            "cylinders": cylinders,
            "shafts": shafts,
            "gears": gears,
            "cones": cones,
        },
    }
    return report


def _gears_in_cluster(pts: np.ndarray, spacing: float) -> list[Gear]:
    """Extract all gears from one point cluster.

    Meshing gears touch at their pitch circles and land in the same
    connected cluster, where the mixed angular profile defeats a direct
    measurement. Decomposition: gear tip circles *are* cylinders, so run the
    cylinder detector inside the cluster and measure a gear around each
    detected tip circle (axis/extent as hints), claiming points greedily.
    """
    if len(pts) < 500:
        return []
    gear = measure_gear(pts)
    if gear is not None:
        rel = pts - gear.center
        t = rel @ gear.axis
        rr = np.linalg.norm(rel - np.outer(t, gear.axis), axis=1)
        margin = 4 * spacing
        inside = (np.abs(t) < gear.width / 2 + margin) & (
            rr < gear.tip_diameter / 2 + margin
        )
        rest = pts[~inside]
        return [gear] + (_gears_in_cluster(rest, spacing) if len(rest) < len(pts) else [])

    normals = _quick_normals(pts)
    cylinders, _ = detect_cylinders(
        pts,
        normals,
        distance_threshold=3.0 * spacing,
        min_inliers=300,
        max_cylinders=8,
        seed=1,
    )
    gears: list[Gear] = []
    claimed = np.zeros(len(pts), dtype=bool)
    for c in sorted(cylinders, key=lambda c: -c.radius):
        rel = pts - c.center
        t = rel @ c.axis
        rr = np.linalg.norm(rel - np.outer(t, c.axis), axis=1)
        near = (
            (np.abs(t) < c.length / 2 + 4 * spacing)
            & (rr < c.radius + 2 * spacing)
            & (rr > 0.3 * c.radius)
            & ~claimed
        )
        if near.sum() < 500:
            continue
        gear = measure_gear(pts[near], axis_hint=c.axis)
        if gear is not None and abs(gear.tip_diameter / 2 - c.radius) < 0.08 * c.radius:
            gears.append(gear)
            rel_g = pts - gear.center
            t_g = rel_g @ gear.axis
            rr_g = np.linalg.norm(rel_g - np.outer(t_g, gear.axis), axis=1)
            margin = 4 * spacing
            claimed |= (np.abs(t_g) < gear.width / 2 + margin) & (
                rr_g < gear.tip_diameter / 2 + margin
            )
    return gears


def _cluster_points(points: np.ndarray, radius: float) -> list[np.ndarray]:
    """Connected components of the radius graph, largest first."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    n = len(points)
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if len(pairs) == 0:
        return []
    graph = sparse.csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)
    clusters = [np.flatnonzero(labels == i) for i in range(n_comp)]
    clusters.sort(key=len, reverse=True)
    return clusters
