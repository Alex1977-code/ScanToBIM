"""Automatic axis alignment (Aufstellung ins Weltkoordinatensystem).

Scans arrive in arbitrary (scanner or georeferenced) coordinates. For CAD/BIM
import it is usually wanted that the dominant wall direction runs along +X,
the vertical along +Z and the floor sits at Z=0. The alignment is derived
from the regularized plane directions (weighted by supporting points) and is
reported as a 4x4 matrix so it stays reversible / documentable.
"""

from __future__ import annotations

import numpy as np

from scantobim.core.mesh import Mesh
from scantobim.core.planes import Plane


def compute_alignment(planes: list[Plane], max_vertical_tilt_deg: float = 40.0) -> np.ndarray:
    """4x4 world-alignment matrix from the model's dominant directions.

    * The direction group closest to vertical (within ``max_vertical_tilt_deg``)
      is rotated onto +Z.
    * The strongest direction group orthogonal to it is rotated onto +X.
    * Returns identity when no reliable vertical exists (e.g. a single wall).

    Translation is *not* included here — floors are placed at Z=0 by
    :func:`apply_alignment` once the rotated geometry is known.
    """
    identity = np.eye(4)
    if not planes:
        return identity

    # Direction groups (regularized planes share exact normals up to sign).
    dirs: list[np.ndarray] = []
    weights: list[float] = []
    for p in planes:
        n = p.normal if p.normal[np.abs(p.normal).argmax()] >= 0 else -p.normal
        for i, d in enumerate(dirs):
            if abs(float(n @ d)) > 0.9999:
                weights[i] += len(p.inliers)
                break
        else:
            dirs.append(n)
            weights.append(len(p.inliers))

    up = np.array([0.0, 0.0, 1.0])
    cos_max = np.cos(np.deg2rad(max_vertical_tilt_deg))
    vertical = None
    best_w = 0.0
    for d, w in zip(dirs, weights):
        if abs(float(d @ up)) >= cos_max and w > best_w:
            vertical = d if d @ up > 0 else -d
            best_w = w
    if vertical is None:
        return identity

    x_dir = None
    best_w = 0.0
    for d, w in zip(dirs, weights):
        if abs(float(d @ vertical)) < 0.1 and w > best_w:
            x_dir = d
            best_w = w
    if x_dir is None:
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(helper @ vertical)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        x_dir = helper - (helper @ vertical) * vertical
    # Orthonormalize exactly.
    x_dir = x_dir - (x_dir @ vertical) * vertical
    x_dir = x_dir / np.linalg.norm(x_dir)
    y_dir = np.cross(vertical, x_dir)

    rot = np.eye(4)
    rot[:3, :3] = np.vstack([x_dir, y_dir, vertical])  # world_from_scan rotation
    return rot


def apply_alignment(
    mesh: Mesh,
    points: np.ndarray | None,
    transform: np.ndarray,
    floor_to_zero: bool = True,
) -> tuple[Mesh, np.ndarray | None, np.ndarray]:
    """Apply ``transform`` to mesh (in place) and optional extra points.

    When ``floor_to_zero`` is set, the lowest mesh vertex is shifted to Z=0
    afterwards and the translation is folded into the returned final matrix.
    """
    rot = transform[:3, :3]
    mesh.vertices = mesh.vertices @ rot.T + transform[:3, 3]
    if points is not None:
        points = points @ rot.T + transform[:3, 3]

    final = transform.copy()
    if floor_to_zero and len(mesh.vertices):
        dz = -float(mesh.vertices[:, 2].min())
        mesh.vertices[:, 2] += dz
        if points is not None:
            points = points.copy()
            points[:, 2] += dz
        final = np.eye(4) @ final
        final[2, 3] += dz
    return mesh, points, final


def transform_normal(normal: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Rotate a (unit) normal by the rotational part of ``transform``."""
    return transform[:3, :3] @ normal
