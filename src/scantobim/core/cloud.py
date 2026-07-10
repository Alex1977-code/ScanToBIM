"""Core point cloud container used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PointCloud:
    """A point cloud with optional per-point attributes.

    Attributes
    ----------
    points:
        ``(N, 3)`` float64 XYZ coordinates.
    colors:
        Optional ``(N, 3)`` uint8 RGB.
    normals:
        Optional ``(N, 3)`` float64 unit normals.
    intensity:
        Optional ``(N,)`` float32 sensor intensity (normalized 0..1 when read
        from LAS).
    """

    points: np.ndarray
    colors: np.ndarray | None = None
    normals: np.ndarray | None = None
    intensity: np.ndarray | None = None
    source: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        self.points = np.ascontiguousarray(self.points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got {self.points.shape}")
        n = len(self.points)
        if self.colors is not None:
            self.colors = np.ascontiguousarray(self.colors, dtype=np.uint8)
            if self.colors.shape != (n, 3):
                raise ValueError("colors must match points shape (N, 3)")
        if self.normals is not None:
            self.normals = np.ascontiguousarray(self.normals, dtype=np.float64)
            if self.normals.shape != (n, 3):
                raise ValueError("normals must match points shape (N, 3)")
        if self.intensity is not None:
            self.intensity = np.ascontiguousarray(self.intensity, dtype=np.float32)
            if self.intensity.shape != (n,):
                raise ValueError("intensity must be (N,)")

    def __len__(self) -> int:
        return len(self.points)

    def select(self, indices: np.ndarray) -> "PointCloud":
        """Return a new cloud containing only ``indices`` (bool mask or ints)."""
        return PointCloud(
            points=self.points[indices],
            colors=None if self.colors is None else self.colors[indices],
            normals=None if self.normals is None else self.normals[indices],
            intensity=None if self.intensity is None else self.intensity[indices],
            source=self.source,
        )

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box as ``(min_xyz, max_xyz)``."""
        return self.points.min(axis=0), self.points.max(axis=0)

    @property
    def centroid(self) -> np.ndarray:
        return self.points.mean(axis=0)
