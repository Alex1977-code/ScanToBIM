"""SLAM scanner project import (SHARE SLAM S20, GeoSLAM, LiGrip, …).

Handheld and mobile SLAM scanners export a project folder rather than a
single file — typically a point cloud, undistorted photos, COLMAP-format
camera poses, a ``trajectory.txt`` of the scanner path, logs/JSON metadata
and the vendor's raw recording (e.g. a ROS ``.bag``). This module finds the
usable pieces automatically:

* **Point cloud** — the geometry backbone (``.las/.laz/.e57/.ply/…``).
* **COLMAP model + undistorted images** — feed the photo-projection
  texturing (sharpest possible photorealism on the clean geometry).
* **Trajectory** — orients the normals towards the scanner path, making
  plane/cylinder detection markedly more robust on real scans.
* The raw ``.bag`` recording is the vendor's input format (raw LiDAR/IMU
  streams before SLAM) — its processed results ARE the files above, so it
  is intentionally left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CLOUD_EXTS = (".las", ".laz", ".e57", ".ply", ".pcd", ".pts", ".xyz")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
# Preference when several clouds exist (native scanner formats first).
_CLOUD_PRIORITY = {".e57": 0, ".las": 1, ".laz": 1, ".ply": 2, ".pcd": 3, ".pts": 4, ".xyz": 5}


@dataclass
class SlamProject:
    root: Path
    cloud: Path | None = None
    clouds: list[Path] = field(default_factory=list)
    colmap_model: Path | None = None
    images_dir: Path | None = None
    image_count: int = 0
    trajectory: Path | None = None
    bags: list[Path] = field(default_factory=list)

    def describe(self) -> list[str]:
        """Human-readable German summary of what was found."""
        lines = []
        if self.cloud is not None:
            mb = self.cloud.stat().st_size / 1e6
            extra = f" (+{len(self.clouds) - 1} weitere)" if len(self.clouds) > 1 else ""
            lines.append(f"Punktwolke:   {self.cloud.name} ({mb:.1f} MB){extra}")
        if self.colmap_model is not None:
            lines.append(f"Kameraposen:  {self.colmap_model.relative_to(self.root)} (COLMAP)")
        if self.images_dir is not None:
            lines.append(
                f"Fotos:        {self.image_count} Bilder in "
                f"{self.images_dir.relative_to(self.root)}"
            )
        if self.trajectory is not None:
            lines.append(f"Trajektorie:  {self.trajectory.relative_to(self.root)}")
        if self.bags:
            lines.append(
                f"Rohdaten:     {', '.join(b.name for b in self.bags)} "
                "(.bag — Rohaufnahme, wird nicht benötigt)"
            )
        return lines


def scan_project_dir(root: str | Path, max_depth: int = 4) -> SlamProject:
    """Detect the usable components of a SLAM scanner project folder."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root}: kein Projektordner")
    project = SlamProject(root=root)

    clouds: list[Path] = []
    traj_candidates: list[Path] = []
    image_dirs: dict[Path, int] = {}
    colmap_dirs: list[Path] = []

    def walk(d: Path, depth: int):
        try:
            entries = sorted(d.iterdir())
        except PermissionError:
            return
        n_images = 0
        has_cam_txt = (d / "cameras.txt").exists() and (d / "images.txt").exists()
        has_cam_bin = (d / "cameras.bin").exists() and (d / "images.bin").exists()
        if has_cam_txt or has_cam_bin:
            colmap_dirs.append(d)
        for e in entries:
            if e.name.startswith("."):
                continue
            if e.is_dir():
                if depth < max_depth:
                    walk(e, depth + 1)
                continue
            ext = e.suffix.lower()
            name = e.name.lower()
            if ext in CLOUD_EXTS:
                clouds.append(e)
            elif ext in IMAGE_EXTS:
                n_images += 1
            elif ext == ".bag":
                project.bags.append(e)
            elif ext in (".txt", ".csv") and ("traj" in name or "path" in name):
                traj_candidates.append(e)
        if n_images:
            image_dirs[d] = n_images

    walk(root, 0)

    # Point cloud: preferred format, then largest file.
    clouds.sort(key=lambda p: (_CLOUD_PRIORITY.get(p.suffix.lower(), 9), -p.stat().st_size))
    project.clouds = clouds
    project.cloud = clouds[0] if clouds else None

    # COLMAP model: shallowest hit.
    if colmap_dirs:
        project.colmap_model = min(colmap_dirs, key=lambda d: len(d.parts))

    # Image folder: prefer undistorted, then the one with the most images.
    if image_dirs:
        def score(item):
            d, n = item
            undist = any("undist" in part.lower() for part in d.parts)
            return (0 if undist else 1, -n)

        best = min(image_dirs.items(), key=score)
        project.images_dir, project.image_count = best

    if traj_candidates:
        project.trajectory = max(traj_candidates, key=lambda p: p.stat().st_size)

    return project


def read_trajectory(path: str | Path) -> np.ndarray:
    """Read a scanner trajectory file → ``(N, 3)`` positions.

    Tolerant of the common export layouts:

    * ``x y z``                              (plain path)
    * ``time x y z [...]``                   (timestamped)
    * ``time x y z qx qy qz qw``             (TUM / SLAM pose format)

    Comma or whitespace separated; comment/header lines are skipped.
    """
    path = Path(path)
    rows: list[list[float]] = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip().lstrip("#")
            if not line:
                continue
            parts = line.replace(",", " ").split()
            try:
                values = [float(p) for p in parts]
            except ValueError:
                continue  # header line
            if len(values) >= 3:
                rows.append(values)
    if not rows:
        raise ValueError(f"{path}: keine Trajektorien-Punkte gefunden")
    n_cols = min(len(r) for r in rows)
    data = np.array([r[:n_cols] for r in rows])

    if n_cols >= 8:
        return data[:, 1:4]  # time x y z qx qy qz qw
    if n_cols >= 4:
        first = data[:, 0]
        if np.all(np.diff(first) >= 0) and len(first) > 2:
            return data[:, 1:4]  # monotonic first column = time
        return data[:, 0:3]
    return data[:, 0:3]
