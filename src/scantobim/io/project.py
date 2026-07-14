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

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CLOUD_EXTS = (".las", ".laz", ".e57", ".ply", ".pcd", ".pts", ".xyz")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
# Preference when several clouds exist (native scanner formats first).
_CLOUD_PRIORITY = {".e57": 0, ".las": 1, ".laz": 1, ".ply": 2, ".pcd": 3, ".pts": 4, ".xyz": 5}


def cloud_has_colors(path: str | Path) -> bool | None:
    """Header-only probe: does this point cloud file carry RGB colors?

    Returns ``True``/``False`` when the file header states it, ``None`` when
    it cannot be told without reading point data (``.pts/.xyz/...``) or the
    optional reader dependency is unavailable. Never reads the point data
    itself, so it is cheap even on multi-GB files.
    """
    path = Path(path)
    ext = path.suffix.lower()
    try:
        if ext == ".e57":
            import pye57

            reader = pye57.E57(str(path))
            for i in range(reader.scan_count):
                if "colorRed" in reader.get_header(i).point_fields:
                    return True
            return False
        if ext in (".las", ".laz"):
            import laspy

            with laspy.open(str(path)) as fh:
                dims = {d.lower() for d in fh.header.point_format.dimension_names}
            return {"red", "green", "blue"} <= dims
        if ext == ".ply":
            with open(path, "rb") as fh:
                head = fh.read(65536).decode("ascii", errors="replace")
            head = head.split("end_header")[0].lower()
            return re.search(r"property\s+\S+\s+(red|diffuse_red)\b", head) is not None
        if ext == ".pcd":
            with open(path, "rb") as fh:
                head = fh.read(4096).decode("ascii", errors="replace").lower()
            for line in head.splitlines():
                if line.startswith("fields"):
                    return "rgb" in line
    except Exception:
        return None
    return None


@dataclass
class SlamProject:
    root: Path
    cloud: Path | None = None
    clouds: list[Path] = field(default_factory=list)
    cloud_colored: bool | None = None
    cloud_note: str | None = None
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
            color = {True: ", mit Farben", False: ", ohne Farben"}.get(
                self.cloud_colored, ""
            )
            extra = f" (+{len(self.clouds) - 1} weitere)" if len(self.clouds) > 1 else ""
            lines.append(f"Punktwolke:   {self.cloud.name} ({mb:.1f} MB{color}){extra}")
            if self.cloud_note:
                lines.append(f"              → {self.cloud_note}")
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

    # Point cloud: SLAM exports often contain BOTH a colorized and a larger
    # uncolorized cloud of the same scan — picking by size alone loses the
    # colors (and with them the point-color texture). Rank by name hints
    # first, then verify with a header-only color probe.
    def _name_rank(p: Path):
        name = p.name.lower()
        uncolored = 1 if re.search(r"uncolor|nocolor|no_color|ohne_?farb", name) else 0
        colored = 0 if (not uncolored and re.search(r"color|farb|rgb", name)) else 1
        return (
            uncolored,
            colored,
            _CLOUD_PRIORITY.get(p.suffix.lower(), 9),
            -p.stat().st_size,
        )

    clouds.sort(key=_name_rank)
    project.clouds = clouds
    if clouds:
        chosen = clouds[0]
        chosen_colors = cloud_has_colors(chosen)
        if chosen_colors is not True and len(clouds) > 1:
            # The favourite has no verified colors — promote the first
            # candidate whose header proves RGB (probe a handful at most).
            for cand in clouds[1:6]:
                if cloud_has_colors(cand) is True:
                    project.cloud_note = (
                        f"farbige Wolke bevorzugt ({cand.name} statt {chosen.name})"
                    )
                    chosen, chosen_colors = cand, True
                    break
        project.cloud = chosen
        project.cloud_colored = chosen_colors

    # COLMAP model: shallowest hit.
    if colmap_dirs:
        project.colmap_model = min(colmap_dirs, key=lambda d: len(d.parts))

    # Image folder: prefer undistorted, then the one with the most images.
    if image_dirs:
        def score(item):
            d, n = item
            undist = any("undist" in part.lower() for part in d.parts)
            return (0 if undist else 1, -n)

        best_dir, best_count = min(image_dirs.items(), key=score)
        # Stereo/multi-camera rigs (left/right …): several image folders
        # sharing one parent → use the parent, so COLMAP names like
        # "left/frame_0001.jpg" resolve for BOTH cameras.
        siblings = {
            d: n for d, n in image_dirs.items() if d.parent == best_dir.parent
        }
        if len(siblings) > 1:
            project.images_dir = best_dir.parent
            project.image_count = sum(siblings.values())
        else:
            project.images_dir, project.image_count = best_dir, best_count

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
