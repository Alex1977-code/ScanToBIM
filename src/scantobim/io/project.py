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


def cloud_header_info(path: str | Path) -> tuple[int | None, bool | None]:
    """Header-only probe: ``(point count, has RGB colors)`` of a cloud file.

    Each element is ``None`` when it cannot be told without reading point
    data (``.pts/.xyz/...``) or the optional reader dependency is
    unavailable. Never reads the point data itself, so it is cheap even on
    multi-GB files.
    """
    path = Path(path)
    ext = path.suffix.lower()
    try:
        if ext == ".e57":
            import pye57

            reader = pye57.E57(str(path))
            count, colored = 0, False
            for i in range(reader.scan_count):
                header = reader.get_header(i)
                count += int(header.point_count)
                if "colorRed" in header.point_fields:
                    colored = True
            return count, colored
        if ext in (".las", ".laz"):
            import laspy

            with laspy.open(str(path)) as fh:
                dims = {d.lower() for d in fh.header.point_format.dimension_names}
                return int(fh.header.point_count), {"red", "green", "blue"} <= dims
        if ext == ".ply":
            with open(path, "rb") as fh:
                head = fh.read(65536).decode("ascii", errors="replace")
            head = head.split("end_header")[0].lower()
            m = re.search(r"element\s+vertex\s+(\d+)", head)
            colored = re.search(r"property\s+\S+\s+(red|diffuse_red)\b", head)
            return (int(m.group(1)) if m else None), colored is not None
        if ext == ".pcd":
            with open(path, "rb") as fh:
                head = fh.read(4096).decode("ascii", errors="replace").lower()
            count = colored = None
            for line in head.splitlines():
                if line.startswith("points"):
                    try:
                        count = int(line.split()[1])
                    except (IndexError, ValueError):
                        pass
                elif line.startswith("fields"):
                    colored = "rgb" in line
            return count, colored
    except Exception:
        return None, None
    return None, None


def cloud_has_colors(path: str | Path) -> bool | None:
    """Header-only probe: does this point cloud file carry RGB colors?"""
    return cloud_header_info(path)[1]


def _fmt_count(count: int | None) -> str:
    if count is None:
        return ""
    if count >= 1e6:
        return f", {count / 1e6:.1f} Mio Punkte"
    return f", {count:,} Punkte"


@dataclass
class SlamProject:
    root: Path
    cloud: Path | None = None
    clouds: list[Path] = field(default_factory=list)
    cloud_points: int | None = None
    cloud_colored: bool | None = None
    cloud_note: str | None = None
    color_source: Path | None = None
    colmap_model: Path | None = None
    xyzopk: Path | None = None
    imgpose: Path | None = None
    calibration: Path | None = None
    calibration_data: dict | None = None
    images_dir: Path | None = None
    image_count: int = 0
    trajectory: Path | None = None
    bags: list[Path] = field(default_factory=list)
    all_files: list = field(default_factory=list)  # [(Path, bytes), …]
    traj_candidates: list = field(default_factory=list)
    colmap_dirs: list = field(default_factory=list)

    def describe(self) -> list[str]:
        """Human-readable German summary of what was found."""
        lines = []
        if self.cloud is not None:
            mb = self.cloud.stat().st_size / 1e6
            color = {True: ", mit Farben", False: ", ohne Farben"}.get(
                self.cloud_colored, ""
            )
            extra = f" (+{len(self.clouds) - 1} weitere)" if len(self.clouds) > 1 else ""
            lines.append(
                f"Punktwolke:   {self.cloud.name} "
                f"({mb:.1f} MB{_fmt_count(self.cloud_points)}{color}){extra}"
            )
            if self.cloud_note:
                lines.append(f"              → {self.cloud_note}")
            if self.color_source is not None:
                lines.append(
                    f"Farbquelle:   {self.color_source.name} "
                    "(Farben werden auf die dichte Wolke übertragen)"
                )
        if self.colmap_model is not None:
            lines.append(f"Kameraposen:  {self.colmap_model.relative_to(self.root)} (COLMAP)")
        if self.images_dir is not None:
            lines.append(
                f"Fotos:        {self.image_count} Bilder in "
                f"{self.images_dir.relative_to(self.root)}"
            )
        if self.colmap_model is None and self.imgpose is not None:
            lines.append(
                f"Kameraposen:  {self.imgpose.relative_to(self.root)} "
                "(ImgPose — Quaternionen, eindeutige Rotationen)"
            )
        elif self.colmap_model is None and self.xyzopk is not None:
            lines.append(
                f"Kameraposen:  {self.xyzopk.relative_to(self.root)} "
                "(xyzopk — Ausrichtung & Brennweite werden am Scan "
                "selbstkalibriert)"
            )
        if (
            self.images_dir is not None
            and self.colmap_model is None
            and self.xyzopk is None
            and self.imgpose is None
        ):
            lines.append(
                "Kameraposen:  NICHT gefunden — Foto-Projektion nicht möglich "
                "(weder COLMAP-Modell noch xyzopk.txt)"
            )
        if self.calibration is not None and self.calibration_data:
            cams = self.calibration_data
            if isinstance(cams, dict):
                cams = [cams]
            parts = ", ".join(
                (c.get("name") or "Kamera")
                + (f" {c['width']}×{c['height']}" if c.get("width") else "")
                + f" fx={c['fx']:.0f}"
                for c in cams[:4]
            )
            lines.append(
                f"Kalibrierung: {self.calibration.relative_to(self.root)} "
                f"— {len(cams)} Kamera(s): {parts} "
                "(Auswahl nach Bildgröße der Fotos)"
            )
        if self.trajectory is not None:
            lines.append(f"Trajektorie:  {self.trajectory.relative_to(self.root)}")
        if self.bags:
            lines.append(
                f"Rohdaten:     {', '.join(b.name for b in self.bags)} "
                "(.bag — Rohaufnahme, wird nicht benötigt)"
            )
        return lines

    def _classify(self, f: Path) -> tuple[bool, str]:
        """(used?, reason) for one file of the project folder."""
        ext = f.suffix.lower()
        if f == self.cloud:
            return True, "gewählte Punktwolke"
        if f == self.color_source:
            return True, "Farbquelle für die Farbübertragung"
        if f in self.clouds:
            return False, (
                "weitere Punktwolke — das Ranking nach Dichte und Farben "
                f"wählte {self.cloud.name if self.cloud else '—'}"
            )
        if ext in IMAGE_EXTS:
            if self.images_dir is not None and (
                self.images_dir in f.parents or f.parent == self.images_dir
            ):
                return True, "Foto (Projektion/Farben)"
            return False, (
                "Bild außerhalb des gewählten Foto-Ordners "
                "(z. B. Vorschau/Thumbnail)"
            )
        if f == self.trajectory:
            return True, "Trajektorie (Normalen-Orientierung)"
        if f in self.traj_candidates:
            return False, "weitere Trajektorien-Datei — die größte wurde gewählt"
        if f == self.imgpose:
            return True, "Kameraposen (ImgPose — Quaternionen, bevorzugt)"
        if f == self.xyzopk:
            return True, "Kameraposen (xyzopk, selbstkalibriert)"
        if f == self.calibration:
            return True, (
                "Kamera-Kalibrierung — echte Brennweite/Bildmitte "
                "für die Foto-Projektion"
            )
        if self.colmap_model is not None and f.parent == self.colmap_model:
            if f.stem in ("cameras", "images"):
                return True, "Kameraposen (COLMAP)"
            return False, (
                "Teil des COLMAP-Modells — für die Projektion nicht benötigt"
            )
        if any(f.parent == d for d in self.colmap_dirs):
            return False, "weiteres COLMAP-Modell — das flachste wurde gewählt"
        if ext == ".bag":
            return False, (
                "Rohaufnahme des Scanners — ihre verarbeiteten Ergebnisse "
                "(Punktwolke, Fotos, Posen, Trajektorie) werden direkt genutzt"
            )
        if ext in (".json", ".log"):
            return False, "Scanner-Protokoll/Metadaten — enthält keine Geometrie"
        if ext in (".txt", ".csv"):
            return False, "Textdatei ohne erkannte Rolle"
        return False, f"Format {ext or '(ohne Endung)'} wird nicht verwendet"

    def unused_report(self, max_groups: int = 20) -> list[str]:
        """Grouped listing of files that are NOT used, with the reason.

        Files are grouped per (folder, extension, reason) so 1000 photos or
        logs collapse to one line; sorted by size, largest first.
        """
        groups: dict[tuple, list] = {}
        for f, size in self.all_files:
            used, reason = self._classify(f)
            if used:
                continue
            try:
                rel = f.relative_to(self.root).parent
            except ValueError:
                rel = f.parent
            key = (str(rel), f.suffix.lower(), reason)
            g = groups.setdefault(key, [0, 0, f.name])
            g[0] += 1
            g[1] += size
        lines = []
        ordered = sorted(groups.items(), key=lambda kv: -kv[1][1])
        for (rel, ext, reason), (count, size, first_name) in ordered[:max_groups]:
            where = f"{rel}/" if rel not in (".", "") else ""
            label = (
                f"{where}{first_name}" if count == 1
                else f"{where}*{ext or ''} ({count} Dateien)"
            )
            mb = size / 1e6
            size_txt = f"{mb / 1000:.2f} GB" if mb >= 1000 else f"{mb:.1f} MB"
            lines.append(f"{label} ({size_txt}) — {reason}")
        if len(ordered) > max_groups:
            lines.append(f"… und {len(ordered) - max_groups} weitere Gruppen")
        return lines


def scan_project_dir(root: str | Path, max_depth: int = 6) -> SlamProject:
    """Detect the usable components of a SLAM scanner project folder."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root}: kein Projektordner")
    project = SlamProject(root=root)

    clouds: list[Path] = []
    traj_candidates: list[Path] = []
    calib_candidates: list[Path] = []
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
            try:
                project.all_files.append((e, e.stat().st_size))
            except OSError:
                pass
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
            elif ext == ".txt" and "xyzopk" in name:
                project.xyzopk = e
            elif ext == ".txt" and "imgpose" in name:
                project.imgpose = e
            elif ext in (".yaml", ".yml") and "calib" in name:
                calib_candidates.append(e)
        if n_images:
            image_dirs[d] = n_images

    walk(root, 0)

    # Scanner camera calibration: real intrinsics beat self-calibration.
    for cand in sorted(calib_candidates):
        from scantobim.photogrammetry.calibration import read_camera_calibration

        data = read_camera_calibration(cand)
        if data is not None:
            project.calibration = cand
            project.calibration_data = data
            break

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
        # Header probe (point count + colors, never the point data) on the
        # leading candidates. Colors are worth a moderate density loss —
        # SLAM exports pair e.g. a 12M colorized with a 16M uncolorized
        # cloud — but NOT a preview-thin one (a 150k colored quicklook must
        # not displace the real multi-million-point scan). In that case the
        # dense cloud wins and the colored sibling becomes the color source
        # for a nearest-neighbour transfer.
        info = {p: cloud_header_info(p) for p in clouds[:8]}

        def weight(p: Path) -> float:
            count = info[p][0]
            # ≈20 bytes/point across LAS/PLY/E57 — rough, but plenty for
            # the orders-of-magnitude comparison needed here.
            return float(count) if count else p.stat().st_size / 20.0

        densest = max(info, key=weight)

        def _truly_colored(p: Path) -> bool:
            """Header colors alone lie: uncolorized exports often carry the
            INTENSITY as R=G=B grey. Name hints and a small data probe
            keep grey ramps out of the color ranking."""
            if info[p][1] is not True:
                return False
            if re.search(r"uncolor|nocolor|no_color|ohne_?farb|intens|grau|gr[ae]y",
                         p.name.lower()):
                return False
            return _colors_grey_probe(p) is not True

        colored = [p for p in info if _truly_colored(p)]
        best_colored = max(colored, key=weight) if colored else None
        # Geometry first: the densest cloud wins unless the colored sibling
        # is practically the same size (≥95%). SLAM exports pair e.g. a
        # 20.0M uncolorized with an 18.1M colorized cloud — those extra 10%
        # points are real geometry, and the colors transfer losslessly
        # enough via nearest neighbour from the colored sibling.
        if best_colored is not None and weight(best_colored) >= 0.95 * weight(densest):
            chosen = best_colored
            if chosen is not clouds[0]:
                project.cloud_note = (
                    f"farbige Wolke bevorzugt ({chosen.name} statt {clouds[0].name})"
                )
        else:
            chosen = densest
            if best_colored is not None and chosen not in colored:
                project.color_source = best_colored
                project.cloud_note = (
                    f"dichteste Wolke als Geometriequelle gewählt "
                    f"({chosen.name}); Farben werden von "
                    f"{best_colored.name} übertragen"
                )
            elif chosen is not clouds[0]:
                project.cloud_note = (
                    f"dichteste Wolke gewählt ({chosen.name} statt {clouds[0].name})"
                )
        project.cloud = chosen
        project.cloud_points = info[chosen][0]
        # "colored" means REAL camera colors — grey intensity ramps don't count.
        project.cloud_colored = (
            (chosen in colored) if info[chosen][1] else info[chosen][1]
        )

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
    project.traj_candidates = traj_candidates
    project.colmap_dirs = colmap_dirs

    return project


def _colors_grey_probe(path: Path, sample: int = 4000) -> bool | None:
    """True when a cloud's RGB is only a grey ramp (R≈G≈B, intensity).

    Reads a small sample of ACTUAL color data (LAS/LAZ chunked); None when
    the format can't be probed cheaply — the caller then trusts the header.
    """
    ext = path.suffix.lower()
    if ext not in (".las", ".laz"):
        return None
    try:
        import laspy

        with laspy.open(str(path)) as reader:
            for chunk in reader.chunk_iterator(sample):
                dims = set(chunk.point_format.dimension_names)
                if not {"red", "green", "blue"} <= dims:
                    return None
                r = np.asarray(chunk.red, dtype=np.int64)
                g = np.asarray(chunk.green, dtype=np.int64)
                b = np.asarray(chunk.blue, dtype=np.int64)
                if not len(r):
                    return None
                peak = max(int(r.max()), int(g.max()), int(b.max()), 1)
                dev = max(int(np.abs(r - g).max()), int(np.abs(g - b).max()))
                return dev <= max(2, peak // 100)
    except Exception:  # noqa: BLE001 — probe is best-effort
        return None
    return None


def read_trajectory(path: str | Path) -> np.ndarray:
    """Read a scanner trajectory file → ``(N, 3)`` or ``(N, 4)`` array.

    Column 4 (when present) is the pose TIMESTAMP — it lets the pipeline
    match every LAS point (``gps_time``) to the scanner position at its
    exact moment of capture. Tolerant of the common export layouts:

    * ``x y z``                                          (plain path)
    * ``time x y z [qx qy qz qw]``                       (TUM / SLAM)
    * ``x y z roll pitch yaw qx qy qz qw time``          (SHARE S20)

    The time column is recognized by monotony — first OR last column.
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

    def _monotonic(col: np.ndarray) -> bool:
        return len(col) > 2 and bool(np.all(np.diff(col) >= 0)) and col[-1] > col[0]

    if n_cols >= 4:
        first, last = data[:, 0], data[:, n_cols - 1]
        # Time LAST (e.g. SHARE S20: x y z r p y qx qy qz qw time). A large
        # magnitude separates true timestamps from a monotonic coordinate.
        if _monotonic(last) and (n_cols >= 5 or np.abs(last).max() > 1e5):
            return np.column_stack([data[:, 0:3], last])
        if _monotonic(first):
            return np.column_stack([data[:, 1:4], first])
        return data[:, 0:3]
    return data[:, 0:3]
