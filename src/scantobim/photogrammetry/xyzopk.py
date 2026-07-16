"""Photogrammetric exterior orientation: ``xyzopk.txt`` (SHARE SLAM S20 & Co.).

Some SLAM/photogrammetry exports ship camera poses not as a COLMAP model but
as one text line per photo: name, position X/Y/Z and the rotation angles
Omega/Phi/Kappa. The file carries **no intrinsics and no stated rotation
convention** — both are recovered here by *self-calibration against the
colorized point cloud*: the cloud was colored from exactly these photos, so
the correct convention + focal length is the one whose test projections
reproduce the point colors best.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scantobim.photogrammetry.colmap import CameraPose


def read_xyzopk(path: str | Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Parse an xyzopk file → ``[(image name, xyz, opk in degrees), …]``.

    Tolerant of headers/comments, comma or whitespace separation, and the
    image name in the first or last column. Angles already in radians
    (all magnitudes ≤ 2π) are converted to degrees.
    """
    entries: list[tuple[str, np.ndarray, np.ndarray]] = []
    max_angle = 0.0
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip().lstrip("#")
        if not line:
            continue
        parts = line.replace(",", " ").replace(";", " ").split()
        if len(parts) < 7:
            continue
        name = None
        nums: list[float] = []
        for tok in parts:
            try:
                nums.append(float(tok))
            except ValueError:
                if name is None:
                    name = tok
        if name is None or len(nums) < 6:
            continue  # header line
        xyz = np.array(nums[0:3])
        opk = np.array(nums[3:6])
        max_angle = max(max_angle, float(np.abs(opk).max()))
        entries.append((name, xyz, opk))
    if entries and max_angle <= 2 * np.pi + 0.01:
        entries = [(n, xyz, np.rad2deg(opk)) for n, xyz, opk in entries]
    return entries


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# World→camera (COLMAP: x right, y down, z forward) rotation builders for
# the conventions seen in the wild. D flips the photogrammetric camera
# frame (x right, y up, z backward) into the COLMAP frame.
_D = np.diag([1.0, -1.0, -1.0])


def _conventions():
    def m1(o, p, k):
        return _rx(o) @ _ry(p) @ _rz(k)

    def m2(o, p, k):
        return _rz(k) @ _ry(p) @ _rx(o)

    return {
        "opk/cam2world": lambda o, p, k: _D @ m1(o, p, k).T,
        "kpo/cam2world": lambda o, p, k: _D @ m2(o, p, k).T,
        "opk/world2cam": lambda o, p, k: _D @ m1(o, p, k),
        "kpo/world2cam": lambda o, p, k: _D @ m2(o, p, k),
        "opk/z-forward": lambda o, p, k: m1(o, p, k).T,
        "kpo/z-forward": lambda o, p, k: m2(o, p, k).T,
        "opk/z-forward-w2c": lambda o, p, k: m1(o, p, k),
        "kpo/z-forward-w2c": lambda o, p, k: m2(o, p, k),
    }


def _build_cameras(entries, convention, fx, width, height) -> list[CameraPose]:
    cams = []
    for name, xyz, opk in entries:
        o, p, k = np.deg2rad(opk)
        rot = convention(o, p, k)
        cams.append(
            CameraPose(
                name=name,
                width=width,
                height=height,
                fx=fx,
                fy=fx,
                cx=width / 2.0,
                cy=height / 2.0,
                k1=0.0,
                rotation=rot,
                translation=-rot @ xyz,
            )
        )
    return cams


def cameras_from_xyzopk(
    path: str | Path,
    images_dir: str | Path,
    cloud,
    stats_out: dict | None = None,
    sample_points: int = 30_000,
    probe_cameras: int = 8,
) -> list[CameraPose]:
    """xyzopk poses → calibrated ``CameraPose`` list.

    Rotation convention and focal length are recovered by projecting a
    sample of the (photo-)colorized point cloud into a handful of probe
    photos: the combination that best reproduces the point colors wins.
    Without cloud colors the in-image fraction decides.
    """
    from scantobim.core.texture import _index_images

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "xyzopk-Kalibrierung benötigt Pillow: pip install scantobim[photos]"
        ) from exc

    entries = read_xyzopk(path)
    if not entries:
        raise ValueError(f"{path}: keine Kameraposen gefunden")
    image_index = _index_images(Path(images_dir))

    # Image size from the first resolvable photo.
    width = height = None
    probe_photos: list[tuple[int, np.ndarray]] = []
    step = max(1, len(entries) // probe_cameras)
    for ei in range(0, len(entries), step):
        name = entries[ei][0]
        p = image_index.get(name) or image_index.get(Path(name).name)
        if p is None:
            # Names in xyzopk sometimes lack the extension.
            for ext in (".jpg", ".jpeg", ".png"):
                p = image_index.get(Path(name).stem + ext)
                if p is not None:
                    break
        if p is None or not p.exists():
            continue
        img = np.asarray(Image.open(p).convert("RGB"))
        if width is None:
            height, width = img.shape[:2]
        if img.shape[0] == height and img.shape[1] == width:
            probe_photos.append((ei, img))
        if len(probe_photos) >= probe_cameras:
            break
    if width is None:
        raise ValueError(
            f"{images_dir}: keine der xyzopk-Kameras hat ein auffindbares Foto"
        )

    pts = np.asarray(cloud.points, dtype=np.float64)
    colors = cloud.colors
    if len(pts) > sample_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(pts), sample_points, replace=False)
        pts = pts[sel]
        colors = None if colors is None else colors[sel]

    fx_grid = np.geomspace(0.35, 1.8, 14) * width
    best = None  # (score, conv_name, fx)
    for conv_name, conv in _conventions().items():
        for fx in fx_grid:
            score = _score(
                entries, conv, float(fx), width, height,
                probe_photos, pts, colors,
            )
            if best is None or score > best[0]:
                best = (score, conv_name, float(fx))

    score, conv_name, fx = best
    if stats_out is not None:
        stats_out["convention"] = conv_name
        stats_out["fx"] = round(fx, 1)
        stats_out["score"] = round(float(score), 4)
        stats_out["cameras"] = len(entries)
    return _build_cameras(entries, _conventions()[conv_name], fx, width, height)


def _score(entries, conv, fx, width, height, probe_photos, pts, colors) -> float:
    """Mean agreement of projected cloud points with the probe photos."""
    total = 0.0
    n = 0
    for ei, photo in probe_photos:
        name, xyz, opk = entries[ei]
        o, p, k = np.deg2rad(opk)
        rot = conv(o, p, k)
        pc = (pts - xyz) @ rot.T
        in_front = pc[:, 2] > 0.2
        if in_front.sum() < 50:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            px = fx * pc[:, 0] / pc[:, 2] + width / 2.0
            py = fx * pc[:, 1] / pc[:, 2] + height / 2.0
        ok = in_front & (px >= 0) & (px <= width - 1) & (py >= 0) & (py <= height - 1)
        if ok.sum() < 50:
            continue
        frac = float(ok.mean())
        if colors is None:
            total += frac
            n += 1
            continue
        sy = py[ok].round().astype(int)
        sx = px[ok].round().astype(int)
        diff = np.abs(
            photo[sy, sx].astype(np.float64) - colors[ok].astype(np.float64)
        ).mean()
        # Color agreement dominates; in-image fraction breaks ties.
        total += (1.0 - diff / 255.0) + 0.1 * frac
        n += 1
    return total / n if n else -1.0
