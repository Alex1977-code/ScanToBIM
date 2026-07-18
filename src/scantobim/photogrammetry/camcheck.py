"""Per-camera color self-check against the colored point cloud.

The pose set as a whole can validate brilliantly while INDIVIDUAL cameras
still sample the wrong image — the classic case being stereo exports where
``left/0123.jpg`` and ``right/0123.jpg`` share a basename and the filename
index silently hands every camera the same side. Symptom on real data:
walls textured in sky colors although the pose score says 92 %.

This module checks every camera separately: project a sample of the
colored cloud into each CANDIDATE image file (all files matching the
camera's name or basename), measure the color agreement, keep the best
file, and drop cameras that agree with none. The result is a per-camera
``name → file`` map that the texture stages use instead of the ambiguous
basename index.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _candidate_files(images_dir: Path) -> dict[str, list[Path]]:
    """name → candidate files; relative paths, folder-prefixed and plain
    basenames all map, and a basename keeps EVERY file that carries it."""
    roots = [images_dir]
    try:
        for sibling in sorted(images_dir.parent.iterdir()):
            if sibling.is_dir() and sibling != images_dir:
                roots.append(sibling)
    except OSError:
        pass
    cands: dict[str, list[Path]] = {}
    for root in roots:
        try:
            files = sorted(root.rglob("*"))
        except OSError:
            continue
        for f in files:
            if not f.is_file() or f.suffix.lower() not in _SUFFIXES:
                continue
            rel = f.relative_to(root).as_posix()
            for key in (rel, f"{root.name}/{rel}", f.name):
                lst = cands.setdefault(key, [])
                if f not in lst:
                    lst.append(f)
    return cands


def _score_camera(cam, paths, pts, cols, scale: float, min_px: int):
    """(best_path, best_score) — color agreement of the projected sample."""
    from PIL import Image

    pc = pts @ cam.rotation.T + cam.translation
    in_front = pc[:, 2] > 0.3
    if int(in_front.sum()) < min_px:
        return None, -1.0
    pc = pc[in_front]
    cc = cols[in_front]
    px, py = cam.project(pc)
    if cam.model == "POLYFISHEYE":
        rho = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2)
        ok = np.arctan2(rho, pc[:, 2]) <= np.deg2rad(cam.max_theta_deg)
    else:
        ok = np.ones(len(pc), dtype=bool)
    ok &= (px >= 0) & (px <= cam.width - 1) & (py >= 0) & (py <= cam.height - 1)
    if int(ok.sum()) < min_px:
        return None, -1.0
    px, py, cc = px[ok], py[ok], cc[ok]

    best_path, best_score = None, -1.0
    for path in paths:
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail(
                    (max(2, int(cam.width * scale)),
                     max(2, int(cam.height * scale)))
                )
                photo = np.asarray(im)
        except Exception:  # noqa: BLE001 — unreadable candidate
            continue
        sy = np.clip(
            (py * photo.shape[0] / cam.height).astype(int), 0,
            photo.shape[0] - 1,
        )
        sx = np.clip(
            (px * photo.shape[1] / cam.width).astype(int), 0,
            photo.shape[1] - 1,
        )
        sampled = photo[sy, sx].astype(np.float64)
        diff = np.abs(sampled - cc.astype(np.float64)).mean()
        score = max(0.0, 1.0 - diff / 128.0)
        if score > best_score:
            best_path, best_score = path, score
    return best_path, best_score


def validate_cameras(
    cameras: list,
    images_dir,
    cloud,
    sample_points: int = 20_000,
    min_score: float = 0.35,
    min_px: int = 40,
    stats_out: dict | None = None,
):
    """Color-validate every camera; returns ``(kept, image_map)``.

    ``image_map`` maps each kept camera's name to ITS validated file —
    passed to the texture stages so basename collisions can never hand a
    camera the wrong stereo side again.
    """
    images_dir = Path(images_dir)
    if cloud.colors is None or not len(cameras):
        return cameras, None
    pts = cloud.points
    cols = cloud.colors
    if len(pts) > sample_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(pts), sample_points, replace=False)
        pts, cols = pts[sel], cols[sel]

    cands = _candidate_files(images_dir)

    def _paths_for(cam) -> list[Path]:
        return (
            cands.get(cam.name)
            or cands.get(Path(cam.name).name)
            or []
        )

    scale = 0.15  # thumbnail factor — plenty for color agreement
    results: list[tuple] = [None] * len(cameras)
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {
            pool.submit(
                _score_camera, cam, _paths_for(cam), pts, cols, scale, min_px
            ): i
            for i, cam in enumerate(cameras)
        }
        for fut in futs:
            results[futs[fut]] = fut.result()

    kept, image_map, scores = [], {}, []
    ambiguous_fixed = 0
    for cam, res in zip(cameras, results):
        path, score = res if res is not None else (None, -1.0)
        if path is None or score < min_score:
            continue
        kept.append(cam)
        image_map[cam.name] = path
        scores.append(score)
        if len(_paths_for(cam)) > 1:
            ambiguous_fixed += 1
    if stats_out is not None:
        stats_out["gesamt"] = len(cameras)
        stats_out["validiert"] = len(kept)
        stats_out["median_score"] = (
            round(float(np.median(scores)), 3) if scores else 0.0
        )
        stats_out["mehrdeutige_namen"] = ambiguous_fixed
    if not kept:
        return cameras, None  # never leave the pipeline empty-handed
    return kept, image_map
