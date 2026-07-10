"""COLMAP integration: photos → dense point cloud.

COLMAP (https://colmap.github.io) is the de-facto standard open-source
structure-from-motion + multi-view-stereo system; it is also the camera-pose
backbone behind NeRF and 3D Gaussian Splatting workflows. This module shells
out to a locally installed ``colmap`` binary and returns the fused dense
cloud, ready for :func:`scantobim.reconstruct`.

The wrapper prefers the full MVS path (requires a CUDA build of COLMAP)::

    feature_extractor → exhaustive_matcher → mapper
        → image_undistorter → patch_match_stereo → stereo_fusion

and automatically falls back to the sparse SfM cloud when no CUDA-enabled
``patch_match_stereo`` is available. Sparse clouds are much thinner but still
usable for planar reconstruction of well-textured scenes.

Alternative photo pipelines whose output feeds straight into this package:

* 3D Gaussian Splatting / NeRF exports (``.ply`` point clouds)
* Apple RoomPlan / iPhone-LiDAR apps (``.ply``/``.e57`` exports)
* Meshroom, Metashape, RealityCapture (dense cloud export as ``.ply``/``.las``)
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scantobim.core.cloud import PointCloud
from scantobim.io.readers import read_point_cloud

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".heic"}


@dataclass
class CameraPose:
    """One registered photo: intrinsics + world→camera pose (COLMAP convention).

    ``p_cam = rotation @ p_world + translation``; the camera center in world
    coordinates is ``-rotation.T @ translation``.
    """

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float  # radial distortion (SIMPLE_RADIAL), 0 for pinhole models
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)

    def project(self, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project camera-space points to pixel coordinates."""
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = pts_cam[:, 0] / pts_cam[:, 2]
            yn = pts_cam[:, 1] / pts_cam[:, 2]
        if self.k1 != 0.0:
            r2 = xn * xn + yn * yn
            factor = 1.0 + self.k1 * r2
            xn, yn = xn * factor, yn * factor
        return self.fx * xn + self.cx, self.fy * yn + self.cy


def read_colmap_text_model(model_dir: str | Path) -> list[CameraPose]:
    """Read a COLMAP text model (``cameras.txt`` + ``images.txt``).

    Export a binary model with ``colmap model_converter --output_type TXT``.
    Supported camera models: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL
    (higher radial terms are ignored with a warning-free best effort).
    """
    model_dir = Path(model_dir)
    cams_file = model_dir / "cameras.txt"
    images_file = model_dir / "images.txt"
    if not cams_file.exists() or not images_file.exists():
        raise FileNotFoundError(
            f"{model_dir}: cameras.txt / images.txt not found — export the "
            "COLMAP model as text (model_converter --output_type TXT)"
        )

    intrinsics: dict[int, tuple] = {}
    for line in cams_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cam_id, model, w, h = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        params = [float(p) for p in parts[4:]]
        if model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy, k1 = params[1], params[2], 0.0
        elif model == "PINHOLE":
            fx, fy, cx, cy, k1 = params[0], params[1], params[2], params[3], 0.0
        elif model in ("SIMPLE_RADIAL", "RADIAL"):
            fx = fy = params[0]
            cx, cy, k1 = params[1], params[2], params[3]
        else:
            raise ValueError(f"unsupported COLMAP camera model: {model}")
        intrinsics[cam_id] = (w, h, fx, fy, cx, cy, k1)

    poses: list[CameraPose] = []
    lines = [
        ln.strip()
        for ln in images_file.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    # images.txt: two lines per image (pose line, 2D-points line).
    for pose_line in lines[::2]:
        parts = pose_line.split()
        qw, qx, qy, qz = (float(p) for p in parts[1:5])
        tx, ty, tz = (float(p) for p in parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]
        w, h, fx, fy, cx, cy, k1 = intrinsics[cam_id]
        poses.append(
            CameraPose(
                name=name,
                width=w,
                height=h,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                k1=k1,
                rotation=_quat_to_rot(qw, qx, qy, qz),
                translation=np.array([tx, ty, tz]),
            )
        )
    return poses


def _quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class ColmapNotFoundError(RuntimeError):
    pass


def find_colmap() -> str:
    exe = shutil.which("colmap")
    if exe is None:
        raise ColmapNotFoundError(
            "COLMAP binary not found on PATH. Install it from "
            "https://colmap.github.io/install.html (Ubuntu: apt install colmap) "
            "or export a dense cloud from another photogrammetry tool and feed "
            "the .ply/.las file to `scantobim reconstruct` directly."
        )
    return exe


def photos_to_point_cloud(
    image_dir: str | Path,
    work_dir: str | Path,
    quality: str = "high",
    use_gpu: bool = True,
    verbose: bool = False,
) -> PointCloud:
    """Run COLMAP on ``image_dir`` and return the dense (or sparse) cloud.

    Parameters
    ----------
    image_dir:
        Directory containing the photos (jpg/png/…).
    work_dir:
        Scratch directory for the COLMAP database and models; kept on disk so
        re-runs can resume and the user can inspect intermediate results.
    quality:
        ``low`` / ``medium`` / ``high`` — trades density for runtime.
    """
    image_dir = Path(image_dir)
    work_dir = Path(work_dir)
    exe = find_colmap()

    images = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    if len(images) < 3:
        raise ValueError(
            f"{image_dir}: found {len(images)} images, need at least 3 "
            "overlapping photos for reconstruction"
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    database = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    dense_dir = work_dir / "dense"
    sparse_dir.mkdir(exist_ok=True)

    sift_gpu = "1" if use_gpu else "0"
    max_image_size = {"low": "1600", "medium": "2400", "high": "3200"}[quality]

    def run(args: list[str]) -> None:
        result = subprocess.run(
            [exe, *args],
            capture_output=not verbose,
            text=True,
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "")[-2000:] if not verbose else ""
            raise RuntimeError(f"colmap {args[0]} failed (exit {result.returncode})\n{tail}")

    run(
        [
            "feature_extractor",
            "--database_path", str(database),
            "--image_path", str(image_dir),
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", sift_gpu,
        ]
    )
    run(
        [
            "exhaustive_matcher",
            "--database_path", str(database),
            "--SiftMatching.use_gpu", sift_gpu,
        ]
    )
    run(
        [
            "mapper",
            "--database_path", str(database),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir),
        ]
    )
    model_dir = sparse_dir / "0"
    if not model_dir.exists():
        raise RuntimeError(
            "COLMAP mapper produced no model — the photos likely lack overlap "
            "or texture. Shoot with ≥70% overlap and avoid blank walls/glass."
        )

    # Export the camera poses as text so `scantobim reconstruct
    # --texture-photos … --colmap-model <work_dir>/model_txt` can project the
    # original photos onto the reconstructed model later.
    model_txt = work_dir / "model_txt"
    model_txt.mkdir(exist_ok=True)
    try:
        run(
            [
                "model_converter",
                "--input_path", str(model_dir),
                "--output_path", str(model_txt),
                "--output_type", "TXT",
            ]
        )
    except RuntimeError:
        pass  # texturing stays available via cloud colors

    # --- dense MVS (needs CUDA); fall back to sparse otherwise -------------
    try:
        dense_dir.mkdir(exist_ok=True)
        run(
            [
                "image_undistorter",
                "--image_path", str(image_dir),
                "--input_path", str(model_dir),
                "--output_path", str(dense_dir),
                "--max_image_size", max_image_size,
            ]
        )
        run(
            [
                "patch_match_stereo",
                "--workspace_path", str(dense_dir),
                "--PatchMatchStereo.geom_consistency", "true",
            ]
        )
        fused = dense_dir / "fused.ply"
        run(
            [
                "stereo_fusion",
                "--workspace_path", str(dense_dir),
                "--output_path", str(fused),
            ]
        )
        return read_point_cloud(fused)
    except RuntimeError:
        # No CUDA build — export the sparse SfM points instead.
        sparse_ply = work_dir / "sparse.ply"
        run(
            [
                "model_converter",
                "--input_path", str(model_dir),
                "--output_path", str(sparse_ply),
                "--output_type", "PLY",
            ]
        )
        cloud = read_point_cloud(sparse_ply)
        cloud.source = f"{image_dir} (COLMAP sparse fallback)"
        return cloud
