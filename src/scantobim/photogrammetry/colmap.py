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
from pathlib import Path

from scantobim.core.cloud import PointCloud
from scantobim.io.readers import read_point_cloud

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".heic"}


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
