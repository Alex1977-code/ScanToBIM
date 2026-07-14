"""ScanToBIM — reconstruct clean-edged 3D models from LiDAR point clouds and photos.

The package implements a structured scan-to-model pipeline:

1.  Ingest point clouds from LAS/LAZ, PLY, PCD, E57, XYZ/PTS/CSV — or a dense
    cloud produced from photos via the bundled COLMAP wrapper.
2.  Preprocess: voxel thinning, statistical outlier removal, normal estimation.
3.  Detect planar structure with normal-aware RANSAC refined by connected
    component analysis and total-least-squares fitting.
4.  Regularize plane orientations (parallel / orthogonal / Manhattan snapping)
    so that walls, floors and ceilings meet at exact angles.
5.  Extract crisp boundary polygons per plane (alpha shape → simplification →
    dominant-direction straightening) and snap them onto the exact
    plane-plane intersection lines and 3-plane corner points.
6.  Triangulate and export watertight-where-possible meshes to OBJ, PLY, STL
    and glTF/GLB together with a machine-readable quality report.
"""

from scantobim.core.cloud import PointCloud
from scantobim.core.mesh import Mesh
from scantobim.core.pipeline import (
    PipelineConfig,
    ReconstructionResult,
    SurfaceGeometry,
    reconstruct,
)
from scantobim.core.registration import merge_clouds, register_point_to_plane

__version__ = "2.7.0"

__all__ = [
    "PointCloud",
    "Mesh",
    "PipelineConfig",
    "ReconstructionResult",
    "SurfaceGeometry",
    "reconstruct",
    "register_point_to_plane",
    "merge_clouds",
    "__version__",
]
