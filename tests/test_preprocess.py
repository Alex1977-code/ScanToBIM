import numpy as np

from scantobim.core.cloud import PointCloud
from scantobim.core.preprocess import (
    estimate_normals,
    estimate_point_spacing,
    remove_statistical_outliers,
    voxel_downsample,
)


def _grid_cloud(step=0.1, n=20):
    xs = np.arange(n) * step
    xx, yy = np.meshgrid(xs, xs)
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n * n)])
    return PointCloud(points=pts)


def test_voxel_downsample_reduces_and_keeps_extent():
    cloud = _grid_cloud(step=0.1, n=20)
    out = voxel_downsample(cloud, voxel_size=0.35)
    assert len(out) < len(cloud)
    lo1, hi1 = cloud.aabb
    lo2, hi2 = out.aabb
    assert np.all(lo2 >= lo1 - 1e-9) and np.all(hi2 <= hi1 + 1e-9)
    # Representative points are original points, not averages.
    orig = {tuple(p) for p in cloud.points}
    assert all(tuple(p) in orig for p in out.points)


def test_voxel_downsample_zero_is_noop():
    cloud = _grid_cloud()
    assert voxel_downsample(cloud, 0) is cloud


def test_outlier_removal():
    cloud = _grid_cloud(step=0.05, n=30)
    junk = np.array([[10.0, 10.0, 10.0], [-5.0, -5.0, 3.0]])
    noisy = PointCloud(points=np.vstack([cloud.points, junk]))
    cleaned, keep = remove_statistical_outliers(noisy, k_neighbors=8, std_ratio=2.0)
    assert not keep[-1] and not keep[-2]  # both outliers dropped
    assert len(cleaned) >= len(cloud) - 5  # grid mostly intact


def test_normal_estimation_flat_plane():
    cloud = _grid_cloud(step=0.05, n=25)
    estimate_normals(cloud, k_neighbors=8)
    dots = np.abs(cloud.normals @ np.array([0.0, 0.0, 1.0]))
    assert np.all(dots > 0.999)


def test_normal_orientation_towards_sensor():
    cloud = _grid_cloud(step=0.05, n=15)
    estimate_normals(cloud, k_neighbors=8, orient_towards=np.array([0, 0, 5.0]))
    assert np.all(cloud.normals[:, 2] > 0.99)


def test_point_spacing():
    cloud = _grid_cloud(step=0.1, n=20)
    spacing = estimate_point_spacing(cloud)
    assert abs(spacing - 0.1) < 0.01
