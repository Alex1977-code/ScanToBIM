"""ICP registration of roughly pre-aligned scans."""

import numpy as np

from scantobim.core.cloud import PointCloud
from scantobim.core.registration import (
    apply_transform,
    merge_clouds,
    register_point_to_plane,
)
from tests.synthetic import make_box_scan


def _rotz(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def test_icp_recovers_known_transform():
    target = make_box_scan(density=600, noise=0.003, seed=1)
    source_raw = make_box_scan(density=600, noise=0.003, seed=2)
    rot = _rotz(4.0)
    shift = np.array([0.15, -0.08, 0.05])
    source = PointCloud(points=source_raw.points @ rot.T + shift)

    res = register_point_to_plane(source, target, seed=3)
    aligned = apply_transform(source, res.transform)

    # The aligned cloud must coincide with the target geometry.
    assert res.rmse < 0.01
    from scipy.spatial import cKDTree

    d, _ = cKDTree(target.points).query(aligned.points[::10], k=1, workers=-1)
    assert np.median(d) < 0.02


def test_icp_identity_when_aligned():
    target = make_box_scan(density=500, noise=0.003, seed=5)
    source = make_box_scan(density=500, noise=0.003, seed=6)
    res = register_point_to_plane(source, target, seed=1)
    # Transform should stay near identity.
    assert np.linalg.norm(res.transform[:3, 3]) < 0.02
    assert abs(np.trace(res.transform[:3, :3]) - 3.0) < 1e-3


def test_merge_clouds_with_voxel():
    a = make_box_scan(density=300, seed=1)
    b = make_box_scan(density=300, seed=2)
    merged = merge_clouds([a, b])
    assert len(merged) == len(a) + len(b)
    thinned = merge_clouds([a, b], voxel_size=0.2)
    assert len(thinned) < len(merged)
