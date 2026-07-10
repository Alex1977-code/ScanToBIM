import numpy as np

from scantobim.core.planes import detect_planes, fit_plane_tls, plane_adjacency
from scantobim.core.preprocess import estimate_normals
from scantobim.core.regularize import regularize_planes
from tests.synthetic import make_box_scan


def test_tls_fit_exact():
    rng = np.random.default_rng(0)
    uv = rng.uniform(-1, 1, (200, 2))
    normal = np.array([1.0, 2.0, 3.0])
    normal /= np.linalg.norm(normal)
    u = np.array([-2.0, 1.0, 0.0]) / np.sqrt(5)
    v = np.cross(normal, u)
    pts = 0.7 * normal + uv[:, :1] * u + uv[:, 1:] * v
    n_fit, d_fit, rms = fit_plane_tls(pts)
    assert abs(abs(float(n_fit @ normal)) - 1.0) < 1e-9
    assert rms < 1e-9
    # plane passes through 0.7*normal
    assert abs(float(n_fit @ (0.7 * normal)) + d_fit) < 1e-9


def test_detect_box_planes():
    cloud = make_box_scan(density=400, noise=0.003)
    estimate_normals(cloud, k_neighbors=12)
    planes, unassigned = detect_planes(
        cloud.points,
        cloud.normals,
        distance_threshold=0.012,
        min_inliers=300,
        seed=1,
    )
    assert len(planes) == 6
    # every plane's normal is close to an axis
    for p in planes:
        assert np.max(np.abs(p.normal)) > 0.995
    # nearly all points explained (edge points with blended kNN normals
    # legitimately fail the normal gate, so allow a small remainder)
    assert unassigned.mean() < 0.08


def test_regularize_makes_axes_exact():
    cloud = make_box_scan(density=400, noise=0.004)
    estimate_normals(cloud, k_neighbors=12)
    planes, _ = detect_planes(
        cloud.points, cloud.normals, distance_threshold=0.015, min_inliers=300, seed=2
    )
    planes = regularize_planes(planes, cloud.points)
    assert len(planes) == 6
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            c = abs(float(planes[i].normal @ planes[j].normal))
            # exactly parallel or exactly orthogonal
            assert min(c, abs(c - 1.0)) < 1e-6


def test_coplanar_merge():
    # Two coplanar patches far apart, plus a supporting orthogonal wall.
    rng = np.random.default_rng(5)

    def patch(x0):
        n = 800
        pts = np.column_stack(
            [rng.uniform(x0, x0 + 1, n), rng.uniform(0, 1, n), rng.normal(0, 0.002, n)]
        )
        return pts

    pts = np.vstack([patch(0.0), patch(5.0)])
    from scantobim.core.cloud import PointCloud

    cloud = PointCloud(points=pts)
    estimate_normals(cloud, k_neighbors=10)
    planes, _ = detect_planes(
        cloud.points, cloud.normals, distance_threshold=0.008, min_inliers=200, seed=3
    )
    # Connectivity filter must split the patches …
    assert len(planes) == 2
    # … and since v1.6 the coplanar merge requires overlapping footprints:
    # two distant patches on the same infinite plane stay separate surfaces
    # (each gets its own boundary polygon instead of one swallowing the other).
    merged = regularize_planes(planes, cloud.points, merge_offset_tol=0.05)
    assert len(merged) == 2


def test_adjacency_box():
    cloud = make_box_scan(density=400, noise=0.003)
    estimate_normals(cloud, k_neighbors=12)
    planes, _ = detect_planes(
        cloud.points, cloud.normals, distance_threshold=0.012, min_inliers=300, seed=4
    )
    planes = regularize_planes(planes, cloud.points)
    adj = plane_adjacency(planes, cloud.points, contact_radius=0.15)
    # A closed box has 12 edges → 12 adjacent pairs.
    assert len(adj) == 12
