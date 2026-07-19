"""Foto-Geometrie-Stufe: Fisheye-Rückprojektion, Posen-Feinschliff,
LiDAR-geführtes MVS, photokonsistentes Feintuning, Render-Regelkreis."""

import numpy as np
import pytest

from scantobim.photogrammetry.colmap import CameraPose


def _cam(name, center, fx=350.0, w=400, h=300):
    r = np.eye(3)
    t = -r @ np.asarray(center, dtype=np.float64)
    return CameraPose(
        name=name, width=w, height=h, fx=fx, fy=fx, cx=w / 2, cy=h / 2,
        k1=0.0, rotation=r, translation=t,
    )


def _checker(x, y, period):
    return (
        np.floor(x / period).astype(np.int64)
        + np.floor(y / period).astype(np.int64)
    ) % 2


def _plane_photo(cam, plane_z, period, blur=1):
    """Analytic photo of a checker plane at world ``z = plane_z``."""
    from scipy.ndimage import uniform_filter

    xs, ys = np.meshgrid(
        np.arange(cam.width, dtype=np.float64),
        np.arange(cam.height, dtype=np.float64),
    )
    rays = cam.unproject(xs.ravel(), ys.ravel())
    c = -cam.rotation.T @ cam.translation
    dirs = rays @ cam.rotation
    tt = (plane_z - c[2]) / dirs[:, 2]
    wx = c[0] + tt * dirs[:, 0]
    wy = c[1] + tt * dirs[:, 1]
    img = 40.0 + 180.0 * _checker(wx, wy, period)
    img = img.reshape(cam.height, cam.width)
    if blur:
        img = uniform_filter(img, 3)
    return img.astype(np.uint8)


def _save(tmp_path, name, img):
    PIL = pytest.importorskip("PIL.Image")
    p = tmp_path / (name.replace("/", "_") + ".png")
    arr = img if img.ndim == 3 else np.repeat(img[:, :, None], 3, axis=2)
    PIL.fromarray(arr).save(p)
    return p


def test_polyfisheye_unproject_roundtrip():
    cam = CameraPose(
        name="f", width=3040, height=3040, fx=1480.0, fy=1480.0,
        cx=1520.0, cy=1520.0, k1=0.0, rotation=np.eye(3),
        translation=np.zeros(3), model="POLYFISHEYE",
        poly=(-0.05, 0.01, -0.001, 0.0, 0.0, 0.0),
    )
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.05, np.deg2rad(55), 200)
    phi = rng.uniform(0, 2 * np.pi, 200)
    rays = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    px, py = cam.project(rays)
    back = cam.unproject(px, py)
    align = (rays * back).sum(axis=1)
    assert float(align.min()) > 0.99999


def _two_plane_photo(cam, blur=1):
    """Photo of a far checker plane (z=2) with a NEARER center patch
    (z=1.2) — the parallax between the planes makes the pose fully
    observable (a single plane leaves translation/rotation degenerate)."""
    from scipy.ndimage import uniform_filter

    xs, ys = np.meshgrid(
        np.arange(cam.width, dtype=np.float64),
        np.arange(cam.height, dtype=np.float64),
    )
    rays = cam.unproject(xs.ravel(), ys.ravel())
    c = -cam.rotation.T @ cam.translation
    dirs = rays @ cam.rotation
    t_far = (2.0 - c[2]) / dirs[:, 2]
    wxf = c[0] + t_far * dirs[:, 0]
    wyf = c[1] + t_far * dirs[:, 1]
    img = 40.0 + 180.0 * _checker(wxf, wyf, 0.25)
    t_nr = (1.2 - c[2]) / dirs[:, 2]
    wxn = c[0] + t_nr * dirs[:, 0]
    wyn = c[1] + t_nr * dirs[:, 1]
    near = (np.abs(wxn) < 0.3) & (np.abs(wyn) < 0.3)
    img[near] = 40.0 + 180.0 * _checker(wxn[near], wyn[near], 0.15)
    img = img.reshape(cam.height, cam.width)
    if blur:
        img = uniform_filter(img, 3)
    return img.astype(np.uint8)


def _two_plane_points():
    g = np.arange(-0.9, 0.9, 0.03)
    xx, yy = np.meshgrid(g, g)
    far = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 2.0)])
    # Points occluded by the near patch (seen from the origin) are cut.
    occl = (np.abs(far[:, 0]) < 0.5) & (np.abs(far[:, 1]) < 0.5)
    far = far[~occl]
    lum_f = 40.0 + 180.0 * _checker(far[:, 0], far[:, 1], 0.25)
    gn = np.arange(-0.29, 0.29, 0.015)
    nx, ny = np.meshgrid(gn, gn)
    near = np.column_stack(
        [nx.ravel(), ny.ravel(), np.full(nx.size, 1.2)]
    )
    lum_n = 40.0 + 180.0 * _checker(near[:, 0], near[:, 1], 0.15)
    pts = np.vstack([far, near])
    lum = np.concatenate([lum_f, lum_n])
    return pts, np.repeat(lum[:, None], 3, axis=1).astype(np.uint8)


def test_pose_refinement_recovers_perturbation(tmp_path):
    true_cam = _cam("left/p.jpg", (0.0, 0.0, 0.0))
    photo = _two_plane_photo(true_cam)
    path = _save(tmp_path, "left/p.jpg", photo)
    pts, cols = _two_plane_points()

    from scantobim.photogrammetry.posefine import (
        _rodrigues,
        refine_camera_poses,
    )

    bad = _cam("left/p.jpg", (0.0, 0.0, 0.0))
    r_p = _rodrigues(np.array([0.0, 0.006, 0.0]))
    bad.rotation = r_p @ bad.rotation
    bad.translation = r_p @ bad.translation + np.array([0.02, 0.0, 0.0])

    def _reproj_err(cam):
        p_true = np.stack(true_cam.project(pts), axis=1)
        p_cam = np.stack(
            cam.project(pts @ cam.rotation.T + cam.translation), axis=1
        )
        return float(np.linalg.norm(p_cam - p_true, axis=1).mean())

    err_before = _reproj_err(bad)
    stats: dict = {}
    n = refine_camera_poses(
        [bad], tmp_path, pts, cols,
        image_map={"left/p.jpg": path}, scale=1.0, stats_out=stats,
    )
    assert n == 1
    # What MVS needs is REPROJECTION accuracy — the photometric fit must
    # cut the pixel error several-fold (center distance alone is a poor
    # metric: rotation and translation trade off along the view axis).
    err_after = _reproj_err(bad)
    assert err_after < err_before * 0.4
    assert stats["posen_feinschliff"]["kameras_nachgefuehrt"] == 1


def test_mvs_recovers_true_depth_against_offset_prior(tmp_path):
    left = _cam("left/a.jpg", (0.0, 0.0, 0.0))
    right = _cam("right/a.jpg", (0.2, 0.0, 0.0))
    p_l = _save(tmp_path, "left/a.jpg", _plane_photo(left, 2.0, 0.25))
    p_r = _save(tmp_path, "right/a.jpg", _plane_photo(right, 2.0, 0.25))

    # LiDAR prior deliberately 3 cm off the photographic truth.
    g = np.arange(-0.9, 0.9, 0.02)
    xx, yy = np.meshgrid(g, g)
    cloud = np.column_stack(
        [xx.ravel(), yy.ravel(), np.full(xx.size, 2.03)]
    )

    from scantobim.core.mvs import mvs_points

    stats: dict = {}
    got = mvs_points(
        [left, right], tmp_path, cloud,
        image_map={"left/a.jpg": p_l, "right/a.jpg": p_r},
        stats_out=stats, scale=1.0, band=0.08, steps=33,
    )
    assert got is not None
    pts, cols = got
    assert len(pts) > 100
    med_z = float(np.median(pts[:, 2]))
    assert abs(med_z - 2.0) < 0.008  # photos win over the 3 cm-off prior
    assert stats["mvs"]["punkte"] == len(pts)


def test_photorefine_pulls_spike_onto_photo_surface(tmp_path):
    from scantobim.core.mesh import Mesh

    fx, w, h = 800.0, 800, 600
    cam_a = _cam("left/r.jpg", (-0.5, 0.0, -2.0), fx=fx, w=w, h=h)
    cam_b = _cam("right/r.jpg", (0.5, 0.0, -2.0), fx=fx, w=w, h=h)
    p_a = _save(tmp_path, "left/r.jpg", _plane_photo(cam_a, 0.0, 0.02, blur=0))
    p_b = _save(tmp_path, "right/r.jpg", _plane_photo(cam_b, 0.0, 0.02, blur=0))

    n = 41
    g = np.linspace(-0.4, 0.4, n)
    xx, yy = np.meshgrid(g, g)
    verts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    spike = (n // 2) * n + n // 2
    verts[spike, 2] = 0.008
    idx = lambda i, j: i * n + j  # noqa: E731
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            faces.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
            faces.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])
    mesh = Mesh(vertices=verts, faces=np.asarray(faces, dtype=np.int64))

    from scantobim.core.photorefine import refine_mesh_photoconsistent

    stats: dict = {}
    moved = refine_mesh_photoconsistent(
        mesh, [cam_a, cam_b], tmp_path,
        image_map={"left/r.jpg": p_a, "right/r.jpg": p_b},
        stats_out=stats, scale=1.0,
    )
    assert moved > 0
    assert abs(float(mesh.vertices[spike, 2])) < 0.004


def test_render_feedback_flags_wrong_region(tmp_path):
    from scantobim.core.mesh import Mesh
    from scantobim.core.renderloop import hot_face_boxes, render_feedback

    n = 121
    g = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(g, g)
    verts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    idx = lambda i, j: i * n + j  # noqa: E731
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            faces.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
            faces.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])
    faces = np.asarray(faces, dtype=np.int64)
    colors = np.full((len(verts), 3), 30, dtype=np.uint8)
    colors[verts[:, 0] > 0] = 220  # model: right half bright
    mesh = Mesh(vertices=verts, faces=faces, vertex_colors=colors)

    cam_a = _cam("left/v.jpg", (-0.05, 0.0, -2.0))
    cam_b = _cam("right/v.jpg", (0.05, 0.0, -2.0))

    def _dark_photo(cam):
        img = np.full((cam.height, cam.width), 30, dtype=np.uint8)
        return img  # photo: EVERYTHING dark -> right half of model wrong

    p_a = _save(tmp_path, "left/v.jpg", _dark_photo(cam_a))
    p_b = _save(tmp_path, "right/v.jpg", _dark_photo(cam_b))

    stats: dict = {}
    fb = render_feedback(
        mesh, [cam_a, cam_b], tmp_path,
        image_map={"left/v.jpg": p_a, "right/v.jpg": p_b},
        stats_out=stats, scale=0.25, tile=8,
    )
    assert fb is not None
    heat, scores = fb
    assert "score_median" in stats
    cent_x = mesh.vertices[mesh.faces].mean(axis=1)[:, 0]
    left_heat = np.nanmean(heat[cent_x < -0.1])
    right_heat = np.nanmean(heat[cent_x > 0.1])
    assert right_heat > left_heat + 0.1
    boxes = hot_face_boxes(mesh, heat, heat_min=(left_heat + right_heat) / 2)
    assert boxes
    los = np.array([b[0] for b in boxes])
    assert float(los[:, 0].min()) > -0.3  # regions sit on the right side
