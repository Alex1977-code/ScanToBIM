"""Foto-Verifikations-Culling: Geisterlappen sterben, echte Flaechen bleiben."""

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


def _world_noise(wx, wy, cell=0.01):
    ix = np.floor(wx / cell)
    iy = np.floor(wy / cell)
    v = np.sin(ix * 12.9898 + iy * 78.233) * 43758.5453
    return (np.abs(v - np.floor(v)) * 255.0)


def _noise_photo(cam, plane_z):
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
    img = _world_noise(wx, wy).reshape(cam.height, cam.width)
    return uniform_filter(img, 2).astype(np.uint8)


def _save(tmp_path, name, img):
    PIL = pytest.importorskip("PIL.Image")
    p = tmp_path / (name.replace("/", "_") + ".png")
    arr = np.repeat(img[:, :, None], 3, axis=2)
    PIL.fromarray(arr).save(p)
    return p


def _grid(x0, x1, y0, y1, z, n):
    g = np.linspace(x0, x1, n)
    h = np.linspace(y0, y1, n)
    xx, yy = np.meshgrid(g, h)
    verts = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])
    idx = lambda i, j: i * n + j  # noqa: E731
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            faces.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
            faces.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])
    return verts, np.asarray(faces, dtype=np.int64)


def test_ghost_flap_is_culled_real_plane_survives(tmp_path):
    from scantobim.core.mesh import Mesh
    from scantobim.core.photocull import cull_ghost_faces

    cams = [
        _cam("left/a.jpg", (0.0, 0.0, 0.0)),
        _cam("right/a.jpg", (0.2, 0.0, 0.0)),
        _cam("left/b.jpg", (0.05, 0.12, 0.0)),
        _cam("right/b.jpg", (0.25, 0.12, 0.0)),
    ]
    imap = {}
    for cam in cams:
        imap[cam.name] = _save(tmp_path, cam.name, _noise_photo(cam, 2.0))

    pv, pf = _grid(-0.9, 0.9, -0.7, 0.7, 2.0, 41)   # echte Ebene
    fv, ff = _grid(-0.14, 0.14, -0.12, 0.12, 1.91, 8)  # Geisterlappen 9 cm davor
    verts = np.vstack([pv, fv])
    faces = np.vstack([pf, ff + len(pv)])
    n_plane = len(pf)
    mesh = Mesh(vertices=verts, faces=faces)

    stats: dict = {}
    n = cull_ghost_faces(
        mesh, cams, tmp_path, image_map=imap, stats_out=stats,
        max_views=4, scale=1.0,
    )
    assert n > 0
    # Der Lappen (Dreiecke mit Vertices >= len(pv)) muss praktisch weg sein.
    flap_left = int((mesh.faces >= len(pv)).any(axis=1).sum())
    assert flap_left <= len(ff) * 0.1
    # Die echte Ebene bleibt nahezu vollstaendig.
    plane_left = int((mesh.faces < len(pv)).all(axis=1).sum())
    assert plane_left >= n_plane * 0.97
    assert stats["foto_verifikation"]["entfernt"] == n
