"""Camera self-check (stereo collisions) + photo-based edge refinement."""

import numpy as np
import pytest

from scantobim.core.cloud import PointCloud
from scantobim.photogrammetry.colmap import CameraPose


def _pinhole(name, pos, look_z=True, f=400.0, w=400, h=400):
    rot = np.diag([1.0, -1.0, -1.0]) if look_z else np.eye(3)
    return CameraPose(
        name=name, width=w, height=h, fx=f, fy=f, cx=w / 2, cy=h / 2,
        k1=0.0, rotation=rot, translation=-rot @ np.asarray(pos, float),
    )


def test_validate_cameras_resolves_stereo_basename_collision(tmp_path):
    """left/cam.png and right/cam.png collide — the check picks the right
    file per camera and reports the resolved ambiguity."""
    PIL = pytest.importorskip("PIL.Image")
    from scantobim.photogrammetry.camcheck import validate_cameras

    rng = np.random.default_rng(0)
    pts = np.column_stack([
        rng.random(4000) * 2 - 1, rng.random(4000) * 2 - 1, np.zeros(4000)
    ])
    colors = np.full((len(pts), 3), (200, 40, 40), dtype=np.uint8)  # red floor
    cloud = PointCloud(points=pts, colors=colors)

    imgdir = tmp_path / "left"
    imgdir.mkdir()
    good = np.zeros((400, 400, 3), np.uint8)
    good[:] = (200, 40, 40)  # matches the cloud
    PIL.fromarray(good).save(imgdir / "cam.png")
    wrong = np.zeros((400, 400, 3), np.uint8)
    wrong[:] = (30, 90, 220)  # sky-blue impostor with the SAME basename
    (tmp_path / "right").mkdir()
    PIL.fromarray(wrong).save(tmp_path / "right" / "cam.png")

    cam = _pinhole("cam.png", (0.0, 0.0, 3.0))  # looks straight down
    kept, image_map = validate_cameras([cam], imgdir, cloud)
    assert len(kept) == 1 and image_map is not None
    assert image_map["cam.png"] == imgdir / "cam.png"  # correct side won

    # A camera whose photo matches nothing gets dropped.
    bad = _pinhole("bad.png", (0.0, 0.0, 3.0))
    PIL.fromarray(wrong).save(imgdir / "bad.png")
    stats: dict = {}
    kept, image_map = validate_cameras([cam, bad], imgdir, cloud,
                                       stats_out=stats)
    assert [c.name for c in kept] == ["cam.png"]
    assert stats["validiert"] == 1 and stats["gesamt"] == 2


def test_refine_detail_edges_moves_edge_toward_photo(tmp_path):
    """A photo edge shifted +10 mm pulls the mesh edge band toward it."""
    PIL = pytest.importorskip("PIL.Image")
    from scantobim.core.edgerefine import refine_detail_edges
    from scantobim.core.mesh import Mesh
    from scantobim.core.pipeline import SurfaceGeometry

    # Two planes: floor z=0 and wall x=0 → edge along y at x=0, z=0.
    floor = SurfaceGeometry(
        plane_index=0, normal=np.array([0.0, 0, 1]),
        outer=np.array([[0.0, -1, 0], [1.5, -1, 0], [1.5, 1, 0], [0.0, 1, 0]]),
        holes=[], surface_class="slab",
    )
    wall = SurfaceGeometry(
        plane_index=1, normal=np.array([1.0, 0, 0]),
        outer=np.array([[0.0, -1, 0], [0.0, 1, 0], [0.0, 1, 1.5], [0.0, -1, 1.5]]),
        holes=[], surface_class="wall",
    )

    # Detail mesh: a strip of vertices along the edge.
    ys = np.linspace(-0.9, 0.9, 40)
    verts = []
    for y in ys:
        verts += [[0.0, y, 0.0], [0.02, y, 0.0], [0.0, y, 0.02]]
    verts = np.array(verts)
    faces = []
    for i in range(len(ys) - 1):
        a = 3 * i
        b = 3 * (i + 1)
        faces += [[a, a + 1, b], [a + 1, b + 1, b], [a, b, a + 2],
                  [a + 2, b, b + 2]]
    mesh = Mesh(vertices=verts.copy(), faces=np.array(faces))

    # Camera above, looking down; photo: bright where x > 0.01 (the
    # photographic edge sits at x = +10 mm), dark elsewhere.
    f, w, h = 400.0, 500, 500
    cam = _pinhole("edge.png", (0.3, 0.0, 2.0), f=f, w=w, h=h)
    xx = np.arange(w)[None, :].repeat(h, axis=0)
    yy = np.arange(h)[:, None].repeat(w, axis=1)
    # Invert the projection for z=0 plane: x = (px - cx) * z_dist / f + cam_x
    x_world = (xx - w / 2) * 2.0 / f + 0.3
    img = np.where(x_world[..., None] > 0.01, 230, 25).astype(np.uint8)
    img = np.repeat(img, 3, axis=2)
    imgdir = tmp_path
    PIL.fromarray(img).save(imgdir / "edge.png")

    cam2 = _pinhole("edge2.png", (-0.3, 0.1, 2.0), f=f, w=w, h=h)
    x_world2 = (xx - w / 2) * 2.0 / f + (-0.3)
    img2 = np.where(x_world2[..., None] > 0.01, 230, 25).astype(np.uint8)
    img2 = np.repeat(img2, 3, axis=2)
    PIL.fromarray(img2).save(imgdir / "edge2.png")
    cams = [cam, cam2]

    before = mesh.vertices.copy()
    stats: dict = {}
    moved = refine_detail_edges(
        mesh, [floor, wall], cams, imgdir, stats_out=stats
    )
    assert stats["kanten_kandidaten"] >= 1
    assert moved >= 1
    shift = mesh.vertices - before
    # Vertices near the edge moved several millimeters, toward +x.
    mags = np.linalg.norm(shift, axis=1)
    assert mags.max() > 0.004
    assert abs(shift[mags > 1e-6, 0]).mean() > abs(
        shift[mags > 1e-6, 1]
    ).mean()
