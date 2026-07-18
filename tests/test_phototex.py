"""Photo texture atlas on the complete mesh (view-dependent texture mapping)."""

import struct

import numpy as np
import pytest

from scantobim.core.mesh import Mesh


def _floor_mesh(n=30, half=1.0):
    """Regular triangulated floor patch around the origin at z=0."""
    g = np.linspace(-half, half, n)
    xx, yy = np.meshgrid(g, g)
    verts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    idx = lambda i, j: i * n + j  # noqa: E731
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            faces.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
            faces.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])
    return Mesh(vertices=verts, faces=np.array(faces))


def _write_binary_model(model_dir, poses_spec):
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "cameras.bin", "wb") as fh:
        fh.write(struct.pack("<Q", 1))
        fh.write(struct.pack("<iiQQ", 1, 1, 400, 300))  # PINHOLE 400x300
        fh.write(struct.pack("<4d", 350.0, 350.0, 200.0, 150.0))
    with open(model_dir / "images.bin", "wb") as fh:
        fh.write(struct.pack("<Q", len(poses_spec)))
        for i, (name, q, t) in enumerate(poses_spec):
            fh.write(struct.pack("<idddddddi", i + 1, *q, *t, 1))
            fh.write(name.encode() + b"\x00")
            fh.write(struct.pack("<Q", 0))


def _camera_setup(tmp_path, color=(200, 40, 90)):
    PIL = pytest.importorskip("PIL.Image")
    model = tmp_path / "sparse"
    _write_binary_model(model, [("foto.jpg", (1.0, 0, 0, 0), (0.0, 0.0, 5.0))])
    img_dir = tmp_path / "bilder"
    img_dir.mkdir(exist_ok=True)
    PIL.new("RGB", (400, 300), color).save(img_dir / "foto.jpg")
    return model, img_dir


def test_bake_photo_atlas_samples_photo(tmp_path):
    """Texels of a camera-facing floor take the photo's color."""
    from scantobim.core.phototex import bake_photo_atlas

    mesh = _floor_mesh()
    mesh.vertex_colors = np.full((len(mesh.vertices), 3), 30, dtype=np.uint8)
    model, img_dir = _camera_setup(tmp_path)

    stats = {}
    textured = bake_photo_atlas(
        mesh, model, img_dir, max_atlas=512, stats_out=stats
    )
    assert textured is not None
    assert textured.texture is not None and textured.uvs is not None
    assert len(textured.faces) == len(mesh.faces)
    assert np.all(textured.uvs >= 0) and np.all(textured.uvs <= 1)
    assert stats["photo_fraction"] > 0.5
    assert stats["cameras_used"] == 1
    assert stats["coverage"] > 0.3

    # Sample the atlas at the face-center UVs: photo color survives JPEG.
    h, w = textured.texture.shape[:2]
    uv_faces = textured.uvs[textured.faces].mean(axis=1)
    px = np.clip((uv_faces[:, 0] * w).astype(int), 0, w - 1)
    py = np.clip((uv_faces[:, 1] * h).astype(int), 0, h - 1)
    med = np.median(textured.texture[py, px], axis=0)
    assert np.all(np.abs(med - [200, 40, 90]) < 15)


def test_bake_photo_atlas_charts_cover_cube(tmp_path):
    """A cube unwraps into ≥6 charts, every face keeps valid UVs; faces the
    camera cannot see fall back to the vertex colors."""
    from scantobim.core.phototex import bake_photo_atlas

    # Cube out of 12 triangles.
    v = np.array(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float
    )
    f = np.array([
        [0, 1, 3], [0, 3, 2],  # x = 0
        [4, 6, 7], [4, 7, 5],  # x = 1
        [0, 4, 5], [0, 5, 1],  # y = 0
        [2, 3, 7], [2, 7, 6],  # y = 1
        [0, 2, 6], [0, 6, 4],  # z = 0
        [1, 5, 7], [1, 7, 3],  # z = 1
    ])
    mesh = Mesh(vertices=v, faces=f)
    base = np.full((8, 3), 90, dtype=np.uint8)
    mesh.vertex_colors = base
    model, img_dir = _camera_setup(tmp_path, color=(10, 250, 10))

    stats = {}
    textured = bake_photo_atlas(
        mesh, model, img_dir, max_atlas=256, stats_out=stats
    )
    assert textured is not None
    assert stats["charts"] >= 6
    assert np.all(textured.uvs >= 0) and np.all(textured.uvs <= 1)
    # Chart seams duplicate vertices.
    assert len(textured.vertices) > len(mesh.vertices)
    # The camera under the floor sees only the z=0 face — the atlas must
    # contain both photo-green and vertex-grey regions.
    tex = textured.texture.reshape(-1, 3)
    greenish = (tex[:, 1] > 180) & (tex[:, 0] < 100)
    grey = np.all(np.abs(tex.astype(int) - 90) < 20, axis=1)
    assert greenish.any()
    assert grey.any()


def test_bake_photo_atlas_without_images_returns_none(tmp_path):
    from scantobim.core.phototex import bake_photo_atlas

    mesh = _floor_mesh(n=5)
    model = tmp_path / "sparse"
    _write_binary_model(model, [("fehlt.jpg", (1.0, 0, 0, 0), (0.0, 0.0, 5.0))])
    empty = tmp_path / "leer"
    empty.mkdir()
    assert bake_photo_atlas(mesh, model, empty) is None


def test_textured_freeform_in_viewer_and_glb(tmp_path):
    """The photo-textured complete mesh flows into HTML viewer and GLB."""
    pytest.importorskip("PIL.Image")
    from scantobim.core.phototex import bake_photo_atlas
    from scantobim.io.writers import write_mesh

    mesh = _floor_mesh()
    mesh.vertex_colors = np.full((len(mesh.vertices), 3), 30, dtype=np.uint8)
    model, img_dir = _camera_setup(tmp_path)
    textured = bake_photo_atlas(mesh, model, img_dir, max_atlas=512)
    assert textured is not None

    html = tmp_path / "modell.html"
    empty = Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    write_mesh(empty, html, freeform=textured, freeform_label="Komplett-Mesh (Scan)")
    text = html.read_text(encoding="utf-8")
    assert '"ff_textured": true' in text or '"ff_textured":true' in text
    assert "data:image/" in text  # embedded atlas

    glb = tmp_path / "modell_foto.glb"
    write_mesh(textured, glb)
    raw = glb.read_bytes()
    assert raw[:4] == b"glTF"
    assert b"TEXCOORD_0" in raw
    assert b"image/jpeg" in raw or b"image/png" in raw


def test_encode_texture_jpeg_for_large_png_for_small():
    from scantobim.io.teximg import encode_texture

    small = np.zeros((16, 16, 3), dtype=np.uint8)
    data, mime = encode_texture(small)
    assert mime == "image/png"

    pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(0)
    big = rng.integers(0, 255, (1024, 1024, 3)).astype(np.uint8)
    data, mime = encode_texture(big)
    assert mime == "image/jpeg"
    assert len(data) > 0


def test_charts_stay_coarse_on_bumpy_mesh():
    """Normal smoothing + mini-chart merge: no chart explosion on noise."""
    from scantobim.core.cloud import PointCloud
    from scantobim.core.freeform import freeform_mesh_from_points
    from scantobim.core.phototex import _build_charts

    rng = np.random.default_rng(0)
    u = rng.random(120_000)[:, None] * 4.0
    v = rng.random(120_000)[:, None] * 3.0
    pts = (
        np.hstack([u, v, np.zeros_like(u)])
        + rng.normal(0, 0.015, (120_000, 3))  # bumpy like a real scan
    )
    mesh = freeform_mesh_from_points(PointCloud(points=pts), voxel=0.03)
    assert mesh is not None
    chart_of_face, chart_axes = _build_charts(mesh)
    # Old behavior fragmented this into thousands of charts; the fix keeps
    # a handful of large ones.
    assert len(chart_axes) < max(20, len(mesh.faces) // 500)
    sizes = np.bincount(chart_of_face)
    assert sizes.max() > len(mesh.faces) * 0.5  # one dominant chart


def test_viewer_detail_layer_default(tmp_path):
    """The photorealistic detail layer is embedded and is the default view."""
    from scantobim.core.mesh import Mesh
    from scantobim.io.html_viewer import write_html_viewer

    base = Mesh(
        vertices=np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
    )
    detail = Mesh(
        vertices=np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        uvs=np.array([[0.0, 0], [1, 0], [1, 1], [0, 1]]),
        texture=np.full((8, 8, 3), 90, dtype=np.uint8),
    )
    out = write_html_viewer(
        base, tmp_path / "v.html", detail=detail,
        detail_label="Detail-Mesh Gebäude (fotorealistisch)",
    )
    text = out.read_text(encoding="utf-8")
    assert "Detail-Mesh Gebäude (fotorealistisch)" in text
    assert '"dt_indices": 6' in text and '"dt_textured": true' in text
    assert "data:image/" in text  # embedded detail texture
