"""Texturing: PNG encoder, cloud-color baking, photo projection with occlusion."""

import numpy as np
import pytest

from scantobim import PipelineConfig, reconstruct
from scantobim.core.cloud import PointCloud
from scantobim.core.texture import bake_texture_from_cloud, bake_texture_from_photos
from scantobim.io.png import encode_png
from scantobim.io.writers import write_mesh
from tests.synthetic import make_box_scan

BOX_SIZE = (4.0, 3.0, 2.5)


def _colored_box_cloud():
    """Box scan where each point's color encodes its dominant axis plane."""
    cloud = make_box_scan(size=BOX_SIZE, density=900, noise=0.003)
    pts = cloud.points
    colors = np.zeros((len(pts), 3), dtype=np.uint8)
    sx, sy, sz = BOX_SIZE
    # walls x≈0 red, x≈sx green, y≈0 blue, y≈sy yellow, floor/ceiling gray
    colors[:] = (120, 120, 120)
    colors[np.abs(pts[:, 0]) < 0.05] = (200, 40, 40)
    colors[np.abs(pts[:, 0] - sx) < 0.05] = (40, 200, 40)
    colors[np.abs(pts[:, 1]) < 0.05] = (40, 40, 200)
    colors[np.abs(pts[:, 1] - sy) < 0.05] = (220, 220, 60)
    cloud.colors = colors
    return cloud


def test_png_roundtrip(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    raw = encode_png(img)
    assert raw.startswith(b"\x89PNG")
    path = tmp_path / "test.png"
    path.write_bytes(raw)
    back = np.asarray(PIL.open(path).convert("RGB"))
    np.testing.assert_array_equal(back, img)


@pytest.fixture(scope="module")
def box_setup():
    cloud = _colored_box_cloud()
    result = reconstruct(cloud, PipelineConfig.preset("building"))
    return cloud, result


def _sample_texture_at(mesh, world, plane_tol=0.03):
    """Texture color at a 3D point: find the containing face, interpolate UVs."""
    th, tw = mesh.texture.shape[:2]
    for tri in mesh.faces:
        bary = _barycentric(world, mesh.vertices[tri], plane_tol)
        if bary is None:
            continue
        uv = (mesh.uvs[tri] * bary[:, None]).sum(axis=0)
        return mesh.texture[
            min(int(uv[1] * th), th - 1), min(int(uv[0] * tw), tw - 1)
        ]
    return None


def _barycentric(p, tri, plane_tol=0.03):
    v0, v1 = tri[1] - tri[0], tri[2] - tri[0]
    n = np.cross(v0, v1)
    norm = np.linalg.norm(n)
    if norm < 1e-14:
        return None
    v2 = p - tri[0]
    if abs(v2 @ (n / norm)) > plane_tol:
        return None  # too far from the triangle's plane
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-14:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    if u < -1e-6 or v < -1e-6 or w < -1e-6:
        return None
    return np.clip(np.array([u, v, w]), 0, 1)


def test_bake_from_cloud(box_setup):
    cloud, result = box_setup
    mesh = bake_texture_from_cloud(result, cloud)
    assert mesh.texture is not None and mesh.uvs is not None
    assert mesh.uvs.shape == (len(mesh.vertices), 2)
    assert np.all(mesh.uvs >= 0) and np.all(mesh.uvs <= 1)
    h, w = mesh.texture.shape[:2]
    assert (h & (h - 1)) == 0 and (w & (w - 1)) == 0  # power of two

    # Sample the texture at each wall's center: color must match the paint.
    sx, sy, sz = BOX_SIZE
    checks = [
        (np.array([0.0, sy / 2, sz / 2]), (200, 40, 40)),
        (np.array([sx, sy / 2, sz / 2]), (40, 200, 40)),
        (np.array([sx / 2, 0.0, sz / 2]), (40, 40, 200)),
        (np.array([sx / 2, sy, sz / 2]), (220, 220, 60)),
    ]
    for target, expected in checks:
        got = _sample_texture_at(mesh, target)
        assert got is not None, f"no face found at {target}"
        assert np.abs(got.astype(int) - expected).max() < 60, (
            f"wall {target}: expected ~{expected}, got {tuple(got)}"
        )


def test_bake_requires_colors():
    cloud = make_box_scan(density=400, noise=0.004)
    result = reconstruct(cloud, PipelineConfig.preset("fast"))
    with pytest.raises(ValueError, match="no colors"):
        bake_texture_from_cloud(result, cloud)


def test_textured_glb_and_obj_and_html(box_setup, tmp_path):
    cloud, result = box_setup
    mesh = bake_texture_from_cloud(result, cloud)

    glb = write_mesh(mesh, tmp_path / "model.glb")
    import json as _json
    import struct

    raw = glb.read_bytes()
    json_len = struct.unpack("<I", raw[12:16])[0]
    gltf = _json.loads(raw[20 : 20 + json_len])
    prim = gltf["meshes"][0]["primitives"][0]
    assert "TEXCOORD_0" in prim["attributes"]
    assert gltf["images"][0]["mimeType"] == "image/png"
    assert gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0

    obj = write_mesh(mesh, tmp_path / "model.obj")
    text = obj.read_text()
    assert "mtllib model.mtl" in text and "vt " in text and "usemtl scan_texture" in text
    assert (tmp_path / "model.mtl").exists()
    assert (tmp_path / "model_texture.png").stat().st_size > 100

    html = write_mesh(mesh, tmp_path / "model.html")
    content = html.read_text()
    assert "data:image/png;base64," in content
    assert '"textured": true' in content


# --------------------------------------------------------------- photo projection

def _make_camera_scene(tmp_path):
    """A floor plane scan + one synthetic overhead camera with a checkerboard
    photo whose projection onto the floor is analytically known."""
    PIL = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(0)
    from tests.synthetic import sample_rect

    x, y = np.eye(3)[0], np.eye(3)[1]
    floor = sample_rect((0, 0, 0), x, y, 4.0, 4.0, 2500, 0.002, rng)
    # a second small wall so plane detection has 2 planes (more realistic)
    wall = sample_rect((0, 0, 0), x, np.eye(3)[2], 4.0, 1.0, 2500, 0.002, rng)
    cloud = PointCloud(points=np.vstack([floor, wall]))
    result = reconstruct(cloud, PipelineConfig.preset("building"))

    # Camera 5 m above the floor center, looking straight down.
    # COLMAP convention: p_cam = R p_world + t; camera looks along +z_cam.
    # Looking down: z_cam = -z_world → R = diag(1, -1, -1).
    rotation = np.diag([1.0, -1.0, -1.0])
    center_world = np.array([2.0, 2.0, 5.0])
    translation = -rotation @ center_world
    w, h, f = 400, 400, 300.0

    # Checkerboard image: 40px squares.
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    board = (((xx // 40) + (yy // 40)) % 2).astype(np.uint8)
    img = np.stack([board * 255, board * 255, np.full_like(board, 128) * 0 + 60], axis=-1)

    images_dir = tmp_path / "photos"
    images_dir.mkdir()
    PIL.fromarray(img).save(images_dir / "cam0.png")

    model_dir = tmp_path / "model_txt"
    model_dir.mkdir()
    (model_dir / "cameras.txt").write_text(
        f"# cameras\n1 PINHOLE {w} {h} {f} {f} {w/2} {h/2}\n"
    )
    q = _rot_to_quat(rotation)
    (model_dir / "images.txt").write_text(
        "# images\n"
        f"1 {q[0]} {q[1]} {q[2]} {q[3]} "
        f"{translation[0]} {translation[1]} {translation[2]} 1 cam0.png\n"
        "\n"
    )
    return result, model_dir, images_dir, img, rotation, translation, f, w, h


def _rot_to_quat(m):
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1.0) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (m[k, j] - m[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (m[j, i] + m[i, j]) / s
    q[k + 1] = (m[k, i] + m[i, k]) / s
    return q


def test_photo_projection(tmp_path):
    result, model_dir, images_dir, img, rotation, translation, f, w, h = (
        _make_camera_scene(tmp_path)
    )
    mesh = bake_texture_from_photos(
        result, model_dir, images_dir, texel_size=0.02
    )
    assert mesh.texture is not None

    # Verify: pick points on the floor, project into the camera analytically,
    # and compare the photo pixel with the baked texture there.
    floor_surface = None
    for s in result.surfaces:
        if abs(float(s.normal @ [0, 0, 1])) > 0.99:
            floor_surface = s
            break
    assert floor_surface is not None

    checks = 0
    for world in [np.array([1.0, 1.0, 0.0]), np.array([2.5, 1.7, 0.0]), np.array([3.0, 3.0, 0.0])]:
        pc = rotation @ world + translation
        px = int(round(f * pc[0] / pc[2] + w / 2))
        py = int(round(f * pc[1] / pc[2] + h / 2))
        expected = img[py, px]
        got = _sample_texture_at(mesh, world)
        if got is None:
            continue
        assert np.abs(got.astype(int) - expected.astype(int)).max() < 70
        checks += 1
    assert checks >= 2


def test_occlusion_ray():
    from scantobim.core.texture import _occluded

    cam = np.array([0.0, 0.0, 5.0])
    pts = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    # a triangle blocking the first ray at z=2
    blocker = np.array([[[-1, -1, 2.0], [1, -1, 2.0], [0, 2, 2.0]]])
    blocked = _occluded(pts, cam, blocker)
    assert blocked[0] and not blocked[1]
