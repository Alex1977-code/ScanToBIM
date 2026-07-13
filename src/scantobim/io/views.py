"""True-to-scale orthographic views (Fassadenansichten, Draufsicht).

Renders the reconstructed (optionally photo-textured) model orthographically
from the four compass directions and from above — the classic 2D deliverables
of an as-built survey, pixel-accurate to scale:

* ``ansicht_nord/ost/sued/west.png`` + ``draufsicht.png``
* one **world file** (``.pgw``) per view, so CAD/GIS tools import the image
  at the correct scale and position

Pure numpy z-buffer rasterizer — no GPU, no external renderer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scantobim.core.mesh import Mesh
from scantobim.io.png import encode_png

# name → (right, up, depth) unit vectors; depth points TOWARDS the viewer.
_VIEWS = {
    "draufsicht": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "ansicht_sued": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),  # seen from south
    "ansicht_nord": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),  # seen from north
    "ansicht_ost": ((0, -1, 0), (0, 0, 1), (1, 0, 0)),  # seen from east
    "ansicht_west": ((0, 1, 0), (0, 0, 1), (-1, 0, 0)),  # seen from west
}

_BACKGROUND = 248


def render_ortho_view(
    mesh: Mesh,
    right: np.ndarray,
    up: np.ndarray,
    depth: np.ndarray,
    pixels_per_unit: float,
    padding: float = 0.05,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Rasterize an orthographic view; returns ``(image, world_top_left)``.

    ``world_top_left`` is the (right, up) coordinate of the top-left pixel
    center — the anchor for the world file.
    """
    right = np.asarray(right, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)

    sx = mesh.vertices @ right
    sy = mesh.vertices @ up
    sz = mesh.vertices @ depth
    pad = padding * max(float(np.ptp(sx)), float(np.ptp(sy)), 1e-9)
    x0, x1 = sx.min() - pad, sx.max() + pad
    y0, y1 = sy.min() - pad, sy.max() + pad
    width = max(int(np.ceil((x1 - x0) * pixels_per_unit)), 8)
    height = max(int(np.ceil((y1 - y0) * pixels_per_unit)), 8)

    img = np.full((height, width, 3), _BACKGROUND, dtype=np.uint8)
    zbuf = np.full((height, width), -np.inf)

    # Pixel coordinates of vertices (y down).
    px = (sx - x0) * pixels_per_unit
    py = (y1 - sy) * pixels_per_unit

    has_texture = mesh.uvs is not None and mesh.texture is not None
    tex = mesh.texture
    colors = mesh.vertex_colors

    for f_idx, face in enumerate(mesh.faces):
        xs, ys, zs = px[face], py[face], sz[face]
        min_x = max(int(np.floor(xs.min())), 0)
        max_x = min(int(np.ceil(xs.max())), width - 1)
        min_y = max(int(np.floor(ys.min())), 0)
        max_y = min(int(np.ceil(ys.max())), height - 1)
        if min_x > max_x or min_y > max_y:
            continue
        gx, gy = np.meshgrid(
            np.arange(min_x, max_x + 1) + 0.5, np.arange(min_y, max_y + 1) + 0.5
        )
        # Barycentric coordinates.
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(d) < 1e-12:
            continue
        w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d
        w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        z = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
        rows = gy.astype(int)
        cols = gx.astype(int)
        closer = inside & (z > zbuf[rows, cols])
        if not closer.any():
            continue
        rr, cc = rows[closer], cols[closer]
        zbuf[rr, cc] = z[closer]
        b0, b1, b2 = w0[closer], w1[closer], w2[closer]
        if has_texture:
            uv = (
                b0[:, None] * mesh.uvs[face[0]]
                + b1[:, None] * mesh.uvs[face[1]]
                + b2[:, None] * mesh.uvs[face[2]]
            )
            th, tw = tex.shape[:2]
            tx = np.clip((uv[:, 0] * tw).astype(int), 0, tw - 1)
            ty = np.clip((uv[:, 1] * th).astype(int), 0, th - 1)
            img[rr, cc] = tex[ty, tx]
        elif colors is not None:
            col = (
                b0[:, None] * colors[face[0]]
                + b1[:, None] * colors[face[1]]
                + b2[:, None] * colors[face[2]]
            )
            img[rr, cc] = np.clip(col, 0, 255).astype(np.uint8)
        else:
            img[rr, cc] = 190

    return img, (x0 + 0.5 / pixels_per_unit, y1 - 0.5 / pixels_per_unit)


def write_ortho_views(
    mesh: Mesh,
    out_dir: str | Path,
    pixels_per_unit: float | None = None,
    max_size: int = 2000,
) -> list[Path]:
    """Render all five standard views into ``out_dir`` (PNG + world file)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if pixels_per_unit is None:
        extent = float(
            np.max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
        )
        pixels_per_unit = max_size / max(extent * 1.1, 1e-9)

    written: list[Path] = []
    for name, (right, up, depth) in _VIEWS.items():
        img, (wx, wy) = render_ortho_view(
            mesh, np.array(right, float), np.array(up, float),
            np.array(depth, float), pixels_per_unit,
        )
        png_path = out_dir / f"{name}.png"
        png_path.write_bytes(encode_png(img))
        # ESRI world file: pixel size + world coordinate of the top-left
        # pixel center (in the view plane's right/up coordinates).
        unit = 1.0 / pixels_per_unit
        (out_dir / f"{name}.pgw").write_text(
            f"{unit:.8f}\n0.0\n0.0\n{-unit:.8f}\n{wx:.6f}\n{wy:.6f}\n"
        )
        written.append(png_path)
    return written
