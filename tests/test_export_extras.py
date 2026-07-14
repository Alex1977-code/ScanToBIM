"""DXF floor plan export and standalone HTML viewer."""

import base64
import json
import re

import numpy as np
import pytest

from scantobim import PipelineConfig, reconstruct
from scantobim.io.dxf import slice_mesh, write_floorplan_dxf
from scantobim.io.html_viewer import extract_crease_edges, write_html_viewer
from scantobim.io.writers import write_mesh
from tests.synthetic import make_box_scan

BOX_SIZE = (4.0, 3.0, 2.5)


@pytest.fixture(scope="module")
def box_mesh():
    cloud = make_box_scan(size=BOX_SIZE, density=800, noise=0.004)
    return reconstruct(cloud, PipelineConfig.preset("building")).mesh


def test_slice_box(box_mesh):
    segments = slice_mesh(box_mesh, z=1.0)
    assert len(segments) >= 4
    total = sum(np.linalg.norm(s[1] - s[0]) for s in segments)
    perimeter = 2 * (BOX_SIZE[0] + BOX_SIZE[1])
    assert abs(total - perimeter) < 0.3


def test_dxf_floorplan(box_mesh, tmp_path):
    path = write_floorplan_dxf(box_mesh, tmp_path / "plan.dxf", height_above_floor=1.0)
    text = path.read_text()
    assert text.startswith("0\nSECTION")
    assert text.rstrip().endswith("EOF")
    assert "SCHNITT" in text
    # Extract all X coordinates (group code 10/11) and check extents.
    xs = [
        float(v)
        for code, v in re.findall(r"^(10|11)\n([-\d.]+)$", text, flags=re.M)
    ]
    assert abs(min(xs) - 0.0) < 0.05 and abs(max(xs) - BOX_SIZE[0]) < 0.05


def test_dxf_bad_height(box_mesh, tmp_path):
    with pytest.raises(ValueError, match="intersects no geometry"):
        write_floorplan_dxf(box_mesh, tmp_path / "plan.dxf", height_above_floor=99.0)


def test_crease_edges_box(box_mesh):
    creases = extract_crease_edges(box_mesh)
    # A box has exactly 12 sharp edges; triangulated faces may split an edge
    # into 2 segments, but all crease segments must lie on the 12 box edges.
    assert len(creases) >= 12
    verts = box_mesh.vertices
    for a, b in creases:
        pa, pb = verts[a], verts[b]
        # On a box edge, two of the three coordinates are on the boundary.
        on_boundary = 0
        for axis, size in enumerate(BOX_SIZE):
            if (abs(pa[axis]) < 0.02 and abs(pb[axis]) < 0.02) or (
                abs(pa[axis] - size) < 0.02 and abs(pb[axis] - size) < 0.02
            ):
                on_boundary += 1
        assert on_boundary >= 2


def test_html_viewer(box_mesh, tmp_path):
    path = write_html_viewer(box_mesh, tmp_path / "model.html", title="Testmodell")
    html = path.read_text()
    assert "<canvas" in html and "Testmodell" in html
    # Embedded metadata must be valid JSON and consistent with the mesh.
    meta = json.loads(re.search(r"const META = (\{.*?\});", html).group(1))
    assert meta["vertices"] == len(box_mesh.vertices)
    assert meta["triangles"] == len(box_mesh.faces)
    # Positions buffer decodes to the right size.
    b64 = re.search(r'decode\("([A-Za-z0-9+/=]+)", Float32Array\)', html).group(1)
    raw = base64.b64decode(b64)
    assert len(raw) == len(box_mesh.vertices) * 3 * 4
    # Self-contained: no external resources.
    assert "http://" not in html and "https://" not in html


def test_write_mesh_html_dispatch(box_mesh, tmp_path):
    out = write_mesh(box_mesh, tmp_path / "viewer.html")
    assert out.stat().st_size > 5000
    # No residual → the layer toggle stays absent from the metadata.
    meta = json.loads(re.search(r"const META = (\{.*?\});", out.read_text()).group(1))
    assert meta["points"] == 0


def test_html_viewer_residual_point_layer(box_mesh, tmp_path):
    """Unexplained scan points ride along as a toggleable colored layer."""
    import numpy as np

    from scantobim.core.cloud import PointCloud

    rng = np.random.default_rng(0)
    residual = PointCloud(
        points=rng.uniform(0, 2, (5000, 3)),
        colors=rng.integers(0, 255, (5000, 3)).astype(np.uint8),
    )
    out = write_mesh(box_mesh, tmp_path / "viewer.html", residual=residual)
    html = out.read_text()
    meta = json.loads(re.search(r"const META = (\{.*?\});", html).group(1))
    assert meta["points"] == 5000
    assert "Scan-Restpunkte" in html and "gl.POINTS" in html

    # Subsampling caps the embedded layer.
    big = PointCloud(points=rng.uniform(0, 2, (30000, 3)))
    out2 = write_mesh(box_mesh, tmp_path / "viewer2.html", residual=big)
    from scantobim.io.html_viewer import write_html_viewer as _w

    path3 = _w(box_mesh, tmp_path / "viewer3.html", points=big, max_layer_points=10000)
    meta3 = json.loads(re.search(r"const META = (\{.*?\});", path3.read_text()).group(1))
    assert meta3["points"] == 10000
