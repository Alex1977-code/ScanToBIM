"""STEP (AP214) and IFC4 export: structural validity + reference integrity."""

import re

import numpy as np
import pytest

from scantobim import PipelineConfig, reconstruct
from scantobim.io.ifc import ifc_guid, write_ifc
from scantobim.io.step import write_step
from tests.synthetic import make_box_scan, make_l_room_scan

BOX_SIZE = (4.0, 3.0, 2.5)


@pytest.fixture(scope="module")
def box_result():
    cloud = make_box_scan(size=BOX_SIZE, density=800, noise=0.004)
    return reconstruct(cloud, PipelineConfig.preset("building"))


@pytest.fixture(scope="module")
def room_result():
    cloud = make_l_room_scan(density=900, noise=0.004, window=True)
    return reconstruct(cloud, PipelineConfig.preset("indoor"))


def _check_reference_integrity(text: str):
    defined = set(re.findall(r"^#(\d+)\s*=", text, flags=re.M))
    referenced = set(re.findall(r"#(\d+)", text))
    missing = referenced - defined
    assert not missing, f"undefined entity references: {sorted(missing)[:10]}"


def test_step_box_is_solid(box_result, tmp_path):
    path = write_step(box_result.surfaces, tmp_path / "box.stp")
    text = path.read_text()
    assert text.startswith("ISO-10303-21;")
    assert text.rstrip().endswith("END-ISO-10303-21;")
    assert "AUTOMOTIVE_DESIGN" in text
    # A watertight box exports as a true solid with shared edges.
    assert "MANIFOLD_SOLID_BREP" in text
    assert "CLOSED_SHELL" in text
    assert text.count("ADVANCED_FACE") == 6
    # A box has 12 edges; every EDGE_CURVE is shared between two faces.
    assert text.count("EDGE_CURVE") == 12
    assert text.count("ORIENTED_EDGE") == 24
    _check_reference_integrity(text)


def test_step_open_geometry_is_shell(room_result, tmp_path):
    # The room with a window opening has boundary edges → surface model.
    path = write_step(room_result.surfaces, tmp_path / "room.stp")
    text = path.read_text()
    assert "SHELL_BASED_SURFACE_MODEL" in text or "MANIFOLD_SOLID_BREP" in text
    # The window appears as an inner FACE_BOUND on one face.
    assert "FACE_BOUND" in text
    _check_reference_integrity(text)


def test_step_deterministic_geometry(box_result, tmp_path):
    a = write_step(box_result.surfaces, tmp_path / "a.stp", timestamp="2026-01-01T00:00:00")
    b = write_step(box_result.surfaces, tmp_path / "b.stp", timestamp="2026-01-01T00:00:00")
    assert a.read_text().replace("a.stp", "x") == b.read_text().replace("b.stp", "x")


def test_step_parses_with_steputils(box_result, tmp_path):
    steputils = pytest.importorskip("steputils.p21")
    path = write_step(box_result.surfaces, tmp_path / "box.stp")
    doc = steputils.readfile(str(path))
    assert doc is not None


def test_ifc_guid_format():
    guids = {ifc_guid() for _ in range(50)}
    assert len(guids) == 50
    for g in guids:
        assert len(g) == 22
        assert g[0] in "0123"


def test_ifc_room(room_result, tmp_path):
    path = write_ifc(room_result.surfaces, tmp_path / "room.ifc")
    text = path.read_text()
    assert "FILE_SCHEMA(('IFC4'))" in text
    assert text.count("IFCWALL(") == 6
    assert text.count("IFCSLAB(") >= 1  # floor
    assert text.count("IFCCOVERING(") >= 1  # ceiling
    assert "IFCPROJECT(" in text and "IFCBUILDINGSTOREY(" in text
    assert "IFCRELCONTAINEDINSPATIALSTRUCTURE(" in text
    # Window opening → inner IFCFACEBOUND on the wall face.
    assert "IFCFACEBOUND(" in text
    _check_reference_integrity(text)


def test_ifc_semantic_names(room_result, tmp_path):
    path = write_ifc(room_result.surfaces, tmp_path / "room.ifc")
    text = path.read_text()
    assert re.search(r"IFCWALL\('[^']+',\$,'wall_\d+'", text)


def test_empty_surfaces_raise(tmp_path):
    with pytest.raises(ValueError):
        write_step([], tmp_path / "x.stp")
    with pytest.raises(ValueError):
        write_ifc([], tmp_path / "x.ifc")
