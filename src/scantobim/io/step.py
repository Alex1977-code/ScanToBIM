"""STEP export (ISO 10303-21, AP214) — real CAD geometry, not a mesh.

Every reconstructed surface becomes an ``ADVANCED_FACE`` with an analytic
``PLANE``, its exact boundary polygon as an ``EDGE_LOOP`` of shared
``EDGE_CURVE``/``LINE`` entities and openings as inner ``FACE_BOUND`` loops.
Watertight models are exported as a ``MANIFOLD_SOLID_BREP`` (a true solid),
open models as a ``SHELL_BASED_SURFACE_MODEL``.

The .stp file imports directly into SolidWorks, Inventor, Fusion, FreeCAD,
AutoCAD, **HiCAD** (Datei → Import → STEP; ein natives .SZA kann nur HiCAD
selbst speichern, STEP ist das dokumentierte Austauschformat) and every
other STEP-capable CAD system. Units: meters.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def write_step(
    surfaces: list,
    path: str | Path,
    name: str = "scantobim_model",
    timestamp: str | None = None,
) -> Path:
    """Write ``surfaces`` (list of :class:`SurfaceGeometry`) as AP214 STEP."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not surfaces:
        raise ValueError("no surfaces to export")

    w = _StepWriter()

    # --- shared topology: vertices and edges are welded across faces -------
    quant = 1e-7
    point_ids: dict[tuple, int] = {}  # quantized xyz -> CARTESIAN_POINT id
    vertex_ids: dict[int, int] = {}  # point entity id -> VERTEX_POINT id
    edge_ids: dict[tuple[int, int], int] = {}  # (vp_a, vp_b) canonical -> EDGE_CURVE
    edge_use_count: dict[tuple[int, int], int] = {}

    def point(p: np.ndarray) -> int:
        key = tuple(np.round(np.asarray(p, dtype=np.float64) / quant).astype(np.int64))
        if key not in point_ids:
            point_ids[key] = w.add(
                f"CARTESIAN_POINT('',({_f(p[0])},{_f(p[1])},{_f(p[2])}))"
            )
        return point_ids[key]

    def vertex(pid: int) -> int:
        if pid not in vertex_ids:
            vertex_ids[pid] = w.add(f"VERTEX_POINT('',#{pid})")
        return vertex_ids[pid]

    def direction(d: np.ndarray) -> int:
        return w.add(f"DIRECTION('',({_f(d[0])},{_f(d[1])},{_f(d[2])}))")

    def edge(pa: np.ndarray, pb: np.ndarray) -> tuple[int, bool]:
        """Shared EDGE_CURVE between two points; returns (id, same_sense)."""
        ida, idb = point(pa), point(pb)
        va, vb = vertex(ida), vertex(idb)
        key = (va, vb) if va < vb else (vb, va)
        same_sense = (va, vb) == key
        if key not in edge_ids:
            p_start = pa if same_sense else pb
            p_end = pb if same_sense else pa
            d = np.asarray(p_end, dtype=np.float64) - np.asarray(p_start, dtype=np.float64)
            length = float(np.linalg.norm(d))
            d = d / length if length > 0 else np.array([1.0, 0.0, 0.0])
            vec = w.add(f"VECTOR('',#{direction(d)},{_f(length)})")
            start_pid = ida if same_sense else idb
            line = w.add(f"LINE('',#{start_pid},#{vec})")
            edge_ids[key] = w.add(
                f"EDGE_CURVE('',#{key[0]},#{key[1]},#{line},.T.)"
            )
            edge_use_count[key] = 0
        edge_use_count[key] += 1
        return edge_ids[key], same_sense

    def loop(poly: np.ndarray) -> int:
        oriented = []
        n = len(poly)
        for i in range(n):
            eid, same = edge(poly[i], poly[(i + 1) % n])
            flag = ".T." if same else ".F."
            oriented.append(w.add(f"ORIENTED_EDGE('',*,*,#{eid},{flag})"))
        refs = ",".join(f"#{o}" for o in oriented)
        return w.add(f"EDGE_LOOP('',({refs}))")

    face_ids = []
    for geo in surfaces:
        normal = np.asarray(geo.normal, dtype=np.float64)
        origin = np.asarray(geo.outer[0], dtype=np.float64)
        first = np.asarray(geo.outer[1], dtype=np.float64) - origin
        first = first - (first @ normal) * normal
        norm = np.linalg.norm(first)
        first = first / norm if norm > 0 else _any_perpendicular(normal)
        placement = w.add(
            f"AXIS2_PLACEMENT_3D('',#{point(origin)},#{direction(normal)},#{direction(first)})"
        )
        plane = w.add(f"PLANE('',#{placement})")

        bounds = [w.add(f"FACE_OUTER_BOUND('',#{loop(geo.outer)},.T.)")]
        for hole in geo.holes:
            # Hole loops run clockwise around the face normal.
            bounds.append(w.add(f"FACE_BOUND('',#{loop(hole[::-1])},.T.)"))
        refs = ",".join(f"#{b}" for b in bounds)
        label = getattr(geo, "name", "") or "surface"
        face_ids.append(w.add(f"ADVANCED_FACE('{label}',({refs}),#{plane},.T.)"))

    closed = edge_use_count and all(c == 2 for c in edge_use_count.values())
    face_refs = ",".join(f"#{f}" for f in face_ids)
    if closed:
        shell = w.add(f"CLOSED_SHELL('',({face_refs}))")
        model = w.add(f"MANIFOLD_SOLID_BREP('{name}',#{shell})")
        rep_kind = "ADVANCED_BREP_SHAPE_REPRESENTATION"
    else:
        shell = w.add(f"OPEN_SHELL('',({face_refs}))")
        model = w.add(f"SHELL_BASED_SURFACE_MODEL('{name}',(#{shell}))")
        rep_kind = "MANIFOLD_SURFACE_SHAPE_REPRESENTATION"

    # --- units, context, product structure ---------------------------------
    length_unit = w.add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.METRE.))")
    angle_unit = w.add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    solid_unit = w.add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    uncertainty = w.add(
        f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),#{length_unit},"
        "'distance_accuracy_value','confusion accuracy')"
    )
    context = w.add(
        f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)"
        f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncertainty}))"
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length_unit},#{angle_unit},#{solid_unit}))"
        f"REPRESENTATION_CONTEXT('scan context','3D'))"
    )
    world = w.add(
        f"AXIS2_PLACEMENT_3D('',#{point(np.zeros(3))},"
        f"#{direction(np.array([0.0, 0.0, 1.0]))},#{direction(np.array([1.0, 0.0, 0.0]))})"
    )
    shape_rep = w.add(f"{rep_kind}('{name}',(#{world},#{model}),#{context})")

    app = w.add("APPLICATION_CONTEXT('automotive design')")
    w.add(
        f"APPLICATION_PROTOCOL_DEFINITION('international standard',"
        f"'automotive_design',2010,#{app})"
    )
    product_ctx = w.add(f"PRODUCT_CONTEXT('',#{app},'mechanical')")
    product = w.add(f"PRODUCT('{name}','{name}','reconstructed from scan',(#{product_ctx}))")
    w.add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part','',(#{product}))")
    formation = w.add(f"PRODUCT_DEFINITION_FORMATION('','',#{product})")
    def_ctx = w.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')")
    definition = w.add(f"PRODUCT_DEFINITION('design','',#{formation},#{def_ctx})")
    def_shape = w.add(f"PRODUCT_DEFINITION_SHAPE('','',#{definition})")
    w.add(f"SHAPE_DEFINITION_REPRESENTATION(#{def_shape},#{shape_rep})")

    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    header = (
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_DESCRIPTION(('ScanToBIM reconstructed model'),'2;1');\n"
        f"FILE_NAME('{path.name}','{stamp}',('ScanToBIM'),(''),"
        "'ScanToBIM','ScanToBIM','');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 3 1 1 }'));\n"
        "ENDSEC;\n"
        "DATA;\n"
    )
    footer = "ENDSEC;\nEND-ISO-10303-21;\n"
    path.write_text(header + w.dump() + footer, encoding="ascii")
    return path


class _StepWriter:
    def __init__(self) -> None:
        self.entities: list[str] = []

    def add(self, body: str) -> int:
        self.entities.append(body)
        return len(self.entities)  # ids are 1-based

    def dump(self) -> str:
        return "".join(
            f"#{i + 1}={body};\n" for i, body in enumerate(self.entities)
        )


def _f(x: float) -> str:
    """STEP real literal (always with a decimal point)."""
    s = f"{float(x):.10g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += "."
    return s


def _any_perpendicular(n: np.ndarray) -> np.ndarray:
    helper = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0.0, 1, 0])
    u = np.cross(n, helper)
    return u / np.linalg.norm(u)
