"""IFC4 export — reconstructed surfaces as real BIM building elements.

The classified surfaces map to IFC entities:

===========  =========================
class        IFC entity
===========  =========================
wall         IfcWall
floor        IfcSlab (.FLOOR.)
ceiling      IfcCovering (.CEILING.)
slab         IfcSlab (.LANDING.)
sloped       IfcSlab (.ROOF.)
===========  =========================

Geometry is exported per element as an IfcShellBasedSurfaceModel whose face
carries the exact boundary polygon incl. window/door openings as inner
loops. The file follows the IFC4 schema (ISO 16739) and opens in Revit,
ArchiCAD, BIMcollab, Solibri, BlenderBIM and every IFC viewer. Units: meters.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_CLASS_MAP = {
    "wall": ("IFCWALL", None),
    "floor": ("IFCSLAB", ".FLOOR."),
    "ceiling": ("IFCCOVERING", ".CEILING."),
    "slab": ("IFCSLAB", ".LANDING."),
    "sloped": ("IFCSLAB", ".ROOF."),
    "roof": ("IFCROOF", ".NOTDEFINED."),
}

_GUID_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"


def ifc_guid() -> str:
    """Compress a random UUID into the 22-character IFC GlobalId encoding
    (2 bits in the first character, 6 bits in each of the remaining 21)."""
    n = uuid.uuid4().int
    result = [_GUID_CHARS[(n >> 126) & 0x3]]
    for i in range(21):
        result.append(_GUID_CHARS[(n >> (120 - 6 * i)) & 0x3F])
    return "".join(result)


def write_ifc(
    surfaces: list,
    path: str | Path,
    name: str = "ScanToBIM Modell",
    storey_name: str = "Erdgeschoss",
    storeys: list[dict] | None = None,
    timestamp: str | None = None,
) -> Path:
    """Write classified :class:`SurfaceGeometry` objects as an IFC4 file.

    ``storeys`` (from the pipeline report) creates one IfcBuildingStorey per
    entry (``{"elevation", "height"}``); elements are assigned by the height
    of their geometry. Without it a single storey is created.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not surfaces:
        raise ValueError("no surfaces to export")

    w = _IfcWriter()

    # --- units + context ----------------------------------------------------
    u_len = w.add("IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.)")
    u_area = w.add("IFCSIUNIT(*,.AREAUNIT.,$,.SQUARE_METRE.)")
    u_vol = w.add("IFCSIUNIT(*,.VOLUMEUNIT.,$,.CUBIC_METRE.)")
    u_ang = w.add("IFCSIUNIT(*,.PLANEANGLEUNIT.,$,.RADIAN.)")
    units = w.add(f"IFCUNITASSIGNMENT((#{u_len},#{u_area},#{u_vol},#{u_ang}))")

    origin = w.add("IFCCARTESIANPOINT((0.,0.,0.))")
    z_dir = w.add("IFCDIRECTION((0.,0.,1.))")
    x_dir = w.add("IFCDIRECTION((1.,0.,0.))")
    world_axis = w.add(f"IFCAXIS2PLACEMENT3D(#{origin},#{z_dir},#{x_dir})")
    context = w.add(
        f"IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.0E-05,#{world_axis},$)"
    )

    # --- spatial structure ----------------------------------------------------
    project = w.add(
        f"IFCPROJECT('{ifc_guid()}',$,'{name}',"
        f"'Rekonstruiert aus Punktwolke (ScanToBIM)',$,$,$,(#{context}),#{units})"
    )
    site_lp = w.add(f"IFCLOCALPLACEMENT($,#{world_axis})")
    site = w.add(
        f"IFCSITE('{ifc_guid()}',$,'Gelaende',$,$,#{site_lp},$,$,.ELEMENT.,$,$,$,$,$)"
    )
    bld_lp = w.add(f"IFCLOCALPLACEMENT(#{site_lp},#{world_axis})")
    building = w.add(
        f"IFCBUILDING('{ifc_guid()}',$,'Gebaeude',$,$,#{bld_lp},$,$,.ELEMENT.,$,$,$)"
    )
    if not storeys:
        storeys = [{"index": 0, "elevation": 0.0, "height": 0.0}]
    storey_ids = []
    storey_lps = []
    for k, st in enumerate(storeys):
        lp = w.add(f"IFCLOCALPLACEMENT(#{bld_lp},#{world_axis})")
        label = storey_name if len(storeys) == 1 else f"Geschoss {k}"
        storey_ids.append(
            w.add(
                f"IFCBUILDINGSTOREY('{ifc_guid()}',$,'{label}',$,$,#{lp},$,$,"
                f".ELEMENT.,{_f(st.get('elevation', 0.0))})"
            )
        )
        storey_lps.append(lp)
    w.add(f"IFCRELAGGREGATES('{ifc_guid()}',$,$,$,#{project},(#{site}))")
    w.add(f"IFCRELAGGREGATES('{ifc_guid()}',$,$,$,#{site},(#{building}))")
    storey_refs = ",".join(f"#{s}" for s in storey_ids)
    w.add(f"IFCRELAGGREGATES('{ifc_guid()}',$,$,$,#{building},({storey_refs}))")

    def storey_for(mean_z: float) -> int:
        """Index of the storey whose [elevation, elevation+height) holds z."""
        for k in range(len(storeys) - 1, -1, -1):
            if mean_z >= storeys[k].get("elevation", 0.0) - 0.3:
                return k
        return 0

    # --- building elements ----------------------------------------------------
    def cartesian(p) -> int:
        return w.add(
            f"IFCCARTESIANPOINT(({_f(p[0])},{_f(p[1])},{_f(p[2])}))"
        )

    def poly_loop(poly: np.ndarray) -> int:
        refs = ",".join(f"#{cartesian(p)}" for p in poly)
        return w.add(f"IFCPOLYLOOP(({refs}))")

    elements_by_storey: dict[int, list[int]] = {k: [] for k in range(len(storeys))}
    for geo in surfaces:
        cls = getattr(geo, "surface_class", "") or "wall"
        entity, predefined = _CLASS_MAP.get(cls, ("IFCBUILDINGELEMENTPROXY", None))
        sk = storey_for(float(np.mean(geo.outer[:, 2])))
        storey_lp = storey_lps[sk]

        bounds = [w.add(f"IFCFACEOUTERBOUND(#{poly_loop(geo.outer)},.T.)")]
        for hole in geo.holes:
            bounds.append(w.add(f"IFCFACEBOUND(#{poly_loop(hole[::-1])},.T.)"))
        bound_refs = ",".join(f"#{b}" for b in bounds)
        face = w.add(f"IFCFACE(({bound_refs}))")
        shell = w.add(f"IFCOPENSHELL((#{face}))")
        model = w.add(f"IFCSHELLBASEDSURFACEMODEL((#{shell}))")
        shape = w.add(
            f"IFCSHAPEREPRESENTATION(#{context},'Body','SurfaceModel',(#{model}))"
        )
        pds = w.add(f"IFCPRODUCTDEFINITIONSHAPE($,$,(#{shape}))")
        lp = w.add(f"IFCLOCALPLACEMENT(#{storey_lp},#{world_axis})")
        label = getattr(geo, "name", "") or cls
        if entity == "IFCCOVERING":
            args = f"'{ifc_guid()}',$,'{label}',$,$,#{lp},#{pds},$,{predefined}"
        elif entity in ("IFCSLAB", "IFCROOF"):
            args = f"'{ifc_guid()}',$,'{label}',$,$,#{lp},#{pds},$,{predefined}"
        elif entity == "IFCWALL":
            args = f"'{ifc_guid()}',$,'{label}',$,$,#{lp},#{pds},$,$"
        else:
            args = f"'{ifc_guid()}',$,'{label}',$,$,#{lp},#{pds},$,$"
        elements_by_storey[sk].append(w.add(f"{entity}({args})"))

    for k, elems in elements_by_storey.items():
        if not elems:
            continue
        elem_refs = ",".join(f"#{e}" for e in elems)
        w.add(
            f"IFCRELCONTAINEDINSPATIALSTRUCTURE('{ifc_guid()}',$,$,$,"
            f"({elem_refs}),#{storey_ids[k]})"
        )

    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    header = (
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_DESCRIPTION(('ViewDefinition [ReferenceView]'),'2;1');\n"
        f"FILE_NAME('{path.name}','{stamp}',('ScanToBIM'),(''),"
        "'ScanToBIM','ScanToBIM','');\n"
        "FILE_SCHEMA(('IFC4'));\n"
        "ENDSEC;\n"
        "DATA;\n"
    )
    footer = "ENDSEC;\nEND-ISO-10303-21;\n"
    path.write_text(header + w.dump() + footer, encoding="ascii")
    return path


class _IfcWriter:
    def __init__(self) -> None:
        self.entities: list[str] = []

    def add(self, body: str) -> int:
        self.entities.append(body)
        return len(self.entities)

    def dump(self) -> str:
        return "".join(f"#{i + 1}={body};\n" for i, body in enumerate(self.entities))


def _f(x: float) -> str:
    s = f"{float(x):.10g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += "."
    return s
