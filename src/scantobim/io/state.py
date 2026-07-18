"""Persisted reconstruction state for on-demand exports.

The GUI computes a model ONCE and lets the user save it in any format
afterwards ("Speichern als …") — no format pre-selection, no re-run. For
that, the job process persists everything the exporters need: the exact
surface geometry (STEP/IFC), the storeys (IFC), the clean structure mesh
(DXF floor plan) and the display mesh with its texture (GLB/OBJ/STL/PLY).

Textures are stored JPEG/PNG-encoded — a photographic 8192² atlas would
otherwise blow the state file up to hundreds of MB.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from scantobim.core.mesh import Mesh

STATE_NAME = "_export_state.pkl"


def _pack_mesh(mesh: Mesh | None) -> dict | None:
    if mesh is None:
        return None
    d = {
        "vertices": mesh.vertices,
        "faces": mesh.faces,
        "vertex_colors": mesh.vertex_colors,
        "face_groups": mesh.face_groups,
        "group_names": mesh.group_names,
        "uvs": mesh.uvs,
        "texture": None,
        "texture_raw": None,
    }
    if mesh.texture is not None:
        try:
            from scantobim.io.teximg import encode_texture

            d["texture"] = encode_texture(mesh.texture)  # (bytes, mime)
        except Exception:  # noqa: BLE001 — fall back to the raw array
            d["texture_raw"] = mesh.texture
    return d


def _unpack_mesh(d: dict | None) -> Mesh | None:
    if d is None:
        return None
    texture = d.get("texture_raw")
    if texture is None and d.get("texture") is not None:
        try:
            import io as _io

            from PIL import Image

            data, _mime = d["texture"]
            texture = np.asarray(Image.open(_io.BytesIO(data)).convert("RGB"))
        except Exception:  # noqa: BLE001 — texture is optional for exports
            texture = None
    return Mesh(
        vertices=d["vertices"],
        faces=d["faces"],
        vertex_colors=d.get("vertex_colors"),
        face_groups=d.get("face_groups"),
        group_names=d.get("group_names"),
        uvs=d.get("uvs"),
        texture=texture,
    )


def save_export_state(
    path,
    surfaces,
    storeys,
    structure_mesh: Mesh | None,
    display_mesh: Mesh | None,
) -> Path | None:
    """Write the export state next to the results. Best-effort: returns the
    path or ``None`` (a failed state save must never fail the model run)."""
    try:
        path = Path(path)
        state = {
            "version": 1,
            "surfaces": surfaces,
            "storeys": storeys,
            "structure_mesh": _pack_mesh(structure_mesh),
            "display_mesh": _pack_mesh(display_mesh),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=4)
        return path
    except Exception:  # noqa: BLE001
        return None


def load_export_state(path) -> dict | None:
    """Load a saved state; meshes come back as ``Mesh`` objects."""
    try:
        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301 — file written by this app
        state["structure_mesh"] = _unpack_mesh(state.get("structure_mesh"))
        state["display_mesh"] = _unpack_mesh(state.get("display_mesh"))
        return state
    except Exception:  # noqa: BLE001
        return None
