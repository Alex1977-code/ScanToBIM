"""Mesh and point cloud writers: OBJ, PLY (binary), STL (binary), glTF 2.0 (GLB)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from scantobim.core.cloud import PointCloud
from scantobim.core.mesh import Mesh


def write_mesh(mesh: Mesh, path: str | Path) -> Path:
    """Write ``mesh`` to ``path``; format is chosen by extension
    (``.obj``, ``.ply``, ``.stl``, ``.glb``, ``.gltf``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".obj":
        _write_obj(mesh, path)
    elif ext == ".ply":
        _write_ply_mesh(mesh, path)
    elif ext == ".stl":
        _write_stl(mesh, path)
    elif ext in (".glb", ".gltf"):
        _write_glb(mesh, path)
    else:
        raise ValueError(f"Unsupported mesh format: {ext!r} (use .obj .ply .stl .glb)")
    return path


def write_point_cloud(cloud: PointCloud, path: str | Path) -> Path:
    """Write a point cloud as binary PLY or xyz text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".ply":
        _write_ply_cloud(cloud, path)
    elif ext in (".xyz", ".txt"):
        np.savetxt(path, cloud.points, fmt="%.6f")
    else:
        raise ValueError(f"Unsupported point cloud output format: {ext!r} (use .ply .xyz)")
    return path


# --------------------------------------------------------------------------- OBJ

def _write_obj(mesh: Mesh, path: Path) -> None:
    lines = ["# ScanToBIM reconstruction", "o scan_model"]
    for v in mesh.vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    normals = mesh.vertex_normals()
    for n in normals:
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    prev_group = None
    groups = mesh.face_groups if mesh.face_groups is not None else np.zeros(len(mesh.faces), dtype=int)
    for face, group in zip(mesh.faces, groups):
        if group != prev_group:
            lines.append(f"g surface_{int(group):03d}")
            prev_group = group
        a, b, c = (int(i) + 1 for i in face)
        lines.append(f"f {a}//{a} {b}//{b} {c}//{c}")
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- PLY

def _write_ply_mesh(mesh: Mesh, path: Path) -> None:
    n_v, n_f = len(mesh.vertices), len(mesh.faces)
    has_color = mesh.vertex_colors is not None
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n_v}"]
    header += ["property float x", "property float y", "property float z"]
    if has_color:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += [f"element face {n_f}", "property list uchar int vertex_indices", "end_header"]

    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        if has_color:
            vdt = np.dtype([("xyz", "<f4", (3,)), ("rgb", "u1", (3,))])
            varr = np.empty(n_v, dtype=vdt)
            varr["xyz"] = mesh.vertices.astype(np.float32)
            varr["rgb"] = mesh.vertex_colors
        else:
            vdt = np.dtype([("xyz", "<f4", (3,))])
            varr = np.empty(n_v, dtype=vdt)
            varr["xyz"] = mesh.vertices.astype(np.float32)
        fh.write(varr.tobytes())
        fdt = np.dtype([("n", "u1"), ("idx", "<i4", (3,))])
        farr = np.empty(n_f, dtype=fdt)
        farr["n"] = 3
        farr["idx"] = mesh.faces.astype(np.int32)
        fh.write(farr.tobytes())


def _write_ply_cloud(cloud: PointCloud, path: Path) -> None:
    n = len(cloud)
    has_color = cloud.colors is not None
    has_normal = cloud.normals is not None
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += ["property float x", "property float y", "property float z"]
    if has_normal:
        header += ["property float nx", "property float ny", "property float nz"]
    if has_color:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += ["end_header"]

    fields: list[tuple] = [("xyz", "<f4", (3,))]
    if has_normal:
        fields.append(("n", "<f4", (3,)))
    if has_color:
        fields.append(("rgb", "u1", (3,)))
    arr = np.empty(n, dtype=np.dtype(fields))
    arr["xyz"] = cloud.points.astype(np.float32)
    if has_normal:
        arr["n"] = cloud.normals.astype(np.float32)
    if has_color:
        arr["rgb"] = cloud.colors
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(arr.tobytes())


# --------------------------------------------------------------------------- STL

def _write_stl(mesh: Mesh, path: Path) -> None:
    tri = mesh.vertices[mesh.faces]  # (F, 3, 3)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)

    n_f = len(mesh.faces)
    with open(path, "wb") as fh:
        fh.write(b"ScanToBIM binary STL".ljust(80, b"\0"))
        fh.write(struct.pack("<I", n_f))
        record = np.zeros(
            n_f,
            dtype=np.dtype(
                [("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")]
            ),
        )
        record["normal"] = normals.astype(np.float32)
        record["verts"] = tri.astype(np.float32)
        fh.write(record.tobytes())


# --------------------------------------------------------------------------- glTF / GLB

def _write_glb(mesh: Mesh, path: Path) -> None:
    positions = mesh.vertices.astype(np.float32)
    normals = mesh.vertex_normals().astype(np.float32)
    indices = mesh.faces.astype(np.uint32).ravel()

    colors = None
    if mesh.vertex_colors is not None:
        colors = (mesh.vertex_colors.astype(np.float32) / 255.0).astype(np.float32)

    def _pad(b: bytes, pad_byte: bytes = b"\0") -> bytes:
        return b + pad_byte * (-len(b) % 4)

    buffers = []
    buffer_views = []
    accessors = []
    offset = 0

    def _add_view(data: bytes, target: int) -> int:
        nonlocal offset
        data = _pad(data)
        buffer_views.append(
            {"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target}
        )
        buffers.append(data)
        offset += len(data)
        return len(buffer_views) - 1

    pos_view = _add_view(positions.tobytes(), 34962)
    accessors.append(
        {
            "bufferView": pos_view,
            "componentType": 5126,
            "count": len(positions),
            "type": "VEC3",
            "min": positions.min(axis=0).tolist() if len(positions) else [0, 0, 0],
            "max": positions.max(axis=0).tolist() if len(positions) else [0, 0, 0],
        }
    )
    nrm_view = _add_view(normals.tobytes(), 34962)
    accessors.append(
        {"bufferView": nrm_view, "componentType": 5126, "count": len(normals), "type": "VEC3"}
    )
    attributes = {"POSITION": 0, "NORMAL": 1}
    if colors is not None:
        col_view = _add_view(colors.tobytes(), 34962)
        accessors.append(
            {"bufferView": col_view, "componentType": 5126, "count": len(colors), "type": "VEC3"}
        )
        attributes["COLOR_0"] = len(accessors) - 1

    idx_view = _add_view(indices.tobytes(), 34963)
    accessors.append(
        {"bufferView": idx_view, "componentType": 5125, "count": len(indices), "type": "SCALAR"}
    )

    gltf = {
        "asset": {"version": "2.0", "generator": "ScanToBIM"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "scan_model"}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": len(accessors) - 1,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "materials": [
            {
                "name": "scan_surface",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.9,
                },
                "doubleSided": True,
            }
        ],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "buffers": [{"byteLength": offset}],
    }

    bin_chunk = b"".join(buffers)
    json_chunk = _pad(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")

    if path.suffix.lower() == ".gltf":
        # Standalone .gltf with embedded base64 buffer.
        import base64

        gltf["buffers"] = [
            {
                "byteLength": len(bin_chunk),
                "uri": "data:application/octet-stream;base64,"
                + base64.b64encode(bin_chunk).decode("ascii"),
            }
        ]
        path.write_text(json.dumps(gltf, indent=1))
        return

    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))  # "glTF"
        fh.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))  # "JSON"
        fh.write(json_chunk)
        fh.write(struct.pack("<II", len(bin_chunk), 0x004E4942))  # "BIN"
        fh.write(bin_chunk)
