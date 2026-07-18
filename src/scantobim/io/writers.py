"""Mesh and point cloud writers: OBJ, PLY (binary), STL (binary), glTF 2.0 (GLB)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from scantobim.core.cloud import PointCloud
from scantobim.core.mesh import Mesh


def write_mesh(
    mesh: Mesh,
    path: str | Path,
    residual: PointCloud | None = None,
    freeform: Mesh | None = None,
    freeform_label: str = "Freiform-Restgeometrie",
    detail: Mesh | None = None,
    detail_label: str = "Detail-Mesh (fotorealistisch)",
) -> Path:
    """Write ``mesh`` to ``path``; format is chosen by extension
    (``.obj``, ``.ply``, ``.stl``, ``.glb``, ``.gltf``, ``.html``).

    ``residual`` (scan points the model does not explain) and ``freeform``
    (the hybrid free-form skin of those points) are embedded as toggleable
    layers in the HTML viewer; other formats ignore them.
    """
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
    elif ext in (".html", ".htm"):
        from scantobim.io.html_viewer import write_html_viewer

        write_html_viewer(
            mesh, path, points=residual, freeform=freeform,
            freeform_label=freeform_label,
            detail=detail, detail_label=detail_label,
        )
    else:
        raise ValueError(
            f"Unsupported mesh format: {ext!r} (use .obj .ply .stl .glb .html)"
        )
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
    textured = mesh.texture is not None and mesh.uvs is not None
    lines = [f"# {_generator()} reconstruction"]
    if textured:
        mtl_path = path.with_suffix(".mtl")
        png_path = path.with_name(path.stem + "_texture.png")
        lines.append(f"mtllib {mtl_path.name}")
    lines.append("o scan_model")
    for v in mesh.vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    if textured:
        for uv in mesh.uvs:
            lines.append(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}")  # OBJ v runs bottom-up
    normals = mesh.vertex_normals()
    for n in normals:
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    if textured:
        lines.append("usemtl scan_texture")
    prev_group = None
    groups = mesh.face_groups if mesh.face_groups is not None else np.zeros(len(mesh.faces), dtype=int)
    names = mesh.group_names or {}
    for face, group in zip(mesh.faces, groups):
        if group != prev_group:
            lines.append(f"g {names.get(int(group), f'surface_{int(group):03d}')}")
            prev_group = group
        a, b, c = (int(i) + 1 for i in face)
        if textured:
            lines.append(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")
        else:
            lines.append(f"f {a}//{a} {b}//{b} {c}//{c}")
    path.write_text("\n".join(lines) + "\n")

    if textured:
        from scantobim.io.png import encode_png

        png_path.write_bytes(encode_png(mesh.texture))
        mtl_path.write_text(
            "newmtl scan_texture\n"
            "Ka 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n"
            f"map_Kd {png_path.name}\n"
        )


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

def _generator() -> str:
    from scantobim import __version__

    return f"ScanToBIM {__version__}"


def _write_glb(mesh: Mesh, path: Path) -> None:
    positions = mesh.vertices.astype(np.float32)
    normals = mesh.vertex_normals().astype(np.float32)
    indices = mesh.faces.astype(np.uint32).ravel()

    textured = mesh.texture is not None and mesh.uvs is not None
    colors = None
    if mesh.vertex_colors is not None and not textured:
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
    if textured:
        uv_view = _add_view(mesh.uvs.astype(np.float32).tobytes(), 34962)
        accessors.append(
            {"bufferView": uv_view, "componentType": 5126, "count": len(mesh.uvs), "type": "VEC2"}
        )
        attributes["TEXCOORD_0"] = len(accessors) - 1

    idx_view = _add_view(indices.tobytes(), 34963)
    accessors.append(
        {"bufferView": idx_view, "componentType": 5125, "count": len(indices), "type": "SCALAR"}
    )

    material = {
        "name": "scan_surface",
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.9,
        },
        "doubleSided": True,
    }
    gltf = {
        "asset": {"version": "2.0", "generator": _generator()},
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
        "materials": [material],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "buffers": [{"byteLength": offset}],
    }
    if textured:
        from scantobim.io.teximg import encode_texture

        img_bytes, mime = encode_texture(mesh.texture)
        img_view = len(buffer_views)
        data = _pad(img_bytes)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data)})
        buffers.append(data)
        offset += len(data)
        gltf["buffers"][0]["byteLength"] = offset
        gltf["images"] = [{"bufferView": img_view, "mimeType": mime}]
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
        gltf["textures"] = [{"sampler": 0, "source": 0}]
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}

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
