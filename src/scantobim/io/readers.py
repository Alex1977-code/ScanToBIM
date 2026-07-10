"""Point cloud readers.

Supported formats
-----------------
- ``.las`` / ``.laz``  — airborne / terrestrial / mobile LiDAR (via laspy;
  ``.laz`` needs the ``lazrs`` backend, install extra ``scantobim[laz]``)
- ``.ply``             — ascii and binary_little_endian, including the point
  clouds exported by photogrammetry tools (COLMAP, Metashape, RealityCapture)
  and 3D Gaussian Splatting trainers (positions are read, SH coefficients
  ignored)
- ``.pcd``             — PCL point cloud data, ascii and binary
- ``.e57``             — ASTM E57 scanner exchange (optional ``pye57``)
- ``.xyz`` / ``.pts`` / ``.txt`` / ``.csv`` — whitespace/comma separated
  ``x y z [intensity] [r g b]`` text formats (``.pts`` may carry a leading
  point-count line, as written by Leica Cyclone)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from scantobim.core.cloud import PointCloud

_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


def read_point_cloud(path: str | Path) -> PointCloud:
    """Read a point cloud, dispatching on the file extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()
    if ext in (".las", ".laz"):
        return _read_las(path)
    if ext == ".ply":
        return _read_ply(path)
    if ext == ".pcd":
        return _read_pcd(path)
    if ext == ".e57":
        return _read_e57(path)
    if ext in (".xyz", ".pts", ".txt", ".csv", ".asc"):
        return _read_text(path)
    raise ValueError(
        f"Unsupported point cloud format: {ext!r} "
        "(supported: .las .laz .ply .pcd .e57 .xyz .pts .txt .csv .asc)"
    )


def _read_las(path: Path) -> PointCloud:
    import laspy

    try:
        las = laspy.read(str(path))
    except laspy.errors.LaspyException as exc:  # pragma: no cover - env specific
        if path.suffix.lower() == ".laz":
            raise RuntimeError(
                "Reading .laz requires a LAZ backend: pip install lazrs"
            ) from exc
        raise

    points = np.column_stack([las.x, las.y, las.z]).astype(np.float64)

    colors = None
    dims = set(las.point_format.dimension_names)
    if {"red", "green", "blue"} <= dims:
        rgb = np.column_stack([las.red, las.green, las.blue]).astype(np.float64)
        # LAS colors are usually 16 bit; scale down if needed.
        if rgb.max() > 255:
            rgb = rgb / 257.0
        colors = np.clip(rgb, 0, 255).astype(np.uint8)

    intensity = None
    if "intensity" in dims:
        raw = np.asarray(las.intensity, dtype=np.float32)
        peak = raw.max() if len(raw) else 0.0
        if peak > 0:
            intensity = raw / peak

    return PointCloud(points=points, colors=colors, intensity=intensity, source=str(path))


def _read_ply(path: Path) -> PointCloud:
    with open(path, "rb") as fh:
        magic = fh.readline().strip()
        if magic != b"ply":
            raise ValueError(f"{path}: not a PLY file")

        fmt = None
        elements: list[tuple[str, int, list[tuple[str, str]]]] = []
        cur_props: list[tuple[str, str]] = []
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: unexpected EOF in PLY header")
            tokens = line.decode("ascii", "replace").strip().split()
            if not tokens or tokens[0] == "comment" or tokens[0] == "obj_info":
                continue
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                cur_props = []
                elements.append((tokens[1], int(tokens[2]), cur_props))
            elif tokens[0] == "property":
                if tokens[1] == "list":
                    # (count_type, item_type, name)
                    cur_props.append((tokens[4], f"list:{tokens[2]}:{tokens[3]}"))
                else:
                    cur_props.append((tokens[2], tokens[1]))
            elif tokens[0] == "end_header":
                break

        if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
            raise ValueError(f"{path}: unsupported PLY format {fmt!r}")

        vertex_data = None
        for name, count, props in elements:
            if any(t.startswith("list:") for _, t in props):
                if name == "vertex":
                    raise ValueError(f"{path}: list properties on vertex element unsupported")
                # Skip non-vertex list elements (e.g. faces) — clouds only.
                if fmt == "ascii":
                    for _ in range(count):
                        fh.readline()
                else:
                    _skip_ply_list_element(fh, count, props, fmt)
                continue

            if fmt == "ascii":
                rows = np.loadtxt(
                    (fh.readline() for _ in range(count)), dtype=np.float64, ndmin=2
                )
                data = {p_name: rows[:, i] for i, (p_name, _) in enumerate(props)}
            else:
                endian = "<" if fmt == "binary_little_endian" else ">"
                dtype = np.dtype([(p, endian + _PLY_TYPES[t]) for p, t in props])
                raw = fh.read(dtype.itemsize * count)
                if len(raw) < dtype.itemsize * count:
                    raise ValueError(f"{path}: truncated PLY payload")
                arr = np.frombuffer(raw, dtype=dtype, count=count)
                data = {p: arr[p] for p, _ in props}

            if name == "vertex":
                vertex_data = data

        if vertex_data is None:
            raise ValueError(f"{path}: PLY file has no vertex element")

    if not {"x", "y", "z"} <= vertex_data.keys():
        raise ValueError(f"{path}: PLY vertex element lacks x/y/z")
    points = np.column_stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]]
    ).astype(np.float64)

    colors = None
    if {"red", "green", "blue"} <= vertex_data.keys():
        colors = np.column_stack(
            [vertex_data["red"], vertex_data["green"], vertex_data["blue"]]
        )
        if colors.dtype != np.uint8:
            colors = np.clip(
                colors * 255.0 if colors.max() <= 1.0 else colors, 0, 255
            )
        colors = colors.astype(np.uint8)

    normals = None
    if {"nx", "ny", "nz"} <= vertex_data.keys():
        normals = np.column_stack(
            [vertex_data["nx"], vertex_data["ny"], vertex_data["nz"]]
        ).astype(np.float64)

    intensity = None
    for key in ("intensity", "scalar_intensity", "scalar_Intensity"):
        if key in vertex_data:
            raw = np.asarray(vertex_data[key], dtype=np.float32)
            peak = raw.max() if len(raw) else 0.0
            intensity = raw / peak if peak > 0 else raw
            break

    return PointCloud(points, colors=colors, normals=normals, intensity=intensity, source=str(path))


def _skip_ply_list_element(fh, count: int, props, fmt: str) -> None:
    endian = "<" if fmt == "binary_little_endian" else ">"
    for _ in range(count):
        for _p_name, p_type in props:
            if p_type.startswith("list:"):
                _, count_t, item_t = p_type.split(":")
                cnt_dtype = np.dtype(endian + _PLY_TYPES[count_t])
                n_items = int(np.frombuffer(fh.read(cnt_dtype.itemsize), cnt_dtype)[0])
                fh.read(np.dtype(_PLY_TYPES[item_t]).itemsize * n_items)
            else:
                fh.read(np.dtype(_PLY_TYPES[p_type]).itemsize)


def _read_pcd(path: Path) -> PointCloud:
    with open(path, "rb") as fh:
        header: dict[str, list[str]] = {}
        while True:
            line = fh.readline().decode("ascii", "replace").strip()
            if line.startswith("#") or not line:
                continue
            key, *vals = line.split()
            header[key.upper()] = vals
            if key.upper() == "DATA":
                break

        fields = header.get("FIELDS", [])
        sizes = [int(s) for s in header.get("SIZE", [])]
        types = header.get("TYPE", [])
        counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
        n_points = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])
        data_mode = header["DATA"][0].lower()

        np_types = {"F": "f", "I": "i", "U": "u"}
        dtype_fields = []
        for name, size, typ, cnt in zip(fields, sizes, types, counts):
            base = f"<{np_types[typ]}{size}"
            dtype_fields.append((name, base, (cnt,)) if cnt > 1 else (name, base))
        dtype = np.dtype(dtype_fields)

        if data_mode == "ascii":
            rows = np.loadtxt(fh, dtype=np.float64, ndmin=2, max_rows=n_points)
            col = 0
            data = {}
            for name, cnt in zip(fields, counts):
                data[name] = rows[:, col] if cnt == 1 else rows[:, col : col + cnt]
                col += cnt
        elif data_mode == "binary":
            arr = np.frombuffer(fh.read(dtype.itemsize * n_points), dtype=dtype, count=n_points)
            data = {name: arr[name] for name in fields}
        elif data_mode == "binary_compressed":
            raise ValueError(
                f"{path}: binary_compressed PCD is not supported — "
                "convert with `pcl_convert_pcd_ascii_binary in.pcd out.pcd 1`"
            )
        else:
            raise ValueError(f"{path}: unknown PCD data mode {data_mode!r}")

    if not {"x", "y", "z"} <= data.keys():
        raise ValueError(f"{path}: PCD lacks x/y/z fields")
    points = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]

    colors = None
    if "rgb" in data:
        packed = np.asarray(data["rgb"])[finite]
        as_int = packed.view(np.uint32) if packed.dtype.kind == "f" else packed.astype(np.uint32)
        colors = np.column_stack(
            [(as_int >> 16) & 0xFF, (as_int >> 8) & 0xFF, as_int & 0xFF]
        ).astype(np.uint8)

    intensity = None
    if "intensity" in data:
        raw = np.asarray(data["intensity"], dtype=np.float32)[finite]
        peak = raw.max() if len(raw) else 0.0
        intensity = raw / peak if peak > 0 else raw

    return PointCloud(points, colors=colors, intensity=intensity, source=str(path))


def _read_e57(path: Path) -> PointCloud:
    try:
        import pye57
    except ImportError as exc:
        raise RuntimeError(
            "Reading .e57 requires the optional dependency pye57: "
            "pip install scantobim[e57]"
        ) from exc

    e57 = pye57.E57(str(path))
    clouds = []
    colors_parts = []
    intensity_parts = []
    has_color = True
    has_intensity = True
    for i in range(e57.scan_count):
        data = e57.read_scan(i, ignore_missing_fields=True, colors=True, intensity=True)
        clouds.append(
            np.column_stack(
                [data["cartesianX"], data["cartesianY"], data["cartesianZ"]]
            )
        )
        if all(k in data for k in ("colorRed", "colorGreen", "colorBlue")):
            colors_parts.append(
                np.column_stack([data["colorRed"], data["colorGreen"], data["colorBlue"]])
            )
        else:
            has_color = False
        if "intensity" in data:
            intensity_parts.append(np.asarray(data["intensity"], dtype=np.float32))
        else:
            has_intensity = False

    points = np.vstack(clouds)
    colors = np.clip(np.vstack(colors_parts), 0, 255).astype(np.uint8) if has_color and colors_parts else None
    intensity = None
    if has_intensity and intensity_parts:
        raw = np.concatenate(intensity_parts)
        peak = raw.max() if len(raw) else 0.0
        intensity = raw / peak if peak > 0 else raw
    return PointCloud(points, colors=colors, intensity=intensity, source=str(path))


def _read_text(path: Path) -> PointCloud:
    delimiter = "," if path.suffix.lower() == ".csv" else None
    with open(path, "r", errors="replace") as fh:
        first = fh.readline()
        skip = 0
        # Leica .pts files start with a single point-count line.
        if re.fullmatch(r"\s*\d+\s*", first):
            skip = 1
        # Header line with column names.
        elif first and not _is_numeric_row(first, delimiter):
            skip = 1
    data = np.loadtxt(path, skiprows=skip, delimiter=delimiter, ndmin=2)
    if data.shape[1] < 3:
        raise ValueError(f"{path}: expected at least 3 columns (x y z)")

    points = data[:, :3].astype(np.float64)
    colors = None
    intensity = None
    extra = data.shape[1] - 3
    if extra == 1:
        intensity = _normalize_intensity(data[:, 3])
    elif extra == 3:
        colors = np.clip(data[:, 3:6], 0, 255).astype(np.uint8)
    elif extra >= 4:
        intensity = _normalize_intensity(data[:, 3])
        colors = np.clip(data[:, 4:7], 0, 255).astype(np.uint8)
    return PointCloud(points, colors=colors, intensity=intensity, source=str(path))


def _is_numeric_row(line: str, delimiter: str | None) -> bool:
    parts = line.strip().split(delimiter)
    try:
        [float(p) for p in parts if p != ""]
        return True
    except ValueError:
        return False


def _normalize_intensity(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    if len(raw) and raw.min() >= -2048 and raw.max() <= 0:
        # Leica .pts convention: intensity in [-2048, 0]
        return (raw + 2048.0) / 2048.0
    peak = np.abs(raw).max() if len(raw) else 0.0
    return raw / peak if peak > 0 else raw
