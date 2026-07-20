"""Fallback-Texturierung: OpenMVS-Mesh + eigener Foto-Atlas (ScanToBIM).

Nimmt das Delaunay-Mesh der Photogrammetrie-Kette (scene_mesh.ply),
dezimiert es per 2-cm-Vertex-Clustering und backt den bewaehrten
ScanToBIM-Foto-Atlas darauf (Best-Kamera je Face, Kamera-Farb-
harmonisierung, 4x8192, EDT-Lochfuellung). Kameras kommen aus dem
metrisch ausgerichteten COLMAP-Modell (sparse_tri) — gleiche Weltlage
wie das Mesh, praezise SfM-Posen.
"""

from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(r"C:\Users\alexanderm\Desktop\ScanToBIM\colmap_ws")
OUT = WS / "mvs" / "scene_bake.glb"


def read_binary_ply(path: Path):
    """Minimaler Reader fuer OpenMVS-PLY (binary_little_endian)."""
    with open(path, "rb") as f:
        n_vert = n_face = 0
        vert_props: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                n_vert = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element face"):
                n_face = int(line.split()[-1])
                in_vertex = False
            elif line.startswith("property") and in_vertex:
                _, typ, name = line.split()[:3]
                vert_props.append((typ, name))
            elif line == "end_header":
                break
        sizes = {"float": 4, "float32": 4, "double": 8, "uchar": 1,
                 "uint8": 1, "int": 4, "uint": 4}
        stride = sum(sizes[t] for t, _ in vert_props)
        raw = f.read(n_vert * stride)
        # x,y,z liegen als erste drei float-Properties vor.
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_vert, stride)
        off = 0
        cols = {}
        for typ, name in vert_props:
            if typ in ("float", "float32") and name in ("x", "y", "z"):
                cols[name] = arr[:, off:off + 4].copy().view(np.float32)[:, 0]
            off += sizes[typ]
        verts = np.column_stack([cols["x"], cols["y"], cols["z"]]).astype(
            np.float64
        )
        # Faces: uchar Anzahl + 3x int32
        fraw = f.read(n_face * 13)
        fa = np.frombuffer(fraw, dtype=np.uint8).reshape(n_face, 13)
        assert (fa[:, 0] == 3).all(), "nur Dreiecke erwartet"
        faces = fa[:, 1:].copy().view(np.int32).reshape(n_face, 3).astype(
            np.int64
        )
    return verts, faces


def decimate_grid(verts: np.ndarray, faces: np.ndarray, cell: float):
    """Vertex-Clustering auf ein Raster (schnell, topologie-tolerant)."""
    key = np.floor(verts / cell).astype(np.int64)
    packed = (
        (key[:, 0] + 2**20) * 2**42
        + (key[:, 1] + 2**20) * 2**21
        + (key[:, 2] + 2**20)
    )
    uniq, inv = np.unique(packed, return_inverse=True)
    acc = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(acc, inv, verts)
    np.add.at(cnt, inv, 1.0)
    new_verts = acc / cnt[:, None]
    nf = inv[faces]
    keep = (
        (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2])
        & (nf[:, 0] != nf[:, 2])
    )
    nf = nf[keep]
    # Doppelte Dreiecke entfernen
    sf = np.sort(nf, axis=1)
    code = (sf[:, 0] * len(new_verts) + sf[:, 1]) * len(new_verts) + sf[:, 2]
    _, first = np.unique(code, return_index=True)
    return new_verts, nf[np.sort(first)]


def main():
    t0 = time.time()
    from scantobim.core.mesh import Mesh
    from scantobim.core.phototex import bake_photo_atlas
    from scantobim.io.writers import write_mesh
    from scantobim.photogrammetry.colmap import read_colmap_model

    print("PLY laden …", flush=True)
    verts, faces = read_binary_ply(WS / "mvs" / "scene_mesh.ply")
    print(f"  {len(verts):,} Vertices, {len(faces):,} Faces")
    verts_d, faces_d = decimate_grid(verts, faces, cell=0.02)
    print(f"  dezimiert (2 cm): {len(verts_d):,} Vertices, "
          f"{len(faces_d):,} Faces")
    mesh = Mesh(vertices=verts_d, faces=faces_d)

    cameras = read_colmap_model(WS / "sparse_tri")
    print(f"  {len(cameras)} Kameras aus sparse_tri")
    stats: dict = {}
    textured = bake_photo_atlas(
        mesh, cameras, WS / "images",
        stats_out=stats, max_atlas=8192, max_pages=4,
    )
    assert textured is not None, "Atlas-Bake lieferte nichts"
    print("  Atlas:", stats.get("atlas"), "Seiten:", stats.get("pages"),
          "Texel:", stats.get("texel_cm"), "cm  Foto-Anteil:",
          stats.get("photo_fraction"))
    write_mesh(textured, OUT)
    print(f"FERTIG in {time.time() - t0:.0f}s: {OUT}")


if __name__ == "__main__":
    main()
