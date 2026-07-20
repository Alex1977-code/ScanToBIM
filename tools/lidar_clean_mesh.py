"""LiDAR-Säuberung des Photogrammetrie-Modells + Genauigkeitsmessung.

Der Vorteil gegenüber reiner Photogrammetrie: Die LiDAR-Wolke ist ein
EXISTENZ-BEWEIS. Mesh-Flächen, in deren Nähe kein einziger LiDAR-Punkt
liegt (Himmel-Fahnen, Wasser-Spiegelungen, Textur-Geister), werden
entfernt — echte Geometrie hat immer LiDAR-Belege. Danach fallen
isolierte Kruemel-Inseln, und als Beleg fuer "genau" wird der Abstand
Mesh->LiDAR gemessen (Median/RMS).
"""

from __future__ import annotations

import io
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

GLB_IN = Path(r"C:\Users\alexanderm\Desktop\ScanToBIM\colmap_ws\mvs\scene_bake.glb")
GLB_OUT = Path(r"C:\Users\alexanderm\Desktop\ScanToBIM\colmap_ws\mvs\scene_clean.glb")
LAS = Path(
    r"C:\Users\alexanderm\Desktop\steuerhaus_nur_aussen"
    r"\Steuerhaus nur aussen\steuerhaus_nur_aussen\output"
    r"\steuerhaus_nur_aussen_uncolorized.las"
)
DIST_KEEP = 0.30      # m — so nah muss LiDAR-Beleg liegen
MIN_COMPONENT = 120   # Faces — kleinere Inseln fliegen


def load_glb(path):
    with open(path, "rb") as f:
        f.read(12)
        clen, _ = struct.unpack("<2I", f.read(8))
        gltf = json.loads(f.read(clen))
        blen, _ = struct.unpack("<2I", f.read(8))
        buf = f.read(blen)
    return gltf, buf


def accessor(gltf, buf, idx):
    acc = gltf["accessors"][idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    comp = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32,
            5126: np.float32}[acc["componentType"]]
    n_comp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
    arr = np.frombuffer(buf, dtype=comp, count=acc["count"] * n_comp,
                        offset=off)
    return arr.reshape(acc["count"], n_comp) if n_comp > 1 else arr


def main():
    t0 = time.time()
    import laspy
    from PIL import Image
    from scipy.spatial import cKDTree

    from scantobim.core.mesh import Mesh
    from scantobim.core.photocull import _drop_small_components
    from scantobim.io.writers import write_mesh

    gltf, buf = load_glb(GLB_IN)
    prim0 = gltf["meshes"][0]["primitives"][0]
    verts = accessor(gltf, buf, prim0["attributes"]["POSITION"]).astype(
        np.float64
    )
    uvs = accessor(gltf, buf, prim0["attributes"]["TEXCOORD_0"]).astype(
        np.float64
    )
    faces_l, page_l = [], []
    images = []
    for img in gltf.get("images", []):
        bv = gltf["bufferViews"][img["bufferView"]]
        off = bv.get("byteOffset", 0)
        images.append(np.asarray(Image.open(
            io.BytesIO(buf[off:off + bv["byteLength"]])
        ).convert("RGB")))
    for p_i, prim in enumerate(gltf["meshes"][0]["primitives"]):
        idxs = accessor(gltf, buf, prim["indices"]).astype(np.int64)
        idxs = idxs.reshape(-1, 3)
        faces_l.append(idxs)
        page_l.append(np.full(len(idxs), p_i, dtype=np.int32))
    faces = np.vstack(faces_l)
    face_page = np.concatenate(page_l)
    print(f"Modell: {len(verts):,} Vertices, {len(faces):,} Faces, "
          f"{len(images)} Atlas-Seiten")

    las = laspy.read(LAS)
    pts = np.column_stack([las.x, las.y, las.z])
    if len(pts) > 4_000_000:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), 4_000_000, replace=False)]
    tree = cKDTree(pts)
    print(f"LiDAR-Beleg: {len(pts):,} Punkte im Baum")

    centers = verts[faces].mean(axis=1)
    dist = np.empty(len(centers))
    for s in range(0, len(centers), 500_000):
        d, _ = tree.query(centers[s:s + 500_000], k=1, workers=-1)
        dist[s:s + 500_000] = d
    keep = dist <= DIST_KEEP
    print(f"LiDAR-Beleg fehlt bei {(~keep).sum():,} Faces "
          f"({(~keep).mean() * 100:.1f}%) — werden entfernt")
    keep = _drop_small_components(faces, keep, min_faces=MIN_COMPONENT)
    print(f"nach Kruemel-Sweep: {(~keep).sum():,} Faces entfernt")

    mesh = Mesh(
        vertices=verts, faces=faces[keep], uvs=uvs,
        texture=images[0] if images else None,
        face_page=face_page[keep] if len(images) > 1 else None,
    )
    mesh.textures = images
    write_mesh(mesh, GLB_OUT)
    print(f"wrote {GLB_OUT}")

    # Genauigkeit: Mesh-Vertices -> LiDAR (nur behaltene Flaechen).
    used = np.unique(faces[keep])
    sample = used
    if len(sample) > 400_000:
        rng = np.random.default_rng(1)
        sample = rng.choice(used, 400_000, replace=False)
    dv, _ = tree.query(verts[sample], k=1, workers=-1)
    inlier = dv[dv <= 0.2]
    print(
        f"GENAUIGKEIT Mesh->LiDAR: Median {np.median(dv) * 1000:.1f} mm, "
        f"RMS(<=20cm) {np.sqrt((inlier**2).mean()) * 1000:.1f} mm, "
        f"Anteil <=5cm: {(dv <= 0.05).mean() * 100:.1f}%"
    )
    print(f"FERTIG in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
