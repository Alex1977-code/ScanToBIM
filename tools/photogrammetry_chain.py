"""Photogrammetrie-Kette: COLMAP (CUDA) + OpenMVS auf dem S20-Workspace.

Voraussetzungen: tools/colmap_export.py hat colmap_ws/{images,sparse_known}
erzeugt; COLMAP und OpenMVS liegen entpackt unter C:\\Users\\alexanderm\\Tools.

Stufen (jede überspringbar, wenn ihr Ergebnis existiert — Neustart-sicher):
  1. feature_extractor  (SIFT, GPU)
  2. sequential_matcher (Trajektorien-Reihenfolge, GPU)
  3. images.txt-IDs an die Datenbank angleichen (COLMAP verlangt DB-IDs)
  4. point_triangulator (Posen FIX — metrischer Massstab bleibt)
  5. image_undistorter  (Workspace-Layout; Bilder sind schon pinhole)
  6. patch_match_stereo (CUDA-Tiefenkarten)
  7. stereo_fusion      (dichte Wolke)
  8. OpenMVS: InterfaceCOLMAP -> ReconstructMesh -> RefineMesh -> TextureMesh
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from pathlib import Path

WS = Path(r"C:\Users\alexanderm\Desktop\ScanToBIM\colmap_ws")
TOOLS = Path(r"C:\Users\alexanderm\Tools")
COLMAP = None
for cand in TOOLS.rglob("colmap.exe"):
    COLMAP = cand
    break
OPENMVS = None
for cand in TOOLS.rglob("InterfaceCOLMAP.exe"):
    OPENMVS = cand.parent
    break

DB = WS / "db.db"
SPARSE_KNOWN = WS / "sparse_known"
SPARSE_TRI = WS / "sparse_tri"
DENSE = WS / "dense"
MVS = WS / "mvs"


def run(name: str, args: list, done_marker: Path | None = None):
    if done_marker is not None and done_marker.exists():
        print(f"[{name}] übersprungen (Ergebnis vorhanden)")
        return
    t0 = time.time()
    print(f"[{name}] startet …", flush=True)
    r = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"[{name}] FEHLER nach {dt:.0f}s:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        sys.exit(1)
    print(f"[{name}] fertig in {dt:.0f}s", flush=True)


def rewrite_model_41():
    """sparse_known im COLMAP-4.1-Format neu schreiben, deckungsgleich zur
    Datenbank: pro Bild eigene Kamera/Rig/Frame mit identischer ID (so hat
    sie der feature_extractor angelegt). Posen kommen aus der bestehenden
    images.txt (per NAME gematcht) — sie bleiben die metrischen
    LiDAR-registrierten Posen."""
    # Bestehende Posen nach Name einlesen (altes wie neues Format).
    poses = {}
    for line in (SPARSE_KNOWN / "images.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 10 and p[9].endswith(".jpg"):
            poses[p[9]] = p[1:8]  # QW QX QY QZ TX TY TZ
    con = sqlite3.connect(DB)
    rows = sorted(con.execute(
        "SELECT image_id, name, camera_id FROM images"
    ))
    cam_params = {}
    for cid, model, w, h, params in con.execute(
        "SELECT camera_id, model, width, height, params FROM cameras"
    ):
        import struct as _st

        vals = _st.unpack(f"<{len(params) // 8}d", params)
        cam_params[cid] = (int(w), int(h), vals)
    # Die ECHTEN Zuordnungen der Datenbank uebernehmen — image_id und
    # camera_id laufen NICHT synchron (asynchrone Extractor-Threads):
    # Bild -> Frame via frame_data(data_id), Frame -> Rig via frames,
    # Rig -> Referenzsensor via rigs.
    frame_of_img = {}
    sensor_of_frame = {}
    for fid, did, sid, stype in con.execute(
        "SELECT frame_id, data_id, sensor_id, sensor_type FROM frame_data"
    ):
        frame_of_img[did] = fid
        sensor_of_frame[fid] = sid
    rig_of_frame = dict(con.execute("SELECT frame_id, rig_id FROM frames"))
    ref_of_rig = dict(con.execute("SELECT rig_id, ref_sensor_id FROM rigs"))
    con.close()

    cams, imgs, rigs, frames = [], [], [], []
    seen_rigs = set()
    n = 0
    for iid, name, cid in rows:
        if name not in poses or iid not in frame_of_img:
            continue
        q = poses[name]
        w, h, vals = cam_params[cid]
        fid = frame_of_img[iid]
        rid = rig_of_frame[fid]
        cams.append(
            f"{cid} PINHOLE {w} {h} " + " ".join(f"{v:.10g}" for v in vals)
        )
        imgs.append(f"{iid} {' '.join(q)} {cid} {name}")
        imgs.append("")
        if rid not in seen_rigs:
            rigs.append(f"{rid} 1 CAMERA {ref_of_rig[rid]}")
            seen_rigs.add(rid)
        frames.append(
            f"{fid} {rid} {' '.join(q)} 1 CAMERA "
            f"{sensor_of_frame[fid]} {iid}"
        )
        n += 1
    (SPARSE_KNOWN / "cameras.txt").write_text(
        "# COLMAP 4.1\n" + "\n".join(cams) + "\n", encoding="utf-8"
    )
    (SPARSE_KNOWN / "images.txt").write_text(
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
        + "\n".join(imgs) + "\n", encoding="utf-8"
    )
    (SPARSE_KNOWN / "rigs.txt").write_text(
        "# RIG_ID NUM_SENSORS REF_SENSOR_TYPE REF_SENSOR_ID\n"
        + "\n".join(rigs) + "\n", encoding="utf-8"
    )
    (SPARSE_KNOWN / "frames.txt").write_text(
        "# FRAME_ID RIG_ID QW QX QY QZ TX TY TZ NUM CAMERA SENSOR DATA\n"
        + "\n".join(frames) + "\n", encoding="utf-8"
    )
    (SPARSE_KNOWN / "points3D.txt").write_text("", encoding="utf-8")
    print(f"[modell] {n} Bilder im 4.1-Format geschrieben "
          "(Kamera/Rig/Frame je Bild, DB-deckungsgleich)")


def main():
    assert COLMAP is not None, "colmap.exe nicht unter C:\\Users\\alexanderm\\Tools gefunden"
    assert OPENMVS is not None, "OpenMVS nicht gefunden"
    print(f"COLMAP:  {COLMAP}\nOpenMVS: {OPENMVS}")
    n_img = len(list((WS / "images").glob("*.jpg")))
    print(f"Workspace: {WS} — {n_img} Bilder")

    run("features", [
        COLMAP, "feature_extractor",
        "--database_path", DB, "--image_path", WS / "images",
        "--ImageReader.camera_model", "PINHOLE",
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.max_image_size", "2400",
    ], done_marker=DB)

    matched = WS / ".matched"
    run("matching", [
        COLMAP, "sequential_matcher",
        "--database_path", DB,
        "--SequentialMatching.overlap", "12",
        "--FeatureMatching.use_gpu", "1",
    ], done_marker=matched)
    matched.touch()

    if not (SPARSE_TRI / "images.bin").exists():
        rewrite_model_41()
        SPARSE_TRI.mkdir(exist_ok=True)
        run("triangulation", [
            COLMAP, "point_triangulator",
            "--database_path", DB, "--image_path", WS / "images",
            "--input_path", SPARSE_KNOWN, "--output_path", SPARSE_TRI,
        ])
    else:
        print("[triangulation] übersprungen")

    run("undistort", [
        COLMAP, "image_undistorter",
        "--image_path", WS / "images", "--input_path", SPARSE_TRI,
        "--output_path", DENSE,
    ], done_marker=DENSE / "sparse")

    run("patchmatch", [
        COLMAP, "patch_match_stereo",
        "--workspace_path", DENSE,
        "--PatchMatchStereo.max_image_size", "1600",
        "--PatchMatchStereo.geom_consistency", "1",
    ], done_marker=DENSE / "stereo" / "depth_maps" / "left_00002.jpg.geometric.bin")

    run("fusion", [
        COLMAP, "stereo_fusion",
        "--workspace_path", DENSE,
        "--output_path", DENSE / "fused.ply",
    ], done_marker=DENSE / "fused.ply")

    MVS.mkdir(exist_ok=True)
    run("interface", [
        OPENMVS / "InterfaceCOLMAP.exe",
        "-i", DENSE, "-o", MVS / "scene.mvs",
        "--image-folder", DENSE / "images",
        "-w", MVS,
    ], done_marker=MVS / "scene.mvs")

    run("mesh", [
        OPENMVS / "ReconstructMesh.exe", MVS / "scene.mvs",
        "-w", MVS, "-o", MVS / "scene_mesh.mvs",
    ], done_marker=MVS / "scene_mesh.mvs")

    run("refine", [
        OPENMVS / "RefineMesh.exe", MVS / "scene_mesh.mvs",
        "-w", MVS, "-o", MVS / "scene_refined.mvs",
        "--resolution-level", "2", "--max-face-area", "16",
    ], done_marker=MVS / "scene_refined.mvs")

    run("texture", [
        OPENMVS / "TextureMesh.exe", MVS / "scene_refined.mvs",
        "-w", MVS, "-o", MVS / "scene_textured.mvs",
        "--export-type", "obj",
    ], done_marker=MVS / "scene_textured.obj")

    print("FERTIG:", MVS / "scene_textured.obj")


if __name__ == "__main__":
    main()
