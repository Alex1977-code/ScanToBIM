"""Foto-Verifikations-Culling: Geister-Dreiecke sterben am Sehstrahl.

Das Prinzip des Nutzers, direkt umgesetzt: Die LiDAR-Wolke gibt die
Struktur vor — aber ob ein Oberflaechenstueck WIRKLICH existiert,
entscheiden die Fotos aus mehreren Blickrichtungen. Ein Dreieck der
zerfransten Sorte (SLAM-Geister, Streifschuss-Punkte, Voxel-Schaum am
Dachrand) haengt frei im Raum: Kameras, die es sehen muessten, sehen an
seiner Bildposition uebereinstimmend eine ANDERE Tiefe (die echte Wand
oder den Hintergrund dahinter). Genau das wird gemessen:

1. Kandidaten sind Dreiecke nahe offener Raender (dort lebt der Franz).
2. Pro Ansicht: Z-Buffer-Sichtbarkeit, dann NCC-Tiefensuche entlang des
   Sehstrahls um die aktuelle Dreieckstiefe (wie MVS, aber am Dreieck
   verankert statt am LiDAR-Prior).
3. Sagen >= 2 Ansichten unabhaengig "die fotokonsistente Flaeche liegt
   deutlich woanders auf diesem Strahl" und bestaetigt KEINE Ansicht das
   Dreieck, wird es entfernt. Danach fliegen isolierte Kruemel-Inseln.

Echte Geometrie besteht den Test, denn an ihrer Position stimmen die
Fotos ueberein (Confirm); texturlose oder verdeckte Stellen bleiben
unangetastet (kein Verdikt = kein Loeschen).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_NCC_REFUTE = 0.68    # so klar muss die Alternativ-Tiefe belegt sein
_NCC_CONFIRM = 0.50   # so viel Konsistenz bestaetigt die Ist-Position
_DEV_REFUTE = 0.055   # m — Alternativ-Tiefe muss deutlich woanders liegen
_DEV_CONFIRM = 0.030  # m — Best-Tiefe nahe der Ist-Position = bestaetigt
_MARGIN_GAIN = 0.15   # Best-NCC muss die Ist-Position klar schlagen


def _candidate_faces(
    mesh, rings: int = 1, chaos_dot: float = 0.55, cap: int = 900_000
):
    """Fransen-Kandidaten: offene Raender UND Normalen-Chaos-Zonen.

    Voxel-Meshes sind fast geschlossen — der Konfetti-Franz besteht aus
    DUENNEN GESCHLOSSENEN Schlaeuchen ohne offene Kanten (beobachtet:
    nur 793 Rand-Kandidaten auf einem 4-Mio-Dreiecke-Mesh). Sein
    verlaessliches Kennzeichen ist das Normalen-Chaos: benachbarte
    Dreiecke zeigen in wild verschiedene Richtungen. ACHTUNG Kalibrierung:
    Marching-Cubes-Nachbarn sind auch auf sauberen Flaechen maessig
    uneinig — mit lockerer Schwelle (0.8) war praktisch das GANZE Mesh
    Kandidat und das Pro-Ansicht-Budget verduennte die Pruefung wirkungslos
    (beobachtet: 6.65 Mio Kandidaten, 24 Widerlegungen). Deshalb: nur
    ECHTES Chaos (mittleres |dot| < 0.55), kaum Ring-Aufblaehung, und ein
    globales Budget, das die chaotischsten Dreiecke zuerst nimmt.
    Rueckgabe: (kandidaten, prioritaet) — Prioritaet klein = chaotisch.
    """
    faces = mesh.faces
    n_v = int(faces.max()) + 1
    ea = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    eb = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    codes = np.minimum(ea, eb) * n_v + np.maximum(ea, eb)
    face_of = np.tile(np.arange(len(faces), dtype=np.int64), 3)
    order = np.argsort(codes, kind="stable")
    cs, fo = codes[order], face_of[order]
    first = np.concatenate([[True], cs[1:] != cs[:-1]])
    starts = np.flatnonzero(first)
    counts = np.diff(np.concatenate([starts, [len(cs)]]))
    seed_v = np.zeros(n_v, dtype=bool)
    boundary_codes = cs[starts[counts == 1]]
    if len(boundary_codes):
        b_set = np.isin(codes, boundary_codes)
        seed_v[ea[b_set]] = True
        seed_v[eb[b_set]] = True
    # Normalen-Chaos ueber Kanten-Nachbarn.
    same = cs[1:] == cs[:-1]
    fa, fb = fo[:-1][same], fo[1:][same]
    tri = mesh.vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    dot = np.abs(np.einsum("ij,ij->i", fn[fa], fn[fb]))
    acc = np.zeros(len(faces))
    cnt = np.zeros(len(faces))
    np.add.at(acc, fa, dot)
    np.add.at(acc, fb, dot)
    np.add.at(cnt, fa, 1.0)
    np.add.at(cnt, fb, 1.0)
    mean_dot = acc / np.maximum(cnt, 1.0)
    chaotic = mean_dot < chaos_dot
    seed_v[faces[chaotic].ravel()] = True
    mark = np.zeros(len(faces), dtype=bool)
    for _ in range(max(1, rings)):
        mark |= seed_v[faces].any(axis=1)
        seed_v[faces[mark]] = True
    cand = np.flatnonzero(mark)
    if len(cand) > cap:
        order = np.argsort(mean_dot[cand])
        cand = cand[order[:cap]]
    return np.sort(cand), mean_dot


def _drop_small_components(faces: np.ndarray, keep: np.ndarray,
                           min_faces: int = 40) -> np.ndarray:
    """Also drop face-connected islands smaller than ``min_faces``."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    sel = np.flatnonzero(keep)
    if not len(sel):
        return keep
    f = faces[sel]
    n_v = int(faces.max()) + 1
    ea = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    eb = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    codes = np.minimum(ea, eb) * n_v + np.maximum(ea, eb)
    face_of = np.tile(np.arange(len(f), dtype=np.int64), 3)
    order = np.argsort(codes, kind="stable")
    cs, fo = codes[order], face_of[order]
    same = cs[1:] == cs[:-1]
    fa, fb = fo[:-1][same], fo[1:][same]
    graph = sparse.coo_matrix(
        (np.ones(len(fa), dtype=np.int8), (fa, fb)), shape=(len(f), len(f))
    )
    _, comp = connected_components(graph, directed=False)
    sizes = np.bincount(comp)
    keep2 = keep.copy()
    keep2[sel[sizes[comp] < min_faces]] = False
    return keep2


def cull_ghost_faces(
    mesh,
    cameras,
    images_dir,
    image_map: dict | None = None,
    stats_out: dict | None = None,
    max_views: int = 60,
    scale: float = 0.35,
    band: float = 0.12,
    steps: int = 21,
    patch: int = 2,
    min_refutes: int = 2,
    rings: int = 3,
):
    """Remove photo-refuted boundary-zone faces from ``mesh`` (in place).

    Returns the number of removed faces (0 when nothing was refuted or
    prerequisites are missing).
    """
    try:
        from PIL import Image
    except ImportError:
        return 0
    from scantobim.core import accel
    from scantobim.core.mvs import (
        _bilinear_xp,
        _gray,
        _pick_neighbors,
        _project_xp,
    )
    from scantobim.core.texture import _index_images, _resolve_cameras

    cameras = _resolve_cameras(cameras)
    if not cameras or mesh is None or len(mesh.faces) < 100:
        return 0
    index = dict(image_map) if image_map else _index_images(Path(images_dir))

    faces = mesh.faces
    cand, chaos_prio = _candidate_faces(mesh, rings=rings)
    if not len(cand):
        return 0
    tri = mesh.vertices[faces]
    centers = tri.mean(axis=1)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)

    n_cam = len(cameras)
    ref_idx = (
        list(np.linspace(0, n_cam - 1, max_views).astype(int))
        if n_cam > max_views else list(range(n_cam))
    )
    cam_centers = np.array([-c.rotation.T @ c.translation for c in cameras])
    needed = set(ref_idx)
    for ci in list(needed):
        for cj in _pick_neighbors(ci, cameras, cam_centers.copy()):
            needed.add(cj)

    grays: dict[int, np.ndarray] = {}

    def _load(ci):
        cam = cameras[ci]
        path = index.get(cam.name) or index.get(Path(cam.name).name)
        if path is None or not path.exists():
            return
        try:
            img = Image.open(path).convert("L")
            w = max(2, int(round(cam.width * scale)))
            h = max(2, int(round(cam.height * scale)))
            grays[ci] = np.asarray(
                img.resize((w, h), Image.BILINEAR), dtype=np.float32
            )
        except Exception:  # noqa: BLE001
            pass

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(_load, sorted(needed)))

    refutes = np.zeros(len(faces), dtype=np.int16)
    confirms = np.zeros(len(faces), dtype=np.int16)
    offs = np.arange(-patch, patch + 1)
    dgx, dgy = np.meshgrid(offs, offs)
    dgx, dgy = dgx.ravel(), dgy.ravel()
    p_n = len(dgx)
    deltas = np.linspace(-band, band, steps)
    k_mid = steps // 2
    xp = accel.gpu() or np

    for ci in ref_idx:
        if ci not in grays:
            continue
        cam = cameras[ci]
        gray = grays[ci]
        h, w = gray.shape
        view_scale = w / cam.width
        pc_all = centers @ cam.rotation.T + cam.translation
        front = pc_all[:, 2] > 0.5
        px_all, py_all = cam.project(pc_all)
        gx = np.round(px_all * view_scale).astype(np.int64)
        gy = np.round(py_all * view_scale).astype(np.int64)
        inside = front & (gx >= 4) & (gx < w - 4) & (gy >= 4) & (gy < h - 4)
        # Z-Buffer aus ALLEN Face-Zentren: verdeckte Kandidaten kriegen
        # kein Verdikt (die Fotos koennen sie nicht beurteilen).
        zw = 300
        zh = max(2, int(round(zw * h / w)))
        zx = np.clip((gx * zw) // max(w, 1), 0, zw - 1)
        zy = np.clip((gy * zh) // max(h, 1), 0, zh - 1)
        zbuf = np.full(zw * zh, np.inf, dtype=np.float64)
        lin_all = zy * zw + zx
        np.minimum.at(zbuf[:], lin_all[inside], pc_all[inside][:, 2])
        cam_c = -cam.rotation.T @ cam.translation
        view_v = cam_c[None, :] - centers
        d_len = np.linalg.norm(view_v, axis=1)
        facing = np.abs((fn * view_v).sum(axis=1)) / np.maximum(d_len, 1e-9)
        vis = (
            inside
            & (facing > 0.25)
            & (pc_all[:, 2] <= zbuf[lin_all] + np.maximum(
                0.12, 0.05 * pc_all[:, 2]
            ))
        )
        sel = cand[vis[cand]]
        if len(sel) < 20:
            continue
        if len(sel) > 45_000:
            # Die chaotischsten Dreiecke zuerst — ein Zufallsschnitt
            # verduennt die Pruefung unter die 2-Widerlegungen-Schwelle.
            order = np.argsort(chaos_prio[sel])
            sel = np.sort(sel[order[:45_000]])

        neighbors = [
            cj for cj in _pick_neighbors(ci, cameras, cam_centers.copy())
            if cj in grays
        ]
        if not neighbors:
            continue
        n = len(sel)
        px_f = (gx[sel][:, None] + dgx[None, :]) / view_scale
        py_f = (gy[sel][:, None] + dgy[None, :]) / view_scale
        rays = cam.unproject(px_f.ravel(), py_f.ravel()).reshape(n, p_n, 3)
        rz = np.maximum(rays[:, :, 2], 1e-6)
        ref_patch = gray[
            gy[sel][:, None] + dgy[None, :], gx[sel][:, None] + dgx[None, :]
        ].astype(np.float32)
        ref_n = ref_patch - ref_patch.mean(axis=1, keepdims=True)
        ref_std = np.sqrt((ref_n**2).mean(axis=1)) + 1e-4
        d_face = pc_all[sel][:, 2]
        depths = np.maximum(d_face[:, None] + deltas[None, :], 0.3)

        rays_x = xp.asarray(rays, dtype=xp.float32)
        rz_x = xp.asarray(rz, dtype=xp.float32)
        ref_n_x = xp.asarray(ref_n, dtype=xp.float32)
        ref_std_x = xp.asarray(ref_std, dtype=xp.float32)
        depths_x = xp.asarray(depths, dtype=xp.float32)
        r_ref_t = xp.asarray(cam.rotation.T, dtype=xp.float32)
        t_ref = xp.asarray(cam.translation, dtype=xp.float32)
        ncc_sum = xp.zeros((n, steps), dtype=xp.float32)
        ncc_cnt = xp.zeros((n, steps), dtype=xp.float32)
        for cj in neighbors:
            nb = cameras[cj]
            gray_n = xp.asarray(grays[cj], dtype=xp.float32)
            nb_scale = grays[cj].shape[1] / nb.width
            r_n = xp.asarray(nb.rotation, dtype=xp.float32)
            t_n = xp.asarray(nb.translation, dtype=xp.float32)
            rel_r = r_n @ r_ref_t
            rel_t = t_n - rel_r @ t_ref
            for k in range(steps):
                z = depths_x[:, k][:, None] / rz_x
                pc_ref = rays_x * z[:, :, None]
                pc_n = pc_ref @ rel_r.T + rel_t
                flat = pc_n.reshape(-1, 3)
                if xp is np:
                    qx, qy = nb.project(flat)
                else:
                    qx, qy = _project_xp(xp, nb, flat)
                vals = _bilinear_xp(
                    xp, gray_n,
                    (qx * nb_scale).reshape(n, p_n),
                    (qy * nb_scale).reshape(n, p_n),
                )
                behind = (flat[:, 2] <= 0.1).reshape(n, p_n)
                vals = xp.where(behind, xp.float32(np.nan), vals)
                v_n = vals - xp.nanmean(vals, axis=1, keepdims=True)
                v_n = xp.where(xp.isnan(v_n), xp.float32(0.0), v_n)
                v_std = xp.sqrt((v_n**2).mean(axis=1)) + 1e-4
                ncc = (ref_n_x * v_n).mean(axis=1) / (ref_std_x * v_std)
                good = xp.isfinite(ncc)
                ncc_sum[:, k] += xp.where(good, ncc, 0.0)
                ncc_cnt[:, k] += good.astype(xp.float32)
        ncc = ncc_sum / xp.maximum(ncc_cnt, 1.0)
        kbest = xp.argmax(ncc, axis=1)
        rows = xp.arange(n)
        from scantobim.core.accel import asnumpy

        best = asnumpy(ncc[rows, kbest])
        at_face = asnumpy(ncc[:, k_mid])
        d_best = asnumpy(depths_x[rows, kbest]).astype(np.float64)
        dev = np.abs(d_best - d_face)
        refute = (
            (best >= _NCC_REFUTE)
            & (dev > _DEV_REFUTE)
            & (best - at_face >= _MARGIN_GAIN)
        )
        confirm = (at_face >= _NCC_CONFIRM) | (dev <= _DEV_CONFIRM)
        refutes[sel[refute]] += 1
        confirms[sel[confirm & ~refute]] += 1

    kill = (refutes >= min_refutes) & (confirms == 0)
    keep = ~kill
    keep = _drop_small_components(faces, keep)
    n_removed = int((~keep).sum())
    if n_removed and n_removed < 0.5 * len(faces):
        mesh.faces = faces[keep]
        if mesh.face_groups is not None and len(mesh.face_groups) == len(keep):
            mesh.face_groups = mesh.face_groups[keep]
        if (
            getattr(mesh, "face_page", None) is not None
            and len(mesh.face_page) == len(keep)
        ):
            mesh.face_page = mesh.face_page[keep]
    else:
        n_removed = 0
    if stats_out is not None:
        stats_out["foto_verifikation"] = {
            "kandidaten": int(len(cand)),
            "widerlegt": int(kill.sum()),
            "entfernt": n_removed,
        }
    return n_removed
