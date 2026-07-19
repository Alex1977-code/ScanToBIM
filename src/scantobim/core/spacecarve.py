"""Freiraum-Carving: Geisterpunkte sterben am LiDAR-Sehstrahl.

Jeder LiDAR-Punkt ist ein Beweis: Zwischen Sensorposition und Punkt war
FREIER RAUM. Ein Geisterpunkt (SLAM-Doppelwand, Streifschuss-Artefakt,
Mixed-Pixel am Dachrand) schwebt vor der echten Oberflaeche — aus vielen
anderen Sensorpositionen betrachtet liegt er klar VOR der Konsens-Tiefe
seiner Blickrichtung, waehrend ihn fast keine Ansicht stuetzt. Genau die
Punkte werden entfernt, BEVOR gemesht wird — die Fransen verlieren ihr
Baumaterial an der Wurzel statt im Mesh-Nachhinein.

Umsetzung: entlang der Trajektorie werden virtuelle Panorama-Ansichten
aufgespannt (Azimut/Elevations-Raster). Pro Zelle entsteht ueber ein
Tiefen-Histogramm eine robuste Konsens-Tiefe (oberes Quantil — die echte
Oberflaeche, nicht der vorgelagerte Ausreisser). Ein Punkt gilt in einer
Ansicht als DURCHSCHOSSEN, wenn er deutlich vor dem Konsens liegt, und
als GESTUETZT, wenn er nahe daran liegt. Wer ueber alle Ansichten
mehrfach durchschossen und kaum gestuetzt wird, fliegt.
"""

from __future__ import annotations

import numpy as np

_AZ_BINS = 512
_EL_BINS = 160
_DEPTH_BUCKETS = 96
_R_MIN = 0.8
_R_MAX = 120.0
_CONSENSUS_Q = 0.60   # oberes Quantil je Zelle = Oberflaechen-Konsens


def _view_positions(trajectory: np.ndarray, max_views: int = 110):
    pos = trajectory[:, :3] if trajectory.shape[1] >= 3 else trajectory
    if len(pos) > max_views:
        idx = np.linspace(0, len(pos) - 1, max_views).astype(np.int64)
        pos = pos[idx]
    return pos


def carve_ghost_points(
    points: np.ndarray,
    trajectory: np.ndarray,
    stats_out: dict | None = None,
    max_views: int = 110,
    min_transit: int = 3,
    support_slack: float = 0.10,
) -> np.ndarray | None:
    """Maske der Punkte, die BLEIBEN (True = behalten), oder None.

    ``min_transit``: so oft muss ein Punkt durchschossen werden, bevor er
    faellt — und er darf hoechstens gleich oft gestuetzt sein.
    """
    if trajectory is None or len(trajectory) < 2 or len(points) < 10_000:
        return None
    from scantobim.core import accel

    xp = accel.gpu() or np
    pts = xp.asarray(points, dtype=xp.float32)
    n = len(points)
    transit = xp.zeros(n, dtype=xp.int16)
    support = xp.zeros(n, dtype=xp.int16)
    log_rmin = np.log(_R_MIN)
    log_span = np.log(_R_MAX) - log_rmin

    for view_pos in _view_positions(np.asarray(trajectory), max_views):
        s = xp.asarray(view_pos[:3], dtype=xp.float32)
        rel = pts - s
        r = xp.sqrt((rel * rel).sum(axis=1))
        ok = (r > _R_MIN) & (r < _R_MAX)
        az = xp.arctan2(rel[:, 1], rel[:, 0])
        el = xp.arcsin(xp.clip(rel[:, 2] / xp.maximum(r, 1e-6), -1, 1))
        ai = ((az + np.pi) / (2 * np.pi) * _AZ_BINS).astype(xp.int64)
        ai = xp.clip(ai, 0, _AZ_BINS - 1)
        ei = ((el + np.pi / 2) / np.pi * _EL_BINS).astype(xp.int64)
        ei = xp.clip(ei, 0, _EL_BINS - 1)
        cell = ai * _EL_BINS + ei
        # Tiefe logarithmisch quantisieren (nah fein, fern grob).
        db = (
            (xp.log(xp.maximum(r, _R_MIN)) - log_rmin) / log_span
            * _DEPTH_BUCKETS
        ).astype(xp.int64)
        db = xp.clip(db, 0, _DEPTH_BUCKETS - 1)
        # Histogramm (Zelle x Tiefenklasse) und Konsens-Quantil je Zelle.
        hist = xp.zeros(_AZ_BINS * _EL_BINS * _DEPTH_BUCKETS, dtype=xp.int32)
        flat = cell * _DEPTH_BUCKETS + db
        if xp is np:
            np.add.at(hist, flat[ok], 1)
        else:  # pragma: no cover — GPU
            xp.add.at(hist, flat[ok], 1)
        hist = hist.reshape(-1, _DEPTH_BUCKETS)
        total = hist.sum(axis=1)
        csum = xp.cumsum(hist, axis=1)
        target = (total * _CONSENSUS_Q).astype(xp.int64)[:, None]
        cons_bucket = (csum < target).sum(axis=1)
        # Konsens-Tiefe: Untergrenze des Quantil-Buckets.
        cons_r = xp.exp(
            log_rmin + cons_bucket.astype(xp.float32)
            / _DEPTH_BUCKETS * log_span
        )
        cell_valid = total >= 6  # zu duenn besetzte Zellen urteilen nicht
        c_r = cons_r[cell]
        c_ok = ok & cell_valid[cell]
        margin = xp.maximum(0.25, 0.05 * c_r)
        is_transit = c_ok & (r < c_r - margin)
        is_support = c_ok & (xp.abs(r - c_r) <= xp.maximum(
            support_slack + 0.02 * c_r, 0.12
        ))
        transit += is_transit.astype(xp.int16)
        support += is_support.astype(xp.int16)

    from scantobim.core.accel import asnumpy

    transit = asnumpy(transit)
    support = asnumpy(support)
    ghost = (transit >= min_transit) & (transit > 2 * support)
    keep = ~ghost
    if stats_out is not None:
        stats_out["freiraum_carving"] = {
            "ansichten": int(len(_view_positions(
                np.asarray(trajectory), max_views
            ))),
            "entfernt": int(ghost.sum()),
            "anteil": round(float(ghost.mean()), 4),
        }
    return keep
