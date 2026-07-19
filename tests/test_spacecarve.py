"""Freiraum-Carving: Geister vor der Wand fallen, Wand und Boden bleiben."""

import numpy as np

from scantobim.core.spacecarve import carve_ghost_points


def test_ghosts_carved_surfaces_survive():
    rng = np.random.default_rng(5)
    # Wand bei x=5 (y/z-Raster), Boden bei z=0.
    wy = rng.uniform(-4, 4, 60_000)
    wz = rng.uniform(0, 4, 60_000)
    wall = np.column_stack([
        np.full(60_000, 5.0) + rng.normal(0, 0.01, 60_000), wy, wz
    ])
    gx = rng.uniform(0.5, 4.5, 50_000)
    gy = rng.uniform(-4, 4, 50_000)
    ground = np.column_stack([gx, gy, rng.normal(0.0, 0.01, 50_000)])
    # Geister: duenne Wolke, die 2.5 m VOR der Wand schwebt.
    ghost = np.column_stack([
        np.full(2_000, 2.5) + rng.normal(0, 0.05, 2_000),
        rng.uniform(-3, 3, 2_000),
        rng.uniform(0.8, 3.0, 2_000),
    ])
    pts = np.vstack([wall, ground, ghost])
    n_wall, n_ground, n_ghost = len(wall), len(ground), len(ghost)

    # Trajektorie: Sensorpositionen auf einer Linie vor der Szene.
    ty = np.linspace(-3.5, 3.5, 60)
    traj = np.column_stack([
        np.full(60, 0.0), ty, np.full(60, 1.5)
    ])

    stats: dict = {}
    keep = carve_ghost_points(pts, traj, stats_out=stats, max_views=60)
    assert keep is not None
    ghost_kept = float(keep[n_wall + n_ground:].mean())
    wall_kept = float(keep[:n_wall].mean())
    ground_kept = float(keep[n_wall:n_wall + n_ground].mean())
    assert ghost_kept < 0.15   # Geister praktisch weg
    assert wall_kept > 0.98    # Wand bleibt
    assert ground_kept > 0.98  # Boden bleibt
    assert stats["freiraum_carving"]["entfernt"] > 1_500
