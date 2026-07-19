"""Objekt- und Geländeerkennung: Klassifikation + DTM-Gelände-Mesh."""

import numpy as np

from scantobim.core.scenesplit import (
    FAHRZEUG,
    GELAENDE,
    REST,
    STABWERK,
    VEGETATION,
    classify_scene,
    terrain_mesh,
)


def _scene():
    rng = np.random.default_rng(3)
    parts = {}
    g = rng.uniform(0, 30, (60_000, 2))
    parts["ground"] = np.column_stack(
        [g, rng.normal(0.0, 0.02, len(g))]
    )
    # Building 8x6, walls + flat roof at z=5 (attached to the ground).
    zs = rng.uniform(0, 5, 30_000)
    side = rng.integers(0, 4, len(zs))
    t = rng.uniform(0, 1, len(zs))
    wx = np.where(side < 2, t * 8, np.where(side == 2, 0, 8))
    wy = np.where(side < 2, np.where(side == 0, 0, 6), t * 6)
    walls = np.column_stack([2 + wx, 20 + wy, zs])
    roof = np.column_stack([
        rng.uniform(2, 10, 12_000), rng.uniform(20, 26, 12_000),
        np.full(12_000, 5.0) + rng.normal(0, 0.01, 12_000),
    ])
    parts["building"] = np.vstack([walls, roof])
    # Car 4.2 x 1.8 x 1.5 on the ground.
    parts["car"] = np.column_stack([
        rng.uniform(16, 20.2, 6_000), rng.uniform(5, 6.8, 6_000),
        rng.uniform(0.25, 1.55, 6_000),
    ])
    # Mast: vertical thin line.
    parts["pole"] = np.column_stack([
        rng.normal(25, 0.04, 2_000), rng.normal(12, 0.04, 2_000),
        rng.uniform(0.0, 4.5, 2_000),
    ])
    # Tree: green blob.
    blob = rng.normal(0, 1.0, (8_000, 3)) * [1.2, 1.2, 1.0]
    parts["tree"] = blob + [8.0, 8.0, 2.8]

    names, arrays = zip(*parts.items())
    pts = np.vstack(arrays)
    colors = np.full((len(pts), 3), 120, dtype=np.uint8)
    ofs = np.cumsum([0] + [len(a) for a in arrays])
    seg = {n: slice(ofs[i], ofs[i + 1]) for i, n in enumerate(names)}
    colors[seg["tree"]] = (60, 160, 50)
    colors[seg["car"]] = (40, 60, 150)
    return pts, colors, seg


def test_classify_scene_labels_all_classes():
    pts, colors, seg = _scene()
    stats: dict = {}
    labels = classify_scene(pts, colors, stats_out=stats)
    assert labels is not None

    def frac(name, cls):
        lab = labels[seg[name]]
        return float((lab == cls).mean())

    assert frac("ground", GELAENDE) > 0.9
    assert frac("car", FAHRZEUG) > 0.8
    assert frac("tree", VEGETATION) > 0.8
    assert frac("pole", STABWERK) > 0.8
    # The building must NEVER be classified as a removable Stoerer.
    lab_b = labels[seg["building"]]
    bad = np.isin(lab_b, [FAHRZEUG, VEGETATION]).mean()
    assert bad < 0.05
    assert (lab_b == REST).mean() > 0.6
    assert stats["szene"]["cluster"]["fahrzeuge"] == 1
    assert stats["szene"]["cluster"]["vegetation"] >= 1


def test_terrain_mesh_is_closed_grid():
    pts, colors, seg = _scene()
    labels = classify_scene(pts, colors)
    mesh = terrain_mesh(pts, labels, colors, cell=0.5)
    assert mesh is not None
    assert len(mesh.faces) > 1000
    # Terrain must stay near z=0 (no roof heights leaking in).
    assert float(np.abs(mesh.vertices[:, 2]).max()) < 0.6
    # The grid mesh has no holes in the observed interior: every interior
    # vertex is shared by ~6 triangles -> face/vertex ratio near 2.
    assert len(mesh.faces) > 1.5 * len(np.unique(mesh.faces))
    assert mesh.vertex_colors is not None
