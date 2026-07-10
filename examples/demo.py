"""Demo: generate a synthetic noisy scanner point cloud of an L-shaped room,
reconstruct a clean-edged model and write every output format.

Run from the repository root:

    python examples/demo.py [output_dir]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from synthetic import add_outliers, make_l_room_scan  # noqa: E402

from scantobim import PipelineConfig, reconstruct  # noqa: E402
from scantobim.io.writers import write_mesh, write_point_cloud  # noqa: E402


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("generating synthetic L-room scan with window (4 mm noise, 1% outliers) …")
    cloud = add_outliers(
        make_l_room_scan(density=900, noise=0.004, window=True), fraction=0.01
    )
    _paint_cloud(cloud)  # give the scan realistic RGB, like a photogrammetry cloud
    write_point_cloud(cloud, out_dir / "scan.ply")
    print(f"  {len(cloud):,} points → {out_dir / 'scan.ply'}")

    print("reconstructing with preset 'indoor' …")
    result = reconstruct(cloud, PipelineConfig.preset("indoor"))

    rep = result.report
    openings = sum(s.get("openings", 0) for s in rep["surfaces"])
    print(f"  planes:   {rep['planes']}  (angles: {rep['plane_angles']})")
    print(f"  corners:  {rep['exact_corners']}")
    print(f"  openings: {openings}")
    print(f"  classes:  {rep['quantities']['surface_count_by_class']}")
    print(f"  mesh:     {rep['mesh']['vertices']} vertices / {rep['mesh']['triangles']} triangles")
    print(f"  runtime:  {rep['runtime_seconds']} s")

    # model.html is the interactive standalone viewer — open it in a browser.
    for ext in (".obj", ".ply", ".stl", ".glb", ".html"):
        path = write_mesh(result.mesh, out_dir / f"model{ext}")
        print(f"  wrote {path}")

    # Photo-realistic version: bake the cloud colors into a texture atlas.
    from scantobim.core.texture import bake_texture_from_cloud

    textured = bake_texture_from_cloud(result, cloud)
    th, tw = textured.texture.shape[:2]
    for ext in (".glb", ".html"):
        path = write_mesh(textured, out_dir / f"model_texturiert{ext}")
        print(f"  wrote {path} (Textur {tw}x{th})")

    from scantobim.io.dxf import write_floorplan_dxf

    plan = write_floorplan_dxf(result.mesh, out_dir / "grundriss.dxf")
    print(f"  wrote {plan}")
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2))
    print(f"  wrote {out_dir / 'report.json'}")


def _paint_cloud(cloud) -> None:
    """Synthetic 'photo colors': parquet floor, painted walls with a dado."""
    import numpy as np

    pts = cloud.points
    rng = np.random.default_rng(1)
    colors = np.zeros((len(pts), 3), dtype=np.uint8)
    h = 2.5
    is_floor = pts[:, 2] < 0.05
    is_ceiling = pts[:, 2] > h - 0.05
    is_wall = ~is_floor & ~is_ceiling

    stripe = ((pts[:, 0] / 0.12).astype(int) % 2).astype(np.float64)
    grain = rng.normal(0, 8, len(pts))
    colors[is_floor, 0] = np.clip(165 + 18 * stripe[is_floor] + grain[is_floor], 0, 255)
    colors[is_floor, 1] = np.clip(120 + 14 * stripe[is_floor] + grain[is_floor], 0, 255)
    colors[is_floor, 2] = np.clip(80 + 8 * stripe[is_floor] + grain[is_floor], 0, 255)
    colors[is_ceiling] = (235, 232, 226)
    wall_col = np.array([210, 205, 196], dtype=np.float64)
    dado = np.array([96, 130, 158], dtype=np.float64)
    wmask = is_wall & (pts[:, 2] >= 1.0)
    colors[wmask] = np.clip(wall_col + rng.normal(0, 5, (wmask.sum(), 3)), 0, 255).astype(np.uint8)
    dmask = is_wall & (pts[:, 2] < 1.0)
    colors[dmask] = np.clip(dado + rng.normal(0, 6, (dmask.sum(), 3)), 0, 255).astype(np.uint8)
    cloud.colors = colors


if __name__ == "__main__":
    main()
