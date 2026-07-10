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

    print("generating synthetic L-room scan (4 mm noise, 1% outliers) …")
    cloud = add_outliers(make_l_room_scan(density=900, noise=0.004), fraction=0.01)
    write_point_cloud(cloud, out_dir / "scan.ply")
    print(f"  {len(cloud):,} points → {out_dir / 'scan.ply'}")

    print("reconstructing with preset 'indoor' …")
    result = reconstruct(cloud, PipelineConfig.preset("indoor"))

    rep = result.report
    print(f"  planes:  {rep['planes']}  (angles: {rep['plane_angles']})")
    print(f"  corners: {rep['exact_corners']}")
    print(f"  mesh:    {rep['mesh']['vertices']} vertices / {rep['mesh']['triangles']} triangles")
    print(f"  runtime: {rep['runtime_seconds']} s")

    for ext in (".obj", ".ply", ".stl", ".glb"):
        path = write_mesh(result.mesh, out_dir / f"model{ext}")
        print(f"  wrote {path}")
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2))
    print(f"  wrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
