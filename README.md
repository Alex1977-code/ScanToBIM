# ScanToBIM – Punktwolken & Fotos → 3D-Modelle mit sauberen Kanten

**ScanToBIM** rekonstruiert aus LiDAR-Punktwolken beliebiger Herkunft
(terrestrische Scanner, Drohnen, iPhone/iPad-LiDAR, Mobile Mapping) und aus
Fotos (Photogrammetrie) professionelle 3D-Modelle mit **sauberen, geraden
Kanten und exakten Winkeln** – statt der üblichen „verwaschenen“
Dreiecksnetze klassischer Flächenrekonstruktion.

![Demo](docs/images/demo.png)

> Aus 75.000 verrauschten Scanpunkten (4 mm Sensor­rauschen, 1 % Ausreißer)
> entsteht die **minimale exakte Repräsentation** des Raums: 12 Vertices,
> 20 Dreiecke, alle Kanten schnurgerade, alle Winkel exakt 90°.

> English summary at the bottom of this document.

## Warum saubere Kanten?

Klassische Rekonstruktion (Poisson, Ball-Pivoting, Marching Cubes) mittelt
das Sensorrauschen in die Oberfläche hinein – Kanten werden rund, Wände
wellig, und das Ergebnis ist für CAD/BIM praktisch unbrauchbar. ScanToBIM
geht den Weg der aktuellen Forschung zu strukturierter Rekonstruktion
(Efficient RANSAC, GlobFit, PolyFit, Kinetic Space Partitioning, Point2CAD):

1. **Vorverarbeitung** – Voxel-Ausdünnung (behält echte Messpunkte statt
   Mittelwerte), statistische Ausreißerentfernung, PCA-Normalenschätzung.
   Alle Längen-Parameter werden automatisch aus dem gemessenen Punktabstand
   abgeleitet – dieselben Presets funktionieren für Räume in Metern und
   Bauteile in Millimetern.
2. **Ebenen-Detektion** – normalengestützter RANSAC (1-Punkt-Hypothesen mit
   adaptivem Abbruchkriterium) mit Total-Least-Squares-Refit. Drei
   statistische Gates (RMS, Kern-Konzentration, Surface Variation) weisen
   diffuse Rausch-„Ebenen“ ab; ein Zusammenhangs-Filter trennt koplanare,
   aber räumlich getrennte Flächen.
3. **Regularisierung** – Normalen-Clustering und Manhattan-Snapping: fast
   parallele Ebenen werden exakt parallel, fast rechtwinklige exakt
   rechtwinklig (Toleranzen konfigurierbar); koplanare Duplikate werden zu
   einer logischen Fläche verschmolzen.
4. **Exakte Kantengeometrie** – Kanten werden nicht aus verrauschten Punkten
   „nachgezeichnet“, sondern **mathematisch abgeleitet**: benachbarte
   Ebenen schneiden sich in einer exakten Geraden, drei Ebenen in einem
   exakten Eckpunkt.
5. **Konturpolygone** – pro Ebene: Alpha-Shape-Rand → Douglas-Peucker →
   Begradigung an dominanten Richtungen (O-Snap-Stil) → Snapping der
   Polygone auf die exakten Schnittgeraden und Eckpunkte → Entfernung
   kollinearer Restpunkte.
6. **Vermaschung & Export** – Ear-Clipping-Triangulierung, Vertex-Welding
   (gemeinsame Kanten werden echte Falze statt Risse – geschlossene Räume
   werden wasserdicht), Export als **OBJ** (mit Flächengruppen), **PLY**,
   **STL**, **glTF/GLB** (mit Flächenfarben) plus maschinenlesbarem
   **Qualitätsbericht** (JSON).

## Unterstützte Eingaben

| Quelle | Format | Hinweis |
| --- | --- | --- |
| Terrestrische/mobile Laserscanner | `.las`, `.laz` | via laspy, `.laz` mit `pip install scantobim[laz]` |
| Scanner-Austauschformat | `.e57` | optional: `pip install scantobim[e57]` |
| Photogrammetrie (COLMAP, Metashape, RealityCapture, Meshroom) | `.ply`, `.las` | dichte Punktwolke exportieren |
| **Fotos direkt** | `.jpg`, `.png`, … | `scantobim photos` ruft ein lokal installiertes [COLMAP](https://colmap.github.io) auf (SfM + MVS, mit Sparse-Fallback ohne CUDA) |
| 3D Gaussian Splatting / NeRF | `.ply` | Positionen werden gelesen, SH-Koeffizienten ignoriert |
| iPhone/iPad LiDAR-Apps | `.ply`, `.e57`, `.xyz` | z. B. Scaniverse, Polycam, 3d Scanner App |
| PCL / Leica Cyclone / Textformate | `.pcd`, `.pts`, `.xyz`, `.txt`, `.csv` | inkl. Intensität und RGB |

## Installation

Voraussetzung: Python ≥ 3.10.

```bash
pip install .            # aus diesem Repository
pip install .[laz]       # + LAZ-Unterstützung (empfohlen)
pip install .[laz,e57]   # + E57
```

Für den Foto-Workflow zusätzlich COLMAP installieren
(`apt install colmap` bzw. [Anleitung](https://colmap.github.io/install.html)).

## Schnellstart

```bash
# Punktwolke inspizieren
scantobim info scan.laz

# Rekonstruktion: Gebäude von außen
scantobim reconstruct scan.laz -o model.glb --preset building --report report.json

# Innenraum-Scan (Normalen zeigen nach innen)
scantobim reconstruct raum.e57 -o raum.obj --preset indoor

# Fotos → Punktwolke → Modell
scantobim photos ./fotos -o wolke.ply
scantobim reconstruct wolke.ply -o model.glb

# Nicht erklärte Punkte (Möbel, Vegetation, …) separat sichern
scantobim reconstruct scan.las -o model.obj --residual-out rest.ply
```

Als Python-Bibliothek:

```python
from scantobim import PipelineConfig, reconstruct
from scantobim.io import read_point_cloud, write_mesh

cloud = read_point_cloud("scan.laz")
result = reconstruct(cloud, PipelineConfig.preset("indoor"))
write_mesh(result.mesh, "model.glb")
print(result.report["plane_angles"])   # z. B. 21x exakt orthogonal, 7x exakt parallel
```

## Presets & wichtige Parameter

| Preset | Einsatz | Besonderheit |
| --- | --- | --- |
| `building` | Gebäude/Objekte von außen | Normalen zeigen nach außen |
| `indoor` | Innenraum-Scans | Normalen zeigen nach innen, tolerantere Normalen-Gates |
| `object` | Bauteile, Möbel, CAD-artige Objekte | höhere Mindest-Inlier-Quote |
| `fast` | Schneller Vorschau-Durchlauf | weniger RANSAC-Iterationen |

Alle Parameter sind über `PipelineConfig` bzw. CLI-Flags zugänglich, u. a.
`--voxel` (Ausdünnung), `--dist` (RANSAC-Toleranz), `--max-planes`,
`--no-regularize`, `--no-straighten`, `--seed`. Nicht gesetzte Längen
werden automatisch aus dem Punktabstand abgeleitet.

## Qualitätsbericht

`--report report.json` schreibt u. a.: Punktzahlen je Verarbeitungsschritt,
Ebenen mit RMS/Fläche/Normale, Winkelstatistik (wie viele Ebenenpaare exakt
parallel/orthogonal sind), Anzahl exakter Ecken, Mesh-Statistik inkl.
Wasserdichtigkeit (`boundary_edges == 0`) und Laufzeit.

## Grenzen & Ausblick

- Die Pipeline ist auf **planare Strukturen** optimiert (Architektur,
  Innenräume, CAD-artige Objekte). Organische Formen landen im
  Residuum (`--residual-out`) – dafür optional Open3D-Poisson nutzen
  (`pip install scantobim[viz]`).
- Öffnungen (Fenster/Türen als Löcher in Flächen) werden derzeit als
  Außenkontur rekonstruiert; Loch-Topologie ist der nächste Schritt.
- Geplant: IFC-Export (Wände/Decken als BIM-Bauteile), Zylinder-Detektion
  für Rohrleitungen.

## Entwicklung

```bash
pip install -e .[dev]
pytest          # 56 Tests: IO-Roundtrips, Algorithmen, End-to-End-Qualitätsgates
python examples/demo.py   # erzeugt Beispiel-Scan + Modell in demo_output/
```

Die End-to-End-Tests prüfen harte Qualitätskriterien: exakte 90°-Winkel,
Eckpunkt-Genauigkeit < 3 cm bei 4 mm Rauschen, Wasserdichtigkeit
geschlossener Räume, Erhalt offener Ränder, Determinismus bei festem Seed
und Abweisung strukturlosen Rauschens.

## Stand der Technik – Referenzen

- Schnabel, Wahl, Klein: *Efficient RANSAC for Point-Cloud Shape Detection*, CGF 2007
- Li et al.: *GlobFit: Consistently Fitting Primitives by Discovering Global Relations*, SIGGRAPH 2011
- Arikan et al.: *O-Snap: Optimization-Based Snapping for Modeling Architecture*, TOG 2013
- Nan, Wonka: *PolyFit: Polygonal Surface Reconstruction from Point Clouds*, ICCV 2017
- Bauchet, Lafarge: *Kinetic Shape Reconstruction*, TOG 2020
- Schönberger, Frahm: *Structure-from-Motion Revisited* (COLMAP), CVPR 2016
- Kerbl et al.: *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023 (als Foto-Capture-Frontend)
- Liu et al.: *Point2CAD: Reverse Engineering CAD Models from 3D Point Clouds*, CVPR 2024
- Pauly, Gross, Kobbelt: *Efficient Simplification of Point-Sampled Surfaces* (Surface Variation), VIS 2002

---

## English summary

**ScanToBIM** turns LiDAR point clouds of any origin (`.las/.laz`, `.e57`,
`.ply`, `.pcd`, `.pts/.xyz/.csv` — terrestrial scanners, drones,
iPhone/iPad LiDAR, photogrammetry exports, Gaussian-Splatting PLYs) and photo
sets (via a bundled COLMAP wrapper with sparse fallback) into **clean-edged,
CAD/BIM-ready 3D models**. Instead of smoothing noise into wavy meshes, it
detects planar structure with normal-aware RANSAC (with statistical noise
gates and connected-component filtering), regularizes plane orientations to
exact parallel/orthogonal relations (Manhattan snapping), derives edges
*mathematically* as plane–plane intersection lines and 3-plane corner
points, extracts per-plane boundary polygons (alpha shape → simplification →
dominant-direction straightening) snapped onto those exact edges, and welds
everything into a watertight-where-possible mesh. Exports: OBJ (surface
groups), PLY, STL, glTF/GLB (per-surface colors) plus a JSON quality report
(exact-angle statistics, corner count, watertightness, residual points).
A synthetic 75k-point room scan with 4 mm noise reconstructs to its minimal
exact representation — 12 vertices, 20 triangles, all angles exactly 90°.
Pure Python (numpy/scipy/laspy), 56 tests, MIT license. CLI:
`scantobim reconstruct scan.laz -o model.glb --preset indoor`.

## Lizenz

MIT – siehe [LICENSE](LICENSE).
