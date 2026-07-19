# steuerhaus_mini_testdaten — S20-Test-Fixture für ScanToBIM

Kompakter, aus dem echten Scan abgeleiteter Testdatensatz (134 MB statt 8,5 GB,
**jede Einzeldatei < 25 MB** → auch per GitHub-Web-Upload einspielbar).
**Ordnerstruktur identisch zum Original-SHARE-S20-Projekt**, damit die automatische
Projekterkennung und alle Codepfade (LAS-Ranking, Farbtransfer, Posen, Kalibrierung,
Zeitzuordnung) exakt wie in Produktion laufen.

## Ableitung aus dem Original (`Desktop\steuerhaus_nur_aussen`)

| Inhalt | Original | Fixture | Methode |
|---|---|---|---|
| uncolorized.las | 19.995.710 Pkt | 666.524 Pkt (22,9 MB) | jeder 30. Punkt, Header 1.4 gepatcht, EVLRs entfernt |
| colorized.las | 18.110.690 Pkt | 603.690 Pkt (20,7 MB) | dito |
| Fotos left/right | je 374, ~4 MB | je 19 (jedes 20. Paar), **Originalauflösung 3504×4672**, JPEG q82 | Auflösung unverändert — Kamera-Auswahl nach Bildgröße muss funktionieren! |
| ImgPose.txt / xyzopk.txt | 747 Zeilen | 36 Zeilen | auf gewählte Bilder gefiltert, Format unverändert |
| leftImgTime/rightImgTime | 374 Zeilen | 19 Zeilen | nach Frame-Index gefiltert |
| calibration.yaml, trajectory.txt, project_info.json, frame_pose.txt, *.opt, preview | vollständig | **unverändert kopiert** | — |

Nicht enthalten (im Tool ohnehin ignoriert): ROS-Bags, Potree-Ordner, pointcloud.pcd, Logs.

## Erwartungswerte für automatische Tests (aus den Vollläufen bekannt)

- uncolorized-RGB ist **Grauwert** (R=G=B, Intensität) → darf nie als Farbquelle gewählt werden
- colorized ist die Farbquelle; Farbtransfer sollte ≳95 % der Punkte einfärben
- Kameras: fisheye_left/right 3504×4672, POLYFISHEYE, fx≈1480/1477; fx=548 = Navi-Kamera (keine Fotos!)
- Posen: Quaternion, Zeit-Zuordnung zur Trajektorie (Residuum ~5 cm, Rotation ~0,07°)
- Szene lokal, keine Georeferenz; Ausdehnung 152×111×25 m; Boden nahe z≈0, leicht verkippt (Normal 0.0037/-0.0030/-1)
- 39 Ebenen / 32 Fenster im Strukturmodell des Vollscans (Fixture: weniger, aber Größenordnung prüfbar)

## Grenzen des Fixtures

- 1/30-Punktdichte (~3,7 cm Punktabstand) → Detail-Mesh-Qualität nicht repräsentativ; Raster-/Textur-Metriken nur relativ bewerten
- 19 Bildpaare → MVS/Texturabdeckung deutlich geringer als in Produktion
- **Vollqualitäts-Abnahme weiterhin nur gegen `Desktop\steuerhaus_nur_aussen`** (macht die andere Claude-Sitzung mit Datenzugriff)

## Stellschrauben (Regenerieren)

Skript: `make_fixture.py` (liegt in diesem Ordner).
`LAS_STEP` (30), `IMG_STEP` (20), `IMG_QUALITY` (82) anpassen → Größe vs. Aussagekraft.
Achtung: Nach einem Neulauf die ImgTime-Filter prüfen (Index-Format `<nr> <zeit>`).
