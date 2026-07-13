## ScanToBIM für Windows

**Download:** `scantobim-windows-x64.zip` entpacken — keine Installation, kein Python nötig.

> SmartScreen-Hinweis beim ersten Start (Datei ist nicht code-signiert): *Weitere Informationen → Trotzdem ausführen*.

### Neu in 1.9.0

- **Textur-Fix**: Wenn die Foto-Projektion ins Leere läuft (Kameraposen in anderem Koordinatensystem, Bilder nicht auffindbar), erkennt das Programm das jetzt selbst (Abdeckungsmessung + Koordinaten-Plausibilitätsprüfung) und fällt **automatisch auf die Punktfarben zurück** — nie wieder ein graues Modell. Die Texturquelle steht sichtbar im Bericht („Textur: foto-projektion (87 % Abdeckung)“ / „punktfarben“).
- **Soll-Ist-Abweichungsanalyse** (QS): Abstand jedes Scanpunkts zum Modell — RMS, P95, Maximum, Toleranzquote und mm-Histogramm im Bericht plus blau-weiß-rot eingefärbte Abweichungswolke (`--deviation`, in der GUI ein Häkchen).
- **Orthofoto-Ansichten**: maßstabsgetreue Fassadenansichten N/O/S/W + Draufsicht als PNG mit World-File für CAD/GIS (`--views`, GUI-Häkchen) — gerendert mit der Fototextur.
- **Bemaßter Grundriss**: `--floorplan` beschriftet jede Wand automatisch mit ihrer Länge (Layer BEMASSUNG).

### Neu in 1.8.0

- **SLAM-Scanner-Projekte** (SHARE SLAM S20 & Co.): `scantobim project <ordner>` — oder in der GUI einfach den Projektordner-Pfad einfügen. Automatisch erkannt und genutzt: Punktwolke, **undistorted Fotos + COLMAP-Posen** (Foto-Projektion in voller Schärfe, jetzt auch **binäre** COLMAP-Modelle), **trajectory.txt** (Normalen werden zum Scannerpfad orientiert — deutlich robustere Erkennung auf Realdaten). Die `.bag`-Rohaufnahme wird bewusst nicht angefasst — ihre verarbeiteten Ergebnisse sind genau diese Dateien.
- `reconstruct --trajectory pfad.txt` für einzelne Wolken mit Scannerpfad.

### Neu in 1.7.1

- **E57 direkt in der Windows-EXE**: `.e57`-Scans (auch farbig, mehrere Standpunkte) werden jetzt ohne Zusatzinstallation gelesen — der Fehler „Reading .e57 requires pye57“ ist behoben. Der Build wird automatisch mit einem farbigen E57-Testscan verifiziert.

### Neu in 1.7.0

- **Professionelle grafische Oberfläche**: Doppelklick auf `scantobim.exe` öffnet die GUI im Browser (lokal, ohne Internet) — Drag & Drop, Modus-Auswahl, Live-Protokoll, integrierter 3D-Viewer, Messbericht, alle Exporte als Download. Auf die EXE gezogene Dateien sind beim Start bereits geladen. Konsolen-Fans: `scantobim.exe wizard`.
- **Zahnräder mit korrekter Zahngeometrie**: `analyze --mesh` schreibt Stirnräder als echte Evolventenverzahnung (DIN 867, Eingriffswinkel 20°), inkl. Bohrung aus der koaxialen Welle.
- **Stahltragwerke als korrektes 3D-Modell**: erkannte Profile werden als extrudierte Katalog-Querschnitte (I/U/L/Hohlprofil mit Steg-/Flanschdicken nach DIN 1025/1026) in der erkannten Einbaulage ausgegeben.
- **Brücken als korrektes 3D-Modell** (`bridge --mesh`): Überbau, Pfeiler, Widerlager, Bogen-Tonnengewölbe, Pylone und Seile als Volumenkörper mit benannten Bauteilgruppen.
- **Detailgetreue Rekonstruktion** (`--preset detail`): behält kleine Strukturen und rekonstruiert Stützen/Rohre als echte Zylinder (`--cylinders` für jedes Preset).

### Funktionen

- **Rekonstruktion**: Punktwolken (`.las/.laz/.e57/.ply/.pcd/.pts/.xyz`) → Modelle mit schnurgeraden Kanten, exakten 90°-Winkeln, Fenster-/Türöffnungen, Bauteil-Klassifikation, Mengenermittlung, Mehrgeschoss-Erkennung
- **Fotorealistische Texturierung** aus Punktwolkenfarben oder per Foto-Projektion (COLMAP-Posen, verdeckungsgeprüft)
- **Export**: STEP (CAD-B-Rep, auch der Weg nach HiCAD), IFC4, GLB/OBJ/STL, DXF-Grundriss, interaktiver HTML-Viewer
- **Industrie-Analyse**: Wellen, Zahnräder (Zähnezahl + DIN-Modul), Getriebestufen, Kegel, Stahlprofile (IPE/HEA/HEB/UPN/SHS/RHS/L), Anschlussdetails
- **Blechkonstruktionen**: Blechdicken, Schweißnahtlängen, Gewichte, Brennschnitt-DXF 1:1
- **Brückenbauwerke**: Typ-Klassifikation, Feldweiten, Pfeiler mit Lagerpunkten, Bogen, Pylone, Seile
- **Multi-Scan**: ICP-Registrierung und Verschmelzung mehrerer Standpunkte

Schnellstart und alle Befehle: `LIESMICH.txt` im Archiv. Die Exe wurde auf dem Windows-Runner automatisch verifiziert (Rekonstruktion + STEP-Export eines Testscans).
