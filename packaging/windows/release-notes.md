## ScanToBIM für Windows

**Download:** `scantobim-windows-x64.zip` entpacken — keine Installation, kein Python nötig.

> SmartScreen-Hinweis beim ersten Start (Datei ist nicht code-signiert): *Weitere Informationen → Trotzdem ausführen*.

### Neu in 2.4.0 — Stereo-Kamera & Farb-Fix (S20-Praxis)

- **Stereo-Fotos (links/rechts) werden jetzt beide genutzt**: Der S20 registriert seine Bilder als `left/…` und `right/…` in zwei Ordnern — der Projekt-Import wählt jetzt den gemeinsamen Überordner, und die Foto-Projektion findet jede Datei über einen Namensindex (voller Pfad, Ordner-Präfix oder Dateiname). Neue Diagnosezeile: „Fotos gefunden: X von Y registrierten Kameras“.
- **Weißes Modell behoben**: E57-Farben in 16-Bit (0–65535, wie beim S20) oder als 0–1-Float wurden bisher auf Weiß bzw. Schwarz abgeschnitten — jetzt werden sie korrekt auf 8-Bit skaliert. Die Punktfarben-Textur zeigt wieder echte Farben.
- **COLMAP-Text-Parser**: Posen-Dateien mit leeren 2D-Punktzeilen verloren jede zweite Kamera — behoben.
- **Fairere Abdeckungszahl**: zusätzlich zur Gesamt-Abdeckung wird die Abdeckung **im Modellbereich** (Bauwerks-Umfeld + 1 m) ausgewiesen — Drift-Ausläufer des Roh-Scans drücken die Zahl nicht mehr.

### Neu in 2.3.0

- **Auto-Tuning-Gewinner als Profil speichern**: nach einem Lauf mit Szene „Automatisch“ erscheint im Messbericht der Knopf **„★ Gewinner-Einstellungen als Profil speichern“** — die objektiv beste Parametrierung wird in die Oberfläche übernommen und dauerhaft als eigenes Profil abgelegt (z. B. „Auto (fein)“). Die Werte sind maßstabsfrei (× Punktabstand) und übertragen sich damit auf künftige Scans derselben Quelle.

### Neu in 2.2.0

- **Quellen-Profile**: auswählbare Voreinstellungen je Aufnahmegerät — SLAM-Handscanner (SHARE S20 & Co.), Stativ-Laserscanner, Drohne/Photogrammetrie, iPhone-LiDAR — geschichtet über jedes Szenen-Preset (`--source`, GUI-Auswahl „Quelle/Scanner“). Das SLAM-Profil verschmilzt z. B. Registrierungs-Doppelwände bis 3 cm automatisch.
- **Eigene Profile, mehrere**: alle Einstellungen (Quelle, Szene, Optionen, erweiterte Werte) als benannte Profile **dauerhaft speichern, laden, löschen** — z. B. „S20 außen“, „S20 innen fein“.
- **Erweiterte Einstellungen mit Erklärung bei Mouseover**: Voxelgröße, Ebenen-Toleranz, Mindest-Flächengröße, Max. Flächen, Winkel-Raster, Geister-Versatz, Öffnungs-Mindestgröße, QS-Toleranz, Punkte-Limit — jedes Feld erklärt sich beim Draufzeigen; leer = Automatik.

### Neu in 2.1.0 — kalibriert am echten SHARE-SLAM-S20-Scan

- **Gelände-Erkennung**: raue Horizontalflächen (Boden, Schotter, Rasen — Ebenen-RMS > 2,5 cm) werden als **Gelände** klassifiziert statt als Bauteil-„slab“ — sie erzeugen keine Phantom-Öffnungen, keine Geschosse und verfälschen keine Bauteilmengen mehr (`IfcGeographicElement` im BIM-Export).
- **Dach-Erkennung relativ zur Wandhöhe** statt zur Szenenhöhe: Bäume, Masten und Gelände strecken Außenscans in z — echte Dächer wurden dadurch übersehen. Jetzt gilt: geneigte Fläche oberhalb der Wände = Dach.
- **Soll-Ist-Analyse für Teilszenen**: getrennte Ausweisung von **Modelltreue** (RMS/P95 nur der modellnahen Punkte, Gelände ausgenommen) und **Modellabdeckung** (Anteil des Scans am Modell) — vorher dominierte nicht modellierte Umgebung (Vegetation, Nachbarobjekte) die Statistik mit Meter-Werten. Die Heatmap-Farbskala löst jetzt den Millimeterbereich am Bauwerk auf.
- **Auto-Tuning bewertet mit denselben robusten Kennzahlen** — Außen-Clutter kann kein Kandidaten-Ranking mehr verzerren.

### Neu in 2.0.0

- **Auto-Tuning** (Preset „Automatisch“): das Programm rekonstruiert mit mehreren Parametersätzen, bewertet jedes Ergebnis objektiv am Scan selbst (Soll-Ist-P95 + erklärte Punkte + Modell-Schlankheit) und behält das beste — transparente Rangliste im Bericht.
- **Feinstruktur-Nachlese**: ein zweiter, feinerer Erkennungspass sammelt kleine echte Flächen (Laibungen, Vorsprünge, Möbel) aus dem Residuum ein — Rausch-Splitter werden weiterhin verworfen.
- **Sehr große Punktwolken**: LAS/LAZ werden blockweise gelesen und adaptiv ausgedünnt (`--max-points`, GUI automatisch) — hunderte Millionen Punkte bei begrenztem Speicher.
- **Dach-Semantik**: Dachflächen mit Neigung, Ausrichtung, Fläche, First- und Traufhöhe im Bericht; `IfcRoof` im BIM-Export.
- **Fenster/Tür-Klassifikation**: Öffnungen werden mit Breite, Höhe und Brüstungshöhe als Fenster oder Tür erkannt und gezählt.
- **Epochen-Vergleich** (`scantobim compare`, GUI-Modus): zwei Scans desselben Objekts → ICP-Feinregistrierung + Verformungs-Heatmap (blau = Setzung, rot = Ausbauchung) mit Statistik — Monitoring für Brücken, Hallen, Stahlbau.
- **Druckfertiger Prüfbericht** (`--report-html`, GUI-Häkchen): alle Kennzahlen, Dach- und Öffnungslisten, Soll-Ist-Histogramm und die maßstabsgetreuen Ansichten in einem A4-Dokument — im Browser öffnen, als PDF drucken.

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
