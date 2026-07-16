## ScanToBIM für Windows

**Download:** `scantobim-windows-x64.zip` entpacken — keine Installation, kein Python nötig.

> SmartScreen-Hinweis beim ersten Start (Datei ist nicht code-signiert): *Weitere Informationen → Trotzdem ausführen*.

### Neu in 3.3.0 — xyzopk-Kameraposen mit Selbstkalibrierung (S20)

- **Dein Posen-Format wird jetzt gelesen**: Der S20 legt die Kameraposen nicht als COLMAP-Modell ab, sondern als **`xyzopk.txt`** im `undistort`-Ordner (je Foto: Position X/Y/Z + Drehwinkel Omega/Phi/Kappa). Der neue Leser versteht Kopfzeilen, Komma/Leerzeichen, Namen vorn oder hinten, Grad oder Radiant.
- **Selbstkalibrierung statt Raten**: Die Datei nennt weder Brennweite noch Winkel-Konvention. Beides ermittelt das Programm selbst: Es projiziert eine Stichprobe der **farbigen Punktwolke** in Probefotos und wählt die Kombination (8 Rotations-Konventionen × Brennweiten-Reihe), deren Projektion die Punktfarben am besten reproduziert — die Wolke wurde ja aus genau diesen Fotos eingefärbt. Ergebnis im Protokoll: „Konvention …, Brennweite … px, Übereinstimmung … %“.
- Damit laufen **Foto-Projektion aufs Strukturmodell** und **Foto-Farben aufs Komplett-Mesh** jetzt auch ohne COLMAP-Export — das Protokoll zeigt „Kameraposen: …xyzopk.txt (Selbstkalibrierung)“, die GUI-Dateiliste „Kameraposen (xyzopk)“.

### Neu in 3.2.0 — hochauflösende Fotos auf dem Netz, runde Bauteile geregelt

- **Gefunden: warum die Foto-Projektion nie lief.** Der S20 legt seine Kameraposen unter `output\colmap\sparse\0` — **eine Ordnerebene tiefer, als die Projektsuche bisher schaute**. Deshalb stand im Protokoll nie „Kameraposen:“ und alles blieb bei den verwaschenen Punktfarben. Die Suche geht jetzt bis Tiefe 6, und wenn Fotos ohne Kameraposen gefunden werden, sagt das Protokoll das ausdrücklich.
- **Foto-Farben auf dem Komplett-Mesh**: Jede Netz-Ecke wird in die am besten passende Original-Kamera projiziert (Blickwinkel × Nähe, zwei Pässe über bis zu 600 Fotos) und erhält die echte Pixel-Farbe des hochauflösenden Fotos — deutlich schärfer als die bisherigen gemittelten Punktfarben. Anteil und Kamerazahl stehen im Protokoll und Bericht (`foto_farben_anteil`).
- **Runde Bauteile werden Regelgeometrie**: Die Zylinder-Erkennung (Rohre, Stützen, Bögen) ist im Projektlauf jetzt **standardmäßig aktiv**, und erkannte Zylinder konturieren das Komplett-Mesh genauso wie die Ebenen: Netz-Ecken auf der Mantelfläche werden **radial exakt auf den Zylinder projiziert** — runde Bauteile sind wirklich rund, mit Durchmesser im Bericht. Ebenen → plan, Kanten → scharf, Rundungen → geregelt: das Modell folgt der realen, geometrisch geregelten Bauwelt.

### Neu in 3.1.0 — konturiert statt geschmolzen

- **Struktur-geführte Kontur-Schärfung**: Die in Stufe 2 gefundenen Ebenen ziehen jetzt die Ecken des Komplett-Meshes auf sich — Wände und Böden werden **exakt plan**, und wo zwei Ebenen zusammenstoßen, werden die Netz-Ecken **auf die Schnittkante projiziert**: messerscharfe Kanten statt verschmolzener Rundungen. Der Anteil konturierter Ecken steht im Bericht (`konturiert_anteil`) und im Protokoll.
- **Taubin- statt Laplace-Glättung**: Die Voxel-Treppen werden weiterhin geglättet, aber ohne das globale Schrumpfen/Anschmelzen dünner Bauteile (λ/μ-Schema erhält das Volumen). Zusätzlich stärkerer Rückzug auf die Messpunkte.
- **Ebenen-Schalter jetzt unübersehbar**: Das Panel oben rechts im Viewer ist größer, mit Überschrift „EBENEN EIN/AUS“ — und hat einen **neuen Schalter „Strukturmodell (Flächen & Kanten)“**, mit dem sich das Flächenmodell ausblenden lässt, um nur das Komplett-Mesh zu sehen (und umgekehrt).

### Neu in 3.0.0 — neue Architektur: erst das Komplett-Netz, dann die Struktur

- **Stufe 1 — Komplett-Mesh zuerst**: Der GESAMTE Scan wird als erstes zu einem farbigen Dreiecksnetz rekonstruiert (dünn besetztes Belegungsgitter → Hüllfläche → Glättung → Rückprojektion auf die Messpunkte → Punktfarben) — **maximale Modellfülle, bevor irgendeine Interpretation stattfindet**. Bis 4 Mio Dreiecke als `komplett.glb`, eine leichtere Variante im Viewer als schaltbare Ebene „Komplett-Mesh (Scan)“.
- **Stufe 2 — Ebenen & Linien als Option**: Die algorithmische Struktursuche (Ebenen, exakte Kanten/Schnittlinien, Öffnungen, Maße, Bauteilklassen, BIM/CAD-Exporte) läuft danach — **zuschaltbar** (GUI-Häkchen „Ebenen & Linien suchen“, CLI `--no-structure`). Im Viewer liegen Strukturmodell und Komplett-Mesh als getrennt schaltbare Ebenen übereinander.
- **Kein Totalausfall mehr**: Scheitert die Struktursuche (z. B. kaum ebene Flächen), bleibt das Komplett-Mesh als vollwertiges Ergebnis erhalten — der Lauf endet erfolgreich statt mit „No planar structure found“.
- Bericht: neuer Block `komplett_mesh` (Dreiecke, Bauteile, Scan-Abdeckung, Rasterweite); die GUI-Zusammenfassung zeigt ihn an erster Stelle.

### Neu in 2.9.0 — Auto-Rettung statt Abbruch, feineres Freiform-Mesh

- **„No planar structure found" ist Geschichte**: Bei SLAM-Scans ohne das SLAM-Quellenprofil (Quelle „Standard") füllen Registrierungs-Doppelwände das komplette Toleranzband — alle Rausch-Tore der Ebenenerkennung lehnten ab, der Lauf brach nach Minuten mit einem Fehler ab. Jetzt startet automatisch ein **zweiter Erkennungsversuch mit gelockerten Toleranzen** (breiteres Band, entschärfte Tore); der Bericht dokumentiert die Rettung und empfiehlt das SLAM-Profil.
- **Deutliche Warnung im Protokoll**, wenn ein SLAM-Projekt (Trajektorie gefunden) mit einer anderen Quelle als „SLAM-Handscanner" gestartet wird — genau das war die Ursache des gemeldeten Abbruchs.
- **Freiform-Mesh deutlich feiner**: Startraster jetzt 2,5× Punktabstand (vorher 4×) und das `freiform.glb` darf bis **2 Mio Dreiecke** nutzen — Geländer, Roste und Stahlprofile kommen sichtbar schärfer heraus. Der eingebettete Viewer erhält eine eigene, leichtere Variante (≤ 600 000 Dreiecke), damit die HTML-Datei handlich bleibt.

### Neu in 2.8.0 — Hybrid-Modell: Freiform-Rekonstruktion der Restgeometrie

- **Alles wird jetzt Modell**: Was das parametrische Flächen-/Zylindermodell nicht erklärt — gekrümmte Stahltore, Bögen, Geländer, Maschinen, Rohrleitungen — wird jetzt als **eigenes farbiges Dreiecksnetz rekonstruiert** (Freiform-Skin). Verfahren: dünn besetztes Voxel-Belegungsgitter → Hüllflächen-Extraktion → Laplace-Glättung → Rückprojektion jeder Netz-Ecke auf die echten Messpunkte → Einfärbung aus den Punktfarben. Rausch-Sprenkel werden über eine Zusammenhangs-Analyse verworfen; das Dreiecks-Budget wird über die Rasterweite eingehalten, nie durch Weglassen von Geometrie.
- **Im Viewer als Ebene schaltbar** („Freiform-Restgeometrie“, oben rechts), zusätzlich als eigene Datei `freiform.glb` zum Weiterverarbeiten in CAD/Blender. Kennzahlen (Dreiecke, Bauteile, abgedeckte Restpunkte, Rasterweite) stehen im Messbericht.
- Standardmäßig aktiv (GUI-Häkchen „Freiform-Restgeometrie vernetzen“, CLI `--no-freeform` zum Abschalten). BIM/CAD-Exporte (IFC, STEP) bleiben bewusst rein parametrisch.

### Neu in 2.7.0 — endlich Detail aus dichten Scans, nichts verschwindet mehr

- **Kleine echte Flächen werden jetzt gefunden**: Die Mindestgröße einer Fläche war als *Prozentsatz der Wolke* definiert — bei einem 10-Mio-Punkte-Scan hieß „1 %" plötzlich: jede Fläche braucht ~100 000 Punkte (≈ 6 m²!). Fensterlaibungen, Pfeiler, kleine Dachflächen konnten prinzipiell nie erkannt werden. Der Prozentsatz zählt jetzt gegen maximal 2 Mio Punkte — auf dichten Scans sinkt die Mindestfläche damit um Faktor 5–20.
- **Scan-Restpunkte im 3D-Viewer**: Alles, was das Flächenmodell nicht erklärt (Stahlbogen, Geländer, Maschinen, Vegetation), verschwand bisher einfach — das Modell wirkte leer gegenüber dem Scan. Jetzt werden bis zu 800 000 Restpunkte **in Originalfarbe als schaltbare Ebene** in modell.html eingebettet („Scan-Restpunkte“-Schalter oben rechts): strukturierte Flächen wo möglich, echte Messpunkte für den Rest — das Modell zeigt wieder das ganze Bauwerk.
- **Auto-Tuning bevorzugt jetzt Vollständigkeit**: Der Kandidat „detail“ erklärte im Praxis-Test 11 Prozentpunkte mehr der Szene, verlor aber wegen eines zu hohen Straf-Terms pro Fläche. Die Gewichte sind neu ausbalanciert — Abdeckung dominiert, Sparsamkeit entscheidet nur noch bei Gleichstand.
- **Plausibilitätsfilter für Öffnungen**: Löcher unter 25 cm Kantenlänge (Scan-Schatten, Rauschen) werden nicht mehr als „Fenster“ gezählt — die Fensterliste enthält nur noch echte Öffnungen.

### Neu in 2.6.1 — kritischer Fix: große Areale wurden beim Einlesen zerquetscht

- **Die eingebaute Ausdünnung hat große Scans ruiniert**: Beim Einlesen von Wolken über dem Punkte-Limit (GUI: 40 Mio) startete das Ausdünnungsraster bei *Szenendiagonale ÷ 1000* und konnte nur gröber werden — ein 45,7-Mio-Scan eines ~230-m-Areals kollabierte damit auf 156 848 Punkte mit 22 cm Raster (statt der erlaubten 40 Mio!). Ergebnis: nur grobe Riesenflächen, kaum Öffnungen, 3 % Abdeckung. Jetzt sucht die Ausdünnung das Raster **in beide Richtungen** (Bisektion auf das Punktebudget): das Ergebnis liegt garantiert nahe am Limit, typisch 70–100 % davon. Derselbe Scan liefert jetzt ~40 Mio Punkte in echter Zentimeter-Auflösung.
- Betroffen waren LAS/LAZ über dem Limit, E57 mit mehreren Scans und alle nachgelagerten Ausdünnungen (`--max-points`). Kleinere Wolken (unter dem Limit) waren nie betroffen.
- Hinweis: mit der vollen Punktdichte dauert die Rekonstruktion großer Areale entsprechend länger — insbesondere Szene „Automatisch“ (4 Kandidaten). Das Protokoll zeigt den Fortschritt.

### Neu in 2.6.0 — Punktdichte zählt: nie wieder die Vorschau-Wolke

- **Dichte UND Farbe entscheiden jetzt gemeinsam**: 2.5.0 bevorzugte die farbige Wolke — auch wenn sie eine 100×-ausgedünnte Vorschau war (z. B. 156 000 Punkte mit 22 cm Abstand statt des echten Multi-Millionen-Scans → nur grobe Riesenflächen, kaum Öffnungen). Jetzt wird zusätzlich die **Punktzahl aus dem Datei-Header** gelesen: die farbige Wolke gewinnt nur, wenn sie mindestens ¼ der Punktdichte der dichtesten Wolke hat — sonst gewinnt die dichte Wolke.
- **Farbübertragung von der Schwester-Wolke**: Hat die dichte Wolke keine Farben, aber eine (dünnere) farbige liegt daneben, werden deren Farben **per Nächster-Nachbar-Suche auf die dichte Geometrie übertragen** — volle Detailtreue *und* Punktfarben-Textur. Punkte ohne farbigen Nachbarn bleiben neutral grau; das Protokoll zeigt „Farben übertragen von: … → X % der Punkte eingefärbt“.
- **Transparenz im Protokoll**: „Punktwolke: … (245.1 MB, 12.4 Mio Punkte, mit Farben)“, dazu ggf. „→ dichteste Wolke gewählt — X ist zwar farbig, aber stark ausgedünnt“ und die Zeile „Farbquelle: …“. Auch der Ordner-Eintrag in der GUI-Dateiliste zeigt Punktzahl und Farbstatus.
- **Warnung bei ausgedünnter Wolke**: liegt der mittlere Punktabstand über 5 cm, weist das Protokoll ausdrücklich darauf hin, die hochauflösende Export-Datei des Scanners zu verwenden — eine grobe Wolke kann kein detailtreues Modell ergeben.

### Neu in 2.5.0 — richtige Punktwolke im Projektordner

- **Farbige Punktwolke wird bevorzugt**: Liegen im Projektordner mehrere Punktwolken (z. B. `…_colorized.e57` **und** eine größere unkolorierte), wählte das Programm bisher schlicht die größte — oft die ohne Farben, und die Textur blieb leer („Textur: keine“). Jetzt wird die Datei-**Kopfzeile** jeder Kandidatin geprüft (ohne die Punktdaten zu lesen, also auch bei GB-Dateien sofort) und die Wolke mit echten RGB-Farben bevorzugt. Das Protokoll zeigt die Wahl transparent: „Punktwolke: … (mit Farben)“ bzw. „→ farbige Wolke bevorzugt (X statt Y)“.
- **Klare Meldung statt Absturz bei gemischter Liste**: Punktwolken-Datei **und** Projektordner zusammen in einer Liste führten zu „Unsupported point cloud format: ''“ — jetzt kommt vor dem Start eine verständliche Meldung: entweder Dateien ODER genau einen Projektordner starten (überzählige Einträge mit ✕ entfernen). Projektordner in anderen Modi (Analyse, Brücke, …) werden ebenfalls sauber abgewiesen.
- Der Farb-Status der gewählten Wolke steht auch im GUI-Dateieintrag des Ordners („mit Farben“ / „ohne Farben“).

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
