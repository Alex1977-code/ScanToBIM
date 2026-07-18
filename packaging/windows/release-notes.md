## ScanToBIM für Windows

**Download:** `scantobim-windows-x64.zip` entpacken — keine Installation, kein Python nötig.

> SmartScreen-Hinweis beim ersten Start (Datei ist nicht code-signiert): *Weitere Informationen → Trotzdem ausführen*.

### Neu in 3.20.0 — Posen-Registrierung mit allen Hypothesen: der Bericht zeigte, warum 3.19 nicht griff

Der 3.19-Lauf hat geliefert, was er sollte — die Diagnose: `posen_ausrichtung: angewendet=false, residuum 130,8 m, zuordnung "zeit"`. Die Registrierung LIEF, aber die Zeit-Zuordnung erzeugte falsche Punktpaare (Foto-Zeiten außerhalb des Trajektorien-Fensters werden von der Interpolation ans Ende geklemmt — viele-zu-eins-Paare vergiften den Fit), und der Bogenlängen-Fallback kam nie zum Zug, weil die Zeit-Zuordnung formal „funktionierte".

- **Alle Zuordnungs-Hypothesen werden jetzt gerechnet, die beste gewinnt**: Zeit (nur Fotos IM Trajektorien-Fenster, keine geklemmten Paare mehr), Zeit mit konstantem Uhren-Versatz, Bogenlänge vorwärts und rückwärts — jede als starrer Fit (Kabsch) UND als Ähnlichkeits-Fit (Umeyama, fängt Einheiten-Fehler wie cm↔m). Angenommen wird nur ein Residuum unter der pfadgrößen-relativen Schwelle.
- **Der Bericht zeigt die Residuen ALLER Hypothesen** (`residuen_aller_hypothesen_m`) — auch im Fehlerfall. Damit ist beim nächsten Lauf ohne Rätselraten ablesbar, welche Zuordnung passt und welche nicht: das Ende der schleppenden Fehlersuche bei den Posen.
- Getestet gegen die drei realistischen Störfälle: fremde Foto-Uhr (Bogenlänge gewinnt), Zentimeter-Koordinaten (Skalierung 0,01 erkannt), Zufalls-Posen (alle Hypothesen abgelehnt, vollständiger Residuen-Bericht).

### Neu in 3.19.0 — Grundursache gefunden: die ImgPose-Posen lagen 150 m neben der Wolke

Die Kamera-Selbstprüfung aus 3.18 hat geliefert: `kamera_abstand_p25_m: 151 m` — die ImgPose-Kamerapositionen liegen in einem EIGENEN Koordinatensystem, ~150 m neben der Punktwolke. Deshalb sampelten die Wände Himmel und Wasser, die Texel-Ableitung lief gegen ihren Sicherheitsdeckel, nur 103 von 746 Kameras bestanden die Prüfung und der Kanten-Fotoabgleich fand (korrekt!) keine verwertbaren Gradienten. Der weiche Posen-Score von 92 % hatte das jahrelang kaschiert.

- **ImgPose wird jetzt auf die Trajektorie registriert**: Fotos und Trajektorie beschreiben denselben Lauf — die Fotoposen werden per Zeitstempel (Fallback: gleicher Weganteil/Bogenlänge) den Trajektorienpunkten zugeordnet, eine starre Transformation (Kabsch) rechnet den Kamerapfad ins Wolken-System, die Rotationen drehen mit. Ein Residuum-Tor (≤ 1 m) verhindert falsche Registrierungen; das Protokoll meldet „ImgPose lag im eigenen Koordinatensystem — per Trajektorie ins Wolken-System geholt (Versatz X m, Restfehler Y cm)", der Bericht führt `posen_ausrichtung`.
- **Damit fällt die gesamte Foto-Kette in sich zusammen … in die richtige Richtung**: korrekte Kamerazentren → echte GSD (mm-Texel statt 2,5-cm-Deckel), korrektes Kamera-Ranking (Ziegel statt Himmel), Kamera-Selbstprüfung sollte >600 von 746 validieren, und der Kanten-Fotoabgleich findet endlich Gradienten zum Nachjustieren.
- Verifiziert am synthetischen Gegenstück des Fehlers: Posen mit 170-m-Versatz und 30°-Drehung werden auf < 5 cm genau ins Wolken-System zurückgeholt, Farb-Score > 0,8, alle Kamerazentren auf dem wahren Pfad.

### Neu in 3.18.0 — Kamera-Selbstprüfung findet den Foto-Dieb, Kanten-Fotoabgleich ist drin

Diagnose zum 3.17-Lauf: texel_cm stand exakt am 2,5-cm-Deckel (Kamera-Abstände wirkten um ein Vielfaches zu groß) und Wände zeigten Himmelsfarben — obwohl der Posen-Score 92 % sagte. Der Hauptverdächtige: **Stereo-Namenskollision** — `left/0123.jpg` und `right/0123.jpg` teilen sich den Dateinamen, der Bild-Index reichte etlichen Kameras stillschweigend die falsche Seite.

- **Kamera-Selbstprüfung (neu, immer aktiv)**: JEDE Kamera wird einzeln gegen die farbige Punktwolke geprüft — bei mehrdeutigen Dateinamen werden ALLE Kandidaten-Dateien getestet und die farblich passende gewählt; Kameras, deren Foto nirgends zur Wolke passt, fliegen ganz raus. Protokoll: „Kamera-Selbstprüfung: X von Y Kameras farb-validiert (Median-Score …, N mehrdeutige Dateinamen aufgelöst)"; Bericht: `kamera_pruefung`. Alle Textur-Stufen (Strukturmodell, Komplett-Mesh, Detail-Mesh, Viewer) nutzen ab jetzt die validierte Kamera→Datei-Zuordnung.
- **GSD-Diagnose im Bericht**: Der Atlas dokumentiert jetzt `kamera_abstand_p25_m`, `kamera_abstand_median_m` und `gsd_mm` — steht der Texel wieder am Anschlag, zeigt der Bericht sofort, welcher Abstand ihn dorthin gezwungen hat. Die Texel-Ableitung nutzt das robuste 25-%-Perzentil der Abstände zur nächsten Kamera.
- **Kanten-Fotoabgleich (neu)**: Die Fotos lösen 3-5 mm pro Pixel auf — deutlich feiner als der LiDAR. Kandidatenkanten (Schnittgeraden benachbarter Strukturebenen) werden in die am besten sehenden validierten Fotos projiziert, dort per **Subpixel-Gradientensuche** quer zur Kante lokalisiert (robuster Median über Abtastpunkte und mehrere Kameras, Kappung 2,5 cm), und die Detail-Mesh-Kanten werden VOR dem Textur-Backen auf die fotografische Kante gezogen. Protokoll: „Kanten-Fotoabgleich: X von Y Kanten nachjustiert (Median … mm)". Die vermessenen Ebenen (Maße, Flächen, Bericht) bleiben unangetastet.

### Neu in 3.17.0 — Atlas-Packer neu gebaut: mm-Texel aus der Foto-GSD, mehrere Atlas-Seiten, plane Wände

Umsetzung der Diagnose zum 3.16-Lauf (Atlas 12288×256, Texel weiter 6,2 cm, Foto-Anteil 49 %):

- **Texelgröße kommt jetzt AUS DEN FOTOS**: Ziel-Texel = Foto-GSD der jeweils nächsten Kamera (Abstand ÷ Brennweite), geklemmt auf 3–25 mm. Vorher diktierten Flächensummen und Packdruck die Auflösung — beobachtet 6 cm Texel auf 4-mm-Fotos. Die Packung entscheidet nur noch über die Platzierung, nie über die Auflösung.
- **Mehrere Atlas-Seiten pro Modell**: Reicht eine 8192²-Seite nicht, verteilt der neue Packer die Charts auf bis zu **4 quadratische Seiten** (glTF: ein Material + eine Textur je Seite; `_detail.glb`, HTML-Viewer und die „Speichern als"-Exporte tragen alle Seiten). Jede Seite wird auf ihren Inhalt beschnitten — keine 98 % leeren Riesen-Atlanten mehr (der 12288×256-Streifen von 3.16 war ein Packer-Defekt).
- **Spiegelglatte Wände und Dächer**: Vor dem Detail-Meshing werden Punkte, die eindeutig zu EINER Strukturebene gehören, exakt auf diese Ebene projiziert (die Ebenen sind auf Bruchteile von Millimetern gefittet). Punkte an Kanten/Ecken (zwei Ebenen) bleiben unangetastet — nichts rundet ab. Protokoll: „Ebenen-Glättung: X Punkte exakt auf ihre Strukturebene projiziert".
- **Störer abgetrennt**: Kleinst-Komponenten (Geländer-Fragmente, Masten-Reste, Sprenkel) fliegen aus dem Detail-Mesh — Protokoll nennt Anzahl und verbleibende Komponenten; im Komplett-Mesh bleibt alles erhalten. Das beseitigt die „Fransen" strukturell.
- **Viewer: reduzierte Geometrie, volle Textur-Schärfe**: Die eingebettete Detail-Ansicht rechnet mit demselben GSD-Texel wie die große `_detail.glb` — weniger Dreiecke, aber gleiche Foto-Auflösung (der 3.16-Doppelverlust „913k Dreiecke + Mini-Atlas" ist behoben).
- Oberfläche: Standard-Hinweise aus den Gruppen-Beschriftungen in Tooltips verschoben („Berechnung", „Zusatzauswertungen").
- Nächster Schritt (angekündigt): Kanten-Fotoabgleich — Strukturkanten in die Fotos projizieren, an Subpixel-Gradienten ausrichten und die Mesh-Kanten darauf ziehen. Die Basis (scharfe, korrekt platzierte Texturen) liefert dieses Release.

### Neu in 3.16.0 — Textur-Auflösung ×10, Gebäude freigestellt, Detail-Mesh als Standard-Ansicht

Umsetzung der verifizierten Fixliste zum 3.15-Lauf (6,2 cm/Texel, 3,6 statt 2 cm Raster, 18 % Foto-Anteil, Viewer zeigte das reduzierte Komplett-Mesh):

- **Ursache der groben Textur gefunden — Chart-Explosion**: Das bucklige Voxel-Netz zerfiel beim Atlas-Entfalten in **461.337 Mini-Charts** (12 Dreiecke pro Chart!); deren Rand-Polster fraßen das Atlas-Budget, das Raster wurde auf 6,2 cm/Texel gezwungen und die winzigen Charts sampelten kaum Foto-Pixel (18 % Foto-Anteil). Jetzt werden die Flächennormalen vor der Chart-Bildung **geglättet** und Mini-Charts in ihre großen Nachbarn **verschmolzen** — im Regressionstest sinkt dieselbe Szene von 165.572 auf **153 Charts**. Ergebnis: das Budget geht in Auflösung statt Polster, der Foto-Anteil steigt entsprechend.
- **Gebäude wirklich freigestellt**: Die „Gebäuderegion" behielt 19 von 20 Mio Punkten — die Hüllbox der Wandflächen spannte fast die ganze Szene auf. Jetzt zählen nur Punkte im **Abstand ≤ 0,8 m zu den Gebäudeflächen selbst** (Wände, Dächer, Decken — abgetastet und per Nachbarschaftssuche geprüft). Gelände, Bewuchs, Autos und Straßenmöbel fliegen raus — das beseitigt den Großteil der Fransen und macht das 2-cm-Ziel erreichbar.
- **2-cm-Raster wird gehalten statt still vergröbert**: Flächen-Budget des Detail-Mesh auf 12 Mio erhöht; wird das Raster dennoch vergröbert, steht jetzt eine **deutliche ACHTUNG-Zeile** im Protokoll („Detail-Raster gehalten: 2,0 cm" bzw. „auf X cm vergröbert"). Der Detail-Atlas darf zusätzlich auf **12288 px Kantenlänge** wachsen.
- **Der Viewer zeigt jetzt standardmäßig das fotorealistische Detail-Mesh**: neue Ebene „Detail-Mesh Gebäude (fotorealistisch)" — beim Öffnen aktiv; Strukturmodell und Komplett-Mesh sind zuschaltbare Overlays mit ehrlichen Namen („Komplett-Mesh (ganze Szene, reduziert)"). Beim Drehen/Zoomen schaltet der Viewer auf eine ⅓-LOD und beim Loslassen sofort zurück auf volle Auflösung. Die Datei `_detail.glb` trägt weiterhin die volle Dreiecks- und Textur-Auflösung.
- Hinweis zur Einordnung: >95 % Foto-Abdeckung am Gebäude und normalen-gewichtetes Überblenden mehrerer Ansichten stehen als nächster Schritt an — die Projektion selbst rechnet bereits mit dem vollen POLYFISHEYE-Modell (kein Pinhole).

### Neu in 3.15.0 — aufgeräumte Oberfläche: klare Standards, „Speichern als …" statt Format-Vorauswahl

- **Exportformate kommen jetzt NACH der Berechnung**: Das Modell wird einmal gerechnet — danach erscheint unter „Ergebnisdateien" die Leiste **„Speichern als …"**: STEP (CAD), IFC (BIM), DXF-Grundriss, GLB, OBJ, STL, PLY werden **per Klick aus dem fertigen Ergebnis erzeugt** und sofort heruntergeladen. Keine Format-Vorauswahl mehr, kein neuer Rechenlauf.
- **Einstellungen in drei klaren Gruppen**: **Berechnung** (Komplett-/Detail-Mesh, Strukturmodell, Textur — Standard: alle an, einfach so lassen), **Zusatzauswertungen** (eingeklappt, Standard: alle aus) und **Erweiterte Einstellungen** (eingeklappt, nur Feinjustierung). Die Standards stehen sichtbar dabei.
- **„Was gilt wann?" steht jetzt in der Oberfläche**: Szene setzt die Grundwerte → das Quelle-Profil (z. B. SLAM) passt sie an den Scanner an → **ausgefüllte** erweiterte Felder überschreiben beides. Leere erweiterte Felder = Automatik. Genau in dieser Reihenfolge rechnet auch das Programm.
- **SLAM wird automatisch erkannt**: Beim Hinzufügen eines Projektordners mit Trajektorie stellt die Oberfläche die Quelle selbst auf „SLAM-Handscanner" — die bisher leicht zu übersehende Handauswahl entfällt.
- **„↺ Alles auf Standard zurücksetzen"**: ein Klick, und alle Haken, Auswahlfelder und erweiterten Werte stehen wieder auf den empfohlenen Standardwerten.

### Neu in 3.14.0 — Foto-Textur garantiert, Detail-Mesh wirklich 2 cm, Strukturanalyse mehrkernig

Analyse des Steuerhaus-Laufs von 3.13.0 (Bericht: `"texture": "keine"`, Detail-Raster 4,9 statt 2 cm): Drei Ursachen gefunden und behoben.

- **Der Foto-Atlas hängt nicht mehr am Strukturmodell**: Bisher galt — scheiterte die Foto-Texturierung des STRUKTURMODELLS (bei 540 m² Gelände in den Flächen fast zwangsläufig unter der 20-%-Schwelle), wurden **auch** Foto-Farben und Foto-Atlas des Komplett-Meshes übersprungen. Genau das machte das Modell „verwaschen": beste Kameraposen (Score 92 %), aber keine einzige Foto-Textur. Jetzt hat jede Stufe ihr eigenes Qualitäts-Tor — der fotorealistische Atlas läuft, sobald Kameraposen existieren.
- **Detail-Mesh: Gebäude-Region ohne Gelände**: Die Detail-Box wurde bisher über ALLE erkannten Flächen gespannt — inklusive der 59×41-m-Geländeflächen. Damit umfasste „Detail" die ganze Szene, und das Flächen-Budget zwang das Raster von 2 auf 4,9 cm hoch. Jetzt zählen nur Gebäudeflächen (Wände, Dächer, Decken) — das Detail-Mesh erreicht die bestellten **1–2 cm am Bauwerk**.
- **Detail-Mesh bekommt den Foto-Atlas**: `<name>_detail.glb` ist jetzt selbst fotorealistisch texturiert (gleiches Verfahren wie das Komplett-Mesh) — das ist das Modell für „realitätsnah": feinstes Raster + echte Fotos.
- **Strukturmodell-Textur auf neuem Fundament**: Die Foto-Projektion aufs Strukturmodell nutzt jetzt dieselbe Atlas-Maschinerie (Z-Buffer-Sichtbarkeit gegen eine dichte Wolken-Stichprobe statt Strahltest gegen jedes Dreieck, exaktes POLYFISHEYE, GPU) — schneller und ohne Durchscheinen verdeckter Flächen. Als Basisschicht unter den Fotos liegen echte Punktfarben.
- **Strukturanalyse deutlich schneller und mehrkernig** (der „6 % CPU"-Befund): Die Zusammenhangs-Analyse pro Fläche baute Millionen-Kanten-Graphen (Minuten pro Fläche!) — ersetzt durch ein Belegungsgitter (Sekunden). Die Polygon-Rekonstruktion aller Flächen läuft jetzt **parallel über die Kerne**, das Zylinder-RANSAC bewertet Hypothesen auf einer Stichprobe (Gewinner exakt auf allen Punkten), Flächen-Nachbarschaften werden C-parallel gezählt. Messung am Testdatensatz: Strukturphase ~4× schneller.
- GUI: Der Ordner-Chip zeigt jetzt korrekt „Kameraposen (ImgPose)", wenn die Quaternion-Posen verwendet werden; die Textur-Checkbox heißt klar „Textur fürs Strukturmodell" (der Foto-Atlas läuft unabhängig davon immer).

### Neu in 3.13.0 — höhere Auslastung: unabhängige Stufen laufen parallel

- **Komplett-Mesh ∥ Strukturanalyse**: Die beiden großen Rechenstufen sind unabhängig und laufen jetzt **gleichzeitig** statt nacheinander — die Wartezeit ist das Maximum statt der Summe, die CPU-Auslastung verdoppelt sich in dieser Phase. (Die Protokollzeilen beider Stufen erscheinen dadurch verschränkt.)
- **Detail-Mesh ∥ Foto-Textur-Atlas**: ebenfalls parallelisiert.
- **Foto-Dekodierung breiter**: bis 16 parallele Decoder-Threads (vorher 8), Atlas-Prefetch 4 Bilder.
- Einordnung zur Auslastung: Zwischen den voll parallelen Phasen (Nachbarschaftssuchen, Foto-Dekodierung, GPU-Kerne) liegen Einzelkern-Abschnitte (RANSAC-Logik, Voxel-Verwaltung) — 100 % Dauerauslastung ist bei dieser Art Pipeline nicht das Ziel, kürzere Gesamtzeit schon. Der Fortschrittsbalken zeigt, wo der Lauf steht; die Zeile „GPU-Nutzung: …" belegt die Grafikkartenarbeit.

### Neu in 3.12.0 — POLYFISHEYE: die Fisheye-Fotos werden exakt projiziert

- **Das vollständige POLYFISHEYE-Modell der S20-Fotokameras ist umgesetzt** (aus der echten `calibration.yaml`): `r(θ) = θ + k2·θ² + … + k7·θ⁷` über den Einfallswinkel, Abbildung über die A-Matrix (A11/A12/A22) und den Bildhauptpunkt (u0/v0), gültig bis 120° Einfallswinkel. Keine Pinhole-Näherung mehr — die Foto-Projektion stimmt jetzt bis in die Bildränder.
- **Jede Stereo-Seite bekommt ihre eigene Kalibrierung**: Fotos unter `left/` nutzen fisheye_left (A11=1480,03), unter `right/` fisheye_right (A11=1476,93) — mit jeweils eigenen Verzeichnungskoeffizienten und Hauptpunkten.
- **Kein Brennweiten-Raten mehr**: Mit dem exakten Werksmodell entfällt die Suche komplett — nur die Rotationskonvention wird noch farb-validiert. Protokoll: „Brennweite 1480 px (calibration.yaml/fisheye_left POLYFISHEYE)".
- Verifiziert mit einem synthetisch über das echte Polynom gerenderten Testfoto (Übereinstimmung > 80 % in der Selbstvalidierung) und den exakten Werkskoeffizienten als Regressionstest.

### Neu in 3.11.0 — verifizierte Korrekturen + Fortschrittsbalken

- **Grauwert-Falle entschärft**: `uncolorized.las` trägt die Intensität als R=G=B — das zählte fälschlich als „farbig". Jetzt prüft die Wolkenauswahl echte Farbdaten (Stichprobe): Grau-Rampen gelten als unfarbig, die Farben kommen zwingend von `colorized.las` bzw. den Fotos. Ein graues Modell kann nicht mehr entstehen.
- **Kamera-Auswahl nach Bildgröße**: `calibration.yaml` enthält MEHRERE Kameras — bisher wurde die erste genommen (die 640×480-Navigationskamera, fx≈548!). Jetzt werden alle Kameras gelesen und die zur Foto-Auflösung (3504×4672) passende gewählt: fisheye_left/right mit A11≈1480, inkl. POLYFISHEYE-Koeffizienten k2… Ohne Größen-Match wird lieber selbstkalibriert als falsch kalibriert.
- **Detail-Mesh der Gebäuderegion**: Neben dem Übersichts-Mesh der Gesamtszene entsteht jetzt ein hochaufgelöstes **`_detail.glb` mit 1–2 cm Raster** (Standard 2 cm) der Region um die erkannten Strukturflächen — einstellbar unter Erweiterte Einstellungen → „Detail-Raster [cm]" bzw. `--detail-raster`.
- **Fortschrittsbalken mit Restzeit-Schätzung**: Die Oberfläche zeigt jetzt einen Balken mit Phase („Komplett-Mesh vernetzen", „Foto-Textur-Atlas", …), Prozent und geschätzter Restdauer.
- **CPU/GPU/RAM live**: Unter dem Balken laufen die Systemwerte mit — CPU %, RAM belegt/gesamt, GPU % (CUDA) und VRAM belegt/gesamt (über nvidia-smi).

### Neu in 3.10.0 — bestmögliches Netz: alle Informationsquellen des S20-Exports

- **Dichteste Wolke als Geometriequelle**: `uncolorized.las` (20,0 Mio Punkte) schlägt jetzt `colorized.las` (18,1 Mio) — die farbige Wolke gewinnt nur noch, wenn sie praktisch gleich groß ist (≥95 %). Die Farben werden verlustarm von der farbigen Schwester übertragen. **+10 % echte Geometrie.**
- **Normalen ZEIT-exakt orientiert**: LAS 1.4 trägt `gps_time` pro Punkt, die Trajektorie Zeitstempel @10 Hz — jeder Punkt zeigt jetzt zur Scannerposition **im Moment seiner Aufnahme** (interpoliert), nicht zur räumlich nächsten. Konstante Uhren-Versätze werden automatisch ausgeglichen; ohne Zeitstempel wie bisher räumlich. Wichtig, wo der Pfad zweimal an derselben Fassade vorbeiführt.
- **Trajektorie-Format-Fix**: Das S20-Layout `x y z roll pitch yaw qx qy qz qw time` (Zeit HINTEN) wurde bisher falsch gelesen (Spaltenverwechslung!) — die Zeitspalte wird jetzt per Monotonie erkannt, vorn oder hinten.
- **`ImgPose.txt` (Quaternionen) wird bevorzugt**: Quaternionen sind eindeutig — statt 8 Rotationskonventionen × 14 Brennweiten reichen wenige farb-validierte Kandidaten. Präzisere Posen → schärfere Foto-Textur. `xyzopk.txt` bleibt Fallback.
- Hinweis POLYFISHEYE: Die Fisheye-Originale (left/right) werden weiterhin nicht direkt projiziert — Bildquelle bleibt der `undistort`-Ordner. Echtes Fisheye-Sampling ist der nächste geplante Schritt.

### Neu in 3.9.1 — Scanner-Kalibrierung wird genutzt (bestmögliches Modell aus dem Projektordner)

- **`info/calibration.yaml` wird jetzt gelesen und verwendet**: Die Werks-Kalibrierung der Scanner-Kamera (echte Brennweite, Bildhauptpunkt) ersetzt das breite Raten der Selbstkalibrierung — gesucht wird nur noch ein schmales Band um den Werkswert, weiterhin farb-validiert am Scan. Ergebnis: **präzisere Foto-Projektion → schärfere Foto-Textur**, und schnellere Kalibrierung. Protokoll: „Brennweite … px (calibration.yaml)" und neue Zeile „Kalibrierung: info/calibration.yaml (fx=… px)". Beide gängigen Formate werden verstanden (flache fx/fy/cx/cy-Schlüssel und OpenCV-`camera_matrix`).
- **Punktlimit sichtbar gemacht**: Wird die Wolke auf das Punktlimit ausgedünnt, sagt das Protokoll jetzt ausdrücklich „Wolke hat X Punkte — auf Y Mio ausgedünnt" mit dem Hinweis, wo sich das Limit erhöhen lässt (GUI: Erweiterte Einstellungen → Max. Punkte; CLI: `--max-points`). Für das bestmögliche Netz zählt jeder Punkt.
- Datei-Inventar: `calibration.yaml` gilt jetzt als VERWENDET mit Begründung; `.opt`-Dateien und Scanner-Logs bleiben zu Recht ungenutzt (keine Geometrie).

### Neu in 3.9.0 — EINE Version: GPU mit automatischem CPU-Fallback

- **Nur noch ein Download**: `scantobim-windows-x64.zip` enthält jetzt immer die GPU-Unterstützung (CUDA-Laufzeit gebündelt) und fällt ohne NVIDIA-Karte automatisch auf die CPU zurück. Der bisherige Link `…-gpu.zip` bleibt gültig und liefert dieselbe Datei. Kein Rätselraten mehr, welche Version läuft.
- **GPU-Nutzung schwarz auf weiß**: Am Ende jedes Laufs listet das Protokoll, was die Grafikkarte wirklich gerechnet hat — z. B. „GPU-Nutzung: Normalen für 15,6 Mio Punkte · 4 Voxel-Sortierungen (84 Mio Schlüssel) · 3 weitere GPU-Läufe" (auch im Bericht als `gpu_nutzung`).
- **Task-Manager-Hinweis direkt im Protokoll**: CUDA-Last erscheint im Task-Manager unter dem Diagramm **„CUDA"/„Compute_0"** — nicht unter „3D". Deshalb wirkt die GPU dort scheinbar untätig, obwohl sie rechnet.
- Die kosmetische CuPy-Warnung „CUDA path could not be detected" ist unterdrückt — mit der gebündelten Laufzeit ist sie bedeutungslos und hat nur verwirrt.
- Die GPU-Tiefendiagnose läuft nur noch einmal pro Programmstart (Caching).

### Neu in 3.8.6 — GPU-Grundursache gefunden und behoben: fehlendes `graphlib`

- **Das neue Release-Gate hat die eigentliche Ursache aller GPU-Fehlstarts seit 3.6.0 gefangen**: CuPy lädt das Python-Standardmodul `graphlib` dynamisch — PyInstallers Abhängigkeits-Scan übersah es, im Exe fehlte es, der CuPy-Import brach ab (`ModuleNotFoundError: No module named 'graphlib'`). Treiber, Karte und CUDA-DLLs waren die ganze Zeit in Ordnung. Das Modul wird jetzt explizit mitgebündelt; das Gate lässt nur noch Builds durch, in denen `import cupy` im fertigen Exe nachweislich gelingt.
- Konsolen-Ausgaben stürzen nicht mehr über Sonderzeichen (→, ✓) auf cp1252-Konsolen ab (`errors="replace"`).

### Neu in 3.8.5 — GPU-Kette wird jetzt VOR jeder Veröffentlichung im fertigen Exe getestet

- **Release-Gate**: Der Build führt die GPU-Diagnose jetzt **im fertigen Windows-Exe auf dem Build-Server aus** — schlägt `import cupy` dort fehl, wird das GPU-Zip gar nicht erst veröffentlicht. Die Diagnose des gebauten Exe wird zusätzlich als Build-Artefakt archiviert.
- **Verschluckte Fehler behoben**: Ein `No module named …`-Fehler wurde bisher pauschal als „normale CPU-Version" gewertet — auch wenn nur ein Teilmodul von CuPy fehlte. Dann gab es weder Fehlerzeile noch `gpu_diagnose.txt`. Jetzt wird nur das komplette Fehlen von CuPy still übergangen; alles andere landet mit Ursache im Protokoll und erzeugt die Diagnose.
- **GPU-Status immer sichtbar**: Die Oberfläche zeigt in der Kopfzeile dauerhaft „GPU: <Karte> ✓", „GPU: CPU-Modus ⚠" (mit Ursache als Tooltip) oder „CPU-Version" — geprüft direkt beim Start, kein Rechenlauf nötig.

### Neu in 3.8.4 — GPU-Fix: CuPys Pfad-Raterei entschärft

- **Die per `gpu_diagnose.txt` gefundene Ursache ist behoben**: CuPy leitet den CUDA-Pfad vom Fundort der cudart-DLL ab; im entpackten Onefile-Paket hielt es `%TEMP%` für die CUDA-Installation und stürzte beim Registrieren des nicht existierenden `%TEMP%\bin` ab (`FileNotFoundError` in `_setup_win32_dll_directory`). Der Import läuft jetzt mit einem nachsichtigen `add_dll_directory` — die korrekten, gebündelten Verzeichnisse sind zu diesem Zeitpunkt längst registriert. Diagnose des betroffenen Systems: Treiber und Karte einwandfrei, alle DLLs einzeln ladbar — nur dieser Startlogik-Absturz stand der GPU im Weg.

### Neu in 3.8.3 — GPU-Diagnose direkt im eigenen Ordner

- Die `gpu_diagnose.txt` wird jetzt **zusätzlich in den Projekt- bzw. Scan-Ordner kopiert** — also dorthin, wo man sie sucht. (Die Oberfläche rechnet in einem versteckten Temp-Ordner; dort lag die Datei bisher nur im Download-Bereich.) Zur Erinnerung: Die Datei entsteht **nur bei einem GPU-Fehlstart** — läuft CUDA, gibt es sie absichtlich nicht.

### Neu in 3.8.2 — automatische GPU-Tiefendiagnose

- **`gpu_diagnose.txt` wird bei jedem GPU-Fehlstart automatisch geschrieben** (neben die Ergebnisdateien; in der Oberfläche erscheint sie bei den Downloads). Sie testet jede Stufe der CUDA-Kette einzeln: Umgebungsvariablen (inkl. `CUDA_PATH`-Altlasten), die gebündelten Laufzeit-Dateien mit Fundort und Größe, jeden einzelnen DLL-Ladevorgang mit Windows-Fehlercode — **auch `nvcuda.dll`, also den NVIDIA-Treiber selbst, samt unterstützter CUDA-Version** —, den CuPy-Import mit vollständigem Traceback und die `nvidia-smi`-Ausgabe. Eine Datei, eine eindeutige Ursache.
- Neuer Befehl **`scantobim.exe gpu`**: schreibt dieselbe Diagnose jederzeit auf Kommando (ohne einen Rechenlauf zu starten).

### Neu in 3.8.1 — GPU-Start doppelt abgesichert

- **CUDA-DLLs liegen jetzt zusätzlich im Paket-Stammverzeichnis** — dort registriert PyInstaller die DLL-Suche selbst, CuPys Module finden cudart/NVRTC damit ganz ohne Eigenlogik, unabhängig von Importreihenfolge und Prozess (auch im neuen Abbrechen-Rechenprozess).
- **`CUDA_PATH`-Altlasten werden überschrieben**: Zeigt die Umgebungsvariable noch auf ein längst deinstalliertes CUDA-Toolkit, gewann sie bisher gegen die gebündelte Laufzeit — jetzt setzt das Programm sie hart auf das mitgelieferte Verzeichnis.
- **Aussagekräftige GPU-Fehlzeile**: CuPys mehrzeilige Fehlermeldung wurde bisher auf ihre (leere) erste Zeile gekürzt → „(ImportError: )". Jetzt wird die eigentliche Ursache extrahiert („Original error: …") und dazu protokolliert, wie viele CUDA-Laufzeit-Ordner im Paket gefunden wurden und wo.
- LIESMICH: Anleitung, wie die Windows-SmartScreen-Warnung („schädlich") **dauerhaft** verschwindet (Datei entpacken → Rechtsklick → Eigenschaften → „Zulassen"). Die Warnung ist bei nicht code-signierter Open-Source-Software normal.

### Neu in 3.8.0 — Abbrechen-Button + GPU-Diagnose

- **⛔ Abbrechen-Button in der Oberfläche**: Eine laufende Berechnung kann jetzt jederzeit sofort abgebrochen werden. Dafür läuft jeder Auftrag in einem **eigenen Prozess** (hartes Beenden mitten in der Rechnung möglich — Python-Threads können das nicht); bereits geschriebene Ergebnisdateien bleiben erhalten, die Oberfläche ist sofort wieder frei für den nächsten Auftrag. Stürzt eine Berechnung ab (z. B. Speicher voll), meldet die Oberfläche das jetzt sauber statt ewig zu laufen.
- **GPU-Treiber-Hinweis**: Meldet CUDA einen zu alten Treiber, sagt das Protokoll jetzt direkt „NVIDIA-Grafiktreiber ist zu alt für CUDA 12 — bitte aktualisieren" mit Download-Link. (Die gebündelte CUDA-12-Laufzeit der GPU-Version braucht Treiber ab Version ~525.)
- Geprüft: Die CUDA-DLLs (cudart, NVRTC) liegen nachweislich im veröffentlichten GPU-Exe — wer die Warnung „CUDA path could not be detected" sieht, hat noch eine Version vor 3.7.0 laufen. Erste Protokollzeile checken: dort steht jetzt immer „ScanToBIM <Version>".

### Neu in 3.7.2 — Programmname mit Versionsnummer überall

- **„ScanToBIM v3.7.2" steht jetzt überall**, wo das Programm sich meldet: Browser-Tab-Titel der Oberfläche, Kopfzeile des 3D-Viewers (`<name>.html`), `bericht.json` (Feld `programm`), GLB-Metadaten (`generator`), OBJ-Kopfzeile — zusätzlich zur ersten Protokollzeile und dem Versions-Badge im GUI-Kopf.

### Neu in 3.7.1 — eindeutige Diagnose

- Das Protokoll beginnt jetzt mit **„ScanToBIM 3.7.1"** — damit ist sofort sichtbar, welche Version wirklich läuft (Verwechslung mit einem älteren entpackten Zip ausgeschlossen).
- Schlägt das **Laden der CUDA-DLLs** fehl (ImportError „DLL load failed …"), steht der konkrete Grund jetzt ebenfalls in der GPU-Protokollzeile statt der allgemeinen CPU-Modus-Meldung.

### Neu in 3.7.0 — GPU-Fix + fotorealistischer Textur-Atlas

- **GPU-Version läuft jetzt ohne CUDA-Toolkit** — Fix für „cuda path could not be detected": Das GPU-Zip von 3.6.0 erwartete die CUDA-Laufzeitbibliotheken vom (auf den meisten PCs nicht installierten) CUDA-Toolkit — deshalb blieb die RTX bei 0 %. Jetzt sind **cudart + NVRTC direkt im Exe gebündelt**; es genügt der normale NVIDIA-Grafiktreiber. Zusätzlich rechnet die Normalenschätzung ihre Eigenvektoren jetzt in geschlossener Form (elementweise Kernels statt cuSOLVER) — weniger Abhängigkeiten, gleicher Output. Wenn CUDA dennoch nicht startet, nennt das Protokoll jetzt den **konkreten Grund** („GPU: CUDA nicht nutzbar (…) — CPU-Modus").
- **Fotorealistisches Komplett-Mesh (View-Dependent Texture Mapping)**: Statt einer Farbe pro Netz-Ecke bekommt das Komplett-Mesh jetzt einen echten **Foto-Textur-Atlas in Foto-Auflösung** — das Verfahren hinter dem „textured mesh" der Photogrammetrie-Tools: Das Netz wird in Charts entfaltet (bis 8192×8192-Atlas), für jedes Dreieck wählt ein Z-Buffer-Sichtbarkeitstest die **am besten blickende, nicht verdeckte Kamera**, und jeder Texel wird bilinear aus dem Originalfoto abgetastet. Texel ohne Fotoabdeckung behalten die Scan-Farben — der Atlas ist immer vollständig.
- Neue Ausgabe **`<name>_foto.glb`**: das fotorealistische Komplett-Mesh mit JPEG-Textur-Atlas — direkt nutzbar in Blender, 3D-Viewern, Präsentationen. Der HTML-Viewer zeigt die Foto-Textur ebenfalls (Layer „Komplett-Mesh (Scan)").
- Die Kamera-Zuweisung des Atlas (alle Kameras × alle Dreiecke + Tiefenpuffer) läuft **auf der GPU**, wenn vorhanden — zusammen mit dem CUDA-Fix wird die Grafikkarte jetzt wirklich ausgelastet.
- Protokoll/Bericht: „Foto-Textur: Atlas … px, … cm/Texel, … % Foto-Anteil (… Kameras)" bzw. `komplett_mesh.foto_textur`.

### Neu in 3.6.0 — GPU-Beschleunigung (NVIDIA CUDA)

- **Neue GPU-Ausgabe `scantobim-windows-x64-gpu.zip`**: nutzt NVIDIA-Grafikkarten (CUDA) für die rechenintensivsten Kerne — Normalenschätzung (Kovarianz + Eigenvektoren für Millionen Nachbarschaften), die Voxel-Sortierung von Komplett-Mesh und Einlese-Ausdünnung sowie die Netz-Glättung. Beim Start zeigt das Protokoll „GPU: <Kartenname> — CUDA-Beschleunigung aktiv“.
- **Automatischer CPU-Rückfall**: keine NVIDIA-Karte, kein Treiber, GPU-Speicher voll — jede GPU-Operation fällt transparent auf die CPU zurück, Ergebnisse sind identisch. Die normale `scantobim-windows-x64.zip` bleibt die kompakte CPU-Version.
- Auf der GPU wird bewusst mit fp32 gerechnet, wo die Genauigkeit es erlaubt (Normalen, Glättung) — Consumer-Karten rechnen fp64 mit 1/32-Rate; die maßkritischen Ebenen-Fits (TLS) bleiben fp64 auf der CPU.

### Neu in 3.5.0 — Turbo: deutlich schnellere Berechnung (CPU)

- **Vorverarbeitung nur noch einmal statt viermal**: Szene „Automatisch“ hat für jeden der vier Kandidaten die komplette Vorverarbeitung wiederholt (Ausdünnen, Entrauschen, Normalenschätzung auf Millionen Punkten) — obwohl alle Kandidaten dieselben Parameter dafür nutzen. Jetzt läuft sie **einmal** und wird geteilt (auch für den finalen wasserdichten Lauf).
- **Stichproben-RANSAC**: Die Ebenensuche bewertete jede Hypothese gegen die *gesamte* Wolke — bei 10 Mio Punkten × 600 Versuchen der größte Einzelposten. Jetzt wird auf einer 400 000-Punkte-Stichprobe gesucht; Gewinner-Ebenen werden weiterhin **exakt auf allen Punkten** nachgesammelt und verfeinert — gleiche Ergebnisse, Bruchteil der Zeit.
- **Foto-Dekodierung parallel**: Die Foto-Farben fürs Komplett-Mesh dekodieren die (bis zu 600) Bilder jetzt in 8 parallelen Threads.
- Hinweis zur Grafikkarte: Die Berechnung ist bewusst CPU-basiert (läuft überall, ohne Treiber-/CUDA-Abhängigkeiten in einer einzigen EXE) — die 1 % GPU-Auslastung ist normal. Die Beschleunigung kommt aus den Algorithmen; auf großen Scans sollte Szene „Automatisch“ jetzt grob 2–4× schneller sein.

### Neu in 3.4.0 — Datei-Inventar: was liegt im Ordner und warum wird es (nicht) verwendet?

- Nach der Projektordner-Analyse listet das Protokoll jetzt ein **Datei-Inventar** aller gefundenen, aber **nicht verwendeten** Dateien — jede mit Größe und Begründung: weitere Punktwolken (das Dichte/Farb-Ranking hat entschieden), die `.bag`-Rohaufnahme (ihre verarbeiteten Ergebnisse werden direkt genutzt), Scanner-Logs/Metadaten, Bilder außerhalb des gewählten Foto-Ordners (Vorschauen), überzählige COLMAP-Modelle oder Trajektorien, unbekannte Formate. Gleiche Dateien werden pro Ordner gruppiert (z. B. „*.log (5 Dateien)“), sortiert nach Größe.
- Werden alle erkannten Dateien genutzt, steht das ebenso ausdrücklich im Protokoll.

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
