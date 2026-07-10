## ScanToBIM für Windows

**Download:** `scantobim-windows-x64.zip` entpacken und `scantobim.exe` in der Eingabeaufforderung nutzen — keine Installation, kein Python nötig.

> SmartScreen-Hinweis beim ersten Start (Datei ist nicht code-signiert): *Weitere Informationen → Trotzdem ausführen*.

### Funktionen

- **Rekonstruktion**: Punktwolken (`.las/.laz/.e57/.ply/.pcd/.pts/.xyz`) → Modelle mit schnurgeraden Kanten, exakten 90°-Winkeln, Fenster-/Türöffnungen, Bauteil-Klassifikation, Mengenermittlung, Mehrgeschoss-Erkennung
- **Fotorealistische Texturierung** aus Punktwolkenfarben oder per Foto-Projektion (COLMAP-Posen, verdeckungsgeprüft)
- **Export**: STEP (CAD-B-Rep, auch der Weg nach HiCAD), IFC4, GLB/OBJ/STL, DXF-Grundriss, interaktiver HTML-Viewer
- **Industrie-Analyse**: Wellen, Zahnräder (Zähnezahl + DIN-Modul), Getriebestufen, Kegel, Stahlprofile (IPE/HEA/HEB/UPN/SHS/RHS/L), Anschlussdetails
- **Blechkonstruktionen**: Blechdicken, Schweißnahtlängen, Gewichte, Brennschnitt-DXF 1:1
- **Brückenbauwerke**: Typ-Klassifikation, Feldweiten, Pfeiler mit Lagerpunkten, Bogen, Pylone, Seile
- **Multi-Scan**: ICP-Registrierung und Verschmelzung mehrerer Standpunkte

Schnellstart und alle Befehle: `LIESMICH.txt` im Archiv. Die Exe wurde auf dem Windows-Runner automatisch verifiziert (Rekonstruktion + STEP-Export eines Testscans).
