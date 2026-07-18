"""The GUI app page — one self-contained HTML document (inline CSS/JS)."""

from __future__ import annotations

from scantobim import __version__


def render_page() -> str:
    return _PAGE.replace("__VERSION__", __version__)


_PAGE = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ScanToBIM v__VERSION__ — Punktwolke zu 3D-Modell</title>
<style>
:root{
  --bg:#0e1116; --panel:#161b23; --panel2:#1b212b; --line:#28303d;
  --text:#dce3ec; --muted:#8b96a5; --accent:#5c9bff; --accent2:#3f7ce0;
  --ok:#4dc27d; --err:#e06c6c; --radius:10px;
  font-size:15px;
}
*{box-sizing:border-box; margin:0; padding:0}
body{
  background:var(--bg); color:var(--text); min-height:100vh;
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
}
header{
  display:flex; align-items:center; gap:.75rem;
  padding:.8rem 1.4rem; border-bottom:1px solid var(--line);
  background:linear-gradient(180deg,#141922,#10141b);
  position:sticky; top:0; z-index:10;
}
.logo{
  width:26px;height:26px;border-radius:7px;
  background:linear-gradient(135deg,var(--accent),#7fd0a8);
  box-shadow:0 0 14px rgba(92,155,255,.45);
}
header h1{font-size:1.05rem; font-weight:650; letter-spacing:.02em}
header .ver{color:var(--muted); font-size:.8rem; margin-left:.1rem}
#status{
  margin-left:auto; font-size:.8rem; padding:.28rem .8rem; border-radius:99px;
  border:1px solid var(--line); color:var(--muted); background:var(--panel);
}
#status.run{color:#ffd479; border-color:#5a4a22}
#status.ok{color:var(--ok); border-color:#2c5e41}
#status.err{color:var(--err); border-color:#6e3434}
main{
  display:grid; grid-template-columns:minmax(320px,380px) 1fr; gap:1rem;
  padding:1rem 1.4rem; max-width:1700px; margin:0 auto;
}
@media (max-width:980px){ main{grid-template-columns:1fr} }
.card{
  background:var(--panel); border:1px solid var(--line); border-radius:var(--radius);
  padding:1rem 1.1rem;
}
.card h2{
  font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--muted); margin-bottom:.7rem; font-weight:600;
}
.side{display:flex; flex-direction:column; gap:1rem}
.content{display:flex; flex-direction:column; gap:1rem; min-width:0}

/* --- drop zone --- */
#drop{
  border:1.5px dashed #34404f; border-radius:var(--radius);
  padding:1.1rem; text-align:center; color:var(--muted);
  cursor:pointer; transition:all .15s; font-size:.88rem; line-height:1.5;
}
#drop:hover,#drop.hover{border-color:var(--accent); color:var(--text); background:#182031}
#drop b{color:var(--text)}
.pathrow{display:flex; gap:.5rem; margin-top:.6rem}
.pathrow input{
  flex:1; background:var(--panel2); border:1px solid var(--line); color:var(--text);
  border-radius:7px; padding:.45rem .6rem; font-size:.82rem;
}
.pathrow button{white-space:nowrap}
#filelist{display:flex; flex-direction:column; gap:.4rem; margin-top:.6rem}
.file{
  display:flex; align-items:center; gap:.55rem; background:var(--panel2);
  border:1px solid var(--line); border-radius:8px; padding:.45rem .6rem; font-size:.84rem;
}
.file .nm{flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
.file .sz{color:var(--muted); font-size:.75rem}
.file .rm{cursor:pointer; color:var(--muted); border:none; background:none; font-size:1rem}
.file .rm:hover{color:var(--err)}

/* --- mode cards --- */
.modes{display:grid; grid-template-columns:1fr 1fr; gap:.55rem}
.mode{
  border:1px solid var(--line); border-radius:9px; background:var(--panel2);
  padding:.65rem .7rem; cursor:pointer; transition:all .12s;
}
.mode:hover{border-color:#3a4a5f}
.mode.sel{border-color:var(--accent); background:#17233a; box-shadow:0 0 0 1px var(--accent) inset}
.mode .t{font-size:.86rem; font-weight:600}
.mode .d{font-size:.72rem; color:var(--muted); margin-top:.15rem; line-height:1.35}

/* --- options --- */
.opt{display:flex; align-items:center; gap:.55rem; padding:.3rem 0; font-size:.87rem}
.opt input[type=checkbox]{accent-color:var(--accent); width:15px; height:15px}
select{
  background:var(--panel2); color:var(--text); border:1px solid var(--line);
  border-radius:7px; padding:.42rem .55rem; font-size:.85rem; width:100%;
}
label.sel-label{font-size:.78rem; color:var(--muted); display:block; margin:.4rem 0 .25rem}
.fmt{display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.3rem}
.fmt label{
  border:1px solid var(--line); background:var(--panel2); border-radius:99px;
  padding:.25rem .7rem; font-size:.78rem; cursor:pointer; user-select:none;
}
.fmt label.on{border-color:var(--accent); color:var(--accent); background:#17233a}
.fmt label.busy{opacity:.55; cursor:progress}
.fmt input{display:none}

button.primary{
  width:100%; padding:.75rem; border:none; border-radius:9px; cursor:pointer;
  background:linear-gradient(180deg,var(--accent),var(--accent2));
  color:#fff; font-size:.95rem; font-weight:650; letter-spacing:.02em;
  transition:filter .12s;
}
button.primary:hover{filter:brightness(1.1)}
button.primary:disabled{filter:grayscale(.6) brightness(.7); cursor:not-allowed}
button.ghost{
  background:var(--panel2); border:1px solid var(--line); color:var(--text);
  border-radius:7px; padding:.45rem .8rem; font-size:.82rem; cursor:pointer;
}
button.ghost:hover{border-color:var(--accent)}

/* --- viewer --- */
#viewerwrap{position:relative; height:min(56vh,560px)}
#viewer{
  width:100%; height:100%; border:none; border-radius:8px; background:#0b0e12;
  display:none;
}
#placeholder{
  position:absolute; inset:0; display:flex; flex-direction:column; gap:.6rem;
  align-items:center; justify-content:center; color:var(--muted);
  border:1px dashed #2b3442; border-radius:8px; font-size:.9rem; text-align:center;
  padding:1rem;
}
#placeholder .big{font-size:2.2rem; opacity:.5}

/* --- log + report --- */
.duo{display:grid; grid-template-columns:1fr 1fr; gap:1rem}
@media (max-width:1200px){ .duo{grid-template-columns:1fr} }
#log{
  font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:.76rem;
  background:#0b0e13; border:1px solid var(--line); border-radius:8px;
  padding:.7rem .8rem; height:240px; overflow-y:auto; white-space:pre-wrap;
  color:#a9b6c6; line-height:1.5;
}
#report{width:100%; border-collapse:collapse; font-size:.85rem}
#report td{padding:.34rem .4rem; border-bottom:1px solid #202835; vertical-align:top}
#report td:first-child{color:var(--muted); width:46%}
#downloads{display:flex; flex-wrap:wrap; gap:.5rem}
.dl{
  display:inline-flex; align-items:center; gap:.45rem; text-decoration:none;
  color:var(--text); background:var(--panel2); border:1px solid var(--line);
  border-radius:8px; padding:.45rem .8rem; font-size:.83rem; transition:all .12s;
}
.dl:hover{border-color:var(--accent); color:var(--accent)}
.dl .sz{color:var(--muted); font-size:.72rem}
.hint{color:var(--muted); font-size:.78rem; line-height:1.5}
.qinfo{
  display:inline-block; color:var(--accent); cursor:help; font-size:.8rem;
  margin-left:.2rem;
}
.adv{display:flex; flex-direction:column; gap:.35rem; margin:.5rem 0}
.adv label{
  display:flex; align-items:center; justify-content:space-between; gap:.6rem;
  font-size:.8rem; color:var(--text); cursor:help;
  border-bottom:1px dotted #242c38; padding:.15rem 0;
}
.adv input{
  width:6.5rem; background:var(--panel2); border:1px solid var(--line);
  color:var(--text); border-radius:6px; padding:.3rem .45rem; font-size:.8rem;
  text-align:right;
}
#advdetails summary::-webkit-details-marker{color:var(--muted)}
</style>
</head>
<body>
<header>
  <div class="logo"></div>
  <h1>ScanToBIM</h1><span class="ver">v__VERSION__</span>
  <span class="ver" id="gpubadge" title="GPU-Status wird geprüft …">GPU: prüfe …</span>
  <div id="status">bereit</div>
</header>
<main>
  <div class="side">
    <div class="card">
      <h2>1 · Punktwolke</h2>
      <div id="drop">
        <b>Dateien hierher ziehen</b> oder klicken<br>
        .las · .laz · .e57 · .ply · .pcd · .pts · .xyz
      </div>
      <input type="file" id="fileinput" multiple style="display:none"
             accept=".las,.laz,.e57,.ply,.pcd,.pts,.xyz,.txt,.csv,.asc">
      <div class="pathrow">
        <input id="pathinput" placeholder="… oder Datei-/Projektordner-Pfad einfügen">
        <button class="ghost" id="addpath">Hinzufügen</button>
      </div>
      <div class="hint" style="margin-top:.45rem">
        SLAM-Scanner-Projekte (z.&nbsp;B. SHARE&nbsp;SLAM&nbsp;S20): einfach den
        <b>Projektordner-Pfad</b> einfügen — Punktwolke, Fotos, Kameraposen
        und Trajektorie werden automatisch erkannt.
      </div>
      <div id="filelist"></div>
    </div>

    <div class="card">
      <h2>2 · Was wurde gescannt?</h2>
      <div class="modes">
        <div class="mode sel" data-mode="reconstruct">
          <div class="t">Gebäude / Raum</div>
          <div class="d">Modell mit sauberen Kanten, Öffnungen, Mengen</div>
        </div>
        <div class="mode" data-mode="analyze">
          <div class="t">Maschinenbau / Stahl</div>
          <div class="d">Wellen, Zahnräder, Getriebe, Profile, Anschlüsse</div>
        </div>
        <div class="mode" data-mode="sheetmetal">
          <div class="t">Blechkonstruktion</div>
          <div class="d">Dicken, Schweißnähte, Zuschnitt, Abwicklung</div>
        </div>
        <div class="mode" data-mode="bridge">
          <div class="t">Brückenbauwerk</div>
          <div class="d">Typ, Felder, Pfeiler, Bogen, Seile — als 3D-Modell</div>
        </div>
        <div class="mode" data-mode="compare" style="grid-column:1/-1">
          <div class="t">Epochen-Vergleich (Monitoring)</div>
          <div class="d">Zwei Scans desselben Objekts: Verformung/Setzung als Heatmap — erst den alten, dann den neuen Scan hinzufügen</div>
        </div>
      </div>
    </div>

    <div class="card" id="optcard">
      <h2>3 · Einstellungen</h2>
      <div id="opts-reconstruct">
        <label class="sel-label">Quelle / Scanner
          <span class="qinfo" title="Sensor-Profil: passt Toleranzen an das Rausch- und Driftverhalten des Aufnahmegeräts an. Bei SLAM-Projektordnern wird automatisch „SLAM-Handscanner“ gewählt.">ⓘ</span>
        </label>
        <select id="source">
          <option value="standard" title="Neutrale Standardwerte — wenn die Quelle unbekannt ist.">Standard / unbekannt</option>
          <option value="slam" title="Handheld-SLAM (SHARE SLAM S20, GeoSLAM, LiGrip …): toleranter gegen cm-Drift, verschmilzt Registrierungs-Doppelwände bis 3 cm.">SLAM-Handscanner (z.&nbsp;B. SHARE S20)</option>
          <option value="tls" title="Terrestrischer Laserscanner auf Stativ: mm-Rauschen — engere Toleranzen, feinere Konturen.">Terrestrischer Scanner (Stativ)</option>
          <option value="drohne" title="Drohnen-Photogrammetrie / Luft-LiDAR: rauere Oberflächen und Ausreißer — großzügigere Toleranzen.">Drohne / Photogrammetrie</option>
          <option value="iphone" title="iPhone/iPad-LiDAR-Apps: grobe, geglättete Tiefe — deutlich größere Toleranzen und größere Mindestflächen.">iPhone / iPad LiDAR</option>
        </select>
        <label class="sel-label">Szene</label>
        <select id="preset">
          <option value="auto">Automatisch — selbstoptimierend (dauert länger)</option>
          <option value="building" selected>Gebäude außen</option>
          <option value="indoor">Innenraum</option>
          <option value="object">Einzelobjekt / Bauteil</option>
          <option value="detail">Detailgetreu (mehr Flächen + Stützen/Rohre)</option>
          <option value="fast">Schnell (Vorschau)</option>
        </select>
        <div class="hint" style="margin-top:.45rem" title="Beispiel: Szene „Gebäude außen“ setzt die Grundwerte, Quelle „SLAM“ ändert davon Toleranz und Geister-Versatz, und ein ausgefülltes erweitertes Feld ersetzt beides.">
          <b>Was gilt wann?</b> Szene setzt die Grundwerte → das Quelle-Profil
          passt sie an den Scanner an → <b>ausgefüllte</b> erweiterte Felder
          überschreiben beides. Leere erweiterte Felder = Automatik.
        </div>
        <div style="margin-top:.6rem">
          <div class="sel-label">Berechnung <span style="color:var(--ok)">(Standard: alle drei an — einfach so lassen)</span></div>
          <label class="opt" title="Stufe 1: Der GESAMTE Scan wird als farbiges Dreiecksnetz rekonstruiert (komplett.glb) plus Detail-Mesh der Gebäuderegion (_detail.glb, 2 cm) — beide mit Foto-Atlas, wenn Fotos vorliegen."><input type="checkbox" id="freeform" checked> Komplett-Mesh + Detail-Mesh (fotorealistisch)</label>
          <label class="opt" title="Stufe 2: Ebenen, Kanten und Linien werden gesucht — liefert das Strukturmodell mit Maßen, Öffnungen, Bauteilklassen. Nur mit dieser Stufe sind die CAD/BIM-Exporte (STEP, IFC, DXF) möglich."><input type="checkbox" id="structure" checked> Strukturmodell: Ebenen &amp; Kanten, Maße, Öffnungen</label>
          <label class="opt" title="Textur des Strukturmodells: Fotos, sonst Punktfarben. Der Foto-Atlas von Komplett- und Detail-Mesh läuft davon unabhängig immer."><input type="checkbox" id="texture" checked> Strukturmodell texturieren</label>
        </div>
        <details style="margin-top:.4rem">
          <summary class="sel-label" style="cursor:pointer">Zusatzauswertungen <span style="color:var(--muted)">(Standard: alle aus)</span></summary>
          <label class="opt" title="Schließt Scanschatten zu einem geschlossenen Volumenkörper — für Volumenermittlung."><input type="checkbox" id="watertight"> Wasserdichtes Volumenmodell (schließt Scanschatten)</label>
          <label class="opt"><input type="checkbox" id="align"> Achsen ausrichten, Boden auf Z=0</label>
          <label class="opt"><input type="checkbox" id="register"> Mehrere Scans automatisch registrieren (ICP)</label>
          <label class="opt"><input type="checkbox" id="deviation"> Soll-Ist-Abweichungsanalyse (QS-Heatmap + Statistik)</label>
          <label class="opt"><input type="checkbox" id="views"> Orthofoto-Ansichten N/O/S/W + Draufsicht (maßstabsgetreu)</label>
          <label class="opt"><input type="checkbox" id="reporthtml"> Druckfertiger Prüfbericht (HTML → PDF)</label>
        </details>
        <details id="advdetails">
          <summary class="sel-label" style="cursor:pointer">Erweiterte Einstellungen
            <span class="qinfo" title="Nur für Feinjustierung. Ausgefüllte Felder überschreiben Szene UND Quelle-Profil. Leere Felder = Automatik. Erklärung: Maus über die Bezeichnung halten.">ⓘ</span>
          </summary>
          <div class="hint" style="margin:.3rem 0">Ausgefüllte Felder überschreiben
          Szene <b>und</b> Quelle-Profil — leer lassen heißt: die Automatik
          entscheidet.</div>
          <div class="adv">
            <label title="Ausdünnungsraster der Vorverarbeitung in Metern. Leer = automatisch (2× Punktabstand). 0 = keine Ausdünnung (langsamer, maximales Detail).">Voxelgröße [m] <input id="adv_voxel_size" placeholder="auto"></label>
            <label title="Wie weit ein Messpunkt von einer Ebene entfernt sein darf, um noch dazuzugehören — als Vielfaches des Punktabstands. Größer = robuster gegen Rauschen/Drift, kleiner = mehr Detailtreue. Standard 3 (SLAM 3,5 / Stativ 2,5 / iPhone 5).">Ebenen-Toleranz [×&nbsp;Punktabstand] <input id="adv_distance_factor" placeholder="auto"></label>
            <label title="Kleinste erkannte Fläche als Anteil der Punktwolke in Prozent. Kleiner = auch kleine Flächen (Laibungen, Möbel), aber mehr Rechenzeit. Standard 1 %.">Mindest-Flächengröße [%] <input id="adv_min_inlier" placeholder="auto"></label>
            <label title="Obergrenze der erkannten Flächen. Mehr Flächen = mehr Details, längere Rechenzeit. Standard 64.">Max. Flächen <input id="adv_max_planes" placeholder="auto"></label>
            <label title="Bis zu diesem Winkel werden fast-parallele bzw. fast-rechtwinklige Flächen auf exakt 0°/90° gerastet. Standard 8°. Bei bewusst schiefen Bauwerken kleiner wählen.">Winkel-Raster-Toleranz [°] <input id="adv_angle_tol" placeholder="auto"></label>
            <label title="SLAM-Registrierung erzeugt manchmal doppelte Wände mit kleinem Versatz (Geisterflächen). Koplanare Flächen bis zu diesem Versatz in Metern werden verschmolzen. 0 = aus. SLAM-Profil: 0,03.">Geister-Versatz [m] <input id="adv_ghost" placeholder="auto"></label>
            <label title="Mindestgröße erkannter Öffnungen (Fenster/Türen) als Vielfaches des Punktabstands. Größer = weniger falsche Öffnungen durch Abschattungen. Standard 8.">Öffnungs-Mindestgröße [×&nbsp;Punktabstand] <input id="adv_min_opening" placeholder="auto"></label>
            <label title="Toleranz der Soll-Ist-Abweichungsanalyse in Millimetern — bestimmt die Quote 'innerhalb Toleranz'. Standard 5 mm.">QS-Toleranz [mm] <input id="adv_tolerance" placeholder="auto"></label>
            <label title="Speicherschutz: riesige Scans werden beim Einlesen blockweise auf diese Punktzahl (in Millionen) ausgedünnt. Standard 40.">Max. Punkte [Mio.] <input id="adv_max_points" placeholder="auto"></label>
            <label title="Raster des hochaufgelösten Detail-Mesh der Gebäuderegion in Zentimetern (wird zusätzlich als _detail.glb geschrieben). Standard 2 cm. 0 = aus.">Detail-Raster [cm] <input id="adv_detail" placeholder="auto"></label>
          </div>
        </details>
        <button class="ghost" id="resetopts" style="margin-top:.55rem;width:100%"
          title="Setzt Quelle, Szene, alle Haken und alle erweiterten Felder auf die Standardwerte zurück.">↺ Alles auf Standard zurücksetzen</button>
        <div class="hint" style="margin-top:.5rem">
          <b>Exportformate?</b> Erst rechnen — danach unter „Ergebnisdateien“
          per Klick speichern: STEP, IFC, DXF, GLB, OBJ, STL, PLY.
        </div>
      </div>
      <div id="opts-sheetmetal" style="display:none">
        <label class="opt"><input type="checkbox" id="unfold" checked>
          Kantungen abwickeln (Flachteile mit Biegelinien als DXF)</label>
      </div>
      <div id="opts-none" style="display:none">
        <div class="hint">Alle Messungen laufen automatisch — Ergebnis rechts als 3D-Modell und Bericht.</div>
      </div>
    </div>

    <div class="card">
      <h2>Profile</h2>
      <div class="pathrow">
        <select id="profsel" style="flex:1">
          <option value="">— gespeichertes Profil laden —</option>
        </select>
        <button class="ghost" id="profdel" title="Ausgewähltes Profil löschen">Löschen</button>
      </div>
      <div class="pathrow">
        <input id="profname" placeholder="Name für aktuelle Einstellungen">
        <button class="ghost" id="profsave" title="Alle aktuellen Einstellungen (Quelle, Szene, Optionen, erweiterte Werte) dauerhaft speichern">Speichern</button>
      </div>
      <div class="hint" style="margin-top:.45rem">Profile werden dauerhaft auf
      diesem Rechner gespeichert — z.&nbsp;B. „S20 außen“, „S20 innen fein“,
      „Drohne Halle“.</div>
    </div>

    <button class="primary" id="run">Modell erstellen</button>
    <button class="ghost" id="cancel" style="display:none"
      title="Bricht die laufende Berechnung sofort ab. Bereits geschriebene Ergebnisdateien bleiben erhalten.">⛔ Abbrechen</button>
    <div id="progwrap" style="display:none;margin-top:.6rem">
      <div style="height:8px;border-radius:99px;background:var(--line);overflow:hidden">
        <div id="progbar" style="height:100%;width:0%;background:linear-gradient(90deg,var(--accent),#7fd0a8);transition:width .6s"></div>
      </div>
      <div id="progtext" style="font-size:.75rem;color:var(--muted);margin-top:.3rem"></div>
    </div>
    <div id="sysstats" style="font-size:.72rem;color:var(--muted);margin-top:.4rem"></div>
  </div>

  <div class="content">
    <div class="card">
      <h2>3D-Modell</h2>
      <div id="viewerwrap">
        <iframe id="viewer" title="3D-Modell"></iframe>
        <div id="placeholder">
          <div class="big">◇</div>
          <div>Noch kein Modell — links Punktwolke wählen und<br>„Modell erstellen“ drücken.</div>
        </div>
      </div>
    </div>
    <div class="card" id="dlcard" style="display:none">
      <h2>Ergebnisdateien</h2>
      <div id="downloads"></div>
      <div id="exportrow" style="display:none;margin-top:.7rem">
        <div class="sel-label" title="Das Modell ist bereits berechnet — jedes Format wird beim Klick aus dem Ergebnis erzeugt und heruntergeladen. Kein neuer Rechenlauf nötig.">Speichern als … (Strukturmodell)</div>
        <div class="fmt" id="exports">
          <label data-f="step" title="CAD-B-Rep für HiCAD, SolidWorks, Inventor … (Datei &gt; Import &gt; STEP)">STEP (CAD)</label>
          <label data-f="ifc" title="BIM-Austausch für Revit, ArchiCAD, BIMcollab …">IFC (BIM)</label>
          <label data-f="dxf" title="Bemaßter 2D-Grundriss für AutoCAD &amp; Co.">DXF-Grundriss</label>
          <label data-f="glb" title="3D-Netz mit Textur für Blender, Viewer, Präsentationen">GLB</label>
          <label data-f="obj" title="3D-Netz mit Textur (klassisches Austauschformat)">OBJ</label>
          <label data-f="stl" title="Reines Dreiecksnetz, z. B. für 3D-Druck">STL</label>
          <label data-f="ply" title="Dreiecksnetz mit Farben (Punktwolken-Software)">PLY</label>
        </div>
        <div class="hint" id="exporthint" style="margin-top:.35rem"></div>
      </div>
    </div>
    <div class="duo">
      <div class="card">
        <h2>Messbericht</h2>
        <table id="report"><tbody><tr><td colspan="2" class="hint">—</td></tr></tbody></table>
      </div>
      <div class="card">
        <h2>Protokoll</h2>
        <div id="log">bereit.</div>
      </div>
    </div>
  </div>
</main>

<script>
"use strict";
const $ = s => document.querySelector(s);
const state = { files: [], mode: "reconstruct", job: null, timer: null, profiles: {} };

function fmtSize(b){
  if (b > 1e9) return (b/1e9).toFixed(2) + " GB";
  if (b > 1e6) return (b/1e6).toFixed(1) + " MB";
  return Math.max(1, Math.round(b/1e3)) + " kB";
}
function setStatus(txt, cls){
  const el = $("#status"); el.textContent = txt; el.className = cls || "";
}
function renderFiles(){
  const list = $("#filelist"); list.innerHTML = "";
  state.files.forEach((f, i) => {
    const div = document.createElement("div"); div.className = "file";
    div.innerHTML = `<span class="nm" title="${f.path}">${f.name}</span>` +
      `<span class="sz">${fmtSize(f.size)}</span>`;
    const rm = document.createElement("button");
    rm.className = "rm"; rm.textContent = "✕"; rm.title = "entfernen";
    rm.onclick = () => { state.files.splice(i, 1); renderFiles(); };
    div.appendChild(rm); list.appendChild(div);
  });
}
async function uploadFile(file){
  setStatus("lade " + file.name + " …", "run");
  const res = await fetch("/api/upload", {
    method: "POST", body: file,
    headers: {"X-Filename": encodeURIComponent(file.name)},
  });
  const data = await res.json();
  if (data.error) { alert(data.error); setStatus("bereit"); return; }
  state.files.push(data); renderFiles(); setStatus("bereit");
}

/* drop zone */
const drop = $("#drop");
drop.onclick = () => $("#fileinput").click();
$("#fileinput").onchange = e => [...e.target.files].forEach(uploadFile);
drop.ondragover = e => { e.preventDefault(); drop.classList.add("hover"); };
drop.ondragleave = () => drop.classList.remove("hover");
drop.ondrop = e => {
  e.preventDefault(); drop.classList.remove("hover");
  [...e.dataTransfer.files].forEach(uploadFile);
};
$("#addpath").onclick = async () => {
  const p = $("#pathinput").value.trim(); if (!p) return;
  const res = await fetch("/api/addpath", {
    method: "POST", body: JSON.stringify({path: p}),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  state.files.push(data); $("#pathinput").value = ""; renderFiles();
  /* SLAM project folder → pre-select the matching source profile */
  if (data.slam && $("#source").value === "standard") {
    $("#source").value = "slam";
    syncSourceTitle();
    setStatus("SLAM-Projekt erkannt — Quelle auf „SLAM-Handscanner“ gestellt", "ok");
  }
};

/* modes */
document.querySelectorAll(".mode").forEach(el => {
  el.onclick = () => {
    document.querySelectorAll(".mode").forEach(m => m.classList.remove("sel"));
    el.classList.add("sel");
    state.mode = el.dataset.mode;
    $("#opts-reconstruct").style.display = state.mode === "reconstruct" ? "" : "none";
    $("#opts-sheetmetal").style.display = state.mode === "sheetmetal" ? "" : "none";
    $("#opts-none").style.display =
      (state.mode === "analyze" || state.mode === "bridge"
       || state.mode === "compare") ? "" : "none";
  };
});

/* "Speichern als …" — formats are generated on demand AFTER the run */
document.querySelectorAll("#exports label").forEach(el => {
  el.onclick = async () => {
    if (!state.job || el.classList.contains("busy")) return;
    el.classList.add("busy");
    const orig = el.textContent;
    el.textContent = orig + " …";
    $("#exporthint").textContent = "";
    try {
      const res = await fetch("/api/export", {
        method: "POST",
        body: JSON.stringify({job: state.job, fmt: el.dataset.f}),
      });
      const data = await res.json();
      if (data.error) { $("#exporthint").textContent = "⚠ " + data.error; return; }
      renderDownloads(data.outputs);
      /* trigger the download right away */
      const a = document.createElement("a");
      a.href = "/api/output?job=" + state.job + "&name=" + encodeURIComponent(data.name);
      a.download = data.name;
      document.body.appendChild(a); a.click(); a.remove();
      $("#exporthint").textContent = "✓ " + data.name + " erzeugt (" + fmtSize(data.size) + ") — Download gestartet, liegt auch unter Ergebnisdateien.";
    } finally {
      el.classList.remove("busy");
      el.textContent = orig;
    }
  };
});

/* reset all options to their defaults */
const DEFAULTS = {
  preset: "building", source: "standard",
  freeform: true, structure: true, texture: true,
  watertight: false, align: false, register: false,
  deviation: false, views: false, report_html: false,
  advanced: {},
};
$("#resetopts").onclick = () => {
  applySettings(DEFAULTS);
  $("#adv_angle_tol").value = ""; $("#adv_tolerance").value = "";
  $("#adv_max_points").value = ""; $("#adv_detail").value = "";
  $("#profsel").value = ""; $("#profname").value = "";
  setStatus("Standardwerte wiederhergestellt", "ok");
};

/* ---- settings gathering / applying (also used by profiles) ---- */
const ADV_MAP = {  /* input id → backend key + scale */
  adv_voxel_size: ["voxel_size", 1],
  adv_distance_factor: ["distance_factor", 1],
  adv_min_inlier: ["min_inlier_ratio", 0.01],
  adv_max_planes: ["max_planes", 1],
  adv_ghost: ["ghost_offset_tol", 1],
  adv_min_opening: ["min_opening_factor", 1],
};
function gatherOptions(){
  const options = {};
  options.preset = $("#preset").value;
  options.source = $("#source").value;
  options.watertight = $("#watertight").checked;
  options.texture = $("#texture").checked;
  options.freeform = $("#freeform").checked;
  options.structure = $("#structure").checked;
  options.align = $("#align").checked;
  options.register = $("#register").checked && state.files.length > 1;
  options.deviation = $("#deviation").checked;
  options.views = $("#views").checked;
  options.report_html = $("#reporthtml").checked;
  options.advanced = {};
  for (const [id, [key, scale]] of Object.entries(ADV_MAP)) {
    const v = parseFloat($("#" + id).value.replace(",", "."));
    if (!isNaN(v)) options.advanced[key] = v * scale;
  }
  const angle = parseFloat($("#adv_angle_tol").value.replace(",", "."));
  if (!isNaN(angle)) {
    options.advanced.ortho_tol_deg = angle;
    options.advanced.parallel_tol_deg = angle;
  }
  const tol = parseFloat($("#adv_tolerance").value.replace(",", "."));
  if (!isNaN(tol)) options.tolerance = tol / 1000.0;
  const mp = parseFloat($("#adv_max_points").value.replace(",", "."));
  if (!isNaN(mp)) options.max_points = Math.round(mp * 1e6);
  const dr = parseFloat($("#adv_detail").value.replace(",", "."));
  if (!isNaN(dr)) options.detail_raster = dr / 100.0;
  options.unfold = $("#unfold").checked;
  return options;
}
function applySettings(s){
  if (!s) return;
  if (s.preset) $("#preset").value = s.preset;
  if (s.source) $("#source").value = s.source;
  for (const id of ["watertight","texture","freeform","structure","align","register","deviation","views","reporthtml"]) {
    const key = id === "reporthtml" ? "report_html" : id;
    if (key in s) $("#" + id).checked = !!s[key];
  }
  for (const [id, [key, scale]] of Object.entries(ADV_MAP)) {
    $("#" + id).value = (s.advanced && key in s.advanced)
      ? String(s.advanced[key] / scale) : "";
  }
  $("#adv_angle_tol").value =
    (s.advanced && "ortho_tol_deg" in s.advanced) ? String(s.advanced.ortho_tol_deg) : "";
  $("#adv_tolerance").value = ("tolerance" in s) ? String(s.tolerance * 1000) : "";
  $("#adv_max_points").value = ("max_points" in s) ? String(s.max_points / 1e6) : "";
  $("#adv_detail").value = ("detail_raster" in s) ? String(s.detail_raster * 100) : "";
  if ("unfold" in s) $("#unfold").checked = !!s.unfold;
}

/* ---- profiles ---- */
function renderProfiles(profiles){
  const sel = $("#profsel");
  const current = sel.value;
  sel.innerHTML = '<option value="">— gespeichertes Profil laden —</option>';
  Object.keys(profiles).sort().forEach(name => {
    const o = document.createElement("option");
    o.value = name; o.textContent = name;
    sel.appendChild(o);
  });
  if (profiles[current]) sel.value = current;
  state.profiles = profiles;
}
async function loadProfiles(){
  const res = await fetch("/api/profiles");
  renderProfiles((await res.json()).profiles || {});
}
$("#profsel").onchange = () => {
  const name = $("#profsel").value;
  if (name && state.profiles[name]) {
    applySettings(state.profiles[name]);
    $("#profname").value = name;
    setStatus("Profil „" + name + "“ geladen", "ok");
  }
};
$("#profsave").onclick = async () => {
  const name = $("#profname").value.trim();
  if (!name) { alert("Bitte einen Profilnamen eingeben."); return; }
  const res = await fetch("/api/profiles", {
    method: "POST",
    body: JSON.stringify({name, settings: gatherOptions()}),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  renderProfiles(data.profiles);
  $("#profsel").value = name;
  setStatus("Profil „" + name + "“ gespeichert", "ok");
};
$("#profdel").onclick = async () => {
  const name = $("#profsel").value;
  if (!name) return;
  const res = await fetch("/api/profiles/delete", {
    method: "POST", body: JSON.stringify({name}),
  });
  renderProfiles((await res.json()).profiles || {});
  setStatus("Profil gelöscht");
};
loadProfiles();

/* run */
$("#run").onclick = async () => {
  if (!state.files.length) { alert("Bitte zuerst eine Punktwolke wählen."); return; }
  let options = {};
  if (state.mode === "reconstruct") {
    options = gatherOptions();
  } else if (state.mode === "sheetmetal") {
    options.unfold = $("#unfold").checked;
  }
  const res = await fetch("/api/run", {
    method: "POST",
    body: JSON.stringify({mode: state.mode, files: state.files.map(f => f.path), options}),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  state.job = data.job;
  $("#run").disabled = true;
  $("#cancel").style.display = "";
  $("#cancel").disabled = false;
  $("#log").textContent = "";
  $("#dlcard").style.display = "none";
  $("#exportrow").style.display = "none";
  $("#exporthint").textContent = "";
  $("#progwrap").style.display = "";
  $("#progbar").style.width = "0%";
  $("#progtext").textContent = "";
  state.progT0 = null;
  setStatus("Berechnung läuft …", "run");
  state.timer = setInterval(poll, 800);
};

$("#cancel").onclick = async () => {
  if (!state.job) return;
  $("#cancel").disabled = true;
  setStatus("wird abgebrochen …", "run");
  await fetch("/api/cancel", {
    method: "POST", body: JSON.stringify({job: state.job}),
  });
};

async function poll(){
  const res = await fetch("/api/status?job=" + state.job);
  const s = await res.json();
  const log = $("#log");
  log.textContent = s.log.join("\n") || "…";
  log.scrollTop = log.scrollHeight;
  if (s.progress != null) {
    $("#progwrap").style.display = "";
    $("#progbar").style.width = (s.progress * 100).toFixed(1) + "%";
    if (!state.progT0 && s.progress > 0) state.progT0 = Date.now() / 1000;
    let eta = "";
    if (state.progT0 && s.progress > 0.05 && s.progress < 1) {
      const el = Date.now() / 1000 - state.progT0;
      const rem = el * (1 - s.progress) / s.progress;
      eta = " · ≈ " + fmtDur(rem) + " verbleibend";
    }
    $("#progtext").textContent =
      Math.round(s.progress * 100) + "% — " + (s.phase || "") + eta;
  }
  if (s.state === "running") return;

  clearInterval(state.timer);
  $("#run").disabled = false;
  $("#cancel").style.display = "none";
  if (s.state === "done") { $("#progbar").style.width = "100%"; $("#progtext").textContent = "100% — fertig"; }
  else { $("#progwrap").style.display = "none"; }
  if (s.state === "cancelled") {
    setStatus("abgebrochen", "err");
  } else if (s.state === "error") {
    setStatus("Fehler", "err");
  } else {
    setStatus("fertig", "ok");
    if (s.has_viewer) {
      $("#viewer").src = "/api/view?job=" + state.job + "&t=" + Date.now();
      $("#viewer").style.display = "block";
      $("#placeholder").style.display = "none";
    }
  }
  /* downloads + on-demand exports */
  if (s.outputs && s.outputs.length) {
    renderDownloads(s.outputs);
    $("#exportrow").style.display = s.can_export ? "" : "none";
    if (!s.can_export) $("#exporthint").textContent = "";
  }
  /* report table */
  if (s.summary) {
    const tb = $("#report tbody"); tb.innerHTML = "";
    Object.entries(s.summary).forEach(([k, v]) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${k}</td><td>${v}</td>`;
      tb.appendChild(tr);
    });
  }
  /* auto-tuning winner → offer one-click profile save */
  if (s.state === "done" && s.auto_winner && s.auto_winner.advanced) {
    const tb = $("#report tbody");
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 2;
    const btn = document.createElement("button");
    btn.className = "ghost";
    btn.textContent = "★ Gewinner-Einstellungen als Profil speichern (" +
      s.auto_winner.candidate + ")";
    btn.title = "Übernimmt die vom Auto-Tuning ermittelten Parameter in die " +
      "Oberfläche und speichert sie dauerhaft als eigenes Profil.";
    btn.onclick = async () => {
      const opts = gatherOptions();
      opts.preset = s.auto_winner.preset || "building";
      opts.advanced = s.auto_winner.advanced;
      applySettings(opts);
      const name = "Auto (" + s.auto_winner.candidate + ")";
      const res = await fetch("/api/profiles", {
        method: "POST", body: JSON.stringify({name, settings: opts}),
      });
      const data = await res.json();
      if (data.error) { alert(data.error); return; }
      renderProfiles(data.profiles);
      $("#profsel").value = name;
      $("#profname").value = name;
      setStatus("Profil „" + name + "“ gespeichert", "ok");
    };
    td.appendChild(btn);
    tr.appendChild(td);
    tb.appendChild(tr);
  }
}

/* source select: show the selected profile's explanation as its own tooltip */
const srcSel = $("#source");
function syncSourceTitle(){
  const opt = srcSel.options[srcSel.selectedIndex];
  srcSel.title = opt ? opt.title : "";
}
srcSel.onchange = syncSourceTitle;
syncSourceTitle();

function renderDownloads(outputs){
  const dl = $("#downloads"); dl.innerHTML = "";
  (outputs || []).forEach(o => {
    const a = document.createElement("a");
    a.className = "dl";
    a.href = "/api/output?job=" + state.job + "&name=" + encodeURIComponent(o.name);
    a.innerHTML = `⬇ ${o.name} <span class="sz">${fmtSize(o.size)}</span>`;
    dl.appendChild(a);
  });
  $("#dlcard").style.display = "";
}

function fmtDur(sec){
  sec = Math.max(0, Math.round(sec));
  const m = Math.floor(sec / 60), s2 = sec % 60;
  if (m >= 60) { const h = Math.floor(m / 60); return h + " h " + (m % 60) + " min"; }
  return m > 0 ? m + " min " + s2 + " s" : s2 + " s";
}

/* CPU/RAM/GPU/VRAM live stats (server-side cached, cheap) */
async function pollStats(){
  try {
    const st = await (await fetch("/api/sysstats")).json();
    const parts = [];
    if (st.cpu != null) parts.push("CPU " + Math.round(st.cpu) + "%");
    if (st.ram_used != null) parts.push("RAM " + st.ram_used + "/" + st.ram_total + " GB");
    if (st.gpu != null) parts.push("GPU " + Math.round(st.gpu) + "%");
    if (st.vram_used != null) parts.push("VRAM " + st.vram_used + "/" + st.vram_total + " GB");
    $("#sysstats").textContent = parts.join(" · ");
  } catch (e) { /* Server weg — still bleiben */ }
}
setInterval(pollStats, 2500);
pollStats();

/* preloaded files (drag & drop onto the exe) + GPU badge */
function applyMeta(m, first){
  if (first) {
    (m.initial_files || []).forEach(f => state.files.push(f));
    renderFiles();
  }
  const badge = $("#gpubadge");
  if (!m.gpu_geprueft) {
    setTimeout(() => fetch("/api/meta").then(r => r.json()).then(x => applyMeta(x, false)), 2000);
    return;
  }
  if (m.gpu) {
    badge.textContent = "GPU: " + m.gpu + " ✓";
    badge.style.color = "#7fd0a8";
    badge.title = "CUDA-Beschleunigung aktiv";
  } else if (m.gpu_fehler) {
    badge.textContent = "GPU: CPU-Modus ⚠";
    badge.style.color = "#ffd479";
    badge.title = "CUDA nicht nutzbar: " + m.gpu_fehler +
      " — Details in gpu_diagnose.txt (wird beim Lauf erzeugt)";
  } else {
    badge.textContent = "CPU-Version";
    badge.title = "Diese Version rechnet ohne Grafikkarte. " +
      "Für NVIDIA-Karten: scantobim-windows-x64-gpu.zip";
  }
}
fetch("/api/meta").then(r => r.json()).then(m => applyMeta(m, true));
</script>
</body>
</html>
"""
