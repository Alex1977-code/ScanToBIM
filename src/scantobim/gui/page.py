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
<title>ScanToBIM — Punktwolke zu 3D-Modell</title>
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
</style>
</head>
<body>
<header>
  <div class="logo"></div>
  <h1>ScanToBIM</h1><span class="ver">v__VERSION__</span>
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
      </div>
    </div>

    <div class="card" id="optcard">
      <h2>3 · Optionen</h2>
      <div id="opts-reconstruct">
        <label class="sel-label">Szene</label>
        <select id="preset">
          <option value="building">Gebäude außen</option>
          <option value="indoor">Innenraum</option>
          <option value="object">Einzelobjekt / Bauteil</option>
          <option value="detail">Detailgetreu (mehr Flächen + Stützen/Rohre)</option>
          <option value="fast">Schnell (Vorschau)</option>
        </select>
        <div style="margin-top:.6rem">
          <label class="opt"><input type="checkbox" id="watertight"> Wasserdichtes Volumenmodell (schließt Scanschatten)</label>
          <label class="opt"><input type="checkbox" id="texture" checked> Fototextur aus Punktfarben</label>
          <label class="opt"><input type="checkbox" id="align"> Achsen ausrichten, Boden auf Z=0</label>
          <label class="opt"><input type="checkbox" id="register"> Mehrere Scans automatisch registrieren (ICP)</label>
          <label class="opt"><input type="checkbox" id="deviation"> Soll-Ist-Abweichungsanalyse (QS-Heatmap + Statistik)</label>
          <label class="opt"><input type="checkbox" id="views"> Orthofoto-Ansichten N/O/S/W + Draufsicht (maßstabsgetreu)</label>
        </div>
        <label class="sel-label">Zusätzliche Exportformate</label>
        <div class="fmt" id="formats">
          <label data-f="step">STEP (CAD)</label>
          <label data-f="ifc">IFC (BIM)</label>
          <label data-f="glb">GLB</label>
          <label data-f="obj">OBJ</label>
          <label data-f="dxf">DXF-Grundriss</label>
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

    <button class="primary" id="run">Modell erstellen</button>
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
const state = { files: [], mode: "reconstruct", job: null, timer: null };

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
      (state.mode === "analyze" || state.mode === "bridge") ? "" : "none";
  };
});
/* format chips */
document.querySelectorAll("#formats label").forEach(el => {
  el.onclick = () => el.classList.toggle("on");
});

/* run */
$("#run").onclick = async () => {
  if (!state.files.length) { alert("Bitte zuerst eine Punktwolke wählen."); return; }
  const options = {};
  if (state.mode === "reconstruct") {
    options.preset = $("#preset").value;
    options.watertight = $("#watertight").checked;
    options.texture = $("#texture").checked;
    options.align = $("#align").checked;
    options.register = $("#register").checked && state.files.length > 1;
    options.deviation = $("#deviation").checked;
    options.views = $("#views").checked;
    options.formats = [...document.querySelectorAll("#formats label.on")]
      .map(el => el.dataset.f);
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
  $("#log").textContent = "";
  $("#dlcard").style.display = "none";
  setStatus("Berechnung läuft …", "run");
  state.timer = setInterval(poll, 800);
};

async function poll(){
  const res = await fetch("/api/status?job=" + state.job);
  const s = await res.json();
  const log = $("#log");
  log.textContent = s.log.join("\n") || "…";
  log.scrollTop = log.scrollHeight;
  if (s.state === "running") return;

  clearInterval(state.timer);
  $("#run").disabled = false;
  if (s.state === "error") {
    setStatus("Fehler", "err");
  } else {
    setStatus("fertig", "ok");
    if (s.has_viewer) {
      $("#viewer").src = "/api/view?job=" + state.job + "&t=" + Date.now();
      $("#viewer").style.display = "block";
      $("#placeholder").style.display = "none";
    }
  }
  /* downloads */
  if (s.outputs && s.outputs.length) {
    const dl = $("#downloads"); dl.innerHTML = "";
    s.outputs.forEach(o => {
      const a = document.createElement("a");
      a.className = "dl";
      a.href = "/api/output?job=" + state.job + "&name=" + encodeURIComponent(o.name);
      a.innerHTML = `⬇ ${o.name} <span class="sz">${fmtSize(o.size)}</span>`;
      dl.appendChild(a);
    });
    $("#dlcard").style.display = "";
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
}

/* preloaded files (drag & drop onto the exe) */
fetch("/api/meta").then(r => r.json()).then(m => {
  (m.initial_files || []).forEach(f => state.files.push(f));
  renderFiles();
});
</script>
</body>
</html>
"""
