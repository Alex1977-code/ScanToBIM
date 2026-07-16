"""Self-contained interactive 3D viewer (single HTML file, zero dependencies).

``write_html_viewer`` embeds the reconstructed mesh into one HTML file with a
minimal WebGL renderer: orbit/pan/zoom controls, per-surface colors, and the
model's *crease edges* (surface borders and boundaries) drawn as crisp black
lines — exactly the clean edges the pipeline produces. The file works offline
in any modern browser, so results can be shared with clients by simply
sending one file. No CDN, no installation, no data leaves the machine.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from scantobim.core.mesh import Mesh


def extract_crease_edges(mesh: Mesh, dihedral_deg: float = 25.0) -> np.ndarray:
    """Edge index pairs worth drawing: boundaries, surface borders, creases."""
    faces = mesh.faces
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edge_face = np.tile(np.arange(len(faces)), 3)
    edges_sorted = np.sort(edges, axis=1)
    uniq, inverse, counts = np.unique(
        edges_sorted, axis=0, return_inverse=True, return_counts=True
    )

    keep = np.zeros(len(uniq), dtype=bool)
    keep[counts == 1] = True  # open boundaries

    face_normals = mesh.face_normals()
    groups = mesh.face_groups
    # For manifold edges check group change / dihedral angle.
    order = np.argsort(inverse, kind="stable")
    sorted_inv = inverse[order]
    sorted_face = edge_face[order]
    start = 0
    cos_th = np.cos(np.deg2rad(dihedral_deg))
    while start < len(sorted_inv):
        end = start
        while end < len(sorted_inv) and sorted_inv[end] == sorted_inv[start]:
            end += 1
        eid = sorted_inv[start]
        if end - start == 2 and not keep[eid]:
            f1, f2 = sorted_face[start], sorted_face[end - 1]
            if groups is not None and groups[f1] != groups[f2]:
                keep[eid] = True
            elif float(face_normals[f1] @ face_normals[f2]) < cos_th:
                keep[eid] = True
        elif end - start > 2:
            keep[eid] = True  # non-manifold — always show
        start = end
    return uniq[keep]


def write_html_viewer(
    mesh: Mesh,
    path: str | Path,
    title: str = "ScanToBIM Modell",
    points=None,
    freeform: Mesh | None = None,
    freeform_label: str = "Freiform-Restgeometrie",
    max_layer_points: int = 800_000,
) -> Path:
    """Write the standalone viewer.

    ``points`` (a PointCloud, e.g. the reconstruction residual) is embedded
    as a toggleable colored point layer; ``freeform`` (the hybrid free-form
    skin of the residual) as a toggleable second mesh layer with vertex
    colors — so railings, steel members and other unmodelled structure stay
    visible next to the parametric surfaces."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    positions = mesh.vertices.astype(np.float32)
    normals = mesh.vertex_normals().astype(np.float32)
    if mesh.vertex_colors is not None:
        colors = mesh.vertex_colors.astype(np.uint8)
    else:
        colors = np.full((len(positions), 3), 200, dtype=np.uint8)
    indices = mesh.faces.astype(np.uint32)
    creases = extract_crease_edges(mesh).astype(np.uint32)

    textured = mesh.texture is not None and mesh.uvs is not None
    if textured:
        uvs = mesh.uvs.astype(np.float32)
        from scantobim.io.png import encode_png

        texture_uri = "data:image/png;base64," + base64.b64encode(
            encode_png(mesh.texture)
        ).decode("ascii")
    else:
        uvs = np.zeros((len(positions), 2), dtype=np.float32)
        texture_uri = ""

    # Optional residual scan points as a toggleable layer.
    pt_positions = np.zeros((0, 3), dtype=np.float32)
    pt_colors = np.zeros((0, 3), dtype=np.uint8)
    if points is not None and len(points):
        pt_positions = points.points
        if len(pt_positions) > max_layer_points:
            rng = np.random.default_rng(0)
            keep = rng.choice(len(pt_positions), max_layer_points, replace=False)
            keep.sort()
            pt_positions = pt_positions[keep]
            pt_colors_src = None if points.colors is None else points.colors[keep]
        else:
            pt_colors_src = points.colors
        pt_positions = pt_positions.astype(np.float32)
        pt_colors = (
            pt_colors_src.astype(np.uint8)
            if pt_colors_src is not None
            else np.full((len(pt_positions), 3), 150, dtype=np.uint8)
        )

    # Optional free-form mesh layer (hybrid model).
    if freeform is not None and len(freeform.faces):
        ff_positions = freeform.vertices.astype(np.float32)
        ff_normals = freeform.vertex_normals().astype(np.float32)
        ff_colors = (
            freeform.vertex_colors.astype(np.uint8)
            if freeform.vertex_colors is not None
            else np.full((len(ff_positions), 3), 150, dtype=np.uint8)
        )
        ff_indices = freeform.faces.astype(np.uint32)
    else:
        ff_positions = np.zeros((0, 3), dtype=np.float32)
        ff_normals = np.zeros((0, 3), dtype=np.float32)
        ff_colors = np.zeros((0, 3), dtype=np.uint8)
        ff_indices = np.zeros((0, 3), dtype=np.uint32)

    all_pos = positions if not len(ff_positions) else np.vstack([positions, ff_positions])
    center = (all_pos.min(axis=0) + all_pos.max(axis=0)) / 2.0 if len(all_pos) else np.zeros(3)
    radius = float(np.linalg.norm(all_pos - center, axis=1).max()) if len(all_pos) else 1.0

    def b64(arr: np.ndarray) -> str:
        return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")

    n_surfaces = len(np.unique(mesh.face_groups)) if mesh.face_groups is not None else 1
    meta = {
        "vertices": int(len(positions)),
        "triangles": int(len(indices)),
        "surfaces": int(n_surfaces),
        "center": [float(c) for c in center],
        "radius": radius if radius > 0 else 1.0,
        "points": int(len(pt_positions)),
        "ff_indices": int(ff_indices.size),
        "ff_triangles": int(len(ff_indices)),
    }

    meta["textured"] = bool(textured)
    html = (
        _TEMPLATE.replace("__TITLE__", title)
        .replace("__FF_LABEL__", freeform_label)
        .replace("__META__", json.dumps(meta))
        .replace("__POSITIONS__", b64(positions))
        .replace("__NORMALS__", b64(normals))
        .replace("__COLORS__", b64(colors))
        .replace("__UVS__", b64(uvs))
        .replace("__TEXTURE_URI__", texture_uri)
        .replace("__INDICES__", b64(indices))
        .replace("__CREASES__", b64(creases))
        .replace("__PT_POSITIONS__", b64(pt_positions))
        .replace("__PT_COLORS__", b64(pt_colors))
        .replace("__FF_POSITIONS__", b64(ff_positions))
        .replace("__FF_NORMALS__", b64(ff_normals))
        .replace("__FF_COLORS__", b64(ff_colors))
        .replace("__FF_INDICES__", b64(ff_indices))
    )
    path.write_text(html, encoding="utf-8")
    return path


_TEMPLATE = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; background: #1a1d21; font-family: system-ui, sans-serif; }
  canvas { display: block; width: 100%; height: 100%; touch-action: none; }
  #hud { position: fixed; left: 12px; top: 10px; color: #dde3ea; pointer-events: none; }
  #hud h1 { font-size: 15px; margin: 0 0 4px; font-weight: 600; }
  #hud div { font-size: 12px; opacity: .75; }
  #help { position: fixed; right: 12px; bottom: 10px; color: #99a3ad; font-size: 11px; pointer-events: none; }
  #layers { position: fixed; right: 12px; top: 10px; color: #e8edf4; font-size: 13px;
            background: rgba(24,28,35,.94); border: 1px solid #4a5568; border-radius: 10px;
            padding: 10px 14px; user-select: none; box-shadow: 0 4px 18px rgba(0,0,0,.45);
            min-width: 220px; }
  #layers .lt { font-size: 10px; letter-spacing: .14em; color: #93a0b4;
                margin-bottom: 6px; font-weight: 700; }
  #layers label { display: flex; gap: 8px; align-items: center; cursor: pointer;
                  padding: 3px 0; }
  #layers input { width: 15px; height: 15px; accent-color: #5c9bff; }
  #layers label[hidden] { display: none; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="hud"><h1>__TITLE__</h1><div id="stats"></div></div>
<div id="layers" hidden>
  <div class="lt">EBENEN EIN/AUS</div>
  <label id="baseRow" hidden><input type="checkbox" id="baseToggle" checked>
    Strukturmodell (Fl&auml;chen &amp; Kanten)</label>
  <label id="ffRow" hidden><input type="checkbox" id="ffToggle" checked>
    __FF_LABEL__ (<span id="ffCount"></span> Dreiecke)</label>
  <label id="ptsRow" hidden><input type="checkbox" id="ptsToggle" checked>
    Scan-Restpunkte (<span id="ptsCount"></span>)</label>
</div>
<div id="help">Ziehen: drehen &nbsp;•&nbsp; Shift/Rechts: verschieben &nbsp;•&nbsp; Rad/Pinch: zoomen</div>
<script>
"use strict";
const META = __META__;
function decode(b64, T) {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new T(bytes.buffer);
}
const positions = decode("__POSITIONS__", Float32Array);
const normals   = decode("__NORMALS__", Float32Array);
const colors    = decode("__COLORS__", Uint8Array);
const uvs       = decode("__UVS__", Float32Array);
const indices   = decode("__INDICES__", Uint32Array);
const creases   = decode("__CREASES__", Uint32Array);
const ptPositions = decode("__PT_POSITIONS__", Float32Array);
const ptColors    = decode("__PT_COLORS__", Uint8Array);
const ffPositions = decode("__FF_POSITIONS__", Float32Array);
const ffNormals   = decode("__FF_NORMALS__", Float32Array);
const ffColors    = decode("__FF_COLORS__", Uint8Array);
const ffIndices   = decode("__FF_INDICES__", Uint32Array);
const TEXTURE_URI = "__TEXTURE_URI__";

document.getElementById("stats").textContent =
  META.vertices + " Vertices · " + META.triangles + " Dreiecke · " + META.surfaces + " Flächen";
let showPoints = META.points > 0;
let showFF = META.ff_indices > 0;
let showBase = true;
if (META.points > 0 || META.ff_indices > 0) {
  document.getElementById("layers").hidden = false;
  if (META.triangles > 0) {
    document.getElementById("baseRow").hidden = false;
    document.getElementById("baseToggle").addEventListener("change", e => {
      showBase = e.target.checked;
    });
  }
}
if (META.points > 0) {
  document.getElementById("ptsRow").hidden = false;
  document.getElementById("ptsCount").textContent = META.points.toLocaleString("de-DE");
  document.getElementById("ptsToggle").addEventListener("change", e => {
    showPoints = e.target.checked;
  });
}
if (META.ff_indices > 0) {
  document.getElementById("ffRow").hidden = false;
  document.getElementById("ffCount").textContent = META.ff_triangles.toLocaleString("de-DE");
  document.getElementById("ffToggle").addEventListener("change", e => {
    showFF = e.target.checked;
  });
}

const canvas = document.getElementById("c");
const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
if (!gl) { document.body.innerHTML = "WebGL nicht verfügbar"; throw new Error("no webgl"); }
if (!(gl instanceof (window.WebGL2RenderingContext || Object))) gl.getExtension("OES_element_index_uint");

function shader(type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
const VS = "attribute vec3 aPos; attribute vec3 aNrm; attribute vec3 aCol; attribute vec2 aUV;" +
  "uniform mat4 uMVP; uniform float uBias; varying vec3 vN; varying vec3 vC; varying vec2 vUV;" +
  "void main(){ vN=aNrm; vC=aCol; vUV=aUV; gl_Position=uMVP*vec4(aPos,1.0); gl_Position.z-=uBias*gl_Position.w; }";
const FS = "precision mediump float; varying vec3 vN; varying vec3 vC; varying vec2 vUV;" +
  "uniform vec3 uEye; uniform float uFlat; uniform vec3 uLine;" +
  "uniform float uTextured; uniform sampler2D uTex;" +
  "void main(){ vec3 n=normalize(vN);" +
  " float d=abs(dot(n,normalize(uEye)));" +
  " vec3 base=mix(vC, texture2D(uTex, vUV).rgb, uTextured);" +
  " vec3 lit=base*(0.45+0.55*d);" +
  " gl_FragColor=vec4(mix(lit,uLine,uFlat),1.0); }";
const prog = gl.createProgram();
gl.attachShader(prog, shader(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog);
gl.useProgram(prog);

// Minimal second program for the residual scan-point layer.
const PVS = "attribute vec3 aPos; attribute vec3 aCol; uniform mat4 uMVP;" +
  "varying vec3 vC; void main(){ vC=aCol; gl_Position=uMVP*vec4(aPos,1.0); gl_PointSize=2.0; }";
const PFS = "precision mediump float; varying vec3 vC;" +
  "void main(){ gl_FragColor=vec4(vC,1.0); }";
const pprog = gl.createProgram();
gl.attachShader(pprog, shader(gl.VERTEX_SHADER, PVS));
gl.attachShader(pprog, shader(gl.FRAGMENT_SHADER, PFS));
gl.linkProgram(pprog);

function buffer(target, data) {
  const b = gl.createBuffer(); gl.bindBuffer(target, b); gl.bufferData(target, data, gl.STATIC_DRAW); return b;
}
const posBuf = buffer(gl.ARRAY_BUFFER, positions);
const nrmBuf = buffer(gl.ARRAY_BUFFER, normals);
const colBuf = buffer(gl.ARRAY_BUFFER, colors);
const uvBuf = buffer(gl.ARRAY_BUFFER, uvs);
const idxBuf = buffer(gl.ELEMENT_ARRAY_BUFFER, indices);
const lineBuf = buffer(gl.ELEMENT_ARRAY_BUFFER, creases);
const ptPosBuf = META.points ? buffer(gl.ARRAY_BUFFER, ptPositions) : null;
const ptColBuf = META.points ? buffer(gl.ARRAY_BUFFER, ptColors) : null;
const ffPosBuf = META.ff_indices ? buffer(gl.ARRAY_BUFFER, ffPositions) : null;
const ffNrmBuf = META.ff_indices ? buffer(gl.ARRAY_BUFFER, ffNormals) : null;
const ffColBuf = META.ff_indices ? buffer(gl.ARRAY_BUFFER, ffColors) : null;
const ffIdxBuf = META.ff_indices ? buffer(gl.ELEMENT_ARRAY_BUFFER, ffIndices) : null;
const pAPos = gl.getAttribLocation(pprog, "aPos");
const pACol = gl.getAttribLocation(pprog, "aCol");
const pUMVP = gl.getUniformLocation(pprog, "uMVP");

const aPos = gl.getAttribLocation(prog, "aPos");
const aNrm = gl.getAttribLocation(prog, "aNrm");
const aCol = gl.getAttribLocation(prog, "aCol");
const aUV = gl.getAttribLocation(prog, "aUV");
const uMVP = gl.getUniformLocation(prog, "uMVP");
const uEye = gl.getUniformLocation(prog, "uEye");
const uFlat = gl.getUniformLocation(prog, "uFlat");
const uLine = gl.getUniformLocation(prog, "uLine");
const uBias = gl.getUniformLocation(prog, "uBias");
const uTextured = gl.getUniformLocation(prog, "uTextured");
const uTex = gl.getUniformLocation(prog, "uTex");

let texReady = false;
if (META.textured && TEXTURE_URI) {
  const tex = gl.createTexture();
  const img = new Image();
  img.onload = () => {
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    texReady = true;
  };
  img.src = TEXTURE_URI;
}

let theta = -1.0, phi = 1.1, dist = META.radius * 2.6;
const target = META.center.slice();

function mat4mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[k*4+r]*b[c*4+k];
    o[c*4+r] = s;
  }
  return o;
}
function perspective(fov, aspect, near, far) {
  const f = 1/Math.tan(fov/2), nf = 1/(near-far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}
function lookAt(eye, at, up) {
  const z = norm3(sub3(eye, at)), x = norm3(cross3(up, z)), y = cross3(z, x);
  return new Float32Array([
    x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
    -dot3(x,eye), -dot3(y,eye), -dot3(z,eye), 1]);
}
function sub3(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
function cross3(a,b){return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
function dot3(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function norm3(a){const l=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/l,a[1]/l,a[2]/l];}

function eyePos() {
  const cp = Math.cos(phi), sp = Math.sin(phi);
  return [
    target[0] + dist*cp*Math.cos(theta),
    target[1] + dist*cp*Math.sin(theta),
    target[2] + dist*sp];
}

function draw() {
  const w = canvas.clientWidth * devicePixelRatio, h = canvas.clientHeight * devicePixelRatio;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.viewport(0, 0, w, h);
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0.10, 0.11, 0.13, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const eye = eyePos();
  const mvp = mat4mul(
    perspective(0.9, w/h, META.radius*0.01, META.radius*40),
    lookAt(eye, target, [0,0,1]));
  gl.useProgram(prog);
  gl.uniformMatrix4fv(uMVP, false, mvp);
  const dir = sub3(eye, target);
  gl.uniform3f(uEye, dir[0], dir[1], dir[2]);

  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf);
  gl.enableVertexAttribArray(aNrm); gl.vertexAttribPointer(aNrm, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
  gl.enableVertexAttribArray(aCol); gl.vertexAttribPointer(aCol, 3, gl.UNSIGNED_BYTE, true, 0, 0);
  if (aUV >= 0) {
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
    gl.enableVertexAttribArray(aUV); gl.vertexAttribPointer(aUV, 2, gl.FLOAT, false, 0, 0);
  }
  gl.uniform1f(uTextured, texReady ? 1.0 : 0.0);
  gl.uniform1i(uTex, 0);

  gl.uniform1f(uFlat, 0.0); gl.uniform1f(uBias, 0.0);
  if (showBase) {
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
    gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_INT, 0);
  }

  if (showBase && creases.length) {
    gl.uniform1f(uFlat, 1.0); gl.uniform3f(uLine, 0.05, 0.05, 0.06); gl.uniform1f(uBias, 0.0012);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lineBuf);
    gl.drawElements(gl.LINES, creases.length, gl.UNSIGNED_INT, 0);
  }

  if (META.ff_indices && showFF) {
    // Free-form layer: same lighting, vertex colors, never textured.
    gl.uniform1f(uFlat, 0.0); gl.uniform1f(uBias, 0.0); gl.uniform1f(uTextured, 0.0);
    if (aUV >= 0) gl.disableVertexAttribArray(aUV);
    gl.bindBuffer(gl.ARRAY_BUFFER, ffPosBuf);
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, ffNrmBuf);
    gl.vertexAttribPointer(aNrm, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, ffColBuf);
    gl.vertexAttribPointer(aCol, 3, gl.UNSIGNED_BYTE, true, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ffIdxBuf);
    gl.drawElements(gl.TRIANGLES, META.ff_indices, gl.UNSIGNED_INT, 0);
  }

  if (META.points && showPoints) {
    gl.useProgram(pprog);
    gl.uniformMatrix4fv(pUMVP, false, mvp);
    // Disable every main-program array before switching layouts — enabled
    // arrays are range-validated even when the program ignores them.
    gl.disableVertexAttribArray(aPos);
    gl.disableVertexAttribArray(aNrm);
    gl.disableVertexAttribArray(aCol);
    if (aUV >= 0) gl.disableVertexAttribArray(aUV);
    gl.bindBuffer(gl.ARRAY_BUFFER, ptPosBuf);
    gl.enableVertexAttribArray(pAPos); gl.vertexAttribPointer(pAPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, ptColBuf);
    gl.enableVertexAttribArray(pACol); gl.vertexAttribPointer(pACol, 3, gl.UNSIGNED_BYTE, true, 0, 0);
    gl.drawArrays(gl.POINTS, 0, META.points);
    gl.disableVertexAttribArray(pAPos);
    gl.disableVertexAttribArray(pACol);
  }
  requestAnimationFrame(draw);
}

const pointers = new Map();
let pinchDist = 0;
canvas.addEventListener("pointerdown", e => {
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 });
  canvas.setPointerCapture(e.pointerId);
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
  }
});
canvas.addEventListener("pointermove", e => {
  const p = pointers.get(e.pointerId);
  if (!p) return;
  const dx = e.clientX - p.x, dy = e.clientY - p.y;
  p.x = e.clientX; p.y = e.clientY;
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    const d = Math.hypot(a.x - b.x, a.y - b.y);
    if (pinchDist > 0) {
      dist *= pinchDist / d;
      dist = Math.min(META.radius * 30, Math.max(META.radius * 0.05, dist));
    }
    pinchDist = d;
    panBy(dx / 2, dy / 2);
  } else if (p.pan) {
    panBy(dx, dy);
  } else {
    theta -= dx * 0.006;
    phi = Math.min(1.55, Math.max(-1.55, phi + dy * 0.006));
  }
});
function panBy(dx, dy) {
  const eye = eyePos(), z = norm3(sub3(eye, target));
  const x = norm3(cross3([0,0,1], z)), y = cross3(z, x);
  const s = dist * 0.0016;
  for (let i = 0; i < 3; i++) target[i] += (-dx*x[i] + dy*y[i]) * s;
}
canvas.addEventListener("pointerup", e => { pointers.delete(e.pointerId); pinchDist = 0; });
canvas.addEventListener("pointercancel", e => { pointers.delete(e.pointerId); pinchDist = 0; });
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  dist *= Math.exp(e.deltaY * 0.0012);
  dist = Math.min(META.radius * 30, Math.max(META.radius * 0.05, dist));
}, { passive: false });
canvas.addEventListener("contextmenu", e => e.preventDefault());

requestAnimationFrame(draw);
</script>
</body>
</html>
"""
