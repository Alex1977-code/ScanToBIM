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


def write_html_viewer(mesh: Mesh, path: str | Path, title: str = "ScanToBIM Modell") -> Path:
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

    center = (positions.min(axis=0) + positions.max(axis=0)) / 2.0 if len(positions) else np.zeros(3)
    radius = float(np.linalg.norm(positions - center, axis=1).max()) if len(positions) else 1.0

    def b64(arr: np.ndarray) -> str:
        return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")

    n_surfaces = len(np.unique(mesh.face_groups)) if mesh.face_groups is not None else 1
    meta = {
        "vertices": int(len(positions)),
        "triangles": int(len(indices)),
        "surfaces": int(n_surfaces),
        "center": [float(c) for c in center],
        "radius": radius if radius > 0 else 1.0,
    }

    meta["textured"] = bool(textured)
    html = (
        _TEMPLATE.replace("__TITLE__", title)
        .replace("__META__", json.dumps(meta))
        .replace("__POSITIONS__", b64(positions))
        .replace("__NORMALS__", b64(normals))
        .replace("__COLORS__", b64(colors))
        .replace("__UVS__", b64(uvs))
        .replace("__TEXTURE_URI__", texture_uri)
        .replace("__INDICES__", b64(indices))
        .replace("__CREASES__", b64(creases))
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
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="hud"><h1>__TITLE__</h1><div id="stats"></div></div>
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
const TEXTURE_URI = "__TEXTURE_URI__";

document.getElementById("stats").textContent =
  META.vertices + " Vertices · " + META.triangles + " Dreiecke · " + META.surfaces + " Flächen";

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

function buffer(target, data) {
  const b = gl.createBuffer(); gl.bindBuffer(target, b); gl.bufferData(target, data, gl.STATIC_DRAW); return b;
}
const posBuf = buffer(gl.ARRAY_BUFFER, positions);
const nrmBuf = buffer(gl.ARRAY_BUFFER, normals);
const colBuf = buffer(gl.ARRAY_BUFFER, colors);
const uvBuf = buffer(gl.ARRAY_BUFFER, uvs);
const idxBuf = buffer(gl.ELEMENT_ARRAY_BUFFER, indices);
const lineBuf = buffer(gl.ELEMENT_ARRAY_BUFFER, creases);

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
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
  gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_INT, 0);

  if (creases.length) {
    gl.uniform1f(uFlat, 1.0); gl.uniform3f(uLine, 0.05, 0.05, 0.06); gl.uniform1f(uBias, 0.0012);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lineBuf);
    gl.drawElements(gl.LINES, creases.length, gl.UNSIGNED_INT, 0);
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
