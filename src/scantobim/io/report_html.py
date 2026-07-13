"""Druckfertiger Prüfbericht als eigenständige HTML-Datei.

One click → the document you hand to the client: all measured numbers,
the Soll-Ist statistics with a histogram, roof and opening schedules and
the true-to-scale orthographic views — print-ready A4 layout, no external
resources, opens in any browser and prints straight to PDF.
"""

from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path

from scantobim import __version__

_CLASS_LABELS = {
    "wall": "Wände", "floor": "Böden", "ceiling": "Decken",
    "slab": "Zwischenebenen", "sloped": "Schrägen", "roof": "Dachflächen",
    "terrain": "Gelände",
}


def _esc(value) -> str:
    return html.escape(str(value))


def _row(label: str, value) -> str:
    return f"<tr><td>{_esc(label)}</td><td>{_esc(value)}</td></tr>"


def render_report_html(
    report: dict,
    path: str | Path,
    title: str = "ScanToBIM Prüfbericht",
    views_dir: str | Path | None = None,
) -> Path:
    """Render the quality report (+ optional view images) to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []

    # ---- Kennzahlen -----------------------------------------------------
    q = report.get("quantities", {})
    rows = [
        _row("Eingangspunkte", f"{report.get('input_points', 0):,}".replace(",", " ")),
        _row("Flächen im Modell", report.get("planes", "–")),
        _row("Exakte Ecken", report.get("exact_corners", "–")),
    ]
    angles = report.get("plane_angles", {})
    if angles.get("pairs"):
        rows.append(_row(
            "Winkelpaare exakt 90° / parallel",
            f"{angles.get('exactly_orthogonal', 0)} / {angles.get('exactly_parallel', 0)}"
            f" (abweichend: {angles.get('other', 0)})",
        ))
    if "volume" in q:
        vol = f"{q['volume']:.3f} m³"
        if q.get("volume_sigma"):
            vol += f" ± {q['volume_sigma']:.3f}"
        rows.append(_row("Umbauter Raum (wasserdicht)", vol))
    if report.get("detail_surfaces"):
        rows.append(_row("Nachgeführte Detailflächen", report["detail_surfaces"]))
    if report.get("cylinders"):
        rows.append(_row("Stützen/Rohre (Zylinder)", len(report["cylinders"])))
    if report.get("trajectory_positions"):
        rows.append(_row("Scanner-Trajektorie", f"{report['trajectory_positions']} Positionen"))
    tex = report.get("texture")
    if tex:
        label = tex.get("source", "keine")
        if tex.get("coverage") is not None:
            label += f" ({tex['coverage'] * 100:.0f}% Abdeckung)"
        rows.append(_row("Textur", label))
    rows.append(_row("Restpunkte (unerklärt)", report.get("residual_points", "–")))
    rows.append(_row("Rechenzeit", f"{report.get('runtime_seconds', '–')} s"))
    sections.append(f"<h2>Modell-Kennzahlen</h2><table>{''.join(rows)}</table>")

    # ---- Bauteile -------------------------------------------------------
    by_class = q.get("surface_count_by_class", {})
    if by_class:
        rows = [
            _row(_CLASS_LABELS.get(cls, cls), n) for cls, n in sorted(by_class.items())
        ]
        if q.get("windows") is not None:
            rows.append(_row("Fenster", q.get("windows", 0)))
        if q.get("doors") is not None:
            rows.append(_row("Türen", q.get("doors", 0)))
        sections.append(f"<h2>Bauteile</h2><table>{''.join(rows)}</table>")

    storeys = report.get("storeys") or []
    if len(storeys) > 1:
        rows = [
            _row(f"Geschoss {s['index']}",
                 f"OKFF {s['elevation']:.2f} m, Höhe {s['height']:.2f} m")
            for s in storeys
        ]
        sections.append(f"<h2>Geschosse</h2><table>{''.join(rows)}</table>")

    # ---- Dach -----------------------------------------------------------
    roof = report.get("roof")
    if roof:
        head = (
            "<tr><th>Fläche</th><th>Neigung</th><th>Ausrichtung</th>"
            "<th>Fläche</th></tr>"
        )
        rows = [
            f"<tr><td>{_esc(f['name'])}</td><td>{f['slope_deg']:.1f}°</td>"
            f"<td>{f['azimuth_deg']:.0f}°</td><td>{f['area']:.2f} m²</td></tr>"
            for f in roof["faces"]
        ]
        summary = (
            f"<p>Firsthöhe <b>{roof['ridge_height']:.2f} m</b> · "
            f"Traufhöhe <b>{roof['eaves_height']:.2f} m</b> · "
            f"Dachfläche gesamt <b>{roof['total_area']:.2f} m²</b></p>"
        )
        sections.append(f"<h2>Dach</h2>{summary}<table>{head}{''.join(rows)}</table>")

    # ---- Öffnungen ------------------------------------------------------
    openings = report.get("opening_details") or []
    if openings:
        head = (
            "<tr><th>Bauteil</th><th>Typ</th><th>Breite</th><th>Höhe</th>"
            "<th>Brüstung</th></tr>"
        )
        rows = [
            f"<tr><td>{_esc(o['surface'])}</td>"
            f"<td>{'Tür' if o['type'] == 'tuer' else 'Fenster'}</td>"
            f"<td>{o['width'] * 100:.0f} cm</td><td>{o['height'] * 100:.0f} cm</td>"
            f"<td>{o['sill_height'] * 100:.0f} cm</td></tr>"
            for o in openings
        ]
        sections.append(f"<h2>Öffnungen</h2><table>{head}{''.join(rows)}</table>")

    # ---- Soll-Ist -------------------------------------------------------
    dev = report.get("deviation")
    if dev:
        rows = [_row("Geprüfte Punkte", f"{dev['points']:,}".replace(",", " "))]
        fid = dev.get("fidelity")
        if fid:
            rows += [
                _row("Modelltreue RMS (modellnahe Punkte)",
                     f"{fid['rms'] * 1000:.1f} mm"),
                _row("Modelltreue 95%-Quantil", f"{fid['p95'] * 1000:.1f} mm"),
                _row(
                    f"Modellnah innerhalb ±{dev['tolerance'] * 1000:.0f} mm",
                    f"{fid['within_tolerance'] * 100:.1f}%",
                ),
            ]
        if dev.get("coverage") is not None:
            rows.append(_row(
                f"Modellabdeckung des Scans (±{dev.get('coverage_band', 0.1) * 100:.0f} cm)",
                f"{dev['coverage'] * 100:.1f}%",
            ))
        rows += [
            _row("Gesamt-RMS (inkl. nicht modellierter Umgebung)",
                 f"{dev['rms'] * 1000:.1f} mm"),
            _row("Maximum", f"{dev['max'] * 1000:.1f} mm"),
            _row(
                f"Gesamt innerhalb ±{dev['tolerance'] * 1000:.0f} mm",
                f"{dev['within_tolerance'] * 100:.1f}%",
            ),
        ]
        hist = dev.get("histogram_mm", {})
        total = max(sum(hist.values()), 1)
        bars = "".join(
            f"<div class='bar'><span class='lbl'>{_esc(rng)} mm</span>"
            f"<span class='track'><span class='fill' "
            f"style='width:{100 * n / total:.1f}%'></span></span>"
            f"<span class='cnt'>{100 * n / total:.1f}%</span></div>"
            for rng, n in hist.items()
        )
        sections.append(
            "<h2>Soll-Ist-Abweichung (Scan → Modell)</h2>"
            f"<table>{''.join(rows)}</table>"
            f"<div class='hist'>{bars}</div>"
        )

    # ---- Auto-Tuning ----------------------------------------------------
    board = report.get("auto_tuning")
    if board:
        head = (
            "<tr><th>Kandidat</th><th>Score</th><th>P95</th>"
            "<th>unerklärt</th><th>Flächen</th></tr>"
        )
        rows = []
        for e in board:
            if "score" not in e:
                continue
            mark = " ✓" if e.get("selected") else ""
            rows.append(
                f"<tr><td>{_esc(e['candidate'])}{mark}</td><td>{e['score']:.3f}</td>"
                f"<td>{e['p95'] * 1000:.1f} mm</td>"
                f"<td>{e['unexplained'] * 100:.1f}%</td><td>{e['surfaces']}</td></tr>"
            )
        sections.append(
            "<h2>Auto-Tuning (selbstoptimierende Rekonstruktion)</h2>"
            f"<table>{head}{''.join(rows)}</table>"
        )

    # ---- Ansichten ------------------------------------------------------
    if views_dir is not None:
        views_dir = Path(views_dir)
        imgs = []
        for png in sorted(views_dir.glob("*.png")):
            b64 = base64.b64encode(png.read_bytes()).decode()
            caption = png.stem.replace("_", " ").title()
            imgs.append(
                f"<figure><img src='data:image/png;base64,{b64}' "
                f"alt='{_esc(caption)}'><figcaption>{_esc(caption)}"
                "</figcaption></figure>"
            )
        if imgs:
            sections.append(
                "<h2>Maßstabsgetreue Ansichten</h2>"
                f"<div class='views'>{''.join(imgs)}</div>"
            )

    stamp = datetime.now().strftime("%d.%m.%Y %H:%M")
    doc = f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8">
<title>{_esc(title)}</title>
<style>
@page {{ size: A4; margin: 18mm; }}
body {{
  font-family: system-ui, "Segoe UI", Arial, sans-serif; color: #16202b;
  max-width: 190mm; margin: 0 auto; padding: 24px; line-height: 1.45;
}}
header {{ border-bottom: 3px solid #2c5f9e; padding-bottom: 10px; margin-bottom: 20px; }}
h1 {{ font-size: 1.5rem; margin: 0 0 4px }}
.meta {{ color: #5c6b7a; font-size: .85rem }}
h2 {{
  font-size: 1.02rem; margin: 22px 0 8px; color: #2c5f9e;
  border-bottom: 1px solid #d5dde5; padding-bottom: 3px;
  page-break-after: avoid;
}}
table {{ border-collapse: collapse; width: 100%; font-size: .88rem }}
td, th {{ border: 1px solid #d5dde5; padding: 5px 9px; text-align: left }}
th {{ background: #eef3f8 }}
tr td:first-child {{ color: #46566a; width: 46% }}
.hist {{ margin-top: 10px; font-size: .78rem }}
.bar {{ display: flex; align-items: center; gap: 8px; margin: 2px 0 }}
.bar .lbl {{ width: 70px; text-align: right; color: #46566a }}
.bar .track {{ flex: 1; background: #eef1f5; border-radius: 3px; height: 12px; overflow: hidden; display:block }}
.bar .fill {{ display: block; height: 100%; background: #4a86c9 }}
.bar .cnt {{ width: 52px; color: #46566a }}
.views {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px }}
figure {{ margin: 0; page-break-inside: avoid }}
img {{ width: 100%; border: 1px solid #d5dde5 }}
figcaption {{ font-size: .78rem; color: #5c6b7a; text-align: center; margin-top: 3px }}
footer {{ margin-top: 28px; padding-top: 8px; border-top: 1px solid #d5dde5;
  color: #8b98a6; font-size: .75rem; display: flex; justify-content: space-between }}
</style></head><body>
<header>
  <h1>{_esc(title)}</h1>
  <div class="meta">Erstellt am {stamp} · ScanToBIM {__version__}</div>
</header>
{''.join(sections)}
<footer><span>ScanToBIM {__version__} — Rekonstruktion mit sauberen Kanten</span>
<span>Seite&nbsp;·&nbsp;{stamp}</span></footer>
</body></html>
"""
    path.write_text(doc, encoding="utf-8")
    return path
