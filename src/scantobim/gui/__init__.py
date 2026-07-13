"""Local web GUI: a professional front end served on 127.0.0.1.

No extra dependencies — Python's http.server plus a self-contained HTML
app (inline CSS/JS, WebGL viewer reuse). Started via ``scantobim gui`` or
by double-clicking the packaged executable.
"""

from scantobim.gui.server import run_gui

__all__ = ["run_gui"]
