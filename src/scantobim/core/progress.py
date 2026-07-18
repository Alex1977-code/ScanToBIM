"""Coarse pipeline progress for the GUI progress bar.

The GUI runs jobs in a child process and streams stdout; progress is
emitted as ``##PROGRESS <frac> <phase>`` marker lines that the server
parses out of the stream (they never reach the visible protokoll). The
markers are only emitted when ``SCANTOBIM_PROGRESS=1`` — plain console
runs stay clean.

Fractions are coarse milestone estimates. They are monotonic within one
run and calibrated against real project runs, which is enough for the
GUI to show a bar and a remaining-time estimate.
"""

from __future__ import annotations

import os

MARKER = "##PROGRESS"


def report(frac: float, text: str = "") -> None:
    if os.environ.get("SCANTOBIM_PROGRESS") != "1":
        return
    frac = min(max(float(frac), 0.0), 1.0)
    print(f"{MARKER} {frac:.3f} {text}", flush=True)
