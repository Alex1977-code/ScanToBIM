"""PyInstaller entry point for the Windows build of ScanToBIM."""

import multiprocessing
import sys

# Imported explicitly so PyInstaller bundles the GUI (loaded lazily at runtime).
import scantobim.gui.page  # noqa: F401
import scantobim.gui.server  # noqa: F401
from scantobim.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
