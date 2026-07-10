"""PyInstaller entry point for the Windows build of ScanToBIM."""

import multiprocessing
import sys

from scantobim.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
