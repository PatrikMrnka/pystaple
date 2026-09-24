"""Entry point of the standalone executable (PyInstaller)."""

import sys

from pystaple.cli import main

if __name__ == "__main__":
    sys.exit(main())
