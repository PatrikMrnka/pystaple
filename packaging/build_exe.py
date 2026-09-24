"""Build the standalone executable with PyInstaller and zip it.

    pip install ".[viz]" pyinstaller
    python packaging/build_exe.py

Result: dist/pystaple/pystaple(.exe) and dist/pystaple-<version>-<platform>.zip.
The executable needs neither Python nor OpenSim.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import PyInstaller.__main__

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
NAME = "pystaple"


def platform_tag() -> str:
    system = {"win32": "windows", "darwin": "macos"}.get(sys.platform, sys.platform)
    machine = platform.machine().lower().replace("amd64", "x64").replace("x86_64", "x64")
    return f"{system}-{machine}"


def main() -> None:
    from pystaple import __version__

    PyInstaller.__main__.run([
        str(ROOT / "packaging" / "pystaple_exe.py"),
        "--name", NAME,
        "--onedir",  # a folder, not one file: starts faster, fewer antivirus false alarms
        "--console",
        "--noconfirm",
        "--clean",
        "--distpath", str(DIST),
        "--workpath", str(ROOT / "build" / "pyinstaller"),
        "--specpath", str(ROOT / "build"),
        # the executable writes models itself; these are not needed
        "--exclude-module", "opensim",
        "--exclude-module", "matplotlib",
        "--exclude-module", "tkinter",
        "--exclude-module", "pytest",
        "--hidden-import", "fast_simplification",
    ])

    app = DIST / NAME
    for f in ("LICENSE", "NOTICE", "CHANGELOG.md"):
        shutil.copy2(ROOT / f, app / f)
    shutil.copy2(ROOT / "packaging" / "README_exe.txt", app / "README.txt")

    exe = app / (NAME + (".exe" if sys.platform == "win32" else ""))
    out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    print(f"smoke test: {out}")
    if __version__ not in out:
        sys.exit(f"unexpected version output: {out!r}")

    zip_path = DIST / f"{NAME}-{__version__}-{platform_tag()}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(app.rglob("*")):
            zf.write(p, Path(NAME) / p.relative_to(app))
    print(f"{zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
