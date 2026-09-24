"""Reading meshes and MATLAB reference outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .mesh import TriMesh


def read_stl(path: str | Path) -> TriMesh:
    """Read a binary or ASCII STL and merge duplicate vertices."""
    path = Path(path)
    data = path.read_bytes()
    corners = _parse_binary_stl(data)
    if corners is None:
        corners = _parse_ascii_stl(data.decode("utf-8", errors="replace"))
    unique_pts, inverse = np.unique(corners.reshape(-1, 3), axis=0, return_inverse=True)
    return TriMesh(unique_pts, inverse.reshape(-1, 3))


def _parse_binary_stl(data: bytes) -> np.ndarray | None:
    if len(data) < 84:
        return None
    n_faces = int(np.frombuffer(data, dtype="<u4", count=1, offset=80)[0])
    if len(data) != 84 + 50 * n_faces:
        return None  # not a consistent binary STL -> try ASCII
    record = np.dtype(
        [("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")]
    )
    faces = np.frombuffer(data, dtype=record, count=n_faces, offset=84)
    return faces["v"].astype(np.float64)


def _parse_ascii_stl(text: str) -> np.ndarray:
    coords = [
        [float(t) for t in line.split()[1:4]]
        for line in text.splitlines()
        if line.strip().startswith("vertex")
    ]
    if not coords or len(coords) % 3:
        raise ValueError("Could not parse STL file")
    return np.asarray(coords, dtype=np.float64).reshape(-1, 3, 3)


def load_mesh(path: str | Path) -> TriMesh:
    """Load .stl, or a .mat exported by export_reference_outputs.m."""
    path = Path(path)
    if path.suffix.lower() == ".stl":
        return read_stl(path)
    if path.suffix.lower() == ".mat":
        return load_reference_mesh(path)
    raise ValueError(f"Unsupported mesh format: {path}")


def load_reference_mesh(path: str | Path) -> TriMesh:
    """Mesh saved by export_reference_outputs.m (1-based ConnectivityList)."""
    mat = loadmat(str(path))
    faces = np.asarray(mat["ConnectivityList"], dtype=np.int64) - 1
    return TriMesh(np.asarray(mat["Points"], dtype=np.float64), faces)


def load_reference_json(path: str | Path) -> dict:
    """reference.json with JSON ``null`` (MATLAB NaN) converted to ``nan``."""
    with open(path, encoding="utf-8") as f:
        return _nulls_to_nan(json.load(f))


def _nulls_to_nan(x):
    if isinstance(x, dict):
        return {k: _nulls_to_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_nulls_to_nan(v) for v in x]
    return float("nan") if x is None else x
