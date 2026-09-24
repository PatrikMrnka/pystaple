"""Visualization geometries of the model bodies.

Ports of writeModelGeometriesFolder.m, reduceTriObjGeometry.m, writeOBJfile.m.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np

from ..mesh import TriMesh

log = logging.getLogger(__name__)


def write_obj(mesh: TriMesh, path: str | Path) -> None:
    """writeOBJfile.m: vertices and 1-based faces, with MATLAB's number format."""
    lines = [f"v {x:.5f} {y:.5f} {z:12.8f}\n" for x, y, z in mesh.points.tolist()]
    lines += [f"f {a} {b} {c}\n" for a, b, c in (mesh.faces + 1).tolist()]
    Path(path).write_text("".join(lines), encoding="ascii")


def write_stl_ascii(mesh: TriMesh, path: str | Path, name: str = "") -> None:
    """ASCII STL (OpenSim does not read binary STL)."""
    out = [f"solid {name}\n"]
    for (p1, p2, p3), n in zip(mesh.points[mesh.faces], mesh.face_normals()):
        out.append(f"facet normal {n[0]:.7e} {n[1]:.7e} {n[2]:.7e}\nouter loop\n")
        out += [f"vertex {p[0]:.7e} {p[1]:.7e} {p[2]:.7e}\n" for p in (p1, p2, p3)]
        out.append("endloop\nendfacet\n")
    out.append(f"endsolid {name}\n")
    Path(path).write_text("".join(out), encoding="ascii")


def reduce_tri_obj_geometry(mesh: TriMesh, coeff_reduc: float = 0.3) -> TriMesh:
    """reduceTriObjGeometry.m: keep about `coeff_reduc` of the faces.

    MATLAB uses ``reducepatch``, which cannot be replicated exactly; here the
    quadric decimation of the optional package ``fast-simplification`` is used.
    This only affects the visualization files, not the model. Without the
    package the full-resolution mesh is returned.
    """
    if coeff_reduc >= 1:
        return mesh
    try:
        import fast_simplification
    except ImportError:
        warnings.warn(
            "fast-simplification is not installed: visualization geometries are written "
            "at full resolution (pip install fast-simplification).",
            stacklevel=2,
        )
        return mesh
    pts, faces = fast_simplification.simplify(
        mesh.points.astype(np.float64), mesh.faces.astype(np.int64), target_reduction=1 - coeff_reduc
    )
    return TriMesh(pts, faces)


def write_model_geometries_folder(
    geom_set: dict[str, TriMesh],
    geom_folder: str | Path = ".",
    file_format: str = "obj",
    coeff_face_reduc: float = 0.3,
) -> Path:
    """writeModelGeometriesFolder.m: one reduced mesh file per bone, named after it."""
    geom_folder = Path(geom_folder)
    geom_folder.mkdir(parents=True, exist_ok=True)
    file_format = file_format.lower()
    if file_format not in ("obj", "stl"):
        raise ValueError("file_format must be 'obj' or 'stl'")
    for name, mesh in geom_set.items():
        reduced = reduce_tri_obj_geometry(mesh, coeff_face_reduc)
        path = geom_folder / f"{name}.{file_format}"
        if file_format == "obj":
            write_obj(reduced, path)
        else:
            write_stl_ascii(reduced, path, name)
    log.info("Stored %s files in folder %s", file_format, geom_folder)
    return geom_folder
