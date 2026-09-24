"""The hip_model.m workflow: bone geometries -> OpenSim model.

    from pystaple.workflow import create_tri_geom_set, build_hip_model
    geom_set = create_tri_geom_set(["pelvis_no_sacrum", "femur_r", "tibia_r"], "data/stl")
    build_hip_model(geom_set, "output", body_mass=64)   # no OpenSim needed
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
import warnings
from pathlib import Path

from .io import load_mesh
from .mesh import TriMesh
from .osim.geometry_files import write_model_geometries_folder
from .osim.joints import infer_body_side
from .osim.writer import model_description, write_osim
from .processing import process_tri_geom_bone_set

log = logging.getLogger(__name__)

HIP_MODEL_BONES = ("pelvis_no_sacrum", "femur_r", "tibia_r")


@contextmanager
def working_directory(path: str | Path | None):
    """Temporarily change the working directory (no-op for None)."""
    if path is None:
        yield
        return
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def create_tri_geom_set(bones_list, geom_folder: str | Path) -> dict[str, TriMesh]:
    """createTriGeomSet.m: read <bone>.stl, <bone>.mat or mesh_<bone>.mat.

    .mat files must contain Points and 1-based ConnectivityList arrays (as saved
    by export_reference_outputs.m); MATLAB triangulation objects cannot be read.
    Missing bones are skipped, as in MATLAB.
    """
    geom_folder = Path(geom_folder)
    geom_set = {}
    for name in bones_list:
        candidates = [geom_folder / f"{name}.stl", geom_folder / f"{name}.mat", geom_folder / f"mesh_{name}.mat"]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            warnings.warn(f"No geometry for {name!r} in {geom_folder}", stacklevel=2)
            continue
        geom_set[name] = load_mesh(path)
    if not geom_set:
        raise FileNotFoundError(f"createTriGeomSet: no triangulations read from {geom_folder}")
    return geom_set


def hip_model_description(
    geom_set: dict[str, TriMesh],
    JCS: dict,
    BL: dict,
    model_name: str,
    body_mass: float = 64.0,
    joint_defs: str = "auto2020",
    geometry_folder_name: str = "Geometry",
    vis_geom_format: str = "obj",
) -> dict:
    """The model of hip_model.m as plain data (no OpenSim needed), see osim.writer."""
    return model_description(
        geom_set, JCS, BL, model_name, body_mass, joint_defs, geometry_folder_name, vis_geom_format
    )


def assemble_hip_model(
    geom_set: dict[str, TriMesh],
    JCS: dict,
    BL: dict,
    model_name: str,
    body_mass: float = 64.0,
    joint_defs: str = "auto2020",
    geometry_folder_name: str = "Geometry",
    vis_geom_format: str = "obj",
    model_file: str | Path | None = None,
):
    """OpenSim part of hip_model.m through the OpenSim API (needs ``opensim``).

    Returns an ``opensim.Model``. The default way of building models is the
    OpenSim-free writer (hip_model_description + write_osim); this function is
    kept as an independent check of the writer and for users of the API.

    `model_file` is where the model will be saved. OpenSim loads the mesh files
    (paths relative to the model) while the model is assembled, so the model is
    assembled with the working directory set to the model folder; otherwise
    OpenSim warns "Couldn't find file 'Geometry/...'".
    """
    from .osim.model import (
        add_bodies_from_tri_geom_bone_set,
        add_bone_landmarks_as_markers,
        assign_mass_props_to_segments,
        create_opensim_model_joints,
        initialize_opensim_model,
    )

    model_dir = None if model_file is None else Path(model_file).resolve().parent
    with working_directory(model_dir):
        model = initialize_opensim_model(model_name)
        add_bodies_from_tri_geom_bone_set(model, geom_set, geometry_folder_name, vis_geom_format)
        create_opensim_model_joints(model, JCS, joint_defs)
        assign_mass_props_to_segments(model, JCS, body_mass)
        add_bone_landmarks_as_markers(model, BL)
        model.finalizeConnections()
    return model


def build_hip_model(
    geom_set: dict[str, TriMesh],
    output_folder: str | Path,
    body_mass: float = 64.0,
    joint_defs: str = "auto2020",
    vis_geom_format: str = "obj",
    model_file_name: str = "bone_model.osim",
    coeff_face_reduc: float = 0.3,
    backend: str = "xml",
):
    """hip_model.m for one dataset -> (path of the .osim file, JCS, BL, BCS).

    backend "xml" (default) writes the .osim file directly and needs no OpenSim;
    "opensim" builds it through the OpenSim API (identical file).
    """
    if backend not in ("xml", "opensim"):
        raise ValueError(f"backend must be 'xml' or 'opensim', not {backend!r}")
    t0 = time.perf_counter()
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    side = infer_body_side(geom_set)
    model_name = f"{joint_defs}_hip_{side.upper()}"

    geometry_folder_name = "Geometry"
    write_model_geometries_folder(
        geom_set, output_folder / geometry_folder_name, vis_geom_format, coeff_face_reduc
    )
    JCS, BL, BCS = process_tri_geom_bone_set(geom_set, side)
    out = (output_folder / model_file_name).resolve()
    if backend == "xml":
        model = hip_model_description(
            geom_set, JCS, BL, model_name, body_mass, joint_defs, geometry_folder_name, vis_geom_format
        )
        write_osim(model, out)
    else:
        model = assemble_hip_model(
            geom_set, JCS, BL, model_name, body_mass, joint_defs, geometry_folder_name, vis_geom_format, out
        )
        model.printToXML(str(out))
    log.info("Model generated in %.1f s, saved as %s", time.perf_counter() - t0, out)
    return out, JCS, BL, BCS
