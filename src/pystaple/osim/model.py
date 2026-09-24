"""Building the OpenSim model (requires the ``opensim`` Python package).

Ports of initializeOpenSimModel.m, addBodiesFromTriGeomBoneSet.m,
addBodyFromTriGeomObj.m, createOpenSimModelJoints.m,
createCustomJointFromStruct.m, createSpatialTransformFromStruct.m,
assignMassPropsToSegments.m and addBoneLandmarksAsMarkers.m.

All numbers are computed in the pure-Python modules (joints, anthropometry);
this module only transfers them to OpenSim objects. It is an optional backend:
by default models are written without OpenSim (osim.writer).
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from ..mesh import TriMesh
from .anthropometry import BONE_DENSITY, bone_mass_props, segment_mass_props
from .joints import ROT, TRANS, finalize_joint_struct, infer_body_side

log = logging.getLogger(__name__)

from .writer import CREDITS, GRAVITY  # noqa: E402  (same header as the XML writer)


def import_opensim():
    try:
        import opensim
    except ImportError as exc:
        raise ImportError(
            "The OpenSim Python package is required to build models: "
            "conda install -c opensim-org opensim"
        ) from exc
    return opensim


def _vec3(osim, v) -> "object":
    x, y, z = (float(c) for c in np.asarray(v, dtype=np.float64).ravel())
    return osim.Vec3(x, y, z)


def _to_array(v3) -> np.ndarray:
    return np.array([v3.get(i) for i in range(3)])


def body_names(model) -> list[str]:
    bs = model.getBodySet()
    return [bs.get(i).getName() for i in range(bs.getSize())]


# --- model and bodies -----------------------------------------------------------


def initialize_opensim_model(model_name: str):
    """initializeOpenSimModel.m."""
    osim = import_opensim()
    model = osim.Model()
    model.setGravity(_vec3(osim, GRAVITY))
    model.setName(model_name)
    model.set_credits(CREDITS)
    return model


def add_body_from_tri_geom(
    model, mesh: TriMesh, body_name: str, vis_mesh_file: str | None = None,
    body_density: float = BONE_DENSITY, in_mm: bool = True,
):
    """addBodyFromTriGeomObj.m (body_density in kg/m^3)."""
    osim = import_opensim()
    dim_fact = 0.001 if in_mm else 1.0
    mp = bone_mass_props(mesh, body_density, in_mm)
    body = osim.Body(
        body_name, float(mp["mass"]), _vec3(osim, mp["mass_center"]), osim.Inertia(*map(float, mp["inertia"]))
    )
    model.addBody(body)
    if vis_mesh_file is not None:
        geom = osim.Mesh(vis_mesh_file)
        geom.set_scale_factors(osim.Vec3(dim_fact))
        body.attachGeometry(geom)
    return body


def add_bodies_from_tri_geom_bone_set(
    model, geom_set: dict[str, TriMesh], vis_geom_folder: str = "Geometry",
    vis_geom_format: str = "obj", body_density: float = BONE_DENSITY, in_mm: bool = True,
):
    """addBodiesFromTriGeomBoneSet.m. 'pelvis_no_sacrum' becomes the body 'pelvis'.

    Mesh paths are written with '/' (MATLAB fullfile gives '\\' on Windows);
    OpenSim accepts both.
    """
    for name, mesh in geom_set.items():
        vis_file = f"{vis_geom_folder}/{name}.{vis_geom_format}"
        body_name = "pelvis" if name == "pelvis_no_sacrum" else name
        log.info("Adding body %s", body_name)
        add_body_from_tri_geom(model, mesh, body_name, vis_file, body_density, in_mm)
    return model


# --- joints ---------------------------------------------------------------------


def _axis_vec(label: str) -> np.ndarray:
    return np.eye(3)["xyz".index(label.lower())]


def create_spatial_transform_from_struct(js: dict):
    """createSpatialTransformFromStruct.m."""
    osim = import_opensim()
    names, types = js["coordsNames"], js["coordsTypes"]
    rot = [n for n, t in zip(names, types) if t == ROT]
    trans = [n for n, t in zip(names, types) if t == TRANS]
    if len(rot) + len(trans) != len(names):
        raise ValueError("coordinate types must be 'rotational' or 'translational'")
    coords = rot + [""] * (3 - len(rot)) + trans + [""] * (3 - len(trans))

    v = np.vstack([np.eye(3), np.eye(3)])
    rot_axes = js.get("rotationAxes")
    if isinstance(rot_axes, str):
        v[:3] = [_axis_vec(c) for c in rot_axes]
    elif rot_axes is not None:
        v[: len(rot)] = np.asarray(rot_axes, float)[: len(rot)]
    trans_axes = js.get("translationAxes")
    if isinstance(trans_axes, str):
        v[3:] = [_axis_vec(c) for c in trans_axes]
    elif trans_axes is not None:
        v[3 : 3 + len(rot)] = np.asarray(trans_axes, float)[: len(rot)]  # sic: MATLAB loops over nr_rot

    st = osim.SpatialTransform()
    lin_fun = osim.LinearFunction(1, 0)
    const_fun = osim.Constant(0)
    for n in range(6):
        ta = st.updTransformAxis(n)
        ta.setAxis(_vec3(osim, v[n]))
        arr = osim.ArrayStr()
        if coords[n]:
            arr.append(coords[n])
        ta.setCoordinateNames(arr)
        ta.set_function(lin_fun if coords[n] else const_fun)
    st.constructIndependentAxes(len(rot), 0)
    return st


def create_custom_joint_from_struct(model, js: dict):
    """createCustomJointFromStruct.m."""
    osim = import_opensim()
    parent = model.getGround() if js["parentName"] == "ground" else model.getBodySet().get(js["parentName"])
    child = model.getBodySet().get(js["childName"])
    joint = osim.CustomJoint(
        js["jointName"],
        parent, _vec3(osim, js["parent_location"]), _vec3(osim, js["parent_orientation"]),
        child, _vec3(osim, js["child_location"]), _vec3(osim, js["child_orientation"]),
        create_spatial_transform_from_struct(js),
    )
    model.addJoint(joint)
    if "coordRanges" in js:
        for i, (rom, ctype) in enumerate(zip(js["coordRanges"], js["coordsTypes"])):
            rom = np.asarray(rom, dtype=np.float64)
            if ctype == ROT:
                rom = rom / 180 * np.pi
            coord = joint.upd_coordinates(i)
            coord.setRangeMin(float(rom[0]))
            coord.setRangeMax(float(rom[1]))
    return joint


def create_opensim_model_joints(model, JCS: dict, joint_defs: str = "auto2020") -> dict:
    """createOpenSimModelJoints.m. Returns the joint structs that were added."""
    joint_struct = finalize_joint_struct(JCS, joint_defs)
    for js in joint_struct.values():
        create_custom_joint_from_struct(model, js)
        log.info("Added joint %s", js["jointName"])
    return joint_struct


# --- mass properties and markers ------------------------------------------------------


def assign_mass_props_to_segments(model, JCS: dict, subj_mass: float, side: str | None = None):
    """assignMassPropsToSegments.m: gait2392 masses scaled to `subj_mass`, Winter COMs."""
    osim = import_opensim()
    side = infer_body_side(JCS) if side is None else side[0].lower()
    names = body_names(model)
    bodies = model.updBodySet()
    initial = {n: _to_array(bodies.get(n).getMassCenter()) for n in names}
    for name, mp in segment_mass_props(names, initial, JCS, subj_mass, side).items():
        body = bodies.get(name)
        body.setMass(float(mp["mass"]))
        body.setMassCenter(_vec3(osim, mp["mass_center"]))
        body.setInertia(osim.Inertia(*map(float, mp["inertia"])))
    return model


def add_bone_landmarks_as_markers(model, BL: dict, in_mm: bool = True) -> None:
    """addBoneLandmarksAsMarkers.m (BL = {body: {marker: global position}})."""
    osim = import_opensim()
    dim_fact = 0.001 if in_mm else 1.0
    for body_name, markers in BL.items():
        if model.getBodySet().getIndex(body_name) < 0:
            warnings.warn(
                f"Markers assigned to body {body_name} cannot be added: body is not in the BodySet.",
                stacklevel=2,
            )
            continue
        frame = model.getBodySet().get(body_name)
        for marker_name, loc in (markers or {}).items():
            model.addMarker(osim.Marker(marker_name, frame, _vec3(osim, np.asarray(loc) * dim_fact)))
