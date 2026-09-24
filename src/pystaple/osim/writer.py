"""Writing OpenSim models (.osim) without the OpenSim package.

The model is first described with plain Python data (``model_description``),
then serialized (``write_osim``) in exactly the layout OpenSim 4.x prints: same
elements, order, comments and number format (``%.17g``). The equivalence with
OpenSim is tested by printing the same model through the OpenSim API
(tests/test_osim_model.py, only where ``opensim`` is installed).

Ports of the non-API parts of initializeOpenSimModel.m, addBodiesFromTriGeomBoneSet.m,
createOpenSimModelJoints.m, createSpatialTransformFromStruct.m,
assignMassPropsToSegments.m, addBoneLandmarksAsMarkers.m, and of OpenSim's
SpatialTransform::constructIndependentAxes.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import numpy as np

from ..mesh import TriMesh
from .anthropometry import BONE_DENSITY, bone_mass_props, segment_mass_props
from .joints import ROT, TRANS, finalize_joint_struct, infer_body_side

CREDITS = (
    "Luca Modenese, Jean-Baptiste Renault 2020. Model created using the STAPLE "
    "(Shared Tools for Automatic Personalised Lower Extremity) modelling toolbox. "
    "GitHub page: https://github.com/modenaxe/msk-STAPLE."
)
GRAVITY = (0.0, -9.8081, 0.0)
OSIM_VERSION = 40500  # document version written by OpenSim 4.5 (read by 4.x)
FRAME_GEOMETRY_SCALE = 0.2


# --- model description ---------------------------------------------------------------


def body_name_of(bone: str) -> str:
    """addBodiesFromTriGeomBoneSet.m: 'pelvis_no_sacrum' becomes the body 'pelvis'."""
    return "pelvis" if bone == "pelvis_no_sacrum" else bone


def construct_independent_axes(axes: np.ndarray, n_axes: int, start: int = 0) -> np.ndarray:
    """OpenSim SpatialTransform::constructIndependentAxes (axes: (6, 3), modified copy)."""
    axes = np.array(axes, dtype=np.float64)
    if n_axes in (0, 3):
        return axes
    v1, v2, v3 = axes[start], axes[start + 1], axes[start + 2]
    if n_axes == 2:
        c = np.cross(v1, v2)
    else:
        if abs(abs(v1 @ v2) - 1) < 1e-4:
            axes[start + 1] = v3
        c = np.cross(axes[start + 1], v1)
    axes[start + 2] = c / np.linalg.norm(c)
    return axes


def opensim_orientation(angles) -> np.ndarray:
    """XYZ body-fixed angles as OpenSim stores them in a joint's offset frame.

    OpenSim does not keep the angles it is given: Joint builds a SimTK::Rotation
    from them and OffsetFrame converts the rotation back to angles, which can
    change the last bit. This replicates Simbody's
    setThreeAngleThreeAxesBodyFixedForwardCyclicalRotation and
    convertThreeAxesBodyFixedRotationToThreeAngles (X-Y-Z, body fixed) operation
    by operation, with the C library functions (math.*), as SimTK does.
    """
    a1, a2, a3 = (float(a) for a in np.asarray(angles, dtype=np.float64).ravel())
    c1, s1 = math.cos(a1), math.sin(a1)
    c2, s2 = math.cos(a2), math.sin(a2)
    c3, s3 = math.cos(a3), math.sin(a3)
    s1c3, s3c1, s1s3, c1c3 = s1 * c3, s3 * c1, s1 * s3, c1 * c3
    R00 = c2 * c3
    R01 = -s3 * c2
    R02 = s2
    R10 = s3c1 + s2 * s1c3
    R11 = c1c3 - s2 * s1s3
    R12 = -s1 * c2
    R20 = s1s3 - s2 * c1c3
    R21 = s1c3 + s2 * s3c1
    R22 = c1 * c2

    rsum = math.sqrt((R00 * R00 + R01 * R01 + R12 * R12 + R22 * R22) / 2)
    theta2 = math.atan2(R02, rsum)
    if rsum > 4 * np.finfo(float).eps:
        theta1 = math.atan2(-R12, R22)
        theta3 = math.atan2(-R01, R00)
    elif R02 > 0:
        theta1, theta3 = math.atan2(R10 + R21, R11 - R20), 0.0
    else:
        theta1, theta3 = math.atan2(R21 - R10, R11 + R20), 0.0
    return np.array([theta1, theta2, theta3])


def _axis_vec(label: str) -> np.ndarray:
    return np.eye(3)["xyz".index(label.lower())]


def spatial_transform(js: dict) -> list[dict]:
    """createSpatialTransformFromStruct.m -> 6 transform axes
    ({"name", "coordinate" ("" if none), "axis"})."""
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
    v = construct_independent_axes(v, len(rot), 0)
    labels = ["rotation1", "rotation2", "rotation3", "translation1", "translation2", "translation3"]
    return [{"name": lbl, "coordinate": c, "axis": ax} for lbl, c, ax in zip(labels, coords, v)]


def joint_description(js: dict) -> dict:
    """createCustomJointFromStruct.m (+ the offset frames created by OpenSim)."""
    ranges = []
    if "coordRanges" in js:
        for rom, ctype in zip(js["coordRanges"], js["coordsTypes"]):
            rom = np.asarray(rom, dtype=np.float64)
            ranges.append(rom / 180 * np.pi if ctype == ROT else rom)
    else:
        ranges = [None] * len(js["coordsNames"])
    parent, child = js["parentName"], js["childName"]
    return {
        "name": js["jointName"],
        "parent_frame": {
            "name": f"{parent}_offset",
            "socket_parent": "/ground" if parent == "ground" else f"/bodyset/{parent}",
            "translation": np.asarray(js["parent_location"], float),
            "orientation": opensim_orientation(js["parent_orientation"]),
        },
        "child_frame": {
            "name": f"{child}_offset",
            "socket_parent": f"/bodyset/{child}",
            "translation": np.asarray(js["child_location"], float),
            "orientation": opensim_orientation(js["child_orientation"]),
        },
        "coordinates": list(zip(js["coordsNames"], ranges)),
        "axes": spatial_transform(js),
    }


def model_description(
    geom_set: dict[str, TriMesh],
    JCS: dict,
    BL: dict,
    model_name: str,
    body_mass: float,
    joint_defs: str = "auto2020",
    geometry_folder_name: str = "Geometry",
    vis_geom_format: str = "obj",
    in_mm: bool = True,
) -> dict:
    """Everything hip_model.m puts into the OpenSim model, as plain data."""
    dim_fact = 0.001 if in_mm else 1.0
    side = infer_body_side(JCS)

    bones = {body_name_of(b): b for b in geom_set}
    initial_com = {body: bone_mass_props(geom_set[bone], BONE_DENSITY, in_mm)["mass_center"]
                   for body, bone in bones.items()}
    mass_props = segment_mass_props(list(bones), initial_com, JCS, body_mass, side)
    bodies = [
        {
            "name": body,
            "mass": float(mass_props[body]["mass"]),
            "mass_center": mass_props[body]["mass_center"],
            "inertia": mass_props[body]["inertia"],
            "mesh_file": f"{geometry_folder_name}/{bone}.{vis_geom_format}",
            "mesh_scale": dim_fact,
        }
        for body, bone in bones.items()
    ]

    joints = [joint_description(js) for js in finalize_joint_struct(JCS, joint_defs).values()]

    markers = []
    for body, landmarks in BL.items():
        if body not in bones:
            warnings.warn(f"Markers assigned to body {body} cannot be added: body is not in the model.",
                          stacklevel=2)
            continue
        for name, loc in (landmarks or {}).items():
            markers.append({"name": name, "body": body, "location": np.asarray(loc, float).ravel() * dim_fact})

    return {"name": model_name, "credits": CREDITS, "gravity": np.array(GRAVITY),
            "bodies": bodies, "joints": joints, "markers": markers}


def description_from_osim(parsed: dict) -> dict:
    """Model description from a parsed .osim file (osim.read_osim), e.g. to rewrite it."""
    joints = []
    for name, j in parsed["joints"].items():
        def frame(fname, frames=j["frames"]):
            f = frames[fname]
            return {"name": fname, "socket_parent": f["parent"],
                    "translation": f["translation"], "orientation": f["orientation"]}

        joints.append({
            "name": name,
            "parent_frame": frame(j["parent_frame"]),
            "child_frame": frame(j["child_frame"]),
            "coordinates": [(c, v["range"]) for c, v in j["coordinates"].items()],
            "axes": [{"name": a, "coordinate": (v["coordinates"] or [""])[0], "axis": v["axis"]}
                     for a, v in j["axes"].items()],
        })
    bodies = []
    for name, b in parsed["bodies"].items():
        mesh = b["meshes"][0] if b["meshes"] else None
        bodies.append({"name": name, "mass": b["mass"], "mass_center": b["mass_center"], "inertia": b["inertia"],
                       "mesh_file": mesh["file"] if mesh else None,
                       "mesh_scale": mesh["scale_factors"][0] if mesh else None})
    markers = [{"name": n, "body": m["parent_frame"].rsplit("/", 1)[-1], "location": m["location"]}
               for n, m in parsed["markers"].items()]
    return {"name": parsed["name"], "credits": parsed["credits"], "gravity": parsed["gravity"],
            "bodies": bodies, "joints": joints, "markers": markers}


# --- serialization -------------------------------------------------------------------------


def fmt(x: float) -> str:
    """A double as OpenSim/SimTK prints it (%.17g)."""
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    return "%.17g" % x


def fmt_vec(v) -> str:
    return " ".join(fmt(x) for x in np.asarray(v, dtype=np.float64).ravel())


# property documentation comments, exactly as OpenSim writes them
C = {
    "ground": "The model's ground reference frame.",
    "gravity": "Acceleration due to gravity, expressed in ground.",
    "credits": "Credits (e.g., model author names) associated with the model.",
    "bodyset": "List of bodies that make up this model.",
    "jointset": "List of joints that connect the bodies.",
    "controllerset": "Controllers that provide the control inputs for Actuators.",
    "forceset": "Forces in the model (includes Actuators).",
    "markerset": "Markers in the model.",
    "frame_geometry": "The geometry used to display the axes of this Frame.",
    "socket_frame": "Path to a Component that satisfies the Socket 'frame' of type Frame.",
    "scale_factors": "Scale factors in X, Y, Z directions respectively.",
    "attached_geometry": "List of geometry attached to this Frame. Note, the geometry are treated as fixed to "
                         "the frame and they share the transform of the frame when visualized",
    "mesh_file": "Name of geometry file.",
    "mass": "The mass of the body (kg)",
    "mass_center": "The location (Vec3) of the mass center in the body frame.",
    "inertia": "The elements of the inertia tensor (Vec6) as [Ixx Iyy Izz Ixy Ixz Iyz] measured about the "
               "mass_center and not the body origin.",
    "socket_parent_frame": "Path to a Component that satisfies the Socket 'parent_frame' of type PhysicalFrame "
                           "(description: The parent frame for the joint.).",
    "socket_child_frame": "Path to a Component that satisfies the Socket 'child_frame' of type PhysicalFrame "
                          "(description: The child frame for the joint.).",
    "coordinates": "List containing the generalized coordinates (q's) that parameterize this joint.",
    "range": "The minimum and maximum values that the coordinate can range between. Rotational coordinate "
             "range in radians and Translational in meters.",
    "frames": "Physical offset frames owned by the Joint that are typically used to satisfy the owning Joint's "
              "parent and child frame connections (sockets). PhysicalOffsetFrames are often used to describe "
              "the fixed transformation from a Body's origin to another location of interest on the Body "
              "(e.g., the joint center). When the joint is deleted, so are the PhysicalOffsetFrame components "
              "in this list.",
    "socket_parent": "Path to a Component that satisfies the Socket 'parent' of type C (description: The parent "
                     "frame to this frame.).",
    "translation": "Translational offset (in meters) of this frame's origin from the parent frame's origin, "
                   "expressed in the parent frame.",
    "orientation": "Orientation offset (in radians) of this frame in its parent frame, expressed as a "
                   "frame-fixed x-y-z rotation sequence.",
    "spatial_transform": "Defines how the child body moves with respect to the parent as a function of the "
                         "generalized coordinates.",
    "rot_axes": "3 Axes for rotations are listed first.",
    "trans_axes": "3 Axes for translations are listed next.",
    "ta_coordinates": "Names of the coordinates that serve as the independent variables         of the "
                      "transform function.",
    "ta_axis": "Rotation or translation axis for the transform.",
    "ta_function": "Transform function of the generalized coordinates used to        represent the amount of "
                   "displacement along a specified axis.",
    "marker_parent": "Path to a Component that satisfies the Socket 'parent_frame' of type PhysicalFrame "
                     "(description: The frame to which this station is fixed.).",
    "marker_location": "The fixed location of the station expressed in its parent frame.",
}


class _Writer:
    def __init__(self):
        self.lines: list[str] = []
        self.depth = 0

    def line(self, text: str) -> None:
        self.lines.append("\t" * self.depth + text)

    def comment(self, key: str) -> None:
        self.line(f"<!--{C[key]}-->")

    def prop(self, key: str | None, tag: str, value: str) -> None:
        if key:
            self.comment(key)
        self.line(f"<{tag}>{escape(value)}</{tag}>")

    def open(self, tag: str, name: str | None = None, key: str | None = None) -> None:
        if key:
            self.comment(key)
        attr = f" name={quoteattr(name)}" if name is not None else ""
        self.line(f"<{tag}{attr}>")
        self.depth += 1

    def close(self, tag: str) -> None:
        self.depth -= 1
        self.line(f"</{tag}>")

    def frame_geometry(self) -> None:
        self.open("FrameGeometry", "frame_geometry", "frame_geometry")
        self.prop("socket_frame", "socket_frame", "..")
        self.prop("scale_factors", "scale_factors", fmt_vec([FRAME_GEOMETRY_SCALE] * 3))
        self.close("FrameGeometry")

    def empty_set(self, tag: str, name: str, key: str) -> None:
        self.open(tag, name, key)
        self.line("<objects />")
        self.line("<groups />")
        self.close(tag)


def osim_text(model: dict) -> str:
    """The .osim document of a model description."""
    w = _Writer()
    w.line('<?xml version="1.0" encoding="UTF-8" ?>')
    w.open("OpenSimDocument")
    w.lines[-1] = f'<OpenSimDocument Version="{OSIM_VERSION}">'
    w.open("Model", model["name"])

    w.open("Ground", "ground", "ground")
    w.frame_geometry()
    w.close("Ground")
    w.prop("gravity", "gravity", fmt_vec(model["gravity"]))
    w.prop("credits", "credits", model["credits"])

    w.open("BodySet", "bodyset", "bodyset")
    w.open("objects")
    for b in model["bodies"]:
        w.open("Body", b["name"])
        w.frame_geometry()
        if b.get("mesh_file"):
            w.open("attached_geometry", key="attached_geometry")
            w.open("Mesh", f"{b['name']}_geom_1")
            w.prop("socket_frame", "socket_frame", "..")
            w.prop("scale_factors", "scale_factors", fmt_vec([b["mesh_scale"]] * 3))
            w.prop("mesh_file", "mesh_file", b["mesh_file"])
            w.close("Mesh")
            w.close("attached_geometry")
        w.prop("mass", "mass", fmt(b["mass"]))
        w.prop("mass_center", "mass_center", fmt_vec(b["mass_center"]))
        w.prop("inertia", "inertia", fmt_vec(b["inertia"]))
        w.close("Body")
    w.close("objects")
    w.line("<groups />")
    w.close("BodySet")

    w.open("JointSet", "jointset", "jointset")
    w.open("objects")
    for j in model["joints"]:
        w.open("CustomJoint", j["name"])
        w.prop("socket_parent_frame", "socket_parent_frame", j["parent_frame"]["name"])
        w.prop("socket_child_frame", "socket_child_frame", j["child_frame"]["name"])
        w.open("coordinates", key="coordinates")
        for cname, rng in j["coordinates"]:
            w.open("Coordinate", cname)
            if rng is not None:
                w.prop("range", "range", fmt_vec(rng))
            w.close("Coordinate")
        w.close("coordinates")
        w.open("frames", key="frames")
        for f in (j["parent_frame"], j["child_frame"]):
            w.open("PhysicalOffsetFrame", f["name"])
            w.frame_geometry()
            w.prop("socket_parent", "socket_parent", f["socket_parent"])
            w.prop("translation", "translation", fmt_vec(f["translation"]))
            w.prop("orientation", "orientation", fmt_vec(f["orientation"]))
            w.close("PhysicalOffsetFrame")
        w.close("frames")
        w.open("SpatialTransform", key="spatial_transform")
        for i, ax in enumerate(j["axes"]):
            if i in (0, 3):
                w.comment("rot_axes" if i == 0 else "trans_axes")
            w.open("TransformAxis", ax["name"])
            w.prop("ta_coordinates", "coordinates", ax["coordinate"])
            w.prop("ta_axis", "axis", fmt_vec(ax["axis"]))
            if ax["coordinate"]:
                w.open("LinearFunction", "function", "ta_function")
                w.line("<coefficients> 1 0</coefficients>")
                w.close("LinearFunction")
            else:
                w.open("Constant", "function", "ta_function")
                w.line("<value>0</value>")
                w.close("Constant")
            w.close("TransformAxis")
        w.close("SpatialTransform")
        w.close("CustomJoint")
    w.close("objects")
    w.line("<groups />")
    w.close("JointSet")

    w.empty_set("ControllerSet", "controllerset", "controllerset")
    w.empty_set("ForceSet", "forceset", "forceset")

    w.open("MarkerSet", "markerset", "markerset")
    w.open("objects")
    for m in model["markers"]:
        w.open("Marker", m["name"])
        w.prop("marker_parent", "socket_parent_frame", f"/bodyset/{m['body']}")
        w.prop("marker_location", "location", fmt_vec(m["location"]))
        w.close("Marker")
    w.close("objects")
    w.line("<groups />")
    w.close("MarkerSet")

    w.close("Model")
    w.close("OpenSimDocument")
    return "\n".join(w.lines) + "\n"


def write_osim(model: dict, path: str | Path) -> Path:
    """Write a model description as an .osim file (UTF-8, '\\n' line endings)."""
    path = Path(path)
    path.write_text(osim_text(model), encoding="utf-8", newline="\n")
    return path
