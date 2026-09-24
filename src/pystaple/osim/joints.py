"""Joint definitions for the OpenSim model, without OpenSim.

Ports of getJointParams.m, compileListOfJointsInJCSStruct.m,
inferBodySideFromAnatomicStruct.m, jointDefinitions_auto2020.m,
assembleJointStruct.m, verifyJointStructCompleteness.m and of the
non-OpenSim part of createOpenSimModelJoints.m (= finalizeJointStruct.m).

A "joint struct" is a dict with the keys of the MATLAB structure:
jointName, parentName, childName, coordsNames, coordsTypes, coordRanges,
rotationAxes, parent_location, parent_orientation, child_location,
child_orientation (locations in metres, orientations as XYZ angles in rad).
"""

from __future__ import annotations

import logging

import numpy as np

from ..mesh import normalize_v
from ..utils import compute_xyz_angle_seq

log = logging.getLogger(__name__)

ROT, TRANS = "rotational", "translational"
LOCATION_FIELDS = ("parent_location", "parent_orientation", "child_location", "child_orientation")


def get_joint_params(joint_name: str, root_body: str = "root_body") -> dict:
    """getJointParams.m: names, coordinates and ranges (deg / m) of a joint."""
    side = joint_name[-1] if joint_name[-2:] in ("_r", "_l") else None

    def params(name, parent, child, coords, types, ranges=None):
        p = {
            "jointName": name,
            "parentName": parent,
            "childName": child,
            "coordsNames": list(coords),
            "coordsTypes": list(types),
            "rotationAxes": "zxy",
        }
        if ranges is not None:
            p["coordRanges"] = [list(r) for r in ranges]
        return p

    rot3 = [ROT] * 3
    if joint_name == "ground_pelvis":
        return params(
            "ground_pelvis", "ground", "pelvis",
            ["pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz"],
            rot3 + [TRANS] * 3,
            [[-90, 90]] * 3 + [[-10, 10]] * 3,
        )
    if joint_name == "free_to_ground":
        cb = root_body
        return params(
            f"ground_{cb}", "ground", cb,
            [f"ground_{cb}_{c}" for c in ("rz", "rx", "ry", "tx", "ty", "tz")],
            rot3 + [TRANS] * 3,
            [[-120, 120]] * 3 + [[-10, 10]] * 3,
        )
    if side is not None:
        s = side
        if joint_name == f"hip_{s}":
            return params(
                f"hip_{s}", "pelvis", f"femur_{s}",
                [f"hip_flexion_{s}", f"hip_adduction_{s}", f"hip_rotation_{s}"],
                rot3, [[-120, 120]] * 3,
            )
        if joint_name == f"knee_{s}":
            return params(f"knee_{s}", f"femur_{s}", f"tibia_{s}", [f"knee_angle_{s}"], [ROT], [[-120, 10]])
        if joint_name == f"ankle_{s}":
            return params(f"ankle_{s}", f"tibia_{s}", f"talus_{s}", [f"ankle_angle_{s}"], [ROT], [[-90, 90]])
        if joint_name == f"subtalar_{s}":
            return params(
                f"subtalar_{s}", f"talus_{s}", f"calcn_{s}", [f"subtalar_angle_{s}"], [ROT], [[-90, 90]]
            )
        if joint_name == "patellofemoral_r":  # only the right side, no ranges (as in MATLAB)
            return params(
                f"patellofemoral_{s}", f"femur_{s}", f"patella_{s}", [f"knee_angle_{s}_beta"], [ROT]
            )
        if joint_name == f"mtp_{s}":
            # NOTE: MATLAB names this joint toes_<s>, not mtp_<s>
            return params(f"toes_{s}", f"calcn_{s}", f"toes_{s}", [f"mtp_angle_{s}"], [ROT], [[-90, 90]])
    raise ValueError(f"getJointParams: unsupported joint {joint_name!r}")


def compile_joint_list(JCS: dict) -> list[str]:
    """compileListOfJointsInJCSStruct.m: unique joint names in order of appearance."""
    out: list[str] = []
    for joints in JCS.values():
        for name in joints:
            if name not in out:
                out.append(name)
    return out


def infer_body_side(names) -> str:
    """inferBodySideFromAnatomicStruct.m: 'r' or 'l' from bone / joint names."""
    names = list(names)
    guessed = []
    for prefix in ("femur", "tibia", "talus", "calcn", "hip", "knee", "ankle", "subtalar"):
        guessed += [n[-1] for n in names if n.startswith(prefix)]
    sides = {g.lower() for g in guessed}
    if sides == {"r"}:
        return "r"
    if sides == {"l"}:
        return "l"
    raise ValueError("inferBodySideFromAnatomicStruct: it was not possible to infer the body side")


def joint_definitions_auto2020(JCS: dict, joint_struct: dict) -> dict:
    """jointDefinitions_auto2020.m: ankle parent frame (only with talus and tibia)."""
    side = infer_body_side(JCS)
    tibia, talus = f"tibia_{side}", f"talus_{side}"
    ankle, knee = f"ankle_{side}", f"knee_{side}"
    if talus in JCS and tibia in JCS:
        Zpar = normalize_v(np.asarray(JCS[talus][ankle]["V"])[:, 2])
        Ytemp = np.asarray(JCS[tibia][knee]["V"])[:, 1]
        Ypar = normalize_v(Ytemp - Zpar * (Zpar @ Ytemp) / np.linalg.norm(Zpar))
        Xpar = normalize_v(np.cross(Ytemp, Zpar))
        V = np.column_stack([Xpar, Ypar, Zpar])
        joint_struct[ankle]["V"] = V
        joint_struct[ankle]["parent_orientation"] = compute_xyz_angle_seq(V)
    return joint_struct


def assemble_joint_struct(joint_struct: dict) -> dict:
    """assembleJointStruct.m: a missing location/orientation on one body is copied
    from the other body (e.g. parent_location <- child_location)."""
    fields = LOCATION_FIELDS + LOCATION_FIELDS[:2]
    for name, j in joint_struct.items():
        complete = [f in j for f in fields[:4]]
        if all(complete):
            continue
        if not complete[0] and not complete[2]:
            log.warning("%s cannot be finalized: no joint locations available on either body.", name)
            continue
        if not complete[1] and not complete[3]:
            log.warning("%s cannot be finalized: no joint orientation available on either body.", name)
            continue
        for i, ok in enumerate(complete):
            if not ok:
                j[fields[i]] = j[fields[i + 2]]
                log.info("%s: %s missing, copied from %s", name, fields[i], fields[i + 2])
    return joint_struct


def verify_joint_struct_completeness(joint_struct: dict) -> None:
    """verifyJointStructCompleteness.m."""
    required = (
        "jointName", "parentName", "parent_location", "parent_orientation",
        "childName", "child_location", "child_orientation",
        "coordsNames", "coordsTypes", "rotationAxes",
    )
    problems = {n: [f for f in required if f not in j] for n, j in joint_struct.items()}
    problems = {n: m for n, m in problems.items() if m}
    if problems:
        details = "; ".join(f"{n}: missing {', '.join(m)}" for n, m in problems.items())
        raise ValueError(f"Incomplete joint definition(s), the joints cannot be generated: {details}")


def finalize_joint_struct(JCS: dict, joint_defs: str = "auto2020") -> dict:
    """finalizeJointStruct.m (= first part of createOpenSimModelJoints.m).

    Returns {joint_name: joint struct} in the order the joints are added to the model.
    The input JCS is not modified.
    """
    JCS = {body: dict(joints) for body, joints in JCS.items()}
    JCS["ground"] = {
        "ground_pelvis": {
            "parentName": "ground",
            "parent_location": np.zeros(3),
            "parent_orientation": np.zeros(3),
        }
    }

    joint_struct: dict[str, dict] = {}
    for cur_joint in compile_joint_list(JCS):
        tmp = get_joint_params(cur_joint)
        parent, child = tmp["parentName"], tmp["childName"]

        if parent not in JCS:
            if child not in JCS:
                raise ValueError(f"Incorrect definition of joint {tmp['jointName']}: missing both bones")
            log.info("Partial model detected proximally: connecting %s to ground.", child)
            tmp = get_joint_params("free_to_ground", child)
            old_joint, cur_joint, parent = cur_joint, tmp["jointName"], tmp["parentName"]
            JCS["ground"][cur_joint] = JCS["ground"]["ground_pelvis"]
            JCS[child][cur_joint] = JCS[child][old_joint]

        if child not in JCS:
            if parent in JCS:
                log.info("Partial model detected distally: deleting incomplete joint %s.", cur_joint)
                continue
            raise ValueError(f"Incorrect definition of joint {tmp['jointName']}: missing both bones")

        for joints in JCS.values():  # later bodies overwrite earlier ones, as in MATLAB
            info = joints.get(cur_joint)
            if info is None:
                continue
            for f in LOCATION_FIELDS:
                if f in info:
                    tmp[f] = np.asarray(info[f], dtype=np.float64).ravel()
        joint_struct[cur_joint] = tmp

    if joint_defs == "auto2020":
        joint_struct = joint_definitions_auto2020(JCS, joint_struct)
    elif joint_defs == "Modenese2018":
        raise NotImplementedError("joint_defs 'Modenese2018' not ported yet")
    else:
        raise ValueError(f"unknown joint definitions {joint_defs!r}")

    joint_struct = assemble_joint_struct(joint_struct)
    verify_joint_struct_completeness(joint_struct)
    return joint_struct
