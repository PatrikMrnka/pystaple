"""Bony landmarks: getBoneLandmarkList.m, findLandmarkCoords.m, landmarkBoneGeom.m."""

from __future__ import annotations

import numpy as np

from .mesh import TriMesh, tri_change_cs

# (name, axis, operator, extremity or None)
_LANDMARKS: dict[str, list[tuple[str, str, str, str | None]]] = {
    "pelvis": [
        ("RASI", "x", "max", None), ("LASI", "x", "max", None),
        ("RPSI", "x", "min", None), ("LPSI", "x", "min", None),
    ],
    "femur_r": [("RKNE", "z", "max", "distal"), ("RMFC", "z", "min", "distal"), ("RTRO", "z", "max", "proximal")],
    "femur_l": [("LKNE", "z", "min", "distal"), ("LMFC", "z", "max", "distal"), ("LTRO", "z", "min", "proximal")],
    "tibia_r": [
        ("RTTB", "x", "max", "proximal"), ("RHFB", "z", "max", "proximal"),
        ("RANK", "z", "max", "distal"), ("RMMA", "z", "min", "distal"),
    ],
    "tibia_l": [
        ("LTTB", "x", "max", "proximal"), ("LHFB", "z", "min", "proximal"),
        ("LANK", "z", "min", "distal"), ("LMMA", "z", "max", "distal"),
    ],
    "patella_r": [("RLOW", "y", "min", "distal")],
    "patella_l": [("LLOW", "y", "min", "distal")],
    "calcn_r": [("RHEE", "x", "min", None), ("RD5M", "z", "max", None), ("RD1M", "z", "min", None)],
    "calcn_l": [("LHEE", "x", "min", None), ("LD5M", "z", "min", None), ("LD1M", "z", "max", None)],
}
# NOTE: in MATLAB the pelvis entries have 'z'/'max' as last element, which is
# neither 'proximal' nor 'distal', so the whole geometry is searched (None here).


def get_bone_landmark_list(bone_name: str):
    try:
        return _LANDMARKS[bone_name]
    except KeyError:
        raise ValueError(f"getBoneLandmarkList: bone {bone_name!r} not supported yet") from None


def find_landmark_coords(points: np.ndarray, axis_name: str, operator: str) -> np.ndarray:
    col = "xyz".index(axis_name)
    i = int(np.argmax(points[:, col]) if operator == "max" else np.argmin(points[:, col]))
    return points[i]


def landmark_bone_geom(mesh: TriMesh, cs: dict, bone_name: str) -> dict[str, np.ndarray]:
    """landmarkBoneGeom.m: landmarks as extreme points in the body frame `cs`."""
    V, center = np.asarray(cs["V"]), np.asarray(cs["CenterVol"]).ravel()
    pts = tri_change_cs(mesh, V, center).points
    ub = pts[:, 1].max() * 0.3
    lb = pts[:, 1].max() * (-0.3)  # as in MATLAB, both limits from the max
    landmarks = {}
    for name, axis, op, extremity in get_bone_landmark_list(bone_name):
        if extremity == "proximal":
            subset = pts[pts[:, 1] > ub]
        elif extremity == "distal":
            subset = pts[pts[:, 1] < lb]
        else:
            subset = pts
        landmarks[name] = center + V @ find_landmark_coords(subset, axis, op)
    return landmarks
