"""Segment mass properties, without OpenSim.

Ports of gait2392MassProps.m, mapGait2392MassPropToModel.m, scaleMassProps.m
and of the arithmetic of assignMassPropsToSegments.m.

The final masses and inertias depend only on the gait2392 table and on the
subject mass. From the geometry only the centres of mass remain: the femur
(and tibia / calcaneus, when talus / feet are present) get the COM from
Winter's proportions along the segment, the other bodies keep the COM of the
bone volume.
"""

from __future__ import annotations

import numpy as np

from ..mesh import TriMesh, tri_inertia_ppties

# name: (mass [kg], (Ixx, Iyy, Izz) [kg m^2])
GAIT2392 = {
    "pelvis": (11.777, (0.1028, 0.0871, 0.0579)),
    "femur": (9.3014, (0.1339, 0.0351, 0.1412)),
    "tibia": (3.7075, (0.0504, 0.0051, 0.0511)),
    "talus": (0.1, (0.001, 0.001, 0.001)),
    "calcn": (1.25, (0.0014, 0.0039, 0.0041)),
    "toes": (0.2166, (0.0001, 0.0002, 0.0001)),
    "torso": (34.2366, (1.4745, 0.7555, 1.4314)),  # different from Rajagopal (arms)
    "patella": (0.0862, (2.87e-006, 1.311e-005, 1.311e-005)),
}
GAIT2392_FULL_BODY_MASS = 11.777 + 2 * (9.3014 + 3.7075 + 0.1 + 1.25 + 0.2166 + 0.0862) + 34.2366
BONE_DENSITY = 1420.0  # kg/m^3, addBodiesFromTriGeomBoneSet.m


def gait2392_mass_props(segment_name: str) -> tuple[float, np.ndarray]:
    """gait2392MassProps.m -> (mass, [Ixx, Iyy, Izz])."""
    if segment_name == "full_body":
        return GAIT2392_FULL_BODY_MASS, np.full(3, np.nan)
    key = segment_name if segment_name in ("pelvis", "torso") else segment_name[:-2]
    if key not in GAIT2392:
        raise ValueError(
            f"gait2392MassProps: {segment_name!r} is not a segment of the gait2392 model"
        )
    mass, inertia = GAIT2392[key]
    return mass, np.array(inertia)


def bone_mass_props(mesh: TriMesh, density: float = BONE_DENSITY, in_mm: bool = True) -> dict:
    """addBodyFromTriGeomObj.m: mass properties of a homogeneous bone.

    Returns {"mass", "mass_center" (m), "inertia" (Ixx Iyy Izz Ixy Ixz Iyz, about the COM)}.
    MATLAB uses computeMassProperties_Mirtich1996.m, which has a bug in the products
    of inertia; they are overwritten by assignMassPropsToSegments anyway.
    """
    dim = 0.001 if in_mm else 1.0
    p = tri_inertia_ppties(mesh)
    rho = density * dim**3  # kg per (mesh unit)^3
    I = p.inertia_matrix * rho * dim**2
    return {
        "mass": p.mass * rho,
        "mass_center": p.center_vol * dim,
        "inertia": np.array([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]),
    }


def winter_mass_centers(JCS: dict, side: str) -> dict[str, np.ndarray]:
    """Centres of mass (m) from assignMassPropsToSegments.m (Winter 2015)."""
    femur, tibia, talus, calcn = (f"{b}_{side}" for b in ("femur", "tibia", "talus", "calcn"))
    hip, knee, ankle, toes = (f"{j}_{side}" for j in ("hip", "knee", "ankle", "mtp"))
    out = {}

    def along(proximal, distal, ratio):
        axis = np.asarray(proximal, float) - np.asarray(distal, float)
        length = np.linalg.norm(axis)
        return length * ratio * (axis / length) + np.asarray(distal, float)

    if femur in JCS:
        knee_origin = JCS[femur][knee]["Origin"]
        out[femur] = along(JCS[femur][hip]["Origin"], knee_origin, 0.567) / 1000
        if talus in JCS:
            ankle_origin = JCS[talus][ankle]["Origin"]
            out[tibia] = along(knee_origin, ankle_origin, 0.567) / 1000
            if calcn in JCS:
                out[calcn] = along(ankle_origin, JCS[calcn][toes]["Origin"], 0.5) / 1000
    return out


def segment_mass_props(
    body_names, initial_mass_centers: dict, JCS: dict, subj_mass: float, side: str
) -> dict[str, dict]:
    """Final mass properties of each body, as after assignMassPropsToSegments.m.

    `initial_mass_centers` are the COMs the bodies were created with (m).
    """
    coms = {name: np.asarray(initial_mass_centers[name], float) for name in body_names}
    coms.update({k: v for k, v in winter_mass_centers(JCS, side).items() if k in coms})
    coeff = subj_mass / GAIT2392_FULL_BODY_MASS
    out = {}
    for name in body_names:
        mass, inertia = gait2392_mass_props(name)
        out[name] = {
            "mass": coeff * mass,
            "mass_center": coms[name],
            "inertia": np.r_[inertia * coeff, 0.0, 0.0, 0.0],
        }
    return out
