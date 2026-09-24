"""STAPLE pelvis algorithm.

Port of STAPLE_pelvis.m, private/pelvis_guess_CS.m and CS_pelvis_ISB.m.

The outputs mirror the MATLAB structures as dictionaries with the same keys,
so they can be compared directly with reference.json and later passed to the
OpenSim layer:

    BCS = {"CenterVol", "Origin", "InertiaMatrix", "V"}
    JCS = {"ground_pelvis": {"V", "Origin", "child_location", "child_orientation"},
           "hip_<side>":    {"parent_orientation"}}
    BL  = {"RASI", "LASI", "RPSI", "LPSI", "SYMP"}
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from ..mesh import (
    TriMesh,
    convex_hull_mesh,
    normalize_v,
    tri_change_cs,
    tri_inertia_ppties,
    tri_reduce_mesh,
)
from ..utils import body_side_to_sign, compute_xyz_angle_seq

log = logging.getLogger(__name__)


def pelvis_guess_cs(pelvis: TriMesh) -> tuple[np.ndarray, TriMesh, dict]:
    """pelvis_guess_CS.m: initial ("pseudo-ISB") orientation of the pelvis.

    Returns (RotPseudoISB2Glob, LargestTriangle, BL) where the rotation matrix
    has the X (anterior), Y (cranial), Z (right) axes as columns.
    """
    inertia = tri_inertia_ppties(pelvis)
    V_all, center_vol = inertia.eig_vectors, inertia.center_vol

    # smallest moment of inertia: normally the medio-lateral axis
    Z0 = V_all[:, 0]

    # largest triangle of the convex hull connects the ASISs and the pubis
    hull = convex_hull_mesh(pelvis.points)
    i_max = int(np.argmax(hull.face_areas()))
    largest_triangle = TriMesh(hull.points[hull.faces[i_max]], np.array([[0, 1, 2]]))

    # inertial axis most aligned with the triangle normal points forward
    normal = largest_triangle.face_normals()[0]
    ind_X = int(np.argmax(np.abs(V_all.T @ normal)))
    X0 = V_all[:, ind_X]

    # reorient X0 posterior -> anterior
    anterior_v = largest_triangle.incenters()[0] - center_vol
    X0 = normalize_v(np.sign(anterior_v @ X0) * X0)

    Y0_temp = normalize_v(np.cross(Z0, X0))

    # NOTE: replicated exactly as in MATLAB, including the transpose:
    # TriChangeCS(pelvisTri, [X0, Y0_temp, Z0]', CenterVol)
    pelvis_inertia = tri_change_cs(pelvis, np.column_stack([X0, Y0_temp, Z0]).T, center_vol)
    pts = pelvis_inertia.points

    # candidate points on the iliac crests: largest span along Y or Z
    ind_P1y, ind_P2y = int(np.argmax(pts[:, 1])), int(np.argmin(pts[:, 1]))
    span_y = abs(pts[ind_P1y, 1]) + abs(pts[ind_P2y, 1])
    ind_P1z, ind_P2z = int(np.argmax(pts[:, 2])), int(np.argmin(pts[:, 2]))
    span_z = abs(pts[ind_P1z, 2]) + abs(pts[ind_P2z, 2])
    if span_y > span_z:
        ind_P1, ind_P2 = ind_P1y, ind_P2y
    else:
        ind_P1, ind_P2 = ind_P1z, ind_P2z

    # iliac crest tubercles (ICT) and their midpoint
    P1 = pelvis.points[ind_P1]
    P2 = pelvis.points[ind_P2]
    P3 = (P1 + P2) / 2

    # upward vector, perpendicular to X0
    upw_ini = normalize_v(P3 - center_vol)
    upw = upw_ini - (upw_ini @ X0) * X0

    # inertial axis most aligned with it points cranially
    ind_Z = int(np.argmax(np.abs(V_all.T @ upw)))
    Z0 = V_all[:, ind_Z]
    Z0 = np.sign(upw @ Z0) * Z0

    # GIBOC -> ISB convention: X0 = X_ISB, Z0 = Y_ISB
    rot_pseudo_isb2glob = np.column_stack([X0, Z0, np.cross(X0, Z0)])

    return rot_pseudo_isb2glob, largest_triangle, {"ICT1": P1, "ICT2": P2}


def cs_pelvis_isb(rasis, lasis, rpsis, lpsis) -> np.ndarray:
    """CS_pelvis_ISB.m: ISB pelvis axes (columns X, Y, Z)."""
    Z = normalize_v(rasis - lasis)
    temp_X = (rasis + lasis) / 2.0 - (rpsis + lpsis) / 2.0
    pseudo_X = temp_X / np.linalg.norm(temp_X)
    Y = normalize_v(np.cross(Z, pseudo_X))
    X = normalize_v(np.cross(Y, Z))
    return np.column_stack([X, Y, Z])


def _most_posterior_in_octant(pelvis, pseudo_isb_pts, x_axis, right: bool) -> np.ndarray:
    lat = pseudo_isb_pts[:, 2] > 0 if right else pseudo_isb_pts[:, 2] < 0
    nodes = np.flatnonzero((pseudo_isb_pts[:, 0] < 0) & (pseudo_isb_pts[:, 1] > 0) & lat)
    if nodes.size == 0:
        raise RuntimeError("STAPLE_pelvis: no vertices in the posterior-superior octant")
    region = tri_reduce_mesh(pelvis, nodes_kept=nodes)
    return region.points[int(np.argmin(region.points @ x_axis))]


def staple_pelvis(pelvis: TriMesh, side: str = "r", in_mm: bool = True):
    """STAPLE_pelvis.m: pelvis reference system, joint parameters and landmarks.

    Returns (BCS, JCS, BL) as dictionaries (see module docstring).
    """
    dim_fact = 0.001 if in_mm else 1.0
    _, side_low = body_side_to_sign(side)
    log.info("STAPLE - PELVIS | hip joint: %s | method: convex hull", side_low.upper())

    inertia = tri_inertia_ppties(pelvis)
    center_vol = inertia.center_vol

    rot, largest_triangle, _ = pelvis_guess_cs(pelvis)
    pelvis_pseudo_isb = tri_change_cs(pelvis, rot, center_vol)

    # ASISs and pubic symphysis from the largest convex-hull triangle:
    # the pubis is the vertex closest to the sagittal plane
    lt_pts = largest_triangle.points
    proj = (lt_pts - center_vol) @ rot[:, 2]
    order = np.argsort(np.abs(proj), kind="stable")  # MATLAB sort is stable
    asi_inds = order[1:3]
    ind_r = np.flatnonzero(proj[asi_inds] > 0)
    ind_l = np.flatnonzero(proj[asi_inds] < 0)
    if len(ind_r) != 1 or len(ind_l) != 1:
        raise RuntimeError("STAPLE_pelvis: could not separate right and left ASIS")
    symp = lt_pts[order[0]]
    rasis = lt_pts[asi_inds[ind_r[0]]]
    lasis = lt_pts[asi_inds[ind_l[0]]]

    # PSISs: most posterior points in the posterior-superior right/left octants
    rpsis = _most_posterior_in_octant(pelvis, pelvis_pseudo_isb.points, rot[:, 0], right=True)
    lpsis = _most_posterior_in_octant(pelvis, pelvis_pseudo_isb.points, rot[:, 0], right=False)

    if np.linalg.norm(rasis - lasis) < np.linalg.norm(rpsis - lpsis):
        warnings.warn(
            "Inter-ASIS distance is shorter than inter-PSIS distance. Better check manually.",
            stacklevel=2,
        )

    origin = (rasis + lasis) / 2.0
    V = cs_pelvis_isb(rasis, lasis, rpsis, lpsis)
    angles = compute_xyz_angle_seq(V)

    BCS = {
        "CenterVol": center_vol,
        "Origin": origin,
        "InertiaMatrix": inertia.inertia_matrix,
        "V": V,
    }
    JCS = {
        "ground_pelvis": {
            "V": V,
            "Origin": origin,
            "child_location": origin * dim_fact,
            "child_orientation": angles,
        },
        f"hip_{side_low}": {"parent_orientation": angles},
    }
    BL = {"RASI": rasis, "LASI": lasis, "RPSI": rpsis, "LPSI": lpsis, "SYMP": symp}
    return BCS, JCS, BL
