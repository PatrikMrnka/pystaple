"""Kai2014 tibia algorithm.

Port of Kai2014_tibia.m, private/tibia_guess_CS.m and
private/tibia_identify_lateral_direction.m.

Outputs mirror the MATLAB structures:

    BCS = {"CenterVol", "Origin", "InertiaMatrix", "V"}
    JCS = {"knee_<side>": {"Origin", "V", "child_orientation"}}
    BL  = {"<S>TTB", "<S>HFB", "<S>ANK", "<S>MMA", and "<S>LM" if a fibula is present}
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from ..fitting import fit_ellipse, pca_matlab
from ..landmarks import landmark_bone_geom
from ..mesh import TriMesh, normalize_v, tri_inertia_ppties
from ..sections import (
    cut_long_bone_mesh,
    get_larger_planar_sect,
    plan_polygon_centroid_3d,
    tri_plan_intersect,
    tri_slice_obj_along_axis,
)
from ..utils import body_side_to_sign, compute_xyz_angle_seq

log = logging.getLogger(__name__)


def tibia_guess_cs(tibia: TriMesh) -> np.ndarray:
    """tibia_guess_CS.m: distal-to-proximal axis from where the largest sections are."""
    Z0 = tri_inertia_ppties(tibia).eig_vectors[:, 0]
    proj = tibia.points @ Z0
    alt = np.linspace(proj.min() + 0.5, proj.max() - 0.5, 100)

    areas = np.zeros(len(alt))
    for it, a in enumerate(alt):
        curves, _ = tri_plan_intersect(tibia, Z0, -a)
        max_area = 0.0
        for c in curves:
            _, area_j = plan_polygon_centroid_3d(c.pts)
            if area_j > max_area:  # NaN compares False, as in MATLAB
                max_area = area_j
        areas[it] = max_area

    i_max_area = int(np.argmax(areas)) + 1  # MATLAB 1-based index
    n_it = len(alt)
    if i_max_area > 0.66 * n_it:
        return Z0
    if i_max_area < 0.33 * n_it:
        return -Z0
    warnings.warn(
        "Identification of the initial distal to proximal axis of the tibia went wrong. "
        "Check the tibia geometry",
        stacklevel=2,
    )
    return Z0


def tibia_identify_lateral_direction(dist_tib: TriMesh, Z0: np.ndarray):
    """tibia_identify_lateral_direction.m -> (U_tmp, MostDistalMedialPt, just_tibia)."""
    center_vol = tri_inertia_ppties(dist_tib).center_vol
    d = center_vol @ Z0
    dist_curves, _ = tri_plan_intersect(dist_tib, Z0, -d)

    n_curves = len(dist_curves)
    just_tibia = True
    if n_curves == 2:
        log.info("Tibia and fibula have been detected.")
        just_tibia = False
    elif n_curves > 2:
        raise RuntimeError(
            f"There are {n_curves} section areas. This should not be the case "
            "(only tibia and possibly fibula should be there)."
        )

    most_distal_medial_pt = dist_tib.points[int(np.argmax(dist_tib.points @ -Z0))]

    if just_tibia:
        u_tmp = center_vol - most_distal_medial_pt
    else:
        c1, c2 = dist_curves
        diff = c2.pts.mean(axis=0) - c1.pts.mean(axis=0)
        u_tmp = diff if c1.area > c2.area else -diff
    return u_tmp, most_distal_medial_pt, just_tibia


def kai2014_tibia(tibia: TriMesh, side: str = "r", in_mm: bool = True):
    """Kai2014_tibia.m: tibia reference system, knee child frame and landmarks.

    Returns (BCS, JCS, BL) as dictionaries (see module docstring).
    """
    slices_thickness = 1.0
    side_sign, side_low = body_side_to_sign(side)
    log.info("KAI2014 - TIBIA | side: %s", side_low.upper())

    V_all = pca_matlab(tibia.points)

    u_dist_to_prox = tibia_guess_cs(tibia)
    prox_tib, dist_tib = cut_long_bone_mesh(tibia, u_dist_to_prox)

    inertia = tri_inertia_ppties(tibia)

    # longitudinal axis, pointing proximally
    Y0 = V_all[:, 0]
    Y0 = np.sign((prox_tib.points.mean(axis=0) - dist_tib.points.mean(axis=0)) @ Y0) * Y0

    # largest cross-section of the tibia along its axis
    log.info("Slicing tibia longitudinally...")
    *_, alt_at_max = tri_slice_obj_along_axis(tibia, Y0, slices_thickness)
    curves, _ = tri_plan_intersect(tibia, Y0, -alt_at_max)
    max_area_section, n_curves = get_larger_planar_sect(curves)
    if n_curves > 2:
        raise RuntimeError(
            f"There are {n_curves} section areas at the largest tibial slice. This should "
            "not be the case (only tibia and possibly fibula should be there)."
        )

    # ellipse fitted to that section, in the PCA frame
    pts_curves = max_area_section.pts @ V_all
    ellipse = fit_ellipse(pts_curves[:, 1], pts_curves[:, 2])
    if ellipse.a > ellipse.b:
        z_elps_max = V_all @ np.array([0.0, np.cos(ellipse.phi), -np.sin(ellipse.phi)])
    else:
        z_elps_max = V_all @ np.array([0.0, np.sin(ellipse.phi), np.cos(ellipse.phi)])

    center_ellipse = V_all @ np.array([pts_curves[:, 0].mean(), ellipse.X0_in, ellipse.Y0_in])

    # lateral direction, used only to orient the ellipse major axis
    u_tmp, most_distal_medial_pt, just_tibia = tibia_identify_lateral_direction(dist_tib, Y0)
    u_tmp = side_sign * u_tmp
    z0_temp = normalize_v(u_tmp - (u_tmp @ Y0) * Y0)
    z_elps_max = np.sign(z0_temp @ z_elps_max) * z_elps_max

    Y = normalize_v(Y0)
    Z = normalize_v(z_elps_max)
    X = normalize_v(np.cross(Y, Z))
    Z_cs = normalize_v(np.cross(X, Y))

    BCS = {
        "CenterVol": inertia.center_vol,
        "Origin": center_ellipse,
        "InertiaMatrix": inertia.inertia_matrix,
        "V": np.column_stack([X, Y, Z_cs]),
    }

    Ydp_knee = normalize_v(np.cross(Z, X))
    V_knee = np.column_stack([X, Ydp_knee, Z])
    JCS = {
        f"knee_{side_low}": {
            "Origin": center_ellipse,
            "V": V_knee,
            "child_orientation": compute_xyz_angle_seq(V_knee),
        }
    }

    BL = landmark_bone_geom(tibia, BCS, f"tibia_{side_low}")
    if not just_tibia:
        BL[f"{side_low.upper()}LM"] = most_distal_medial_pt

    return BCS, JCS, BL
