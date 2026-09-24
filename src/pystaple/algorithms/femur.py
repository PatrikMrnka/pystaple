"""GIBOC femur algorithm (work in progress).

Step 5a: everything up to the femoral head centre
    femur_guess_CS.m, computeTriCoeffMorpho.m, sphereFit.m,
    GIBOC_femur_fitSphere2FemHead.m, first part of GIBOC_femur.m
Step 5b: distal epiphysis and posterior condyles, cylinder fit, frames, landmarks
    GIBOC_isolate_epiphysis.m, GIBOC_femur_ArticSurf.m ('post_condyles'),
    GIBOC_femur_processEpiPhysis.m, GIBOC_femur_getCondyleMostProxPoint.m,
    PtsOnCondylesFemur.m, GIBOC_femur_filterCondyleSurf.m,
    CS_femur_SpheresOnCondyles.m, CS_femur_CylinderOnCondyles.m

Outputs mirror MATLAB:
    BCS = {"CenterVol", "Origin", "InertiaMatrix", "V"}
    JCS = {"hip_<s>":  {"V", "child_location", "child_orientation", "Origin"},
           "knee_<s>": {"V", "parent_location", "parent_orientation", "Origin"}}
    BL  = {"<S>KNE", "<S>MFC", "<S>TRO"}
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np

from ..mesh import TriMesh, normalize_v, tri_inertia_ppties, tri_reduce_mesh
from ..morphology import (
    face_neighbors,
    matlab_round,
    tri_dilate_mesh,
    tri_keep_largest_patch,
    tri_open_mesh,
    tri_unite,
)
from ..sections import cut_long_bone_mesh, tri_fill_planar_holes

log = logging.getLogger(__name__)


def compute_tri_coeff_morpho(mesh: TriMesh) -> float:
    """computeTriCoeffMorpho.m: 0.5 / (mean edge length of an equilateral mesh)."""
    mean_edge_length = np.sqrt(4 / np.sqrt(3) * mesh.face_areas().sum() / len(mesh.faces))
    return 0.5 / mean_edge_length


def sphere_fit(X: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """sphereFit.m: algebraic least-squares sphere -> (center, radius, error)."""
    X = np.asarray(X, dtype=np.float64)
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    dx, dy, dz = x - x.mean(), y - y.mean(), z - z.mean()
    A = np.array(
        [
            [np.mean(x * dx), 2 * np.mean(x * dy), 2 * np.mean(x * dz)],
            [0.0, np.mean(y * dy), 2 * np.mean(y * dz)],
            [0.0, 0.0, np.mean(z * dz)],
        ]
    )
    A = A + A.T
    sq = x**2 + y**2 + z**2
    B = np.array([np.mean(sq * dx), np.mean(sq * dy), np.mean(sq * dz)])
    center = np.linalg.solve(A, B)
    d2 = np.sum((X - center) ** 2, axis=1)
    radius = float(np.sqrt(np.mean(d2)))
    return center, radius, d2 - radius**2


def femur_guess_cs(femur: TriMesh) -> np.ndarray:
    """femur_guess_CS.m: distal-to-proximal axis (epiphysis far from the shaft axis
    is the proximal one, because of the femoral neck)."""
    inertia = tri_inertia_ppties(femur)
    V_all, center_vol = inertia.eig_vectors, inertia.center_vol
    Z0 = V_all[:, 0]

    # deform the femur (x2 along the 2nd inertial axis) to accentuate the neck
    pts = femur.points - center_vol
    deformed = (pts @ V_all) @ np.diag([1.0, 2.0, 1.0]) @ V_all.T
    femur_d = TriMesh(deformed + center_vol, femur.faces)

    epi1, epi2 = cut_long_bone_mesh(femur_d, Z0, 0.10)

    proj = femur_d.points @ Z0
    length = proj.max() - proj.min()
    l_ratio = 0.20
    alt_top = proj.max() - l_ratio * length
    tmp1 = tri_fill_planar_holes(
        tri_reduce_mesh(femur_d, faces_kept=np.flatnonzero(femur_d.incenters() @ Z0 < alt_top))
    )
    alt_bottom = proj.min() + l_ratio * length
    diaphysis = tri_fill_planar_holes(
        tri_reduce_mesh(tmp1, faces_kept=np.flatnonzero(tmp1.incenters() @ Z0 > alt_bottom))
    )

    dia = tri_inertia_ppties(diaphysis)
    Z0_dia, center_dia = dia.eig_vectors[:, 0], dia.center_vol
    c1 = tri_inertia_ppties(epi1).center_vol
    c2 = tri_inertia_ppties(epi2).center_vol
    d1 = np.linalg.norm(np.cross(c1 - center_dia, Z0_dia))
    d2 = np.linalg.norm(np.cross(c2 - center_dia, Z0_dia))

    if d1 < d2:
        u_dist_to_prox = c2 - c1
    elif d1 > d2:
        u_dist_to_prox = c1 - c2
    else:
        raise RuntimeError("femur_guess_CS: cannot tell the proximal epiphysis")
    Z0 = np.sign(u_dist_to_prox @ Z0) * Z0

    if abs(d1 - d2) / (d1 + d2) < 0.20:
        warnings.warn(
            "The distance to the femur diaphysis axis for the femur epiphyses were not "
            "very different. Orientation of Z0 could be incorrect.",
            stacklevel=2,
        )
    return Z0


def _top_patch(prox_fem: TriMesh, direction: np.ndarray, coeff_morpho: float) -> TriMesh:
    """Face most extreme along `direction`, its neighbours, dilated by 40*coeff."""
    i_top = int(np.argmax(prox_fem.incenters() @ direction))
    ids = np.r_[i_top, face_neighbors(prox_fem)[i_top]]
    ids = ids[ids >= 0]
    face_top = tri_reduce_mesh(prox_fem, faces_kept=np.unique(ids))
    return tri_dilate_mesh(prox_fem, face_top, 40 * coeff_morpho)


def giboc_femur_fit_sphere_to_head(prox_fem: TriMesh, aux: dict, coeff_morpho: float):
    """GIBOC_femur_fitSphere2FemHead.m -> (aux updated, femoral head mesh).

    Adds aux["Y0"], aux["CenterFH_Renault"], aux["RadiusFH_Renault"].
    """
    Z0, center_vol = aux["Z0"], aux["CenterVol"]

    patch_top = _top_patch(prox_fem, Z0, coeff_morpho)
    OT = patch_top.points.mean(axis=0) - center_vol
    aux["Y0"] = normalize_v(np.cross(np.cross(Z0, OT), Z0))

    patch_mm = _top_patch(prox_fem, aux["Y0"], coeff_morpho)
    fem_head0 = tri_unite(patch_mm, patch_top)

    # fit 1: sphere on the two patches
    center, radius, err = sphere_fit(fem_head0.points)
    sph_rmse = np.mean(np.abs(err))
    log.info("Fit #1: RMSE %.4g", sph_rmse)

    # fit 2: on the dilated patches
    dilated = tri_dilate_mesh(prox_fem, fem_head0, matlab_round(1.5 * radius * coeff_morpho))
    center_fh, _, _ = sphere_fit(dilated.points)

    # keep faces whose normal is radial and that lie close to the first sphere
    inc = dilated.incenters()
    radial = (inc - center_fh) / np.linalg.norm(inc - center_fh, axis=1, keepdims=True)
    cond1 = np.sum(radial * dilated.face_normals(), axis=1) > 0.975
    cond2 = np.abs(np.linalg.norm(inc - center_fh, axis=1) - radius) < 0.1 * radius

    if np.count_nonzero(cond1 & cond2) > 20:
        fem_head = tri_reduce_mesh(dilated, faces_kept=np.flatnonzero(cond1 & cond2))
    else:
        fem_head = tri_keep_largest_patch(tri_reduce_mesh(dilated, faces_kept=np.flatnonzero(cond1)))

    fem_head = tri_open_mesh(prox_fem, fem_head, 3 * coeff_morpho)

    # fit 3: final
    center_fh, radius_fh, _ = sphere_fit(fem_head.points)
    if sph_rmse > 25:
        warnings.warn(f"Large sphere fit RMSE: {sph_rmse:.4g} (>25 mm).", stacklevel=2)

    aux["CenterFH_Renault"] = center_fh
    aux["RadiusFH_Renault"] = radius_fh
    return aux, fem_head


@dataclass
class FemurHeadStage:
    """Intermediate results of GIBOC_femur up to the femoral head."""

    prox_fem: TriMesh
    dist_fem: TriMesh
    coeff_morpho: float
    inertia_matrix: np.ndarray
    fem_head: TriMesh
    aux: dict  # AuxCSInfo: CenterVol, V_all, Z0, Y0, X0, CenterFH_Renault, RadiusFH_Renault


def giboc_femur_head_stage(femur: TriMesh) -> FemurHeadStage:
    """First part of GIBOC_femur.m (up to 'X0 points backwards')."""
    u_dist_to_prox = femur_guess_cs(femur)
    prox_fem, dist_fem = cut_long_bone_mesh(femur, u_dist_to_prox)
    coeff_morpho = compute_tri_coeff_morpho(femur)

    inertia = tri_inertia_ppties(femur)
    aux = {"CenterVol": inertia.center_vol, "V_all": inertia.eig_vectors}
    Z0 = inertia.eig_vectors[:, 0]
    Z0 = np.sign((prox_fem.points.mean(axis=0) - dist_fem.points.mean(axis=0)) @ Z0) * Z0
    aux["Z0"] = Z0

    try:
        aux, fem_head = giboc_femur_fit_sphere_to_head(prox_fem, aux, coeff_morpho)
    except Exception as exc:
        # MATLAB falls back to Kai2014_femur_fitSphere2FemHead here (not ported yet)
        raise NotImplementedError(
            "GIBOC femoral head fit failed; the Kai2014 fallback is not ported yet"
        ) from exc

    aux["X0"] = np.cross(aux["Y0"], aux["Z0"])
    return FemurHeadStage(prox_fem, dist_fem, coeff_morpho, inertia.inertia_matrix, fem_head, aux)


# =============================================================================
# Step 5b: distal epiphysis, posterior condyles, cylinder fit
# =============================================================================

from scipy.spatial import cKDTree  # noqa: E402

from ..curvature import tri_curvature  # noqa: E402
from ..fitting import fit_csa, fit_ellipse, lscylinder, lsplane  # noqa: E402
from ..geometry import (  # noqa: E402
    convex_hull_2d_polygon,
    inpolygon_many,
    largest_edge_conv_hull,
    pc_region_growing,
)
from ..landmarks import landmark_bone_geom  # noqa: E402
from ..mesh import mesh_2d_center  # noqa: E402
from ..morphology import (  # noqa: E402
    tri_close_mesh,
    tri_connected_patch,
    tri_difference_mesh,
)
from ..sections import tri_slice_obj_along_axis  # noqa: E402
from ..utils import body_side_to_sign, compute_xyz_angle_seq  # noqa: E402


def giboc_isolate_epiphysis(tri: TriMesh, Z0: np.ndarray, which: str, z_epi: float | None = None):
    """GIBOC_isolate_epiphysis.m -> (epiphysis mesh, Zepi, altitudes, areas).

    `z_epi` can be given to bypass the fit (used for diagnostics).
    """
    areas, alt, *_ = tri_slice_obj_along_axis(tri, Z0, 1.0)
    if z_epi is None:
        z_epi, _ = fit_csa(alt, areas)
    inc = tri.incenters() @ Z0
    keep = inc > z_epi if which == "proximal" else inc < z_epi
    return tri_reduce_mesh(tri, faces_kept=np.flatnonzero(keep)), z_epi, alt, areas


def giboc_femur_process_epiphysis(epi: TriMesh, aux: dict, edge_threshold=0.5, axes_dev_thresh=0.75):
    """GIBOC_femur_processEpiPhysis.m -> (IdCdlPts (N, 2), U_Axes (N, 3), med_lat_ind)."""
    pairs, lengths = largest_edge_conv_hull(epi.points)
    border = np.unique(epi.free_boundary_edges())
    n_target = int(np.count_nonzero(lengths > edge_threshold * lengths[0]))

    kept, i = [], -1
    while len(kept) != n_target:
        i += 1
        if i >= len(pairs):
            raise RuntimeError("processEpiPhysis: ran out of convex hull edges")
        if not np.isin(pairs[i], border).any():
            kept.append(i)
    id_cdl = pairs[kept]

    axes = epi.points[id_cdl[:, 0]] - epi.points[id_cdl[:, 1]]
    ok = ~(axes @ aux["Y0"] < 0)  # removes the duplicate (reversed) edges
    id_cdl, axes = id_cdl[ok], axes[ok]
    u_axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)

    ok = ~(np.abs(u_axes @ aux["V_all"][:, 1]) < axes_dev_thresh)
    id_cdl, u_axes = id_cdl[ok], u_axes[ok]

    good = pc_region_growing(u_axes, normalize_v(u_axes.mean(axis=0)), 0.1)
    lia = np.isin(_rows(u_axes), _rows(good))
    id_cdl, u_axes = id_cdl[lia], u_axes[lia]

    orientation = matlab_round(np.mean(np.sign(u_axes @ aux["Y0"])))
    if orientation < 0:
        warnings.warn("Unexpected orientation of the lateral->medial condyle axes.", stacklevel=2)
        return id_cdl, u_axes, (1, 0)
    return id_cdl, u_axes, (0, 1)


def _rows(a):
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def giboc_femur_get_condyle_most_prox_point(epi, aux, trace, U):
    """GIBOC_femur_getCondyleMostProxPoint.m."""
    p_centr, normal = lsplane(trace)
    d = -p_centr @ normal
    on_plan = np.flatnonzero(
        (np.abs(epi.points @ normal + d) < 2.5)
        & (epi.points @ aux["Z0"] > np.max(trace @ aux["Z0"] - 2.5))
    )
    near = cKDTree(epi.points).query_ball_point(trace, 7.5)
    near = np.unique(np.concatenate([np.asarray(n, dtype=np.int64) for n in near]))
    iok = np.intersect1d(on_plan, near)
    i_max = int(np.argmax(epi.vertex_normals()[iok] @ U))
    return epi.points[iok[i_max]]


def pts_on_condyles_femur(pts_condyle_0, pts_epi, cut_angle, inset_ratio, dilat_fact):
    """PtsOnCondylesFemur.m: epiphysis points on the articular surface of a condyle.

    Works in the (z, x) plane of the condyle frame, as the MATLAB code.
    """
    elps = fit_ellipse(pts_condyle_0[:, 2], pts_condyle_0[:, 0])
    Ux = np.array([np.cos(elps.phi), -np.sin(elps.phi)])
    Uy = np.array([np.sin(elps.phi), np.cos(elps.phi)])
    R = np.column_stack([Ux, Uy])

    theta = np.linspace(0, 2 * np.pi, 36)
    ex = elps.X0 + inset_ratio * elps.a * np.cos(theta)
    ey = elps.Y0 + inset_ratio * elps.b * np.sin(theta)
    ellipse = (R @ np.vstack([ex, ey])).T
    zq, xq = pts_epi[:, 2], pts_epi[:, 0]
    out_elps = ~inpolygon_many(zq, xq, ellipse[:, 0], ellipse[:, 1])

    K = convex_hull_2d_polygon(pts_condyle_0[:, [2, 0]])
    hz = pts_condyle_0[K, 2] + dilat_fact * (pts_condyle_0[K, 2] - elps.X0_in)
    hx = pts_condyle_0[K, 0] + dilat_fact * (pts_condyle_0[K, 0] - elps.Y0_in)
    in_ch = inpolygon_many(zq, xq, hz, hx)

    rel = pts_epi[:, [2, 0]] - np.array([elps.X0_in, elps.Y0_in])
    sqrd = (rel @ Ux) ** 2 + (rel @ Uy) ** 2
    i_far = int(np.argmax(sqrd))
    u = rel / np.linalg.norm(rel, axis=1, keepdims=True)
    thr = np.cos(np.deg2rad(90 - cut_angle))
    if rel[i_far] @ Uy < 0:
        ext_post = (u @ Uy < -thr) | ((u @ Uy < 0) & (u @ Ux > 0))
    else:
        ext_post = (u @ Uy > thr) | ((u @ Uy > 0) & (u @ Ux > 0))

    return pts_epi[out_elps & in_ch & ~ext_post]


def giboc_femur_filter_condyle_surf(epi, aux, pts_condyle, pts_0_c, coeff_morpho, rng):
    """GIBOC_femur_filterCondyleSurf.m: clean the condyle surface using curvature,
    normal orientation and connectivity."""
    Y1, Z0 = aux["Y1"], aux["Z0"]
    center, _, _ = sphere_fit(pts_condyle)
    condyle = tri_reduce_mesh(epi, nodes_kept=pts_condyle)
    condyle = tri_close_mesh(epi, condyle, 4 * coeff_morpho)

    c_mean, c_gauss, _, _ = tri_curvature(condyle, rng)
    curvtr = np.sqrt(4 * c_mean**2 - 2 * c_gauss)

    cyl = condyle.points - center
    ui = cyl - np.outer(cyl @ Y1, Y1)
    ui /= np.linalg.norm(ui, axis=1, keepdims=True)
    vn = condyle.vertex_normals()
    alpha = np.abs(90 - np.rad2deg(np.arccos(np.clip(np.sum(vn * ui, axis=1), -1, 1))))
    gamma = np.rad2deg(np.arccos(np.clip(vn @ Z0, -1, 1)))

    p_angle = 1 / (1 + np.exp((alpha - 50) / 10))
    p_angle /= p_angle.max()
    p_curv = 1 / (1 + np.exp(-(curvtr - 0.25) / 0.05))
    p_curv /= p_curv.max()
    p_up = 1 / (1 + np.exp((gamma - 45) / 15))
    p_up /= p_up.max()
    p_edge = 0.6 * np.sqrt(p_angle * p_curv) + 0.05 * p_curv + 0.15 * p_angle + 0.2 * p_up

    edges = tri_reduce_mesh(condyle, nodes_kept=np.flatnonzero(p_curv * p_angle > 0.5))
    end = tri_reduce_mesh(condyle, nodes_kept=np.flatnonzero(p_edge < 0.20))
    end = tri_connected_patch(end, pts_0_c)
    end = tri_close_mesh(epi, end, 10 * coeff_morpho)
    end = tri_keep_largest_patch(end)
    return tri_difference_mesh(end, edges)


def giboc_femur_artic_surf_post(epi: TriMesh, aux: dict, coeff_morpho: float, rng):
    """GIBOC_femur_ArticSurf.m with art_surface='post_condyles'
    -> (medial condyle, lateral condyle). Adds X1, Y1, Z1, PtNotch to aux."""
    cut_lat, cut_med, inset, dilat = 10, 25, 0.6, 0.025
    edge_threshold = 0.5
    Z0 = aux["Z0"]

    id_cdl, u_axes, (i_med, i_lat) = giboc_femur_process_epiphysis(epi, aux, edge_threshold, 0.75)
    pts_med = epi.points[id_cdl[:, i_med]]
    pts_lat = epi.points[id_cdl[:, i_lat]]

    Y1 = normalize_v(u_axes.sum(axis=0))
    X1 = normalize_v(np.cross(Y1, Z0))
    Z1 = np.cross(X1, Y1)
    VC = np.column_stack([X1, Y1, Z1])

    n_in = int(np.ceil(len(id_cdl) * edge_threshold))
    mid_post = np.mean(0.5 * (pts_med[:n_in] + pts_lat[:n_in]), axis=0)
    mid_epi = epi.points.mean(axis=0)
    X1 = np.sign((mid_epi - mid_post) @ X1) * X1
    U = normalize_v(3 * Z0 - X1)

    top_med = giboc_femur_get_condyle_most_prox_point(epi, aux, pts_med, U)
    top_lat = giboc_femur_get_condyle_most_prox_point(epi, aux, pts_lat, U)

    pt_axis_proj = np.mean(0.5 * (pts_med + pts_lat), axis=0) @ VC
    epi_rc = epi.points @ VC
    proj_lat = np.vstack([pts_lat, top_lat, top_lat]) @ VC
    proj_med = np.vstack([pts_med, top_med, top_med]) @ VC
    pts_0_c1 = proj_lat @ VC.T
    pts_0_c2 = proj_med @ VC.T
    c1_pts = epi_rc[epi_rc[:, 1] - pt_axis_proj[1] < 0]
    c2_pts = epi_rc[epi_rc[:, 1] - pt_axis_proj[1] > 0]

    art_lat = pts_on_condyles_femur(proj_lat, c1_pts, cut_lat, inset, dilat) @ VC.T
    art_med = pts_on_condyles_femur(proj_med, c2_pts, cut_med, inset, dilat) @ VC.T

    mid_post2 = np.vstack([art_lat, art_med]).mean(axis=0)
    X1 = np.sign((mid_epi - mid_post2) @ X1) * X1
    U = normalize_v(-Z0 - 3 * X1)
    nodes_ok = epi.points[epi.vertex_normals() @ U > 0.98]
    U = normalize_v(Z0 - 3 * X1)
    pt_notch = nodes_ok[int(np.argmin(nodes_ok @ U))]

    aux.update(PtNotch=pt_notch, X1=X1, Y1=Y1, Z1=Z1)

    notch = pt_notch @ X1
    art_lat = art_lat[~(art_lat @ X1 > notch)]
    art_med = art_med[~(art_med @ X1 > notch)]
    pts_0_c1 = pts_0_c1[~(pts_0_c1 @ X1 > notch)]
    pts_0_c2 = pts_0_c2[~(pts_0_c2 @ X1 > notch)]

    # MATLAB order: lateral first, then medial (matters for the rand() stream)
    lat = giboc_femur_filter_condyle_surf(epi, aux, art_lat, pts_0_c1, coeff_morpho, rng)
    med = giboc_femur_filter_condyle_surf(epi, aux, art_med, pts_0_c2, coeff_morpho, rng)
    return med, lat


def cs_femur_spheres_on_condyles(cond_lat, cond_med, aux, side):
    """CS_femur_SpheresOnCondyles.m (only what the cylinder method needs)."""
    side_sign, _ = body_side_to_sign(side)
    c_lat, r_lat, _ = sphere_fit(cond_lat.points)
    c_med, r_med, _ = sphere_fit(cond_med.points)
    aux.update(sphere_center_lat=c_lat, sphere_radius_lat=r_lat,
               sphere_center_med=c_med, sphere_radius_med=r_med)
    Z_knee = normalize_v(c_lat - c_med) * side_sign
    return c_lat, c_med, r_lat, r_med, Z_knee


def cs_femur_cylinder_on_condyles(cond_lat, cond_med, aux, side, in_mm=True, tolp=0.001, tolg=0.001):
    """CS_femur_CylinderOnCondyles.m -> JCS with hip_<s> and knee_<s>."""
    dim_fact = 0.001 if in_mm else 1.0
    _, s = body_side_to_sign(side)
    pts = np.vstack([cond_lat.points, cond_med.points])

    c_lat, c_med, r_lat, r_med, z_dir = cs_femur_spheres_on_condyles(cond_lat, cond_med, aux, side)
    x0n, an, rn = lscylinder(pts, 0.5 * (c_lat + c_med), c_lat - c_med, 0.5 * (r_lat + r_med), tolp, tolg)
    Y2 = normalize_v(an)

    on_axis_lat = x0n + ((mesh_2d_center(cond_lat) - x0n) @ Y2) * Y2
    on_axis_med = x0n + ((mesh_2d_center(cond_med) - x0n) @ Y2) * Y2
    knee_center = 0.5 * on_axis_lat + 0.5 * on_axis_med
    aux.update(Cyl_Y=Y2, Cyl_Pt=x0n, Cyl_Radius=rn, Cyl_Range=float(np.ptp((pts - x0n) @ Y2)))

    center_fh = aux["CenterFH_Renault"]
    Y = normalize_v(center_fh - knee_center)
    Z = normalize_v(np.sign(Y2 @ z_dir) * Y2)
    X = normalize_v(np.cross(Y, Z))
    Zml_hip = normalize_v(np.cross(X, Y))
    V_hip = np.column_stack([X, Y, Zml_hip])
    Y_knee = normalize_v(np.cross(Z, X))
    V_knee = np.column_stack([X, Y_knee, Z])

    return {
        f"hip_{s}": {
            "V": V_hip,
            "child_location": center_fh * dim_fact,
            "child_orientation": compute_xyz_angle_seq(V_hip),
            "Origin": center_fh,
        },
        f"knee_{s}": {
            "V": V_knee,
            "parent_location": knee_center * dim_fact,
            "parent_orientation": compute_xyz_angle_seq(V_knee),
            "Origin": knee_center,
        },
    }


def giboc_femur(
    femur: TriMesh,
    side: str = "r",
    fit_method: str = "cylinder",
    in_mm: bool = True,
    z_epi: float | None = None,
):
    """GIBOC_femur.m with the cylinder method -> (BCS, JCS, BL, aux).

    `aux` holds intermediate results (AuxCSInfo in MATLAB) and, under keys
    starting with "_", intermediate meshes, useful for debugging.
    `z_epi` overrides the fitted start of the distal epiphysis (diagnostics).
    """
    if fit_method != "cylinder":
        raise NotImplementedError(f"fit_method {fit_method!r} not ported (only 'cylinder')")
    _, s = body_side_to_sign(side)
    log.info("GIBOC - FEMUR | side: %s | fit: %s", s.upper(), fit_method)

    stage = giboc_femur_head_stage(femur)
    aux = stage.aux
    epi, aux["Zepi"], aux["_slice_alt"], aux["_slice_areas"] = giboc_isolate_epiphysis(
        stage.dist_fem, aux["Z0"], "distal", z_epi
    )
    aux["_EpiFem"] = epi

    # MATLAB also extracts the full condyles first; that does not change any
    # output of the cylinder method, so it is skipped here.
    rng = np.random.RandomState(5489)  # = MATLAB rng(0), used by TriCurvature
    post_med, post_lat = giboc_femur_artic_surf_post(epi, aux, stage.coeff_morpho, rng)
    aux["_postCondyle_Med"], aux["_postCondyle_Lat"] = post_med, post_lat

    JCS = cs_femur_cylinder_on_condyles(post_lat, post_med, aux, side, in_mm)
    BCS = {
        "CenterVol": aux["CenterVol"],
        "Origin": aux["CenterFH_Renault"],
        "InertiaMatrix": stage.inertia_matrix,
        "V": JCS[f"hip_{s}"]["V"],
    }
    BL = landmark_bone_geom(femur, BCS, f"femur_{s}")
    return BCS, JCS, BL, aux
