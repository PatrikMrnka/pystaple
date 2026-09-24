"""Planar sections and hole filling of triangulated meshes.

Ports of GIBOC-core functions:
    TriPlanIntersect.m, TriFillPlanarHoles.m, PlanPolygonCentroid3D.m,
    TriSliceObjAlongAxis.m, getLargerPlanarSect.m, cutLongBoneMesh.m

The MATLAB code is replicated closely, including details that influence the
numerical results, e.g. section curves are closed by repeating their first
point, and curves are traced in the same order as in MATLAB.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .mesh import TriMesh, tri_inertia_ppties, tri_reduce_mesh


# --- polygon helpers ---------------------------------------------------------


def polyarea(x: np.ndarray, y: np.ndarray) -> float:
    """MATLAB polyarea: absolute area of a (implicitly closed) polygon."""
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def inpolygon(xq: float, yq: float, xv: np.ndarray, yv: np.ndarray) -> bool:
    """Point-in-polygon test (even-odd rule), like MATLAB inpolygon for one point.

    Points exactly on an edge are not treated specially (MATLAB counts them as
    inside); for the section curves used here this case practically never occurs.
    """
    x1, y1 = xv, yv
    x2, y2 = np.roll(xv, -1), np.roll(yv, -1)
    crosses = (y1 > yq) != (y2 > yq)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at_y = x1 + (yq - y1) * (x2 - x1) / (y2 - y1)
    return bool(np.count_nonzero(crosses & (xq < x_at_y)) % 2)


def _eig_cov(pts: np.ndarray) -> np.ndarray:
    """Eigenvectors of cov(pts), ascending eigenvalues (MATLAB eig on symmetric)."""
    return np.linalg.eigh(np.cov(pts, rowvar=False))[1]


def plan_polygon_centroid_3d(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """PlanPolygonCentroid3D.m: centroid and area of a planar 3D polygon."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        warnings.warn("PlanPolygonCentroid3D: empty Pts variable.", stacklevel=2)
        return np.full(3, np.nan), float("nan")
    # MATLAB `if Pts(1,:) ~= Pts(end,:)` is true only if ALL coordinates differ
    if np.all(pts[0] != pts[-1]):
        pts = np.vstack([pts, pts[:1]])
    center0 = pts[:-1].mean(axis=0)
    pts_middle = pts[:-1] + np.diff(pts, axis=0) / 2
    triangles_centroid = pts_middle - (pts_middle - center0) / 3
    n = _eig_cov(pts[:-1])[:, 0]  # normal to the polygon plane
    triangles_area = 0.5 * np.cross(np.diff(pts, axis=0), -(pts[:-1] - center0)) @ n
    centroid = (triangles_centroid * triangles_area[:, None]).sum(axis=0) / triangles_area.sum()
    return centroid, float(abs(triangles_area.sum()))


# --- tracing closed curves from segments ---------------------------------------


def _trace_segments(
    segments: np.ndarray, remove_all_matches: bool = False
) -> list[tuple[list[int], list[int]]]:
    """Chain segments into curves exactly like the MATLAB while-loops.

    Returns a list of (node_ids, traversed_segment_rows) per curve. A closed
    curve ends with its first node repeated, as in MATLAB. When looking for the
    next segment, MATLAB's column-major `find` is replicated: matches in the
    first column win over the second, then the lowest row wins.
    With ``remove_all_matches`` every remaining segment containing the node is
    removed (TriKeepLargestPatch.m), otherwise only the one followed.
    """
    n_seg = len(segments)
    alive = np.ones(n_seg, dtype=bool)
    occurrences: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row, (a, b) in enumerate(segments.tolist()):
        occurrences[a].append((0, row))
        occurrences[b].append((1, row))

    def take_next(node):
        cands = [cr for cr in occurrences[node] if alive[cr[1]]]
        if not cands:
            return None, None
        col, row = min(cands)
        if remove_all_matches:
            for _, r in cands:
                alive[r] = False
        else:
            alive[row] = False
        return int(segments[row, 1 - col]), row

    curves = []
    start = 0
    while True:
        while start < n_seg and not alive[start]:
            start += 1
        if start == n_seg:
            break
        a, b = (int(v) for v in segments[start])
        alive[start] = False
        nodes, rows = [a, b], [start]
        nk, row = take_next(b)
        while nk is not None:
            rows.append(row)
            nodes.append(nk)
            nk, row = take_next(nk)
        curves.append((nodes, rows))
    return curves


# --- TriPlanIntersect -----------------------------------------------------------


@dataclass
class SectionCurve:
    pts: np.ndarray  # (K, 3), closed curves repeat the first point at the end
    area: float
    hole: int  # 1 if hole, -1 if filled (MATLAB convention)


def tri_plan_intersect(mesh: TriMesh, n: np.ndarray, d: float) -> tuple[list[SectionCurve], float]:
    """TriPlanIntersect.m: intersection of the mesh with the plane n.x + d = 0.

    Only the scalar-altitude form of `d` is supported (the only one used here).
    Returns (curves, total_area). With no intersection, one empty curve with
    zero area is returned, as in MATLAB.
    """
    pts = mesh.points
    n = np.asarray(n, dtype=np.float64).ravel()[:3]
    n = n / np.linalg.norm(n)

    # a point on the plane, computed as in MATLAB
    k = int(np.argmax(np.abs(n)))
    pts1 = pts[0].copy()
    op = pts[0].copy()
    pts1[k] = 0.0
    op[k] = (-(pts1 @ n) - d) / n[k]

    signed = pts @ n + d
    over_under = (signed > 0).astype(np.int64) - (signed < 0).astype(np.int64)
    if np.any(over_under == 0):
        # MATLAB calls the non-existent `warnings(...)` here and would crash
        warnings.warn("Points lying exactly on the intersecting plane", stacklevel=2)

    elmts = mesh.faces
    score = over_under[elmts].sum(axis=1)
    elmts_x = elmts[np.abs(score) < 3]
    if len(elmts_x) == 0:
        warnings.warn("No intersection found between the plane and the triangulation", stacklevel=2)
        return [SectionCurve(np.empty((0, 3)), 0.0, 0)], 0.0

    # edges of intersecting faces: (1,2), (2,3), (3,1) for each face, face by face
    edges = np.stack([elmts_x[:, [0, 1]], elmts_x[:, [1, 2]], elmts_x[:, [2, 0]]], axis=1).reshape(-1, 2)
    i_x = np.flatnonzero(over_under[edges].sum(axis=1) == 0)

    p0 = pts[edges[i_x, 0]]
    u = pts[edges[i_x, 1]] - p0
    v = p0 - op
    ratio = (-(v @ n)) / (u @ n)
    pts_inter = p0 + u * ratio[:, None]

    # each intersecting edge and its twin (reversed edge) map to the lower index
    row_of = {(int(a), int(b)): int(r) for r, (a, b) in zip(i_x, edges[i_x])}
    corr = np.array(
        [min(r, row_of.get((int(b), int(a)), r)) for r, (a, b) in zip(i_x, edges[i_x])],
        dtype=np.int64,
    )
    pt_index = np.full(len(edges), -1, dtype=np.int64)
    pt_index[i_x] = np.arange(len(i_x))
    pt_index[i_x] = pt_index[corr]
    segments = pt_index[pt_index >= 0].reshape(-1, 2)

    traced = _trace_segments(segments)

    curve_pts = [pts_inter[nodes] for nodes, _ in traced]
    areas, bases = [], []
    for cp in curve_pts:
        V = _eig_cov(cp)
        planar = cp @ V
        areas.append(polyarea(planar[:, 1], planar[:, 2]))
        bases.append(V)

    n_curves = len(curve_pts)
    inside_count = np.zeros(n_curves, dtype=np.int64)
    for i in range(n_curves):
        first = curve_pts[i][0] @ bases[i]
        for j in range(n_curves):
            if i != j:
                other = curve_pts[j] @ bases[i]
                if inpolygon(first[1], first[2], other[:, 1], other[:, 2]):
                    inside_count[i] += 1

    curves, tot_area = [], 0.0
    for cp, area, cnt in zip(curve_pts, areas, inside_count):
        add_or_subtract = 1 - 2 * (cnt % 2)
        hole = -add_or_subtract
        tot_area -= hole * area
        curves.append(SectionCurve(cp, area, hole))
    return curves, tot_area


def get_larger_planar_sect(curves: list[SectionCurve]) -> tuple[SectionCurve, int]:
    """getLargerPlanarSect.m: the curve with the largest area, and the count."""
    i = int(np.argmax([c.area for c in curves]))
    return curves[i], len(curves)


def matlab_colon(start: float, step: float, stop: float) -> np.ndarray:
    """MATLAB start:step:stop."""
    n = int(np.floor((stop - start) / step + 1e-10)) + 1
    return start + step * np.arange(max(n, 0))


def tri_slice_obj_along_axis(mesh: TriMesh, axis: np.ndarray, step: float, cut_offset: float = 0.5):
    """TriSliceObjAlongAxis.m: section areas along an axis.

    Returns (areas, alt, max_area, max_area_ind, max_alt).
    """
    proj = mesh.points @ axis
    alt = matlab_colon(proj.min() + cut_offset, step, proj.max() - cut_offset)
    areas = np.array([tri_plan_intersect(mesh, axis, -a)[1] for a in alt])
    i_max = int(np.argmax(areas))
    return areas, alt, areas[i_max], i_max, alt[i_max]


# --- hole filling and long bone cutting -------------------------------------------


def tri_fill_planar_holes(mesh: TriMesh) -> TriMesh:
    """TriFillPlanarHoles.m: close each boundary loop with a fan of triangles."""
    fb = mesh.free_boundary_edges()
    if len(fb) == 0:
        return mesh

    new_points = [mesh.points]
    new_faces = [mesh.faces]
    n_nodes = len(mesh.points)
    tri_center = mesh.points.mean(axis=0)

    for nodes, rows in _trace_segments(fb):
        curve_fb = fb[rows]
        hole_center = mesh.points[nodes].mean(axis=0)  # first node counted twice, as in MATLAB
        new_node = n_nodes
        n_nodes += 1
        new_points.append(hole_center[None, :])

        fan = np.column_stack([curve_fb, np.full(len(curve_fb), new_node)])
        all_pts = np.vstack(new_points)
        v1 = all_pts[fan[:, 1]] - all_pts[fan[:, 0]]
        v2 = all_pts[fan[:, 2]] - all_pts[fan[:, 0]]
        normals = np.cross(v1, v2)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        U = hole_center - tri_center
        U /= np.linalg.norm(U)
        if np.mean(normals @ U) < 0:
            fan = np.column_stack([curve_fb[:, 0], np.full(len(curve_fb), new_node), curve_fb[:, 1]])
        new_faces.append(fan)

    return TriMesh(np.vstack(new_points), np.vstack(new_faces))


def cut_long_bone_mesh(mesh: TriMesh, u0: np.ndarray, l_ratio: float = 0.33) -> tuple[TriMesh, TriMesh]:
    """cutLongBoneMesh.m: proximal and distal epiphyses (hole-filled)."""
    Z0 = tri_inertia_ppties(mesh).eig_vectors[:, 0]
    Z0 = np.sign(np.asarray(u0).ravel() @ Z0) * Z0
    proj = mesh.points @ Z0
    length = proj.max() - proj.min()
    inc = mesh.incenters() @ Z0

    z_prox = proj.max() - l_ratio * length
    prox = tri_fill_planar_holes(tri_reduce_mesh(mesh, faces_kept=np.flatnonzero(inc > z_prox)))
    z_dist = proj.min() + l_ratio * length
    dist = tri_fill_planar_holes(tri_reduce_mesh(mesh, faces_kept=np.flatnonzero(inc < z_dist)))
    return prox, dist
