"""Morphological operations on triangulated surfaces.

Ports of GIBOC-core: TriDilateMesh.m, TriErodeMesh.m, TriOpenMesh.m,
TriCloseMesh.m, TriUnite.m, TriConnectedPatch.m, TriKeepLargestPatch.m,
TriDifferenceMesh.m, and of the triangulation method ``neighbors``.

The MATLAB growth loops (``ElmtsInitial = neighbors(ElmtsInitial)``) are
replicated literally rather than replaced by a BFS, to keep identical results.
"""

from __future__ import annotations

import numpy as np

from .mesh import TriMesh, tri_reduce_mesh
from .sections import _trace_segments


# --- adjacency ------------------------------------------------------------------


def face_neighbors(mesh: TriMesh) -> np.ndarray:
    """(M, 3) array of edge-adjacent faces, -1 where there is none.

    Column j is the neighbour across the edge opposite vertex j, as in MATLAB
    ``neighbors``. For non-manifold edges one of the other faces is used.
    The result is cached on the (immutable) mesh object.
    """
    cached = mesh.__dict__.get("_face_neighbors")
    if cached is not None:
        return cached
    faces = mesh.faces
    n_faces = len(faces)
    # edge opposite vertex j
    opp = np.stack([faces[:, [1, 2]], faces[:, [2, 0]], faces[:, [0, 1]]], axis=1)
    edges = np.sort(opp.reshape(-1, 2), axis=1)
    _, edge_id = np.unique(edges, axis=0, return_inverse=True)
    edge_id = edge_id.ravel()
    order = np.argsort(edge_id, kind="stable")
    sorted_ids = edge_id[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    counts = np.diff(np.r_[starts, len(order)])

    neighbors = np.full(3 * n_faces, -1, dtype=np.int64)
    slot_face = np.repeat(np.arange(n_faces), 3)
    pair = counts == 2  # manifold edges, vectorised
    a, b = order[starts[pair]], order[starts[pair] + 1]
    neighbors[a], neighbors[b] = slot_face[b], slot_face[a]
    for s, c in zip(starts[counts > 2], counts[counts > 2]):  # non-manifold edges
        group = order[s : s + c]
        neighbors[group] = slot_face[np.roll(group, -1)]
    neighbors = neighbors.reshape(n_faces, 3)
    object.__setattr__(mesh, "_face_neighbors", neighbors)
    return neighbors


def _neighbors_of(neighbors: np.ndarray, face_ids: np.ndarray) -> np.ndarray:
    nb = neighbors[face_ids].ravel()
    return np.unique(nb[nb >= 0])


def _grow(neighbors: np.ndarray, faces_ok: np.ndarray, n_steps: int) -> np.ndarray:
    """MATLAB loop: ElmtsInitial = neighbors(ElmtsInitial); ElmtsOK = union(...)."""
    current = faces_ok
    for _ in range(n_steps):
        current = _neighbors_of(neighbors, current)
        faces_ok = np.union1d(faces_ok, current)
    return faces_ok


def _row_view(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def shared_point_ids(sup: TriMesh, sub_points: np.ndarray) -> np.ndarray:
    """intersect(sup.Points, sub_points, 'rows', 'stable') -> indices into sup."""
    v_sup = _row_view(sup.points)
    mask = np.isin(v_sup, _row_view(np.asarray(sub_points, dtype=np.float64)))
    idx = np.flatnonzero(mask)
    _, first = np.unique(v_sup[idx], return_index=True)  # first occurrence of each row
    return np.sort(idx[first])


def matlab_round(x: float) -> int:
    """MATLAB round (half away from zero); Python's round is half-to-even."""
    return int(np.sign(x) * np.floor(abs(x) + 0.5))


# --- morphology -------------------------------------------------------------------


def tri_unite(tr1: TriMesh, tr2: TriMesh) -> TriMesh:
    """TriUnite.m: concatenate two meshes (no merging of shared points)."""
    return TriMesh(np.vstack([tr1.points, tr2.points]), np.vstack([tr1.faces, tr2.faces + len(tr1.points)]))


def tri_dilate_mesh(sup: TriMesh, tr_in: TriMesh, n_elmts: float) -> TriMesh:
    """TriDilateMesh.m: grow tr_in by n face-adjacency steps on the surface sup."""
    n = int(np.ceil(n_elmts))
    start = sup.faces_attached_to(shared_point_ids(sup, tr_in.points))
    faces_ok = _grow(face_neighbors(sup), start, n)
    return tri_reduce_mesh(sup, faces_kept=faces_ok)


def tri_erode_mesh(tr_in: TriMesh, n_elmts: float) -> TriMesh:
    """TriErodeMesh.m: remove n rows of faces along the free boundary."""
    n = int(np.ceil(n_elmts))
    border_nodes = np.unique(tr_in.free_boundary_edges())
    border = tr_in.faces_attached_to(border_nodes)
    if n > 1:
        border = _grow(face_neighbors(tr_in), border, n - 1)
    keep = np.ones(len(tr_in.faces), dtype=bool)
    keep[border] = False
    return tri_reduce_mesh(tr_in, faces_kept=np.flatnonzero(keep))


def tri_open_mesh(sup: TriMesh, tr_in: TriMesh, n_elmts: float) -> TriMesh:
    """TriOpenMesh.m: erosion followed by dilation."""
    return tri_dilate_mesh(sup, tri_erode_mesh(tr_in, n_elmts), n_elmts)


def tri_close_mesh(sup: TriMesh, tr_in: TriMesh, n_elmts: float) -> TriMesh:
    """TriCloseMesh.m: dilation followed by erosion."""
    return tri_erode_mesh(tri_dilate_mesh(sup, tr_in, n_elmts), n_elmts)


def tri_difference_mesh(tr1: TriMesh, tr2: TriMesh) -> TriMesh:
    """TriDifferenceMesh.m: remove from tr1 the faces touching points of tr2."""
    ia = shared_point_ids(tr1, tr2.points)
    if len(ia) == 0:
        ia = shared_point_ids(TriMesh(np.round(tr1.points, 5), tr1.faces), np.round(tr2.points, 5))
    keep = np.ones(len(tr1.faces), dtype=bool)
    keep[tr1.faces_attached_to(ia)] = False
    return tri_reduce_mesh(tr1, faces_kept=np.flatnonzero(keep))


def tri_connected_patch(tr: TriMesh, pts_initial: np.ndarray) -> TriMesh:
    """TriConnectedPatch.m: faces connected to the vertices nearest to pts_initial."""
    nodes = np.unique(tr.nearest_vertex(pts_initial))
    neighbors = face_neighbors(tr)
    current = tr.faces_attached_to(nodes)
    connected = current
    while True:
        previous = len(connected)
        current = _neighbors_of(neighbors, current)
        connected = np.union1d(connected, current)
        if len(connected) == previous:
            break
    return tri_reduce_mesh(tr, faces_kept=connected)


def tri_keep_largest_patch(tr_in: TriMesh) -> TriMesh:
    """TriKeepLargestPatch.m: keep the patch with the largest area."""
    tr2 = tri_erode_mesh(tr_in, 1)
    segments = tr2.free_boundary_edges()
    curves = _trace_segments(segments, remove_all_matches=True) if len(segments) else []
    if len(curves) > 1:
        best, best_area = None, -np.inf
        for nodes, _ in curves:
            patch = tri_connected_patch(tr_in, tr2.points[nodes])
            area = patch.face_areas().sum()
            if area > best_area:  # first maximum wins, as MATLAB max
                best, best_area = patch, area
        return best
    return tri_dilate_mesh(tr_in, tr2, 1)
