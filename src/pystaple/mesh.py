"""Triangulated surface meshes and basic operations.

Port of the MATLAB ``triangulation`` object methods used by STAPLE and of
several GIBOC-core functions (TriInertiaPpties, TriChangeCS, TriReduceMesh,
TriMesh2DProperties, normalizeV).

Conventions used across the whole package:
    * points are float64 arrays of shape (N, 3)
    * faces are int64 arrays of shape (M, 3) with **0-based** indices
    * a single vector is a 1-D array of shape (3,)  (MATLAB uses 3x1 columns)
    * rotation matrices keep MATLAB's layout: axes are the *columns*
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class TriMesh:
    """Minimal equivalent of MATLAB's ``triangulation`` object."""

    points: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        points = np.ascontiguousarray(self.points, dtype=np.float64)
        faces = np.ascontiguousarray(self.faces, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {points.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must have shape (M, 3), got {faces.shape}")
        if faces.size and (faces.min() < 0 or faces.max() >= len(points)):
            raise ValueError("faces reference non-existent points (are they 0-based?)")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "faces", faces)

    # --- equivalents of triangulation methods -----------------------------

    def _corners(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.points
        return p[self.faces[:, 0]], p[self.faces[:, 1]], p[self.faces[:, 2]]

    def face_normals(self) -> np.ndarray:
        """Unit normals, same orientation as MATLAB ``faceNormal``."""
        p1, p2, p3 = self._corners()
        n = np.cross(p2 - p1, p3 - p1)
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    def vertex_normals(self) -> np.ndarray:
        """MATLAB ``vertexNormal``: normalised *unweighted* mean of the unit
        normals of the attached faces (verified against MATLAB, see
        mathworks.com/matlabcentral/answers/511341)."""
        acc = np.zeros_like(self.points)
        fn = self.face_normals()
        for j in range(3):
            np.add.at(acc, self.faces[:, j], fn)
        return acc / np.linalg.norm(acc, axis=1, keepdims=True)

    def face_areas(self) -> np.ndarray:
        p1, p2, p3 = self._corners()
        return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1), axis=1)

    def incenters(self) -> np.ndarray:
        """Incenters of the faces, as MATLAB ``incenter`` (not centroids!)."""
        p1, p2, p3 = self._corners()
        a = np.linalg.norm(p2 - p3, axis=1)[:, None]  # side opposite p1
        b = np.linalg.norm(p3 - p1, axis=1)[:, None]  # side opposite p2
        c = np.linalg.norm(p1 - p2, axis=1)[:, None]  # side opposite p3
        return (a * p1 + b * p2 + c * p3) / (a + b + c)

    def free_boundary_edges(self) -> np.ndarray:
        """Edges used by exactly one face (MATLAB ``freeBoundary``), unordered."""
        edges = np.concatenate(
            [self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]]
        )
        edges_sorted = np.sort(edges, axis=1)
        _, inverse, counts = np.unique(
            edges_sorted, axis=0, return_inverse=True, return_counts=True
        )
        return edges[counts[inverse.ravel()] == 1]

    def is_closed(self) -> bool:
        return len(self.free_boundary_edges()) == 0

    def faces_attached_to(self, vertex_ids: np.ndarray) -> np.ndarray:
        """Sorted ids of faces touching any of the vertices (``vertexAttachments``)."""
        mask = np.isin(self.faces, np.asarray(vertex_ids, dtype=np.int64)).any(axis=1)
        return np.flatnonzero(mask)

    def nearest_vertex(self, query_points: np.ndarray) -> np.ndarray:
        """Index of the nearest mesh vertex (MATLAB ``nearestNeighbor``)."""
        _, idx = cKDTree(self.points).query(np.atleast_2d(query_points))
        return idx


# --- GIBOC-core functions -------------------------------------------------


def normalize_v(v: np.ndarray) -> np.ndarray:
    """normalizeV.m: unit vector, or row-wise normalisation of an (N, 3) array."""
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1 or 1 in v.shape:
        v = v.ravel()
        return v / np.linalg.norm(v)
    if v.shape[1] != 3:
        v = v.T
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def mesh_2d_center(mesh: TriMesh) -> np.ndarray:
    """TriMesh2DProperties.m ``Center``: area-weighted mean of face incenters."""
    areas = mesh.face_areas()
    return (mesh.incenters() * areas[:, None]).sum(axis=0) / areas.sum()


def tri_change_cs(mesh: TriMesh, V: np.ndarray, T: np.ndarray) -> TriMesh:
    """TriChangeCS.m (3-argument form): express the mesh in the frame (V, T).

    V holds the new axes as columns, T is the new origin.
    """
    new_points = (mesh.points - np.asarray(T).ravel()) @ np.asarray(V)
    return TriMesh(new_points, mesh.faces)


def tri_reduce_mesh(
    mesh: TriMesh,
    faces_kept: np.ndarray | None = None,
    nodes_kept: np.ndarray | None = None,
) -> TriMesh:
    """TriReduceMesh.m: keep a subset of faces (or faces attached to given nodes).

    ``nodes_kept`` may be integer vertex ids (0-based) or point coordinates,
    which are snapped to the nearest vertices, as in MATLAB.
    """
    if nodes_kept is not None:
        nodes_kept = np.asarray(nodes_kept)
        if np.issubdtype(nodes_kept.dtype, np.integer):
            node_ids = nodes_kept.ravel()
        else:
            node_ids = np.unique(mesh.nearest_vertex(nodes_kept))
        faces_kept = mesh.faces_attached_to(node_ids)
    if faces_kept is None:
        raise ValueError("provide faces_kept or nodes_kept")

    kept_faces = mesh.faces[np.asarray(faces_kept, dtype=np.int64)]
    kept_nodes = np.unique(kept_faces)
    old2new = np.zeros(len(mesh.points), dtype=np.int64)
    old2new[kept_nodes] = np.arange(len(kept_nodes))
    return TriMesh(mesh.points[kept_nodes], old2new[kept_faces])


@dataclass(frozen=True)
class InertiaProperties:
    eig_vectors: np.ndarray  # (3, 3), columns = principal axes, ascending moments
    center_vol: np.ndarray  # (3,)
    inertia_matrix: np.ndarray  # (3, 3), about center_vol, unit density
    eig_values: np.ndarray  # (3,), ascending (diag of MATLAB's D)
    mass: float  # = enclosed volume for unit density


def _subexpressions(w0, w1, w2):
    temp0 = w0 + w1
    f1 = temp0 + w2
    temp1 = w0 * w0
    temp2 = temp1 + w1 * temp0
    f2 = temp2 + w2 * f1
    f3 = w0 * temp1 + w1 * temp2 + w2 * f2
    g0 = f2 + w0 * (f1 + w0)
    g1 = f2 + w1 * (f1 + w1)
    g2 = f2 + w2 * (f1 + w2)
    return f1, f2, f3, g0, g1, g2


def tri_inertia_ppties(mesh: TriMesh) -> InertiaProperties:
    """TriInertiaPpties.m: volume, centroid and inertia of a closed mesh.

    Vectorised version of the per-triangle loop in the MATLAB code
    (Eberly's polyhedral mass properties algorithm).

    NOTE: eigenvalues are ascending as in MATLAB, but the *sign* of each
    eigenvector may differ from MATLAB. Callers must not rely on it.
    """
    if not mesh.is_closed():
        raise ValueError(
            "The inertia properties are for hole-free triangulations. "
            "Close your mesh before use."
        )

    pseudo_center = mesh.points.mean(axis=0)
    nodes = mesh.points - pseudo_center
    p1, p2, p3 = nodes[mesh.faces[:, 0]], nodes[mesh.faces[:, 1]], nodes[mesh.faces[:, 2]]
    d = np.cross(p2 - p1, p3 - p1)
    d0, d1, d2 = d[:, 0], d[:, 1], d[:, 2]

    x0, x1, x2 = p1[:, 0], p2[:, 0], p3[:, 0]
    y0, y1, y2 = p1[:, 1], p2[:, 1], p3[:, 1]
    z0, z1, z2 = p1[:, 2], p2[:, 2], p3[:, 2]
    f1x, f2x, f3x, g0x, g1x, g2x = _subexpressions(x0, x1, x2)
    _, f2y, f3y, g0y, g1y, g2y = _subexpressions(y0, y1, y2)
    _, f2z, f3z, g0z, g1z, g2z = _subexpressions(z0, z1, z2)

    intg = np.array(
        [
            np.sum(d0 * f1x),
            np.sum(d0 * f2x),
            np.sum(d1 * f2y),
            np.sum(d2 * f2z),
            np.sum(d0 * f3x),
            np.sum(d1 * f3y),
            np.sum(d2 * f3z),
            np.sum(d0 * (y0 * g0x + y1 * g1x + y2 * g2x)),
            np.sum(d1 * (z0 * g0y + z1 * g1y + z2 * g2y)),
            np.sum(d2 * (x0 * g0z + x1 * g1z + x2 * g2z)),
        ]
    )
    mult = np.array([1 / 6, 1 / 24, 1 / 24, 1 / 24, 1 / 60, 1 / 60, 1 / 60, 1 / 120, 1 / 120, 1 / 120])
    intg = intg * mult

    mass = intg[0]
    c = intg[1:4] / mass
    I = np.zeros((3, 3))
    I[0, 0] = intg[5] + intg[6] - mass * (c[1] ** 2 + c[2] ** 2)
    I[1, 1] = intg[4] + intg[6] - mass * (c[2] ** 2 + c[0] ** 2)
    I[2, 2] = intg[4] + intg[5] - mass * (c[0] ** 2 + c[1] ** 2)
    I[0, 1] = I[1, 0] = -(intg[7] - mass * c[0] * c[1])
    I[1, 2] = I[2, 1] = -(intg[8] - mass * c[1] * c[2])
    I[2, 0] = I[0, 2] = -(intg[9] - mass * c[2] * c[0])

    eig_values, eig_vectors = np.linalg.eigh(I)
    return InertiaProperties(
        eig_vectors=eig_vectors,
        center_vol=c + pseudo_center,
        inertia_matrix=I,
        eig_values=eig_values,
        mass=float(mass),
    )


def convex_hull_mesh(points: np.ndarray) -> TriMesh:
    """Convex hull as a mesh with only the hull vertices (convhull + reindexing)."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    used = np.unique(hull.simplices)
    old2new = np.zeros(len(points), dtype=np.int64)
    old2new[used] = np.arange(len(used))
    return TriMesh(points[used], old2new[hull.simplices])
