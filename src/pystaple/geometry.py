"""Assorted geometry helpers: hull edges, 2D point-in-polygon, region growing."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree


def inpolygon_many(xq: np.ndarray, yq: np.ndarray, xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
    """Vectorised point-in-polygon (even-odd rule), like MATLAB inpolygon.

    The polygon is closed implicitly; a repeated closing vertex is harmless.
    """
    xq, yq = np.asarray(xq, float)[:, None], np.asarray(yq, float)[:, None]
    x1, y1 = np.asarray(xv, float)[None, :], np.asarray(yv, float)[None, :]
    x2, y2 = np.roll(x1, -1, axis=1), np.roll(y1, -1, axis=1)
    crosses = (y1 > yq) != (y2 > yq)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at = x1 + (yq - y1) * (x2 - x1) / (y2 - y1)
    return (np.count_nonzero(crosses & (xq < x_at), axis=1) % 2).astype(bool)


def convex_hull_2d_polygon(pts2d: np.ndarray) -> np.ndarray:
    """Indices of the 2D convex hull vertices, counter-clockwise (convhull)."""
    return ConvexHull(pts2d).vertices


def convex_hull_with_coplanar_points(pts: np.ndarray) -> np.ndarray:
    """Outward-oriented 3D convex hull triangles that keep points lying on flat
    hull facets as vertices, like MATLAB ``convhull`` does.

    scipy's ConvexHull drops such points, which changes the hull edges on
    meshes with many coplanar points (e.g. surfaces segmented from voxel
    grids). The boundary of a Delaunay tetrahedralisation keeps them.
    """
    tri = Delaunay(pts)
    simp, nb = tri.simplices, tri.neighbors
    s_idx, opp = np.nonzero(nb == -1)  # face opposite vertex `opp` is on the hull
    all_local = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    K = simp[s_idx[:, None], all_local[opp]]
    inside = pts[simp[s_idx, opp]]
    n = np.cross(pts[K[:, 1]] - pts[K[:, 0]], pts[K[:, 2]] - pts[K[:, 0]])
    flip = np.sum(n * (pts[K[:, 0]] - inside), axis=1) < 0
    K[flip] = K[flip][:, [0, 2, 1]]
    return K


def largest_edge_conv_hull(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LargestEdgeConvHull.m (3D): hull edges sorted by decreasing length.

    Returns (IdxPointsPair (E, 2), EdgesLength (E,)). Each hull edge appears
    twice, once per adjacent triangle and in opposite directions, because hull
    triangles are oriented outwards as in MATLAB convhull.

    NOTE: edges of exactly equal length keep the order of the hull triangles,
    which is implementation specific (MATLAB and scipy triangulate flat facets
    differently). This only matters for meshes on a regular grid.
    """
    K = convex_hull_with_coplanar_points(pts)
    pairs = np.vstack([K[:, [0, 1]], K[:, [1, 2]], K[:, [2, 0]]])
    lengths = np.linalg.norm(pts[pairs[:, 1]] - pts[pairs[:, 0]], axis=1)
    order = np.argsort(-lengths, kind="stable")  # sortrows(...,'descend') is stable
    return pairs[order], lengths[order]


def pc_region_growing(pts: np.ndarray, seeds: np.ndarray, r: float) -> np.ndarray:
    """PCRegionGrowing.m: points reachable from the seeds by steps <= r."""
    pts = np.asarray(pts, dtype=float)
    remaining = np.arange(len(pts))
    _, first = cKDTree(pts).query(np.atleast_2d(seeds))  # dsearchn
    seed_pts = pts[np.atleast_1d(first)]
    out = []
    while len(seed_pts) and len(remaining):
        tree = cKDTree(pts[remaining])
        found = np.unique(np.concatenate(tree.query_ball_point(seed_pts, r)).astype(np.int64))
        if found.size == 0:
            break
        seed_pts = pts[remaining[found]]
        out.append(seed_pts)
        remaining = np.delete(remaining, found)
    return np.vstack(out) if out else np.empty((0, pts.shape[1]))
