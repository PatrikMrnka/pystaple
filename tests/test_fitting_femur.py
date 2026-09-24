"""Unit tests for step 5b building blocks on synthetic data."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from pystaple.curvature import mldivide, tri_curvature
from pystaple.fitting import fit_csa, lscylinder, lsplane
from pystaple.geometry import inpolygon_many, largest_edge_conv_hull, pc_region_growing
from pystaple.mesh import TriMesh
from test_mesh import box


def sphere_mesh(n=800, radius=10.0) -> TriMesh:
    i = np.arange(n) + 0.5
    phi, theta = np.arccos(1 - 2 * i / n), np.pi * (1 + 5**0.5) * i
    pts = radius * np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])
    hull = ConvexHull(pts)
    K = hull.simplices.copy()
    nrm = np.cross(pts[K[:, 1]] - pts[K[:, 0]], pts[K[:, 2]] - pts[K[:, 0]])
    flip = np.sum(nrm * pts[K].mean(axis=1), axis=1) < 0
    K[flip] = K[flip][:, [0, 2, 1]]
    return TriMesh(pts, K)


def test_vertex_normals_of_sphere_are_radial():
    m = sphere_mesh()
    radial = m.points / np.linalg.norm(m.points, axis=1, keepdims=True)
    assert np.min(np.sum(m.vertex_normals() * radial, axis=1)) > 0.99


def test_curvature_of_sphere():
    cmean, cgauss, l1, l2 = tri_curvature(sphere_mesh(radius=10.0))
    np.testing.assert_allclose(np.abs(l1), 0.1, rtol=0.05)
    np.testing.assert_allclose(np.abs(l2), 0.1, rtol=0.05)
    np.testing.assert_allclose(cgauss, 0.01, rtol=0.1)


def test_mldivide_basic_solution_when_rank_deficient():
    A = np.array([[1.0, 1.0], [2.0, 2.0]])
    x = mldivide(A, np.array([2.0, 4.0]))
    assert np.count_nonzero(x) == 1  # MATLAB basic solution, not min-norm [1, 1]
    np.testing.assert_allclose(A @ x, [2, 4])


def test_lscylinder_recovers_cylinder():
    rng = np.random.default_rng(3)
    axis = np.array([1.0, 0.2, 0.1])
    axis /= np.linalg.norm(axis)
    u = np.cross(axis, [0, 0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    t, h = rng.uniform(0, 2 * np.pi, 400), rng.uniform(-20, 20, 400)
    pts = np.array([5.0, -3, 2]) + np.outer(h, axis) + 12.0 * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
    x0n, an, rn = lscylinder(pts, pts.mean(axis=0), axis + [0.05, -0.05, 0.02], 10.0)
    assert rn == pytest.approx(12.0, abs=1e-6)
    assert abs(abs(an @ axis) - 1) < 1e-9
    assert np.linalg.norm(np.cross(x0n - [5.0, -3, 2], axis)) < 1e-6  # x0n lies on the axis


def test_lsplane():
    pts = np.array([[0, 0, 1.0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    x0, n = lsplane(pts)
    np.testing.assert_allclose(x0, [0.5, 0.5, 1])
    assert abs(abs(n[2]) - 1) < 1e-12


def test_fit_csa_finds_the_rise_of_the_epiphysis():
    z = np.arange(0, 200.0)
    area = 1 + 3 * np.exp(-(((z - 190) / 25) ** 2)) + 0.001 * z
    z_epi, orient = fit_csa(z, area)
    assert orient == 1
    assert z_epi == pytest.approx(190 - 1.5 * 25, abs=1.0)


def test_largest_edges_come_in_opposite_pairs():
    pairs, lengths = largest_edge_conv_hull(box(4, 2, 1).points)
    assert np.all(np.diff(lengths) <= 0)
    as_set = {tuple(p) for p in pairs.tolist()}
    assert all((b, a) in as_set for a, b in as_set)  # outward-oriented hull


def test_region_growing_and_inpolygon():
    pts = np.array([[0, 0.0], [0.05, 0], [0.1, 0], [1, 1]])
    grown = pc_region_growing(pts, [0.0, 0.0], 0.06)
    assert len(grown) == 3
    inside = inpolygon_many([0.5, 2.0], [0.5, 0.5], [0, 1, 1, 0], [0, 0, 1, 1])
    assert inside.tolist() == [True, False]


def test_hull_keeps_points_on_flat_facets_like_matlab():
    from pystaple.geometry import convex_hull_with_coplanar_points

    corners = box(2, 2, 2).points
    face_centres = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])
    pts = np.vstack([corners, face_centres])
    K = convex_hull_with_coplanar_points(pts)
    assert len(np.unique(K)) == 14  # scipy ConvexHull would keep only the 8 corners
    n = np.cross(pts[K[:, 1]] - pts[K[:, 0]], pts[K[:, 2]] - pts[K[:, 0]])
    assert np.all(np.sum(n * pts[K].mean(axis=1), axis=1) > 0)  # outward
