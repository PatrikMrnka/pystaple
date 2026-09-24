"""Unit tests of mesh morphology and sphere fitting on synthetic meshes."""

import numpy as np
import pytest

from pystaple.algorithms.femur import compute_tri_coeff_morpho, sphere_fit
from pystaple.mesh import TriMesh, tri_reduce_mesh
from pystaple.morphology import (
    face_neighbors,
    matlab_round,
    tri_dilate_mesh,
    tri_erode_mesh,
    tri_keep_largest_patch,
    tri_unite,
)
from test_mesh import box


def grid(n=10, offset=(0.0, 0.0)) -> TriMesh:
    """Flat n x n grid of unit squares, each split into two triangles."""
    xs, ys = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    pts = np.column_stack([xs.ravel() + offset[0], ys.ravel() + offset[1], np.zeros(xs.size)])
    vid = lambda i, j: i * (n + 1) + j  # noqa: E731
    faces = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            faces += [[a, b, c], [a, c, d]]
    return TriMesh(pts, np.array(faces))


def test_face_neighbors_of_closed_box():
    nb = face_neighbors(box())
    assert np.all(nb >= 0)  # closed mesh: every face has 3 neighbours
    for f, row in enumerate(nb):
        for g in row:
            assert f in nb[g]  # symmetric


def test_dilate_then_erode_grid():
    g = grid(10)
    center_face = tri_reduce_mesh(g, faces_kept=np.array([100]))
    grown = tri_dilate_mesh(g, center_face, 2)
    assert len(grown.faces) > len(center_face.faces)
    eroded = tri_erode_mesh(g, 1)
    assert len(eroded.faces) < len(g.faces)
    assert eroded.points[:, :2].min() >= 1 and eroded.points[:, :2].max() <= 9


def test_keep_largest_patch():
    two = tri_unite(grid(4), grid(8, offset=(20, 0)))
    largest = tri_keep_largest_patch(two)
    assert largest.face_areas().sum() == pytest.approx(64)


def test_sphere_fit_exact():
    rng = np.random.default_rng(1)
    d = rng.normal(size=(300, 3))
    pts = np.array([1.0, -2.0, 3.0]) + 7.5 * d / np.linalg.norm(d, axis=1, keepdims=True)
    c, r, err = sphere_fit(pts)
    np.testing.assert_allclose(c, [1, -2, 3], atol=1e-9)
    assert r == pytest.approx(7.5)
    assert np.max(np.abs(err)) < 1e-8


def test_coeff_morpho_and_round():
    g = grid(10)  # right triangles with legs 1
    assert compute_tri_coeff_morpho(g) == pytest.approx(0.5 / np.sqrt(4 / np.sqrt(3) * 0.5))
    assert [matlab_round(x) for x in (0.5, 1.5, 2.5, -0.5)] == [1, 2, 3, -1]
