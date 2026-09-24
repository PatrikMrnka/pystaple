"""Unit tests of planar sections, hole filling and fitting on synthetic data."""

import numpy as np
import pytest

from pystaple.fitting import fit_ellipse, pca_matlab
from pystaple.mesh import TriMesh, tri_inertia_ppties
from pystaple.sections import (
    matlab_colon,
    plan_polygon_centroid_3d,
    tri_fill_planar_holes,
    tri_plan_intersect,
)
from test_mesh import box


def merge(*meshes, flip=()):
    pts, faces, off = [], [], 0
    for i, m in enumerate(meshes):
        f = m.faces[:, ::-1] if i in flip else m.faces
        pts.append(m.points)
        faces.append(f + off)
        off += len(m.points)
    return TriMesh(np.vstack(pts), np.vstack(faces))


def test_box_section_is_one_closed_rectangle():
    curves, tot = tri_plan_intersect(box(2, 3, 4), [0, 0, 1], -0.3)  # plane z = 0.3
    assert len(curves) == 1
    c = curves[0]
    np.testing.assert_allclose(c.pts[0], c.pts[-1])  # closed like in MATLAB
    assert c.area == pytest.approx(6)
    assert tot == pytest.approx(6)
    assert c.hole == -1


def test_two_separate_sections():
    m = merge(box(1, 1, 4), box(2, 2, 4, center=(10, 0, 0)))
    curves, tot = tri_plan_intersect(m, [0, 0, 1], 0.1)
    assert sorted(round(c.area, 9) for c in curves) == [1, 4]
    assert tot == pytest.approx(5)


def test_section_with_hole():
    tube = merge(box(4, 4, 2), box(2, 2, 2), flip={1})  # inner box as a cavity
    curves, tot = tri_plan_intersect(tube, [0, 0, 1], 0.1)
    assert tot == pytest.approx(16 - 4)
    assert sorted(c.hole for c in curves) == [-1, 1]


def test_polygon_centroid_and_area():
    square = np.array([[0, 0, 5], [2, 0, 5], [2, 2, 5], [0, 2, 5], [0, 0, 5.0]])
    centroid, area = plan_polygon_centroid_3d(square)
    np.testing.assert_allclose(centroid, [1, 1, 5], atol=1e-12)
    assert area == pytest.approx(4)


def test_fill_planar_holes_closes_open_box():
    closed = box(2, 3, 4)
    top = np.flatnonzero(np.all(closed.points[closed.faces][:, :, 2] > 0, axis=1))
    open_box = TriMesh(closed.points, np.delete(closed.faces, top, axis=0))
    assert not open_box.is_closed()
    filled = tri_fill_planar_holes(open_box)
    assert filled.is_closed()
    assert tri_inertia_ppties(filled).mass == pytest.approx(24)


def test_fit_ellipse_recovers_known_ellipse():
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    phi = 0.4
    x0, y0, a, b = 3.0, -2.0, 5.0, 2.0
    xe, ye = a * np.cos(t), b * np.sin(t)
    x = x0 + xe * np.cos(phi) - ye * np.sin(phi)
    y = y0 + xe * np.sin(phi) + ye * np.cos(phi)
    e = fit_ellipse(x, y)
    assert e.X0_in == pytest.approx(x0)
    assert e.Y0_in == pytest.approx(y0)
    assert e.long_axis == pytest.approx(2 * a)
    assert e.short_axis == pytest.approx(2 * b)


def test_pca_sign_convention_and_order():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(500, 3)) * [10, 3, 1]
    coeff = pca_matlab(pts)
    np.testing.assert_allclose(coeff.T @ coeff, np.eye(3), atol=1e-12)
    assert np.all(coeff[np.argmax(np.abs(coeff), axis=0), range(3)] > 0)
    assert abs(coeff[0, 0]) > 0.99  # first component = direction of largest spread


def test_matlab_colon():
    np.testing.assert_allclose(matlab_colon(0.5, 1, 3.5), [0.5, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(matlab_colon(0.5, 1, 3.4), [0.5, 1.5, 2.5])
