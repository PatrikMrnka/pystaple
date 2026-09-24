"""Unit tests of mesh utilities on synthetic geometries with known answers."""

import numpy as np
import pytest

from pystaple.mesh import (
    TriMesh,
    convex_hull_mesh,
    normalize_v,
    tri_change_cs,
    tri_inertia_ppties,
    tri_reduce_mesh,
)
from pystaple.utils import body_side_to_sign, compute_xyz_angle_seq


def box(a=1.0, b=1.0, c=1.0, center=(0, 0, 0)) -> TriMesh:
    """Closed, outward-oriented box mesh."""
    x, y, z = a / 2, b / 2, c / 2
    pts = np.array(
        [[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
         [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]]
    ) + np.asarray(center)
    faces = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
         [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    )
    return TriMesh(pts, faces)


def rot_xyz(a, b, c):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def test_box_inertia_matches_analytic():
    a, b, c = 2.0, 3.0, 5.0
    props = tri_inertia_ppties(box(a, b, c, center=(10, -4, 7)))
    m = a * b * c
    assert props.mass == pytest.approx(m)
    np.testing.assert_allclose(props.center_vol, [10, -4, 7], atol=1e-12)
    expected = np.diag([m * (b**2 + c**2), m * (a**2 + c**2), m * (a**2 + b**2)]) / 12
    np.testing.assert_allclose(props.inertia_matrix, expected, atol=1e-9)
    assert np.all(np.diff(props.eig_values) >= 0)  # ascending, as MATLAB eig


def test_inertia_rejects_open_mesh():
    closed = box()
    open_mesh = TriMesh(closed.points, closed.faces[:-1])
    assert not open_mesh.is_closed()
    with pytest.raises(ValueError):
        tri_inertia_ppties(open_mesh)


def test_incenter_of_345_triangle():
    tri = TriMesh(np.array([[0, 0, 0], [4, 0, 0], [0, 3, 0]]), np.array([[0, 1, 2]]))
    np.testing.assert_allclose(tri.incenters()[0], [1, 1, 0])  # inradius = 1
    np.testing.assert_allclose(tri.face_normals()[0], [0, 0, 1])
    assert tri.face_areas()[0] == pytest.approx(6)


def test_tri_change_cs_roundtrip():
    mesh = box(1, 2, 3)
    R = rot_xyz(0.3, -0.2, 1.1)
    T = np.array([5.0, -1.0, 2.0])
    moved = tri_change_cs(mesh, R, T)
    np.testing.assert_allclose(moved.points @ R.T + T, mesh.points, atol=1e-12)


def test_tri_reduce_mesh_keeps_attached_faces():
    mesh = box()
    reduced = tri_reduce_mesh(mesh, nodes_kept=np.array([6]))
    assert len(reduced.faces) == np.isin(mesh.faces, [6]).any(axis=1).sum()
    assert reduced.faces.max() == len(reduced.points) - 1


def test_convex_hull_of_box_is_box():
    hull = convex_hull_mesh(box(2, 2, 2).points)
    assert len(hull.points) == 8
    assert hull.face_areas().sum() == pytest.approx(24)


def test_xyz_angle_sequence_roundtrip():
    angles = np.array([0.4, -0.7, 1.2])
    np.testing.assert_allclose(compute_xyz_angle_seq(rot_xyz(*angles)), angles, atol=1e-12)
    np.testing.assert_allclose(compute_xyz_angle_seq(np.eye(3)), 0, atol=1e-15)


def test_normalize_v_and_side():
    np.testing.assert_allclose(normalize_v([3, 0, 4]), [0.6, 0, 0.8])
    assert body_side_to_sign("Right") == (1, "r")
    assert body_side_to_sign("l") == (-1, "l")
    with pytest.raises(ValueError):
        body_side_to_sign("x")
