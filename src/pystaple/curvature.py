"""TriCurvature.m: principal curvatures from local quadratic fits (2-ring)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr, solve_triangular

from .mesh import TriMesh


def mldivide(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MATLAB ``A\\b`` for a rectangular A: least squares via column-pivoted QR.

    For rank-deficient systems this returns MATLAB's *basic* solution (at most
    `rank` non-zeros), not numpy's minimum-norm solution.
    """
    m, n = A.shape
    Q, R, P = qr(A, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    tol = max(m, n) * np.finfo(float).eps * (diag[0] if diag.size else 0.0)
    r = int(np.count_nonzero(diag > tol))
    x = np.zeros(n)
    if r:
        x[P[:r]] = solve_triangular(R[:r, :r], (Q.T @ b)[:r])
    return x


def _tangent_frame(v: np.ndarray, k: np.ndarray) -> np.ndarray:
    """VectorRotationMatrix: Minv = [v l k] with l = k x v, k = l x v."""
    v = v / np.linalg.norm(v)
    l = np.cross(k, v)
    l /= np.linalg.norm(l)
    k = np.cross(l, v)
    k /= np.linalg.norm(k)
    return np.column_stack([v, l, k])


def _eig2(dxx, dxy, dyy):
    tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy**2)
    mu1 = 0.5 * (dxx + dyy + tmp)
    mu2 = 0.5 * (dxx + dyy - tmp)
    return (mu1, mu2) if abs(mu1) < abs(mu2) else (mu2, mu1)


def tri_curvature(mesh: TriMesh, rng: np.random.RandomState | None = None):
    """TriCurvature.m (usethird=false) -> (Cmean, Cgaussian, Lambda1, Lambda2).

    The local tangent frame uses a random vector, as in MATLAB. Curvatures do
    not depend on it (up to rounding); passing
    ``np.random.RandomState(5489)`` reproduces MATLAB's ``rng(0)`` stream, which
    only matters for degenerate neighbourhoods. Principal directions are not
    computed (unused by STAPLE).
    """
    if rng is None:
        rng = np.random.RandomState(5489)
    pts, faces = mesh.points, mesh.faces
    nv = len(pts)
    normals = mesh.vertex_normals()
    frames = [_tangent_frame(normals[i], rng.random_sample(3)) for i in range(nv)]

    # vertex -> attached faces
    order = np.argsort(faces.ravel(), kind="stable")
    counts = np.bincount(faces.ravel(), minlength=nv)
    starts = np.r_[0, np.cumsum(counts)]
    face_of_slot = order // 3

    def attached(vertices):
        return np.concatenate([face_of_slot[starts[v] : starts[v + 1]] for v in vertices])

    lam1 = np.zeros(nv)
    lam2 = np.zeros(nv)
    for i in range(nv):
        ring1 = np.unique(faces[face_of_slot[starts[i] : starts[i + 1]]])
        nce = np.unique(faces[attached(ring1)])  # 2-ring vertices
        W = pts[nce] @ frames[i]
        f, x, y = W[:, 0], W[:, 1], W[:, 2]
        FM = np.column_stack([x**2, y**2, x * y, x, y, np.ones_like(x)])
        a, b, c = mldivide(FM, f)[:3]
        lam1[i], lam2[i] = _eig2(2 * a, c, 2 * b)

    return (lam1 + lam2) / 2, lam1 * lam2, lam1, lam2
