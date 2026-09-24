"""Fitting helpers: MATLAB-compatible PCA and fit_ellipse.m."""

from __future__ import annotations

from dataclasses import dataclass

from scipy.linalg import solve_triangular

import numpy as np


def pca_matlab(points: np.ndarray) -> np.ndarray:
    """Principal component coefficients like MATLAB ``pca`` (Statistics Toolbox).

    Columns are ordered by decreasing variance, and each column is signed so
    that its largest-magnitude element is positive (MATLAB's sign convention).
    """
    x = np.asarray(points, dtype=np.float64)
    x = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    coeff = vt.T
    idx = np.argmax(np.abs(coeff), axis=0)
    signs = np.sign(coeff[idx, np.arange(coeff.shape[1])])
    signs[signs == 0] = 1
    return coeff * signs


@dataclass
class Ellipse:
    a: float  # semi-axis along the (rotated) x direction
    b: float  # semi-axis along the (rotated) y direction
    phi: float  # orientation [rad]
    X0: float  # centre in the rotated frame
    Y0: float
    X0_in: float  # centre in the input frame
    Y0_in: float
    long_axis: float
    short_axis: float


def fit_ellipse(x: np.ndarray, y: np.ndarray) -> Ellipse:
    """fit_ellipse.m (Ohad Gal): least-squares ellipse fit, no plotting.

    Raises ValueError where MATLAB would return an empty / non-ellipse result.
    """
    orientation_tolerance = 1e-3
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mean_x, mean_y = x.mean(), y.mean()
    x = x - mean_x
    y = y - mean_y

    X = np.column_stack([x**2, x * y, y**2, x, y])
    XtX = X.T @ X
    if np.linalg.cond(XtX) > 1 / np.finfo(float).eps:
        raise ValueError("fit_ellipse: singular matrix (MATLAB stops with a warning)")
    a, b, c, d, e = np.linalg.solve(XtX, X.sum(axis=0))  # = sum(X)/(X'*X)

    if min(abs(b / a), abs(b / c)) > orientation_tolerance:
        orientation_rad = 0.5 * np.arctan(b / (c - a))
        cos_phi, sin_phi = np.cos(orientation_rad), np.sin(orientation_rad)
        a, b, c, d, e = (
            a * cos_phi**2 - b * cos_phi * sin_phi + c * sin_phi**2,
            0.0,
            a * sin_phi**2 + b * cos_phi * sin_phi + c * cos_phi**2,
            d * cos_phi - e * sin_phi,
            d * sin_phi + e * cos_phi,
        )
        mean_x, mean_y = (
            cos_phi * mean_x - sin_phi * mean_y,
            sin_phi * mean_x + cos_phi * mean_y,
        )
    else:
        orientation_rad = 0.0
        cos_phi, sin_phi = 1.0, 0.0

    if a * c <= 0:
        raise ValueError("fit_ellipse: did not locate an ellipse (parabola or hyperbola)")

    if a < 0:
        a, c, d, e = -a, -c, -d, -e
    X0 = mean_x - d / 2 / a
    Y0 = mean_y - e / 2 / c
    F = 1 + d**2 / (4 * a) + e**2 / (4 * c)
    semi_a, semi_b = np.sqrt(F / a), np.sqrt(F / c)
    R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
    X0_in, Y0_in = R @ np.array([X0, Y0])
    return Ellipse(
        a=float(semi_a),
        b=float(semi_b),
        phi=float(orientation_rad),
        X0=float(X0),
        Y0=float(Y0),
        X0_in=float(X0_in),
        Y0_in=float(Y0_in),
        long_axis=float(2 * max(semi_a, semi_b)),
        short_axis=float(2 * min(semi_a, semi_b)),
    )


# --- fitCSA.m: cross-sectional area profile of a long bone ----------------------

from scipy.optimize import least_squares  # noqa: E402

_TIGHT = dict(xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)


def _gauss2(p, x):
    a1, b1, c1, a2, b2, c2 = p
    return a1 * np.exp(-(((x - b1) / c1) ** 2)) + a2 * np.exp(-(((x - b2) / c2) ** 2))


def _gauss2_jac(p, x):
    J = np.empty((len(x), 6))
    for k, (a, b, c) in enumerate((p[0:3], p[3:6])):
        u = (x - b) / c
        e = np.exp(-(u**2))
        J[:, 3 * k] = e
        J[:, 3 * k + 1] = a * e * 2 * u / c
        J[:, 3 * k + 2] = a * e * 2 * u**2 / c
    return J


def _gauss_lin(p, x):
    a1, b1, c1, d, e = p
    return a1 * np.exp(-(((x - b1) / c1) ** 2)) + d * x + e


def _gauss_lin_jac(p, x):
    a1, b1, c1, _, _ = p
    u = (x - b1) / c1
    ex = np.exp(-(u**2))
    return np.column_stack([ex, a1 * ex * 2 * u / c1, a1 * ex * 2 * u**2 / c1, x, np.ones_like(x)])


def fit_csa(Z: np.ndarray, area: np.ndarray) -> tuple[float, int]:
    """fitCSA.m: altitude where the epiphysis starts -> (Zepi, orientation).

    MATLAB uses the Curve Fitting Toolbox (``fit`` with 'gauss2', then a
    Gaussian plus a line). Here the same models are fitted with
    scipy.optimize.least_squares to tight tolerances; MATLAB stops at its
    default tolerances (1e-6), so Zepi agrees only to ~1e-4 mm.
    The diaphysis limits (Zdiaph) are not computed (unused by STAPLE here).
    """
    Z = np.asarray(Z, dtype=float)
    area = np.asarray(area, dtype=float)
    z0 = Z.mean()
    Z = Z - z0
    area = area / area.mean()
    i_max = int(np.argmax(area))
    orient = -1 if Z[i_max] < Z.mean() else 1

    start1 = [area[i_max], Z[i_max], 20.0, area.mean(), Z.mean(), 75.0]
    lower = [-np.inf, -np.inf, 0.0, -np.inf, -np.inf, 0.0]
    fit1 = least_squares(
        lambda p: _gauss2(p, Z) - area, start1, jac=lambda p: _gauss2_jac(p, Z),
        bounds=(lower, [np.inf] * 6), method="trf", **_TIGHT,
    )
    a1, b1, c1 = fit1.x[:3]

    fit2 = least_squares(
        lambda p: _gauss_lin(p, Z) - area, [a1, b1, c1, 0.0, Z.mean()],
        jac=lambda p: _gauss_lin_jac(p, Z), method="trf", **_TIGHT,
    )
    b1, c1 = fit2.x[1], fit2.x[2]
    z_epi = z0 + b1 - 1.5 * orient * c1
    return float(z_epi), orient


# --- lsplane.m -------------------------------------------------------------------


def lsplane(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """lsplane.m: centroid and normal of the least-squares plane."""
    X = np.asarray(X, dtype=float)
    if len(X) < 3:
        raise ValueError("At least 3 data points required")
    x0 = X.mean(axis=0)
    _, s, vt = np.linalg.svd(X - x0, full_matrices=False)
    return x0, vt[int(np.argmin(s))]


# --- lscylinder.m (NPL LSGE library: Gauss-Newton cylinder fit) --------------------


def _gr(x, y):
    if y == 0:
        return 1.0, 0.0
    if abs(y) >= abs(x):
        t = x / y
        s = 1 / np.sqrt(1 + t * t)
        return t * s, s
    t = y / x
    c = 1 / np.sqrt(1 + t * t)
    return c, t * c


def _rot3z(a):
    c1, s1 = _gr(a[1], a[2])
    z = c1 * a[1] + s1 * a[2]
    V = np.array([[1, 0, 0], [0, s1, -c1], [0, c1, s1]])
    c2, s2 = _gr(a[0], z)
    if c2 * a[0] + s2 * z < 0:
        c2, s2 = -c2, -s2
    W = np.array([[s2, 0, -c2], [0, 1, 0], [c2, 0, s2]])
    return W @ V


def _fgrrot3(theta):
    ct, st = np.cos(theta), np.sin(theta)
    R1 = np.array([[1, 0, 0], [0, ct[0], -st[0]], [0, st[0], ct[0]]])
    R2 = np.array([[ct[1], 0, st[1]], [0, 1, 0], [-st[1], 0, ct[1]]])
    R3 = np.array([[ct[2], -st[2], 0], [st[2], ct[2], 0], [0, 0, 1]])
    dR1 = np.array([[0, 0, 0], [0, -R1[2, 1], -R1[1, 1]], [0, R1[1, 1], -R1[2, 1]]])
    dR2 = np.array([[-R2[0, 2], 0, R2[0, 0]], [0, 0, 0], [-R2[0, 0], 0, -R2[0, 2]]])
    dR3 = np.array([[-R3[1, 0], -R3[0, 0], 0], [R3[0, 0], -R3[1, 0], 0], [0, 0, 0]])
    R = R3 @ R2 @ R1
    return R, R3 @ R2 @ dR1, R3 @ dR2 @ R1, dR3 @ R2 @ R1


def _fgcylinder(a, X):
    """fgcylinder.m: residuals and Jacobian of the cylinder fit (unit weights)."""
    x0, y0, alpha, beta, s = a
    R, DR1, DR2, _ = _fgrrot3(np.array([alpha, beta, 0.0]))
    D = X - np.array([x0, y0, 0.0])
    Xt = D @ R.T
    rt = np.sqrt(Xt[:, 0] ** 2 + Xt[:, 1] ** 2)
    Nt = np.column_stack([Xt[:, 0] / rt, Xt[:, 1] / rt, np.zeros(len(X))])
    f = np.sum(Xt * Nt, axis=1) - s
    J = np.column_stack(
        [
            Nt @ (R @ np.array([-1.0, 0, 0])),
            Nt @ (R @ np.array([0, -1.0, 0])),
            np.sum((D @ DR1.T) * Nt, axis=1),
            np.sum((D @ DR2.T) * Nt, axis=1),
            -np.ones(len(X)),
        ]
    )
    return f, J


def _gncc2(f0, f1, p, g, scale, tolr, scalef):
    eps = np.finfo(float).eps
    sp = np.max(np.abs(p * scale))
    sg = np.max(np.abs(g / scale))
    c1 = sp / (scalef * tolr**0.7)
    c2 = abs(f0 - f1) / (tolr * scalef)
    c3 = sg / (tolr**0.7 * scalef)
    c4 = f1 / (scalef * eps**0.7)
    c5 = sg / (eps**0.7 * scalef)
    return (c1 < 1 and c2 < 1 and c3 < 1) or c4 < 1 or c5 < 1


def _nlss11(ai, tol, X):
    """nlss11.m: Gauss-Newton with line search; returns (a, converged)."""
    a0 = np.asarray(ai, dtype=float).copy()
    n = len(a0)
    mxiter = 100 + int(np.ceil(np.sqrt(n)))
    conv, niter, eta = False, 0, 0.01
    scale = None
    while niter < mxiter and not conv:
        f0, J = _fgcylinder(a0, X)
        if niter == 0:
            scale = np.linalg.norm(J, axis=0)
        F0 = np.linalg.norm(f0)
        Ra = np.triu(np.linalg.qr(np.column_stack([J, f0]), mode="r"))
        R, q = Ra[:n, :n], Ra[:n, n]
        p = -solve_triangular(R, q)
        g = 2 * R.T @ q
        G0 = g @ p
        a1 = a0 + p
        niter += 1
        f1, _ = _fgcylinder(a1, X)
        F1 = np.linalg.norm(f1)
        conv = _gncc2(F0, F1, p, g, scale, tol[0], tol[1])
        if not conv:
            rho = (F1 - F0) * (F1 + F0) / G0
            if rho < eta:
                a0 = a0 + max(0.001, 1 / (2 * (1 - rho))) * p
            else:
                a0 = a0 + p
    return a0 + p, conv


def lscylinder(X, x0, a0, r0, tolp=0.001, tolg=0.001):
    """lscylinder.m: least-squares cylinder -> (point on axis, axis, radius)."""
    import warnings

    X = np.asarray(X, dtype=float)
    if len(X) < 5:
        raise ValueError("At least 5 data points required")
    ez = np.array([0.0, 0.0, 1.0])
    xb = X.mean(axis=0)
    R0 = _rot3z(np.asarray(a0, dtype=float))
    x1 = R0 @ np.asarray(x0, dtype=float)
    xb1 = R0 @ xb
    t = x1 + (xb1[2] - x1[2]) * ez
    X2 = X @ R0.T - t
    xb2 = xb1 - t
    a, conv = _nlss11([0, 0, 0, 0, r0], (tolp, tolg), X2)
    if not conv:
        warnings.warn("*** Gauss-Newton algorithm has not converged ***", stacklevel=2)
    R3, _, _, _ = _fgrrot3(np.array([a[2], a[3], 0.0]))
    an = R0.T @ R3.T @ ez
    p = R3 @ (xb2 - np.array([a[0], a[1], 0.0]))
    x0n = R0.T @ (t + np.array([a[0], a[1], 0.0]) + R3.T @ np.array([0.0, 0.0, p[2]]))
    return x0n, an, float(a[4])
