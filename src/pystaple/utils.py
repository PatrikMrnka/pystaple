"""Small helpers ported from STAPLE (bodySide2Sign.m, computeXYZAngleSeq.m)."""

from __future__ import annotations

import numpy as np


def body_side_to_sign(side_raw: str) -> tuple[int, str]:
    """bodySide2Sign.m: 'r...' -> (1, 'r'), 'l...' -> (-1, 'l')."""
    side_low = side_raw[0].lower() if side_raw else ""
    if side_low == "r":
        return 1, "r"
    if side_low == "l":
        return -1, "l"
    raise ValueError("specify right 'r' or left 'l'")


def compute_xyz_angle_seq(rot_mat: np.ndarray) -> np.ndarray:
    """computeXYZAngleSeq.m: body-fixed X-Y-Z angles, R = Rx(a) @ Ry(b) @ Rz(c).

    Returns [alpha, beta, gamma] in radians, as used for OpenSim orientations.
    """
    R = np.asarray(rot_mat, dtype=np.float64)
    beta = np.arctan2(R[0, 2], np.sqrt(R[0, 0] ** 2 + R[0, 1] ** 2))
    cb = np.cos(beta)
    alpha = np.arctan2(-R[1, 2] / cb, R[2, 2] / cb)
    gamma = np.arctan2(-R[0, 1] / cb, R[0, 0] / cb)
    return np.array([alpha, beta, gamma])
