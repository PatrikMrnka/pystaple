"""Shared test helpers: locating the MATLAB reference outputs and comparisons."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

REFERENCE_DIR = Path(
    os.environ.get("PYSTAPLE_REFERENCE_DIR", Path(__file__).parents[1] / "reference")
)


def reference_datasets(required_mesh: str) -> list[str]:
    """Datasets that have reference.json and the given input mesh."""
    if not REFERENCE_DIR.is_dir():
        return []
    return sorted(
        d.name
        for d in REFERENCE_DIR.iterdir()
        if (d / "reference.json").is_file() and (d / f"mesh_{required_mesh}.mat").is_file()
    )


def parametrize_datasets(required_mesh: str):
    datasets = reference_datasets(required_mesh)
    if datasets:
        return pytest.mark.parametrize("dataset", datasets)
    if not REFERENCE_DIR.is_dir():
        reason = f"reference folder does not exist: {REFERENCE_DIR}"
    else:
        subdirs = [d.name for d in REFERENCE_DIR.iterdir() if d.is_dir()]
        reason = (
            f"no dataset in {REFERENCE_DIR} has both reference.json and "
            f"mesh_{required_mesh}.mat (subfolders found: {subdirs or 'none'})"
        )
    return pytest.mark.skip(reason=reason)


def assert_close(actual, expected, atol: float, what: str) -> None:
    actual = np.asarray(actual, dtype=float).ravel()
    expected = np.asarray(expected, dtype=float).ravel()
    assert actual.shape == expected.shape, f"{what}: shape {actual.shape} != {expected.shape}"
    err = np.max(np.abs(actual - expected))
    assert err <= atol, (
        f"{what}: max abs error {err:.3g} > {atol:.3g}\n"
        f"  python: {np.array2string(actual, precision=6)}\n"
        f"  matlab: {np.array2string(expected, precision=6)}"
    )
