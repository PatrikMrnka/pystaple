"""MATLAB reference OpenSim models (hip_model.m, reference/<dataset>/osim/) and
the matching Python bone analyses, computed once per test session."""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pytest

from conftest import REFERENCE_DIR  # noqa: F401  (re-exported)

BONES = ("pelvis_no_sacrum", "femur_r", "tibia_r")
BODY_MASS = 64.0

# joint centres and frames depend on the distal femur -> same tolerances as there
POS_TOL_M = 1e-5  # 0.01 mm
ANGLE_TOL = 1e-4  # rad
MASS_RTOL = 1e-12


def ref_osim(dataset: str):
    return REFERENCE_DIR / dataset / "osim" / "bone_model.osim"


def ref_geometry(dataset: str):
    return REFERENCE_DIR / dataset / "osim" / "Geometry"


def osim_datasets() -> list[str]:
    if not REFERENCE_DIR.is_dir():
        return []
    return sorted(d.name for d in REFERENCE_DIR.iterdir() if ref_osim(d.name).is_file())


def parametrize_osim_datasets():
    datasets = osim_datasets()
    if datasets:
        return pytest.mark.parametrize("dataset", datasets)
    return pytest.mark.skip(reason=f"no reference/<dataset>/osim/bone_model.osim in {REFERENCE_DIR}")


def require_reference(dataset: str):
    needed = [ref_osim(dataset)] + [REFERENCE_DIR / dataset / f"mesh_{b}.mat" for b in BONES]
    missing = [p for p in needed if not p.is_file()]
    if missing:
        pytest.skip(f"missing reference files: {', '.join(str(p) for p in missing)}")


@lru_cache(maxsize=None)
def bone_analysis(dataset: str):
    """(geom_set, JCS, BL) as processTriGeomBoneSet in hip_model.m."""
    from pystaple.processing import process_tri_geom_bone_set
    from pystaple.workflow import create_tri_geom_set

    geom_set = create_tri_geom_set(BONES, REFERENCE_DIR / dataset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        JCS, BL, _ = process_tri_geom_bone_set(geom_set, "r")
    return geom_set, JCS, BL


@lru_cache(maxsize=None)
def reference_model(dataset: str):
    from pystaple.osim import read_osim

    return read_osim(ref_osim(dataset))


def assert_vec(actual, expected, atol, what):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    assert actual.shape == expected.shape, f"{what}: shape {actual.shape} != {expected.shape}"
    err = np.max(np.abs(actual - expected)) if actual.size else 0.0
    assert err <= atol, f"{what}: max abs error {err:.3g} > {atol:.3g}\n  python: {actual}\n  matlab: {expected}"
