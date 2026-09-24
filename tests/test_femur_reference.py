"""GIBOC femur (cylinder) compared with the MATLAB reference outputs.

Step 5a (femoral head) should match to ~1e-6 mm. Step 5b depends on fitCSA,
which MATLAB solves with the Curve Fitting Toolbox at its default tolerance
(1e-6), so tolerances for the distal femur are slightly looser.
"""

import warnings

import pytest

from conftest import REFERENCE_DIR, assert_close, parametrize_datasets
from pystaple.algorithms.femur import giboc_femur
from pystaple.io import load_reference_json, load_reference_mesh

POS_TOL_MM = 1e-3  # femoral head, inertia
POS_TOL_M = POS_TOL_MM * 1e-3
INERTIA_RTOL = 1e-6
# distal femur (condyles, cylinder), see module docstring
DIST_POS_TOL_MM = 0.01
DIST_ROT_TOL = 1e-4
DIST_ANGLE_TOL = 1e-4  # rad (~0.006 deg)

_cache = {}


@pytest.fixture
def femur_case(dataset):
    if dataset not in _cache:
        ref = load_reference_json(REFERENCE_DIR / dataset / "reference.json")
        if "femur_r" not in ref["CS"]:
            pytest.skip(f"MATLAB failed on femur: {ref.get('errors', {}).get('femur_r', '?')}")
        mesh = load_reference_mesh(REFERENCE_DIR / dataset / "mesh_femur_r.mat")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the same warnings are printed by MATLAB
            result = giboc_femur(mesh, side="r")
        _cache[dataset] = (result, (ref["CS"]["femur_r"], ref["JCS"]["femur_r"], ref["BL"]["femur_r"]))
    return _cache[dataset]


@parametrize_datasets("femur_r")
def test_femur_inertia(femur_case):
    (BCS, _, _, _), (ref_BCS, _, _) = femur_case
    assert_close(BCS["CenterVol"], ref_BCS["CenterVol"], POS_TOL_MM, "BCS.CenterVol")
    scale = abs(max(map(max, ref_BCS["InertiaMatrix"]), key=abs))
    assert_close(BCS["InertiaMatrix"], ref_BCS["InertiaMatrix"], INERTIA_RTOL * scale, "BCS.InertiaMatrix")


@parametrize_datasets("femur_r")
def test_femoral_head_centre(femur_case):
    (BCS, JCS, _, _), (ref_BCS, ref_JCS, _) = femur_case
    assert_close(BCS["Origin"], ref_BCS["Origin"], POS_TOL_MM, "BCS.Origin (femoral head centre)")
    assert_close(JCS["hip_r"]["Origin"], ref_JCS["hip_r"]["Origin"], POS_TOL_MM, "JCS.hip_r.Origin")
    assert_close(JCS["hip_r"]["child_location"], ref_JCS["hip_r"]["child_location"], POS_TOL_M, "child_location")


@parametrize_datasets("femur_r")
def test_femur_axes_and_hip_frame(femur_case):
    (BCS, JCS, _, _), (ref_BCS, ref_JCS, _) = femur_case
    assert_close(BCS["V"], ref_BCS["V"], DIST_ROT_TOL, "BCS.V")
    assert_close(JCS["hip_r"]["V"], ref_JCS["hip_r"]["V"], DIST_ROT_TOL, "JCS.hip_r.V")
    assert_close(
        JCS["hip_r"]["child_orientation"], ref_JCS["hip_r"]["child_orientation"],
        DIST_ANGLE_TOL, "JCS.hip_r.child_orientation",
    )


@parametrize_datasets("femur_r")
def test_knee_parent_frame(femur_case):
    (_, JCS, _, _), (_, ref_JCS, _) = femur_case
    knee, ref_knee = JCS["knee_r"], ref_JCS["knee_r"]
    assert_close(knee["Origin"], ref_knee["Origin"], DIST_POS_TOL_MM, "JCS.knee_r.Origin")
    assert_close(knee["parent_location"], ref_knee["parent_location"], DIST_POS_TOL_MM * 1e-3, "parent_location")
    assert_close(knee["V"], ref_knee["V"], DIST_ROT_TOL, "JCS.knee_r.V")
    assert_close(knee["parent_orientation"], ref_knee["parent_orientation"], DIST_ANGLE_TOL, "parent_orientation")


@parametrize_datasets("femur_r")
def test_femur_landmarks(femur_case):
    (_, _, BL, _), (_, _, ref_BL) = femur_case
    assert set(BL) == set(ref_BL)
    for name in ref_BL:
        assert_close(BL[name], ref_BL[name], DIST_POS_TOL_MM, f"BL.{name}")
