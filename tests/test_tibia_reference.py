"""Kai2014_tibia compared with the MATLAB reference outputs."""

import pytest

from conftest import REFERENCE_DIR, assert_close, parametrize_datasets
from pystaple.algorithms.tibia import kai2014_tibia
from pystaple.io import load_reference_json, load_reference_mesh

POS_TOL_MM = 1e-3
ROT_TOL = 1e-6
ANGLE_TOL = 1e-6
INERTIA_RTOL = 1e-6

_cache = {}


@pytest.fixture
def tibia_case(dataset):
    if dataset not in _cache:  # the tibia algorithm takes a few seconds
        ref = load_reference_json(REFERENCE_DIR / dataset / "reference.json")
        if "tibia_r" not in ref["CS"]:
            pytest.skip(f"MATLAB failed on tibia: {ref.get('errors', {}).get('tibia_r', '?')}")
        mesh = load_reference_mesh(REFERENCE_DIR / dataset / "mesh_tibia_r.mat")
        _cache[dataset] = (
            kai2014_tibia(mesh, side="r"),
            (ref["CS"]["tibia_r"], ref["JCS"]["tibia_r"], ref["BL"]["tibia_r"]),
        )
    return _cache[dataset]


@parametrize_datasets("tibia_r")
def test_tibia_landmarks(tibia_case):
    (_, _, BL), (_, _, ref_BL) = tibia_case
    assert set(BL) == set(ref_BL), "different set of landmarks (fibula detection?)"
    for name in ref_BL:
        assert_close(BL[name], ref_BL[name], POS_TOL_MM, f"BL.{name}")


@parametrize_datasets("tibia_r")
def test_tibia_body_cs(tibia_case):
    (BCS, _, _), (ref_BCS, _, _) = tibia_case
    assert_close(BCS["CenterVol"], ref_BCS["CenterVol"], POS_TOL_MM, "BCS.CenterVol")
    assert_close(BCS["Origin"], ref_BCS["Origin"], POS_TOL_MM, "BCS.Origin")
    assert_close(BCS["V"], ref_BCS["V"], ROT_TOL, "BCS.V")
    scale = abs(max(map(max, ref_BCS["InertiaMatrix"]), key=abs))
    assert_close(BCS["InertiaMatrix"], ref_BCS["InertiaMatrix"], INERTIA_RTOL * scale, "BCS.InertiaMatrix")


@parametrize_datasets("tibia_r")
def test_tibia_knee_child_frame(tibia_case):
    (_, JCS, _), (_, ref_JCS, _) = tibia_case
    knee, ref_knee = JCS["knee_r"], ref_JCS["knee_r"]
    assert_close(knee["Origin"], ref_knee["Origin"], POS_TOL_MM, "JCS.knee_r.Origin")
    assert_close(knee["V"], ref_knee["V"], ROT_TOL, "JCS.knee_r.V")
    assert_close(knee["child_orientation"], ref_knee["child_orientation"], ANGLE_TOL, "child_orientation")
