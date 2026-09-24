"""STAPLE_pelvis compared with the MATLAB reference outputs.

The port replicates the MATLAB code step by step, so the results should match
to (almost) machine precision. Larger differences point to a porting error, or
to a different convex hull triangulation (scipy vs MATLAB Qhull options).
"""

import pytest

from conftest import REFERENCE_DIR, assert_close, parametrize_datasets
from pystaple.algorithms.pelvis import staple_pelvis
from pystaple.io import load_reference_json, load_reference_mesh

POS_TOL_MM = 1e-3  # positions in mm
POS_TOL_M = POS_TOL_MM * 1e-3  # child_location is in metres
ROT_TOL = 1e-6  # rotation matrix entries
ANGLE_TOL = 1e-6  # radians
INERTIA_RTOL = 1e-6  # relative to the largest inertia entry


@pytest.fixture
def pelvis_case(dataset):
    ref = load_reference_json(REFERENCE_DIR / dataset / "reference.json")
    if "pelvis" not in ref["CS"]:
        pytest.skip(f"MATLAB failed on pelvis: {ref.get('errors', {}).get('pelvis_no_sacrum', '?')}")
    mesh = load_reference_mesh(REFERENCE_DIR / dataset / "mesh_pelvis_no_sacrum.mat")
    BCS, JCS, BL = staple_pelvis(mesh, side="r", in_mm=True)
    return (BCS, JCS, BL), (ref["CS"]["pelvis"], ref["JCS"]["pelvis"], ref["BL"]["pelvis"])


@parametrize_datasets("pelvis_no_sacrum")
def test_pelvis_landmarks(pelvis_case):
    (_, _, BL), (_, _, ref_BL) = pelvis_case
    for name in ("RASI", "LASI", "RPSI", "LPSI", "SYMP"):
        assert_close(BL[name], ref_BL[name], POS_TOL_MM, f"BL.{name}")


@parametrize_datasets("pelvis_no_sacrum")
def test_pelvis_body_cs(pelvis_case):
    (BCS, _, _), (ref_BCS, _, _) = pelvis_case
    assert_close(BCS["CenterVol"], ref_BCS["CenterVol"], POS_TOL_MM, "BCS.CenterVol")
    assert_close(BCS["Origin"], ref_BCS["Origin"], POS_TOL_MM, "BCS.Origin")
    assert_close(BCS["V"], ref_BCS["V"], ROT_TOL, "BCS.V")
    scale = abs(max(map(max, ref_BCS["InertiaMatrix"]), key=abs))
    assert_close(BCS["InertiaMatrix"], ref_BCS["InertiaMatrix"], INERTIA_RTOL * scale, "BCS.InertiaMatrix")


@parametrize_datasets("pelvis_no_sacrum")
def test_pelvis_joint_cs(pelvis_case):
    (_, JCS, _), (_, ref_JCS, _) = pelvis_case
    gp, ref_gp = JCS["ground_pelvis"], ref_JCS["ground_pelvis"]
    assert_close(gp["V"], ref_gp["V"], ROT_TOL, "JCS.ground_pelvis.V")
    assert_close(gp["Origin"], ref_gp["Origin"], POS_TOL_MM, "JCS.ground_pelvis.Origin")
    assert_close(gp["child_location"], ref_gp["child_location"], POS_TOL_M, "child_location")
    assert_close(gp["child_orientation"], ref_gp["child_orientation"], ANGLE_TOL, "child_orientation")
    assert_close(
        JCS["hip_r"]["parent_orientation"],
        ref_JCS["hip_r"]["parent_orientation"],
        ANGLE_TOL,
        "JCS.hip_r.parent_orientation",
    )
