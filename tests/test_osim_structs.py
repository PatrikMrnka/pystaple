"""OpenSim layer without OpenSim: joint structs, mass properties, markers, files.

The reference tests compare the numbers that go into the model with the
MATLAB model reference/ICL_MRI/osim/bone_model.osim.
"""

import numpy as np
import pytest

from osim_reference import (
    ANGLE_TOL, BODY_MASS, MASS_RTOL, POS_TOL_M, assert_vec, bone_analysis, parametrize_osim_datasets,
    reference_model, require_reference,
)
from pystaple.mesh import tri_inertia_ppties
from pystaple.osim.anthropometry import GAIT2392_FULL_BODY_MASS, gait2392_mass_props, segment_mass_props
from pystaple.osim.geometry_files import write_obj
from pystaple.osim.joints import (
    assemble_joint_struct, compile_joint_list, finalize_joint_struct, get_joint_params, infer_body_side,
)
from test_mesh import box


# --- unit tests -------------------------------------------------------------------


def test_joint_params_and_side():
    knee = get_joint_params("knee_l")
    assert (knee["parentName"], knee["childName"], knee["coordsNames"]) == ("femur_l", "tibia_l", ["knee_angle_l"])
    assert get_joint_params("mtp_r")["jointName"] == "toes_r"  # MATLAB quirk
    assert get_joint_params("free_to_ground", "femur_r")["coordsNames"][0] == "ground_femur_r_rz"
    with pytest.raises(ValueError):
        get_joint_params("elbow_r")
    assert infer_body_side(["pelvis_no_sacrum", "femur_r", "tibia_r"]) == "r"
    with pytest.raises(ValueError):
        infer_body_side(["femur_r", "tibia_l"])


def test_joint_list_order_and_assembly():
    JCS = {"pelvis": {"ground_pelvis": {}, "hip_r": {}}, "femur_r": {"hip_r": {}, "knee_r": {}}}
    assert compile_joint_list(JCS) == ["ground_pelvis", "hip_r", "knee_r"]
    js = assemble_joint_struct({"j": {"parent_orientation": 1, "child_location": 2, "child_orientation": 3}})
    assert js["j"]["parent_location"] == 2


def test_partial_models():
    V = np.eye(3)
    JCS = {"femur_r": {"hip_r": {"child_location": np.ones(3), "child_orientation": np.zeros(3), "V": V},
                       "knee_r": {"parent_location": np.ones(3), "parent_orientation": np.zeros(3)}}}
    js = finalize_joint_struct(JCS)  # no pelvis -> free joint to ground, no tibia -> no knee
    assert list(js) == ["ground_femur_r"]
    assert js["ground_femur_r"]["parentName"] == "ground"
    np.testing.assert_allclose(js["ground_femur_r"]["child_location"], 1)


def test_gait2392_scaling():
    assert GAIT2392_FULL_BODY_MASS == pytest.approx(75.337)
    mass, inertia = gait2392_mass_props("femur_r")
    assert mass == 9.3014 and inertia[1] == 0.0351
    mp = segment_mass_props(["pelvis"], {"pelvis": [0.1, 0.2, 0.3]}, {}, 64, "r")["pelvis"]
    assert mp["mass"] == 10.004751981098265  # bone_model.osim from MATLAB, bit for bit
    np.testing.assert_allclose(mp["mass_center"], [0.1, 0.2, 0.3])
    assert np.all(mp["inertia"][3:] == 0)


def test_write_obj_format(tmp_path):
    m = box(2, 2, 2)
    write_obj(m, tmp_path / "b.obj")
    lines = (tmp_path / "b.obj").read_text().splitlines()
    assert lines[0] == "v -1.00000 -1.00000  -1.00000000"  # MATLAB 'v %.5f %.5f %12.8f'
    assert lines[8] == "f 1 3 2"  # 1-based
    assert len(lines) == 8 + 12


# --- comparison with the MATLAB model -----------------------------------------------


@pytest.fixture
def ref(dataset):
    require_reference(dataset)
    return reference_model(dataset)


@pytest.fixture
def analysis(ref, dataset):
    return bone_analysis(dataset)


@parametrize_osim_datasets()
def test_joint_frames_match_matlab(ref, analysis):
    _, JCS, _ = analysis
    joints = finalize_joint_struct(JCS, "auto2020")
    assert list(joints) == list(ref["joints"])
    for name, js in joints.items():
        rj = ref["joints"][name]
        parent = rj["frames"][rj["parent_frame"]]
        child = rj["frames"][rj["child_frame"]]
        assert_vec(js["parent_location"], parent["translation"], POS_TOL_M, f"{name} parent_location")
        assert_vec(js["parent_orientation"], parent["orientation"], ANGLE_TOL, f"{name} parent_orientation")
        assert_vec(js["child_location"], child["translation"], POS_TOL_M, f"{name} child_location")
        assert_vec(js["child_orientation"], child["orientation"], ANGLE_TOL, f"{name} child_orientation")
        assert js["coordsNames"] == list(rj["coordinates"])
        for cname, ctype, (lo, hi) in zip(js["coordsNames"], js["coordsTypes"], js["coordRanges"]):
            f = np.pi / 180 if ctype == "rotational" else 1.0
            expected = rj["coordinates"][cname]["range"]
            np.testing.assert_allclose([lo * f, hi * f], expected, rtol=1e-15, err_msg=cname)


@parametrize_osim_datasets()
def test_mass_props_match_matlab(ref, analysis):
    geom_set, JCS, _ = analysis
    names = ["pelvis" if n == "pelvis_no_sacrum" else n for n in geom_set]
    assert names == list(ref["bodies"])
    initial = {("pelvis" if n == "pelvis_no_sacrum" else n): tri_inertia_ppties(m).center_vol * 1e-3
               for n, m in geom_set.items()}
    mp = segment_mass_props(names, initial, JCS, BODY_MASS, "r")
    for name, rb in ref["bodies"].items():
        assert mp[name]["mass"] == pytest.approx(rb["mass"], rel=MASS_RTOL)
        np.testing.assert_allclose(mp[name]["inertia"], rb["inertia"], rtol=MASS_RTOL, err_msg=name)
        assert_vec(mp[name]["mass_center"], rb["mass_center"], POS_TOL_M, f"{name} mass_center")


@parametrize_osim_datasets()
def test_markers_match_matlab(ref, analysis):
    _, _, BL = analysis
    python = {m: (body, loc) for body, markers in BL.items() for m, loc in markers.items()}
    assert list(python) == list(ref["markers"])
    for name, rm in ref["markers"].items():
        body, loc = python[name]
        assert rm["parent_frame"] == f"/bodyset/{body}"
        assert_vec(np.asarray(loc) * 1e-3, rm["location"], POS_TOL_M, f"marker {name}")


def test_mesh_distances_on_known_geometry():
    from pystaple.osim.compare import compare_meshes, point_to_surface_distance

    m = box(10, 10, 10)
    pts = m.points * 1.1  # corners pushed out along the diagonal by 0.5*sqrt(3)
    np.testing.assert_allclose(point_to_surface_distance(pts, m), 0.5 * np.sqrt(3), rtol=1e-12)
    r = compare_meshes(m, m, n=2000)
    assert r["hausdorff"] < 1e-12 and r["volume_rel_diff"] == 0


def test_mesh_distances_with_degenerate_faces():
    from pystaple.mesh import TriMesh
    from pystaple.osim.compare import compare_meshes, degenerate_faces, point_to_surface_distance

    pts = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 0, 0.0]])
    m = TriMesh(pts, np.array([[0, 1, 2], [0, 3, 1], [3, 3, 1]]))  # collinear + collapsed face
    assert degenerate_faces(m) == 2
    d = point_to_surface_distance(np.array([[0.5, 0.5, 1.0], [1.0, -1.0, 0.0]]), m, k=3)
    np.testing.assert_allclose(d, [1.0, 1.0])
    assert np.isfinite(compare_meshes(m, m, n=500)["hausdorff"])
