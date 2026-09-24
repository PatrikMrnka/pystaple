"""The OpenSim model built by pystaple vs the MATLAB model (needs opensim)."""

import warnings
from pathlib import PureWindowsPath

import numpy as np
import pytest

osim = pytest.importorskip("opensim")

from osim_reference import (  # noqa: E402
    ANGLE_TOL, BODY_MASS, MASS_RTOL, POS_TOL_M, assert_vec, bone_analysis, parametrize_osim_datasets,
    reference_model, require_reference,
)
from pystaple.osim import read_osim, write_model_geometries_folder  # noqa: E402
from pystaple.workflow import assemble_hip_model  # noqa: E402


_models = {}


@pytest.fixture
def models(dataset, tmp_path_factory):
    require_reference(dataset)
    if dataset not in _models:
        geom_set, JCS, BL = bone_analysis(dataset)
        out_dir = tmp_path_factory.mktemp(f"osim_{dataset}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # fast-simplification may be missing
            write_model_geometries_folder(geom_set, out_dir / "Geometry", "obj")
        path = out_dir / "bone_model.osim"
        model = assemble_hip_model(geom_set, JCS, BL, "auto2020_hip_R", BODY_MASS, model_file=path)
        model.printToXML(str(path))
        _models[dataset] = (read_osim(path), reference_model(dataset), path)
    return _models[dataset]

@parametrize_osim_datasets()
def test_model_header(models):
    py, ref, _ = models
    assert py["name"] == ref["name"]
    np.testing.assert_array_equal(py["gravity"], ref["gravity"])
    assert py["credits"] == ref["credits"]


@parametrize_osim_datasets()
def test_bodies(models):
    py, ref, _ = models
    assert list(py["bodies"]) == list(ref["bodies"])
    for name, rb in ref["bodies"].items():
        pb = py["bodies"][name]
        assert pb["mass"] == pytest.approx(rb["mass"], rel=MASS_RTOL)
        np.testing.assert_allclose(pb["inertia"], rb["inertia"], rtol=MASS_RTOL, err_msg=name)
        assert_vec(pb["mass_center"], rb["mass_center"], POS_TOL_M, f"{name} mass_center")
        assert [m["name"] for m in pb["meshes"]] == [m["name"] for m in rb["meshes"]]
        for pm, rm in zip(pb["meshes"], rb["meshes"]):
            assert PureWindowsPath(pm["file"]).parts == PureWindowsPath(rm["file"]).parts
            np.testing.assert_array_equal(pm["scale_factors"], rm["scale_factors"])


@parametrize_osim_datasets()
def test_joints(models):
    py, ref, _ = models
    assert list(py["joints"]) == list(ref["joints"])
    for name, rj in ref["joints"].items():
        pj = py["joints"][name]
        assert (pj["type"], pj["parent_frame"], pj["child_frame"]) == (rj["type"], rj["parent_frame"], rj["child_frame"])
        assert list(pj["coordinates"]) == list(rj["coordinates"])
        for cname, rc in rj["coordinates"].items():
            np.testing.assert_allclose(pj["coordinates"][cname]["range"], rc["range"], rtol=1e-15)
        assert list(pj["frames"]) == list(rj["frames"])
        for fname, rf in rj["frames"].items():
            pf = pj["frames"][fname]
            assert pf["parent"] == rf["parent"], f"{name}/{fname}"
            assert_vec(pf["translation"], rf["translation"], POS_TOL_M, f"{name}/{fname} translation")
            assert_vec(pf["orientation"], rf["orientation"], ANGLE_TOL, f"{name}/{fname} orientation")
        assert list(pj["axes"]) == list(rj["axes"])
        for aname, ra in rj["axes"].items():
            pa = pj["axes"][aname]
            assert pa["coordinates"] == ra["coordinates"], f"{name}/{aname}"
            assert pa["function"] == ra["function"], f"{name}/{aname}"
            np.testing.assert_allclose(pa["axis"], ra["axis"], atol=1e-12, err_msg=f"{name}/{aname}")


@parametrize_osim_datasets()
def test_markers(models):
    py, ref, _ = models
    assert list(py["markers"]) == list(ref["markers"])
    for name, rm in ref["markers"].items():
        assert py["markers"][name]["parent_frame"] == rm["parent_frame"]
        assert_vec(py["markers"][name]["location"], rm["location"], POS_TOL_M, f"marker {name}")


@parametrize_osim_datasets()
def test_model_loads_and_initializes(models):
    py, ref, path = models
    model = osim.Model(str(path))
    state = model.initSystem()
    assert model.getNumCoordinates() == sum(len(j["coordinates"]) for j in ref["joints"].values())
    assert model.getMarkerSet().getSize() == len(ref["markers"])
    assert np.isfinite(model.calcMassCenterPosition(state).get(0))
