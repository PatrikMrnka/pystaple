"""The OpenSim model built by pystaple vs the MATLAB model.

Both backends are compared with MATLAB: "xml" (the default writer, no OpenSim
needed) and "opensim" (the OpenSim API, only where opensim is installed).
Where opensim is installed, the XML writer must also print exactly the same
text as OpenSim and the written files must load and initialise in OpenSim.
"""

import difflib
import re
import warnings
from pathlib import PureWindowsPath

import numpy as np
import pytest

from osim_reference import (
    ANGLE_TOL, BODY_MASS, MASS_RTOL, POS_TOL_M, assert_vec, bone_analysis, parametrize_osim_datasets,
    reference_model, require_reference,
)
from pystaple.osim import osim_text, read_osim, write_model_geometries_folder, write_osim
from pystaple.workflow import assemble_hip_model, hip_model_description

MODEL_NAME = "auto2020_hip_R"
BACKENDS = ["xml", pytest.param("opensim", marks=pytest.mark.opensim)]

_models = {}


def _has_opensim():
    try:
        import opensim  # noqa: F401
    except ImportError:
        return False
    return True


def build(dataset, backend, tmp_path_factory):
    """(path of the written model, parsed model, parsed MATLAB model)."""
    key = (dataset, backend)
    if key not in _models:
        if backend == "opensim" and not _has_opensim():
            pytest.skip("opensim is not installed")
        require_reference(dataset)
        geom_set, JCS, BL = bone_analysis(dataset)
        out_dir = tmp_path_factory.mktemp(f"osim_{dataset}_{backend}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # fast-simplification may be missing
            write_model_geometries_folder(geom_set, out_dir / "Geometry", "obj")
        path = out_dir / "bone_model.osim"
        if backend == "xml":
            write_osim(hip_model_description(geom_set, JCS, BL, MODEL_NAME, BODY_MASS), path)
        else:
            model = assemble_hip_model(geom_set, JCS, BL, MODEL_NAME, BODY_MASS, model_file=path)
            model.printToXML(str(path))
        _models[key] = (path, read_osim(path), reference_model(dataset))
    return _models[key]


@pytest.fixture(params=BACKENDS)
def models(request, dataset, tmp_path_factory):
    _, py, ref = build(dataset, request.param, tmp_path_factory)
    return py, ref


@parametrize_osim_datasets()
def test_model_header(models):
    py, ref = models
    assert py["name"] == ref["name"]
    np.testing.assert_array_equal(py["gravity"], ref["gravity"])
    assert py["credits"] == ref["credits"]


@parametrize_osim_datasets()
def test_bodies(models):
    py, ref = models
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
    py, ref = models
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
    py, ref = models
    assert list(py["markers"]) == list(ref["markers"])
    for name, rm in ref["markers"].items():
        assert py["markers"][name]["parent_frame"] == rm["parent_frame"]
        assert_vec(py["markers"][name]["location"], rm["location"], POS_TOL_M, f"marker {name}")


# --- checks with OpenSim itself ------------------------------------------------------------


def _normalise(text: str) -> str:
    """Without line-ending differences, the document version and the property
    comments (documentation only; they differ between OpenSim versions and the
    writer keeps those of 4.5, the version of the MATLAB reference models)."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"^[ \t]*<!--.*?-->\n", "", text, flags=re.MULTILINE)
    return re.sub(r'<OpenSimDocument Version="\d+">', '<OpenSimDocument Version="*">', text)


@pytest.mark.opensim
@parametrize_osim_datasets()
def test_xml_writer_prints_exactly_like_opensim(dataset, tmp_path_factory):
    """Same model through the OpenSim API and through the XML writer -> same text
    (every element and every digit; comments and version excluded)."""
    path, _, _ = build(dataset, "opensim", tmp_path_factory)
    geom_set, JCS, BL = bone_analysis(dataset)
    ours = _normalise(osim_text(hip_model_description(geom_set, JCS, BL, MODEL_NAME, BODY_MASS)))
    theirs = _normalise(path.read_text(encoding="utf-8"))
    if ours != theirs:
        diff = "\n".join(list(difflib.unified_diff(theirs.splitlines(), ours.splitlines(),
                                                   "opensim", "pystaple", lineterm="", n=1))[:60])
        pytest.fail(f"the XML writer differs from OpenSim's print:\n{diff}")


@pytest.mark.opensim
@parametrize_osim_datasets()
@pytest.mark.parametrize("backend", ["xml", "opensim"])
def test_model_loads_and_initializes(dataset, backend, tmp_path_factory):
    osim = pytest.importorskip("opensim")
    path, _, ref = build(dataset, backend, tmp_path_factory)
    model = osim.Model(str(path))
    state = model.initSystem()
    assert model.getNumCoordinates() == sum(len(j["coordinates"]) for j in ref["joints"].values())
    assert model.getMarkerSet().getSize() == len(ref["markers"])
    assert np.isfinite(model.calcMassCenterPosition(state).get(0))
