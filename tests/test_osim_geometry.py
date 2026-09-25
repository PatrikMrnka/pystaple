"""Visualization geometries: pystaple vs MATLAB (reference/<dataset>/osim/Geometry).

MATLAB decimates with reducepatch, pystaple with fast-simplification, so the
meshes differ triangle by triangle. What matters is that the Python meshes have
the same size and are at least as faithful to the original bone as MATLAB's.
Maximum deviations of 1-5 mm occur with both algorithms at thin bone edges, so
point-wise limits against the MATLAB mesh would be meaningless.
Limits calibrated on the 7 bone_datasets (tools/compare_osim.py, 2026-09);
MSKPIPE_CT (msk-PIPE segmentation) is well within them.
"""

import warnings

import pytest

pytest.importorskip("fast_simplification")

from osim_reference import (  # noqa: E402
    BONES, bone_analysis, parametrize_osim_datasets, ref_geometry, require_reference,
)
from pystaple.osim.compare import compare_meshes, read_obj  # noqa: E402
from pystaple.osim.geometry_files import write_model_geometries_folder  # noqa: E402

# size (observed: faces identical, volume within 0.16 %)
FACES_RTOL = 0.01
VOLUME_RTOL = 0.005  # vs MATLAB and vs the original bone
AREA_RTOL_ORIGINAL = 0.03  # decimation smooths noisy surfaces (observed up to ~1.7 %)
# sanity: same bone, same units, same place (observed mean distance <= 0.12 mm)
MEAN_DIST_TO_MATLAB_MM = 0.25
# fidelity to the original bone, relative to MATLAB's reducepatch
# (observed: python mean <= matlab + 0.002 mm, hausdorff <= 1.4 x matlab or +0.35 mm)
FIDELITY_MEAN = (1.2, 0.01)  # python <= a * matlab + b [mm]
FIDELITY_HAUSDORFF = (1.5, 0.5)

_results = {}


@pytest.fixture
def geometries(dataset, tmp_path_factory):
    require_reference(dataset)
    ref_dir = ref_geometry(dataset)
    missing = [b for b in BONES if not (ref_dir / f"{b}.obj").is_file()]
    if missing:
        pytest.skip(f"missing MATLAB geometries in {ref_dir}: {missing}")
    if dataset not in _results:
        geom_set, _, _ = bone_analysis(dataset)
        out = tmp_path_factory.mktemp(f"geom_{dataset}")
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # fast-simplification must be used
            write_model_geometries_folder(geom_set, out, "obj")
        res = {}
        for bone in BONES:
            py, ref, orig = read_obj(out / f"{bone}.obj"), read_obj(ref_dir / f"{bone}.obj"), geom_set[bone]
            res[bone] = (compare_meshes(py, ref), compare_meshes(py, orig), compare_meshes(ref, orig))
        _results[dataset] = res
    return _results[dataset]


@parametrize_osim_datasets()
@pytest.mark.parametrize("bone", BONES)
def test_same_size_as_matlab(geometries, bone):
    vs_matlab, vs_orig, _ = geometries[bone]
    n_py, n_ref = vs_matlab["n_faces"]
    assert abs(n_py - n_ref) <= FACES_RTOL * n_ref, f"faces python {n_py} / matlab {n_ref}"
    assert abs(vs_matlab["volume_rel_diff"]) < VOLUME_RTOL
    assert abs(vs_orig["volume_rel_diff"]) < VOLUME_RTOL
    assert abs(vs_orig["area_rel_diff"]) < AREA_RTOL_ORIGINAL


@parametrize_osim_datasets()
@pytest.mark.parametrize("bone", BONES)
def test_same_bone_as_matlab(geometries, bone):
    r, _, _ = geometries[bone]
    assert r["mean_symmetric"] < MEAN_DIST_TO_MATLAB_MM, f"python->matlab {r['a_to_b']}, matlab->python {r['b_to_a']}"


@parametrize_osim_datasets()
@pytest.mark.parametrize("bone", BONES)
def test_as_faithful_to_the_bone_as_matlab(geometries, bone):
    _, py, ref = geometries[bone]
    a, b = FIDELITY_MEAN
    assert py["mean_symmetric"] <= a * ref["mean_symmetric"] + b, (
        f"mean deviation from the original bone: python {py['mean_symmetric']:.3f} mm, "
        f"matlab {ref['mean_symmetric']:.3f} mm"
    )
    a, b = FIDELITY_HAUSDORFF
    assert py["hausdorff"] <= a * ref["hausdorff"] + b, (
        f"max deviation from the original bone: python {py['hausdorff']:.3f} mm, matlab {ref['hausdorff']:.3f} mm"
    )
