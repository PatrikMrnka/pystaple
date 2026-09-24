# pystaple

Python port of [STAPLE](https://github.com/modenaxe/msk-STAPLE) (Modenese & Renault),
limited to the workflow used by `hip_model.m`. Work in progress.

STAPLE is released under CC BY-NC 4.0: academic / non-commercial use only,
with attribution to the original authors.

## Status

| Part | MATLAB | Status |
|---|---|---|
| Mesh utilities | GIBOC-core Tri* | done for this workflow |
| Pelvis | STAPLE_pelvis | ported, verified |
| Tibia | Kai2014_tibia | ported, verified |
| Femur | GIBOC_femur (cylinder) | ported, verified |
| OpenSim model | STAPLE/opensim, anthropometry (auto2020) | ported, verified |

## Usage

```python
from pystaple.workflow import create_tri_geom_set, build_hip_model

geom_set = create_tri_geom_set(["pelvis_no_sacrum", "femur_r", "tibia_r"], "path/to/stl")
build_hip_model(geom_set, "output", body_mass=64)   # output/bone_model.osim + output/Geometry
```

Bone geometries are read from `<bone>.stl` or from `.mat` files with `Points` and
`ConnectivityList` arrays (MATLAB `triangulation` objects cannot be read in Python).

## Development

    conda env create -f environment.yml
    conda activate pystaple
    pytest -v

`opensim` is only available from conda (`opensim-org` channel). Without it, the
tests that build the OpenSim model are skipped; everything else (including the
joint frames, mass properties and markers that go into the model) is still tested.

The reference folder can be moved with the `PYSTAPLE_REFERENCE_DIR`
environment variable.

## Verification against MATLAB

Tests compare every step with MATLAB STAPLE on the 7 datasets of `bone_datasets`
(`reference/<dataset>/`: bone analyses from `export_reference_outputs.m`, OpenSim
models and geometries from `hip_model.m` in `osim/`). `tools/compare_osim.py`
rebuilds all models and prints a full comparison.

* Bone analysis (landmarks, joint frames, inertia): within 0.001 mm
  (distal femur 0.01 mm / 1e-4 rad).
* OpenSim model (`bone_model.osim`): no structural difference (bodies, joints,
  coordinates, ranges, axes, frames, markers); numbers differ by at most
  1e-7 mm and 1e-9 rad, masses and inertias are bit-identical.
* Visualization geometries: same number of triangles as MATLAB, same volume
  (within 0.2 %), mean surface distance to the MATLAB mesh 0.01-0.12 mm. Compared
  with the original bone, the Python meshes are as faithful as MATLAB's or more
  (mean deviation 0.000-0.105 mm vs 0.009-0.108 mm).

## Known differences from MATLAB

* `fitCSA` (start of the femoral epiphysis): MATLAB's Curve Fitting Toolbox stops
  at a looser tolerance, so Zepi differs by ~0.01 mm and a few faces at the cut
  may differ. This does not show in the final model.
* On meshes segmented from voxel grids, many convex-hull edges have exactly
  equal lengths and flat hull facets can be triangulated in several ways; a
  couple of condyle point pairs may then differ from MATLAB. STAPLE itself is
  not unique in that case.
* Visualization meshes (`Geometry/*.obj`) are decimated with `fast-simplification`
  instead of MATLAB `reducepatch`, so they differ triangle by triangle (the model
  is not affected). Both algorithms simplify thin bone edges, with maximum
  deviations of 1-4 mm from the original bone on some datasets. On oversampled
  meshes (JIA_MRI) `reducepatch` creates folded triangles (+1.7 % area, up to
  4.7 mm spikes) while `fast-simplification` is practically lossless.
  Mesh paths in the `.osim` use `/` instead of `\`.
* Initial body inertias use `TriInertiaPpties` instead of
  `computeMassProperties_Mirtich1996` (which has a bug in the products of inertia);
  both are overwritten by the gait2392-based mass properties, so the model is identical.

Diagnostics: `tools/compare_osim.py` (OpenSim models),
`tools/diag_femur_matlab.m` + `tools/diag_femur.py <dataset>` (femur pipeline).