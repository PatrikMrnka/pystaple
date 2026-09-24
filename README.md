# pystaple

Python port of [STAPLE](https://github.com/modenaxe/msk-STAPLE) (Modenese & Renault),
limited to the workflow used by `hip_model.m`. Work in progress.

> **Non-commercial use only.** pystaple is a translation of STAPLE and, like
> STAPLE, is licensed under [CC BY-NC 4.0](LICENSE). If you use it, please cite the
> STAPLE publication (see [Citation](#citation)).

## Status

| Part | MATLAB | Status |
|---|---|---|
| Mesh utilities | GIBOC-core Tri* | done for this workflow |
| Pelvis | STAPLE_pelvis | ported, verified |
| Tibia | Kai2014_tibia | ported, verified |
| Femur | GIBOC_femur (cylinder) | ported, verified |
| OpenSim model | STAPLE/opensim, anthropometry (auto2020) | ported, verified |

## Installation

OpenSim is only distributed via conda, so use a conda environment:

    conda create -n pystaple -c opensim-org -c conda-forge python=3.11 opensim=4.6
    conda activate pystaple
    pip install "pystaple[viz] @ https://github.com/PatrikMrnka/pystaple/releases/download/v0.1.0/pystaple-0.1.0-py3-none-any.whl"
 
 The bone analysesalone (without building the model) only need numpy and scipy.

## Usage

Command line, equivalent to `hip_model.m`:

    pystaple hip-model path/to/bones output --body-mass 64

The bones folder must contain `pelvis_no_sacrum.stl`, `femur_r.stl` and `tibia_r.stl`
(in mm; `.mat` files with `Points` and `ConnectivityList` also work, MATLAB
`triangulation` objects cannot be read). The output folder gets `bone_model.osim`,
`Geometry/` and a log file. `pystaple hip-model --help` lists the options.

From Python:

```python
from pystaple.workflow import create_tri_geom_set, build_hip_model

geom_set = create_tri_geom_set(["pelvis_no_sacrum", "femur_r", "tibia_r"], "path/to/bones")
build_hip_model(geom_set, "output", body_mass=64)
```

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

## Releasing

1. Update `__version__` in `src/pystaple/__init__.py` and add a section
   `## [x.y.z] - date` to `CHANGELOG.md`.
2. Commit, then `git tag vx.y.z` and `git push origin vx.y.z`.
3. The release workflow runs all tests, builds the wheel and creates the GitHub
   release with the notes from the changelog.

## License

pystaple is a Python translation of parts of [STAPLE](https://github.com/modenaxe/msk-STAPLE),
Copyright (c) 2020-2021 Luca Modenese and Jean-Baptiste Renault, and is distributed
under the same license, the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/)
(see [LICENSE](LICENSE)). It is free for academic and other non-commercial use;
uses beyond those permitted by the license must be discussed with the authors of
STAPLE. [NOTICE](NOTICE) lists the changes made to the original work.

The test data in `reference/` are derived from the STAPLE `bone_datasets` and keep
their own terms (two of them are CC BY-NC-SA 2.0 BE), see
[reference/DATA_LICENSES.md](reference/DATA_LICENSES.md).

## Citation

If you use pystaple, please cite the STAPLE publication:

> Modenese L., Renault J.-B. (2021). Automatic generation of personalised skeletal
> models of the lower limb from three-dimensional bone geometries.
> *Journal of Biomechanics* 116, 110186.
> https://doi.org/10.1016/j.jbiomech.2020.110186

```bibtex
@article{Modenese2021auto,
  title   = {Automatic Generation of Personalized Skeletal Models of the Lower Limb from Three-Dimensional Bone Geometries},
  author  = {Luca Modenese and Jean-Baptiste Renault},
  journal = {Journal of Biomechanics},
  volume  = {116},
  pages   = {110186},
  year    = {2021},
  doi     = {10.1016/j.jbiomech.2020.110186}
}
```

and the publications of the algorithms used by the hip model:

| Algorithm | Bone | Reference |
|---|---|---|
| STAPLE | pelvis | Modenese & Renault (2021), see above |
| GIBOC | femur | Renault J.-B. et al. (2018). Articular-surface-based automatic anatomical coordinate systems for the knee bones. *J Biomech* 80, 171-178. https://doi.org/10.1016/j.jbiomech.2018.08.028 |
| Kai2014 | tibia | Kai S. et al. (2014). Automatic construction of an anatomical coordinate system for three-dimensional bone models of the lower extremities - Pelvis, femur, and tibia. *J Biomech* 47(5), 1229-1233. https://doi.org/10.1016/j.jbiomech.2013.12.013 |

The STAPLE software itself can be cited as https://doi.org/10.5281/zenodo.4428103.
GitHub's *Cite this repository* button uses [CITATION.cff](CITATION.cff).
