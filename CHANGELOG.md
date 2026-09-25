# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Reference dataset `MSKPIPE_CT`: pelvis, right femur and tibia segmented
  automatically from an LHDL CT scan (CC BY-NC-SA 2.0 BE) by [msk-PIPE](https://github.com/PatrikMrnka/msk-PIPE)
  (TotalSegmentator, VTK meshing). pystaple and MATLAB STAPLE give the same model
  (no structural difference, positions within 2e-11 mm, angles within 5e-13 rad,
  bit-identical masses and inertias); the Python visualization meshes are closer
  to the original bones than MATLAB's for all three bones.
- `tools/export_reference_dataset.m`: writes a new reference dataset
  (`mesh_<bone>.mat`, `reference.json`, `osim/`) from a folder of bone meshes with
  MATLAB STAPLE.

## [0.1.0] - 2026-09-24

First release: the `hip_model.m` workflow of STAPLE without MATLAB.

### Added
- Bone analyses: STAPLE pelvis, GIBOC femur (cylinder fit on the posterior
  condyles), Kai2014 tibia, with the GIBOC-core mesh utilities they need.
- OpenSim model generation (auto2020 joint definitions, gait2392 mass properties
  scaled to the body mass, bone landmarks as markers, visualization geometries).
  The `.osim` files are written directly, OpenSim is not required; the OpenSim
  API is available as an optional backend (`--backend opensim`).
- `pystaple hip-model` command line interface (also `python -m pystaple`).
- Standalone Windows executable (`pystaple-<version>-windows-x64.zip`), no Python
  or OpenSim needed.
- Verification against MATLAB STAPLE on the 7 datasets of `bone_datasets`
  (`reference/`, `tools/compare_osim.py`): identical models within 1e-7 mm.
- License (CC BY-NC 4.0, as STAPLE), NOTICE with the changes made to STAPLE,
  CITATION.cff and the terms of the reference datasets.
