# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-09-25

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
- Verification against MATLAB STAPLE on 8 datasets (`reference/`,
  `tools/compare_osim.py`): the 7 datasets of STAPLE's `bone_datasets` and
  `MSKPIPE_CT`, bones segmented automatically from an LHDL CT scan
  (CC BY-NC-SA 2.0 BE) by [msk-PIPE](https://github.com/PatrikMrnka/msk-PIPE).
  Identical models: no structural difference, positions within 1e-7 mm,
  bit-identical masses and inertias.
- Reference data as a release asset (`pystaple-<version>-reference-data.zip`,
  the `reference/` folder with the terms of each dataset).
- `tools/export_reference_dataset.m`: writes a new reference dataset
  (`mesh_<bone>.mat`, `reference.json`, `osim/`) from a folder of bone meshes with
  MATLAB STAPLE.
- License (CC BY-NC 4.0, as STAPLE), NOTICE with the changes made to STAPLE,
  CITATION.cff and the terms of the reference datasets.
