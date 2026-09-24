# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-09-24

First release: the `hip_model.m` workflow of STAPLE without MATLAB.

### Added
- Bone analyses: STAPLE pelvis, GIBOC femur (cylinder fit on the posterior
  condyles), Kai2014 tibia, with the GIBOC-core mesh utilities they need.
- OpenSim model generation (auto2020 joint definitions, gait2392 mass properties
  scaled to the body mass, bone landmarks as markers, visualization geometries).
- `pystaple hip-model` command line interface (also `python -m pystaple`).
- Verification against MATLAB STAPLE on the 7 datasets of `bone_datasets`
  (`reference/`, `tools/compare_osim.py`): identical models within 1e-7 mm.
- License (CC BY-NC 4.0, as STAPLE), NOTICE with the changes made to STAPLE,
  CITATION.cff and the terms of the reference datasets.
