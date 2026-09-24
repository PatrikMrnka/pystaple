pystaple - standalone command line tool (no Python, MATLAB or OpenSim needed)
==============================================================================

Builds an OpenSim model of the hip (pelvis, right femur, right tibia) from bone
geometries, like hip_model.m of the STAPLE toolbox.

Usage (in a command prompt / PowerShell, in this folder):

    pystaple.exe hip-model <bones_folder> <output_folder> --body-mass 64

<bones_folder> must contain pelvis_no_sacrum.stl, femur_r.stl and tibia_r.stl
(surface meshes in millimetres). The output folder gets bone_model.osim,
the Geometry folder and a log file. Open bone_model.osim in OpenSim.

    pystaple.exe hip-model --help      all options
    pystaple.exe --version

Keep the whole folder together (pystaple.exe needs the _internal folder).
Windows may warn about an "unknown publisher" the first time: choose
"More info" -> "Run anyway".

License: CC BY-NC 4.0, non-commercial use only (see LICENSE and NOTICE).
Please cite: Modenese L., Renault J.-B. (2021), J Biomech 116, 110186,
https://doi.org/10.1016/j.jbiomech.2020.110186
