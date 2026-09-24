"""OpenSim model generation (port of STAPLE/opensim and STAPLE/anthropometry).

Only ``model`` and ``workflow.build_hip_model`` need the ``opensim`` package;
joint definitions, mass properties, geometry files and the .osim reader are
pure Python.
"""

from .anthropometry import segment_mass_props
from .geometry_files import write_model_geometries_folder
from .joints import finalize_joint_struct, infer_body_side
from .xml_reader import read_osim

__all__ = [
    "finalize_joint_struct",
    "infer_body_side",
    "read_osim",
    "segment_mass_props",
    "write_model_geometries_folder",
]
