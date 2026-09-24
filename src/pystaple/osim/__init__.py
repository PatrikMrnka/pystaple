"""OpenSim model generation (port of STAPLE/opensim and STAPLE/anthropometry).

Everything is pure Python: models are described with plain data and written
as .osim files by ``writer``. Only the optional ``model`` backend (the same model
through the OpenSim API) needs the ``opensim`` package.
"""

from .anthropometry import segment_mass_props
from .geometry_files import write_model_geometries_folder
from .joints import finalize_joint_struct, infer_body_side
from .writer import model_description, osim_text, write_osim
from .xml_reader import read_osim

__all__ = [
    "finalize_joint_struct",
    "infer_body_side",
    "model_description",
    "osim_text",
    "read_osim",
    "segment_mass_props",
    "write_model_geometries_folder",
    "write_osim",
]
