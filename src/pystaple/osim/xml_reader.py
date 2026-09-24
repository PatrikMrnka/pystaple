"""Reading .osim files into plain dictionaries (no OpenSim needed).

Used to compare generated models with the MATLAB reference model.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _vec(text: str | None) -> np.ndarray | None:
    return None if text is None else np.array([float(t) for t in text.split()])


def read_osim(path: str | Path) -> dict:
    """Model name, gravity, credits, bodies, joints and markers of an .osim file.

    Dictionaries keep the order of the file. Values of properties that are not
    written in the file (OpenSim omits defaults) are None.
    """
    model = ET.parse(path).getroot().find("Model")
    out = {
        "name": model.get("name"),
        "gravity": _vec(model.findtext("gravity")),
        "credits": model.findtext("credits"),
        "bodies": {},
        "joints": {},
        "markers": {},
    }
    for b in model.findall("BodySet/objects/Body"):
        out["bodies"][b.get("name")] = {
            "mass": float(b.findtext("mass", "nan")),
            "mass_center": _vec(b.findtext("mass_center")),
            "inertia": _vec(b.findtext("inertia")),
            "meshes": [
                {
                    "name": m.get("name"),
                    "file": m.findtext("mesh_file"),
                    "scale_factors": _vec(m.findtext("scale_factors")),
                }
                for m in b.findall("attached_geometry/Mesh")
            ],
        }
    for j in model.find("JointSet/objects"):
        axes = {}
        for ta in j.findall("SpatialTransform/TransformAxis"):
            fn = [e for e in ta if e.get("name") == "function"]
            axes[ta.get("name")] = {
                "coordinates": (ta.findtext("coordinates") or "").split(),
                "axis": _vec(ta.findtext("axis")),
                "function": fn[0].tag if fn else None,
            }
        out["joints"][j.get("name")] = {
            "type": j.tag,
            "parent_frame": j.findtext("socket_parent_frame"),
            "child_frame": j.findtext("socket_child_frame"),
            "coordinates": {
                c.get("name"): {"range": _vec(c.findtext("range"))}
                for c in j.findall("coordinates/Coordinate")
            },
            "frames": {
                f.get("name"): {
                    "parent": f.findtext("socket_parent"),
                    "translation": _vec(f.findtext("translation")),
                    "orientation": _vec(f.findtext("orientation")),
                }
                for f in j.findall("frames/PhysicalOffsetFrame")
            },
            "axes": axes,
        }
    for m in model.findall("MarkerSet/objects/Marker"):
        out["markers"][m.get("name")] = {
            "parent_frame": m.findtext("socket_parent_frame"),
            "location": _vec(m.findtext("location")),
        }
    return out
