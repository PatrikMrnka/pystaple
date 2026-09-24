"""Port of processTriGeomBoneSet.m (only the algorithms ported so far)."""

from __future__ import annotations

from .algorithms.femur import giboc_femur
from .algorithms.pelvis import staple_pelvis
from .algorithms.tibia import kai2014_tibia
from .mesh import TriMesh
from .utils import body_side_to_sign


def process_tri_geom_bone_set(
    geom_set: dict[str, TriMesh],
    side: str = "r",
    algo_pelvis: str = "STAPLE",
    algo_femur: str = "GIBOC-cylinder",
    algo_tibia: str = "Kai2014",
    in_mm: bool = True,
):
    """Returns (JCS, BL, BCS), each a dict keyed by bone name, as in MATLAB."""
    _, s = body_side_to_sign(side)
    JCS, BL, BCS = {}, {}, {}

    pelvis_key = next((k for k in ("pelvis", "pelvis_no_sacrum") if k in geom_set), None)
    if pelvis_key:
        if algo_pelvis != "STAPLE":
            raise NotImplementedError(f"pelvis algorithm {algo_pelvis!r} not ported yet")
        BCS["pelvis"], JCS["pelvis"], BL["pelvis"] = staple_pelvis(geom_set[pelvis_key], s, in_mm)

    if f"femur_{s}" in geom_set:
        if algo_femur != "GIBOC-cylinder":
            raise NotImplementedError(f"femur algorithm {algo_femur!r} not ported yet")
        key = f"femur_{s}"
        BCS[key], JCS[key], BL[key], _ = giboc_femur(geom_set[key], s, "cylinder", in_mm)
    if f"tibia_{s}" in geom_set:
        if algo_tibia != "Kai2014":
            raise NotImplementedError(f"tibia algorithm {algo_tibia!r} not ported yet")
        key = f"tibia_{s}"
        BCS[key], JCS[key], BL[key] = kai2014_tibia(geom_set[key], s, in_mm)

    return JCS, BL, BCS
