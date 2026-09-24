"""Comparison of two OpenSim models and of their visualization geometries.

Used by tools/compare_osim.py and by the tests to quantify how a model built by
pystaple differs from the one built by MATLAB STAPLE.

Geometries decimated by different algorithms never match vertex by vertex, so
they are compared by shape: area, volume and surface-to-surface distances
(computed from area-uniform samples to the exact closest point on the other
surface).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import numpy as np
from scipy.spatial import cKDTree

from ..mesh import TriMesh
from .xml_reader import read_osim

# --- OBJ files ------------------------------------------------------------------


def read_obj(path: str | Path) -> TriMesh:
    """Minimal OBJ reader (v / f records; 'f a/b/c' and polygons are handled)."""
    verts, faces = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            idx = [int(p.split("/")[0]) for p in parts[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            faces += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return TriMesh(np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64))


def resolve_mesh_file(model_path: str | Path, mesh_file: str) -> Path:
    """Mesh path from an .osim (possibly with Windows separators) relative to the model."""
    parts = PureWindowsPath(mesh_file).parts
    return Path(model_path).parent.joinpath(*parts)


# --- mesh measures ------------------------------------------------------------------


def surface_area(mesh: TriMesh) -> float:
    p = mesh.points[mesh.faces]
    return float(0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum())


def signed_volume(mesh: TriMesh) -> float:
    """Divergence theorem; exact for closed meshes, approximate otherwise."""
    p = mesh.points[mesh.faces]
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6)


def sample_surface(mesh: TriMesh, n: int, seed: int = 0) -> np.ndarray:
    """n points uniformly distributed on the surface (area weighted)."""
    rng = np.random.default_rng(seed)
    p = mesh.points[mesh.faces]
    areas = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    f = rng.choice(len(areas), size=n, p=areas / areas.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    a, b, c = p[f, 0], p[f, 1], p[f, 2]
    return (1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b + (r1 * r2)[:, None] * c


def _closest_on_triangles(P, A, B, C):
    """Closest points on triangles ABC to points P, all (n, 3) (Ericson 2005)."""
    ab, ac = B - A, C - A
    dot = lambda u, v: np.einsum("ij,ij->i", u, v)  # noqa: E731
    ap, bp, cp = P - A, P - B, P - C
    d1, d2 = dot(ab, ap), dot(ac, ap)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1 / (va + vb + vc)
        res = A + ab * (vb * denom)[:, None] + ac * (vc * denom)[:, None]  # interior
        # regions are applied in reverse order of Ericson's early returns (last wins)
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        res[m] = (B + (C - B) * w[:, None])[m]
        w = d2 / (d2 - d6)
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        res[m] = (A + ac * w[:, None])[m]
        m = (d6 >= 0) & (d5 <= d6)
        res[m] = C[m]
        v = d1 / (d1 - d3)
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        res[m] = (A + ab * v[:, None])[m]
        m = (d3 >= 0) & (d4 <= d3)
        res[m] = B[m]
        m = (d1 <= 0) & (d2 <= 0)
        res[m] = A[m]
    return res


def _segment_distance(P, A, B):
    """Distance from points P to segments AB (zero-length segments allowed)."""
    ab = B - A
    L2 = np.einsum("ij,ij->i", ab, ab)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(L2 > 0, np.einsum("ij,ij->i", P - A, ab) / L2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(A + ab * t[:, None] - P, axis=1)


def point_to_surface_distance(points: np.ndarray, mesh: TriMesh, k: int = 24) -> np.ndarray:
    """Distance from each point to the surface of `mesh`.

    Exact distance to the best of the `k` triangles with the nearest centroids
    (sufficient for reasonably uniform meshes; never underestimates).
    Degenerate (zero-area) triangles are handled as their three edges.
    """
    tri = mesh.points[mesh.faces]
    k = min(k, len(tri))
    _, cand = cKDTree(tri.mean(axis=1)).query(points, k=k)
    cand = cand.reshape(len(points), k)
    P = np.repeat(points, k, axis=0)
    t = tri[cand.ravel()]
    A, B, C = t[:, 0], t[:, 1], t[:, 2]
    d = np.linalg.norm(_closest_on_triangles(P, A, B, C) - P, axis=1)

    cross = np.linalg.norm(np.cross(B - A, C - A), axis=1)
    longest2 = np.max([np.einsum("ij,ij->i", u, u) for u in (B - A, C - B, A - C)], axis=0)
    degenerate = (cross <= 1e-12 * np.maximum(longest2, np.finfo(float).tiny)) | ~np.isfinite(d)
    if degenerate.any():
        Pd, Ad, Bd, Cd = P[degenerate], A[degenerate], B[degenerate], C[degenerate]
        d[degenerate] = np.minimum.reduce(
            [_segment_distance(Pd, Ad, Bd), _segment_distance(Pd, Bd, Cd), _segment_distance(Pd, Cd, Ad)]
        )
    return d.reshape(len(points), k).min(axis=1)


def degenerate_faces(mesh: TriMesh) -> int:
    """Number of zero-area (collinear or collapsed) triangles."""
    p = mesh.points[mesh.faces]
    e = [p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]]
    cross = np.linalg.norm(np.cross(e[0], -e[2]), axis=1)
    longest2 = np.max([np.einsum("ij,ij->i", u, u) for u in e], axis=0)
    return int(np.count_nonzero(cross <= 1e-12 * np.maximum(longest2, np.finfo(float).tiny)))


@dataclass
class SurfaceDistance:
    mean: float
    p95: float
    max: float

    def __str__(self) -> str:
        return f"mean {self.mean:.3f} / p95 {self.p95:.3f} / max {self.max:.3f} mm"


def surface_distance(a: TriMesh, b: TriMesh, n: int = 20000, seed: int = 0) -> SurfaceDistance:
    """One-sided distance from the surface of a to the surface of b."""
    d = point_to_surface_distance(sample_surface(a, n, seed), b)
    return SurfaceDistance(float(d.mean()), float(np.percentile(d, 95)), float(d.max()))


def _rel(a: float, b: float) -> float:
    return (a - b) / b if b != 0 else (0.0 if a == b else float("nan"))


def compare_meshes(a: TriMesh, b: TriMesh, n: int = 20000) -> dict:
    """Shape comparison of mesh a (e.g. Python) against mesh b (e.g. MATLAB)."""
    area_a, area_b = surface_area(a), surface_area(b)
    vol_a, vol_b = signed_volume(a), signed_volume(b)
    a_to_b, b_to_a = surface_distance(a, b, n), surface_distance(b, a, n)
    return {
        "n_points": (len(a.points), len(b.points)),
        "n_faces": (len(a.faces), len(b.faces)),
        "n_degenerate": (degenerate_faces(a), degenerate_faces(b)),
        "area_rel_diff": _rel(area_a, area_b),
        "volume_rel_diff": _rel(vol_a, vol_b),
        "bbox_max_diff": float(
            np.max(np.abs(np.r_[a.points.min(0) - b.points.min(0), a.points.max(0) - b.points.max(0)]))
        ),
        "a_to_b": a_to_b,
        "b_to_a": b_to_a,
        "hausdorff": max(a_to_b.max, b_to_a.max),
        "mean_symmetric": 0.5 * (a_to_b.mean + b_to_a.mean),
    }


# --- model comparison ----------------------------------------------------------------


@dataclass
class Diff:
    section: str
    item: str
    quantity: str
    max_abs: float | None  # None for structural (non-numeric) comparisons
    equal: bool
    note: str = ""


def _num(section, item, quantity, a, b) -> Diff:
    if a is None or b is None or np.shape(a) != np.shape(b):
        return Diff(section, item, quantity, None, False, f"python {a} / matlab {b}")
    d = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float)))) if np.size(a) else 0.0
    return Diff(section, item, quantity, d, d == 0.0)


def _same(section, item, quantity, a, b) -> Diff:
    return Diff(section, item, quantity, None, a == b, "" if a == b else f"python {a} / matlab {b}")


def compare_models(py_path: str | Path, ref_path: str | Path) -> list[Diff]:
    """Property-by-property differences between two .osim files."""
    a, b = read_osim(py_path), read_osim(ref_path)
    out = [
        _same("model", "", "name", a["name"], b["name"]),
        _num("model", "", "gravity", a["gravity"], b["gravity"]),
        _same("model", "", "credits", a["credits"], b["credits"]),
        _same("bodies", "", "names (order)", list(a["bodies"]), list(b["bodies"])),
        _same("joints", "", "names (order)", list(a["joints"]), list(b["joints"])),
        _same("markers", "", "names (order)", list(a["markers"]), list(b["markers"])),
    ]
    for name in [n for n in b["bodies"] if n in a["bodies"]]:
        pa, pb = a["bodies"][name], b["bodies"][name]
        out += [
            _num("bodies", name, "mass [kg]", pa["mass"], pb["mass"]),
            _num("bodies", name, "mass_center [m]", pa["mass_center"], pb["mass_center"]),
            _num("bodies", name, "inertia [kg m2]", pa["inertia"], pb["inertia"]),
            _same("bodies", name, "mesh files", [PureWindowsPath(m["file"]).parts for m in pa["meshes"]],
                  [PureWindowsPath(m["file"]).parts for m in pb["meshes"]]),
        ]
        for ma, mb in zip(pa["meshes"], pb["meshes"]):
            out.append(_num("bodies", name, f"{mb['name']} scale_factors", ma["scale_factors"], mb["scale_factors"]))
    for name in [n for n in b["joints"] if n in a["joints"]]:
        ja, jb = a["joints"][name], b["joints"][name]
        out += [
            _same("joints", name, "type", ja["type"], jb["type"]),
            _same("joints", name, "parent/child frames",
                  (ja["parent_frame"], ja["child_frame"]), (jb["parent_frame"], jb["child_frame"])),
            _same("joints", name, "coordinates", list(ja["coordinates"]), list(jb["coordinates"])),
        ]
        for c in [c for c in jb["coordinates"] if c in ja["coordinates"]]:
            out.append(_num("joints", name, f"{c} range", ja["coordinates"][c]["range"], jb["coordinates"][c]["range"]))
        for f in [f for f in jb["frames"] if f in ja["frames"]]:
            fa, fb = ja["frames"][f], jb["frames"][f]
            out += [
                _same("joints", name, f"{f} parent", fa["parent"], fb["parent"]),
                _num("joints", name, f"{f} translation [m]", fa["translation"], fb["translation"]),
                _num("joints", name, f"{f} orientation [rad]", fa["orientation"], fb["orientation"]),
            ]
        for ax in [x for x in jb["axes"] if x in ja["axes"]]:
            xa, xb = ja["axes"][ax], jb["axes"][ax]
            out += [
                _same("joints", name, f"{ax} coordinate/function",
                      (xa["coordinates"], xa["function"]), (xb["coordinates"], xb["function"])),
                _num("joints", name, f"{ax} axis", xa["axis"], xb["axis"]),
            ]
    for name in [n for n in b["markers"] if n in a["markers"]]:
        ma, mb = a["markers"][name], b["markers"][name]
        out += [
            _same("markers", name, "parent frame", ma["parent_frame"], mb["parent_frame"]),
            _num("markers", name, "location [m]", ma["location"], mb["location"]),
        ]
    return out


def compare_model_geometries(
    py_path: str | Path, ref_path: str | Path, originals: dict[str, TriMesh] | None = None, n: int = 20000
) -> dict[str, dict]:
    """Compare the mesh files of each body of two models.

    `originals` (body name -> full-resolution bone mesh) adds the deviation of
    each decimated mesh from the original surface ("py_vs_original",
    "ref_vs_original"), i.e. how faithful each decimation is.
    """
    a, b = read_osim(py_path), read_osim(ref_path)
    out = {}
    for name in [n for n in b["bodies"] if n in a["bodies"]]:
        for ma, mb in zip(a["bodies"][name]["meshes"], b["bodies"][name]["meshes"]):
            pa, pb = resolve_mesh_file(py_path, ma["file"]), resolve_mesh_file(ref_path, mb["file"])
            if not (pa.is_file() and pb.is_file()):
                out[name] = {"missing": [str(p) for p in (pa, pb) if not p.is_file()]}
                continue
            mesh_a, mesh_b = read_obj(pa), read_obj(pb)
            res = compare_meshes(mesh_a, mesh_b, n)
            orig = (originals or {}).get(name)
            if orig is not None:
                res["n_faces_original"] = len(orig.faces)
                res["py_vs_original"] = compare_meshes(mesh_a, orig, n)
                res["ref_vs_original"] = compare_meshes(mesh_b, orig, n)
            out[name] = res
    return out
