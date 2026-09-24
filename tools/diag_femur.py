"""Compare GIBOC_femur intermediate results with MATLAB (diag_femur.json).

    python tools/diag_femur.py JIA_MRI

Prints each quantity from MATLAB and Python in pipeline order; the first row
with a large difference shows where the two implementations diverge.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

from pystaple.algorithms.femur import giboc_femur
from pystaple.fitting import fit_csa
from pystaple.io import load_reference_json, load_reference_mesh

ref_dir = Path(__file__).parents[1] / "reference"
dataset = sys.argv[1] if len(sys.argv) > 1 else "JIA_MRI"
m = load_reference_json(ref_dir / dataset / "diag_femur.json")
mesh = load_reference_mesh(ref_dir / dataset / "mesh_femur_r.mat")


def run(z_epi=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, JCS, _, aux = giboc_femur(mesh, "r", z_epi=z_epi)
    lat, med = aux["_postCondyle_Lat"], aux["_postCondyle_Med"]
    return {
        "Zepi": aux["Zepi"],
        "epi_n_points": len(aux["_EpiFem"].points),
        "epi_n_faces": len(aux["_EpiFem"].faces),
        "X1": aux["X1"], "Y1": aux["Y1"], "PtNotch": aux["PtNotch"],
        "post_lat_n_points": len(lat.points), "post_lat_n_faces": len(lat.faces),
        "post_lat_points_mean": lat.points.mean(axis=0),
        "post_med_n_points": len(med.points), "post_med_n_faces": len(med.faces),
        "post_med_points_mean": med.points.mean(axis=0),
        "sphere_center_lat": aux["sphere_center_lat"], "sphere_center_med": aux["sphere_center_med"],
        "Cyl_Pt": aux["Cyl_Pt"], "Cyl_Y": aux["Cyl_Y"], "Cyl_Radius": aux["Cyl_Radius"],
        "knee_origin": JCS["knee_r"]["Origin"],
    }, aux


def report(title, py):
    print(f"\n=== {title} ===")
    print(f"{'quantity':24s} {'max |diff|':>12s}   matlab / python")
    for key, pv in py.items():
        mv = np.asarray(m[key], dtype=float).ravel()
        pv = np.asarray(pv, dtype=float).ravel()
        diff = np.max(np.abs(mv - pv)) if mv.shape == pv.shape else np.inf
        flag = "  <--" if diff > 1e-6 else ""
        print(f"{key:24s} {diff:12.3g}   {np.array2string(mv, precision=4)} / "
              f"{np.array2string(pv, precision=4)}{flag}")


py, aux = run()
alt_m, areas_m = np.asarray(m["slice_alt"], float), np.asarray(m["slice_areas"], float)
print(f"slices: matlab {len(alt_m)}, python {len(aux['_slice_alt'])}")
if len(alt_m) == len(aux["_slice_alt"]):
    print(f"max |diff| altitudes {np.max(np.abs(alt_m - aux['_slice_alt'])):.3g}, "
          f"areas {np.max(np.abs(areas_m - aux['_slice_areas'])):.3g}")
print(f"fit_csa on MATLAB's slices: Zepi = {fit_csa(alt_m, areas_m)[0]:.9f} "
      f"(MATLAB {m['Zepi']:.9f})")

report("Python pipeline", py)
py2, aux2 = run(z_epi=float(m["Zepi"]))
report("Python pipeline with MATLAB's Zepi", py2)

# --- convex hull / condyle axes, on the identical epiphysis ------------------------
if "hull" in m:
    from scipy.spatial import ConvexHull, Delaunay

    from pystaple.algorithms.femur import giboc_femur_process_epiphysis
    from pystaple.geometry import largest_edge_conv_hull

    h = m["hull"]
    epi = aux2["_EpiFem"]
    ch = ConvexHull(epi.points)
    dl = Delaunay(epi.points).convex_hull
    print("\n=== convex hull of the epiphysis ===")
    print(f"MATLAB convhull simplify=false : {h['n_tri']} triangles, {h['n_vertices']} vertices")
    print(f"MATLAB convhull (simplified)   : {h['simplified_n_tri']} triangles, {h['simplified_n_vertices']} vertices")
    print(f"scipy ConvexHull               : {len(ch.simplices)} triangles, {len(ch.vertices)} vertices")
    print(f"scipy Delaunay(...).convex_hull: {len(dl)} triangles, {len(np.unique(dl))} vertices")

    pairs, lengths = largest_edge_conv_hull(epi.points)
    mp = np.asarray(h["top_pairs"], dtype=np.int64) - 1
    ml = np.asarray(h["top_lengths"], dtype=float)
    n = min(len(mp), len(pairs))
    same_len = np.max(np.abs(ml[:n] - lengths[:n]))
    as_set = lambda a: {tuple(sorted(r)) for r in a.tolist()}  # noqa: E731
    print(f"top {n} hull edges: max |length diff| {same_len:.3g}, "
          f"edges only in MATLAB {len(as_set(mp[:n]) - as_set(pairs[:n]))}, "
          f"only in Python {len(as_set(pairs[:n]) - as_set(mp[:n]))}")

    id_py, u_py, _ = giboc_femur_process_epiphysis(epi, aux2)
    id_m = np.asarray(h["IdCdlPts"], dtype=np.int64).reshape(-1, 2) - 1
    print(f"condyle pairs kept: MATLAB {len(id_m)}, Python {len(id_py)}, "
          f"common {len(as_set(id_m) & as_set(id_py))}")
    print(f"sum(U_Axes): MATLAB {np.round(h['U_Axes_sum'], 4)}, Python {np.round(u_py.sum(axis=0), 4)}")
