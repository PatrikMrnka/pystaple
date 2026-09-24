"""Compare OpenSim models built by pystaple with the MATLAB STAPLE models.

All datasets that have reference/<dataset>/osim/bone_model.osim (+ Geometry/):

    python tools/compare_osim.py                       # build, compare, summary table
    python tools/compare_osim.py --details             # + full report per dataset
    python tools/compare_osim.py ICL_MRI JIA_MRI       # only some datasets
    python tools/compare_osim.py --report compare.txt  # also write everything to a file
    python tools/compare_osim.py --no-build            # reuse models in out/compare

One pair of existing models:

    python tools/compare_osim.py --pair out_icl/bone_model.osim reference/ICL_MRI/osim/bone_model.osim ^
        --originals reference/ICL_MRI

Python models are written to out/compare/<dataset>/ (bone_model.osim + Geometry).
The geometries are compared by shape (Python vs MATLAB) and each decimated mesh
is also compared with the original full-resolution bone ("fidelity").
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

from pystaple.osim.compare import compare_model_geometries, compare_models

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = Path(os.environ.get("PYSTAPLE_REFERENCE_DIR", ROOT / "reference"))
BONES = ("pelvis_no_sacrum", "femur_r", "tibia_r")
BODY_MASS = 64.0


class Out:
    """print() to the console and, optionally, to a report file."""

    def __init__(self, path=None):
        self.file = open(path, "w", encoding="utf-8") if path else None

    def __call__(self, text="", console=True):
        if console:
            print(text)
        if self.file:
            self.file.write(text + "\n")

    def close(self):
        if self.file:
            self.file.close()


def load_originals(folder, body_names):
    """Full-resolution bones, keyed by body name (pelvis_no_sacrum -> pelvis)."""
    from pystaple.workflow import create_tri_geom_set

    out = {}
    for body in body_names:
        candidates = ["pelvis_no_sacrum", "pelvis"] if body == "pelvis" else [body]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for bone in candidates:
                try:
                    out[body] = create_tri_geom_set([bone], folder)[bone]
                    break
                except FileNotFoundError:
                    continue
    return out


def build_python_model(dataset: str, out_dir: Path) -> Path:
    from pystaple.workflow import build_hip_model, create_tri_geom_set

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the same warnings are printed by MATLAB
        geom_set = create_tri_geom_set(BONES, REFERENCE_DIR / dataset)
        path, *_ = build_hip_model(geom_set, out_dir, body_mass=BODY_MASS)
    return path


# --- reports ------------------------------------------------------------------------


def detailed_report(out, py_model, ref_model, diffs, geo, show_all=False, console=True):
    p = lambda text="": out(text, console)  # noqa: E731
    p(f"\n=== .osim: {py_model}  vs  {ref_model} ===")
    structural = [d for d in diffs if d.max_abs is None]
    numeric = [d for d in diffs if d.max_abs is not None]
    bad = [d for d in structural if not d.equal]
    p(f"structural properties: {len(structural)} compared, {len(bad)} different")
    for d in bad:
        p(f"  DIFFERENT  {d.section:8s} {d.item:14s} {d.quantity}: {d.note}")
    p(f"numeric properties: {len(numeric)} compared, {sum(d.equal for d in numeric)} bit-identical")
    p(f"  {'section':8s} {'item':14s} {'quantity':34s} {'max |diff|':>11s}")
    for d in numeric:
        if show_all or not d.equal:
            p(f"  {d.section:8s} {d.item:14s} {d.quantity:34s} {d.max_abs:11.3g}")

    p("\n--- visualization geometries (Python vs MATLAB, distances in mm) ---")
    for body, r in geo.items():
        p(f"{body}")
        if "missing" in r:
            p(f"  missing files: {', '.join(r['missing'])}")
            continue
        n_orig = f"  (original {r['n_faces_original']})" if "n_faces_original" in r else ""
        p(f"  faces     python {r['n_faces'][0]:7d}  matlab {r['n_faces'][1]:7d}{n_orig}")
        p(f"  vertices  python {r['n_points'][0]:7d}  matlab {r['n_points'][1]:7d}")
        if any(r["n_degenerate"]):
            p(f"  zero-area faces  python {r['n_degenerate'][0]}  matlab {r['n_degenerate'][1]}")
        p(f"  area {100 * r['area_rel_diff']:+.3f} %   volume {100 * r['volume_rel_diff']:+.3f} %   "
          f"bounding box max diff {r['bbox_max_diff']:.3f}")
        p(f"  python -> matlab surface: {r['a_to_b']}")
        p(f"  matlab -> python surface: {r['b_to_a']}")
        for key, label in (("py_vs_original", "python"), ("ref_vs_original", "matlab")):
            if key in r:
                o = r[key]
                p(f"  {label:6s} vs original: mean {o['mean_symmetric']:.3f}, hausdorff {o['hausdorff']:.3f}, "
                  f"area {100 * o['area_rel_diff']:+.3f} %, volume {100 * o['volume_rel_diff']:+.3f} %")


def summarize(diffs, geo) -> dict:
    num = [d for d in diffs if d.max_abs is not None]
    pick = lambda cond: max((d.max_abs for d in num if cond(d.quantity)), default=0.0)  # noqa: E731
    ok_geo = [r for r in geo.values() if "missing" not in r]
    gmax = lambda f: max((f(r) for r in ok_geo), default=float("nan"))  # noqa: E731
    return {
        "structural": sum(1 for d in diffs if d.max_abs is None and not d.equal),
        "identical": f"{sum(d.equal for d in num)}/{len(num)}",
        "pos_mm": 1000 * pick(lambda q: "[m]" in q),
        "angle": pick(lambda q: "[rad]" in q or q.endswith(" axis")),
        "mass": pick(lambda q: "[kg" in q),
        "geo_mean": gmax(lambda r: r["mean_symmetric"]),
        "geo_haus": gmax(lambda r: r["hausdorff"]),
        "fid_py": gmax(lambda r: r["py_vs_original"]["mean_symmetric"] if "py_vs_original" in r else float("nan")),
        "fid_ml": gmax(lambda r: r["ref_vs_original"]["mean_symmetric"] if "ref_vs_original" in r else float("nan")),
        "fidh_py": gmax(lambda r: r["py_vs_original"]["hausdorff"] if "py_vs_original" in r else float("nan")),
        "fidh_ml": gmax(lambda r: r["ref_vs_original"]["hausdorff"] if "ref_vs_original" in r else float("nan")),
        "geo_missing": sum(1 for r in geo.values() if "missing" in r),
    }


def summary_table(out, rows):
    out("\n=== SUMMARY (python vs matlab) ===")
    out("osim : struct = structural differences, identical = bit-identical numbers,")
    out("       pos = max position diff [mm] (joint frames, COMs, markers), angle = max [rad], mass = max [kg / kg m2]")
    out("geom : mean / hausdorff = surface distance python<->matlab mesh [mm] (worst bone),")
    out("       fidelity = distance of the decimated mesh to the original bone [mm], mean and hausdorff,")
    out("       python / matlab (lower = more faithful)")
    out(f"{'dataset':22s} {'struct':>6s} {'identical':>9s} {'pos':>9s} {'angle':>9s} {'mass':>9s}"
        f" {'mean':>7s} {'hausd.':>7s} {'fid. py':>8s} {'fid. ml':>8s} {'fidH py':>8s} {'fidH ml':>8s}")
    for name, s in rows:
        if isinstance(s, str):
            out(f"{name:22s} {s}")
            continue
        miss = f"  ({s['geo_missing']} geometries missing)" if s["geo_missing"] else ""
        out(f"{name:22s} {s['structural']:6d} {s['identical']:>9s} {s['pos_mm']:9.2g} {s['angle']:9.2g} "
            f"{s['mass']:9.2g} {s['geo_mean']:7.3f} {s['geo_haus']:7.3f} {s['fid_py']:8.3f} {s['fid_ml']:8.3f} "
            f"{s['fidh_py']:8.3f} {s['fidh_ml']:8.3f}{miss}")


# --- main ---------------------------------------------------------------------------------


def osim_datasets():
    if not REFERENCE_DIR.is_dir():
        return []
    return sorted(d.name for d in REFERENCE_DIR.iterdir() if (d / "osim" / "bone_model.osim").is_file())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help="datasets in the reference folder (default: all with osim/)")
    ap.add_argument("--pair", nargs=2, metavar=("PYTHON_OSIM", "MATLAB_OSIM"), help="compare two existing models")
    ap.add_argument("--originals", help="(--pair) folder with the full-resolution bone meshes")
    ap.add_argument("--out", default=str(ROOT / "out" / "compare"), help="where the Python models are built")
    ap.add_argument("--no-build", action="store_true", help="reuse Python models already in --out")
    ap.add_argument("--details", action="store_true", help="print the full report of each dataset")
    ap.add_argument("--all", action="store_true", help="also list numeric properties that are identical")
    ap.add_argument("--samples", type=int, default=20000, help="surface samples per mesh")
    ap.add_argument("--report", help="write the full report (all details) to this file")
    args = ap.parse_args()
    out = Out(args.report)

    if args.pair:
        py, ref = args.pair
        diffs = compare_models(py, ref)
        from pystaple.osim import read_osim

        originals = load_originals(args.originals, list(read_osim(ref)["bodies"])) if args.originals else None
        geo = compare_model_geometries(py, ref, originals, args.samples)
        detailed_report(out, py, ref, diffs, geo, args.all)
        out.close()
        return

    datasets = args.datasets or osim_datasets()
    if not datasets:
        sys.exit(f"no dataset with osim/bone_model.osim in {REFERENCE_DIR}")
    rows = []
    for ds in datasets:
        ref = REFERENCE_DIR / ds / "osim" / "bone_model.osim"
        py = Path(args.out) / ds / "bone_model.osim"
        print(f"[{ds}] ", end="", flush=True)
        if not ref.is_file():
            rows.append((ds, f"no MATLAB model: {ref}"))
            print("no MATLAB model")
            continue
        try:
            t0 = time.perf_counter()
            if not args.no_build or not py.is_file():
                print("building python model... ", end="", flush=True)
                py = build_python_model(ds, py.parent)
            print("comparing... ", end="", flush=True)
            diffs = compare_models(py, ref)
            from pystaple.osim import read_osim

            originals = load_originals(REFERENCE_DIR / ds, list(read_osim(ref)["bodies"]))
            geo = compare_model_geometries(py, ref, originals, args.samples)
            print(f"done ({time.perf_counter() - t0:.0f} s)")
        except Exception as exc:  # report and continue with the other datasets
            rows.append((ds, f"FAILED: {type(exc).__name__}: {exc}"))
            print("FAILED")
            out(traceback.format_exc(), console=False)
            continue
        out(f"\n##### {ds}", console=args.details)
        detailed_report(out, py, ref, diffs, geo, args.all, console=args.details)
        rows.append((ds, summarize(diffs, geo)))

    summary_table(out, rows)
    if args.report:
        out(f"\nfull report: {args.report}")
    out.close()


if __name__ == "__main__":
    main()
