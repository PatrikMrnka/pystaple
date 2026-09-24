"""Figures comparing the pystaple models with the MATLAB STAPLE models.

    python tools/plot_comparison.py                    # all datasets, figures in out/figures
    python tools/plot_comparison.py JIA_MRI MC22       # some datasets
    python tools/plot_comparison.py --out docs/validation --no-maps

Uses the Python models in out/compare/<dataset>/ (built by tools/compare_osim.py;
missing ones are built here, --build rebuilds all) and the MATLAB models in
reference/<dataset>/osim/. Writes PNG figures and report.md, which embeds them.

Needs matplotlib (pip install matplotlib).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_osim as co  # noqa: E402

from pystaple.osim import read_osim  # noqa: E402
from pystaple.osim.compare import (  # noqa: E402
    compare_models,
    point_to_surface_distance,
    read_obj,
    resolve_mesh_file,
    sample_surface,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm, Normalize
except ImportError:  # pragma: no cover
    sys.exit("matplotlib is required: pip install matplotlib")

PY_COLOR, ML_COLOR = "#1f77b4", "#d62728"
BONE_LABELS = {"pelvis": "pelvis", "femur_r": "femur", "tibia_r": "tibia"}
ZERO = 1e-17  # stands for "bit-identical" on log axes

# quantity groups of the .osim comparison: (label, unit, selector, factor to unit)
GROUPS = [
    ("joint frame origins", "mm", lambda d: d.section == "joints" and "translation" in d.quantity, 1000),
    ("joint frame orientations", "rad", lambda d: d.section == "joints" and "orientation" in d.quantity, 1),
    ("joint axes", "-", lambda d: d.section == "joints" and d.quantity.endswith(" axis"), 1),
    ("coordinate ranges", "rad", lambda d: d.section == "joints" and d.quantity.endswith(" range"), 1),
    ("centres of mass", "mm", lambda d: d.section == "bodies" and "mass_center" in d.quantity, 1000),
    ("masses", "kg", lambda d: d.section == "bodies" and d.quantity.startswith("mass [kg]"), 1),
    ("inertias", "kg m²", lambda d: d.section == "bodies" and d.quantity.startswith("inertia"), 1),
    ("markers", "mm", lambda d: d.section == "markers" and "location" in d.quantity, 1000),
]


# --- data ---------------------------------------------------------------------------


def symmetric_distances(mesh, original, n):
    """Surface samples of mesh -> original and original -> mesh (mm)."""
    a = point_to_surface_distance(sample_surface(mesh, n, seed=1), original)
    b = point_to_surface_distance(sample_surface(original, n, seed=2), mesh)
    return np.r_[a, b]


def _rot_xyz(angles):
    """Rotation of OpenSim body-fixed X-Y-Z angles (columns = frame axes)."""
    a, b, c = angles
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def view_frame(original, body, model=None):
    """Columns: screen horizontal, screen vertical, towards the viewer (front view).

    Uses the anatomical frame STAPLE computed for the bone (a joint frame attached
    to the body: X anterior, Y proximal, Z lateral/right), so bones stand upright
    and "front" is anterior. Falls back to the principal axes of the bone.
    """
    if model is not None:
        for joint in model["joints"].values():
            for frame in joint["frames"].values():
                if frame["parent"] == f"/bodyset/{body}" and frame["orientation"] is not None:
                    R = _rot_xyz(frame["orientation"])
                    X, Y, Z = R[:, 0], R[:, 1], R[:, 2]
                    return np.column_stack([-Z, Y, X])  # viewer in front: subject's right on the left
    pts = original.points - original.points.mean(axis=0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    pc = vt.T
    if body.startswith("pelvis"):  # widest dimension horizontal
        return np.column_stack([pc[:, 0], pc[:, 1], np.cross(pc[:, 0], pc[:, 1])])
    return np.column_stack([pc[:, 1], pc[:, 0], np.cross(pc[:, 1], pc[:, 0])])


def face_deviation(mesh, original):
    """Per triangle: largest distance to the original surface among its vertices,
    edge midpoints and centroid (vertices alone miss folded or bridging triangles,
    e.g. reducepatch keeps the original vertices)."""
    tri = mesh.points[mesh.faces]
    probes = [tri[:, 0], tri[:, 1], tri[:, 2],
              0.5 * (tri[:, 0] + tri[:, 1]), 0.5 * (tri[:, 1] + tri[:, 2]), 0.5 * (tri[:, 2] + tri[:, 0]),
              tri.mean(axis=1)]
    d = point_to_surface_distance(np.vstack(probes), original)
    return d.reshape(len(probes), -1).max(axis=0)


def collect_dataset(ds, models_dir, build, n_samples):
    ref_path = co.REFERENCE_DIR / ds / "osim" / "bone_model.osim"
    py_path = models_dir / ds / "bone_model.osim"
    if build or not py_path.is_file():
        print("building python model... ", end="", flush=True)
        py_path = co.build_python_model(ds, py_path.parent)
    diffs = compare_models(py_path, ref_path)
    ref_model, py_model = read_osim(ref_path), read_osim(py_path)
    originals = co.load_originals(co.REFERENCE_DIR / ds, list(ref_model["bodies"]))
    bones = {}
    for body in ref_model["bodies"]:
        if body not in py_model["bodies"] or body not in originals:
            continue
        m_py = read_obj(resolve_mesh_file(py_path, py_model["bodies"][body]["meshes"][0]["file"]))
        m_ml = read_obj(resolve_mesh_file(ref_path, ref_model["bodies"][body]["meshes"][0]["file"]))
        orig = originals[body]
        d_py, d_ml = symmetric_distances(m_py, orig, n_samples), symmetric_distances(m_ml, orig, n_samples)
        bones[body] = {
            "frame": view_frame(orig, body, ref_model),
            "mesh_py": m_py, "mesh_ml": m_ml, "orig": orig,
            "dist_py": d_py, "dist_ml": d_ml,
            "mean_py": float(d_py.mean()), "mean_ml": float(d_ml.mean()),
            "max_py": float(d_py.max()), "max_ml": float(d_ml.max()),
            "faces": (len(m_py.faces), len(m_ml.faces), len(orig.faces)),
        }
    return {"diffs": diffs, "bones": bones}


def group_max(diffs, select, factor):
    vals = [d.max_abs for d in diffs if d.max_abs is not None and select(d)]
    return max(vals) * factor if vals else np.nan


# --- figures ------------------------------------------------------------------------------


def fig_model_context(data, path):
    """Largest position / angle difference per dataset next to meaningful scales."""
    names = list(data)
    pos = [max(group_max(data[d]["diffs"], s, f) for lbl, u, s, f in GROUPS if u == "mm") for d in names]
    ang = [max(group_max(data[d]["diffs"], s, f) for lbl, u, s, f in GROUPS if u == "rad") for d in names]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.45 * len(names) + 2.6), sharey=True)
    y = np.arange(len(names))
    panels = [
        (axes[0], pos, "largest position difference [mm]\n(joint frames, centres of mass, markers)",
         [(500 * np.finfo(float).eps, "float64 resolution\nat 500 mm"), (0.01, "test tolerance\n0.01 mm"),
          (0.5, "CT/MRI voxel\n~0.5-1 mm")], (1e-16, 10)),
        (axes[1], ang, "largest angle difference [rad]\n(joint frame orientations)",
         [(np.finfo(float).eps, "float64\nresolution"), (1e-4, "test tolerance\n1e-4 rad"),
          (np.deg2rad(1), "1 degree")], (1e-17, 1)),
    ]
    for ax, vals, title, refs, lim in panels:
        vals = np.maximum(np.asarray(vals, float), ZERO)
        ax.barh(y, vals, color=PY_COLOR, height=0.6)
        for yi, v in zip(y, vals):
            ax.text(max(v, lim[0]) * 2, yi, "bit-identical" if v <= ZERO else f"{v:.1e}", va="center", fontsize=9)
        for k, (x, lbl) in enumerate(refs):
            ax.axvline(x, color="0.4", ls="--", lw=1)
            ax.text(x, 1.02 + 0.13 * (k % 2), lbl.replace("\n", " "), transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color="0.3")
        ax.set_xscale("log")
        ax.set_xlim(*lim)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.set_xlabel(title)
        ax.grid(axis="x", which="major", alpha=0.3)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    fig.suptitle("OpenSim model: Python port vs MATLAB STAPLE (all quantities of bone_model.osim)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_model_heatmap(data, path):
    names = list(data)
    M = np.array([[group_max(data[d]["diffs"], s, f) for d in names] for _, _, s, f in GROUPS])
    shown = np.where(np.isnan(M), np.nan, np.maximum(M, ZERO))
    fig, ax = plt.subplots(figsize=(max(1.3 * len(names) + 4.5, 8), 0.55 * len(GROUPS) + 2))
    im = ax.imshow(shown, norm=LogNorm(vmin=ZERO, vmax=1e-2), cmap="viridis", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = "n/a" if np.isnan(v) else ("0" if v == 0 else f"{v:.0e}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if (np.isnan(v) or v < 1e-8) else "black")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(GROUPS)), [f"{lbl} [{u}]" for lbl, u, _, _ in GROUPS])
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("max |python - matlab|")
    ax.set_title("Largest difference per quantity and dataset (0 = bit-identical)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _bone_entries(data):
    for ds, d in data.items():
        for body, b in d["bones"].items():
            yield ds, body, b


def fig_fidelity_bars(data, path):
    bodies = sorted({body for _, body, _ in _bone_entries(data)}, key=list(BONE_LABELS).index)
    names = list(data)
    fig, axes = plt.subplots(2, len(bodies), figsize=(5 * len(bodies), 7.5), squeeze=False)
    x = np.arange(len(names))
    for c, body in enumerate(bodies):
        for r, (key, ylabel) in enumerate((("mean", "mean distance [mm]"), ("max", "maximum distance [mm]"))):
            ax = axes[r, c]
            py = [data[d]["bones"].get(body, {}).get(f"{key}_py", np.nan) for d in names]
            ml = [data[d]["bones"].get(body, {}).get(f"{key}_ml", np.nan) for d in names]
            ax.bar(x - 0.2, py, 0.4, color=PY_COLOR, label="Python (fast-simplification)")
            ax.bar(x + 0.2, ml, 0.4, color=ML_COLOR, label="MATLAB (reducepatch)")
            ax.set_xticks(x, names, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.3)
            if r == 0:
                ax.set_title(BONE_LABELS.get(body, body))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Visualization meshes (30 % of the triangles): distance to the original bone surface\n"
                 "lower = more faithful", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_fidelity_scatter(data, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    markers = {"pelvis": "o", "femur_r": "s", "tibia_r": "^"}
    colors = dict(zip(data, plt.cm.tab10.colors))
    for ax, key, title in ((axes[0], "mean", "mean distance [mm]"), (axes[1], "max", "maximum distance [mm]")):
        allv = []
        for ds, body, b in _bone_entries(data):
            xv, yv = max(b[f"{key}_ml"], 1e-4), max(b[f"{key}_py"], 1e-4)
            allv += [xv, yv]
            ax.scatter(xv, yv, marker=markers.get(body, "o"), color=colors[ds], s=55, edgecolor="k", lw=0.5)
        lo, hi = min(allv) / 1.5, max(allv) * 1.5
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=PY_COLOR, alpha=0.06)
        ax.text(hi / 1.3, lo * 1.3, "Python more faithful", ha="right", va="bottom", color=PY_COLOR, fontsize=9)
        ax.text(lo * 1.3, hi / 1.3, "MATLAB more faithful", ha="left", va="top", color=ML_COLOR, fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"MATLAB: {title}")
        ax.set_ylabel(f"Python: {title}")
        ax.grid(alpha=0.3, which="both")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=ds) for ds, c in colors.items()]
    handles += [plt.Line2D([], [], marker=m, ls="", color="0.5", label=BONE_LABELS[b]) for b, m in markers.items()]
    fig.legend(handles=handles, loc="center right", fontsize=8)
    fig.suptitle("Distance of the decimated meshes to the original bone (one point per bone)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.87, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_histograms(ds, bones, path):
    fig, axes = plt.subplots(1, len(bones), figsize=(5 * len(bones), 3.8), squeeze=False)
    for ax, (body, b) in zip(axes[0], bones.items()):
        top = max(np.percentile(np.r_[b["dist_py"], b["dist_ml"]], 99.9), 0.05)
        bins = np.linspace(0, top, 60)
        for key, color, label in (("py", PY_COLOR, "Python"), ("ml", ML_COLOR, "MATLAB")):
            d = b[f"dist_{key}"]
            ax.hist(np.minimum(d, top), bins=bins, histtype="step", lw=1.5, color=color,
                    label=f"{label}: mean {d.mean():.3f}, p95 {np.percentile(d, 95):.3f}, max {d.max():.2f}")
        ax.set_yscale("log")
        ax.set_xlabel(f"distance to the original bone [mm]  (last bin: >= {bins[-2]:.2f})")
        ax.set_title(BONE_LABELS.get(body, body))
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel("surface samples")
    fig.suptitle(f"{ds}: distribution of distances to the original bone surface", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _render(ax, mesh, fdist, R, back, norm, cmap):
    P = (mesh.points - mesh.points.mean(axis=0)) @ R
    if back:  # rotate 180 degrees about the vertical axis
        P[:, 0], P[:, 2] = -P[:, 0], -P[:, 2]
    F = mesh.faces
    tri = P[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    shade = 0.35 + 0.65 * np.abs(n[:, 2])
    colors = cmap(norm(fdist))
    colors[:, :3] *= shade[:, None]
    order = np.argsort(tri[:, :, 2].mean(axis=1))  # painter's algorithm, viewer at +depth
    ax.add_collection(PolyCollection(tri[order][:, :, :2], facecolors=colors[order],
                                     edgecolors="none", antialiased=False))
    ax.set_xlim(P[:, 0].min(), P[:, 0].max())
    ax.set_ylim(P[:, 1].min(), P[:, 1].max())
    ax.set_aspect("equal")
    ax.axis("off")


def fig_deviation_maps(ds, bones, path, vmax):
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_over("#7a0177")
    norm = Normalize(0, vmax)
    frames = {body: b["frame"] for body, b in bones.items()}
    aspect = []  # height / width of each bone in its view
    for body, b in bones.items():
        P = (b["orig"].points - b["orig"].points.mean(axis=0)) @ frames[body]
        ext = np.ptp(P, axis=0)
        aspect.append(float(np.clip(ext[1] / ext[0], 0.4, 3.0)))
    panel_w = 2.6
    fig = plt.figure(figsize=(4 * panel_w + 1.2, panel_w * sum(aspect) + 0.8 * len(bones) + 0.9))
    gs = fig.add_gridspec(len(bones), 4, height_ratios=aspect, left=0.01, right=0.9, top=0.93, bottom=0.02,
                          hspace=0.25, wspace=0.05)
    for r, (body, b) in enumerate(bones.items()):
        for c, (key, back) in enumerate((("py", False), ("ml", False), ("py", True), ("ml", True))):
            ax = fig.add_subplot(gs[r, c])
            mesh = b[f"mesh_{key}"]
            fd = face_deviation(mesh, b["orig"])
            _render(ax, mesh, fd, frames[body], back, norm, cmap)
            who = "Python" if key == "py" else "MATLAB"
            ax.set_title(f"{BONE_LABELS.get(body, body)}, {who} ({'back' if back else 'front'})\n"
                         f"max {fd.max():.2f} mm", fontsize=9, color=PY_COLOR if key == "py" else ML_COLOR)
    cax = fig.add_axes((0.92, 0.3, 0.018, 0.4))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="max",
                 label="distance to the original bone [mm]")
    fig.suptitle(f"{ds}: where the visualization meshes deviate from the original bone", fontsize=13)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --- report ---------------------------------------------------------------------------------


def write_report(out, data, figs, per_dataset, vmax):
    bones = list(_bone_entries(data))
    n_better = sum(b["mean_py"] <= b["mean_ml"] for _, _, b in bones)
    worst_pos = max(group_max(d["diffs"], s, f) for d in data.values() for _, u, s, f in GROUPS if u == "mm")
    worst_ang = max(group_max(d["diffs"], s, f) for d in data.values() for _, u, s, f in GROUPS if u == "rad")
    n_struct = sum(1 for d in data.values() for x in d["diffs"] if x.max_abs is None and not x.equal)
    mass_diff = max(group_max(d["diffs"], s, f) for d in data.values() for _, u, s, f in GROUPS if u.startswith("kg"))
    mass_txt = "Masses and inertias are bit-identical." if mass_diff == 0 else \
        f"Largest mass / inertia difference: {mass_diff:.1e}."
    verdict = (
        "The differences are rounding noise: many orders of magnitude below the resolution of the "
        "medical images the bones come from, and close to the resolution of double-precision numbers."
        if worst_pos < 1e-6 and worst_ang < 1e-8 else
        "Compare the differences with the test tolerances and the image resolution marked in the figure."
    )
    L = [
        "# pystaple vs MATLAB STAPLE: validation figures",
        "",
        f"Generated by `tools/plot_comparison.py` on {date.today()} for {len(data)} datasets "
        "(`hip_model.m`: pelvis, right femur and tibia, auto2020 joints, 64 kg).",
        "",
        "## OpenSim model",
        "",
        f"* Structural differences (bodies, joints, coordinates, frames, markers, mesh files): **{n_struct}**.",
        f"* Largest position difference: **{worst_pos:.1e} mm**, largest angle difference: "
        f"**{worst_ang:.1e} rad**. {mass_txt}",
        "",
        f"![model differences in context]({figs['context']})",
        "",
        verdict,
        "",
        f"![model differences per quantity]({figs['heatmap']})",
        "",
        "## Visualization geometries",
        "",
        "MATLAB reduces the bone meshes with `reducepatch`, pystaple with `fast-simplification`, both to "
        "30 % of the triangles. The meshes are only used for display (they do not affect the model), so the "
        "relevant question is how faithfully each one represents the original bone.",
        "",
        f"* The Python mesh is at least as faithful as MATLAB's (mean distance) for **{n_better} of {len(bones)}** bones.",
        "",
        f"![fidelity bars]({figs['bars']})",
        "",
        f"![fidelity scatter]({figs['scatter']})",
        "",
        "### Per dataset",
        "",
        f"Deviation maps (bones seen from the front and the back of STAPLE's anatomical frame): colour = "
        f"largest distance of each triangle to the original bone surface (0-{vmax:g} mm, "
        f"purple above {vmax:g} mm). Histograms: distances in both directions between the decimated and the "
        "original surface.",
        "",
    ]
    for ds, f in per_dataset.items():
        L += [f"#### {ds}", ""]
        rows = ["| bone | faces (python / matlab / original) | mean py / ml [mm] | max py / ml [mm] |",
                "|---|---|---|---|"]
        for body, b in data[ds]["bones"].items():
            rows.append(f"| {BONE_LABELS.get(body, body)} | {b['faces'][0]} / {b['faces'][1]} / {b['faces'][2]} | "
                        f"{b['mean_py']:.3f} / {b['mean_ml']:.3f} | {b['max_py']:.2f} / {b['max_ml']:.2f} |")
        L += rows + [""]
        if "maps" in f:
            L += [f"![{ds} deviation maps]({f['maps']})", ""]
        L += [f"![{ds} histograms]({f['hist']})", ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


# --- main -------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help="datasets (default: all with reference/<dataset>/osim)")
    ap.add_argument("--out", default=str(co.ROOT / "out" / "figures"), help="output folder")
    ap.add_argument("--models", default=str(co.ROOT / "out" / "compare"), help="folder with the Python models")
    ap.add_argument("--build", action="store_true", help="rebuild the Python models")
    ap.add_argument("--samples", type=int, default=20000, help="surface samples per mesh and direction")
    ap.add_argument("--vmax", type=float, default=1.0, help="upper limit of the deviation colour scale [mm]")
    ap.add_argument("--no-maps", action="store_true", help="skip the (slow) deviation maps")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or co.osim_datasets()
    if not datasets:
        sys.exit(f"no dataset with osim/bone_model.osim in {co.REFERENCE_DIR}")

    data, per_dataset = {}, {}
    for ds in datasets:
        t0 = time.perf_counter()
        print(f"[{ds}] ", end="", flush=True)
        try:
            d = collect_dataset(ds, Path(args.models), args.build, args.samples)
        except Exception as exc:  # keep going with the other datasets
            print(f"FAILED: {type(exc).__name__}: {exc}")
            continue
        f = {"hist": f"hist_{ds}.png"}
        fig_histograms(ds, d["bones"], out / f["hist"])
        if not args.no_maps:
            print("maps... ", end="", flush=True)
            f["maps"] = f"maps_{ds}.png"
            fig_deviation_maps(ds, d["bones"], out / f["maps"], args.vmax)
        for b in d["bones"].values():  # free the meshes, keep the numbers
            for k in ("mesh_py", "mesh_ml", "orig", "frame"):
                b.pop(k)
        data[ds], per_dataset[ds] = d, f
        print(f"done ({time.perf_counter() - t0:.0f} s)")
    if not data:
        sys.exit("no dataset could be processed")

    figs = {"context": "model_differences.png", "heatmap": "model_differences_heatmap.png",
            "bars": "geometry_fidelity_bars.png", "scatter": "geometry_fidelity_scatter.png"}
    fig_model_context(data, out / figs["context"])
    fig_model_heatmap(data, out / figs["heatmap"])
    fig_fidelity_bars(data, out / figs["bars"])
    fig_fidelity_scatter(data, out / figs["scatter"])
    write_report(out, data, figs, per_dataset, args.vmax)
    print(f"figures and report: {out / 'report.md'}")


if __name__ == "__main__":
    main()