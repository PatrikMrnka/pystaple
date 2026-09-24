"""Command line interface.

    pystaple hip-model <bones_folder> <output_folder> [--body-mass 64]

equivalent to STAPLE's hip_model.m. Also available as ``python -m pystaple``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import __version__

log = logging.getLogger("pystaple")


def build_parser() -> argparse.ArgumentParser:
    from .workflow import HIP_MODEL_BONES

    parser = argparse.ArgumentParser(
        prog="pystaple",
        description="Python port of the STAPLE toolbox: automatic OpenSim models from bone geometries.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    hm = sub.add_parser(
        "hip-model",
        help="build the pelvis-femur-tibia model of hip_model.m",
        description=(
            "Build an OpenSim model (pelvis: STAPLE, femur: GIBOC-cylinder, tibia: Kai2014, "
            "joints: auto2020) from bone geometries <bone>.stl or mesh_<bone>.mat in BONES_FOLDER. "
            "Writes OUTPUT_FOLDER/bone_model.osim, OUTPUT_FOLDER/Geometry and a log file."
        ),
    )
    hm.add_argument("bones_folder", type=Path, help="folder with the bone geometries (mm)")
    hm.add_argument("output_folder", type=Path, help="folder for the model (created if needed)")
    hm.add_argument("--body-mass", type=float, default=64.0, help="subject mass in kg (default: 64)")
    hm.add_argument(
        "--bones", nargs="+", default=list(HIP_MODEL_BONES),
        help=f"bone names (default: {' '.join(HIP_MODEL_BONES)})",
    )
    hm.add_argument("--model-file", default="bone_model.osim", help="model file name (default: bone_model.osim)")
    hm.add_argument("--vis-format", choices=("obj", "stl"), default="obj", help="visualization geometry format")
    hm.add_argument(
        "--reduce", type=float, default=0.3,
        help="fraction of triangles kept in the visualization geometries (default: 0.3, 1 = all)",
    )
    hm.add_argument("--joint-defs", choices=("auto2020",), default="auto2020", help="joint definitions")
    hm.add_argument("-q", "--quiet", action="store_true", help="only errors on the console (the log file is complete)")
    return parser


def _setup_logging(log_file: Path | None, quiet: bool) -> list[logging.Handler]:
    fmt = logging.Formatter("%(message)s")
    handlers: list[logging.Handler] = []
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.ERROR if quiet else logging.INFO)
    handlers.append(console)
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.captureWarnings(True)
    for h in handlers:
        h.setFormatter(fmt)
        for name in ("pystaple", "py.warnings"):
            logging.getLogger(name).addHandler(h)
    log.setLevel(logging.INFO)
    return handlers


def _teardown_logging(handlers: list[logging.Handler]) -> None:
    for h in handlers:
        for name in ("pystaple", "py.warnings"):
            logging.getLogger(name).removeHandler(h)
        h.close()
    logging.captureWarnings(False)


def run_hip_model(args: argparse.Namespace) -> int:
    from .osim.joints import infer_body_side
    from .workflow import build_hip_model, create_tri_geom_set

    try:
        side = infer_body_side(args.bones)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output_folder.mkdir(parents=True, exist_ok=True)
    log_file = args.output_folder / f"{args.joint_defs}_hip_{side.upper()}.log"
    handlers = _setup_logging(log_file, args.quiet)
    t0 = time.perf_counter()
    try:
        log.info("pystaple %s | hip model | bones: %s", __version__, args.bones_folder)
        geom_set = create_tri_geom_set(args.bones, args.bones_folder)
        from .osim.model import import_opensim

        import_opensim()  # fail early, before the bone analysis
        path, *_ = build_hip_model(
            geom_set,
            args.output_folder,
            body_mass=args.body_mass,
            joint_defs=args.joint_defs,
            vis_geom_format=args.vis_format,
            model_file_name=args.model_file,
            coeff_face_reduc=args.reduce,
        )
    except (FileNotFoundError, ImportError, ValueError, RuntimeError, NotImplementedError) as exc:
        log.error("error: %s", exc)
        return 1
    finally:
        _teardown_logging(handlers)
    if not args.quiet:
        print("-------------------------")
        print(f"Model generated in {time.perf_counter() - t0:.1f} s")
        print(f"Saved as {path}")
        print(f"Model geometries saved in folder: {args.output_folder / 'Geometry'}")
        print(f"Log: {log_file}")
        print("-------------------------")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "hip-model":
        return run_hip_model(args)
    return 2  # pragma: no cover (argparse requires a command)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
