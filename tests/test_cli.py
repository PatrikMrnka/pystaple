"""Command line interface (the end-to-end run needs opensim)."""

import pytest

from conftest import REFERENCE_DIR
from pystaple import __version__
from pystaple.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_missing_bones_folder(tmp_path):
    assert main(["hip-model", str(tmp_path / "nope"), str(tmp_path / "out"), "-q"]) == 1
    log = (tmp_path / "out" / "auto2020_hip_R.log").read_text()
    assert "no triangulations" in log


def test_side_cannot_be_inferred(tmp_path, capsys):
    assert main(["hip-model", str(tmp_path), str(tmp_path / "out"), "--bones", "pelvis_no_sacrum"]) == 2
    assert "body side" in capsys.readouterr().err


def test_hip_model_end_to_end(tmp_path):
    pytest.importorskip("opensim")
    bones = REFERENCE_DIR / "ICL_MRI"
    if not (bones / "mesh_femur_r.mat").is_file():
        pytest.skip(f"no reference meshes in {bones}")
    from pystaple.osim import read_osim

    out = tmp_path / "model"
    assert main(["hip-model", str(bones), str(out), "--body-mass", "70", "-q"]) == 0
    model = read_osim(out / "bone_model.osim")
    assert list(model["bodies"]) == ["pelvis", "femur_r", "tibia_r"]
    assert sum(b["mass"] for b in model["bodies"].values()) == pytest.approx(
        70 * (11.777 + 9.3014 + 3.7075) / 75.337
    )
    assert sorted(p.name for p in (out / "Geometry").iterdir()) == [
        "femur_r.obj", "pelvis_no_sacrum.obj", "tibia_r.obj"
    ]
    assert (out / "auto2020_hip_R.log").stat().st_size > 0
