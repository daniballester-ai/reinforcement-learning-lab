import json
import os
import sys
import types


def _setup_cell():
    nb = json.load(open("notebooks/atari_gpu.ipynb", encoding="utf8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    return next(c for c in cells if "drive.mount" in c)


def test_drive_mount_failure_does_not_break_the_setup_cell(monkeypatch):
    # Kaggle tem o módulo google.colab, mas drive.mount levanta NotImplementedError
    def boom(*a, **k):
        raise NotImplementedError("Mounting drive is unsupported in this environment")

    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    drive = types.ModuleType("google.colab.drive")
    drive.mount = boom
    colab.drive = drive
    google.colab = colab
    for name, mod in (("google", google), ("google.colab", colab), ("google.colab.drive", drive)):
        monkeypatch.setitem(sys.modules, name, mod)
    mount_part = _setup_cell().split("def find_code")[0]
    exec(mount_part, {"os": os})  # não pode levantar


def _defs_only(cell):
    start = cell.index("def find_code")
    end = cell.index("# --- uso ---")
    return "import glob, os, sys, zipfile\n" + cell[start:end]


def test_find_code_accepts_a_zip_or_a_folder_extracted_by_kaggle(tmp_path):
    import zipfile as zf

    ns = {}
    exec(_defs_only(_setup_cell()), ns)
    find_code = ns["find_code"]
    out = tmp_path / "out"

    # caso A: o dataset do Kaggle foi descompactado automaticamente (só existe a pasta)
    kin = tmp_path / "kaggle_input" / "rl-lab-code" / "rl_lab"
    kin.mkdir(parents=True)
    (kin / "__init__.py").write_text("")
    assert find_code(str(tmp_path / "kaggle_input"), str(tmp_path / "nodrive"), str(out)) == str(tmp_path / "kaggle_input" / "rl-lab-code")

    # caso B: existe o rl_lab.zip (Colab/Drive)
    drive = tmp_path / "drive"
    drive.mkdir()
    with zf.ZipFile(drive / "rl_lab.zip", "w") as z:
        z.writestr("rl_lab/__init__.py", "")
    assert find_code(str(tmp_path / "nokaggle"), str(drive), str(out)) == str(out)
    assert (out / "rl_lab" / "__init__.py").exists()

    # caso C: nada encontrado
    assert find_code(str(tmp_path / "x"), str(tmp_path / "y"), str(out / "z")) is None


def test_subprocess_can_import_rl_lab_via_pythonpath():
    assert "PYTHONPATH" in _setup_cell()  # o '!python -m rl_lab.run' não herda sys.path do kernel


def _launch_cell():
    nb = json.load(open("notebooks/atari_gpu.ipynb", encoding="utf8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    return next(c for c in cells if "def build_cmd" in c)


def _launch_helpers():
    cell = _launch_cell()
    ns = {}
    exec("import os, sys, subprocess, time\n" + cell[cell.index("COMBOS"):cell.index("# --- uso ---")], ns)
    return ns


def test_parallel_launch_covers_the_four_combinations_with_resume():
    ns = _launch_helpers()
    assert set(ns["COMBOS"]) == {("pong", "dqn"), ("pong", "ddqn"), ("spaceinvaders", "dqn"), ("spaceinvaders", "ddqn")}
    cmd = ns["build_cmd"]("pong", "ddqn", "/out", "--set total_steps=500000")
    assert cmd[1:3] == ["-m", "rl_lab.run"]
    assert "--resume" in cmd and cmd[cmd.index("--env") + 1] == "pong" and cmd[cmd.index("--algo") + 1] == "ddqn"
    assert cmd[-2:] == ["--set", "total_steps=500000"] or "total_steps=500000" in cmd


def test_last_step_reads_progress_from_the_episode_log(tmp_path):
    ns = _launch_helpers()
    assert ns["last_step"](str(tmp_path), "pong", "dqn") == 0  # ainda sem arquivo
    d = tmp_path / "pong" / "dqn" / "seed0"
    d.mkdir(parents=True)
    (d / "episodes.csv").write_text("step,episode,episode_return,epsilon\n900,1,-21.0,0.9\n1800,2,-20.0,0.8\n")
    assert ns["last_step"](str(tmp_path), "pong", "dqn") == 1800
