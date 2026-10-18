"""scripts/affected-tests.py: its map, its environment stamp and its git paths."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "affected-tests.py"

_spec = importlib.util.spec_from_file_location("affected_tests", SCRIPT)
affected = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(affected)

WORKER = ["tests/test_cli.py", "tests/test_lce.py", "tests/test_runner.py", "tests/test_worker.py"]


@pytest.mark.parametrize(
    "path, tests",
    [
        ("torch_op_bench/worker.py", WORKER),
        ("torch_op_bench/runner.py", WORKER),
        ("torch_op_bench/measure.py", WORKER),
        ("torch_op_bench/ops/lce.py", WORKER),
        ("torch_op_bench/ops/sub/new.py", WORKER),
        ("scripts/remote-run.sh", ["tests/test_remote_run.py"]),
        ("scripts/affected-tests.py", ["tests/test_remote_run.py"]),
        ("scripts/setup-env.sh", ["tests/test_envs.py", "tests/test_remote_run.py"]),
        ("envs/liger.yml", ["tests/test_envs.py"]),
        ("pyproject.toml", ["all"]),
        ("tests/conftest.py", ["all"]),
        ("tests/toy_op.py", ["all"]),
        ("tests/synth_op.py", ["all"]),
        ("tests/synth_results.py", ["all"]),
        # testmon's: code that runs in the pytest process, and the tests themselves.
        ("torch_op_bench/plot.py", []),
        ("torch_op_bench/cli.py", []),
        ("tests/test_plot.py", []),
        ("README.md", []),
    ],
)
def test_map(path, tests):
    assert affected.implied([path]) == tests


def test_map_union():
    assert affected.implied(["envs/liger.yml", "torch_op_bench/plot.py", "scripts/x.sh"]) == [
        "tests/test_envs.py",
        "tests/test_remote_run.py",
    ]
    assert affected.implied(["torch_op_bench/ops/lce.py", "pyproject.toml"]) == ["all"]
    assert affected.implied([]) == []


def test_map_files_exist():
    for _, tests in affected.MAP:
        for test in tests:
            assert test == affected.ALL or (ROOT / test).is_file(), test


@pytest.fixture
def fake_torch(monkeypatch):
    """torch reporting a fixed build and GPU; `fake_torch(name=...)` changes them."""

    def plant(version="2.99.0", git_version="abc123", cuda="13.0", gpu="Fake GPU"):
        monkeypatch.setattr(torch, "__version__", version)
        monkeypatch.setattr(torch.version, "git_version", git_version)
        monkeypatch.setattr(torch.version, "cuda", cuda)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu is not None)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: gpu)

    plant()
    return plant


def run(root, *argv):
    return affected.main(list(argv), root=root)


def test_stamp(tmp_path, fake_torch, capsys):
    stamp = tmp_path / ".testmondata.env"
    # First run: no stamp, so everything; the stamp is written.
    run(tmp_path, "envs/liger.yml")
    assert capsys.readouterr().out == "all\n"
    assert json.loads(stamp.read_text()) == {
        "executable": sys.executable,
        "torch": "2.99.0",
        "git_version": "abc123",
        "cuda": "13.0",
        "gpu": "Fake GPU",
    }
    # The same interpreter: the map alone.
    run(tmp_path, "envs/liger.yml")
    assert capsys.readouterr().out == "tests/test_envs.py\n"
    run(tmp_path, "README.md")
    assert capsys.readouterr().out == ""
    # Each part of the environment changed: everything once, then the map again.
    for change in [
        {"version": "2.99.1"},
        {"git_version": "def456"},
        {"cuda": None},
        {"gpu": None},
        {"gpu": "Other GPU"},
    ]:
        fake_torch(**change)
        run(tmp_path, "README.md")
        assert capsys.readouterr().out == "all\n", change
        run(tmp_path, "README.md")
        assert capsys.readouterr().out == "", change
    assert json.loads(stamp.read_text())["gpu"] == "Other GPU"


def test_stamp_dry_run_and_unreadable(tmp_path, fake_torch, capsys):
    stamp = tmp_path / ".testmondata.env"
    run(tmp_path, "--dry-run", "README.md")
    assert capsys.readouterr().out == "all\n"
    assert not stamp.exists()
    stamp.write_text("not json")
    run(tmp_path, "README.md")
    assert capsys.readouterr().out == "all\n"
    run(tmp_path, "README.md")
    assert capsys.readouterr().out == ""


def test_stamp_executable(tmp_path, fake_torch, monkeypatch, capsys):
    run(tmp_path, "README.md")
    capsys.readouterr()
    monkeypatch.setattr(sys, "executable", "/elsewhere/bin/python")
    run(tmp_path, "README.md")
    assert capsys.readouterr().out == "all\n"


def git(root, *args):
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    """A repository whose branch `work` forks `main`, which then moves on."""
    root = tmp_path / "checkout"
    for path in ["README.md", "envs/liger.yml", "scripts/remote-run.sh", "pyproject.toml"]:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("one\n")
    git(root, "init", "-q", "-b", "main")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "init")
    git(root, "checkout", "-q", "-b", "work")
    return root


def test_git_paths(repo, fake_torch, capsys):
    run(repo)  # the first run's `all`, for the stamp
    capsys.readouterr()
    (repo / ".gitignore").write_text(".testmondata*\n")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-q", "-m", "ignore")
    assert affected.changed_paths(repo, "main") == [".gitignore"]
    # A commit on the branch, an uncommitted change and an untracked file.
    (repo / "envs/liger.yml").write_text("two\n")
    git(repo, "commit", "-q", "-am", "envs")
    (repo / "scripts/remote-run.sh").write_text("two\n")
    (repo / "torch_op_bench").mkdir()
    (repo / "torch_op_bench/worker.py").write_text("new\n")
    assert affected.changed_paths(repo, "main") == [
        ".gitignore",
        "envs/liger.yml",
        "scripts/remote-run.sh",
        "torch_op_bench/worker.py",
    ]
    # main moving on is not the branch's change.
    git(repo, "checkout", "-q", "main")
    (repo / "pyproject.toml").write_text("two\n")
    git(repo, "commit", "-q", "-m", "main moves on", "pyproject.toml")
    git(repo, "checkout", "-q", "work")
    run(repo)
    assert capsys.readouterr().out.splitlines() == sorted(
        ["tests/test_envs.py", "tests/test_remote_run.py", *WORKER]
    )
    # --base: since the branch's second commit.
    run(repo, "--base", "work~1")
    assert capsys.readouterr().out.splitlines() == sorted(
        ["tests/test_envs.py", "tests/test_remote_run.py", *WORKER]
    )
    run(repo, "--base", "HEAD")
    assert capsys.readouterr().out.splitlines() == sorted(["tests/test_remote_run.py", *WORKER])
    # Arguments instead of git.
    run(repo, "pyproject.toml")
    assert capsys.readouterr().out == "all\n"


def test_command_line(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "--help"], capture_output=True, text=True
    )
    assert proc.returncode == 0 and "--base REF" in proc.stdout
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "pyproject.toml"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0 and proc.stdout == "all\n"


SHELL_SCRIPT = ROOT / "scripts" / "test-affected.sh"
BASH = shutil.which("bash")


def readme_blocks():
    """The shell blocks of README.md's "Tests" section, as lists of lines."""
    text = (ROOT / "README.md").read_text()
    section = text[text.index("\n## Tests\n") :]
    section = section[: section.index("\n## ", 1)]
    return [block.splitlines() for block in re.findall(r"```sh\n(.*?)```", section, re.S)]


@pytest.fixture
def checkout(tmp_path):
    """A repository holding copies of both scripts, on branch `work` from `main`, with
    the stamp of this interpreter (as `python` on PATH) already written."""
    root = tmp_path / "checkout"
    for path in ["envs/liger.yml", "torch_op_bench/plot.py", "pyproject.toml"]:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("one\n")
    (root / "scripts").mkdir()
    for script in [SCRIPT, SHELL_SCRIPT]:
        shutil.copy(script, root / "scripts" / script.name)
    (root / ".gitignore").write_text(".testmondata*\n")
    git(root, "init", "-q", "-b", "main")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "init")
    git(root, "checkout", "-q", "-b", "work")
    bindir = Path(sys.executable).parent
    assert (bindir / "python").exists()
    env = {k: v for k, v in os.environ.items() if k != "PYTHON"}
    env["PATH"] = f"{bindir}{os.pathsep}{env['PATH']}"
    subprocess.run(
        ["python", "scripts/affected-tests.py", "README.md"], cwd=root, env=env, check=True
    )
    return root, env


def dry_run(checkout, *args):
    root, env = checkout
    stamp = (root / ".testmondata.env").read_bytes()
    proc = subprocess.run(
        [BASH, str(root / "scripts" / "test-affected.sh"), "--dry-run", *args],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert (root / ".testmondata.env").read_bytes() == stamp
    assert all(line.startswith("+ ") for line in proc.stdout.splitlines()), proc.stdout
    return [line[2:] for line in proc.stdout.splitlines()]


def test_test_affected_syntax_and_usage():
    subprocess.run([BASH, "-n", str(SHELL_SCRIPT)], check=True)
    proc = subprocess.run([BASH, str(SHELL_SCRIPT), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0 and "--base REF" in proc.stdout and "--dry-run" in proc.stdout
    for args, message in [(["--base"], "--base needs a value"), (["x"], "unknown argument x")]:
        proc = subprocess.run([BASH, str(SHELL_SCRIPT), *args], capture_output=True, text=True)
        assert proc.returncode == 2 and message in proc.stderr


def test_test_affected_mapped(checkout):
    root, _ = checkout
    (root / "envs/liger.yml").write_text("two\n")
    commands = dry_run(checkout)
    assert commands == [
        "ruff check .",
        "python -m pytest --testmon",
        "python -m pytest --testmon-noselect tests/test_envs.py",
    ]
    assert commands == readme_blocks()[1]
    # --base: nothing changed since the uncommitted change, once committed.
    git(root, "commit", "-q", "-am", "envs")
    assert dry_run(checkout, "--base", "HEAD") == commands[:2]


def test_test_affected_all(checkout):
    root, _ = checkout
    (root / "pyproject.toml").write_text("two\n")
    commands = dry_run(checkout)
    assert commands == ["ruff check .", "python -m pytest --testmon-noselect"]
    assert commands == readme_blocks()[2]


def test_test_affected_new_environment(checkout):
    root, _ = checkout
    (root / ".testmondata.env").write_text("{}\n")
    assert dry_run(checkout) == ["ruff check .", "python -m pytest --testmon-noselect"]


def test_test_affected_testmon_only(checkout):
    root, _ = checkout
    (root / "torch_op_bench/plot.py").write_text("two\n")
    assert dry_run(checkout) == ["ruff check .", "python -m pytest --testmon"]


def test_readme_final_run():
    """The final run: the whole suite through testmon, then CI's own command."""
    final = readme_blocks()[3]
    assert final[:2] == ["ruff check .", "pytest --testmon-noselect"]
    workflow = (ROOT / ".github" / "workflows").glob("*.yml")
    ci = " ".join(" ".join(path.read_text().split()) for path in workflow)
    assert f"run: >- {final[2]} " in ci + " "
    assert "--testmon" not in ci
