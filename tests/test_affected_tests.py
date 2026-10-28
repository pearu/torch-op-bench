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
        ("torch_op_bench/env.py", WORKER),
        ("torch_op_bench/records.py", WORKER),
        ("torch_op_bench/spec.py", WORKER),
        ("torch_op_bench/__init__.py", WORKER),
        ("torch_op_bench/ops/lce.py", WORKER),
        ("torch_op_bench/ops/sub/new.py", WORKER),
        ("scripts/remote-run.sh", ["tests/test_remote_run.py"]),
        ("scripts/affected-tests.py", ["tests/test_affected_tests.py"]),
        ("scripts/test-affected.sh", ["tests/test_affected_tests.py"]),
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
    paths = ["envs/liger.yml", "torch_op_bench/plot.py", "scripts/remote-run.sh"]
    assert affected.implied(paths) == ["tests/test_envs.py", "tests/test_remote_run.py"]
    assert affected.implied(["torch_op_bench/ops/lce.py", "pyproject.toml"]) == ["all"]
    assert affected.implied([]) == []


def test_map_files_exist():
    for _, tests in affected.MAP:
        for test in tests:
            assert test == affected.ALL or (ROOT / test).is_file(), test


def test_map_every_script():
    """Each script has its own row: the tests that run it."""
    scripts = sorted(path.name for path in (ROOT / "scripts").iterdir() if path.is_file())
    assert "test-affected.sh" in scripts
    for name in scripts:
        assert affected.implied([f"scripts/{name}"]), name


def test_map_every_module_the_worker_imports():
    """The worker row covers every torch_op_bench module the worker subprocess imports,
    so a module it starts importing fails here instead of escaping the map."""
    code = (
        "import sys, torch_op_bench.worker\n"
        "for name, module in sorted(sys.modules.items()):\n"
        "    if name.split('.')[0] == 'torch_op_bench':\n"
        "        print(module.__file__)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
    )
    files = [Path(line).resolve().relative_to(ROOT).as_posix() for line in proc.stdout.split()]
    assert "torch_op_bench/worker.py" in files and "torch_op_bench/env.py" in files
    for path in files:
        assert affected.implied([path]) == WORKER, path


@pytest.fixture
def fake_torch(monkeypatch, tmp_path_factory):
    """torch reporting a fixed build, extension and GPU; `fake_torch(name=...)` changes
    them, and `fake_torch.library` is the planted extension's file."""
    library = tmp_path_factory.mktemp("torch") / "_C.cpython-399-fake.so"
    library.write_bytes(b"one")
    os.utime(library, ns=(10**18, 10**18))

    def plant(version="2.99.0", git_version="abc123", cuda="13.0", gpu="Fake GPU"):
        monkeypatch.setattr(torch._C, "__file__", str(library))
        monkeypatch.setattr(torch, "__version__", version)
        monkeypatch.setattr(torch.version, "git_version", git_version)
        monkeypatch.setattr(torch.version, "cuda", cuda)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu is not None)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: gpu)

    plant()
    plant.library = library
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
        "torch_C": {"_C.cpython-399-fake.so": [3, 10**18]},
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
    # torch rebuilt in place, at the same version and commit: a new size, then only
    # a new mtime.
    fake_torch.library.write_bytes(b"two!")
    os.utime(fake_torch.library, ns=(10**18, 10**18))
    for _ in range(2):
        run(tmp_path, "README.md")
        assert capsys.readouterr().out == "all\n"
        run(tmp_path, "README.md")
        assert capsys.readouterr().out == ""
        os.utime(fake_torch.library, ns=(2 * 10**18, 2 * 10**18))


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
        "COVERAGE_CORE=ctrace python -m pytest --testmon",
        "COVERAGE_CORE=ctrace python -m pytest --testmon-noselect tests/test_envs.py",
    ]
    assert commands == readme_blocks()[1]
    # --base: nothing changed since the uncommitted change, once committed.
    git(root, "commit", "-q", "-am", "envs")
    assert dry_run(checkout, "--base", "HEAD") == commands[:2]


def test_test_affected_all(checkout):
    root, _ = checkout
    (root / "pyproject.toml").write_text("two\n")
    commands = dry_run(checkout)
    assert commands == [
        "ruff check .",
        "COVERAGE_CORE=ctrace python -m pytest --testmon-noselect -n auto --maxprocesses 10",
    ]
    assert commands == readme_blocks()[2]


def test_test_affected_new_environment(checkout):
    root, _ = checkout
    (root / ".testmondata.env").write_text("{}\n")
    assert dry_run(checkout) == [
        "ruff check .",
        "COVERAGE_CORE=ctrace python -m pytest --testmon-noselect -n auto --maxprocesses 10",
    ]


def test_test_affected_paths_without_git(checkout, tmp_path):
    """`-- PATH...`: the paths instead of git's, in a copy without .git."""
    root, env = checkout
    copy = tmp_path / "copy"
    shutil.copytree(root, copy, ignore=shutil.ignore_patterns(".git"))
    commands = dry_run((copy, env), "--", "envs/liger.yml", "README.md")
    assert commands == readme_blocks()[1]
    assert dry_run((copy, env), "--", "torch_op_bench/plot.py") == commands[:2]


def test_test_affected_testmon_only(checkout):
    root, _ = checkout
    (root / "torch_op_bench/plot.py").write_text("two\n")
    assert dry_run(checkout) == ["ruff check .", "COVERAGE_CORE=ctrace python -m pytest --testmon"]


def test_readme_final_run():
    """The final run: the whole suite through testmon, then CI's own command."""
    final = readme_blocks()[3]
    assert final[:2] == [
        "ruff check .",
        "COVERAGE_CORE=ctrace pytest --testmon-noselect -n auto --maxprocesses 10",
    ]
    workflow = (ROOT / ".github" / "workflows").glob("*.yml")
    ci = " ".join(" ".join(path.read_text().split()) for path in workflow)
    assert f"run: >- {final[2]} " in ci + " "
    assert "--testmon" not in ci


def test_test_affected_selects_every_test_of_a_changed_line(tmp_path):
    """For real, in a small project: after a change to a line two tests run, testmon
    selects both (not only the first that ran it, as with coverage's sysmon core)."""
    pytest.importorskip("testmon", reason="needs pytest-testmon (in the dev extra)")
    if shutil.which("ruff") is None:
        pytest.skip("needs ruff on PATH (in the dev extra)")
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "tests").mkdir()
    for script in [SCRIPT, SHELL_SCRIPT]:
        shutil.copy(script, root / "scripts" / script.name)
    (root / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\npythonpath = ["."]\n'
    )
    (root / "m.py").write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    (root / "tests/test_m.py").write_text(
        "import m\n\n\ndef test_a():\n    assert m.f() == 1\n\n\n"
        "def test_b():\n    assert m.f() == 1\n\n\ndef test_c():\n    assert m.g() == 2\n"
    )
    (root / ".gitignore").write_text(".testmondata*\n__pycache__/\n.pytest_cache/\n")
    git(root, "init", "-q", "-b", "main")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "init")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHON", "COVERAGE_CORE")}
    env["PYTHON"] = sys.executable

    def ran():
        proc = subprocess.run(
            [BASH, str(root / "scripts/test-affected.sh"), "--base", "HEAD"],
            capture_output=True,
            text=True,
            env={**env, "PYTEST_ADDOPTS": "-v -p no:randomly"},
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        # Serial -v writes "test_m.py::test_a PASSED"; under xdist (the `all` branch),
        # "[gw0] [ 33%] PASSED tests/test_m.py::test_a".
        pattern = r"test_m\.py::(test_\w+) PASSED|PASSED \S*test_m\.py::(test_\w+)"
        found = re.findall(pattern, proc.stdout)
        return sorted({a or b for a, b in found})

    assert ran() == ["test_a", "test_b", "test_c"]  # the first run: all
    assert ran() == []
    (root / "m.py").write_text("def f():\n    return 3 - 2\n\n\ndef g():\n    return 2\n")
    assert ran() == ["test_a", "test_b"]
