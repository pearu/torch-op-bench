"""The environment definitions and scripts/setup-env.sh, without conda or a GPU."""

import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "setup-env.sh"
LIGER_GIT = "git+https://github.com/linkedin/Liger-Kernel@"
BASH = shutil.which("bash")


def run_script(*args, cwd=None, path=None):
    env = dict(os.environ)
    if path is not None:
        env["PATH"] = path
    return subprocess.run(
        [BASH, str(SCRIPT), *args], capture_output=True, text=True, cwd=cwd, env=env
    )


def test_script_syntax():
    subprocess.run([BASH, "-n", str(SCRIPT)], check=True)


def test_script_help(tmp_path):
    proc = run_script("--help", cwd=tmp_path)
    assert proc.returncode == 0
    assert "envs/NAME.yml" in proc.stdout and "--prefix DIR" in proc.stdout


def test_script_usage_errors(tmp_path):
    proc = run_script(cwd=tmp_path)
    assert proc.returncode == 2 and "usage:" in proc.stderr
    proc = run_script("liger", "--bogus", cwd=tmp_path)
    assert proc.returncode == 2 and "unknown option --bogus" in proc.stderr
    proc = run_script("liger", "--prefix", cwd=tmp_path)
    assert proc.returncode == 2 and "--prefix needs a directory" in proc.stderr
    proc = run_script("liger", "other", cwd=tmp_path)
    assert proc.returncode == 2 and "one NAME only" in proc.stderr


def test_script_missing_definition(tmp_path):
    proc = run_script("nonexistent", "--prefix", str(tmp_path), cwd=tmp_path)
    assert proc.returncode != 0
    assert f"expected {ROOT}/envs/nonexistent.yml" in proc.stderr
    assert not (tmp_path / "nonexistent").exists()


def test_script_without_conda(tmp_path):
    # Only the tools the script needs before conda: no mamba, no conda.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ["dirname", "mkdir"]:
        (bin_dir / tool).symlink_to(shutil.which(tool))
    proc = run_script("liger", "--prefix", str(tmp_path / "p"), cwd=tmp_path, path=str(bin_dir))
    assert proc.returncode == 1 and "neither mamba nor conda is on PATH" in proc.stderr


def test_liger_release_definition():
    text = (ROOT / "envs" / "liger.yml").read_text()
    assert re.search(r"^\s+- liger-kernel==\d+\.\d+\.\d+$", text, re.M)
    assert re.search(r"^\s+- torch==\d+\.\d+\.\d+$", text, re.M)
    assert re.search(r"^\s+- python=\d+\.\d+\.\d+$", text, re.M)
    assert "nvidia-cutlass-dsl" not in text


def test_liger_git_definition():
    text = (ROOT / "envs" / "liger-git.yml").read_text()
    assert re.search(rf"^\s+- liger-kernel @ {re.escape(LIGER_GIT)}[0-9a-f]{{40}}$", text, re.M)
    release = (ROOT / "envs" / "liger.yml").read_text()
    pins = re.compile(r"^\s+- (python=|pip=|torch==|triton==).*$", re.M)
    assert pins.findall(text) and [m.group() for m in pins.finditer(text)] == [
        m.group() for m in pins.finditer(release)
    ]


def liger_git_commit() -> str:
    text = (ROOT / "envs" / "liger-git.yml").read_text()
    return re.search(rf"{re.escape(LIGER_GIT)}([0-9a-f]{{40}})", text).group(1)
