"""The environment definitions and scripts/setup-env.sh, without conda or a GPU."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

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


TORCHTITAN_GIT = "git+https://github.com/pytorch/torchtitan@"


def test_torchtitan_release_definition():
    text = (ROOT / "envs" / "torchtitan.yml").read_text()
    for pin in [r"torchtitan==0\.3\.0", r"spmd_types==0\.2\.3", r"torch==\d+\.\d+\.\d+"]:
        assert re.search(rf"^\s+- {pin}$", text, re.M), pin
    assert re.search(r"^\s+- tyro==\d+\.\d+\.\d+$", text, re.M)
    assert re.search(r"^\s+- python=3\.(1[1-9])\.\d+$", text, re.M)  # torchtitan: >= 3.11
    assert "nvidia-cutlass-dsl" not in text and "--no-deps" not in text


def test_torchtitan_git_definition():
    text = (ROOT / "envs" / "torchtitan-git.yml").read_text()
    pin = re.escape(TORCHTITAN_GIT)
    assert re.search(rf"^\s+- torchtitan @ {pin}[0-9a-f]{{40}}$", text, re.M)
    spmd = re.escape("git+https://github.com/meta-pytorch/spmd_types.git@")
    assert re.search(rf"^\s+- spmd_types @ {spmd}[0-9a-f]{{40}}$", text, re.M)
    # One torch build, named exactly: a release, or a nightly from its index.
    torch_pins = re.findall(r"^\s+- torch==(\S+)$", text, re.M)
    assert len(torch_pins) == 1
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d{8}\+cu\d+)?", torch_pins[0])
    if ".dev" in torch_pins[0]:
        assert "--extra-index-url https://download.pytorch.org/whl/nightly/" in text


def torchtitan_git_commit() -> str:
    text = (ROOT / "envs" / "torchtitan-git.yml").read_text()
    return re.search(rf"{re.escape(TORCHTITAN_GIT)}([0-9a-f]{{40}})", text).group(1)


@pytest.mark.parametrize("tool, flag", [("mamba", ["--yes"]), ("conda", [])])
def test_script_never_prompts(tmp_path, tool, flag):
    # mamba 2 asks to confirm; under remote-run.sh stdin is the rest of the remote
    # script, so the tool gets --yes (mamba only) and /dev/null as stdin.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ["dirname", "mkdir"]:
        (bin_dir / name).symlink_to(shutil.which(name))
    log = tmp_path / "args"
    fake = bin_dir / tool
    fake.write_text(
        f'#!{BASH}\nprintf "%s\\n" "$@" > {log}\n'
        f'if read -r line; then echo "stdin: $line" >> {log}; fi\n'
        "exit 3\n"
    )
    fake.chmod(0o755)
    proc = subprocess.run(
        [BASH, str(SCRIPT), "liger", "--prefix", str(tmp_path / "p")],
        input="a line that must not be read\n",
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": str(bin_dir)},
    )
    assert proc.returncode == 3
    args = log.read_text().splitlines()
    assert args[:2] == ["env", "create"] and args[2 : 2 + len(flag)] == flag
    assert not any(line.startswith("stdin:") for line in args)


def test_torch_definitions():
    # The torch environment of a host without a torch build (remote runs): both are
    # the environment named torch, with the override's runtime pinned.
    nightly = (ROOT / "envs" / "torch-nightly.yml").read_text()
    stable = (ROOT / "envs" / "torch-stable.yml").read_text()
    for text in [nightly, stable]:
        assert re.search(r"^name: torch$", text, re.M)
        assert re.search(r"^\s+- python=3\.\d+\.\d+$", text, re.M)
        assert re.search(r"^\s+- nvidia-cutlass-dsl==\d+\.\d+\.\d+$", text, re.M)
        assert re.search(r"^\s+- apache-tvm-ffi==\d+\.\d+\.\d+$", text, re.M)
        assert len(re.findall(r"^\s+- triton==\S+$", text, re.M)) == 1
    # One nightly build, named exactly, from the nightly index.
    assert re.search(r"^\s+- torch==\d+\.\d+\.\d+\.dev\d{8}\+cu\d+$", nightly, re.M)
    assert re.search(
        r"^\s+- --extra-index-url https://download\.pytorch\.org/whl/nightly/cu\d+$",
        nightly,
        re.M,
    )
    assert re.search(r"^\s+- torch==\d+\.\d+\.\d+$", stable, re.M)
    assert "--extra-index-url" not in stable
