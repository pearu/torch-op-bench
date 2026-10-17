"""scripts/remote-run.sh without a remote host: its dry runs, its checks, and its
remote side end to end through a fake ``ssh`` that runs commands locally."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from torch_op_bench import remote

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "remote-run.sh"
BASH = shutil.which("bash")


def run_script(*args, script=SCRIPT, env=None, cwd=None):
    return subprocess.run(
        [BASH, str(script), *args], capture_output=True, text=True, env=env, cwd=cwd
    )


def dry(subcommand, *args):
    # --dry-run before any `--`: after it, it would be a RUN-ARG and ssh would run.
    proc = run_script("tob-ex", subcommand, "--dry-run", *args)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_syntax_and_help():
    subprocess.run([BASH, "-n", str(SCRIPT)], check=True)
    proc = run_script("--help")
    assert proc.returncode == 0
    for word in ["push", "bootstrap", "setup", "run", "status", "wait", "fetch", "shell"]:
        assert f"\n  {word} " in proc.stdout
    assert "--dry-run" in proc.stdout and "--allow-dirty" in proc.stdout


@pytest.mark.parametrize(
    "args, message",
    [
        ([], "usage:"),
        (["tob-ex"], "usage:"),
        (["tob-ex", "bogus"], "unknown subcommand bogus"),
        (["tob-ex", "push", "--frob"], "unknown option --frob"),
        (["tob-ex", "push", "--dir"], "--dir needs a value"),
        (["tob-ex", "status", "--", "x"], "RUN-ARGS (after --) are for run only"),
        (["tob-ex", "run", "--torch-env", "torch-old"], "--torch-env must be"),
        (["tob-ex", "run", "--gpu", "a"], "--gpu takes GPU indices"),
        (["tob-ex", "run", "--dir", "rel/dir"], "--dir must be absolute or start with ~/"),
        (["tob-ex", "run", "--envs", "a;b"], "bad environment name"),
        (["tob-ex", "run", "extra"], "unexpected argument extra"),
    ],
)
def test_usage_errors(args, message):
    proc = run_script(*args)
    assert proc.returncode == 2 and message in proc.stderr


def test_push_dry_run(repo):
    script = repo / "scripts" / "remote-run.sh"
    out = run_script("tob-ex", "push", "--dry-run", script=script).stdout
    (rsync,) = [line for line in out.splitlines() if line.startswith("+ rsync")]
    assert rsync.startswith("+ rsync -a --delete ")
    for exclude in [
        "/.asb/",
        "/envs/.prefix/",
        "/results/",
        "__pycache__/",
        ".pytest_cache/",
        ".ruff_cache/",
        "htmlcov/",
        r".coverage\*",
    ]:
        assert f"--exclude={exclude} " in rsync
    assert "--exclude=/.git " not in rsync  # .git goes: records name the commit
    assert "--exclude=/.git/objects/info/alternates " in rsync
    assert rsync.endswith(f" {repo}/ tob-ex:torch_op_bench-scratch/torch-op-bench/")
    remote_dir = "torch_op_bench-scratch/torch-op-bench"
    assert rf"--rsync-path=mkdir\ -p\ {remote_dir}\ \&\&\ \{{\ \[\ -d\ {remote_dir}/.git\ " in rsync
    assert rf"\&\&\ rm\ -f\ {remote_dir}/.git/objects/info/alternates\ \&\&\ rsync " in rsync


def test_dir_option(repo):
    script = repo / "scripts" / "remote-run.sh"
    out = run_script("tob-ex", "push", "--dry-run", "--dir", "/scratch/tob", script=script).stdout
    assert out.rstrip().endswith("tob-ex:/scratch/tob/torch-op-bench/")
    assert "\nD=/scratch/tob\n" in dry("status", "--dir", "/scratch/tob")
    assert '\nD="$HOME"/torch_op_bench-scratch\n' in dry("status")
    assert '\nD="$HOME"/my\\ scratch\n' in dry("status", "--dir", "~/my scratch")


def test_bootstrap_dry_run():
    out = dry("bootstrap")
    assert out.startswith("+ ssh tob-ex bash -s <<'REMOTE'\nset -euo pipefail\n")
    assert out.rstrip().endswith("\nREMOTE")
    assert "Miniforge3-Linux-$(uname -m).sh" in out and '-b -p "$D/miniforge3"' in out
    assert "command -v mamba" in out and "command -v conda" in out
    for name in [
        "TRITON_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "CUDA_CACHE_PATH",  # else ~/.nv/ComputeCache, as on ex
        "MAMBA_ROOT_PREFIX",
        "CONDA_PKGS_DIRS",
    ]:
        assert f'echo "export {name}=\\"$D/cache/' in out
    assert "CONDA_REGISTER_ENVS=false" in out  # no ~/.conda/environments.txt
    # The installer's conda made an empty ~/.conda on ex: it gets a home of its own.
    assert 'HOME="$D/cache/home" CONDA_REGISTER_ENVS=false bash "$installer"' in out


def test_setup_dry_run():
    out = dry("setup", "--envs", "liger,torchtitan", "--envs", "liger-git")
    assert "for name in torch-nightly liger torchtitan liger-git; do" in out
    # mamba 2 wrote ~/.conda/environments.txt on ex: setup-env.sh gets a home of its own.
    assert 'HOME="$D"/cache/home scripts/setup-env.sh "$name" --prefix "$D"/envs' in out
    assert "nvidia-smi" in out
    assert '"$D"/envs/torch-nightly/bin/python -m torch_op_bench.cli list lce' in out
    out = dry("setup", "--torch-env", "torch-stable", "--op", "toy_op:TOY")
    assert "for name in torch-stable; do" in out
    assert "torch-stable/bin/python -m torch_op_bench.cli list toy_op:TOY" in out


def test_run_dry_run():
    out = dry(
        "run", "--envs", "liger-git", "--gpu", "1", "--", "--dtype", "float16",
        "--case", "N=1024,H=1024,V=32768",
    )  # fmt: skip
    assert "\nexport CUDA_VISIBLE_DEVICES=1\n" in out
    assert (
        "\nenv_args+=(--env \"$(awk '/^name:/ {print $2; exit}' "
        'torch-op-bench/envs/liger-git.yml)=$D/envs/liger-git/bin/python")\n'
    ) in out
    assert (
        'cmd=("$D"/envs/torch-nightly/bin/python -m torch_op_bench.cli run lce --device cuda '
        '--results "$D"/results "${env_args[@]}" --dtype float16 --case N=1024\\,H=1024\\,V=32768)'
    ) in out
    assert "setsid nohup bash -c" in out and "< /dev/null &\n" in out
    assert 'echo $! > "$D"/run.pid' in out and 'log="$D/logs/' in out
    assert "a run is in progress" in out  # no second run on top of the first
    assert "CUDA_VISIBLE_DEVICES" not in dry("run")


def test_status_wait_fetch_shell_dry_run():
    out = dry("status")
    assert "tail -n 20" in out and "-m torch_op_bench.remote count" in out
    out = dry("wait")
    assert 'tail -n +1 -f --pid="$pid"' in out and 'exit "$status"' in out
    out = dry("fetch")
    assert out == f"+ rsync -ai tob-ex:torch_op_bench-scratch/results/ {ROOT}/results/\n"
    assert "--delete" not in out  # never deletes on the remote, nor locally
    out = dry("shell")
    assert out.startswith("+ ssh -t tob-ex ") and "source\\ env.sh" in out


@pytest.fixture
def repo(tmp_path):
    """A git checkout holding a copy of the script, committed."""
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, root / "scripts" / "remote-run.sh")
    (root / "tracked.txt").write_text("one\n")
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], check=True)
    return root


def test_push_refuses_a_dirty_tree(repo):
    script = repo / "scripts" / "remote-run.sh"
    (repo / "untracked.txt").write_text("not a problem\n")
    proc = run_script("tob-ex", "push", "--dry-run", script=script)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("commit: ")
    (repo / "tracked.txt").write_text("two\n")
    proc = run_script("tob-ex", "push", "--dry-run", script=script)
    assert proc.returncode == 1
    assert "uncommitted changes" in proc.stderr and "tracked.txt" in proc.stderr
    assert "+ rsync" not in proc.stdout
    proc = run_script("tob-ex", "push", "--dry-run", "--allow-dirty", script=script)
    assert proc.returncode == 0
    assert "WARNING: pushing UNCOMMITTED changes" in proc.stderr
    assert "+ rsync -a --delete" in proc.stdout


def test_count(tmp_path):
    results = tmp_path / "results" / "lce" / "gpu"
    results.mkdir(parents=True)
    (results / "a.jsonl").write_text(
        '{"op": "lce", "device_slug": "gpu", "status": "ok"}\n'
        '{"op": "lce", "device_slug": "gpu", "status": "error"}\n'
        "\n"
        '{"op": "lce", "device_slug": "gpu", "sta'  # a line still being written
    )
    (results / "b.jsonl").write_text(
        '{"op": "lce", "device_slug": "gpu", "status": "ok"}\n'
        '{"op": "toy", "device_slug": "cpu", "status": "unavailable"}\n'
        "[1, 2]\n"
    )
    counts = remote.count(tmp_path / "results")
    assert counts == {("lce", "gpu"): {"ok": 2, "error": 1}, ("toy", "cpu"): {"unavailable": 1}}
    assert remote.format_counts(counts) == (
        "lce/gpu: 3 records (2 ok, 1 error)\ntoy/cpu: 1 records (1 unavailable)"
    )
    assert remote.format_counts({}) == "no records yet"


def test_count_main(tmp_path, capsys):
    assert remote.main(["count", str(tmp_path / "absent")]) == 0
    assert capsys.readouterr().out == f"no results directory at {tmp_path / 'absent'}\n"
    assert remote.main(["count", str(tmp_path)]) == 0
    assert capsys.readouterr().out == "no records yet\n"


@pytest.fixture
def fake_host(tmp_path, repo):
    """A "remote host" that is this machine: ``ssh HOST CMD...`` runs CMD locally
    with HOME set to a directory of its own, and conda is a stub on PATH. Returns
    (environment for the script, the remote home)."""
    home = tmp_path / "remote-home"
    home.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    # Drop ssh's options and the host; run the rest here, as the remote shell would.
    ssh.write_text(
        "#!/bin/sh\n"
        'while [ $# -gt 0 ]; do case "$1" in -t|-l|-p|-e) shift ;; -*) ;; *) break ;; esac; '
        "shift; done\n"
        "shift\n"
        f'export HOME="{home}"\n'
        'cd "$HOME" && exec /bin/sh -c "$*"\n'
    )
    ssh.chmod(0o755)
    conda = bin_dir / "conda"
    conda.write_text("#!/bin/sh\necho conda stub\n")
    conda.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CONDA", "MAMBA"))}
    env["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
    env["PYTHON"] = sys.executable
    return env, home


def test_push_a_separate_git_dir(fake_host, tmp_path):
    """A checkout whose .git is a "gitdir: ..." file arrives with its repository as
    .git, also over an earlier push that copied the file."""
    env, home = fake_host
    root = tmp_path / "separate"
    gitdir = tmp_path / "separate.git"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, root / "scripts" / "remote-run.sh")
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", f"--separate-git-dir={gitdir}", str(root)], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], check=True)
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    remote_checkout = home / "torch_op_bench-scratch" / "torch-op-bench"
    remote_checkout.mkdir(parents=True)
    (remote_checkout / ".git").write_text(f"gitdir: {tmp_path}/elsewhere\n")  # the old push
    script = root / "scripts" / "remote-run.sh"

    proc = run_script("tob-ex", "push", "--dry-run", script=script)
    assert proc.returncode == 0, proc.stderr
    first, second = [line for line in proc.stdout.splitlines() if line.startswith("+ rsync")]
    assert "--exclude=/.git " in first
    assert second.startswith("+ rsync -a --delete --exclude=/objects/info/alternates ")
    assert second.endswith(f" {gitdir}/ tob-ex:torch_op_bench-scratch/torch-op-bench/.git/")

    proc = run_script("fakehost", "push", script=script, env=env)
    assert proc.returncode == 0, proc.stderr
    assert (remote_checkout / ".git").is_dir()
    remote_head = subprocess.run(
        ["git", "-C", str(remote_checkout), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    assert remote_head.stdout == head
    status = subprocess.run(
        ["git", "-C", str(remote_checkout), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout == ""


def test_push_a_clone_that_borrows_objects(fake_host, repo, tmp_path):
    """A clone with objects/info/alternates (git clone --shared) arrives with every
    object it needs and no alternates file, also over a push that copied one."""
    env, home = fake_host
    shared = tmp_path / "shared"
    subprocess.run(["git", "clone", "-q", "--shared", str(repo), str(shared)], check=True)
    alternates = shared / ".git" / "objects" / "info" / "alternates"
    assert alternates.read_text().strip() == str(repo / ".git" / "objects")
    remote_git = home / "torch_op_bench-scratch" / "torch-op-bench" / ".git"
    (remote_git / "objects" / "info").mkdir(parents=True)
    (remote_git / "objects" / "info" / "alternates").write_text("/nowhere\n")  # the old push
    script = shared / "scripts" / "remote-run.sh"

    proc = run_script("tob-ex", "push", "--dry-run", script=script)
    assert proc.returncode == 0, proc.stderr
    first, second = [line for line in proc.stdout.splitlines() if line.startswith("+ rsync")]
    assert "--exclude=/.git/objects/info/alternates " in first
    assert second == (
        f"+ rsync -a --exclude=/info/ {repo}/.git/objects/ "
        "tob-ex:torch_op_bench-scratch/torch-op-bench/.git/objects/"
    )

    proc = run_script("fakehost", "push", script=script, env=env)
    assert proc.returncode == 0, proc.stderr
    assert not (remote_git / "objects" / "info" / "alternates").exists()
    fsck = subprocess.run(
        ["git", "-C", str(remote_git.parent), "fsck", "--full"], capture_output=True, text=True
    )
    assert fsck.returncode == 0 and "missing" not in fsck.stdout + fsck.stderr, fsck


def test_push_refuses_a_linked_worktree(repo, tmp_path):
    linked = tmp_path / "linked"
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", str(linked)], check=True)
    proc = run_script("tob-ex", "push", "--dry-run", script=linked / "scripts" / "remote-run.sh")
    assert proc.returncode == 2
    assert "is a linked worktree" in proc.stderr and "+ rsync" not in proc.stdout


def test_remote_side_end_to_end(fake_host, repo, tests_dir):
    """push, bootstrap, run (detached), wait, status, fetch and a resumed run, on the
    toy op on CPU, with the torch environment's interpreter this one."""
    env, home = fake_host
    script = repo / "scripts" / "remote-run.sh"
    scratch = home / "torch_op_bench-scratch"

    def call(*args):
        proc = run_script("fakehost", *args, script=script, env=env)
        assert proc.returncode == 0, (args, proc.stdout, proc.stderr)
        return proc.stdout

    out = call("push")
    assert (scratch / "torch-op-bench" / "tracked.txt").read_text() == "one\n"
    assert (scratch / "torch-op-bench" / ".git").is_dir()
    out = call("bootstrap")
    assert "using " in out and "conda" in out  # the stub on PATH: no Miniforge download
    env_sh = (scratch / "env.sh").read_text()
    assert f'TRITON_HOME="{scratch}/cache/triton"' in env_sh and "export PATH=" not in env_sh
    # setup would build the environment; stand in this interpreter for it.
    python = scratch / "envs" / "torch-nightly" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    run_args = [
        "--", "--device", "cpu", "--pythonpath", str(tests_dir), "--impl", "plain",
        "--mode", "fwd", "--case", "M=8,K=4,N=4,layout=contiguous", "--warmup", "1",
        "--repeat", "2",
    ]  # fmt: skip
    out = call("run", "--op", "toy_op:TOY", *run_args)
    assert out.splitlines()[-1].startswith("started pid ")
    out = call("wait")
    # Two records: the toy op's two default dtypes; RUN-ARGS' --device cpu wins.
    assert "2 new records (2 ok)" in out and "the run exited with status 0" in out, out[-700:]
    out = call("status")
    assert "not running: the last run exited with status 0" in out
    assert "toy/cpu: 2 records (2 ok)" in out
    out = call("fetch", "--op", "toy")
    fetched = list((repo / "results" / "toy" / "cpu").glob("*.jsonl"))
    assert len(fetched) == 1 and "== toy/cpu" in out and "# Report: toy on cpu" in out
    # A rerun of the same run resumes: the recorded ok triples are skipped.
    call("run", "--op", "toy_op:TOY", *run_args)
    out = call("wait")
    assert "recorded, skipped" in out and "0 new records" in out
