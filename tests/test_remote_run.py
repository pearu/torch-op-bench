"""scripts/remote-run.sh without a remote host: its dry runs, its checks, and its
remote side end to end through a fake ``ssh`` that runs commands locally."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from torch_op_bench import records, remote

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "remote-run.sh"
BASH = shutil.which("bash")


def run_script(*args, script=SCRIPT, env=None, cwd=None, input=None):
    return subprocess.run(
        [BASH, str(script), *args], capture_output=True, text=True, env=env, cwd=cwd, input=input
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
        (["tob-ex", "run", "--matmul-reduction", "loose"], "--matmul-reduction must be"),
        (["tob-ex", "run", "--", "--results", "x"], "--results is remote-run.sh's own option"),
        (["tob-ex", "setup", "--wheelhouse", "/"], "--wheelhouse is for push only"),
        (
            ["tob-ex", "push", "--wheelhouse", "/nowhere"],
            "--wheelhouse /nowhere is not a directory",
        ),
        (
            ["tob-ex", "run", "--", "--matmul-reduction=strict"],
            "--matmul-reduction is remote-run.sh's own option",
        ),
    ],
)
def test_usage_errors(args, message):
    proc = run_script(*args)
    assert proc.returncode == 2 and message in proc.stderr


def test_push_dry_run(repo):
    script = repo / "scripts" / "remote-run.sh"
    out = run_script("tob-ex", "push", "--dry-run", script=script).stdout
    lines = out.splitlines()
    remote_dir = "torch_op_bench-scratch/torch-op-bench"
    # The repository first, whole, after the remote checks.
    (rsync,) = [line for line in lines if line.startswith("+ rsync")]
    assert rsync.startswith("+ rsync -a --checksum --delete --exclude=/objects/info/alternates ")
    assert rsync.endswith(f" {repo}/.git/ tob-ex:{remote_dir}/.git/")
    assert r"--rsync-path=if\ p=\$\(cat\ torch_op_bench-scratch/run.pid\ " in rsync
    assert r"a\ run\ is\ in\ progress" in rsync
    old_head = "torch_op_bench-scratch/push.old-head"
    assert rf"mkdir\ -p\ {remote_dir}\ \&\&\ \{{\ \[\ -s\ {old_head}\ \]\ \|\|\ " in rsync
    assert rf"\&\&\ \{{\ \[\ -d\ {remote_dir}/.git\ " in rsync
    assert rf"\&\&\ rm\ -f\ {remote_dir}/.git/objects/info/alternates\ \&\&\ rsync " in rsync
    # Then exactly the tracked files: no exclusion list to keep up to date.
    assert (
        f"+ git -C {repo} ls-files -z | rsync -a --force --checksum --from0 --files-from=- "
        "--ignore-missing-args "
        f"{repo}/ tob-ex:{remote_dir}/"
    ) in lines
    assert "--exclude=/.asb/" not in out and "--delete " not in out.split("ls-files")[1]
    # The checkout's HEAD before the copy, for what left the index (R1-16 F4), unless
    # a cut-off push left one (R1-17 F5); asked of git only when .git is there.
    assert (
        rf"\{{\ \[\ -d\ {remote_dir}/.git\ \]\ \&\&\ git\ -C\ {remote_dir}\ rev-parse\ -q\ "
        rf"--verify\ HEAD\ \>{old_head}\ " in rsync
    )
    assert "\ngit clean -fd -e /envs/.prefix/ -e /results/\n" in out
    assert out.index('rm -f -- "$path"') < out.index("git clean -fd")
    assert "wheelhouse" not in out
    wheels = repo / "wheels"
    wheels.mkdir()
    out = run_script(
        "tob-ex", "push", "--dry-run", "--wheelhouse", str(wheels), script=script
    ).stdout
    assert f"\n+ rsync -a {wheels}/ tob-ex:torch_op_bench-scratch/wheelhouse/\n" in out


def test_dir_option(repo):
    script = repo / "scripts" / "remote-run.sh"
    out = run_script("tob-ex", "push", "--dry-run", "--dir", "/scratch/tob", script=script).stdout
    assert " tob-ex:/scratch/tob/torch-op-bench/\n" in out
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
    # A refused download names itself rather than ending in set -e's silence.
    assert 'curl -fsSL -o "$installer" "$url" ||' in out
    assert 'echo "bootstrap: could not download the Miniforge installer, $url" >&2' in out
    # The session's TMPDIR and the proxy's CA, decided when env.sh is sourced.
    assert 'echo "if [ -z \\"\\${TMPDIR:-}\\" ]; then export TMPDIR=\\"$D/tmp\\"; fi"' in out
    assert 'export MAMBA_SSL_VERIFY=\\"\\${MAMBA_SSL_VERIFY:-\\$SSL_CERT_FILE}\\"' in out
    assert 'export CONDA_SSL_VERIFY=\\"\\${CONDA_SSL_VERIFY:-\\$SSL_CERT_FILE}\\"' in out
    assert "} > shell.rc\n" in out


def test_setup_dry_run():
    out = dry("setup", "--envs", "liger,torchtitan", "--envs", "liger-git")
    assert "for name in torch-nightly liger torchtitan liger-git; do" in out
    # mamba 2 wrote ~/.conda/environments.txt on ex: setup-env.sh gets a home of its own.
    assert 'HOME="$D"/cache/home scripts/setup-env.sh "$name" --prefix "$D"/envs' in out
    assert "nvidia-smi" in out and 'echo "hostname: $(hostname)"' in out
    assert '"$D"/envs/torch-nightly/bin/python -m torch_op_bench.cli list lce' in out
    # R1-15 remark 1: the counterparts, for a run without --envs; written after
    # every environment was made.
    assert "\nprintf '%s\\n' liger torchtitan liger-git > \"$D\"/envs.list\n" in out
    assert out.index("done\nprintf") < out.index("hostname")
    assert 'export PIP_FIND_LINKS="$D/wheelhouse${PIP_FIND_LINKS:+ $PIP_FIND_LINKS}"' in out
    out = dry("setup", "--torch-env", "torch-stable", "--op", "toy_op:TOY")
    assert "for name in torch-stable; do" in out
    assert "torch-stable/bin/python -m torch_op_bench.cli list toy_op:TOY" in out
    assert "envs.list" not in out  # another torch environment: the list stays


def test_run_dry_run():
    out = dry(
        "run", "--envs", "liger-git", "--gpu", "1", "--", "--dtype", "float16",
        "--case", "N=1024,H=1024,V=32768",
    )  # fmt: skip
    assert "\nexport CUDA_VISIBLE_DEVICES=1\n" in out
    assert "\nenv_names=(liger-git)\n" in out and "envs.list" not in out
    assert (
        "\n    env_args+=(--env \"$(awk '/^name:/ {print $2; exit}' "
        'torch-op-bench/envs/$name.yml)=$D/envs/$name/bin/python")\n'
    ) in out
    assert "setup --envs $name first" in out  # a missing environment stops the run
    assert (
        'cmd=("$D"/envs/torch-nightly/bin/python -m torch_op_bench.cli run lce --device cuda '
        '--results "$D"/results "${env_args[@]}" --dtype float16 --case N=1024\\,H=1024\\,V=32768)'
    ) in out
    assert "setsid nohup bash -c" in out and "< /dev/null &\n" in out
    assert 'echo $! > "$D"/run.pid' in out and 'log="$D/logs/' in out
    assert "a run is in progress" in out  # no second run on top of the first
    # The log's header: the host, the cards, then the command.
    assert 'echo "# host: $(hostname)"' in out and "nvidia-smi -L 2>&1 | sed 's/^/# /'" in out
    assert "\necho results > run.results\n" in out
    out = dry("run")
    assert "export CUDA_VISIBLE_DEVICES" not in out
    assert "--matmul-reduction" not in out  # the default: nothing passed
    assert "mapfile -t env_names < <(grep . envs.list)" in out  # no --envs: setup's


def test_run_strict_dry_run():
    out = dry("run", "--matmul-reduction", "strict", "--", "--case", "N=1024")
    assert "\nmkdir -p logs results-strict\n" in out
    assert (
        'cmd=("$D"/envs/torch-nightly/bin/python -m torch_op_bench.cli run lce --device cuda '
        '--results "$D"/results-strict --matmul-reduction strict "${env_args[@]}" --case N=1024)'
    ) in out


@pytest.mark.parametrize("strict", [False, True])
def test_run_stable_dry_run(strict):
    """The release's records go apart from the nightly's: the resume and merge key
    holds no build."""
    name = "results-stable-strict" if strict else "results-stable"
    out = dry("run", "--torch-env", "torch-stable", *(["--matmul-reduction", "strict"] * strict))
    assert f"\nmkdir -p logs {name}\n" in out and f"\necho {name} > run.results\n" in out
    assert (
        'cmd=("$D"/envs/torch-stable/bin/python -m torch_op_bench.cli run lce --device cuda '
        f'--results "$D"/{name}'
    ) in out


RESULTS = ("results", "results-strict", "results-stable", "results-stable-strict")


def test_status_wait_fetch_shell_dry_run():
    out = dry("status")
    assert "tail -n 20" in out and "-m torch_op_bench.remote count" in out
    assert f"\nfor name in {' '.join(RESULTS)}; do\n" in out
    assert 'echo "running: pid $(cat run.pid), into $into/"' in out
    out = dry("wait")
    assert 'tail -n +1 -f --pid="$pid"' in out and 'exit "$status"' in out
    assert "(records into $D/$into/)" in out
    out = dry("fetch")
    warning, rsyncs = out.split("\nREMOTE\n")
    assert "WARNING: a run is in progress" in warning  # R1-15 remark 2
    assert rsyncs == "".join(
        f"+ rsync -ai --ignore-missing-args tob-ex:torch_op_bench-scratch/{name} {ROOT}/\n"
        for name in RESULTS
    )
    assert "--delete" not in out  # never deletes on the remote, nor locally
    # R1-15 remark 7: three plain words for whatever login shell, bash does the rest.
    assert dry("shell") == "+ ssh -t tob-ex bash --rcfile torch_op_bench-scratch/shell.rc -i\n"
    out = dry("shell", "--dir", "/home/tob/bench/tob-scratch")
    assert out == "+ ssh -t tob-ex bash --rcfile /home/tob/bench/tob-scratch/shell.rc -i\n"


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
    assert "+ rsync -a --checksum --delete" in proc.stdout


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
    with HOME set to a directory of its own, and conda is a stub on PATH. The
    remote checkout is the ``repo`` fixture's, which has no torch_op_bench: the
    remote interpreter imports it from this checkout, through PYTHONPATH. Returns
    (environment for the script, the remote home)."""
    home = tmp_path / "remote-home"
    home.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    # Drop ssh's options (-l, -p and -e take a value) and the host; run the rest
    # here with /bin/sh, as a remote login shell would.
    ssh.write_text(
        "#!/bin/sh\n"
        'while [ $# -gt 0 ]; do case "$1" in -l|-p|-e) shift ;; -*) ;; *) break ;; esac; '
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
    env["PYTHONPATH"] = str(ROOT)
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
    (rsync,) = [line for line in proc.stdout.splitlines() if line.startswith("+ rsync")]
    assert rsync.startswith("+ rsync -a --checksum --delete --exclude=/objects/info/alternates ")
    assert rsync.endswith(f" {gitdir}/ tob-ex:torch_op_bench-scratch/torch-op-bench/.git/")

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
    assert first.endswith(f" {shared}/.git/ tob-ex:torch_op_bench-scratch/torch-op-bench/.git/")
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


def _git(root, *args):
    cmd = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def _files(root):
    """Every file under ROOT outside .git, relative."""
    return sorted(
        str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".git" not in p.parts
    )


def test_push_sends_exactly_the_tracked_files(fake_host, repo):
    """After push the remote checkout holds the tracked files and nothing else: an
    untracked file and an ignored file stay here, a file deleted since the previous
    push is removed there; envs/.prefix/, results/, the remote's ignored files and
    the run files are untouched."""
    env, home = fake_host
    script = repo / "scripts" / "remote-run.sh"
    scratch = home / "torch_op_bench-scratch"
    checkout = scratch / "torch-op-bench"
    (repo / ".gitignore").write_text("secret.cfg\n__pycache__/\nenvs/.prefix/\n")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "mod.py").write_text("x = 1\n")
    (repo / "old.txt").write_text("gone soon\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "more")
    (repo / "pkg" / "stray.py").write_text("untracked code\n")
    (repo / ".agent-sandbox").write_text("untracked policy\n")
    (repo / "secret.cfg").write_text("ignored\n")
    (repo / "pkg" / "__pycache__").mkdir()
    (repo / "pkg" / "__pycache__" / "mod.pyc").write_text("ignored\n")

    proc = run_script("fakehost", "push", script=script, env=env)
    assert proc.returncode == 0, proc.stderr
    tracked = _git(repo, "ls-files").split()
    assert _files(checkout) == sorted(tracked)
    assert "old.txt" in tracked

    # What the remote has of its own: an ignored cache, an environment, results
    # inside the checkout, and the run files beside it.
    kept = [
        checkout / "pkg" / "__pycache__" / "remote.pyc",
        checkout / "envs" / ".prefix" / "torch" / "bin" / "python",
        checkout / "results" / "toy" / "cpu" / "r.jsonl",
        scratch / "run.status",
        scratch / "results" / "toy" / "cpu" / "r.jsonl",
    ]
    for path in kept:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("remote\n")
    _git(repo, "rm", "-q", "old.txt")
    _git(repo, "commit", "-q", "-m", "drop old.txt")

    proc = run_script("fakehost", "push", script=script, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "\nno longer tracked, removing old.txt\n" in proc.stdout  # before git clean
    assert not (checkout / "old.txt").exists()
    for path in kept:
        assert path.read_text() == "remote\n", path
    left = [f for f in _files(checkout) if not f.startswith(("envs/.prefix/", "results/"))]
    assert left == sorted(_git(repo, "ls-files").split() + ["pkg/__pycache__/remote.pyc"])
    assert _git(checkout, "rev-parse", "HEAD") == _git(repo, "rev-parse", "HEAD")
    assert _git(checkout, "status", "--porcelain", "--untracked-files=no") == ""


def test_push_removes_what_left_the_index(fake_host, repo):
    """R1-16 F4: a file that stops being tracked and becomes ignored in one commit
    is not something git clean removes; push removes what the previous push
    tracked and HEAD does not. The first push has no previous HEAD."""
    env, home = fake_host
    script = repo / "scripts" / "remote-run.sh"
    scratch = home / "torch_op_bench-scratch"
    checkout = scratch / "torch-op-bench"
    (repo / "pkg").mkdir()
    (repo / "pkg" / "a.py").write_text("a = 1\n")
    (repo / "pkg" / "b.py").write_text("b = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "pkg")

    proc = run_script("fakehost", "push", script=script, env=env)  # the first push
    assert proc.returncode == 0, proc.stderr
    assert "\nno longer tracked, removing" not in proc.stdout
    assert "previous push" not in proc.stderr
    assert (checkout / "pkg" / "b.py").exists()
    assert not (scratch / "push.old-head").exists()

    _git(repo, "rm", "-q", "--cached", "pkg/b.py")
    (repo / ".gitignore").write_text("pkg/b.py\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-q", "-m", "pkg/b.py: untracked and ignored")
    assert (repo / "pkg" / "b.py").exists()  # still here, ignored

    proc = run_script("fakehost", "push", script=script, env=env)
    assert proc.returncode == 0, proc.stderr
    assert not (checkout / "pkg" / "b.py").exists()
    assert "\nno longer tracked, removing pkg/b.py\n" in proc.stdout
    assert _files(checkout) == sorted(_git(repo, "ls-files").split())
    assert not (scratch / "push.old-head").exists()


def test_push_after_an_interrupted_push(fake_host, repo, tmp_path):
    """R1-17 F5: a push cut off after its .git rsync leaves push.old-head with the
    commit whose files are on the host; the next push keeps it, and removes what
    left the index then."""
    env, home = fake_host
    script = repo / "scripts" / "remote-run.sh"
    scratch = home / "torch_op_bench-scratch"
    checkout = scratch / "torch-op-bench"
    (repo / "pkg").mkdir()
    (repo / "pkg" / "b.py").write_text("b = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "pkg")
    first = _git(repo, "rev-parse", "HEAD").strip()
    proc = run_script("fakehost", "push", script=script, env=env)
    assert proc.returncode == 0, proc.stderr

    _git(repo, "rm", "-q", "--cached", "pkg/b.py")
    (repo / ".gitignore").write_text("pkg/b.py\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-q", "-m", "pkg/b.py: untracked and ignored")
    # The connection drops during the tracked files' rsync, once: a wrapper on PATH.
    bin_dir = tmp_path / "drop"
    bin_dir.mkdir()
    marker = tmp_path / "drop-once"
    marker.touch()
    wrapper = bin_dir / "rsync"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'case " $* " in *" --files-from=- "*) if [ -e {marker} ]; then rm {marker}; '
        'echo "connection dropped" >&2; exit 30; fi ;; esac\n'
        f'exec {shutil.which("rsync")} "$@"\n'
    )
    wrapper.chmod(0o755)
    dropping = {**env, "PATH": f"{bin_dir}:{env['PATH']}"}
    proc = run_script("fakehost", "push", script=script, env=dropping)
    assert proc.returncode == 30 and "connection dropped" in proc.stderr
    assert _git(checkout, "rev-parse", "HEAD") == _git(repo, "rev-parse", "HEAD")  # .git went
    assert (scratch / "push.old-head").read_text().strip() == first
    assert (checkout / "pkg" / "b.py").exists()

    proc = run_script("fakehost", "push", script=script, env=dropping)  # in full
    assert proc.returncode == 0, proc.stderr
    assert not (checkout / "pkg" / "b.py").exists()
    assert "\nno longer tracked, removing pkg/b.py\n" in proc.stdout
    assert _files(checkout) == sorted(_git(repo, "ls-files").split())
    assert not (scratch / "push.old-head").exists()


def test_push_refuses_during_a_run(fake_host, repo):
    """push refuses while run.pid is alive, also with --allow-dirty, and changes
    nothing on the remote."""
    env, home = fake_host
    script = repo / "scripts" / "remote-run.sh"
    scratch = home / "torch_op_bench-scratch"
    checkout = scratch / "torch-op-bench"
    proc = run_script("fakehost", "push", script=script, env=env)
    assert proc.returncode == 0, proc.stderr
    head = _git(checkout, "rev-parse", "HEAD")
    (repo / "tracked.txt").write_text("two\n")
    _git(repo, "commit", "-q", "-am", "two")
    (repo / "tracked.txt").write_text("three, uncommitted\n")
    sleeper = subprocess.Popen(["sleep", "60"])
    try:
        (scratch / "run.pid").write_text(f"{sleeper.pid}\n")
        proc = run_script("fakehost", "push", "--allow-dirty", script=script, env=env)
        assert proc.returncode != 0
        assert f"a run is in progress (pid {sleeper.pid}): wait for it or stop it" in proc.stderr
        assert (checkout / "tracked.txt").read_text() == "one\n"
        assert _git(checkout, "rev-parse", "HEAD") == head
    finally:
        sleeper.kill()
        sleeper.wait()
    # The run is over: the same push goes through.
    proc = run_script("fakehost", "push", "--allow-dirty", script=script, env=env)
    assert proc.returncode == 0, proc.stderr
    assert (checkout / "tracked.txt").read_text() == "three, uncommitted\n"


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
    # shell: the fake host's login shell is /bin/sh, and env.sh is sourced all the same.
    proc = run_script(
        "fakehost", "shell", script=script, env=env, input='echo "in $PWD: $TRITON_HOME"\nexit\n'
    )
    assert proc.returncode == 0, proc.stderr
    assert f"in {scratch}: {scratch}/cache/triton" in proc.stdout
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
    assert "not running: the last run exited with status 0, into results/" in out
    assert "== records in results/\ntoy/cpu: 2 records (2 ok)" in out
    out = call("fetch", "--op", "toy")
    fetched = list((repo / "results" / "toy" / "cpu").glob("*.jsonl"))
    assert len(fetched) == 1 and "# Report: toy on cpu" in out
    assert "== results/toy/cpu: 2 new records\n" in out
    assert not (repo / "results-strict").exists()  # none on the host: skipped
    # A rerun of the same run resumes: the recorded ok triples are skipped.
    call("run", "--op", "toy_op:TOY", *run_args)
    out = call("wait")
    assert "recorded, skipped" in out and "0 new records" in out
    # The strict run: into results-strict/ beside results/, and fetched back.
    call("run", "--op", "toy_op:TOY", "--matmul-reduction", "strict", *run_args)
    out = call("wait")
    assert "2 new records (2 ok)" in out and "results-strict/toy/cpu/" in out, out[-700:]
    out = call("fetch", "--op", "toy")
    (strict,) = (repo / "results-strict" / "toy" / "cpu").glob("*.jsonl")
    assert "== results-strict/toy/cpu: 2 new records\n" in out and "== results/" not in out
    assert all(r["settings"]["matmul_reduction"] == "strict" for r in records.read(strict))


def test_push_a_tracked_file_and_a_directory_trade_places(fake_host, repo):
    """R1-17's remark: a tracked file that replaces a directory of the same name
    replaces it on the host, also when the host's directory holds an ignored file;
    and the other way round."""
    env, home = fake_host
    script = repo / "scripts" / "remote-run.sh"
    checkout = home / "torch_op_bench-scratch" / "torch-op-bench"
    (repo / "p").mkdir()
    (repo / "p" / "a.py").write_text("a = 1\n")
    (repo / "q").write_text("a file\n")
    (repo / ".gitignore").write_text("__pycache__/\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "p/ a directory, q a file")
    proc = run_script("fakehost", "push", script=script, env=env)
    assert proc.returncode == 0, proc.stderr
    (checkout / "p" / "__pycache__").mkdir()
    (checkout / "p" / "__pycache__" / "a.pyc").write_text("the host's own\n")

    _git(repo, "rm", "-q", "-r", "p", "q")
    (repo / "p").write_text("now a file\n")
    (repo / "q").mkdir()
    (repo / "q" / "b.py").write_text("b = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "p a file, q/ a directory")
    proc = run_script("fakehost", "push", script=script, env=env)
    assert proc.returncode == 0, proc.stderr
    assert (checkout / "p").read_text() == "now a file\n"
    assert (checkout / "q" / "b.py").read_text() == "b = 1\n"
    assert _files(checkout) == sorted(_git(repo, "ls-files").split())
    assert _git(checkout, "status", "--porcelain", "--untracked-files=no") == ""


# A conda (and mamba) for setup: `env create|update --file F --prefix P` makes P/bin/python,
# which stands in pip and runs everything else with this interpreter; a definition
# named refused.yml fails as a proxy's refusal does. Each call says what it saw.
STUB_CONDA = """#!/bin/sh
echo "$(basename "$0") $1 $2 HOME=$HOME PIP_FIND_LINKS=${PIP_FIND_LINKS:-}" >&2
while [ $# -gt 0 ]; do
    case "$1" in --prefix) prefix="$2"; shift ;; --file) file="$2"; shift ;; esac
    shift
done
[ -n "${file:-}" ] || exit 0
case "$file" in
    */refused.yml)
        echo "CONNECT tunnel failed, response 403: https://conda.anaconda.org/x" >&2
        exit 1 ;;
esac
mkdir -p "$prefix/bin"
cat > "$prefix/bin/python" <<PY
#!/bin/sh
if [ "\\$1 \\$2" = "-m pip" ]; then
    echo "pip \\$* PIP_FIND_LINKS=\\${PIP_FIND_LINKS:-}" >&2
    exit 0
fi
exec PYTHON "\\$@"
PY
chmod +x "$prefix/bin/python"
"""


@pytest.fixture(params=["sshd-profile", "setup-remote"])
def profile_host(request, tmp_path, repo, tests_dir):
    """A fake host behind agent-sandbox's sandboxed login, of either kind: the older
    sshd profile (HOME read-only, one writable directory elsewhere, an absolute
    --dir) or a login made by `asb --setup-remote` (HOME the one writable directory,
    in a read-only parent; the default --dir). Both with a proxy's CA in
    SSL_CERT_FILE and no TMPDIR (env.sh then makes one). The checkout has
    setup-env.sh and environment definitions. Returns (environment, the scratch
    directory, run)."""
    if request.param == "sshd-profile":
        home = tmp_path / "remote-home"
        read_only = home
        bench = tmp_path / "bench"
        bench.mkdir()
        scratch = bench / "tob-scratch"
        dir_args = ("--dir", str(scratch))
    else:
        read_only = tmp_path / "remote-home"
        home = read_only / "tob-ey"
        scratch = home / "torch_op_bench-scratch"
        dir_args = ()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text(
        "#!/bin/sh\n"
        'while [ $# -gt 0 ]; do case "$1" in -l|-p|-e) shift ;; -*) ;; *) break ;; esac; '
        "shift; done\n"
        "shift\n"
        f'export HOME="{home}"\n'
        'cd "$HOME" && exec /bin/sh -c "$*"\n'
    )
    for name in ("conda", "mamba"):
        (bin_dir / name).write_text(STUB_CONDA.replace("PYTHON", sys.executable))
    for path in bin_dir.iterdir():
        path.chmod(0o755)
    shutil.copy(ROOT / "scripts" / "setup-env.sh", repo / "scripts")
    (repo / "envs").mkdir()
    for name, defined in [("torch-nightly", "torch"), ("liger", "liger"), ("refused", "x")]:
        (repo / "envs" / f"{name}.yml").write_text(
            f"name: {defined}\ndependencies:\n  - pip:\n      - pkg==1.0\n"
        )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "setup")
    home.mkdir(parents=True)
    read_only.chmod(0o555)
    if os.access(read_only, os.W_OK):
        pytest.skip("a read-only directory is writable here (root)")
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("CONDA", "MAMBA", "PIP_")) and k != "TMPDIR"
    }
    env["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
    env["PYTHON"] = sys.executable
    env["PYTHONPATH"] = f"{ROOT}:{tests_dir}"
    env["SSL_CERT_FILE"] = "/the/proxy/ca.pem"
    script = repo / "scripts" / "remote-run.sh"

    def run(subcommand, *args, check=True, input=None):
        # --dir before any `--`, after which it would be a RUN-ARG.
        args = ("fakehost", subcommand, *dir_args, *args)
        proc = run_script(*args, script=script, env=env, input=input)
        if check:
            assert proc.returncode == 0, (args, proc.stdout, proc.stderr)
        return proc

    yield env, scratch, run
    read_only.chmod(0o755)


def test_profile_host_end_to_end(profile_host, repo, tmp_path, tests_dir):
    """The whole workflow on either kind of login: nothing needs a writable home
    but --dir's own expansion; env.sh makes TMPDIR and hands the CA to mamba and conda; setup
    lists its counterparts in envs.list, and run without --envs passes them; fetch
    says what each slug gained."""
    env, scratch, run = profile_host
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / "pkg-1.0-py3-none-any.whl").write_text("a wheel\n")

    run("push", "--wheelhouse", str(wheels))
    assert (scratch / "wheelhouse" / "pkg-1.0-py3-none-any.whl").read_text() == "a wheel\n"
    run("bootstrap")
    probe = 'echo "tmp=$TMPDIR mamba=$MAMBA_SSL_VERIFY conda=$CONDA_SSL_VERIFY"; exit\n'
    proc = run("shell", input=probe)
    assert f"tmp={scratch}/tmp mamba=/the/proxy/ca.pem conda=/the/proxy/ca.pem" in proc.stdout
    assert (scratch / "tmp").is_dir()

    proc = run("setup", "--envs", "liger", "--op", "toy_op:TOY")
    out, err = proc.stdout, proc.stderr
    # Every environment made with HOME in the scratch and the wheelhouse searched.
    wheelhouse = f"PIP_FIND_LINKS={scratch}/wheelhouse"
    for name in ("torch-nightly", "liger"):
        assert f"creating {scratch}/envs/{name} from" in err
    assert f"mamba env create HOME={scratch}/cache/home {wheelhouse}\n" in err
    assert f"pip -m pip install --quiet -e {scratch}/torch-op-bench {wheelhouse}\n" in err
    assert (scratch / "envs.list").read_text() == "liger\n"
    assert "envs.list: liger\n" in out and f"hostname: {os.uname().nodename}\n" in out
    assert "plain" in out  # list toy_op:TOY, from the torch environment

    run_args = [
        "--", "--device", "cpu", "--pythonpath", str(tests_dir), "--impl", "plain",
        "--mode", "fwd", "--case", "M=8,K=4,N=4,layout=contiguous", "--warmup", "1",
        "--repeat", "2",
    ]  # fmt: skip
    out = run("run", "--op", "toy_op:TOY", *run_args).stdout
    assert "the environments of envs.list: liger\n" in out
    out = run("wait").stdout
    assert f"(records into {scratch}/results/)" in out
    assert "2 new records (2 ok)" in out and "the run exited with status 0" in out, out[-700:]
    (log,) = (scratch / "logs").glob("*.log")
    header = log.read_text().splitlines()
    assert header[0] == f"# host: {os.uname().nodename}"
    assert header[1].startswith("# CUDA_VISIBLE_DEVICES: ")
    assert f" --env liger={scratch}/envs/liger/bin/python " in log.read_text()
    out = run("fetch", "--op", "toy").stdout
    assert "== results/toy/cpu: 2 new records\n" in out
    out = run("fetch", "--op", "toy").stdout  # nothing new: no report
    assert "new records" not in out and "# Report" not in out

    # A listed environment that is not there stops the run before it starts.
    (scratch / "envs.list").write_text("liger\nmissing\n")
    proc = run("run", "--op", "toy_op:TOY", *run_args, check=False)
    assert proc.returncode == 1
    assert f"no environment missing in {scratch}/envs: setup --envs missing first" in proc.stderr
    assert not (scratch / "run.pid").read_text().strip() == ""  # the first run's, kept


def test_profile_host_setup_names_what_failed(profile_host):
    """A definition whose host the proxy refuses: setup stops there, naming the
    environment and the step, and lists no counterparts."""
    env, scratch, run = profile_host
    run("push")
    run("bootstrap")
    proc = run("setup", "--envs", "refused,liger", check=False)
    assert proc.returncode == 1
    assert "CONNECT tunnel failed, response 403" in proc.stderr
    assert "setup-env.sh: refused: failed (status 1): " in proc.stderr
    assert "setup: the environment refused failed: the step and its error are above" in (
        proc.stderr
    )
    assert "creating" not in proc.stderr.split("refused: failed")[1]  # liger not tried
    assert not (scratch / "envs.list").exists()


def test_fetch_warns_while_a_run_is_alive(fake_host, repo):
    """R1-15 remark 2: a fetch during a run says so (and still fetches)."""
    env, home = fake_host
    scratch = home / "torch_op_bench-scratch"
    (scratch / "results" / "toy" / "cpu").mkdir(parents=True)
    (scratch / "results" / "toy" / "cpu" / "r.jsonl").write_text('{"half": ')
    sleeper = subprocess.Popen(["sleep", "60"])
    try:
        (scratch / "run.pid").write_text(f"{sleeper.pid}\n")
        proc = run_script(
            "fakehost", "fetch", "--op", "toy", script=repo / "scripts" / "remote-run.sh", env=env
        )
    finally:
        sleeper.kill()
        sleeper.wait()
    assert proc.returncode == 0, proc.stderr
    assert f"WARNING: a run is in progress (pid {sleeper.pid}): the last line" in proc.stderr
    assert (repo / "results" / "toy" / "cpu" / "r.jsonl").exists()
