#!/usr/bin/env python
"""The tests a change implies that pytest-testmon cannot see. See "Tests" in README.md.

usage: scripts/affected-tests.py [--base REF] [--dry-run] [PATH...]

testmon selects tests from the coverage it recorded in the pytest process; code that
runs elsewhere (the worker's subprocess, a shell script) or that every test depends
on is outside it. This script maps the changed paths to test files through MAP and
prints them one per line, or the single word `all` for the full suite. The paths are
the arguments, or else every path changed since the merge base of REF (default
main) and HEAD: the branch's commits, the uncommitted changes and untracked files.

It also prints `all` when the interpreter differs from the one the stamp
.testmondata.env (beside .testmondata) records, and then rewrites the stamp; a new
or rebuilt torch, or a new GPU, therefore runs everything once. --dry-run leaves the
stamp alone.
"""

import argparse
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAMP = ".testmondata.env"
ALL = "all"

WORKER = ["tests/test_worker.py", "tests/test_runner.py", "tests/test_cli.py", "tests/test_lce.py"]

# (changed path patterns, implied tests): fnmatch patterns, where `*` crosses `/`.
MAP = [
    # Every module the worker subprocess imports: the worker runs them where testmon
    # records nothing. tests/test_affected_tests.py checks the list against the worker.
    (
        [
            "torch_op_bench/__init__.py",
            "torch_op_bench/worker.py",
            "torch_op_bench/runner.py",
            "torch_op_bench/measure.py",
            "torch_op_bench/env.py",
            "torch_op_bench/records.py",
            "torch_op_bench/spec.py",
            "torch_op_bench/ops/*",
        ],
        WORKER,
    ),
    # Per script, the tests that run it through bash or in a subprocess.
    (["scripts/remote-run.sh"], ["tests/test_remote_run.py"]),
    (["scripts/setup-env.sh"], ["tests/test_remote_run.py", "tests/test_envs.py"]),
    (["scripts/test-affected.sh"], ["tests/test_affected_tests.py"]),
    (["scripts/affected-tests.py"], ["tests/test_affected_tests.py"]),
    # The tests read the definitions as files, or setup-env.sh runs them.
    (["envs/*"], ["tests/test_envs.py"]),
    # Every test depends on these: the configuration and the shared test helpers.
    (
        ["pyproject.toml", "tests/conftest.py", "tests/toy_op.py", "tests/synth_*.py"],
        [ALL],
    ),
]


def implied(paths):
    """The test files `paths` imply through MAP, sorted, or [ALL]."""
    tests = set()
    for path in paths:
        for patterns, implies in MAP:
            if any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns):
                tests.update(implies)
                break
    return [ALL] if ALL in tests else sorted(tests)


def git_lines(root, *args):
    proc = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return [line for line in proc.stdout.splitlines() if line]


def changed_paths(root, base):
    """The paths changed since the merge base of `base` and HEAD, untracked ones too."""
    (merge_base,) = git_lines(root, "merge-base", base, "HEAD")
    changed = git_lines(root, "diff", "--name-only", merge_base)
    untracked = git_lines(root, "ls-files", "--others", "--exclude-standard")
    return sorted(set(changed + untracked))


def environment():
    """What identifies this interpreter's test environment."""
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    # A development build rebuilt in place keeps its version and commit: the size and
    # mtime of the loaded extension tell it apart.
    library = {}
    for path in sorted(Path(torch._C.__file__).parent.glob("_C*.so")):
        stat = path.stat()
        library[path.name] = [stat.st_size, stat.st_mtime_ns]
    return {
        "executable": sys.executable,
        "torch": torch.__version__,
        "git_version": torch.version.git_version,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "torch_C": library,
    }


def environment_changed(root, write=True):
    """Whether the stamp in `root` differs from this interpreter; rewrite it if so."""
    stamp = Path(root) / STAMP
    current = environment()
    try:
        recorded = json.loads(stamp.read_text())
    except (OSError, ValueError):
        recorded = None
    if recorded == current:
        return False
    if write:
        stamp.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
    return True


def main(argv=None, root=ROOT):
    parser = argparse.ArgumentParser(
        prog="scripts/affected-tests.py",
        description="Print the test files a change implies beyond testmon, or `all`.",
    )
    parser.add_argument(
        "--base", default="main", metavar="REF", help="compare with REF (default main)"
    )
    parser.add_argument("--dry-run", action="store_true", help="leave the stamp alone")
    parser.add_argument("paths", nargs="*", metavar="PATH", help="changed paths (not git's)")
    args = parser.parse_args(argv)
    paths = args.paths or changed_paths(root, args.base)
    tests = implied(paths)
    if environment_changed(root, write=not args.dry_run):
        tests = [ALL]
    for test in tests:
        print(test)


if __name__ == "__main__":
    main()
