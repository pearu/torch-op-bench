#!/usr/bin/env bash
# Lint, then the tests a change affects. See "Tests" in README.md.
set -euo pipefail

usage() {
    cat <<'USAGE'
usage: scripts/test-affected.sh [--base REF] [--dry-run] [-- PATH...]

While iterating: `ruff check .`, then the tests pytest-testmon selects
(`pytest --testmon`), then the test files scripts/affected-tests.py maps the
changes since REF to, unconditionally (`pytest --testmon-noselect FILE...`).
When the map says `all` (a change every test depends on, or a new interpreter,
torch build or GPU), the whole suite once instead of both, in parallel
(`-n auto --maxprocesses 10`, pytest-xdist). Each pytest run
records testmon's data, under COVERAGE_CORE=ctrace (with Python 3.14's
default core, testmon's data misses tests). The exit status is lint's, else the
first failing pytest's.

options:
  --base REF   the changes since the merge base of REF and HEAD, uncommitted and
               untracked ones too (default main)
  --dry-run    print the commands, run none, and leave the stamp alone
  -- PATH...   the changed paths, instead of git's (e.g. `git diff --name-only`
               of a change applied to a copy without .git)
  -h, --help   this message

The interpreter is $PYTHON (default python): the one whose tests run and whose
build the stamp records.
USAGE
}

die() {
    echo "test-affected.sh: $*" >&2
    exit 2
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${PYTHON:-python}"
base="main"
dry_run=""
paths=()

while [ $# -gt 0 ]; do
    case "$1" in
        --base)
            [ $# -ge 2 ] || die "--base needs a value"
            base="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --)
            shift
            paths=("$@")
            break
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *) die "unknown argument $1" ;;
    esac
done

cd "$root"

run() {
    echo "+ $*"
    [ -n "$dry_run" ] || "$@"
}

# testmon records which test ran each line by switching coverage's context per test;
# coverage's sysmon core (Python 3.14's default) records a line for the first test
# that runs it only, so testmon would miss the others.
run_pytest() {
    echo "+ COVERAGE_CORE=ctrace $python -m pytest $*"
    [ -n "$dry_run" ] || COVERAGE_CORE=ctrace "$python" -m pytest "$@"
}

run ruff check .
implied="$("$python" scripts/affected-tests.py --base "$base" ${dry_run:+--dry-run} -- "${paths[@]}")"
tests=()
[ -z "$implied" ] || mapfile -t tests <<<"$implied"

status=0
if [ "${tests[*]-}" = "all" ]; then
    # The whole suite in parallel (see "Tests" in README.md); a selected run stays
    # serial, as K interpreters importing torch would cost more than it saves.
    run_pytest --testmon-noselect -n auto --maxprocesses 10 || status=$?
else
    run_pytest --testmon || status=$?
    if [ ${#tests[@]} -gt 0 ]; then
        run_pytest --testmon-noselect "${tests[@]}" || {
            s=$?
            [ "$status" -ne 0 ] || status=$s
        }
    fi
fi
exit "$status"
