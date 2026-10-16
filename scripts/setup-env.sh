#!/usr/bin/env bash
# Create (or update) a benchmark environment from its definition envs/NAME.yml and
# install this checkout into it. See "Environments" in README.md.
set -euo pipefail

usage() {
    cat <<USAGE
usage: scripts/setup-env.sh NAME [--prefix DIR]

Create the conda environment defined in envs/NAME.yml at DIR/NAME (default DIR:
envs/.prefix in this checkout), or update it if it exists, then install this
checkout into it (pip install -e, no extras). Uses mamba if it is on PATH, conda
otherwise.

Prints the provenance the environment's records will carry (torch-op-bench env,
with the packages the definition's pip section names) and, on the last line of
stdout, the environment's interpreter, to pass as --env NAME=PYTHON:

    torch-op-bench run lce --env liger=\$(scripts/setup-env.sh liger | tail -1) ...
USAGE
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name=""
prefix_dir="$root/envs/.prefix"
while (($#)); do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --prefix)
            [[ $# -ge 2 ]] || { echo "setup-env.sh: --prefix needs a directory" >&2; exit 2; }
            prefix_dir="$2"
            shift 2
            ;;
        -*)
            echo "setup-env.sh: unknown option $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            [[ -z "$name" ]] || { echo "setup-env.sh: one NAME only" >&2; exit 2; }
            name="$1"
            shift
            ;;
    esac
done
if [[ -z "$name" ]]; then
    usage >&2
    exit 2
fi

definition="$root/envs/$name.yml"
if [[ ! -f "$definition" ]]; then
    echo "setup-env.sh: no definition for '$name': expected $definition" >&2
    exit 1
fi
mkdir -p "$prefix_dir"
prefix="$(cd "$prefix_dir" && pwd)/$name"

if command -v mamba >/dev/null 2>&1; then
    tool=mamba
elif command -v conda >/dev/null 2>&1; then
    tool=conda
else
    echo "setup-env.sh: neither mamba nor conda is on PATH" >&2
    exit 1
fi

# Everything but the provenance and the interpreter goes to stderr.
if [[ -d "$prefix" ]]; then
    echo "setup-env.sh: updating $prefix from $definition with $tool" >&2
    "$tool" env update --file "$definition" --prefix "$prefix" >&2
else
    echo "setup-env.sh: creating $prefix from $definition with $tool" >&2
    "$tool" env create --file "$definition" --prefix "$prefix" >&2
fi

python="$prefix/bin/python"
echo "setup-env.sh: installing $root into $prefix" >&2
"$python" -m pip install --quiet -e "$root" >&2

# The distributions named in the definition's pip section ("name==1.0" or
# "name @ git+https://..."), whose versions or commits the records carry.
mapfile -t packages < <(
    "$python" - "$definition" <<'PY'
import re, sys

in_pip = False
for line in open(sys.argv[1]):
    text = line.split("#", 1)[0].rstrip()
    if re.fullmatch(r"\s*- pip:", text):
        in_pip = True
    elif in_pip and (m := re.fullmatch(r"\s+- ([A-Za-z0-9._-]+)\s*(==|@).*", text)):
        print(m.group(1))
PY
)
args=()
for package in "${packages[@]}"; do
    args+=(--package "$package")
done
"$python" -m torch_op_bench.cli env "${args[@]}"
echo "$python"
