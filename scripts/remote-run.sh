#!/usr/bin/env bash
# Run benchmarks on a remote host from this checkout. See "Remote runs" in README.md.
set -euo pipefail

usage() {
    cat <<'USAGE'
usage: scripts/remote-run.sh HOST SUBCOMMAND [options] [-- RUN-ARGS...]

Run torch-op-bench on HOST (an ssh destination or alias) in a scratch directory
there, from this checkout. Each subcommand is one ssh session or one rsync (push:
more when the repository is outside .git or borrows objects).

subcommands:
  push       copy this checkout (with .git) to REMOTE_DIR/torch-op-bench; refuses
             unless the tracked files are clean, so every record names one commit
  bootstrap  install Miniforge into REMOTE_DIR/miniforge3 unless mamba or conda is
             on PATH, and write REMOTE_DIR/env.sh (caches under REMOTE_DIR/cache)
  setup      create the torch environment and each of --envs under REMOTE_DIR/envs
             with scripts/setup-env.sh; show their provenance, nvidia-smi and
             `torch-op-bench list OP`
  run        start `torch-op-bench run OP --device cuda` detached (nohup, setsid),
             RUN-ARGS after `--`; returns at once; rerun after a disconnect to
             resume (recorded ok triples are skipped)
  status     whether the run is alive, the end of its log, records per status
  wait       follow the log until the run ends; exit with its status
  fetch      copy REMOTE_DIR/results/ into this checkout's results/ (never deletes
             anything on HOST) and show the first lines of each slug's report
  shell      an interactive shell on HOST in REMOTE_DIR with env.sh sourced

options:
  --dir REMOTE_DIR     scratch directory on HOST (default ~/torch_op_bench-scratch)
  --torch-env NAME     torch-nightly (default) or torch-stable: the definition of the
                       torch environment, whose interpreter runs torch-op-bench
  --envs NAME,...      counterpart environments to create and pass to run
                       (e.g. liger,torchtitan)
  --gpu N              CUDA_VISIBLE_DEVICES=N on HOST for run (default: all)
  --op OP              the op to run, list and report (default lce)
  --allow-dirty        push uncommitted changes (development only: the records then
                       name a commit that is not what ran)
  --dry-run            print every local and remote command, run none
  -h, --help           this message

Nothing is written on HOST outside REMOTE_DIR.
USAGE
}

die() {
    echo "remote-run.sh: $*" >&2
    exit 2
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
host=""
subcommand=""
remote_dir="~/torch_op_bench-scratch"
torch_env="torch-nightly"
envs=()
gpu=""
op="lce"
allow_dirty=0
dry_run=0
run_args=()
while (($#)); do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --dir | --torch-env | --envs | --gpu | --op)
            [[ $# -ge 2 ]] || die "$1 needs a value"
            case "$1" in
                --dir) remote_dir="$2" ;;
                --torch-env) torch_env="$2" ;;
                --envs) IFS=, read -r -a more <<<"$2" && envs+=("${more[@]}") ;;
                --gpu) gpu="$2" ;;
                --op) op="$2" ;;
            esac
            shift 2
            ;;
        --allow-dirty)
            allow_dirty=1
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --)
            shift
            run_args=("$@")
            break
            ;;
        -*)
            echo "remote-run.sh: unknown option $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -z "$host" ]]; then
                host="$1"
            elif [[ -z "$subcommand" ]]; then
                subcommand="$1"
            else
                die "unexpected argument $1 (RUN-ARGS go after --)"
            fi
            shift
            ;;
    esac
done
if [[ -z "$host" || -z "$subcommand" ]]; then
    usage >&2
    exit 2
fi
case "$torch_env" in
    torch-nightly | torch-stable) ;;
    *) die "--torch-env must be torch-nightly or torch-stable, not $torch_env" ;;
esac
[[ -z "$gpu" || "$gpu" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--gpu takes GPU indices, not $gpu"
if ((${#run_args[@]})) && [[ "$subcommand" != run ]]; then
    die "RUN-ARGS (after --) are for run only"
fi
for name in "${envs[@]}"; do
    [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || die "bad environment name '$name'"
done

# REMOTE_DIR as the remote shell expands it (D=... at the top of every remote
# script), and as rsync addresses it: relative to the remote home for ~/...
case "$remote_dir" in
    "~") remote_d='"$HOME"' rsync_dir="." ;;
    "~/"*) remote_d="\"\$HOME\"/$(printf '%q' "${remote_dir#\~/}")" rsync_dir="${remote_dir#\~/}" ;;
    /*) remote_d="$(printf '%q' "$remote_dir")" rsync_dir="$remote_dir" ;;
    *) die "--dir must be absolute or start with ~/, not $remote_dir" ;;
esac

# Every command is printed before it runs; --dry-run prints and runs nothing.
local_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if ((!dry_run)); then
        "$@"
    fi
}

# One ssh session running SCRIPT (fed on stdin to bash) on HOST, in REMOTE_DIR.
remote() {
    local script
    script="set -euo pipefail
D=$remote_d
mkdir -p \"\$D\"
cd \"\$D\"
$1"
    printf '+ ssh %q bash -s <<'"'"'REMOTE'"'"'\n%s\nREMOTE\n' "$host" "$script"
    if ((!dry_run)); then
        ssh "$host" bash -s <<<"$script"
    fi
}

# The lines that put the remote environment in place, for the subcommands after
# bootstrap.
source_env='[ -f env.sh ] || { echo "no env.sh in $D: run bootstrap first" >&2; exit 1; }
source env.sh'
torch_python="\"\$D\"/envs/$torch_env/bin/python"

cmd_push() {
    local dirty
    dirty="$(git -C "$root" status --porcelain --untracked-files=no)"
    if [[ -n "$dirty" ]]; then
        if ((!allow_dirty)); then
            echo "remote-run.sh: the checkout has uncommitted changes:" >&2
            echo "$dirty" >&2
            echo "remote-run.sh: commit them (every record names one commit), or" \
                "pass --allow-dirty for development" >&2
            exit 1
        fi
        echo "remote-run.sh: WARNING: pushing UNCOMMITTED changes (--allow-dirty): the" \
            "records will name $(git -C "$root" rev-parse --short HEAD) but run other code" >&2
    fi
    echo "commit: $(git -C "$root" rev-parse HEAD)"
    # The remote .git must be a repository of its own, or the records name no
    # commit. A checkout whose .git is a "gitdir: ..." file (git clone
    # --separate-git-dir) gets that directory copied as the remote .git; objects
    # borrowed through objects/info/alternates (git clone --shared or --reference)
    # are copied into it, and the alternates file is left out. One rsync more for
    # each.
    local common gitdir="" alternates=() alt
    common="$(git -C "$root" rev-parse --path-format=absolute --git-common-dir)"
    if [[ ! -d "$root/.git" ]]; then
        gitdir="$(git -C "$root" rev-parse --absolute-git-dir)"
        [[ "$gitdir" == "$common" ]] ||
            die "$root is a linked worktree (its repository is shared): push from a clone"
    fi
    if [[ -f "$common/objects/info/alternates" ]]; then
        while IFS= read -r alt; do
            [[ -n "$alt" && "$alt" != "#"* ]] || continue
            [[ "$alt" == /* ]] || alt="$common/objects/$alt"
            [[ -d "$alt" && ! -e "$alt/info/alternates" ]] ||
                die "alternate $alt of $common is missing or has alternates of its own"
            alternates+=("$alt")
        done <"$common/objects/info/alternates"
    fi
    local dest="$rsync_dir/torch-op-bench" q git_exclude=/.git/objects/info/alternates
    q="$(printf '%q' "$dest")"
    [[ -z "$gitdir" ]] || git_exclude=/.git
    local prepare="mkdir -p $q && { [ -d $q/.git ] || rm -f $q/.git; }"
    prepare+=" && rm -f $q/.git/objects/info/alternates"
    # Before the first rsync, on HOST: make the directory, remove a .git that is a
    # file (rsync does not replace a file with a directory) and an alternates file
    # an older push left.
    local_cmd rsync -a --delete \
        --exclude=/.asb/ --exclude=/envs/.prefix/ --exclude=/results/ \
        --exclude="$git_exclude" \
        --exclude=__pycache__/ --exclude=.pytest_cache/ --exclude=.ruff_cache/ \
        --exclude=htmlcov/ --exclude='.coverage*' \
        --rsync-path="$prepare && rsync" \
        "$root/" "$host:$dest/"
    if [[ -n "$gitdir" ]]; then
        local_cmd rsync -a --delete --exclude=/objects/info/alternates \
            "$gitdir/" "$host:$dest/.git/"
    fi
    for alt in "${alternates[@]}"; do
        local_cmd rsync -a --exclude=/info/ "$alt/" "$host:$dest/.git/objects/"
    done
}

cmd_bootstrap() {
    remote 'mkdir -p cache
conda_bin=""
if [ -x "$D/miniforge3/bin/conda" ]; then
    conda_bin="$D/miniforge3/bin"
    echo "Miniforge already in $D/miniforge3"
elif command -v mamba >/dev/null 2>&1 || command -v conda >/dev/null 2>&1; then
    echo "using $(command -v mamba || command -v conda) from PATH"
else
    installer="$D/cache/Miniforge3-Linux-$(uname -m).sh"
    curl -fsSL -o "$installer" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
    # The installer runs conda, which makes ~/.conda: give it a home in the scratch.
    mkdir -p "$D/cache/home"
    HOME="$D/cache/home" CONDA_REGISTER_ENVS=false bash "$installer" -b -p "$D/miniforge3"
    conda_bin="$D/miniforge3/bin"
fi
{
    echo "# Written by scripts/remote-run.sh bootstrap: everything under $D."
    if [ -n "$conda_bin" ]; then echo "export PATH=\"$conda_bin:\$PATH\""; fi
    echo "export TRITON_HOME=\"$D/cache/triton\""
    echo "export TORCHINDUCTOR_CACHE_DIR=\"$D/cache/inductor\""
    echo "export CUDA_CACHE_PATH=\"$D/cache/nv\""
    echo "export MAMBA_ROOT_PREFIX=\"$D/cache/mamba\""
    echo "export CONDA_PKGS_DIRS=\"$D/cache/conda-pkgs\""
    echo "export CONDA_REGISTER_ENVS=false"
    echo "export PIP_CACHE_DIR=\"$D/cache/pip\""
    echo "export XDG_CACHE_HOME=\"$D/cache\""
    echo "export MPLCONFIGDIR=\"$D/cache/matplotlib\""
} > env.sh
echo "wrote $D/env.sh"'
}

cmd_setup() {
    local names="" name
    for name in "$torch_env" "${envs[@]}"; do
        names+=" $(printf '%q' "$name")"
    done
    remote "$source_env
cd torch-op-bench
# mamba 2 registers environments in ~/.conda/environments.txt whatever
# CONDA_REGISTER_ENVS says: it gets a home in the scratch, as the installer does.
mkdir -p \"\$D\"/cache/home
for name in$names; do
    HOME=\"\$D\"/cache/home scripts/setup-env.sh \"\$name\" --prefix \"\$D\"/envs
done
nvidia-smi || echo 'nvidia-smi failed'
$torch_python -m torch_op_bench.cli list $(printf '%q' "$op")"
}

# --env NAME=PYTHON for each counterpart environment: NAME is the definition's own
# `name:` (envs/liger-git.yml defines the environment liger).
env_args_script() {
    local name
    echo 'env_args=()'
    for name in "${envs[@]}"; do
        echo "env_args+=(--env \"\$(awk '/^name:/ {print \$2; exit}' torch-op-bench/envs/$name.yml)=\$D/envs/$name/bin/python\")"
    done
}

cmd_run() {
    local args="" arg
    for arg in "${run_args[@]}"; do
        args+=" $(printf '%q' "$arg")"
    done
    local visible=""
    if [[ -n "$gpu" ]]; then
        visible="export CUDA_VISIBLE_DEVICES=$gpu"
    fi
    remote "$source_env
$visible
if [ -f run.pid ] && kill -0 \"\$(cat run.pid)\" 2>/dev/null; then
    echo \"a run is in progress (pid \$(cat run.pid)): status or wait\" >&2
    exit 1
fi
mkdir -p logs results
$(env_args_script)
log=\"\$D/logs/\$(date -u +%Y%m%dT%H%M%SZ).log\"
cmd=($torch_python -m torch_op_bench.cli run $(printf '%q' "$op") --device cuda --results \"\$D\"/results \"\${env_args[@]}\"$args)
printf '%q ' \"\${cmd[@]}\" > \"\$log\"
printf '\\n' >> \"\$log\"
rm -f run.status
cd torch-op-bench
D=\"\$D\" setsid nohup bash -c '\"\$@\"; echo \$? > \"\$D\"/run.status' run \"\${cmd[@]}\" >> \"\$log\" 2>&1 < /dev/null &
echo \$! > \"\$D\"/run.pid
echo \"started pid \$(cat \"\$D\"/run.pid), log \$log\""
}

cmd_status() {
    remote "$source_env
if [ -f run.pid ] && kill -0 \"\$(cat run.pid)\" 2>/dev/null; then
    echo \"running: pid \$(cat run.pid)\"
elif [ -f run.status ]; then
    echo \"not running: the last run exited with status \$(cat run.status)\"
else
    echo 'not running'
fi
log=\"\$(ls -1t logs/*.log 2>/dev/null | head -1 || true)\"
if [ -n \"\$log\" ]; then
    echo \"== \$log\"
    tail -n 20 \"\$log\"
fi
echo '== records'
$torch_python -m torch_op_bench.remote count \"\$D\"/results"
}

cmd_wait() {
    remote "[ -f run.pid ] || { echo 'no run started here' >&2; exit 1; }
pid=\"\$(cat run.pid)\"
log=\"\$(ls -1t logs/*.log | head -1)\"
echo \"following \$log until pid \$pid ends\"
tail -n +1 -f --pid=\"\$pid\" \"\$log\"
status=\"\$(cat run.status 2>/dev/null || echo 1)\"
echo \"the run exited with status \$status\"
exit \"\$status\""
}

cmd_fetch() {
    local changes slug
    printf '+ rsync -ai %q %q\n' "$host:$rsync_dir/results/" "$root/results/"
    ((dry_run)) && return 0
    mkdir -p "$root/results"
    changes="$(rsync -ai "$host:$rsync_dir/results/" "$root/results/")"
    echo "$changes"
    for slug in $(sed -n "s|^>f[^ ]* $op/\([^/]*\)/.*\.jsonl$|\1|p" <<<"$changes" | sort -u); do
        echo "== $op/$slug"
        PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python3}" -m torch_op_bench.cli \
            report "$root/results" --op "$op" --device-slug "$slug" 2>&1 | head -n 25
    done
}

cmd_shell() {
    local script
    script="cd $remote_d && { [ -f env.sh ] && source env.sh; }; exec bash -i"
    printf '+ ssh -t %q %q\n' "$host" "$script"
    if ((!dry_run)); then
        ssh -t "$host" "$script"
    fi
}

case "$subcommand" in
    push | bootstrap | setup | run | status | wait | fetch | shell) "cmd_$subcommand" ;;
    *)
        echo "remote-run.sh: unknown subcommand $subcommand" >&2
        usage >&2
        exit 2
        ;;
esac
