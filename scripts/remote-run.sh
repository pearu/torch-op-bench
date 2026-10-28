#!/usr/bin/env bash
# Run benchmarks on a remote host from this checkout. See "Remote runs" in README.md.
set -euo pipefail

usage() {
    cat <<'USAGE'
usage: scripts/remote-run.sh HOST SUBCOMMAND [options] [-- RUN-ARGS...]

Run torch-op-bench on HOST (an ssh destination or alias) in a scratch directory
there, from this checkout. Each subcommand is one ssh session or one rsync, except
push: an rsync of the repository (one more for each alternate), one of the tracked
files, and an ssh session.

subcommands:
  push       copy the tracked files and .git to REMOTE_DIR/torch-op-bench, and
             nothing untracked or ignored; refuses unless the tracked files are
             clean, so every record names one commit, and while a run is alive
  bootstrap  install Miniforge into REMOTE_DIR/miniforge3 unless mamba or conda is
             on PATH, and write REMOTE_DIR/env.sh (caches under REMOTE_DIR/cache)
  setup      create the torch environment and each of --envs under REMOTE_DIR/envs
             with scripts/setup-env.sh; show their provenance, nvidia-smi and
             `torch-op-bench list OP`
  run        start `torch-op-bench run OP --device cuda` detached (nohup, setsid),
             RUN-ARGS after `--`, into REMOTE_DIR/results/ (results-strict/ with
             --matmul-reduction strict); returns at once; rerun after a disconnect
             to resume (recorded ok triples are skipped)
  status     whether the run is alive, the end of its log, records per status
  wait       follow the log until the run ends; exit with its status
  fetch      copy REMOTE_DIR/results/ and results-strict/ (when there) into this
             checkout's results/ and results-strict/ (never deletes anything on
             HOST) and show the first lines of each new slug's report
  shell      an interactive shell on HOST in REMOTE_DIR with env.sh sourced

options:
  --dir REMOTE_DIR     scratch directory on HOST (default ~/torch_op_bench-scratch)
  --torch-env NAME     torch-nightly (default) or torch-stable: the definition of the
                       torch environment, whose interpreter runs torch-op-bench
  --envs NAME,...      counterpart environments to create and pass to run
                       (e.g. liger,torchtitan)
  --gpu N              CUDA_VISIBLE_DEVICES=N on HOST for run (default: all)
  --op OP              the op to run, list and report (default lce)
  --matmul-reduction default|strict
                       run under torch's matmul defaults (default), or strict: no
                       reduced-precision reduction in bf16/fp16 GEMMs, into
                       results-strict/ (the resume key does not hold the setting)
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
matmul_reduction="default"
allow_dirty=0
dry_run=0
run_args=()
while (($#)); do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --dir | --torch-env | --envs | --gpu | --op | --matmul-reduction)
            [[ $# -ge 2 ]] || die "$1 needs a value"
            case "$1" in
                --matmul-reduction) matmul_reduction="$2" ;;
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
case "$matmul_reduction" in
    default) results_name="results" ;;
    strict) results_name="results-strict" ;;
    *) die "--matmul-reduction must be default or strict, not $matmul_reduction" ;;
esac
# Both choose the results directory: a strict run must not land in results/.
for arg in "${run_args[@]}"; do
    case "$arg" in
        --results | --results=* | --matmul-reduction | --matmul-reduction=*)
            die "${arg%%=*} is remote-run.sh's own option (before --), not a RUN-ARG"
            ;;
    esac
done
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
    # The remote checkout is exactly the tracked files plus .git: nothing untracked
    # or ignored leaves this machine (the records' dirty flag is about tracked files,
    # so nothing else may run there). The repository goes first, whole: a checkout
    # whose .git is a "gitdir: ..." file (git clone --separate-git-dir) has that
    # directory copied, and objects borrowed through objects/info/alternates (git
    # clone --shared or --reference) are copied into it, without the alternates
    # file; one rsync more each. Then the files `git ls-files` lists, and on HOST
    # `git clean` removes what is no longer tracked.
    local common gitdir alternates=() alt
    common="$(git -C "$root" rev-parse --path-format=absolute --git-common-dir)"
    gitdir="$(git -C "$root" rev-parse --absolute-git-dir)"
    [[ "$gitdir" == "$common" ]] ||
        die "$root is a linked worktree (its repository is shared): push from a clone"
    if [[ -f "$common/objects/info/alternates" ]]; then
        while IFS= read -r alt; do
            [[ -n "$alt" && "$alt" != "#"* ]] || continue
            [[ "$alt" == /* ]] || alt="$common/objects/$alt"
            [[ -d "$alt" && ! -e "$alt/info/alternates" ]] ||
                die "alternate $alt of $common is missing or has alternates of its own"
            alternates+=("$alt")
        done <"$common/objects/info/alternates"
    fi
    local dest="$rsync_dir/torch-op-bench" q pid_file old_head
    q="$(printf '%q' "$dest")"
    pid_file="$(printf '%q' "$rsync_dir/run.pid")"
    old_head="$(printf '%q' "$rsync_dir/push.old-head")"
    # Before the first rsync, on HOST: refuse while a run is alive (it imports from
    # the checkout; --allow-dirty does not override this), make the directory,
    # record the checkout's HEAD (none on a first push) for the step after the
    # copy, and remove a .git that is a file (rsync does not replace a file with a
    # directory) and an alternates file an older push left. A push.old-head that is
    # already there is kept: only a completed push removes it, so the last push was
    # cut off, and its HEAD is the last one whose files were all on the host. git is
    # asked only when .git is there: otherwise it would look in the directories above.
    local prepare="if p=\$(cat $pid_file 2>/dev/null) && kill -0 \"\$p\" 2>/dev/null; then"
    prepare+=" echo \"remote-run.sh: a run is in progress (pid \$p): wait for it or stop it\" >&2;"
    prepare+=" exit 1; fi; mkdir -p $q"
    prepare+=" && { [ -s $old_head ] || { [ -d $q/.git ] &&"
    prepare+=" git -C $q rev-parse -q --verify HEAD >$old_head 2>/dev/null; } || rm -f $old_head; }"
    prepare+=" && { [ -d $q/.git ] || rm -f $q/.git; }"
    prepare+=" && rm -f $q/.git/objects/info/alternates"
    # --checksum: refs, the index and tracked files change in place, and the quick
    # check (size and time) misses a change of the same size within the same second.
    local_cmd rsync -a --checksum --delete --exclude=/objects/info/alternates \
        --rsync-path="$prepare && rsync" "$gitdir/" "$host:$dest/.git/"
    for alt in "${alternates[@]}"; do
        local_cmd rsync -a --exclude=/info/ "$alt/" "$host:$dest/.git/objects/"
    done
    # A tracked file deleted but not committed (--allow-dirty) is listed and missing.
    local tree_cmd=(rsync -a --checksum --from0 --files-from=- --ignore-missing-args
        "$root/" "$host:$dest/")
    printf '+ git -C %q ls-files -z |' "$root"
    printf ' %q' "${tree_cmd[@]}"
    printf '\n'
    if ((!dry_run)); then
        git -C "$root" ls-files -z | "${tree_cmd[@]}"
    fi
    remote 'cd torch-op-bench
# What the previous push tracked and HEAD does not, ignored now or not: git clean
# leaves a file that left the index and became ignored in one commit.
if [ -s "$D"/push.old-head ]; then
    old="$(cat "$D"/push.old-head)"
    if git cat-file -e "$old^{commit}" 2>/dev/null; then
        gone=(--name-only --no-renames --diff-filter=D "$old" HEAD)
        git diff "${gone[@]}" | sed "s/^/no longer tracked, removing /"
        git diff -z "${gone[@]}" | xargs -0 rm -f --
    else
        echo "the previous push ($old) is not in this history: only git clean below" >&2
    fi
fi
rm -f "$D"/push.old-head
# What is not tracked and not ignored: files deleted since the previous push.
# Ignored files (caches, envs/.prefix/) stay, and so does results/.
git clean -fd -e /envs/.prefix/ -e /results/
echo "remote checkout: $(git rev-parse HEAD)"
git status --short --untracked-files=no'
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
    local reduction_arg=""
    if [[ "$matmul_reduction" != default ]]; then
        reduction_arg=" --matmul-reduction $matmul_reduction"
    fi
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
mkdir -p logs $results_name
$(env_args_script)
log=\"\$D/logs/\$(date -u +%Y%m%dT%H%M%SZ).log\"
cmd=($torch_python -m torch_op_bench.cli run $(printf '%q' "$op") --device cuda --results \"\$D\"/$results_name$reduction_arg \"\${env_args[@]}\"$args)
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
    local changes slug name
    # Each directory, not its contents, into this checkout: results-strict/ is there
    # only after a strict run, and a missing source is skipped (--ignore-missing-args
    # does not skip a missing DIR/, which rsync changes into).
    for name in results results-strict; do
        printf '+ rsync -ai --ignore-missing-args %q %q\n' "$host:$rsync_dir/$name" "$root/"
        ((dry_run)) && continue
        changes="$(rsync -ai --ignore-missing-args "$host:$rsync_dir/$name" "$root/")"
        echo "$changes"
        for slug in $(sed -n "s|^>f[^ ]* $name/$op/\([^/]*\)/.*\.jsonl$|\1|p" <<<"$changes" | sort -u); do
            echo "== $name/$op/$slug"
            PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python3}" -m torch_op_bench.cli \
                report "$root/$name" --op "$op" --device-slug "$slug" 2>&1 | head -n 25
        done
    done
}

cmd_shell() {
    # Run by bash, as every other remote script is, not by the login shell (sh or
    # dash would not source env.sh). Not bash -s: stdin stays the terminal.
    local script
    script="cd $remote_d && { [ -f env.sh ] && source env.sh; }; exec bash -i"
    script="bash -c $(printf '%q' "$script")"
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
