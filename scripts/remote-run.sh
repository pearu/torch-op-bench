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
             clean, so every record names one commit, and while a run is alive;
             with --wheelhouse, also a directory of wheels
  bootstrap  install Miniforge into REMOTE_DIR/miniforge3 unless mamba or conda is
             on PATH, and write REMOTE_DIR/env.sh (caches under REMOTE_DIR/cache)
  setup      create the torch environment and each of --envs under REMOTE_DIR/envs
             with scripts/setup-env.sh, and list --envs in REMOTE_DIR/envs.list;
             show their provenance, hostname, nvidia-smi and `torch-op-bench list OP`
  run        start `torch-op-bench run OP --device cuda` detached (nohup, setsid),
             RUN-ARGS after `--`, into REMOTE_DIR/RESULTS/ (below); returns at
             once; rerun after a disconnect to resume (recorded ok triples are
             skipped); without --envs, the environments of envs.list
  status     whether the run is alive and into which RESULTS/, the end of its log,
             records per status in each RESULTS/ there
  wait       follow the log until the run ends; exit with its status
  fetch      copy each RESULTS/ there into this checkout's (never deletes anything
             on HOST), say how many records each slug gained, and show the first
             lines of its report; warns while a run is alive
  shell      an interactive bash on HOST in REMOTE_DIR with env.sh sourced

RESULTS is results, or results-stable with --torch-env torch-stable, with -strict
appended under --matmul-reduction strict: the resume key holds neither the build
nor the setting, so neither may share a directory with the default.

options:
  --dir REMOTE_DIR     scratch directory on HOST (default ~/torch_op_bench-scratch)
  --torch-env NAME     torch-nightly (default) or torch-stable: the definition of the
                       torch environment, whose interpreter runs torch-op-bench
  --envs NAME,...      counterpart environments to create and pass to run
                       (e.g. liger,torchtitan)
  --gpu N              CUDA_VISIBLE_DEVICES=N on HOST for run (default: all; not
                       on a host whose sandbox shows one card)
  --op OP              the op to run, list and report (default lce)
  --matmul-reduction default|strict
                       run under torch's matmul defaults (default), or strict: no
                       reduced-precision reduction in bf16/fp16 GEMMs, into
                       RESULTS-strict/
  --wheelhouse DIR     push: copy the wheels in DIR to REMOTE_DIR/wheelhouse, which
                       setup's pip searches (--find-links) before the indexes
  --allow-dirty        push uncommitted changes (development only: the records then
                       name a commit that is not what ran)
  --dry-run            print every local and remote command, run none
  -h, --help           this message

Nothing is written on HOST outside REMOTE_DIR (and TMPDIR, which is REMOTE_DIR/tmp
unless the session sets one).
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
wheelhouse=""
allow_dirty=0
dry_run=0
run_args=()
while (($#)); do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --dir | --torch-env | --envs | --gpu | --op | --matmul-reduction | --wheelhouse)
            [[ $# -ge 2 ]] || die "$1 needs a value"
            case "$1" in
                --matmul-reduction) matmul_reduction="$2" ;;
                --dir) remote_dir="$2" ;;
                --torch-env) torch_env="$2" ;;
                --envs) IFS=, read -r -a more <<<"$2" && envs+=("${more[@]}") ;;
                --gpu) gpu="$2" ;;
                --op) op="$2" ;;
                --wheelhouse) wheelhouse="$2" ;;
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
# The results directory: the resume and merge key holds neither the torch build nor
# the matmul setting, so a release run or a strict run must not land in results/
# (it would be skipped against the nightly's records, and duplicate their keys).
case "$torch_env" in
    torch-nightly) results_name="results" ;;
    torch-stable) results_name="results-stable" ;;
    *) die "--torch-env must be torch-nightly or torch-stable, not $torch_env" ;;
esac
[[ -z "$gpu" || "$gpu" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--gpu takes GPU indices, not $gpu"
case "$matmul_reduction" in
    default) ;;
    strict) results_name+="-strict" ;;
    *) die "--matmul-reduction must be default or strict, not $matmul_reduction" ;;
esac
# Every results directory a run can write, for status and fetch.
all_results=(results results-strict results-stable results-stable-strict)
if [[ -n "$wheelhouse" ]]; then
    [[ "$subcommand" == push ]] || die "--wheelhouse is for push only"
    [[ -d "$wheelhouse" ]] || die "--wheelhouse $wheelhouse is not a directory"
fi
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
    # --force: a tracked file that replaced a directory of the same name replaces it
    # on HOST too, also when the directory holds something untracked (a
    # __pycache__); only paths of the list are replaced. A directory that replaced a
    # file needs nothing: rsync replaces a file with a directory by default.
    local tree_cmd=(rsync -a --force --checksum --from0 --files-from=- --ignore-missing-args
        "$root/" "$host:$dest/")
    printf '+ git -C %q ls-files -z |' "$root"
    printf ' %q' "${tree_cmd[@]}"
    printf '\n'
    if ((!dry_run)); then
        git -C "$root" ls-files -z | "${tree_cmd[@]}"
    fi
    if [[ -n "$wheelhouse" ]]; then
        local_cmd rsync -a "$wheelhouse/" "$host:$rsync_dir/wheelhouse/"
    fi
    remote 'cd torch-op-bench
# What the previous push tracked and HEAD does not, ignored now or not: git clean
# leaves a file that left the index and became ignored in one commit.
if [ -s "$D"/push.old-head ]; then
    old="$(cat "$D"/push.old-head)"
    if git cat-file -e "$old^{commit}" 2>/dev/null; then
        # A file that became a directory of tracked files is that directory now.
        git diff -z --name-only --no-renames --diff-filter=D "$old" HEAD |
            while IFS= read -r -d "" path; do
                [ -d "$path" ] && continue
                echo "no longer tracked, removing $path"
                rm -f -- "$path"
            done
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
    url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
    # A proxy that refuses github.com or .githubusercontent.com answers 403 (above).
    curl -fsSL -o "$installer" "$url" ||
        { echo "bootstrap: could not download the Miniforge installer, $url" >&2; exit 1; }
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
    echo "# The TMPDIR of the session (a sandbox sets one), else one here."
    echo "if [ -z \"\${TMPDIR:-}\" ]; then export TMPDIR=\"$D/tmp\"; fi"
    echo "mkdir -p \"\$TMPDIR\""
    echo "# The CA of a proxy, for mamba and conda: libmamba ignores SSL_CERT_FILE."
    echo "if [ -n \"\${SSL_CERT_FILE:-}\" ]; then"
    echo "    export MAMBA_SSL_VERIFY=\"\${MAMBA_SSL_VERIFY:-\$SSL_CERT_FILE}\""
    echo "    export CONDA_SSL_VERIFY=\"\${CONDA_SSL_VERIFY:-\$SSL_CERT_FILE}\""
    echo "fi"
} > env.sh
echo "wrote $D/env.sh"
# What shell starts bash with: the login shell (whatever it is) only runs
# `bash --rcfile shell.rc -i`, and HOME may be read-only, so the history is here.
{
    echo "# Written by scripts/remote-run.sh bootstrap: the rc file of its shell."
    echo "cd \"$D\""
    echo ". \"$D/env.sh\""
    echo "HISTFILE=\"$D/cache/bash_history\""
} > shell.rc
echo "wrote $D/shell.rc"'
}

cmd_setup() {
    local names="" name
    for name in "$torch_env" "${envs[@]}"; do
        names+=" $(printf '%q' "$name")"
    done
    # The counterpart environments, for run without --envs; a setup without --envs
    # (another torch environment) leaves the list as it was.
    local list="" counterparts=""
    if ((${#envs[@]})); then
        for name in "${envs[@]}"; do
            counterparts+=" $(printf '%q' "$name")"
        done
        list="printf '%s\\n'$counterparts > \"\$D\"/envs.list
echo 'envs.list:$counterparts'"
    fi
    remote "$source_env
cd torch-op-bench
# mamba 2 registers environments in ~/.conda/environments.txt whatever
# CONDA_REGISTER_ENVS says: it gets a home in the scratch, as the installer does.
mkdir -p \"\$D\"/cache/home
# The wheels of push --wheelhouse, which pip (that of mamba's pip step too) searches
# before the indexes: PIP_FIND_LINKS is its --find-links, and keeps the session's.
if [ -d \"\$D\"/wheelhouse ]; then
    export PIP_FIND_LINKS=\"\$D/wheelhouse\${PIP_FIND_LINKS:+ \$PIP_FIND_LINKS}\"
    echo \"pip finds links in \$PIP_FIND_LINKS\"
fi
for name in$names; do
    HOME=\"\$D\"/cache/home scripts/setup-env.sh \"\$name\" --prefix \"\$D\"/envs || {
        echo \"setup: the environment \$name failed: the step and its error are above\" \\
            \"(a host the proxy refuses answers 403)\" >&2
        exit 1
    }
done
$list
echo \"hostname: \$(hostname)\"
nvidia-smi || echo 'nvidia-smi failed'
$torch_python -m torch_op_bench.cli list $(printf '%q' "$op")"
}

# --env NAME=PYTHON for each counterpart environment, those of --envs or else those
# setup listed in envs.list: NAME is the definition's own `name:` (envs/liger-git.yml
# defines the environment liger). A missing environment stops the run before it
# starts, rather than recording its entries unavailable.
env_args_script() {
    if ((${#envs[@]})); then
        echo "env_names=(${envs[*]})"
    else
        cat <<'SCRIPT'
env_names=()
if [ -s envs.list ]; then
    mapfile -t env_names < <(grep . envs.list)
    echo "the environments of envs.list: ${env_names[*]}"
else
    echo "no --envs and no envs.list (setup --envs): no counterpart environments" >&2
fi
SCRIPT
    fi
    cat <<'SCRIPT'
env_args=()
for name in "${env_names[@]}"; do
    [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad environment name '$name'" >&2; exit 1; }
    [ -x "$D/envs/$name/bin/python" ] ||
        { echo "no environment $name in $D/envs: setup --envs $name first" >&2; exit 1; }
    env_args+=(--env "$(awk '/^name:/ {print $2; exit}' torch-op-bench/envs/$name.yml)=$D/envs/$name/bin/python")
done
SCRIPT
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
# The log's header: where the run is and which cards it sees, then the command.
{
    echo \"# host: \$(hostname)\"
    echo \"# CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES-(not set)}\"
    nvidia-smi -L 2>&1 | sed 's/^/# /' || echo '# nvidia-smi failed'
    printf '%q ' \"\${cmd[@]}\"
    printf '\\n'
} > \"\$log\"
rm -f run.status
echo $results_name > run.results
cd torch-op-bench
D=\"\$D\" setsid nohup bash -c '\"\$@\"; echo \$? > \"\$D\"/run.status' run \"\${cmd[@]}\" >> \"\$log\" 2>&1 < /dev/null &
echo \$! > \"\$D\"/run.pid
echo \"started pid \$(cat \"\$D\"/run.pid), log \$log\""
}

cmd_status() {
    remote "$source_env
into=\"\$(cat run.results 2>/dev/null || echo results)\"
if [ -f run.pid ] && kill -0 \"\$(cat run.pid)\" 2>/dev/null; then
    echo \"running: pid \$(cat run.pid), into \$into/\"
elif [ -f run.status ]; then
    echo \"not running: the last run exited with status \$(cat run.status), into \$into/\"
else
    echo 'not running'
fi
log=\"\$(ls -1t logs/*.log 2>/dev/null | head -1 || true)\"
if [ -n \"\$log\" ]; then
    echo \"== \$log\"
    tail -n 20 \"\$log\"
fi
found=0
for name in ${all_results[*]}; do
    [ -d \"\$name\" ] || continue
    found=1
    echo \"== records in \$name/\"
    $torch_python -m torch_op_bench.remote count \"\$D\"/\"\$name\"
done
[ \$found = 1 ] || echo '== no records yet'"
}

cmd_wait() {
    remote "[ -f run.pid ] || { echo 'no run started here' >&2; exit 1; }
pid=\"\$(cat run.pid)\"
log=\"\$(ls -1t logs/*.log | head -1)\"
into=\"\$(cat run.results 2>/dev/null || echo results)\"
echo \"following \$log until pid \$pid ends (records into \$D/\$into/)\"
tail -n +1 -f --pid=\"\$pid\" \"\$log\"
status=\"\$(cat run.status 2>/dev/null || echo 1)\"
echo \"the run exited with status \$status\"
exit \"\$status\""
}

# "SLUG RECORDS" for each slug of DIR/OP: the non-empty lines of its files.
slug_counts() {
    local dir
    for dir in "$1/$op"/*/; do
        [[ -d "$dir" ]] || continue
        printf '%s %s\n' "$(basename "$dir")" "$(cat "$dir"*.jsonl 2>/dev/null | grep -c . || true)"
    done
}

cmd_fetch() {
    local slug name before gained
    # A record still being written is a half line, on which report fails until the
    # next fetch.
    remote 'if [ -f run.pid ] && kill -0 "$(cat run.pid)" 2>/dev/null; then
    echo "remote-run.sh: WARNING: a run is in progress (pid $(cat run.pid)): the last line" \
        "of a file may be half-written; fetch again when the run has ended" >&2
fi'
    # Each directory, not its contents, into this checkout: each is there only after
    # its kind of run, and a missing source is skipped (--ignore-missing-args does
    # not skip a missing DIR/, which rsync changes into).
    for name in "${all_results[@]}"; do
        printf '+ rsync -ai --ignore-missing-args %q %q\n' "$host:$rsync_dir/$name" "$root/"
        ((dry_run)) && continue
        before="$(slug_counts "$root/$name")"
        rsync -ai --ignore-missing-args "$host:$rsync_dir/$name" "$root/"
        # The files are append-only: what a slug gained is its new lines.
        gained="$(awk 'NR == FNR { had[$1] = $2; next } $2 > had[$1] { print $1, $2 - had[$1] }' \
            <(echo "$before") <(slug_counts "$root/$name"))"
        [[ -n "$gained" ]] || continue
        while read -r slug count; do
            echo "== $name/$op/$slug: $count new records"
            # head may close the pipe before a long report ends: not a failure.
            PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python3}" -m torch_op_bench.cli \
                report "$root/$name" --op "$op" --device-slug "$slug" 2>&1 | head -n 25 || true
        done <<<"$gained"
    done
}

cmd_shell() {
    # bash with bootstrap's rc file (cd, env.sh), whatever the login shell is: it
    # gets three plain words to run, and no syntax of its own. Not bash -s: stdin
    # stays the terminal. The path is relative to the home for ~/..., where ssh
    # starts.
    local script
    script="bash --rcfile $(printf '%q' "$rsync_dir/shell.rc") -i"
    printf '+ ssh -t %q %s\n' "$host" "$script"
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
