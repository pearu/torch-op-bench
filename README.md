# torch-op-bench

A tool for benchmarking a torch op against its counterparts. It measures throughput,
peak memory and parity against a higher-precision reference. Each measurement runs in
its own worker process, possibly in a different environment (Python and torch) per
counterpart, and is written as a self-describing JSON Lines record. The record
carries the case, the settings and the provenance (host, GPU, driver, CUDA, torch
build, package versions, the torch-op-bench commit), so results from several hosts
and builds can be merged and compared. The core knows nothing about any particular
op: an op is described by a spec, and the core runs on `cpu` as well as `cuda`, so
the whole engine is tested on CPU.

## Install

torch is not a dependency: it is taken from the environment under test. In an
environment that has torch:

```sh
pip install -e .            # the tool
pip install -e .[dev,plot]  # plus tests, lint and plotting dependencies
```

Install torch-op-bench from the same checkout into every environment a benchmark
runs in. When `torch-op-bench run` itself runs from a checkout, its workers import
torch-op-bench from that checkout in every environment, whatever the working
directory, so the parent and the workers always run the same code. A counterpart can also be installed into the torch environment itself, e.g.
`pip install -e .[liger]`; it then runs there instead of in its own environment.

## Environments

A counterpart usually runs in an environment of its own, with its own Python, torch
and packages. `envs/NAME.yml` defines one: a conda environment file with every
version pinned, since a results file names the definition and a definition that
drifts would make the records lie. One script creates it:

```sh
scripts/setup-env.sh liger            # creates envs/.prefix/liger, or updates it
scripts/setup-env.sh liger --prefix /scratch/envs
```

It uses mamba if it is on `PATH`, conda otherwise, installs this checkout into the
environment (`pip install -e .`, no extras), and prints the provenance its records
will carry (`torch-op-bench env` with the packages the definition's `pip:` section
names). The last line of its output is the environment's interpreter, which a run
takes as `--env NAME=PYTHON`:

```sh
torch-op-bench run lce --device cuda --env liger=$(scripts/setup-env.sh liger | tail -1)
torch-op-bench run lce --device cuda --env liger=envs/.prefix/liger/bin/python
```

Each counterpart has two definitions, and both make an environment its
implementations address by the same name:

- `envs/NAME.yml`: the PyPI release;
- `envs/NAME-git.yml`: a commit of the counterpart's repository, for what is under
  development and has no nightly wheels, as `name @ git+https://...@<40-hex hash>`.
  To bump it, replace the hash (a branch name is not a pin) and the date next to it,
  and rerun the script.

The implementations do not know which definition they run in; the record does. For
a package installed from a VCS URL, `env.packages` holds
`{"version", "url", "commit", "requested"}` (read from the `direct_url.json` that pip
writes) instead of a version string, so `env.packages["liger-kernel"]` is `"0.8.4"`
for `envs/liger.yml` and carries the pinned commit for `envs/liger-git.yml`.

**Controls.** A comparison across environments includes the torch build: the PyPI
wheel in `liger` is not the torch of the `torch` environment. So the baseline also
runs in each counterpart's environment, as `torch_eager_<env>`
(`Implementation.as_control(env)`), with `info.role: "control"` and
`info.control_of: "torch_eager"`. Compare a counterpart with the control of its own
environment; `torch_eager` against `torch_eager_<env>` is the difference the
environment makes.

## Run the toy example

The tests define a toy op, `y = (a @ b).sum()`, in `tests/toy_op.py`:

```sh
torch-op-bench list toy_op:TOY --pythonpath tests
torch-op-bench run toy_op:TOY --pythonpath tests --dtype float32 --sweep M \
    --env other=$(which python) --warmup 2 --repeat 5
torch-op-bench run toy_op:TOY --pythonpath tests --device cuda --impl plain,chunked \
    --case M=256,K=256 --dtype float32
torch-op-bench env          # the provenance record this interpreter would write
```

`run` varies one axis at a time around the op's default point (`--sweep AXIS` limits
the axes, `--values AXIS=V1,V2` overrides a swept axis' values), or runs the given points
(`--case AXIS=V,AXIS=V`, repeatable, completed from the default point). Further
flags: `--dtype`, `--impl`, `--mode fwd|fwd_bwd`, `--warmup`, `--repeat`,
`--flush-l2`, `--seed`, `--timeout` (seconds per worker), `--results DIR` (default
`results`), `--force`, `--env NAME=PYTHON` (the interpreter of the environment an
implementation names; `torch` defaults to the running one), `--pythonpath DIR` and
`--device-slug`.

An interrupted run resumes: rerunning the same command skips every
(case, implementation, mode) that already has an `ok` record, unless `--force`.

## Add an op

An op is an `OpSpec` (see `torch_op_bench/spec.py`), addressed either by a short name
registered in `torch_op_bench/ops/__init__.py` or by a dotted path `module:OBJECT`.
The parent process (`run`, `list`) imports the op's module to enumerate cases, so the
module should import torch and any counterpart lazily, inside its callables: then the
parent needs neither, and each worker imports torch only after its environment
variables are set.

- `axes`: a list of `Axis(name, values, kind="int"|"str")` (names and `str` values
  must not contain `,`, `=` or `|`, which structure case keys); `defaults` gives the
  point a sweep varies one axis around (an axis missing from it defaults to its
  first value).
- `dtypes`: dtype names (`"float32"`, `"bfloat16"`, ...); `default_dtypes`: those a
  `run` uses without `--dtype` (default: all of `dtypes`).
- `make_inputs(case, device, generator)`: the inputs of a `Case` (`case.params`,
  `case.dtype`, `case.device`), seeded by the `torch.Generator`.
- `implementations`: a list of `Implementation`:
  - `run(inputs, mode)`: the timed computation, for `mode` in `modes` (`fwd`,
    `fwd_bwd`). It is called many times on the same inputs; before every call the
    core sets `.grad` to `None` on each tensor in the inputs (walking tuples, lists
    and dicts), outside the measured region, so `loss.backward()` and
    `torch.autograd.grad` styles both start from the same state. Any other state a
    `run` leaves behind is its own to undo;
  - `available()`: `True`, or a string saying why not; import optional dependencies
    here, not at module level, so that the spec imports without them;
  - `env_name` (default `torch`): the environment its worker runs in;
    `env_vars`: set in the worker before torch is imported;
  - `setup(case)`: run once in the worker before anything is measured; it may
    return a dict of facts known only at run time (resolved options, a source hash,
    whether it compiled), merged over `info` into the record's `info`;
  - `verify(fn)`: given the timed callable, check (e.g. with a profiler) that the
    intended code path runs; raise if not; return a JSON-serialisable dict or `None`;
  - `packages`: distributions whose versions the record carries; `info`: free-form
    facts recorded as is (e.g. chunk size, accumulation dtype).
- `reference`: an `Implementation` at higher precision, run once per case on the same
  inputs; `compare(outputs, ref_outputs, case)` returns named error metrics, checked
  against `tolerances`; `reference_fits(case)` returns `True` or a reason to skip the
  parity check (a memory guard).
- `work(case)`: work units of a case (e.g. `{"tokens": N, "flops": ...}`) from which
  reports derive throughput.

Each (case, implementation, mode) runs in a fresh worker
(`python -m torch_op_bench.worker`, a JSON payload on stdin, one JSON record on
stdout): the implementation's environment variables are applied, the inputs are
built, availability is checked, `setup`, parity, `verify`, peak memory and timing run
in that order. An exception, an out-of-memory error, a crash or a timeout becomes an
`error` record and the sweep goes on.

## Benchmarking LCE

The in-package op `lce` (`torch_op_bench/ops/lce.py`) measures linear cross entropy,
`cross_entropy(linear(hidden, weight), labels)`, under one contract for every
counterpart:

- inputs `hidden[N,H]` and `weight[V,H]` in the case dtype (bf16 by default), both
  `requires_grad`; int64 `labels[N]`, a fraction `ignore_frac` of them `-100`;
- no bias, no label smoothing; the output is the mean loss over non-ignored tokens;
- `fwd_bwd`: forward and `loss.backward()`, returns `(loss, grad_hidden, grad_weight)`;
- `fwd`: forward under `no_grad`, returns `(loss,)`;
- parity against an fp32 eager reference; work units `tokens = N`, `flops = 6 N H V`.

| implementation | what |
|---|---|
| `torch_eager` | `F.cross_entropy(F.linear(hidden, weight), labels)`; the baseline and the control |
| `torch_compiled` | the same under `torch.compile(fullgraph=True)` |
| `torch_lce` | `F.linear_cross_entropy(..., options=LinearCrossEntropyOptions())`, with the `torch._native` (CuTeDSL) override where eligible |
| `torch_lce_no_override` | the same with the override ops disabled: the eager accumulator |
| `liger` (env `liger`) | Liger-Kernel's `liger_fused_linear_cross_entropy(hidden, weight, labels, ignore_index=-100, reduction="mean", accum_dtype=torch.float32)`, `fwd_bwd` only |
| `liger_native_accum` (env `liger`) | the same with `accum_dtype=None`, Liger's default: the weight gradient accumulated in the parameter dtype; `fwd_bwd` only |
| `torch_eager_liger` (env `liger`) | `torch_eager` in the `liger` environment, the control |

The Liger entries run `fwd_bwd` only: Liger computes the gradients inside its forward
whenever the inputs require grad, so a `fwd` number would include the backward work.
They are `unavailable` without `liger_kernel` or without a CUDA device. Their `info`
records Liger's chunking for the case (`chunk_size`, `num_chunks`), the
`liger_kernel` version and `ce_impl: "triton"`, and `verify.liger_events` the Liger
kernels that ran (an `error` record if none did).

The default sweep (CUDA, bf16) varies one axis at a time around
(N, H, V) = (4096, 4096, 131072): N 1K–16K, H 1K–8K, V 32K–256K, `ignore_frac` 0
and 0.1:

```sh
torch-op-bench run lce --device cuda
```

On a 16 GiB GPU, run only the points that fit (the larger ones become `error`
records with an out-of-memory traceback):

```sh
torch-op-bench run lce --device cuda --case N=1024,V=32768 --case N=4096,V=65536
```

With Liger (see "Environments"):

```sh
scripts/setup-env.sh liger
torch-op-bench run lce --device cuda --env liger=envs/.prefix/liger/bin/python
```

What the records say:

- `verify.override_ran`: the profiler saw a `_native::...linear_cross_entropy...` event.
  `torch_lce` fails with an `error` record if the override was eligible (CUDA sm_80+,
  bf16 or fp16, options resolved to `compact` with fp32 accumulation) and did not
  run, and `torch_lce_no_override` fails if it ran. On CPU, below sm_80 or in fp32,
  `override_ran: false` is simply recorded. The error names why the override was not
  registered: the CuTeDSL runtime is missing, `TORCH_DISABLE_NATIVE_JIT=1` is set, or
  the `nvidia-cutlass-dsl` version is not known-good. `info.override_registration`
  records the same, and `env.packages` the versions of `nvidia-cutlass-dsl` and
  `apache-tvm-ffi`.
- `verify.native_events`: every `torch._native` event seen, not only LCE's. With the
  LCE override disabled, or in fp32, the backward still runs
  `_native::scatter_add__cutedsl_*`, so "no override" means no *LCE* override.
- `parity`: errors against the fp32 reference. The tolerances are 4× the bf16 error
  of `torch_eager` (see `ops/lce.py`). A failure is a finding for the report, not a
  failure of the run.
- `info`: the contract's constants, `compiled`, and for `torch_lce*` the resolved
  `options` and `lce_source_md5`. That is an md5 of torch's LCE source files, which
  the build hash misses when an editable install's Python files are edited.
- `memory.peak_bytes - memory.baseline_bytes` includes the cuBLAS workspace that the
  first matmul in a process allocates, about 30 MB.

## Results

```
results/<op>/<device-slug>/<UTC timestamp>-<host>.jsonl
```

One file per `run`, one record per line, append-only; `<device-slug>` is `cpu` or the
GPU name (e.g. `nvidia-h100-80gb-hbm3`). Files from several hosts merge by
concatenation; `torch_op_bench.records.read(directory)` reads a tree and
`records.merge` keeps the latest `ok` record per key. A record (`schema: 1`) has:

| field | content |
|---|---|
| `schema`, `timestamp` | record format version; UTC time the record was made |
| `op`, `op_ref` | op name; how it was resolved |
| `case`, `case_key`, `device_slug` | `{params, dtype, device}` (device normalised: `cuda` is `cuda:0`); its stable key; results subdirectory |
| `implementation`, `mode`, `env_name`, `python` | what ran, where |
| `settings` | `warmup`, `repeat`, `flush_l2`, `seed`, `timeout` |
| `status`, `reason`, `traceback`, `stderr` | `ok`, `unavailable` or `error`, and why |
| `times_ms` | all timing samples, in milliseconds (statistics are derived, `records.stats`) |
| `memory` | `peak_bytes`, `baseline_bytes` (inputs allocated); `null` on CPU |
| `parity` | `status` (`ok`, `fail`, `skipped`), `reason`, `metrics`, `tolerances`, `failed` |
| `verify`, `work`, `info` | what `verify` returned; work units; the implementation's info |
| `env`, `env_from`, `env_error` | provenance of the worker's interpreter (`worker`), or of the parent's when the worker produced no record (`runner`, without torch); why `env` is `null` if collecting it failed; `env.packages` maps each of the implementation's `packages` to its version (`null` if not installed), or, for a package installed from a VCS URL, to `{version, url, commit, requested}` from its `direct_url.json`; `env.package_root` the directory the worker imported torch-op-bench from |

The resume and merge key is (op, device slug, case key, implementation, mode).

## Tests

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu  # if no torch yet
pip install -e .[dev,plot]
ruff check .
pytest --cov=torch_op_bench --cov-report=term-missing
```

Tests marked `cuda` run when a CUDA device is present and are skipped otherwise. CI
runs the CPU tests with a coverage floor. The CUDA test of the Liger entries runs
once per Liger environment present (`envs/.prefix/liger`, `envs/.prefix/liger-git`,
or the interpreter in `TOB_LIGER_PYTHON`) and is skipped without one.

## Licence

BSD-3-Clause; see `LICENSE`.
