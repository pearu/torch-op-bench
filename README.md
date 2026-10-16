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
runs in.

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
- `dtypes`: dtype names (`"float32"`, `"bfloat16"`, ...).
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
  - `setup(case)`: run once in the worker before anything is measured;
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
| `env`, `env_from`, `env_error` | provenance of the worker's interpreter (`worker`), or of the parent's when the worker produced no record (`runner`, without torch); why `env` is `null` if collecting it failed |

The resume and merge key is (op, device slug, case key, implementation, mode).

## Tests

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu  # if no torch yet
pip install -e .[dev,plot]
ruff check .
pytest --cov=torch_op_bench --cov-report=term-missing
```

Tests marked `cuda` run when a CUDA device is present and are skipped otherwise. CI
runs the CPU tests with a coverage floor.

## Licence

BSD-3-Clause; see `LICENSE`.
