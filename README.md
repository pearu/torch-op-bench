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
pip install -e .[dev,plot]  # plus tests, lint and matplotlib for plots
```

Install torch-op-bench from the same checkout into every environment a benchmark
runs in. When `torch-op-bench run` itself runs from a checkout, its workers import
torch-op-bench from that checkout in every environment, whatever the working
directory, so the parent and the workers always run the same code. A counterpart
can also be installed into the torch environment itself, e.g.
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

Rerunning the script on an existing prefix runs `env update`, which adds and changes
packages but does not remove one dropped from the definition. After removing a
package from a definition, recreate the prefix instead:
`rm -rf envs/.prefix/NAME && scripts/setup-env.sh NAME`.

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

`--matmul-reduction default|strict` (default `default`) chooses the matmul setting
every worker runs under. `default` touches nothing: torch's defaults, under which
bf16 and fp16 GEMMs may reduce in reduced precision, and the sweeps run under them,
for a fair comparison. `strict` makes each worker set, before the implementation's
setup, `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` and
`allow_fp16_reduced_precision_reduction` to `(False, True)` (no reduced-precision
reduction, split-K still allowed) on builds whose flags are pairs, and to `False`
on older ones; `(False, False)` is not offered, since it needs cuBLASLt. The setting
is in `settings.matmul_reduction`, and what was in effect is read back into
`env.torch.matmul`. Since the resume key does not hold the setting, a strict run
goes to a results directory of its own: `run --matmul-reduction strict` refuses
`results/` and requires `--results`, by convention `results-strict/` beside
`results/`. The strict run is the counterfactual, one per GPU at the default point,
for every entry in both modes; the sweeps are never run strict:

```sh
torch-op-bench run lce --device cuda --env liger=PYTHON --env torchtitan=PYTHON \
    --case N=4096 --matmul-reduction strict --results results-strict
torch-op-bench report results --strict results-strict --out report.md
```

## Add an op

An op is an `OpSpec` (see `torch_op_bench/spec.py`), addressed either by a short name
registered in `torch_op_bench/ops/__init__.py` or by a dotted path `module:OBJECT`.
The parent process (`run`, `list`) imports the op's module to enumerate cases, so the
module should import torch and any counterpart lazily, inside its callables: then the
parent needs neither, and each worker imports torch only after its environment
variables are set.

- `description`: what the op computes, in a sentence, for the report's summary.
- `axes`: a list of `Axis(name, values, kind="int"|"str", description="")` (names and
  `str` values must not contain `,`, `=` or `|`, which structure case keys; the
  description says in one line what the axis is, for the report's definitions);
  `defaults` gives the point a sweep varies one axis around (an axis missing from
  it defaults to its first value).
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
    facts recorded as is (e.g. chunk size, accumulation dtype);
  - `description`: what it is, in one line, for the report's summary (a control's
    and a variant's are made from their original's name).
- `reference`: an `Implementation` at higher precision, run once per case on the same
  inputs; `compare(outputs, ref_outputs, case)` returns named error metrics, checked
  against `tolerances`; `metric_definitions` says what each metric is (every
  `tolerances` key needs one, else the spec is refused) and `tolerance_basis` how
  the tolerances were set, both for the report's definitions;
  `reference_fits(case)` returns `True` or a reason to skip the
  parity check (a memory guard). The worker runs the implementation first and keeps
  its outputs in host memory while the reference runs, so the reference has the
  device's free memory to itself.
- `work(case, mode)`: the work of a case in a mode (e.g. `{"tokens": N, "flops":
  ...}`), recorded with each record (a `work` that takes the case only is refused);
  `work_units`: the units the report shows as throughput, each a `WorkUnit(key,
  label, definition, rate_label, scale)`: `key` one that `work` returns at the
  default point in every mode (else the spec is refused), the column headed
  `rate_label` and holding `work[key]` per second of the median over `scale` (LCE:
  tokens/s, and TFLOP/s with `scale` 1e12). An op that declares no unit gets no
  throughput column.
- `parity_notes`: a list of `ParityNote(implementation, metrics, applies, text)`, each
  a known cause of a parity failure outside the tool: `metrics` the metrics it
  explains (`None`: any), `applies(case, mode, metrics, torch_version)` whether it
  holds for a record (its case, mode, parity metrics and the worker's torch
  version), and `text` the cause in a sentence or two, with its source (file and
  line, the torch version it was verified on) and a reference (an issue or PR URL).
  The reports mark a failure a note explains (see "Reports"); the note lives with the
  op, not in the records, so a cause found after a run needs no rerun.
- `notes`: a list of strings, free text the op's author vouches for, each dated and
  sourced (and naming the card or the version it was seen on), on what the records
  cannot show by themselves; the report prints them verbatim in its "Notes" section.

Each (case, implementation, mode) runs in a fresh worker
(`python -m torch_op_bench.worker`, a JSON payload on stdin, one JSON record on
stdout): the implementation's environment variables are applied, the inputs are
built, availability is checked, `setup`, parity, `verify`, peak memory and timing run
in that order. Before the inputs are built, a worker on a CUDA device primes cuBLAS
(`worker.prime_device`: a 64×64 matmul in the case's dtype, with a backward), so the
forward's and autograd's cuBLAS workspaces each sit in a segment of their own, as in
a training process where earlier layers created them; left to the op's first call, a
workspace can land in a freed multi-GiB temporary and pin it, and the later calls run
out of memory where the first fit (`torchtitan_ce` fwd_bwd at V=262144 on a 16 GB
card). A case on the CPU never touches the GPU: its worker creates no CUDA context
(the runner hides the GPUs from it, `CUDA_VISIBLE_DEVICES=""`, as torch.compile's
tracing would otherwise create one), and its record's `env.torch` lists no GPUs and
no cuDNN version, only the build's CUDA version (`env.torch.cuda`), so a CPU record
does not depend on what else runs on the GPU. A point that runs out of the device's
memory in any of these phases becomes an `unavailable` record (see `status` under
"Results"); an exception, a crash or a timeout becomes an `error` record. Either way
the sweep goes on.

## Benchmarking LCE

The in-package op `lce` (`torch_op_bench/ops/lce.py`) measures linear cross entropy,
`cross_entropy(linear(hidden, weight), labels)`, under one contract for every
counterpart:

- inputs `hidden[N,H]` and `weight[V,H]` in the case dtype (bf16 by default), both
  `requires_grad`; int64 `labels[N]`, a fraction `ignore_frac` of them `-100`;
- no bias, no label smoothing; the output is the mean loss over non-ignored tokens;
- `fwd_bwd`: forward and `loss.backward()`, returns `(loss, grad_hidden, grad_weight)`;
- `fwd`: forward under `no_grad`, returns `(loss,)`;
- parity against an fp32 eager reference, computed in chunks of 1024 rows (the same
  quantities: only the order of the sums differs), so it needs one chunk's logits,
  not N rows'; work units `tokens = N`, and `flops` the GEMMs the mode runs:
  `2 N H V` in `fwd` (the logits), `6 N H V` in `fwd_bwd` (and the two backward
  products), an implementation's recomputation not counted.

| implementation | what |
|---|---|
| `torch_eager` | `F.cross_entropy(F.linear(hidden, weight), labels)`; the baseline and the control |
| `torch_compiled` | the same under `torch.compile(fullgraph=True)` |
| `torch_lce` | `F.linear_cross_entropy(..., options=LinearCrossEntropyOptions())`, with the `torch._native` (CuTeDSL) override where eligible; `unavailable` on a torch without the override (`torch/_native/ops/linear_cross_entropy`): every release up to 2.14.1, the newest on 2026-10-17; nightlies only |
| `torch_lce_no_override` | the same with the override ops disabled: the eager accumulator; available wherever `F.linear_cross_entropy` is (2.13 on), and `info.override_present` says whether there was an override to disable |
| `liger` (env `liger`) | Liger-Kernel's `liger_fused_linear_cross_entropy(hidden, weight, labels, ignore_index=-100, reduction="mean", accum_dtype=torch.float32)`, `fwd_bwd` only |
| `liger_native_accum` (env `liger`) | the same with `accum_dtype=None`, Liger's default: the weight gradient accumulated in the parameter dtype; `fwd_bwd` only |
| `torch_eager_liger` (env `liger`) | `torch_eager` in the `liger` environment, the control |
| `torchtitan_chunked` (env `torchtitan`) | torchtitan's `ChunkedLossWrapper` (8 chunks) with its default inner `CrossEntropyLoss`, over an lm_head computing `F.linear(x, weight)`; eager; `fwd_bwd` only |
| `torchtitan_chunked_compiled` (env `torchtitan`) | the same with the inner loss under `torch.compile`, the lm_head not compiled, as torchtitan does it; `fwd_bwd` only |
| `torchtitan_ce` (env `torchtitan`) | torchtitan's default loss, `CrossEntropyLoss`, on the full logits `F.linear(hidden, weight)`; eager |
| `torch_eager_torchtitan` (env `torchtitan`) | `torch_eager` in the `torchtitan` environment, the control |

Six of these entries chunk with a parameter of their own, and each has two
**variants** (`Implementation.with_params`): the same entry with another value of
that parameter, named `<entry>[<parameter>=<value>]` (`lce.VARIANTS`):

| entries | parameter | default | variants |
|---|---|---|---|
| `torch_lce`, `torch_lce_no_override` | `batch_chunk_size`, with `chunking_method=None` | the heuristic's (128 rows at the default point) | 256, 1024 |
| `liger`, `liger_native_accum` | `chunk_mem_const` | 1 | 4, 8 |
| `torchtitan_chunked`, `torchtitan_chunked_compiled` | `num_chunks` | 8 | 4, 32 |

They exist for one comparison, the time at equal peak memory (the report's "Time at
equal peak"): with the default, each entry has three (peak, time) points, from which
its time within a peak budget is read, at budgets that are the same for every entry.
They are not a tuning study: no value is ranked, none is named an optimum, and
whether a default is optimal on a machine is a question for another study. A
variant runs only when `--impl` names it, at the default point only (see "Remote
runs"), never along a sweep; `run` without `--impl` leaves the variants out. A
variant has its entry's modes, environment and availability; its `info` carries
`base` (the entry) and `params` (the value as given), and what its setup resolved
from it, as the entry's does: `options.batch_chunk_size` and `num_chunks` for torch
LCE (a `batch_chunk_size` above N is capped at N by torch; `chunking_method=None`
with no `batch_chunk_size` would be one chunk of N rows, which no entry uses),
`chunk_size` and `num_chunks` for Liger, `num_chunks` and `chunk_tokens` for
torchtitan. Quote a variant's name on a shell command line: `[` is a glob character.

The Liger entries run `fwd_bwd` only: Liger computes the gradients inside its forward
whenever the inputs require grad, so a `fwd` number would include the backward work.
They are `unavailable` without `liger_kernel` or without a CUDA device. Their `info`
records Liger's chunking for the case (`chunk_size`, `num_chunks`), the
`liger_kernel` version and `ce_impl: "triton"`, and `verify.liger_events` the Liger
kernels that ran (an `error` record if none did).

The torchtitan entries need no GPU and no process group: without a device mesh,
torchtitan's losses are plain torch (`F.cross_entropy(..., reduction="sum")` on
`pred.float()`), divided by the number of non-ignored tokens. The entries pass the
count positionally, as the losses' third argument: 0.3.0 names it
`global_valid_tokens`, `main` `global_loss_token_counts`. The chunked entries run
`fwd_bwd` only, for the same reason as Liger's: the wrapper runs the chunk backwards
whenever the hidden states require grad. They need `N` divisible by 8 (every sweep
value is; otherwise an `error` record). Their `info` records `num_chunks`,
`chunk_tokens` (`N / 8`) and two dtypes:

- `logits_dtype: "float32"`: torchtitan upcasts the logits with `.float()` before the
  cross-entropy, per chunk for the wrapper (`chunk_tokens × V × 4` bytes) and in full
  for `torchtitan_ce`;
- `grad_accum_dtype: "float32"`: the wrapper's `GradAccumulator` buffer, which
  collects `grad_hidden` chunk by chunk. `grad_weight` is not in it: the chunk
  backwards accumulate into `weight.grad` in the weight's dtype.

`info.torchtitan_api` says which torchtitan API ran: `compile_config` (0.3.0:
`[1, N, ...]` inputs, the loss compiled through a `CompileConfig`) or `local_compile`
(`main`: `[N, ...]` inputs, compiled through `apply_local_compile(["loss"])`).
`verify.compiled_events` of the compiled entry are the events showing the compiled
loss ran (`CompiledFunction`, or `triton_` kernels on CUDA); an `error` record if
there are none. `main` reports a stale version (`0.2.2` on 2026-10-16), so for
`envs/torchtitan-git.yml` it is the commit in `env.packages["torchtitan"]` that names
it.

The default sweep (CUDA, bf16) varies one axis at a time around
(N, H, V) = (4096, 4096, 131072): N 1K–16K, H 1K–8K, V 32K–256K, `ignore_frac` 0
and 0.1:

```sh
torch-op-bench run lce --device cuda
```

On a 16 GiB GPU, run only the points that fit (the larger ones become `unavailable`
records, "out of memory on" the device):

```sh
torch-op-bench run lce --device cuda --case N=1024,V=32768 --case N=4096,V=65536
```

With Liger (see "Environments"):

```sh
scripts/setup-env.sh liger
torch-op-bench run lce --device cuda --env liger=envs/.prefix/liger/bin/python
```

With torchtitan:

```sh
scripts/setup-env.sh torchtitan
torch-op-bench run lce --device cuda --env torchtitan=envs/.prefix/torchtitan/bin/python
```

What the records say:

- `verify.override_ran`: the profiler saw a `_native::...linear_cross_entropy...`
  event. `torch_lce` fails with an `error` record if the override was eligible (CUDA
  sm_80+, bf16 or fp16, options resolved to `compact` with fp32 accumulation) and did
  not run, and `torch_lce_no_override` fails if it ran. On CPU or in fp32,
  `override_ran: false` is simply recorded. On a CUDA device below sm_80 both
  `torch_lce` entries are `unavailable` ("torch's linear_cross_entropy CuTeDSL path
  needs sm_80 or newer; Tesla T4 is sm_75"): there the backward's `torch._native`
  dispatch fails. On a torch without the override (2.14.1 and older), `torch_lce` is
  `unavailable` with that reason rather than run: it would measure what
  `torch_lce_no_override` measures, under another name. The error names why a build
  that has the override did not register it: the CuTeDSL runtime is missing,
  `TORCH_DISABLE_NATIVE_JIT=1` is set, or the `nvidia-cutlass-dsl` version is not
  known-good. `info.override_registration` records the same, after `present` (whether
  the build has the override), and `env.packages` the versions of `nvidia-cutlass-dsl`
  and `apache-tvm-ffi`.
- `verify.override_kernel_ran` and `verify.override_kernels`: whether a CuTeDSL
  kernel (a name starting `kernel_cutlass_`) ran under the override's dispatch
  event, and which. The dispatch is not the kernel: in fwd (`torch.no_grad`) the
  override dispatches but forms the loss from eager ops, with no CuTeDSL kernel, so
  `torch_lce`'s fwd time is the eager accumulator's (`override_ran: true`,
  `override_kernel_ran: false`); that is recorded, not an error. In fwd_bwd an
  eligible override that dispatches without its kernel fails with an `error`
  record. Under the dispatch only: other `torch._native` overrides launch CuTeDSL
  kernels too (`_native::scatter_add__cutedsl_*` in the backward), and those are
  not LCE's. `torch_lce_no_override` records `false` and `[]`, and fails if a
  kernel ran; on CPU both entries record `false` and `[]`.
- `verify.native_events`, on every entry (baselines, controls and counterparts
  alike): every `torch._native` event seen, not only LCE's. With the LCE override
  disabled, or in fp32, the backward still runs `_native::scatter_add__cutedsl_*`, so
  "no override" means no *LCE* override. `[]` means that none ran, or that the torch
  of that environment has no `torch._native`.
- `parity`: errors against the fp32 reference. The tolerances are 4× the bf16 error
  of `torch_eager` (see `ops/lce.py`). A failure is a finding for the report, not a
  failure of the run. One cause is known and noted (`lce.PARITY_NOTES`):
  `torch_lce_no_override` in bf16 on CUDA fails `loss_rel_err` and
  `grad_hidden_rel_l2` at some shapes where `torch_lce` passes, because torch's eager
  chunked path keeps the running loss in bf16
  (`torch/nn/modules/linear_cross_entropy.py:534` in torch 2.16.0.dev20261017) and
  the override keeps it in fp32; reported as pytorch#201659, fix pytorch#202050.
  Once the fix merges, `lce.BF16_LOSS_FIX_DATE` takes the day it reached torch main,
  and the note no longer applies to a nightly from that day on. Two more facts the
  records cannot show by themselves are the op's `notes` (`lce.NOTES`, printed in
  the report's "Notes"): the rise of `grad_hidden_rel_l2` at N=8192 under the
  default matmul setting, and that a "Reduction setting" time ratio within the
  environment check's band is run-to-run variation.
- `info`: the contract's constants, `compiled`, and for `torch_lce*` the resolved
  `options` and `lce_source_md5`. That is an md5 of torch's LCE source files, which
  the build hash misses when an editable install's Python files are edited.
- `info.op_decisions`, for `torch_lce*`: what torch's LCE decided for the case, read
  from what the build exposes, nothing vendored, `source` naming where. With PR
  pytorch#195707 (as reviewed), the values of `_mm_fp32_out_available(dtype,
  device)` and `_mm_accumulates_in_fp32(dtype, device)` under the matmul setting in
  effect; before it, the fields of `_ChunkContext.build` for the case that do not
  depend on which gradients are asked (`is_cuda`, `use_acc_dtype`,
  `forward_uses_cuda_out_dtype`, the dtypes, ...), built on stride-0 views of the
  case's shapes, so nothing of its size is allocated; `{"unavailable": reason}`
  when neither is there.
- `memory.baseline_bytes` includes the cuBLAS workspaces where the build counts them
  as allocated (torch 2.14.1: 64 MiB, the forward's and autograd's thread's; the
  nightly reserves a 32 MiB segment for them and does not count it); the worker
  primes them before the inputs exist (see the worker paragraph above), so
  `memory.peak_bytes - memory.baseline_bytes` no longer includes them.

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
| `status`, `reason`, `traceback`, `stderr` | `ok`, `unavailable` or `error`, and why; `unavailable` has three sources, the implementation's `available()` (a package or an API missing) and its `setup` raising `spec.Unavailable` (the case cannot run here, e.g. a device the op does not support), both before anything is timed, and the point not fitting in the device's memory, wherever it runs out (`setup`, `parity`, `verify`, `memory`, `warmup`, `timing`): its reason's first line names the device ("out of memory on NVIDIA GeForce RTX 5060 Ti (15.6 GiB)"), its second the figures and the phase ("tried to allocate 4.00 GiB, 2.22 GiB free; 9.19 GiB allocated, 3.97 GiB reserved but unallocated; phase: timing"). Whether a point at the edge fits depends on the allocator's fragmentation in that build: such a record says it did not fit on this device in this build, and a rerun retries it. `unavailable` has no traceback; `error` is for failures nobody foresaw |
| `times_ms` | all timing samples, in milliseconds (statistics are derived, `records.stats`) |
| `memory` | `peak_bytes`, `baseline_bytes` (what is allocated when the call starts: the inputs, and the cuBLAS workspaces where the build counts them); `null` on CPU |
| `parity` | `status` (`ok`, `fail`, `skipped`), `reason`, `metrics`, `tolerances`, `failed`; `skipped` when the reference does not fit (`reference_fits`) or, against that estimate, runs out of memory ("reference out of memory: …", the record staying `ok`) |
| `verify`, `work`, `info` | what `verify` returned; the work of the case in the record's mode, in the op's units (`work(case, mode)`; LCE counts 2·N·H·V FLOP in `fwd` and 6·N·H·V in `fwd_bwd`); the implementation's info |
| `env`, `env_from`, `env_error` | provenance of the worker's interpreter (`worker`), or of the parent's when the worker produced no record (`runner`, without torch); why `env` is `null` if collecting it failed; `env.packages` maps each of the implementation's `packages` to its version (`null` if not installed), or, for a package installed from a VCS URL, to `{version, url, commit, requested}` from its `direct_url.json`; `env.package_root` the directory the worker imported torch-op-bench from; `env.torch.matmul` the process-wide settings that choose how matmuls round (`allow_tf32`, `allow_bf16_reduced_precision_reduction`, `allow_fp16_reduced_precision_reduction`, and, on builds whose reduced-precision flags are pairs `(reduced_precision, split_k)`, `allow_bf16_reduced_precision_reduction_split_k` and `allow_fp16_reduced_precision_reduction_split_k` (`null` otherwise), `fp32_precision`, `float32_matmul_precision`), as the worker had them after the run (`run --matmul-reduction strict` sets the reduced-precision ones; its `_split_k` fields stay `true`, since the second element of a pair only qualifies an allowed reduced-precision reduction: torch refuses `(True, False)` with "allow_splitk=False is not supported when reduced precision reductions are enabled", and `(False, True)` is no reduced-precision reduction at all): they change parity (bf16 reduced-precision reduction, on by default, doubles the `grad_hidden` error of chunked LCE at 128-row chunks) |

The resume and merge key is (op, device slug, case key, implementation, mode).

## Reports

Two commands read a results directory, never an op's code: they work for any op,
and `report` needs nothing but this package. Each takes `--op` and `--device-slug`
(required only when the directory holds several), `--dtype` and `--mode` (default:
each in turn), `--baseline` (default `torch_eager`), and `--point AXIS=VALUE,...`.

The point both centre on is the op's `defaults`. When the op cannot be imported, or
no `ok` record is at its defaults (a small run of a case or two elsewhere), it is
inferred: the most common value of each axis over the `ok` records, or the most
common case when no record is at that combination. The report names the source:
"The default point is inferred (the spec's default point N=4096, H=4096, V=131072,
ignore_frac=0 has no records)." `--point` sets it instead, for the axes it names,
the others keeping theirs, e.g. `--point V=32768` to centre the tables and figures
on the V=32768 sweep.

```sh
torch-op-bench report results --out report.md
torch-op-bench plot results --out figures
torch-op-bench plot results --out figures \
    --overlay other-results --label-a 2.16 --label-b 2.14
```

`report` writes one markdown document (to stdout without `--out`), starting with the
command that made it, and any warning the merge raised (records from different
torch-op-bench commits). Its sections:

1. **Summary**, for a reader who is not an engineer: *What was compared* (the op's
   `description`, each entry in the op's order with its `description`, the
   baseline, the modes in words, the sweep's axes with their descriptions and
   values, the number of points and records, the strict run and the variants);
   *On what* (the GPU and its memory, the hosts, the torch build of each
   environment, the counterparts' versions, the dates); *Findings*; a *Headline
   table* (at the default point, every entry and mode in the op's order: time,
   peak and time × peak against the baseline's, and the equal-peak cell at the 1×
   budget, the same numbers as sections 4 and 6); and *Caveats* (what was not
   measured, by cause, in plain words; where parity was not checked, by reason;
   the parity failures and their notes, "N parity failures, M explained"; the
   run-to-run band). The findings are generated by default, from fixed templates
   in one place (`report.py`): per entry and mode at the default point, its time,
   memory and time × peak against the baseline's and its parity; the controls
   against the environment check's bands; per chunked entry, the point within
   each memory budget; per axis, the baseline's median over it and each entry's
   ratios, within ±10 % of their default-point value or their range; and the
   repeatability across environments. Every sentence is a value, or a ratio to
   the baseline or to a budget, the entries in the op's order; no sentence ranks
   entries or compares two counterparts, and the words fastest, slowest, best,
   worst, wins, optimum and beats never appear (`report.FORBIDDEN`, tested).
   `--summary FINDINGS.md` prints an author's findings in their place, the file's
   paragraphs verbatim under "Findings (written by the author of this result
   set; …)" and a line naming the file and the records' torch-op-bench commit;
   the judgement there is the author's. When the op does not import, the summary
   comes from the records alone and says what it cannot show.
2. **How to read this report**: every column and section defined, in two layers.
   The tool's definitions, the same for every op (`report.DEFINITIONS`, filled in
   with the timing settings as recorded, the baseline and the bands; a test checks
   that every header the report emits has one); then the op's own, read through
   its spec: its units of work (`work_units`), its parity metrics with their
   tolerances and how those were set (`metric_definitions`, `tolerances`,
   `tolerance_basis`), and its axes with their values and default
   (`Axis.description`). When the op does not import, the section says its
   definitions are not shown.
3. **Provenance**: device, hosts, the time span of the records and their number in
   each directory read, named as given (232 records in `results`, 18
   in `results-strict`), and one row per
   environment build: torch and CUDA, Python, the packages (a version, or the first
   12 characters of the commit of a git install), the torch-op-bench commit, and
   the matmul settings. It says so when the commits differ, or when one environment
   holds several builds.
4. **At the default point** (see above; under it, each axis' value with its
   description), per mode and dtype, two tables. The first has
   one row per implementation, the baseline first and each control right after its
   original, with its environment, median and IQR (in one cell, "median (IQR)"; the
   status instead when it is not `ok`), time against the baseline, the throughput
   in each of the op's units (`work_units`: LCE's tokens/s and TFLOP/s; when the op
   does not import, `<key>/s` for each key of the records' `work`, unscaled),
   peak memory above baseline and against the baseline's, **time × peak** (median ms ×
   peak above baseline in MiB, in ms·MiB) and against the baseline's, parity and its
   metrics (`fail: a¹, b`: each failed metric with the mark of the parity note of
   the op that explains it there, `b` unexplained; the note numbered under the
   table: one footnote per distinct note per table, the same in the sweeps' parity
   tables), and `override`: `kernel` (the override dispatched and its kernel ran),
   `dispatch only` (it dispatched, and ran no kernel of its own), or `no` (`dispatch`
   for a record older than `override_kernel_ran`; `—` for an entry that records
   neither), from `verify.override_ran` and `verify.override_kernel_ran`. When an
   entry is `dispatch only`, a sentence under the table says its time is not the
   override kernel's. The second, "Entry details", has one row per implementation in
   the same order and one cell: the `key=value` pairs of the `info` fields that differ
   between records (e.g. `torchtitan_api`, `compile`, `op_decisions.is_cuda`), one
   per line, keys sorted, those the entry lacks left out, so a difference cannot hide in `info`. No
   table of the report is wider than 16 columns.
5. **Sweeps**: for each axis varied around the default point, four tables (median,
   peak, time × peak with its ratio to the baseline's in parentheses, parity with
   the failing metrics) with one column per implementation. A cell is `—` when there
   is no record, or the status when it is not `ok`. For a mode and dtype in which no
   axis varies around the point (one case, or cases that differ in two axes at
   once), the default-point table is repeated at every case present instead, one
   table per case, as `plot` says when it draws nothing for that mode and dtype.
6. **Time at equal peak**, only when the results hold variants: per mode and dtype,
   at the default point, one row per entry with variants and one column per peak
   budget, the baseline's peak above baseline there at 1×, 0.75× and 0.5× (in MiB
   in the header), the same budgets for every entry, so the table reads the same on
   any GPU. A cell is the median time of the entry's default or one of its variants,
   the one whose peak above baseline is the largest within the budget (of two at that
   peak, the faster), with what produced it in small print: the variant's parameter,
   or `default` with the entry's own value when its `info` records it; `—` when none
   fits (measured; no point within the budget), with the entry's lowest measured peak
   over its default and variants in small print (`— lowest 3104.0 MiB`), so the cell
   shows how far it misses; ", parity fail" when that record fails parity. The last
   column is the default's own time and peak. A variant that is not `ok` is in no
   cell, and a sentence under the table names it. When every budget of a row picks
   the same point, a sentence under the table names the row ("every budget admits
   the same point"): none of the entry's points lies between the smallest budget and
   the largest, so the budgets do not separate them there. Variants are in no other table but "Parity
   failures" (a note of an entry covers its variants) and "Not measured".
7. **Reduction setting**, only with `--strict RESULTS_STRICT` (the results of a
   `run --matmul-reduction strict`): first a check that every record of `RESULTS`
   has the default setting and every record of `RESULTS_STRICT` the strict one
   (read from `env.torch.matmul`; a record the runner wrote has none and is not
   checked), else a warning naming the records that do not, and the section still
   renders. Then per mode and dtype, at the default point, one row per entry: the
   median of each and the time ratio strict / default; for each parity metric both
   values (in one cell, "default, strict") and the ratio default / strict; "parity moves", yes with the metrics
   named when a ratio is outside [0.9, 1.1] (the environment check's band); and,
   for the torch LCE entries, `info.op_decisions`, the values in one cell. An entry
   absent from one directory, or not `ok` in one, says "not measured" and why.
   Below each table a computed reading: "Under the defaults, `<entries>` have the
   strict parity (fp32 accumulation or no small-M GEMMs); `<entries>` do not."
   and, for each torch LCE entry, whether its default parity equals its strict
   parity: the check that, once PR pytorch#195707 lands as reviewed, torch LCE
   accumulates in fp32 under the defaults while Liger and torchtitan do not.
8. **Parity failures**: every record whose parity is `fail`, grouped by entry, then
   by "explained: <note>" (the note's text once per entry) when one of the op's
   `parity_notes` applies there, else "**unexplained**", then by metric, with the
   records under each; then "N parity failures, M explained", a record counting
   as explained when every metric it fails is. When the op cannot be imported,
   there are no notes and every failure is unexplained, and a last sentence says so
   ("The op `<op_ref>` does not import here: …").
9. **Environment check**: each control (`torch_eager_<env>`) against its original at
   every point both have, as the time ratio (control / original), the peak
   difference, and the ratio of each parity metric. A time ratio outside [0.95,
   1.05], a difference beyond 64 MiB or a parity ratio outside [0.9, 1.1] is named
   in a sentence: there the environment, not the counterpart, makes a difference,
   and the counterpart must be compared with its own environment's control. Another
   torch build can move the error of the same code (by 23 % in `grad_hidden` at one
   shape between torch 2.14 and 2.16), so a parity comparison across environments
   needs this too. A parity metric on one side only is reported as such. A pair
   where only one side is `ok` (a control out of memory where its original fits) is
   a row too, flagged `status`, with the statuses (control / original) in the time
   ratio's place and a sentence naming the cause: the environment makes that
   difference. A pair where neither is `ok` has no row; the paragraph above the
   table counts the pairs compared, those with one side and those with neither side
   measured, and "Every control is within the limits" is said only when no row is
   flagged.
10. **Not measured**: every `error` and `unavailable` record, grouped by cause, with
   the number of records: for an `error`, the message up to its first number (a
   token of its own, such as `4.00`, `cuda:0` or `4096x1024`; digits inside a word,
   as in `mat1`, `sm_80` or `2D`, are part of the cause), less the unfinished
   sentence there (one group for every CUDA out-of-memory error, whatever the
   sizes); each row then names the entry, mode and case, and what the cause left
   out (for out of memory, the size tried and the free memory). For `unavailable`,
   whose reasons are written to be stable, the first line of the reason, and the
   second line in each row (for a point out of memory, one group per device, each
   row with its figures and phase).
11. **Notes**, only when the op has `notes` and imports here: each note verbatim,
   under a sentence saying they are the op author's, not computed by the report.

Numbers have 3 significant digits and memory is in MiB with one decimal.

Time × peak is there because the comparison that matters for a fused kernel is
elapsed time *and* memory together: a counterpart that is faster by holding more,
or smaller by taking longer, does not win on it unless the product drops (the
user's comparison; the other one is the time at equal peak memory, section 6). The
environment check still compares time and peak separately.

`plot` (`pip install -e .[plot]`; matplotlib, with the Agg backend, so no display
is needed) writes PNGs at 150 dpi, per mode (and per dtype, as `-<dtype>`, when the
results hold several):

- `grid-<mode>.png`: rows are metrics (by default median time, tokens/s, peak above
  baseline and one parity error, the op's `grad_hidden` or `loss` error when it has
  one), columns are the swept axes (int axes on a log scale);
- `ratio-<mode>.png`: speedup (baseline time / time), memory ratio and time × peak
  ratio (`time_x_peak_ratio`, impl / baseline) against the baseline, with the
  baseline as the line at 1. A throughput ratio is drawn only
  when some implementation's work differs from the baseline's at a point; otherwise
  it is the speedup again;
- `overlay-<mode>.png`, with `--overlay`: the grid of both directories on the same
  axes, the first solid and the second dashed, e.g. one device under two torch
  builds; its footer counts what each set leaves out, and gives each set's
  provenance, under the labels;
- `reduction-<mode>.png`, with `--strict RESULTS_STRICT`: per entry at the default
  point a pair of bars, default (filled) and strict (hatched), for the median time
  and for the mode's parity metric (`grad_hidden` or `loss` error), as the report's
  "Reduction setting" compares them, each on a linear axis from zero, so the bars'
  lengths compare as the values do. Each bar carries its value (3 significant
  digits), so a bar too short to see still reads. It is drawn also when nothing is
  swept.
- `equal-peak-<mode>.png`, when the results hold variants: time against peak above
  baseline at the default point, one line per entry with variants through its
  default and variants (by peak), the baseline as a reference point, and the peak
  budgets of the report's "Time at equal peak" as dotted vertical lines; no point is
  labelled with its parameter (the CSV names each point's entry). Each entry's default
  point is ringed in black ("default: edged" in the legend), and the CSV's extra
  column `default` says which point it is (empty for the baseline and the budgets). Two entries whose
  points coincide one by one, in order of peak, are stacked and named as below. The
  footer counts what is left out and the parity failures, as every footer does.
  With `--impl`, the entries named, or one of whose variants is named.

The variants are drawn in no other figure unless `--impl` names them (and never in
the reduction figure).

When no axis varies around the point, `plot` draws nothing and says so on stderr,
once per mode and dtype, and exits 0; the report's per-case tables are then the
result.

Every figure draws the baseline black, dashed and thick, and a control dotted and
thin in its original's colour, with one marker per environment, so two controls of
one original stay apart (`plot.STYLE`, `plot.COLORS`, `plot.CONTROL_MARKERS`).
Colours follow the op's implementation order (sorted names when the op cannot be
imported), so an implementation has the same colour in every figure and every
invocation. A point whose parity failed has a hollow marker, so a fast but wrong
number does not read as a win: in a parity panel when that panel's metric failed, in
every other panel when any metric did. A missing point (`error`, `unavailable`, no
parity) is a gap, and the footer counts what was left out and why, and the parity
failures with how many of them the op's parity notes explain ("Parity failures: 4
(4 explained), drawn hollow."); a failure stays hollow when explained. Two series of a
panel coincide when, at every point, they lie within half a marker of each other as
drawn (a control and its own original excepted, and two controls of one original:
a control is styled apart, and the environment check compares each with the
original). Series
that coincide, directly or through another, are stacked in drawing order: each one
above the first is drawn on top, dashed, at half its line width and marker size, and
from the third on also shifted sideways, alternately left and right, by half a
marker per step, so every colour shows. The footer names each such series and the
first one it lies on, with its panels ("Coincident, …").
A linear y axis whose values exceed 10⁵, or stay below 10⁻⁴, is drawn in a unit
scaled by a power of 10³, which its label names ("time × peak (10⁶ ms·MiB)"), the
same in every panel of a row; tick labels are plain numbers, with no offset or "1e7"
above the panel. The footer also gives the provenance (each environment's torch and
torch-op-bench commit, and the hosts); a figure without it is not a result.
`--metric`, `--sweep` and `--impl` restrict what is drawn (the baseline is always
drawn).

Next to every PNG, a CSV of the same name holds exactly the plotted points, one row
per point: the metric, axis, x, implementation, environment, role, y, parity status,
whether parity failed (on any metric, and, in a parity panel, on the panel's own),
the point, and the work (with the baseline's, in a ratio figure). The write-up
quotes those numbers, not values read off a figure.

`torch_op_bench.analysis` turns records into flat rows (`load`, `to_rows`,
`default_point`, `sweeps`) for anything else; it needs no pandas.

## Remote runs

A benchmark runs on a fresh host that has `ssh`, `git`, `curl`, `rsync`, an NVIDIA
driver and internet access, from `scripts/remote-run.sh` on your machine. Each of its
subcommands is one ssh session or one rsync, and all of them work in one scratch
directory on the host (`--dir`, absolute or `~/…`, default
`~/torch_op_bench-scratch`), outside which nothing is written but temporary files
(in the session's `TMPDIR`, or `REMOTE_DIR/tmp` when it has none). For a host
behind agent-sandbox's sandboxed login, see the subsection below. The five steps
of a run:

```sh
scripts/remote-run.sh HOST push        # 1. the tracked files and .git, to HOST
scripts/remote-run.sh HOST bootstrap   #    Miniforge, unless conda is there
scripts/remote-run.sh HOST setup --envs liger,torchtitan   # 2. environments
                                       # 3. setup shows `env`, nvidia-smi, `list lce`
scripts/remote-run.sh HOST run --envs liger,torchtitan -- --case N=4096,V=65536
                                       #    then the variants (below)
scripts/remote-run.sh HOST status      # 4. while it runs: alive?, log, records
scripts/remote-run.sh HOST wait        #    or follow it to the end
scripts/remote-run.sh HOST fetch       # 5. results back into results/
```

- **`push`** sends the tracked files (`git ls-files`) and the repository, and
  nothing else: untracked and ignored files never leave your machine, and a file
  the previous push tracked and HEAD does not is removed on the host, also one
  that became ignored (`git clean` keeps the host's own ignored files,
  `envs/.prefix/` and `results/`). It refuses a checkout
  with uncommitted changes to tracked files. Every record carries
  `env.torch_op_bench.hash`; a run from a dirty tree, or one during which the
  checkout changed, names code that is not what ran (the step-5 sweep had records
  from two commits, because a commit landed mid-run). So `push` also refuses while
  a run is alive on the host, and `--allow-dirty`, which is for development only
  and says so loudly, does not override that. The remote `.git` is a repository of
  its own: a checkout whose `.git` is a `gitdir:` file (`git clone
  --separate-git-dir`) has that directory copied, objects borrowed through
  `objects/info/alternates` (`git clone --shared`) are copied in, one rsync more
  each, and a linked worktree is refused (push from a clone). The whole repository
  goes, `.git/config`, every ref and the hooks included: a remote URL with
  credentials in it reaches the host too, so push to a shared host from a clone
  without them. A tracked file that replaced a directory of the same name, or a
  directory that replaced a file, replaces it on the host as well (rsync's
  `--force`, which acts only on the tracked paths: a host's ignored file inside
  such a directory goes with it). `push --wheelhouse DIR` also copies the wheels
  in `DIR` to `REMOTE_DIR/wheelhouse`, for a host whose proxy does not reach an
  index, or to save the downloads: `setup` sets `PIP_FIND_LINKS` to it (pip's
  `--find-links`, also for the pip step of `mamba env create`), keeping the
  session's own `PIP_FIND_LINKS` after it.
- **`bootstrap`** installs Miniforge into the scratch directory when the host has
  neither mamba nor conda. It writes `env.sh`, which puts every cache (Triton,
  Inductor, CUDA's JIT cache, conda packages, pip) in the scratch directory, sets
  `TMPDIR` to `REMOTE_DIR/tmp` when the session has none, and hands a proxy's CA
  (`SSL_CERT_FILE`) to mamba and conda as `MAMBA_SSL_VERIFY` and `CONDA_SSL_VERIFY`
  (libmamba ignores `SSL_CERT_FILE`) unless the session sets them. The
  installer and `setup` run with `HOME` in the scratch directory, since conda and
  mamba write `~/.conda` whatever `CONDA_REGISTER_ENVS` says, so `HOME` may be
  read-only. A download that fails (a proxy's 403) stops `bootstrap` or `setup`
  with the URL or the environment and step that failed.
- **`setup`** runs `scripts/setup-env.sh` for the torch environment and each of
  `--envs`. The torch environment is `--torch-env torch-nightly` (default:
  `envs/torch-nightly.yml`, a pinned nightly wheel, which carries the newest CuTeDSL
  override) or `torch-stable` (`envs/torch-stable.yml`, the PyPI release). Take the
  nightly to measure what torch `main` does now, and the release to measure what
  users have. torch 2.14.1 has `F.linear_cross_entropy` but not its CuTeDSL
  override, so `torch_lce` is `unavailable` there. torch-op-bench
  runs from that environment's interpreter, so it is the `torch` environment and
  needs no `--env torch=...`. `setup --envs` writes the counterparts' names to
  `REMOTE_DIR/envs.list`, one per line, and a `run` without `--envs` takes them
  from there (a `setup` without `--envs`, of the other torch environment, leaves
  the list as it was). Last, `setup` prints the host's name, `nvidia-smi` and
  `torch-op-bench list`.
- **`run`** starts `torch-op-bench run lce --device cuda` with `nohup` and
  `setsid`, writing to `logs/<timestamp>.log` and `results/` in the scratch
  directory, and returns at once. A dropped connection does not end it. Running the
  same `run` again (after a disconnect, or after an error) resumes: the triples
  recorded `ok` are skipped. Arguments after `--` go to `torch-op-bench run`;
  `--gpu N` sets `CUDA_VISIBLE_DEVICES`; `--op` names another op. A counterpart
  environment that `--envs` or `envs.list` names and `setup` did not make stops
  the run before it starts. The log begins with the host's name,
  `CUDA_VISIBLE_DEVICES` and `nvidia-smi -L`.
  `--matmul-reduction strict` runs the strict point into `results-strict/` instead
  (`--matmul-reduction strict --results "$D"/results-strict` on the host), e.g.
  `run --matmul-reduction strict --envs liger,torchtitan -- --case N=4096`, and
  `--torch-env torch-stable` runs into `results-stable/` (and
  `results-stable-strict/`): the resume and merge key holds neither the setting
  nor the torch build, so a release run in `results/` would be skipped against the
  nightly's records. Both choose the results directory, so neither `--results`
  nor `--matmul-reduction` is accepted after `--`.

  Per GPU, the run plan is: the default sweeps (the base entries; `run` without
  `--impl` leaves the variants out), the strict point, then the variants (see
  "Benchmarking LCE") at the default point only, into the same `results/` as the
  sweeps, under the default setting, for the report's "Time at equal peak":

  ```sh
  scripts/remote-run.sh HOST run --envs liger,torchtitan -- --case N=4096 \
      --impl 'torch_lce[batch_chunk_size=256],torch_lce[batch_chunk_size=1024]' \
      --impl 'torch_lce_no_override[batch_chunk_size=256],torch_lce_no_override[batch_chunk_size=1024]' \
      --impl 'liger[chunk_mem_const=4],liger[chunk_mem_const=8]' \
      --impl 'liger_native_accum[chunk_mem_const=4],liger_native_accum[chunk_mem_const=8]' \
      --impl 'torchtitan_chunked[num_chunks=4],torchtitan_chunked[num_chunks=32]' \
      --impl 'torchtitan_chunked_compiled[num_chunks=4],torchtitan_chunked_compiled[num_chunks=32]'
  ```

  The variants exist for that comparison, not to tune an entry: they never run
  along a sweep, nor under `--matmul-reduction strict`.
- **`status`** and **`wait`** say which results directory the run writes;
  `status` counts the records of each one on the host.
- **`fetch`** copies each of the host's results directories (`results/`,
  `results-strict/`, `results-stable/`, `results-stable-strict/`, those it has)
  into this checkout's of the same name without deleting anything on either side
  (the files are append-only, with new names per run), then says how many records
  each slug gained and prints the start of its report as a check. While a run is
  alive it warns: the last line of a file may be half-written, and `report` fails
  on it until a fetch after the run. Commit the new files yourself, one commit per
  host run: `results/<op>/<slug>/<timestamp>-<host>.jsonl`.
- **`shell`** opens bash on the host in the scratch directory, with `env.sh`
  sourced, from the rc file `bootstrap` writes (`REMOTE_DIR/shell.rc`): the login
  shell, whatever it is, only runs `bash --rcfile REMOTE_DIR/shell.rc -i`.

`--dry-run` prints every command, local and remote, and runs none.

### A host behind agent-sandbox's sandboxed login

A host may serve its benchmark account through an agent-sandbox login: every
session runs in a sandbox with one writable directory, `/tmp` a private tmpfs,
egress only through an allowlist proxy, and the cards the policy binds (one,
seen as GPU 0). There are two kinds:

- **A login made by `asb --setup-remote ALIAS`** (from the project's
  `[remote:ALIAS]` section, which names the host, the account, the scratch, the
  cards and the allowed hosts). `asb --setup-remote` writes and prints the ssh
  alias, the key, the host key and the role's `host-port` line. `HOME` is the
  scratch (e.g. `/home/tob/tob-ey`), writable, and a session starts in it, so the
  default `--dir ~/torch_op_bench-scratch` works as on a plain host:

  ```sh
  scripts/remote-run.sh tob-ey push
  scripts/remote-run.sh tob-ey bootstrap   # ... as in the steps above
  ```

- **The older sshd profile**: `HOME` is an empty read-only tmpfs and the one
  writable place is the policy's directory (e.g. `/home/tob/bench`). Give every
  subcommand an absolute subdirectory of it, never the directory itself (the
  policy file lives there, and `push`'s checkout goes under
  `REMOTE_DIR/torch-op-bench`):

  ```sh
  scripts/remote-run.sh HOST push --dir /home/tob/bench/tob-scratch
  ```

Nothing in the script needs a writable `HOME` beyond `--dir` itself, so one script
serves both. On either:

- The policy must allow the workflow's hosts: `github.com` and
  `.githubusercontent.com` (the Miniforge installer, the git pins of the
  counterparts), `conda.anaconda.org`, `pypi.org`, `files.pythonhosted.org`,
  `download.pytorch.org`, `download-r2.pytorch.org` (where the nightly index
  redirects its wheels) and `pypi.nvidia.com` (the nightly's NVIDIA wheels). The
  proxy refuses any other with a 403, which `bootstrap` and `setup` report with
  the URL or the environment that failed. On a `--setup-remote` login the host's
  owner adds one with `asb --apply '[remote:ALIAS]' 'allow = HOST'`, then
  `asb --setup-remote ALIAS`.
- No `--gpu`: the session sees only the card the policy binds, as GPU 0, and
  cannot touch another (`CUDA_VISIBLE_DEVICES`, if the policy's `[env]` sets
  one, passes through). The log's `nvidia-smi -L` shows it.
- A relaunch of the login (`asb --setup-remote` again) or its `--shutdown` ends
  every session's processes, a detached `run` included: the host's owner does
  neither during a set.
- Before the first `run`, a smoke test of the sandbox, in one session:

  ```sh
  ssh tob-ey 'hostname; nvidia-smi -L; touch /home/tob/x; touch /tmp/x && ls -l /tmp/x; for u in https://example.com/ https://pypi.org/simple/; do curl -s -o /dev/null -w "%{http_code} $u\n" "$u"; done'
  ```

  It should print the host's own name (`ey`, not `pasta-…`: the records carry
  it), one card, `Read-only file system` for a path outside the writable
  directory (`/home/tob/x`; on the older profile `$HOME/x` too), `/tmp/x` (in
  the sandbox's private `/tmp`, not the host's), `000 https://example.com/`
  (curl's `CONNECT tunnel failed, response 403`: a host off the list is refused)
  and `200 https://pypi.org/simple/`.

On a host without the sandbox, nothing confines a run to the scratch directory
but the script: check it after a run with `touch MARKER` before it and
`find ~ -newer MARKER -not -path '*/torch_op_bench-scratch/*'` after (empty).

## Tests

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu  # if no torch yet
pip install -e .[dev,plot]
```

While iterating on a change, the implementer and the reviewer run only the tests it
affects. The full suite, CUDA included, runs once, when the reviewer would accept the
change with no findings, and CI runs the full CPU suite with the coverage floor on
every push.

This project's stable test interface, which tools outside it may name, is two
commands: `scripts/test-affected.sh [--base REF] [-- PATH...]` to iterate, and
`COVERAGE_CORE=ctrace python -m pytest --testmon-noselect -n auto --maxprocesses 10`
for the final run, both described below. Their names and arguments stay as they are.

Iterating, `scripts/test-affected.sh [--base REF] [-- PATH...]` lints, then runs the
tests `pytest-testmon` selects, then the test files `scripts/affected-tests.py` maps
the changed paths to, all in the interpreter `$PYTHON` (default `python`). The
changed paths are those since `main` (or REF), or the PATHs given, such as the
`git diff --name-only` of a change applied to a copy without `.git`. For a change to
`envs/` alone, that is

```sh
ruff check .
COVERAGE_CORE=ctrace python -m pytest --testmon
COVERAGE_CORE=ctrace python -m pytest --testmon-noselect tests/test_envs.py
```

The map covers what testmon cannot see: every module the worker subprocess imports,
the scripts and environment definitions the tests run as files, and the files every
test depends on (`pyproject.toml` and the shared test helpers), for which it says
`all`. It also says `all` once for a new interpreter, a new or rebuilt torch, or a
new GPU, from the stamp `.testmondata.env` it keeps beside `.testmondata`. For `all`
the script runs the whole suite once instead, in parallel:

```sh
ruff check .
COVERAGE_CORE=ctrace python -m pytest --testmon-noselect -n auto --maxprocesses 10
```

The final run, in the interpreter with CUDA and the counterpart environments present:
the whole suite, which also refreshes testmon's data, then CI's own command.

```sh
ruff check .
COVERAGE_CORE=ctrace pytest --testmon-noselect -n auto --maxprocesses 10
pytest --cov=torch_op_bench --cov-report=term-missing --cov-fail-under=90 -n auto --maxprocesses 10
```

The whole-suite runs above, and CI's, run in parallel with `pytest-xdist` (in
`dev`): `-n auto` starts a worker per core, at most 10 (more workers gained less
than ten percent on a 32-core host; CI's runner has four). `-n` is on those
commands only, not in the pytest options: a selected run stays serial, as each
worker imports torch first. `-n 0` runs a whole suite serially. When a CUDA device
is present, the `cuda` tests all go to one worker (`--dist loadgroup`, in
`pyproject.toml`'s pytest options, groups them, from `tests/conftest.py`): the
peak-memory and subprocess-CUDA-memory tests measure the device, and a second
worker on it would perturb them. They run under their plain ids, so `pytest-testmon`
records each test once, however the suite ran, and a parallel final run refreshes
the data the serial selection reads. `dev` keeps pytest below 9.2: under pytest
9.2, pytest-xdist's `loadgroup` stops grouping (pytest-dev/pytest-xdist#1386); the
bound goes when a pytest-xdist release groups under pytest 9.2.

Tests marked `cuda` run when a CUDA device is present and are skipped otherwise. The
CUDA test of the Liger entries runs once per Liger environment present
(`envs/.prefix/liger`, `envs/.prefix/liger-git`, or the interpreter in
`TOB_LIGER_PYTHON`) and is skipped without one. The torchtitan tests (a CPU one and a
CUDA one) run the same way, once per torchtitan environment present
(`envs/.prefix/torchtitan`, `envs/.prefix/torchtitan-git`, or
`TOB_TORCHTITAN_PYTHON`).

`pytest-testmon` (in `dev`) selects the tests a change affects from the coverage of
each test it recorded. It does not combine with `pytest-cov` in one run: a run with
`--testmon` never passes `--cov`, and CI passes `--cov` and never `--testmon`. A run
with `--testmon` sets `COVERAGE_CORE=ctrace`: testmon records which test ran each
line by switching coverage's context per test, and coverage's default core on Python
3.14, `sysmon`, records a line for the first test that runs it only. Its data,
`.testmondata`, is per interpreter and per checkout: it lives in the checkout root
and is never shared or committed, so the first run in a fresh copy runs every test.

A pytest marker expression (`-m`, such as `-m "not cuda"`) makes testmon turn its own
selection off, so that run is the whole suite less the deselected tests;
`--testmon-forceselect` in place of `--testmon` keeps the selection under `-m`.

## Licence

BSD-3-Clause; see `LICENSE`.
