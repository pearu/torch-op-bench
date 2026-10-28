"""A synthetic result set for the report and plot tests, written as ``.jsonl`` files.

The op is ``synth_op:SYNTH`` (axes M, K, N and the ``str`` axis ``layout``; two
dtypes; two modes). Four implementations:

- ``base``, the baseline, and ``compiled``, in the environment ``torch``;
- ``chunked`` (``fwd_bwd`` only) and its control ``base_other``, in ``other``, whose
  torch is another build and whose counterpart is installed from git.

Timings follow ``median_ms`` exactly (the samples are ``0.9 t, t, 1.1 t``), peak
memory ``peak_above``. Planted, for the tests to find:

- ``base_other`` is 10 % slower than ``base`` at the default point (``DEVIATION``),
  and holds 100 MiB more at ``M=256`` (``PEAK_DEVIATION``);
- ``base_other``'s ``grad_rel`` is 25 % larger than ``base``'s at ``N=8``
  (``PARITY_DEVIATION``), within tolerance: another build moving the error;
- ``chunked`` fails parity at ``M=256`` (``PARITY_FAIL``);
- ``compiled`` has an ``error`` record (``ERROR``, in the second host's file, with
  no ``env``, as the runner writes one) and no record at all at
  ``N=8`` ``fwd_bwd`` (``MISSING``); ``chunked`` an ``unavailable`` one
  (``UNAVAILABLE``);
- the second host's file also holds an older ``ok`` record of one key
  (``DUPLICATE``), which merge drops;
- the ``other`` environment's records come from another torch-op-bench commit,
  so merge warns.

``write_strict`` writes the strict counterpart at the default point (the records
of ``run --matmul-reduction strict``), every implementation but ``compiled``
(``STRICT_ABSENT``), each ``STRICT_TIME`` times slower. Planted: ``base``'s parity
halves under strict (``STRICT_PARITY``: it moves); ``base_other``'s grows 5 %
(within the band); ``chunked``'s stays, and its records carry ``op_decisions``
(``STRICT_DECISIONS``) as torch LCE's do; ``base_other`` is ``unavailable`` in
``fwd`` ``float32`` (``STRICT_UNAVAILABLE``); one record has the default setting
(``STRICT_ODD``).
"""

from __future__ import annotations

import datetime
from pathlib import Path

from torch_op_bench import records, worker
from torch_op_bench.spec import Case

OP_REF = "synth_op:SYNTH"
SLUG = "synthdev"
DEVICE = "cuda:0"
DEFAULT = {"M": 128, "K": 32, "N": 16, "layout": "a"}
SWEEP = {"M": [64, 128, 256], "K": [16, 32], "N": [8, 16], "layout": ["a", "b"]}
DTYPES = ["float32", "bfloat16"]
MODES = ["fwd", "fwd_bwd"]
MIB = 1024**2
BASELINE_BYTES = 8 * MIB

# implementation: (environment, modes, time factor, memory factor, info)
IMPLS = {
    "base": ("torch", MODES, 1.0, 1.0, {"role": "baseline", "compiled": False}),
    "compiled": ("torch", MODES, 0.5, 1.0, {"compiled": True, "compile": "fullgraph"}),
    "chunked": ("other", ["fwd_bwd"], 2.0, 0.25, {"num_chunks": 4, "api": {"name": "v2"}}),
    "base_other": (
        "other",
        MODES,
        1.0,
        1.0,
        {"role": "control", "control_of": "base", "compiled": False},
    ),
}
DEVIATION = 1.10
PEAK_DEVIATION = 100 * MIB
PARITY_FAIL = ("chunked", {"M": 256})
PARITY_DEVIATION = ("base_other", {"N": 8}, 1.25)
ERROR = ("compiled", "fwd", "float32", {"M": 64})
UNAVAILABLE = ("chunked", "fwd_bwd", "float32", {"K": 16})
MISSING = ("compiled", "fwd_bwd", "float32", {"N": 8})
DUPLICATE = ("base", "fwd_bwd", "float32", {})

ENVS = {
    "torch": {
        "version": "2.16.0a0+gitabc",
        "git_version": "abc" * 13 + "a",
        "packages": {},
        "hash": "a" * 40,
    },
    "other": {
        "version": "2.14.1+cu130",
        "git_version": "def" * 13 + "d",
        "packages": {
            "counterpart": {
                "version": "0.2.2",
                "url": "https://example.org/counterpart",
                "commit": "0123456789abcdef0123456789abcdef01234567",
                "requested": "0123456789abcdef0123456789abcdef01234567",
            },
            "helper": "1.0",
        },
        "hash": "b" * 40,
    },
}


def points() -> list[dict]:
    """The one-at-a-time sweep around the default point, the default first."""
    found = [dict(DEFAULT)]
    for axis, values in SWEEP.items():
        for value in values:
            point = {**DEFAULT, axis: value}
            if point not in found:
                found.append(point)
    return found


def _matches(params: dict, where: dict) -> bool:
    return all(params[k] == v for k, v in where.items())


def median_ms(impl: str, mode: str, dtype: str, params: dict) -> float:
    """The formula the timings follow."""
    factor = IMPLS[impl][2]
    if impl == "base_other" and params == DEFAULT:
        factor *= DEVIATION
    size = params["M"] * params["K"] * params["N"] / 1024
    layout = 1.0 if params["layout"] == "a" else 1.5
    return (
        factor
        * size
        * layout
        * (2.0 if mode == "fwd_bwd" else 1.0)
        * (1.0 if dtype == "float32" else 0.5)
    )


def peak_above(impl: str, params: dict) -> int:
    extra = PEAK_DEVIATION if impl == "base_other" and params["M"] == 256 else 0
    return int(IMPLS[impl][3] * params["M"] * params["K"] * MIB / 64) + extra


def tokens(params: dict) -> int:
    return params["M"]


def _record(impl: str, mode: str, dtype: str, params: dict, when: datetime.datetime) -> dict:
    env_name, _, _, _, info = IMPLS[impl]
    case = Case(dict(params), dtype, DEVICE)
    payload = {
        "op": OP_REF,
        "op_name": "synth",
        "case": case.to_dict(),
        "implementation": impl,
        "mode": mode,
        "settings": {"warmup": 1, "repeat": 3},
        "env_name": env_name,
        "device_slug": SLUG,
        "work": {"tokens": tokens(params), "flops": 2 * params["M"] * params["K"] * params["N"]},
        "info": dict(info),
    }
    record = worker.base_record(payload, f"/envs/{env_name}/bin/python")
    record["timestamp"] = when.isoformat(timespec="seconds")
    environment = ENVS[env_name]
    record["env_from"] = "worker"
    record["env"] = {
        "hostname": "host-a",
        "python": "3.14.8" if env_name == "torch" else "3.13.1",
        "torch": {
            "version": environment["version"],
            "git_version": environment["git_version"],
            "cuda": "13.0",
            "matmul": {
                "allow_bf16_reduced_precision_reduction": True,
                "allow_fp16_reduced_precision_reduction": True,
                "allow_tf32": False,
            },
        },
        "torch_op_bench": {"version": "0.1.0", "hash": environment["hash"], "dirty": False},
        "packages": dict(environment["packages"]),
    }
    if (
        impl == UNAVAILABLE[0]
        and (mode, dtype) == UNAVAILABLE[1:3]
        and _matches(params, UNAVAILABLE[3])
    ):
        record.update(status="unavailable", reason="needs a GPU")
        return record
    t = median_ms(impl, mode, dtype, params)
    failed = impl == PARITY_FAIL[0] and _matches(params, PARITY_FAIL[1])
    metrics = {"loss_rel_err": 1e-3, "grad_rel": 0.5 if failed else 2e-3}
    if impl == PARITY_DEVIATION[0] and _matches(params, PARITY_DEVIATION[1]):
        metrics["grad_rel"] *= PARITY_DEVIATION[2]
    if mode == "fwd":
        metrics = {"loss_rel_err": 1e-3}
    record.update(
        status="ok",
        times_ms=[0.9 * t, t, 1.1 * t],
        memory={
            "baseline_bytes": BASELINE_BYTES,
            "peak_bytes": BASELINE_BYTES + peak_above(impl, params),
        },
        parity={
            "status": "fail" if failed and mode == "fwd_bwd" else "ok",
            "failed": ["grad_rel"] if failed and mode == "fwd_bwd" else [],
            "metrics": metrics,
            "tolerances": {"loss_rel_err": 1e-2, "grad_rel": 1e-2},
            "reason": None,
        },
        verify={"native_events": ["_native::x"] if impl == "compiled" else []},
    )
    return record


def write(results_dir: str | Path) -> Path:
    """Write the result set under ``results_dir``; return ``<results>/synth/<slug>``."""
    when = datetime.datetime(2026, 10, 17, 12, 0, tzinfo=datetime.timezone.utc)
    first = records.result_path(results_dir, "synth", SLUG, timestamp=when, host="host-a")
    for impl, (_, modes, *_) in IMPLS.items():
        for mode in modes:
            for dtype in DTYPES:
                for params in points():
                    if (
                        impl == MISSING[0]
                        and (mode, dtype) == MISSING[1:3]
                        and _matches(params, MISSING[3])
                    ):
                        continue
                    if (
                        impl == ERROR[0]
                        and (mode, dtype) == ERROR[1:3]
                        and _matches(params, ERROR[3])
                    ):
                        continue  # in the second host's file
                    records.append(first, _record(impl, mode, dtype, params, when))
    older = when - datetime.timedelta(days=1)
    second = records.result_path(results_dir, "synth", SLUG, timestamp=older, host="host-b")
    impl, mode, dtype, where = DUPLICATE
    stale = _record(impl, mode, dtype, {**DEFAULT, **where}, older)
    stale["times_ms"] = [999.0, 999.0, 999.0]
    stale["env"] = None  # a record the runner wrote: the host is the file name's
    records.append(second, stale)
    # A worker that died: the runner wrote this record, with no env; its host is the
    # file name's.
    impl, mode, dtype, where = ERROR
    failed = _record(impl, mode, dtype, {**DEFAULT, **where}, when)
    failed.update(status="error", reason="RuntimeError: boom\nsecond line", traceback="tb")
    failed.update(env=None, env_from="runner", times_ms=None, memory=None, parity=None)
    failed["verify"] = None
    records.append(second, failed)
    return first.parent


def subset(results_dir: str | Path, out: str | Path, *wheres: dict) -> Path:
    """The records of ``results_dir`` whose case matches one of ``wheres`` (each
    completed from the default point), as one file under ``out``; returns ``out``."""
    import warnings

    from torch_op_bench import analysis

    points = [{**DEFAULT, **where} for where in wheres]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recs = analysis.load(results_dir)
    target = Path(out) / "synth" / SLUG / "20261017T120000Z-host-a.jsonl"
    for record in recs:
        if record["case"]["params"] in points:
            records.append(target, {k: v for k, v in record.items() if k != analysis.SOURCE})
    return Path(out)


STRICT_TIME = 1.2
STRICT_ABSENT = "compiled"
STRICT_PARITY = {"base": 0.5, "base_other": 1.05, "chunked": 1.0}
STRICT_DECISIONS = {"source": "synth", "_mm_accumulates_in_fp32": True, "other_field": 1}
STRICT_UNAVAILABLE = ("base_other", "fwd", "float32")
STRICT_ODD = ("base", "fwd", "bfloat16")
STRICT_MATMUL = {
    "allow_bf16_reduced_precision_reduction": False,
    "allow_fp16_reduced_precision_reduction": False,
    "allow_bf16_reduced_precision_reduction_split_k": True,
    "allow_fp16_reduced_precision_reduction_split_k": True,
    "allow_tf32": False,
}


def write_strict(results_dir: str | Path) -> Path:
    """Write the strict set under ``results_dir``; return ``<results>/synth/<slug>``."""
    when = datetime.datetime(2026, 10, 17, 13, 0, tzinfo=datetime.timezone.utc)
    path = records.result_path(results_dir, "synth", SLUG, timestamp=when, host="host-a")
    for impl, (_, modes, *_) in IMPLS.items():
        if impl == STRICT_ABSENT:
            continue
        for mode in modes:
            for dtype in DTYPES:
                record = _record(impl, mode, dtype, dict(DEFAULT), when)
                record["settings"]["matmul_reduction"] = "strict"
                if (impl, mode, dtype) != STRICT_ODD:
                    record["env"]["torch"]["matmul"] = dict(STRICT_MATMUL)
                if impl == "chunked":
                    record["info"]["op_decisions"] = dict(STRICT_DECISIONS)
                if (impl, mode, dtype) == STRICT_UNAVAILABLE:
                    record.update(status="unavailable", reason="not here", times_ms=None)
                    record.update(memory=None, parity=None, verify=None)
                else:
                    record["times_ms"] = [STRICT_TIME * t for t in record["times_ms"]]
                    metrics = record["parity"]["metrics"]
                    for name in metrics:
                        metrics[name] *= STRICT_PARITY[impl]
                records.append(path, record)
    return path.parent
