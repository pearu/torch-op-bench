"""Records to rows: what ``report`` and ``plot`` read (R6–R8).

Op-agnostic and stdlib only: it reads records, not an op's module, so ``report``
runs in any environment that has the tool. An op's spec is consulted only for the
order of its axes and its default point, and only when its ``op_ref`` resolves.
"""

from __future__ import annotations

import collections
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import records

Record = dict[str, Any]
Row = dict[str, Any]

MIB = 1024**2

# Where a record came from: the file name, kept beside the record for its host part.
SOURCE = "_source"

HEAD = ["op", "device_slug", "host", "timestamp", "implementation", "mode", "env_name", "dtype"]
RESULTS = [
    "status",
    "reason",
    "n",
    "median_ms",
    "min_ms",
    "mean_ms",
    "iqr_ms",
    "peak_bytes",
    "baseline_bytes",
    "peak_above_baseline",
    "time_x_peak",
    "tokens",
    "flops",
    "tokens_per_s",
    "tflops",
    "parity_status",
    "parity_failed",
]
TAIL = [
    "role",
    "control_of",
    "compiled",
    "override_ran",
    "native_events",
    "torch_version",
    "torch_cuda",
    "torch_git_version",
    "matmul_bf16_reduced",
    "packages",
    "tob_hash",
]
# info keys that are columns of their own, so never repeated as varying info.
INFO_COLUMNS = ("role", "control_of", "compiled")
INFO_PREFIX = "info."
PARITY_PREFIX = "parity_"


def load(
    results_dir: str | Path, *, op: str | None = None, device_slug: str | None = None
) -> list[Record]:
    """The merged records under ``results_dir`` (or ``<results>/<op>/<slug>``).

    ``records.merge`` keeps one record per key and warns when the kept records come
    from different torch-op-bench commits; the warning is not caught here. Each
    record keeps its file name under ``SOURCE``.
    """
    root = Path(results_dir)
    if op is not None:
        root = root / op
        if device_slug is not None:
            root = root / device_slug
    if not root.exists():
        raise FileNotFoundError(f"no results at {root}")
    files = sorted(root.rglob("*.jsonl")) if root.is_dir() else [root]
    found = []
    for file in files:
        for record in records.read(file):
            if op is not None and record["op"] != op:
                continue
            if device_slug is not None and record["device_slug"] != device_slug:
                continue
            found.append({**record, SOURCE: file.name})
    return records.merge(found)


# --- the op's spec, when it resolves ------------------------------------------------

_SPECS: dict[str, Any] = {}


def spec_of(record: Mapping[str, Any]) -> Any:
    """The op spec of ``record["op_ref"]``, or ``None`` when it does not resolve here."""
    ref = record.get("op_ref") or record.get("op")
    if ref not in _SPECS:
        from .spec import resolve_op

        try:
            _SPECS[ref] = resolve_op(ref)
        except Exception:  # not importable here: the rows fall back to the records
            _SPECS[ref] = None
    return _SPECS[ref]


def axes_of(
    recs: Sequence[Mapping[str, Any]], spec: Any = None, *, resolve: bool = True
) -> list[str]:
    """The case axes, in the op's order when its spec resolves, else sorted."""
    names = {name for r in recs for name in r["case"]["params"]}
    if spec is None and resolve and recs:
        spec = spec_of(recs[0])
    if spec is not None:
        ordered = [axis.name for axis in spec.axes if axis.name in names]
        return ordered + sorted(names - set(ordered))
    return sorted(names)


# --- one record, one row --------------------------------------------------------------


def _get(mapping: Any, *path: str) -> Any:
    for part in path:
        if not isinstance(mapping, Mapping):
            return None
        mapping = mapping.get(part)
    return mapping


def flat_info(info: Mapping[str, Any] | None, prefix: str = "") -> dict[str, Any]:
    """``info`` with nested dicts as dotted keys."""
    flat: dict[str, Any] = {}
    for name, value in (info or {}).items():
        if isinstance(value, Mapping):
            flat.update(flat_info(value, f"{prefix}{name}."))
        else:
            flat[f"{prefix}{name}"] = value
    return flat


def host_of(record: Mapping[str, Any]) -> str | None:
    """``env.hostname``, else the host part of the file name ``<stamp>-<host>.jsonl``."""
    host = _get(record, "env", "hostname")
    if host:
        return host
    source = record.get(SOURCE)
    if source:
        stem = Path(source).stem
        return stem.split("-", 1)[1] if "-" in stem else None
    return None


def packages_text(packages: Mapping[str, Any] | None) -> str | None:
    """``name=version`` or ``name=commit[:12]`` for a VCS install, ``;``-joined."""
    if not packages:
        return None
    parts = []
    for name in sorted(packages):
        value = packages[name]
        if isinstance(value, Mapping):
            value = (value.get("commit") or "")[:12] or value.get("version")
        parts.append(f"{name}={'not installed' if value is None else value}")
    return ";".join(parts)


def _cell(value: Any) -> Any:
    # Lists and dicts (e.g. an info field) as JSON text, so a row stays flat.
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def flatten(
    record: Mapping[str, Any],
    *,
    axes: Sequence[str] | None = None,
    info_keys: Sequence[str] = (),
) -> Row:
    """One flat row of ``record``; a field the record lacks is ``None``."""
    case = record.get("case") or {}
    params = case.get("params") or {}
    axes = sorted(params) if axes is None else axes
    row: Row = {
        "op": record.get("op"),
        "device_slug": record.get("device_slug"),
        "host": host_of(record),
        "timestamp": record.get("timestamp"),
        "implementation": record.get("implementation"),
        "mode": record.get("mode"),
        "env_name": record.get("env_name"),
        "dtype": case.get("dtype"),
    }
    row.update({name: params.get(name) for name in axes})

    times = record.get("times_ms") or []
    timing = records.stats(times)
    memory = record.get("memory") or {}
    peak, baseline = memory.get("peak_bytes"), memory.get("baseline_bytes")
    on_cpu = str(case.get("device", "")).startswith("cpu")
    work = record.get("work") or {}
    median_s = timing["median"] / 1e3 if timing["median"] else None
    tokens, flops = work.get("tokens"), work.get("flops")
    parity = record.get("parity") or {}
    above = None if on_cpu or peak is None or baseline is None else peak - baseline
    row.update(
        {
            "status": record.get("status"),
            "reason": record.get("reason"),
            "n": timing["n"] if times else None,
            "median_ms": timing["median"],
            "min_ms": timing["min"],
            "mean_ms": timing["mean"],
            "iqr_ms": timing["iqr"],
            "peak_bytes": peak,
            "baseline_bytes": baseline,
            "peak_above_baseline": above,
            "time_x_peak": time_x_peak(timing["median"], above),
            "tokens": tokens,
            "flops": flops,
            "tokens_per_s": tokens / median_s if tokens is not None and median_s else None,
            "tflops": flops / median_s / 1e12 if flops is not None and median_s else None,
            "parity_status": parity.get("status"),
            "parity_failed": ",".join(parity.get("failed") or []) or None,
        }
    )
    for name, value in sorted((parity.get("metrics") or {}).items()):
        row[PARITY_PREFIX + name] = value

    info = record.get("info") or {}
    verify = record.get("verify") or {}
    native = verify.get("native_events")
    torch = _get(record, "env", "torch") or {}
    row.update(
        {
            "role": info.get("role"),
            "control_of": info.get("control_of"),
            "compiled": info.get("compiled"),
            "override_ran": verify.get("override_ran"),
            "native_events": len(native) if isinstance(native, list) else None,
            "torch_version": torch.get("version"),
            "torch_cuda": torch.get("cuda"),
            "torch_git_version": torch.get("git_version"),
            "matmul_bf16_reduced": _get(torch, "matmul", "allow_bf16_reduced_precision_reduction"),
            "packages": packages_text(_get(record, "env", "packages")),
            "tob_hash": _get(record, "env", "torch_op_bench", "hash"),
        }
    )
    flat = flat_info(info)
    for name in info_keys:
        row[INFO_PREFIX + name] = _cell(flat.get(name))
    return row


def time_x_peak(median_ms: float | None, peak_above_baseline: int | None) -> float | None:
    """Elapsed time × peak memory, in ms·MiB: the median time and the peak above the
    process baseline, as the peak columns are. The user's comparison (Statement
    §6 step 6b): a counterpart that is faster by holding more memory, or smaller by
    taking longer, does not win on it unless the product drops."""
    if median_ms is None or peak_above_baseline is None:
        return None
    return median_ms * peak_above_baseline / MIB


def varying_info(recs: Iterable[Mapping[str, Any]]) -> list[str]:
    """The (dotted) ``info`` keys whose values differ among the ``ok`` records:
    within one implementation or across implementations. A key that some records
    lack differs too. Keys that are columns of their own are left out."""
    ok = [flat_info(r.get("info")) for r in recs if r.get("status") == "ok"]
    keys = {name for info in ok for name in info}
    missing = object()
    varying = []
    for name in sorted(keys - set(INFO_COLUMNS)):
        values = {json.dumps(info.get(name, missing), sort_keys=True, default=id) for info in ok}
        if len(values) > 1:
            varying.append(name)
    return varying


def to_rows(recs: Sequence[Mapping[str, Any]], spec: Any = None) -> list[Row]:
    """``flatten`` of every record, with the same columns in every row: the axes in
    the op's order, every parity metric seen, and the varying info keys."""
    axes = axes_of(recs, spec)
    info_keys = varying_info(recs)
    rows = [flatten(r, axes=axes, info_keys=info_keys) for r in recs]
    columns = columns_of(rows, axes, info_keys)
    return [{name: row.get(name) for name in columns} for row in rows]


def columns_of(rows: Sequence[Row], axes: Sequence[str], info_keys: Sequence[str]) -> list[str]:
    parity = sorted({k for row in rows for k in row if k.startswith(PARITY_PREFIX)} - set(RESULTS))
    return [*HEAD, *axes, *RESULTS, *parity, *TAIL, *(INFO_PREFIX + k for k in info_keys)]


# --- the default point and the sweeps around it -----------------------------------------


def point_text(params: Mapping[str, Any]) -> str:
    return ", ".join(f"{name}={value}" for name, value in params.items())


def _has_ok(recs: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> bool:
    return any(
        r.get("status") == "ok"
        and all(r["case"]["params"].get(name) == value for name, value in params.items())
        for r in recs
    )


def _inferred(recs: Sequence[Mapping[str, Any]], spec: Any = None) -> dict[str, Any]:
    """The most common value of each axis over the ``ok`` records (ties: the first
    seen); when no ``ok`` record is at that combination, the most common case. The
    axes in the spec's order when it is given."""
    ok = [r for r in recs if r.get("status") == "ok"] or list(recs)
    params: dict[str, Any] = {}
    for name in axes_of(ok, spec, resolve=False):
        counts = collections.Counter(r["case"]["params"].get(name) for r in ok)
        params[name] = counts.most_common(1)[0][0]
    if ok and not _has_ok(recs, params):
        cases = collections.Counter(json.dumps(r["case"]["params"], sort_keys=True) for r in ok)
        common = json.loads(cases.most_common(1)[0][0])
        params = {name: common.get(name) for name in params}
    return params


def default_point(
    recs: Sequence[Mapping[str, Any]],
    spec: Any = None,
    *,
    point: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """``{"source": ..., "params": {axis: value}}``: the point the report and the
    figures centre on.

    ``point`` (``--point``, axis to the value as text) wins for the axes it names,
    ``source`` ``"--point"``. The rest is the op's ``defaults`` when its spec
    resolves (here, or given) and some ``ok`` record is at them, ``source``
    ``"spec"``; else the most common value of each axis over the ``ok`` records
    (ties: the first seen), ``source`` ``"inferred"``, or, when the spec's point has
    no records, ``"inferred (the spec's default point ... has no records)"``.
    """
    if spec is None and recs:
        spec = spec_of(recs[0])
    if spec is not None:
        params = dict(spec.default_point)
        if _has_ok(recs, params) or not recs:
            found = {"source": "spec", "params": params}
        else:
            found = {
                "source": (
                    f"inferred (the spec's default point {point_text(params)} has no records)"
                ),
                "params": _inferred(recs, spec),
            }
    else:
        found = {"source": "inferred", "params": _inferred(recs)}
    if point:
        params = dict(found["params"])
        for name, value in point.items():
            if name not in params:
                known = ", ".join(params) or "none"
                raise ValueError(f"--point: unknown axis {name!r}; known: {known}")
            params[name] = _parse_value(recs, spec, name, value)
        found = {"source": "--point", "params": params}
    return found


def _parse_value(recs: Sequence[Mapping[str, Any]], spec: Any, name: str, value: str) -> Any:
    """``value`` in the kind of axis ``name``: the spec's when it resolves, else the
    kind of the values the records hold."""
    if spec is not None:
        for axis in spec.axes:
            if axis.name == name:
                return axis.parse(value)
    seen = [r["case"]["params"].get(name) for r in recs]
    if seen and all(isinstance(v, int) and not isinstance(v, bool) for v in seen):
        return int(value)
    return value


def parse_point(text: str | Sequence[str] | None) -> dict[str, str] | None:
    """``AXIS=VALUE,...`` (repeatable) to ``{axis: value}``, values as text."""
    if not text:
        return None
    parts = [text] if isinstance(text, str) else list(text)
    found: dict[str, str] = {}
    for item in (p for part in parts for p in part.split(",")):
        name, sep, value = item.partition("=")
        if not sep or not name.strip() or not value.strip():
            raise ValueError(f"--point: expected AXIS=VALUE, got {item!r}")
        found[name.strip()] = value.strip()
    return found


def sweeps(rows: Sequence[Row], point: Mapping[str, Any]) -> dict[str, list[Row]]:
    """For each axis, the rows whose other axes are at ``point``: the default point
    itself is in every sweep."""
    result = {}
    for axis in point:
        others = {name: value for name, value in point.items() if name != axis}
        result[axis] = [
            row for row in rows if all(row.get(name) == value for name, value in others.items())
        ]
    return result


def axis_values(rows: Sequence[Row], axis: str) -> list[Any]:
    """The distinct values of ``axis`` in ``rows``: numbers ascending, else as seen."""
    values = list(dict.fromkeys(row[axis] for row in rows if row.get(axis) is not None))
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return sorted(values)
    return values


def at(rows: Iterable[Row], **where: Any) -> list[Row]:
    """The rows whose columns equal ``where``."""
    return [row for row in rows if all(row.get(k) == v for k, v in where.items())]


def case_of(row: Row, axes: Sequence[str]) -> tuple:
    """The case of a row beyond its implementation: dtype and the axes' values."""
    return (row.get("dtype"), *(row.get(name) for name in axes))


def implementations(rows: Sequence[Row], baseline: str | None = None) -> list[str]:
    """Implementations in display order: the baseline first; then each in order of
    first appearance, with its controls right after it."""
    seen = list(dict.fromkeys(row["implementation"] for row in rows))
    control_of = {
        row["implementation"]: row.get("control_of")
        for row in rows
        if row.get("role") == "control" and row.get("control_of")
    }
    order: list[str] = []
    if baseline in seen:
        order.append(baseline)
    for name in seen:
        if name in order or (name in control_of and control_of[name] in seen):
            continue
        order.append(name)
    for control, original in control_of.items():
        if original in seen and control not in order:
            following = [n for n in order if control_of.get(n) == original]
            anchor = following[-1] if following else original
            order.insert(order.index(anchor) + 1, control)
    return order
