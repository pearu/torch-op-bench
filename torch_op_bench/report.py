"""``torch-op-bench report``: the markdown tables of the write-up (R7).

Pure functions ``render_*`` turn rows (``analysis.to_rows``) into markdown;
``report`` loads a results directory and composes them. Stdlib only.
"""

from __future__ import annotations

import math
import shlex
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import analysis
from .analysis import Row
from .spec import MODES

MIB = 1024**2
MISSING = "—"
# A varying info value longer than this is cut in the tables (the record has it
# whole), so one long list cannot make the table unreadable.
INFO_WIDTH = 40
# The environment check: a control within these of its original is the same build.
TIME_RATIO_RANGE = (0.95, 1.05)
PEAK_DIFF_LIMIT = 64 * MIB


def sig3(value: Any) -> str:
    """A number in 3 significant digits; ``—`` for none."""
    if value is None:
        return MISSING
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if value == 0 or not math.isfinite(value):
        return f"{value:g}"
    mag = math.floor(math.log10(abs(value)))
    if math.floor(math.log10(abs(round(value, 2 - mag)))) > mag:  # 99.96 -> 100
        mag += 1
    if -3 <= mag < 6:
        # Rounded to 3 significant digits, also left of the point (1234 -> 1230).
        return f"{round(value, 2 - mag):.{max(0, 2 - mag)}f}"
    return f"{value:.2e}"


def mib(value: Any) -> str:
    """Bytes in MiB with one decimal; ``—`` for none."""
    return MISSING if value is None else f"{value / MIB:.1f}"


def text(value: Any, width: int | None = None) -> str:
    if value is None:
        return MISSING
    value = str(value).replace("\n", " ")
    if width is not None and len(value) > width:
        value = value[: width - 1] + "…"
    return value.replace("|", "\\|")


def table(header: Sequence[str], rows: Sequence[Sequence[str]], numeric: Sequence[bool]) -> str:
    """A markdown table; numeric columns right-aligned."""
    align = ["---:" if n else ":---" for n in numeric]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def ratio(a: Any, b: Any) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def case_text(row: Row, axes: Sequence[str]) -> str:
    return ", ".join(f"{name}={row.get(name)}" for name in axes)


def parity_text(row: Row) -> str:
    status = row.get("parity_status")
    if status == "fail" and row.get("parity_failed"):
        return f"fail: {row['parity_failed']}"
    return text(status)


def impl_label(row_or_name: Row | str, rows: Sequence[Row]) -> str:
    name = row_or_name if isinstance(row_or_name, str) else row_or_name["implementation"]
    for row in rows:
        if row["implementation"] == name and row.get("role") == "control":
            return f"{name} (control of {row.get('control_of')})"
    return name


# --- 1. provenance ----------------------------------------------------------------------

MATMUL_SHORT = {
    "allow_tf32": "tf32",
    "allow_bf16_reduced_precision_reduction": "bf16_reduced",
    "allow_fp16_reduced_precision_reduction": "fp16_reduced",
    "fp32_precision": "fp32_precision",
    "float32_matmul_precision": "float32_matmul_precision",
}


def _matmul_text(matmul: Mapping[str, Any] | None) -> str:
    if not matmul:
        return MISSING
    return ", ".join(f"{MATMUL_SHORT.get(k, k)}={v}" for k, v in matmul.items())


def render_provenance(recs: Sequence[Mapping[str, Any]], rows: Sequence[Row]) -> str:
    """Where and with what the records were made: one row per environment build."""
    slugs = sorted({r["device_slug"] for r in recs})
    hosts = sorted({h for h in (analysis.host_of(r) for r in recs) if h})
    stamps = sorted(r["timestamp"] for r in recs if r.get("timestamp"))
    lines = ["## Provenance", ""]
    lines.append(f"- device: {', '.join(slugs) or MISSING}")
    lines.append(f"- hosts: {', '.join(hosts) or MISSING}")
    if stamps:
        lines.append(f"- records from {stamps[0]} to {stamps[-1]} ({len(recs)} records)")
    builds: dict[tuple, dict[str, set]] = {}
    for r in recs:
        env = r.get("env") or {}
        torch = env.get("torch") or {}
        tob = (env.get("torch_op_bench") or {}).get("hash")
        build = (
            r["env_name"],
            torch.get("version"),
            torch.get("cuda"),
            env.get("python"),
            tob,
            _matmul_text(torch.get("matmul")),
        )
        if not env:
            continue  # written by the runner: no provenance of the worker
        packages = builds.setdefault(build, defaultdict(set))
        for name, value in (env.get("packages") or {}).items():
            packages[name].add(analysis.packages_text({name: value}))
    body = []
    for build in sorted(builds, key=lambda b: tuple(str(x) for x in b)):
        env_name, version, cuda, python, tob, matmul = build
        packages = "; ".join(sorted(v for vs in builds[build].values() for v in vs)) or MISSING
        body.append(
            [
                env_name,
                text(version),
                text(cuda),
                text(python),
                text(packages),
                text(tob[:12] if tob else None),
                matmul,
            ]
        )
    lines += [
        "",
        table(
            ["env", "torch", "CUDA", "Python", "packages", "torch-op-bench", "matmul"],
            body,
            [False] * 7,
        ),
        "",
    ]
    commits = sorted({b[4] for b in builds if b[4]})
    lines.append(f"torch-op-bench commit(s): {', '.join(commits) or MISSING}.")
    if len(commits) > 1:
        lines.append(
            "**The records come from different torch-op-bench commits**: a difference "
            "between environments may be the tool's, not the code under test's."
        )
    # The environment's own build: torch, CUDA, Python, matmul settings, packages;
    # not the torch-op-bench commit, flagged above.
    per_env = defaultdict(set)
    for build, packages in builds.items():
        env_name, version, cuda, python, _, matmul = build
        names = tuple(sorted(v for vs in packages.values() for v in vs))
        per_env[env_name].add((version, cuda, python, matmul, names))
    for env_name, variants in sorted(per_env.items()):
        if len(variants) > 1:
            lines.append(
                f"**Environment `{env_name}` has {len(variants)} different builds** in "
                "these records: its rows mix them."
            )
    return "\n".join(lines) + "\n"


# --- 2. the default point -------------------------------------------------------------


def _info_columns(rows: Sequence[Row]) -> list[str]:
    return [k for k in rows[0] if k.startswith(analysis.INFO_PREFIX)] if rows else []


def _parity_columns(rows: Sequence[Row]) -> list[str]:
    names = {k for row in rows for k in row if k.startswith(analysis.PARITY_PREFIX)}
    names -= {"parity_status", "parity_failed"}
    return sorted(n for n in names if any(row.get(n) is not None for row in rows))


def render_default(
    rows: Sequence[Row], point: Mapping[str, Any], baseline: str, *, title: str
) -> str:
    """One row per implementation at the default point, the baseline first and each
    control right after its original."""
    here = [row for row in rows if all(row.get(k) == v for k, v in point.items())]
    if not here:
        return f"### {title}\n\nNo records at the default point.\n"
    by_impl = {row["implementation"]: row for row in here}
    base = by_impl.get(baseline)
    base_ok = base if base is not None and base.get("status") == "ok" else None
    parity = _parity_columns(here)
    info = [c for c in _info_columns(here) if any(row.get(c) is not None for row in here)]
    header = [
        "implementation",
        "env",
        "status",
        "median ms",
        "IQR ms",
        "time / baseline",
        "tokens/s",
        "TFLOP/s",
        "peak MiB",
        "peak / baseline",
        "parity",
        *(p.removeprefix(analysis.PARITY_PREFIX) for p in parity),
        "override_ran",
        *(c.removeprefix(analysis.INFO_PREFIX) for c in info),
    ]
    numeric = [False, False, False, *[True] * 7, False, *[True] * len(parity), False]
    numeric += [False] * len(info)
    body = []
    for name in analysis.implementations(here, baseline):
        row = by_impl[name]
        body.append(
            [
                impl_label(row, here),
                text(row.get("env_name")),
                text(row.get("status")),
                sig3(row.get("median_ms")),
                sig3(row.get("iqr_ms")),
                sig3(ratio(row.get("median_ms"), base_ok and base_ok.get("median_ms"))),
                sig3(row.get("tokens_per_s")),
                sig3(row.get("tflops")),
                mib(row.get("peak_above_baseline")),
                sig3(
                    ratio(
                        row.get("peak_above_baseline"),
                        base_ok and base_ok.get("peak_above_baseline"),
                    )
                ),
                parity_text(row),
                *(sig3(row.get(p)) for p in parity),
                text(row.get("override_ran")),
                *(text(row.get(c), INFO_WIDTH) for c in info),
            ]
        )
    point_text = ", ".join(f"{k}={v}" for k, v in point.items())
    lines = [f"### {title}", "", f"At {point_text}.", ""]
    if base_ok is None:
        lines += [f"The baseline `{baseline}` has no `ok` record here: no ratios.", ""]
    lines.append(table(header, body, numeric))
    return "\n".join(lines) + "\n"


# --- 3. the sweeps --------------------------------------------------------------------


def _cell(row: Row | None, what: str) -> str:
    if row is None:
        return MISSING
    if what == "parity":
        return parity_text(row) if row.get("status") == "ok" else text(row.get("status"))
    if row.get("status") != "ok":
        return text(row.get("status"))
    if what == "median":
        return sig3(row.get("median_ms"))
    return mib(row.get("peak_above_baseline"))


def render_sweeps(
    rows: Sequence[Row], point: Mapping[str, Any], baseline: str, *, title: str
) -> str:
    """For each axis that varies around the default point: median, peak and parity,
    one row per axis value and one column per implementation."""
    names = analysis.implementations(rows, baseline)
    parts = []
    for axis, swept in analysis.sweeps(rows, point).items():
        values = analysis.axis_values(swept, axis)
        if len(values) < 2:
            continue
        by_key = {(row[axis], row["implementation"]): row for row in swept}
        for what, label in (
            ("median", "median ms"),
            ("peak", "peak above baseline, MiB"),
            ("parity", "parity"),
        ):
            body = [
                [str(value), *(_cell(by_key.get((value, name)), what) for name in names)]
                for value in values
            ]
            numbers = all(isinstance(v, (int, float)) for v in values)
            numeric = [numbers] + [what != "parity"] * len(names)
            parts += [
                f"#### {title}: {label} against {axis}",
                "",
                table([axis, *(impl_label(n, rows) for n in names)], body, numeric),
                "",
            ]
    if not parts:
        return ""
    return "\n".join(parts)


# --- 4. the environment check ------------------------------------------------------------


def environment_check(rows: Sequence[Row], axes: Sequence[str]) -> list[dict[str, Any]]:
    """For every control, against its original, at every point and mode both have ok:
    the time ratio (control / original) and the peak difference (control − original)."""
    originals = {
        (row["implementation"], row["mode"], analysis.case_of(row, axes)): row
        for row in rows
        if row.get("status") == "ok"
    }
    found = []
    for row in rows:
        if row.get("role") != "control" or row.get("status") != "ok":
            continue
        original = originals.get((row.get("control_of"), row["mode"], analysis.case_of(row, axes)))
        if original is None:
            continue
        time_ratio = ratio(row.get("median_ms"), original.get("median_ms"))
        peak_a, peak_b = row.get("peak_above_baseline"), original.get("peak_above_baseline")
        peak_diff = None if peak_a is None or peak_b is None else peak_a - peak_b
        flags = []
        low, high = TIME_RATIO_RANGE
        if time_ratio is not None and not low <= time_ratio <= high:
            flags.append("time")
        if peak_diff is not None and abs(peak_diff) > PEAK_DIFF_LIMIT:
            flags.append("peak")
        found.append(
            {
                "control": row["implementation"],
                "original": row["control_of"],
                "env_name": row.get("env_name"),
                "mode": row["mode"],
                "dtype": row.get("dtype"),
                "case": case_text(row, axes),
                "time_ratio": time_ratio,
                "peak_diff": peak_diff,
                "flags": flags,
            }
        )
    return found


def render_environment_check(rows: Sequence[Row], axes: Sequence[str]) -> str:
    checks = environment_check(rows, axes)
    lines = ["## Environment check", ""]
    if not checks:
        lines.append("No control has a record at a point where its original has one.")
        return "\n".join(lines) + "\n"
    low, high = TIME_RATIO_RANGE
    lines += [
        f"Each control against its original at the same point: the time ratio "
        f"(control / original) and the peak difference (control − original). Outside "
        f"[{low}, {high}] or beyond {PEAK_DIFF_LIMIT // MIB} MiB, the environment, not "
        f"the counterpart, makes a difference there, and a counterpart must be compared "
        f"with the control of its own environment.",
        "",
    ]
    body = [
        [
            f"{c['control']} ({c['env_name']})",
            c["original"],
            c["mode"],
            text(c["dtype"]),
            c["case"],
            sig3(c["time_ratio"]),
            mib(c["peak_diff"]),
            ", ".join(c["flags"]) or "",
        ]
        for c in checks
    ]
    lines.append(
        table(
            ["control", "original", "mode", "dtype", "case", "time ratio", "peak diff MiB", "flag"],
            body,
            [False] * 5 + [True, True, False],
        )
    )
    flagged = [c for c in checks if c["flags"]]
    lines.append("")
    if not flagged:
        lines.append("Every control is within the limits: the environments do not differ here.")
    for c in flagged:
        what = []
        if "time" in c["flags"]:
            what.append(f"time ratio {sig3(c['time_ratio'])} outside [{low}, {high}]")
        if "peak" in c["flags"]:
            what.append(f"peak difference {mib(c['peak_diff'])} MiB")
        lines.append(
            f"- **{c['control']}** against {c['original']}, {c['mode']} {c['dtype']} at "
            f"{c['case']}: {' and '.join(what)}. Here the environment `{c['env_name']}` "
            f"differs from the baseline's, not only the counterpart."
        )
    return "\n".join(lines) + "\n"


# --- 5. not measured -----------------------------------------------------------------------


def render_not_measured(rows: Sequence[Row], axes: Sequence[str]) -> str:
    missing = [row for row in rows if row.get("status") in ("error", "unavailable")]
    lines = ["## Not measured", ""]
    if not missing:
        lines.append("Every record is `ok`.")
        return "\n".join(lines) + "\n"
    groups: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in missing:
        reason = (row.get("reason") or "no reason recorded").strip().splitlines()[0]
        groups[(row["status"], reason[:200])].append(row)
    for (status, reason), group in sorted(groups.items()):
        lines.append(f"- {status}: {text(reason)}")
        for row in group:
            lines.append(
                f"  - {row['implementation']}, {row['mode']}, "
                f"{row.get('dtype')} {case_text(row, axes)}"
            )
    return "\n".join(lines) + "\n"


# --- the document -----------------------------------------------------------------------------


def _one(values: set, what: str, flag: str) -> Any:
    values = values - {None}
    if len(values) > 1:
        raise ValueError(f"the results hold several {what}s ({', '.join(sorted(values))}): {flag}")
    return next(iter(values), None)


def report(
    results_dir: str | Path,
    *,
    op: str | None = None,
    device_slug: str | None = None,
    dtype: str | None = None,
    mode: str | None = None,
    baseline: str = "torch_eager",
    command: Sequence[str] | None = None,
) -> str:
    """The markdown report of a results directory."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        recs = analysis.load(results_dir, op=op, device_slug=device_slug)
    if not recs:
        raise ValueError(f"no records under {results_dir}")
    _one({r["op"] for r in recs}, "op", "choose one with --op")
    _one({r["device_slug"] for r in recs}, "device slug", "choose one with --device-slug")
    rows = analysis.to_rows(recs)
    axes = analysis.axes_of(recs)
    point = analysis.default_point(recs)
    dtypes = [dtype] if dtype else list(dict.fromkeys(row["dtype"] for row in rows))
    modes = [mode] if mode else [m for m in MODES if any(row["mode"] == m for row in rows)]

    if command is None:
        command = ["torch-op-bench", "report", str(results_dir)]
    lines = [
        f"# Report: {recs[0]['op']} on {recs[0]['device_slug']}",
        "",
        f"Made by `{shlex.join(command)}`.",
        "",
    ]
    for warning in caught:
        lines += [f"**Warning:** {warning.message}", ""]
    lines.append(render_provenance(recs, rows))
    source = "the op's defaults" if point["source"] == "spec" else "inferred from the records"
    lines += ["## At the default point", "", f"The default point is {source}.", ""]
    for d in dtypes:
        for m in modes:
            selected = analysis.at(rows, dtype=d, mode=m)
            lines.append(render_default(selected, point["params"], baseline, title=f"{m}, {d}"))
    lines += ["## Sweeps", ""]
    swept = [
        render_sweeps(
            analysis.at(rows, dtype=d, mode=m), point["params"], baseline, title=f"{m}, {d}"
        )
        for d in dtypes
        for m in modes
    ]
    if any(swept):
        lines += [part for part in swept if part]
    else:
        lines += ["No axis varies around the default point in these records.", ""]
    scope = [row for row in rows if row["dtype"] in dtypes and row["mode"] in modes]
    lines.append(render_environment_check(scope, axes))
    lines.append(render_not_measured(scope, axes))
    return "\n".join(lines).rstrip() + "\n"
