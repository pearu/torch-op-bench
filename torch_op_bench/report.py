"""``torch-op-bench report``: the markdown tables of the write-up (R7).

Pure functions ``render_*`` turn rows (``analysis.to_rows``) into markdown;
``report`` loads a results directory and composes them. Stdlib only.
"""

from __future__ import annotations

import math
import re
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
NO_SWEEP = "No axis varies around the default point"
# How "The default point is ..." names analysis.default_point's source.
SOURCES = {
    "spec": "the op's defaults",
    "inferred": "inferred from the records",
    "--point": "set by --point",
}
# A varying info value longer than this is cut in the tables (the record has it
# whole), so one long list cannot make the table unreadable.
INFO_WIDTH = 40
# The environment check: a control within these of its original is the same build.
TIME_RATIO_RANGE = (0.95, 1.05)
PEAK_DIFF_LIMIT = 64 * MIB
# ... and its parity metrics within these of its original's: another build can move
# the error of the same code (R1-12 R1), which a parity comparison must not credit
# to the counterpart. The reduction-setting section uses the same band.
PARITY_RATIO_RANGE = analysis.PARITY_RATIO_RANGE


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
        "peak above baseline, MiB",
        "peak above baseline / baseline's",
        "time × peak, ms·MiB",
        "time × peak / baseline's",
        "parity",
        *(p.removeprefix(analysis.PARITY_PREFIX) for p in parity),
        "override_ran",
        *(c.removeprefix(analysis.INFO_PREFIX) for c in info),
    ]
    numeric = [False, False, False, *[True] * 9, False, *[True] * len(parity), False]
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
                sig3(row.get("time_x_peak")),
                sig3(ratio(row.get("time_x_peak"), base_ok and base_ok.get("time_x_peak"))),
                parity_text(row),
                *(sig3(row.get(p)) for p in parity),
                text(row.get("override_ran")),
                *(text(row.get(c), INFO_WIDTH) for c in info),
            ]
        )
    lines = [f"### {title}", "", f"At {analysis.point_text(point)}.", ""]
    if base_ok is None:
        lines += [f"The baseline `{baseline}` has no `ok` record here: no ratios.", ""]
    lines.append(table(header, body, numeric))
    return "\n".join(lines) + "\n"


# --- 3. the sweeps --------------------------------------------------------------------


def _cell(row: Row | None, what: str, base: Row | None = None) -> str:
    if row is None:
        return MISSING
    if what == "parity":
        return parity_text(row) if row.get("status") == "ok" else text(row.get("status"))
    if row.get("status") != "ok":
        return text(row.get("status"))
    if what == "median":
        return sig3(row.get("median_ms"))
    if what == "time_x_peak":
        # The product and, after it, its ratio to the baseline's at the same value.
        value = row.get("time_x_peak")
        base_value = base.get("time_x_peak") if base and base.get("status") == "ok" else None
        quotient = ratio(value, base_value)
        return sig3(value) + ("" if quotient is None else f" ({sig3(quotient)})")
    return mib(row.get("peak_above_baseline"))


def render_sweeps(
    rows: Sequence[Row], point: Mapping[str, Any], baseline: str, *, title: str
) -> str:
    """For each axis that varies around the default point: median, peak, time × peak
    (with its ratio to the baseline's) and parity, one row per axis value and one
    column per implementation."""
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
            ("time_x_peak", "time × peak, ms·MiB (/ baseline's)"),
            ("parity", "parity"),
        ):
            body = [
                [
                    str(value),
                    *(
                        _cell(by_key.get((value, name)), what, by_key.get((value, baseline)))
                        for name in names
                    ),
                ]
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


def render_per_case(
    rows: Sequence[Row], axes: Sequence[str], baseline: str, dtype: str, mode: str
) -> str:
    """The default-point table at every case present, for records whose cases do not
    form a sweep (one case, or cases that differ in more than one axis)."""
    cases = sorted({analysis.case_of(row, axes)[1:] for row in rows}, key=_case_order)
    parts = []
    for values in cases:
        here = dict(zip(axes, values, strict=True))
        title = f"{mode}, {dtype}: {analysis.point_text(here)}"
        parts.append(render_default(rows, here, baseline, title=title))
    return "\n".join(parts)


def _case_order(values: tuple) -> tuple:
    # Numbers before text, each ascending, so a case order never depends on the files.
    return tuple((0, v, "") if isinstance(v, (int, float)) else (1, 0, str(v)) for v in values)


# --- 4. the reduction setting ------------------------------------------------------------

REDUCTION_TITLE = "## Reduction setting"
# The record names a warning lists at most; the rest is counted.
NAMED_RECORDS = 10
# The op_decisions fields named in the section's cell, when the build has them: the
# PR's predicates, else the _ChunkContext capability decisions.
DECISION_FIELDS = (
    "_mm_fp32_out_available",
    "_mm_accumulates_in_fp32",
    "is_cuda",
    "use_acc_dtype",
    "forward_uses_cuda_out_dtype",
)


def record_text(record: Mapping[str, Any], axes: Sequence[str]) -> str:
    case = record.get("case") or {}
    params = case.get("params") or {}
    where = ", ".join(f"{name}={params.get(name)}" for name in axes)
    return f"{record.get('implementation')} {record.get('mode')} {case.get('dtype')} {where}"


def reduction_warnings(
    recs_default: Sequence[Mapping[str, Any]],
    recs_strict: Sequence[Mapping[str, Any]],
    axes: Sequence[str],
) -> list[str]:
    """A line per set holding a record of another setting than its own, naming them."""
    lines = []
    for label, recs, expected in (
        ("RESULTS", recs_default, "default"),
        ("RESULTS_STRICT", recs_strict, "strict"),
    ):
        odd = analysis.reduction_mismatches(recs, expected)
        if not odd:
            continue
        names = [
            f"{record_text(r, axes)} ({analysis.reduction_setting(r)})" for r in odd[:NAMED_RECORDS]
        ]
        more = f" and {len(odd) - NAMED_RECORDS} more" if len(odd) > NAMED_RECORDS else ""
        lines.append(
            f"**Warning:** {len(odd)} record(s) of {label} do not have the {expected} "
            f"setting in `env.torch.matmul`: {'; '.join(names)}{more}."
        )
    return lines


def decisions_text(decisions: Sequence[Mapping[str, Any] | None]) -> str:
    """``op_decisions`` of the default and the strict record in one cell: the named
    fields the build has, once when both agree."""

    def one(d: Mapping[str, Any] | None) -> str:
        if not d:
            return MISSING
        if "unavailable" in d:
            return f"unavailable: {d['unavailable']}"
        return ", ".join(f"{k}={d[k]}" for k in DECISION_FIELDS if k in d) or MISSING

    default, strict = (one(d) for d in decisions)
    return default if default == strict else f"default: {default}; strict: {strict}"


def _side(entry: Mapping[str, Any], which: str) -> str:
    row = entry[which]
    if row is None:
        return "absent"
    return text(row.get("status"))


def render_reduction(
    entries: Sequence[Mapping[str, Any]], point: Mapping[str, Any], dtype: str, mode: str
) -> str:
    """The table of one dtype and mode, and the reading below it."""
    title = f"### {mode}, {dtype}"
    if not entries:
        return f"{title}\n\nNo records at the default point.\n"
    metrics = sorted({name for e in entries for name in e["parity"]})
    decisions = any(any(d is not None for d in e["op_decisions"]) for e in entries)
    header = ["implementation", "median ms default", "median ms strict", "time strict / default"]
    for name in metrics:
        header += [f"{name} default", f"{name} strict", f"{name} default / strict"]
    header.append("parity moves")
    if decisions:
        header.append("op_decisions")
    numeric = [False, True, True, True, *[True] * (3 * len(metrics)), False]
    numeric += [False] * decisions
    label_rows = [
        {"implementation": e["implementation"], "role": e["role"], "control_of": e["control_of"]}
        for e in entries
    ]
    body = []
    for e in entries:
        cells = [impl_label(e["implementation"], label_rows)]
        if not e["measured"]:
            reason = f"not measured (default: {_side(e, 'default')}, strict: {_side(e, 'strict')})"
            cells += [reason, *[MISSING] * (2 + 3 * len(metrics)), MISSING]
        else:
            cells += [sig3(e["median_default"]), sig3(e["median_strict"]), sig3(e["time_ratio"])]
            for name in metrics:
                value = e["parity"].get(name)
                if value is None:
                    cells += [MISSING] * 3
                else:
                    cells += [sig3(value["default"]), sig3(value["strict"]), sig3(value["ratio"])]
            cells.append(f"yes: {', '.join(e['moves'])}" if e["moves"] else "no")
        if decisions:
            cells.append(text(decisions_text(e["op_decisions"])))
        body.append(cells)
    lines = [title, "", f"At {analysis.point_text(point)}.", "", table(header, body, numeric), ""]
    lines += reduction_reading(entries)
    return "\n".join(lines) + "\n"


def reduction_reading(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    """The reading below a table: which entries keep their strict parity under the
    defaults and which do not; and the torch LCE entries' (those recording
    ``op_decisions``) on a line of their own."""
    compared = [e for e in entries if e["measured"] and e["parity"]]
    keep = [e["implementation"] for e in compared if not e["moves"]]
    move = [e["implementation"] for e in compared if e["moves"]]

    def names(found: Sequence[str]) -> str:
        return ", ".join(f"`{n}`" for n in found) or "none"

    lines = [
        f"Under the defaults, {names(keep)} have the strict parity (fp32 accumulation or "
        f"no small-M GEMMs); {names(move)} do not.",
    ]
    unmeasured = [e["implementation"] for e in entries if e not in compared]
    if unmeasured:
        lines[-1] += f" Not compared (not measured in both, or no parity): {names(unmeasured)}."
    lce = [e for e in compared if any(d is not None for d in e["op_decisions"])]
    for e in lce:
        verdict = "equals" if not e["moves"] else f"does not equal ({', '.join(e['moves'])})"
        lines.append("")
        lines.append(
            f"torch LCE: `{e['implementation']}`'s default parity {verdict} its strict parity."
        )
    return lines


def render_reduction_section(
    recs_default: Sequence[Mapping[str, Any]],
    recs_strict: Sequence[Mapping[str, Any]],
    point: Mapping[str, Any],
    baseline: str,
    *,
    dtypes: Sequence[str],
    modes: Sequence[str],
    strict_dir: str | Path,
) -> str:
    axes = analysis.axes_of([*recs_default, *recs_strict])
    low, high = PARITY_RATIO_RANGE
    lines = [
        REDUCTION_TITLE,
        "",
        f"The default run against the strict one (`{strict_dir}`: no reduced-precision "
        f"reduction in bf16 and fp16 GEMMs) at the default point: the time ratio "
        f"(strict / default) and, per parity metric, the ratio default / strict. A parity "
        f"ratio outside [{low}, {high}] (the environment check's band) means the "
        f"setting moves that entry's parity.",
        "",
    ]
    warned = reduction_warnings(recs_default, recs_strict, axes)
    for line in warned:
        lines += [line, ""]
    entries = analysis.reduction_rows(recs_default, recs_strict, point, baseline=baseline)
    for d in dtypes:
        for m in modes:
            selected = [e for e in entries if e["dtype"] == d and e["mode"] == m]
            if selected:
                lines.append(render_reduction(selected, point, d, m))
    if not any(e["dtype"] in dtypes and e["mode"] in modes for e in entries):
        lines.append("No records at the default point in either set.")
    return "\n".join(lines) + "\n"


# --- 5. the environment check ------------------------------------------------------------


def _parity_metrics(row: Row) -> dict[str, Any]:
    return {
        k.removeprefix(analysis.PARITY_PREFIX): v
        for k, v in row.items()
        if k.startswith(analysis.PARITY_PREFIX)
        and k not in ("parity_status", "parity_failed")
        and v is not None
    }


def parity_ratios(control: Row, original: Row) -> dict[str, dict[str, Any]]:
    """For each parity metric of either side: both values, their ratio (control /
    original) and, when one side lacks it, which side has it."""
    a, b = _parity_metrics(control), _parity_metrics(original)
    found = {}
    for name in sorted(set(a) | set(b)):
        c, o = a.get(name), b.get(name)
        if c is None or o is None:
            side = "control only" if o is None else "original only"
            found[name] = {"control": c, "original": o, "ratio": None, "side": side}
            continue
        value = 1.0 if c == o else ratio(c, o)
        found[name] = {"control": c, "original": o, "ratio": value, "side": None}
    return found


def environment_check(rows: Sequence[Row], axes: Sequence[str]) -> list[dict[str, Any]]:
    """For every control, against its original, at every point and mode both have a
    record: when both are ok, the time ratio (control / original), the peak
    difference (control − original) and the ratio of each parity metric (control /
    original); when only one is ok (``measured`` is "one"), the flag ``status``: the
    environment makes that difference (R1-30 F2); when neither is ("neither"), the
    pair only."""
    by_key = {
        (row["implementation"], row["mode"], analysis.case_of(row, axes)): row for row in rows
    }
    found = []
    for row in rows:
        if row.get("role") != "control":
            continue
        original = by_key.get((row.get("control_of"), row["mode"], analysis.case_of(row, axes)))
        if original is None:
            continue
        check = {
            "control": row["implementation"],
            "original": row["control_of"],
            "env_name": row.get("env_name"),
            "mode": row["mode"],
            "dtype": row.get("dtype"),
            "case": case_text(row, axes),
            "status": (row.get("status"), original.get("status")),
            "measured": "both",
            "time_ratio": None,
            "peak_diff": None,
            "parity": {},
            "flags": [],
        }
        found.append(check)
        both = [r.get("status") == "ok" for r in (row, original)]
        if not all(both):
            check["measured"] = "one" if any(both) else "neither"
            check["flags"] = ["status"] if any(both) else []
            check["reason"] = (row if not both[0] else original).get("reason")
            continue
        time_ratio = ratio(row.get("median_ms"), original.get("median_ms"))
        peak_a, peak_b = row.get("peak_above_baseline"), original.get("peak_above_baseline")
        peak_diff = None if peak_a is None or peak_b is None else peak_a - peak_b
        parity = parity_ratios(row, original)
        flags = []
        low, high = TIME_RATIO_RANGE
        if time_ratio is not None and not low <= time_ratio <= high:
            flags.append("time")
        if peak_diff is not None and abs(peak_diff) > PEAK_DIFF_LIMIT:
            flags.append("peak")
        low, high = PARITY_RATIO_RANGE
        for name, value in parity.items():
            if value["ratio"] is not None and not low <= value["ratio"] <= high:
                flags.append(name)
            elif value["ratio"] is None and value["side"] is None:  # 0 against nonzero
                flags.append(name)
        check.update(time_ratio=time_ratio, peak_diff=peak_diff, parity=parity, flags=flags)
    return found


def _parity_cell(value: Mapping[str, Any] | None) -> str:
    if value is None:
        return MISSING
    if value["side"]:
        return value["side"]
    if value["ratio"] is None:  # zero against nonzero: no ratio, both values
        return f"{sig3(value['control'])} / {sig3(value['original'])}"
    return sig3(value["ratio"])


def render_environment_check(rows: Sequence[Row], axes: Sequence[str]) -> str:
    pairs = environment_check(rows, axes)
    checks = [c for c in pairs if c["measured"] != "neither"]
    lines = ["## Environment check", ""]
    if not pairs:
        lines.append("No control has a record at a point where its original has one.")
        return "\n".join(lines) + "\n"
    low, high = TIME_RATIO_RANGE
    p_low, p_high = PARITY_RATIO_RANGE
    lines += [
        f"Each control against its original at the same point: the time ratio "
        f"(control / original), the peak difference (control − original) and the "
        f"ratio of each parity metric (control / original). A time ratio outside "
        f"[{low}, {high}], a peak difference beyond {PEAK_DIFF_LIMIT // MIB} MiB or a "
        f"parity ratio outside [{p_low}, {p_high}] means the environment, not the "
        f"counterpart, makes a difference there, and a counterpart must be compared "
        f"with the control of its own environment; so does a pair where only one side "
        f"is `ok` (flag `status`, its statuses, control / original, in the time ratio's "
        f"place).",
        "",
    ]
    count = {kind: sum(c["measured"] == kind for c in pairs) for kind in ("both", "one", "neither")}
    lines += [
        f"{count['both']} pairs compared; {count['one']} with one side not measured"
        f"{' (flagged below)' if count['one'] else ''}; {count['neither']} with neither "
        f"side measured.",
        "",
    ]
    if not checks:
        return "\n".join(lines) + "\n"
    metrics = sorted({name for c in checks for name in c["parity"]})
    body = [
        [
            f"{c['control']} ({c['env_name']})",
            c["original"],
            c["mode"],
            text(c["dtype"]),
            c["case"],
            sig3(c["time_ratio"]) if c["measured"] == "both" else " / ".join(c["status"]),
            mib(c["peak_diff"]),
            *(_parity_cell(c["parity"].get(name)) for name in metrics),
            ", ".join(c["flags"]) or "",
        ]
        for c in checks
    ]
    header = ["control", "original", "mode", "dtype", "case", "time ratio", "peak diff MiB"]
    header += [f"{name} ratio" for name in metrics] + ["flag"]
    lines.append(table(header, body, [False] * 5 + [True] * (2 + len(metrics)) + [False]))
    lines.append("")
    noted = [c for c in checks if c["flags"] or any(v["side"] for v in c["parity"].values())]
    if not noted:
        lines.append("Every control is within the limits: the environments do not differ here.")
    for c in noted:
        if c["measured"] == "one":
            control, original = (
                f"`{status}`" + ("" if status == "ok" else f" ({cause(status, c['reason'])[0]})")
                for status in c["status"]
            )
            lines.append(
                f"- **{c['control']}** (`{c['env_name']}`) against {c['original']}, "
                f"{c['mode']} {c['dtype']} at {c['case']}: control {control}, original "
                f"{original}: the environment makes a difference there."
            )
            continue
        what = []
        if "time" in c["flags"]:
            what.append(f"time ratio {sig3(c['time_ratio'])} outside [{low}, {high}]")
        if "peak" in c["flags"]:
            what.append(f"peak difference {mib(c['peak_diff'])} MiB")
        for name, value in c["parity"].items():
            if name in c["flags"]:
                what.append(
                    f"{name} {sig3(value['control'])} against {sig3(value['original'])} "
                    f"(ratio {sig3(value['ratio'])}) outside [{p_low}, {p_high}]"
                )
            elif value["side"]:
                what.append(f"{name} on the {value['side'].removesuffix(' only')} only")
        sentence = (
            f"- **{c['control']}** against {c['original']}, {c['mode']} {c['dtype']} at "
            f"{c['case']}: {' and '.join(what)}."
        )
        if c["flags"]:
            sentence += (
                f" Here the environment `{c['env_name']}` differs from the baseline's, "
                f"not only the counterpart."
            )
        lines.append(sentence)
    return "\n".join(lines) + "\n"


# --- 6. not measured -----------------------------------------------------------------------


REASON_WIDTH = 200
# An error's cause ends where its first number starts: what follows (sizes, free
# memory, indices) differs from record to record of one cause.
FIRST_NUMBER = re.compile(r"\d")
OOM_SPECIFICS = re.compile(
    r"Tried to allocate (?P<tried>[\d.]+ \w+)\..*? of which (?P<free>[\d.]+ \w+) is free"
)


def cause(status: str, reason: str | None) -> tuple[str, str]:
    """The group of a not-measured record: its cause, and what of its reason's first
    line the cause leaves out. For an ``error``, the cause is the message up to its
    first number, with the unfinished sentence there dropped ("OutOfMemoryError: CUDA
    out of memory. Tried to allocate 4.00 GiB. …" is "OutOfMemoryError: CUDA out of
    memory."); for ``unavailable``, whose reasons are written to be stable, the whole
    first line."""
    line = (reason or "no reason recorded").strip().splitlines()[0]
    found = FIRST_NUMBER.search(line) if status == "error" else None
    if found is None:
        return line[:REASON_WIDTH], ""
    head = line[: found.start()]
    # "…: CUDA out of memory. Tried to allocate " -> "…: CUDA out of memory."
    sentence = head.rfind(". ")
    key = head[: sentence + 1] if sentence != -1 else head.rstrip()
    name, colon, message = key.partition(":")
    if not (message if colon and " " not in name else key).strip(" :"):
        # The message starts with a number: no cause to cut it at.
        return line[:REASON_WIDTH], ""
    oom = OOM_SPECIFICS.search(line)
    if oom:
        rest = f"tried to allocate {oom['tried']}, {oom['free']} free"
    else:
        rest = line[len(key) :].strip()
    return key[:REASON_WIDTH], rest


def render_not_measured(rows: Sequence[Row], axes: Sequence[str]) -> str:
    missing = [row for row in rows if row.get("status") in ("error", "unavailable")]
    lines = ["## Not measured", ""]
    if not missing:
        lines.append("Every record is `ok`.")
        return "\n".join(lines) + "\n"
    groups: dict[tuple[str, str], list[tuple[Row, str]]] = defaultdict(list)
    for row in missing:
        key, rest = cause(row["status"], row.get("reason"))
        groups[(row["status"], key)].append((row, rest))
    for (status, key), group in sorted(groups.items()):
        lines.append(f"- {status}: {text(key)} ({len(group)})")
        for row, rest in group:
            where = f"{row['implementation']}, {row['mode']}, {row.get('dtype')}"
            where += f" {case_text(row, axes)}" + (f": {text(rest, REASON_WIDTH)}" if rest else "")
            lines.append(f"  - {where}")
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
    point: Mapping[str, str] | None = None,
    command: Sequence[str] | None = None,
    strict: str | Path | None = None,
) -> str:
    """The markdown report of a results directory; with ``strict``, a strict run's
    results directory, the "Reduction setting" section too."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        recs = analysis.load(results_dir, op=op, device_slug=device_slug)
        if not recs:
            raise ValueError(f"no records under {results_dir}")
        op_name = _one({r["op"] for r in recs}, "op", "choose one with --op")
        slug = _one(
            {r["device_slug"] for r in recs}, "device slug", "choose one with --device-slug"
        )
        # The strict set of the same op and device as RESULTS.
        recs_strict = [] if strict is None else analysis.load(strict, op=op_name, device_slug=slug)
    rows = analysis.to_rows(recs)
    axes = analysis.axes_of(recs)
    point = analysis.default_point(recs, point=point)
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
    source = SOURCES.get(point["source"], point["source"])
    lines += ["## At the default point", "", f"The default point is {source}.", ""]
    for d in dtypes:
        for m in modes:
            selected = analysis.at(rows, dtype=d, mode=m)
            lines.append(render_default(selected, point["params"], baseline, title=f"{m}, {d}"))
    lines += ["## Sweeps", ""]
    # Per (dtype, mode), as plot decides: the sweeps around the point, or, when no
    # axis varies there, the default-point table at every case present.
    pairs = []
    for d in dtypes:
        for m in modes:
            selected = analysis.at(rows, dtype=d, mode=m)
            if selected:
                swept = render_sweeps(selected, point["params"], baseline, title=f"{m}, {d}")
                pairs.append((d, m, selected, swept))
    if not any(swept for *_, swept in pairs):
        # No pair has a sweep: one line for all of them.
        lines += [f"{NO_SWEEP} in these records: one table per case instead.", ""]
        lines += [render_per_case(sel, axes, baseline, d, m) for d, m, sel, _ in pairs]
    else:
        for d, m, selected, swept in pairs:
            if swept:
                lines.append(swept)
                continue
            lines += [f"**{m}, {d}:** {NO_SWEEP.lower()}: one table per case instead.", ""]
            lines.append(render_per_case(selected, axes, baseline, d, m))
    if strict is not None:
        lines.append(
            render_reduction_section(
                recs,
                recs_strict,
                point["params"],
                baseline,
                dtypes=dtypes,
                modes=modes,
                strict_dir=strict,
            )
        )
    scope = [row for row in rows if row["dtype"] in dtypes and row["mode"] in modes]
    lines.append(render_environment_check(scope, axes))
    lines.append(render_not_measured(scope, axes))
    return "\n".join(lines).rstrip() + "\n"
