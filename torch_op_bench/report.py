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
# The widest a table may be (R1-30 F6): wider, a reader loses the row.
MAX_COLUMNS = 16
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


def parity_text(row: Row, footnotes: Footnotes | None = None) -> str:
    """The parity status; for a ``fail``, the metrics, each with the mark of the note
    that explains it (``fail: a¹, b``: ``b`` is unexplained) when ``footnotes``
    collects them (R1-38 R2)."""
    status = row.get("parity_status")
    if status == "fail" and row.get("parity_failed"):
        marks = footnotes.marks(row) if footnotes is not None else {}
        failed = analysis.failed_metrics(row)
        return "fail: " + ", ".join(metric + marks.get(metric, "") for metric in failed)
    return text(status)


SUPERSCRIPTS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def note_text(note: Any) -> str:
    """A parity note with what it covers: the implementation and its metrics."""
    metrics = ", ".join(note.metrics) if note.metrics is not None else "any metric"
    return f"`{note.implementation}`, {metrics}: {note.text}"


class Footnotes:
    """The parity notes of one table, numbered in the order they are first marked."""

    def __init__(self, notes: Sequence[Any], axes: Sequence[str]) -> None:
        self.notes, self.axes = notes, axes
        self.marked: list[Any] = []

    def marks(self, row: Row) -> dict[str, str]:
        """Per metric ``row`` failed on that a note explains, the note's mark."""
        found = {}
        for metric, note in analysis.explained(row, self.notes, self.axes).items():
            if note is None:
                continue
            if note not in self.marked:
                self.marked.append(note)
            found[metric] = self.mark(note)
        return found

    def mark(self, note: Any) -> str:
        return str(self.marked.index(note) + 1).translate(SUPERSCRIPTS)

    def lines(self) -> list[str]:
        """The footnotes, each a paragraph, under the table."""
        return [
            line for note in self.marked for line in ("", f"{self.mark(note)} {note_text(note)}")
        ]


def override_text(row: Row) -> str:
    """Whether an override ran: ``kernel`` (its dispatch and its own kernel),
    ``dispatch only`` (the dispatch, no kernel of its own), ``no``; ``—`` for an
    entry that records neither. ``dispatch`` alone for a record older than
    ``override_kernel_ran``."""
    ran, kernel = row.get("override_ran"), row.get("override_kernel_ran")
    if ran is None:
        return MISSING
    if not ran:
        return "no"
    if kernel is None:
        return "dispatch"
    return "kernel" if kernel else "dispatch only"


def dispatch_only_note(rows: Sequence[Row], labels: Sequence[Row]) -> list[str]:
    """One sentence under a default-point table when an entry's override dispatched
    but ran no kernel of its own: its time is not the kernel's (R1-30 N1)."""
    names = [impl_label(row, labels) for row in rows if override_text(row) == "dispatch only"]
    if not names:
        return []
    mode = rows[0].get("mode")
    who = ", ".join(f"`{name}`" for name in names)
    verb = "dispatches" if len(names) == 1 else "dispatch"
    whose = "its" if len(names) == 1 else "their"
    return [
        "",
        f"{who} in {mode} {verb} to the override, which runs no kernel of its own on "
        f"this path (`override_kernel_ran` false): {whose} {mode} time is not the "
        "override kernel's.",
    ]


def impl_label(row_or_name: Row | str, rows: Sequence[Row]) -> str:
    name = row_or_name if isinstance(row_or_name, str) else row_or_name["implementation"]
    for row in rows:
        if row["implementation"] == name and row.get("role") == "control":
            return f"{name} (control of {row.get('control_of')})"
    return name


# --- 0a. how to read this report (Task I1-23 Part A) -------------------------------------

READ_TITLE = "## How to read this report"

# The tool's definitions, the same for every op, written once: (term, definition). The
# fields in braces are filled from the records and the bands (``tool_definitions``).
DEFINITIONS = [
    (
        "implementation",
        "an entry: one way of computing the op, as the op declares it (the Summary "
        "describes each). `<name> (control of <original>)` is a control: the original's "
        "code run in another environment, so that what the environment does can be told "
        "from what the counterpart does. `<name>[<parameter>=<value>]` is a variant: the "
        "entry with another value of one of its own parameters, measured at the default "
        'point for "Time at equal peak" only.',
    ),
    (
        "env",
        "the environment (interpreter, torch build, packages) the entry's worker ran in; "
        '"Provenance" lists each.',
    ),
    (
        "mode, dtype, case",
        "`fwd`: one call of the forward; `fwd_bwd`: one call of the forward and the "
        "backward, the gradients of the inputs included. The dtype of the inputs. A case "
        "is a value of each of the op's axes (defined below), written `axis=value, ...`.",
    ),
    (
        "median ms (IQR)",
        "the median and, in parentheses, the interquartile range of {repeat} timed calls, "
        "in milliseconds, after {warmup} untimed calls ({settings}): one call at a time, "
        "each timed on the device by a pair of CUDA events (on the CPU, by the wall "
        "clock), the gradients of the inputs cleared before each call, outside the timed "
        "region. The status instead when the record is not `ok`.",
    ),
    (
        "time / baseline",
        "the entry's median over the baseline `{baseline}`'s, at the same point, mode and dtype.",
    ),
    (
        "throughput",
        "a column per unit of work the op declares (defined below), headed by the unit's "
        "rate: the work of one call, as the op counts it, over the median time.",
    ),
    (
        "peak above baseline, MiB",
        "the peak memory allocated during one call minus what was allocated when the call "
        "started (the inputs, and the cuBLAS workspaces where the build counts them), from "
        "the device allocator's statistics; `—` on the CPU. \"/ baseline's\": its ratio to "
        "the baseline's.",
    ),
    (
        "time × peak, ms·MiB",
        "the median times the peak above baseline: one number that trades time against "
        "memory, so that an entry faster by holding more memory, or smaller by taking "
        "longer, shows it. \"/ baseline's\": its ratio to the baseline's.",
    ),
    (
        "parity",
        "the entry's outputs against the op's reference on the same inputs: `ok` when "
        "every metric with a tolerance is within it; `fail: a, b` with the metrics that "
        "are not, a numbered superscript on a metric one of the op's parity notes explains (the "
        "note under the table); `skipped` when the reference was not run, with the "
        "reason. A metric's column holds the record's value; the metrics and their "
        "tolerances are the op's (below).",
    ),
    (
        "override",
        "whether a registered override of the op ran: `kernel` (it dispatched and its own "
        "kernel ran), `dispatch only` (it dispatched and ran no kernel of its own), `no`; "
        "`—` for an entry that records neither.",
    ),
    (
        "info",
        "Entry details: the `info` fields whose values differ between records, one per "
        "line, so that a difference cannot hide there.",
    ),
    (
        "sweeps",
        "for each axis varied around the default point, the other axes at the default "
        "point: a row per value of the axis, a column per entry; the cell is the median "
        "ms, the peak above baseline, time × peak with its ratio to the baseline's in "
        "parentheses, or parity; `—` without a record.",
    ),
    (
        "time at equal peak",
        "memory budgets: the baseline's peak above baseline at the default point times "
        "{budgets}, the same for every entry (headed `≤ 1× (… MiB)`). A cell is the "
        "median of the entry's default or of one of its variants, the one with the "
        "largest peak above baseline within the budget, what produced it in small print; "
        "`—` when none fits, with the entry's lowest measured peak (`— lowest … MiB`). "
        '"default: median ms, peak MiB": the entry\'s own default.',
    ),
    (
        "reduction setting",
        "the default run against a strict one (`run --matmul-reduction strict`: no "
        "reduced-precision reduction in bf16 and fp16 GEMMs) at the default point: the "
        "median of each and their ratio (strict / default); each parity metric's values "
        '(default, strict) and their ratio (default / strict); "parity moves" when that '
        "ratio is outside [{parity_low}, {parity_high}]; `op_decisions` the decisions "
        "the torch LCE path records.",
    ),
    (
        "environment check",
        "each control against its original at the same point: the time ratio (control / "
        "original), the peak difference (control − original, MiB) and each parity "
        "metric's ratio (`<metric> ratio`). The bands: a time ratio within [{time_low}, "
        "{time_high}], a peak difference within {peak_limit} MiB, a parity ratio within "
        "[{parity_low}, {parity_high}]; outside one, the `flag` column names it: there "
        "the environment, not the counterpart, makes a difference.",
    ),
    (
        "not measured",
        "`unavailable`: the entry could not run the point here, with the reason (a "
        "package or an API missing, a device the op does not support, or the point did "
        "not fit in the device's memory); `error`: a failure nobody foresaw, with its "
        "message.",
    ),
    (
        "provenance",
        "per environment build: torch, CUDA, Python, the packages measured (a version, or "
        "the first 12 characters of a git install's commit), the torch-op-bench commit, "
        "and the matmul settings (`tf32`, `bf16_reduced`: reduced-precision reductions "
        "allowed in bf16 GEMMs, ...).",
    ),
]
# Every header a table of the report has, to the term that defines it (fullmatch, in
# order); the op's own headers (its rates, metrics and axes, and the entries' names)
# are matched in ``header_term``.
HEADER_TERMS = [
    (r"implementation", "implementation"),
    (r"env", "env"),
    (r"mode|dtype|case", "mode, dtype, case"),
    (r"median ms \(IQR\)", "median ms (IQR)"),
    (r"time / baseline", "time / baseline"),
    (r"peak above baseline, MiB|peak above baseline / baseline's", "peak above baseline, MiB"),
    (r"time × peak, ms·MiB|time × peak / baseline's", "time × peak, ms·MiB"),
    (r"parity", "parity"),
    (r"override", "override"),
    (r"info", "info"),
    (
        r"≤ [\d.]+× \([\d.]+ MiB\)|default: median ms, peak MiB|≤ 1× budget: median ms",
        "time at equal peak",
    ),
    (
        r"median ms default|median ms strict|time strict / default|parity moves"
        r"|op_decisions|.+ default, strict|.+ default / strict",
        "reduction setting",
    ),
    (r"control|original|time ratio|peak diff MiB|flag|.+ ratio", "environment check"),
    (r"torch|CUDA|Python|packages|torch-op-bench|matmul", "provenance"),
]


def header_term(
    header: str,
    *,
    rates: Sequence[str] = (),
    metrics: Sequence[str] = (),
    axes: Sequence[str] = (),
    entries: Sequence[str] = (),
) -> str | None:
    """The term of ``DEFINITIONS`` that defines a table header, or, for a header the
    op declares, ``throughput`` (its ``rates``), ``parity`` (its ``metrics``),
    ``axis`` (its ``axes``) or ``implementation`` (an entry's label, a column of the
    sweeps); ``None`` when nothing defines it."""
    if header in rates:
        return "throughput"
    if header in metrics:
        return "parity"
    if header in axes:
        return "axis"
    if header in entries:
        return "implementation"
    for pattern, term in HEADER_TERMS:
        if re.fullmatch(pattern, header):
            return term
    return None


def _values_text(recs: Sequence[Mapping[str, Any]], name: str) -> str:
    """The recorded values of setting ``name``, ``" or "``-joined when they vary."""
    found = sorted(
        {str((r.get("settings") or {}).get(name)) for r in recs if r.get("status") == "ok"}
    )
    return " or ".join(found) or MISSING


def tool_definitions(recs: Sequence[Mapping[str, Any]], baseline: str) -> list[tuple[str, str]]:
    """``DEFINITIONS`` with the settings as recorded and the bands filled in."""
    flush = {(r.get("settings") or {}).get("flush_l2") for r in recs if r.get("status") == "ok"}
    if flush == {False}:
        cache = "the L2 cache not flushed between calls"
    elif flush == {True}:
        cache = "the L2 cache flushed before each call"
    else:
        cache = "the L2 cache flushed in some records (`settings.flush_l2`)"
    repeat, warmup = _values_text(recs, "repeat"), _values_text(recs, "warmup")
    fields = {
        "repeat": repeat,
        "warmup": warmup,
        "settings": f"`settings.repeat` {repeat} and `settings.warmup` {warmup} as recorded; "
        + cache,
        "baseline": baseline,
        "budgets": ", ".join(f"{f:g}×" for f in analysis.EQUAL_PEAK_BUDGETS),
        "parity_low": PARITY_RATIO_RANGE[0],
        "parity_high": PARITY_RATIO_RANGE[1],
        "time_low": TIME_RATIO_RANGE[0],
        "time_high": TIME_RATIO_RANGE[1],
        "peak_limit": PEAK_DIFF_LIMIT // MIB,
    }
    return [(term, definition.format(**fields)) for term, definition in DEFINITIONS]


def render_definitions(
    recs: Sequence[Mapping[str, Any]],
    baseline: str,
    spec: Any,
    op_ref: str,
    work_keys: Sequence[str] = (),
) -> str:
    """The section "How to read this report": the tool's definitions, then the op's
    (its units of work, parity metrics and axes), read through its spec;
    ``work_keys`` are the work keys of the records, named when the op does not
    import here."""
    lines = [
        READ_TITLE,
        "",
        "What each column and section of this report measures. The tool's definitions, "
        "the same for every op:",
        "",
    ]
    lines += [f"- **{term}**: {text_}" for term, text_ in tool_definitions(recs, baseline)]
    lines.append("")
    if spec is None:
        lines.append(
            f"The op `{op_ref}` does not import here: its definitions are not shown"
            + (
                f"; the throughput columns ({', '.join(f'{k}/s' for k in work_keys)}) are "
                "the records' work over the median, in units the op defines."
                if work_keys
                else "."
            )
        )
        return "\n".join(lines) + "\n"
    lines += [f"The op `{spec.name}`'s definitions, as its author declares them:", ""]
    if spec.work_units:
        lines.append("- **Units of work** (the throughput columns):")
        for unit in spec.work_units:
            scale = "" if unit.scale == 1 else f", in units of {unit.scale:g}"
            lines.append(
                f"  - **{unit.rate_label}**: {unit.label} per second{scale}; {unit.label}: "
                f"{unit.definition}."
            )
    else:
        lines.append("- **Units of work**: none declared, so no throughput column.")
    lines.append("- **Parity metrics** (the parity columns), each with its tolerance:")
    for name, definition in spec.metric_definitions.items():
        tolerance = spec.tolerances.get(name)
        check = "recorded, not checked" if tolerance is None else f"tolerance {tolerance:g}"
        lines.append(f"  - **{name}**: {definition}; {check}.")
    if spec.tolerance_basis:
        lines.append(f"  {spec.tolerance_basis}")
    lines.append("- **Axes** (the case):")
    default = spec.default_point
    for axis in spec.axes:
        values = ", ".join(str(v) for v in axis.values)
        lines.append(
            f"  - **{axis.name}**: {axis.description or 'not described'}; values {values}; "
            f"default {default[axis.name]}."
        )
    return "\n".join(lines) + "\n"


def axes_line(spec: Any, point: Mapping[str, Any]) -> list[str]:
    """Under the default point: each axis' value with its description."""
    if spec is None:
        return []
    described = {axis.name: axis.description for axis in spec.axes if axis.description}
    parts = [
        f"{name}={value}" + (f" ({described[name]})" if name in described else "")
        for name, value in point.items()
    ]
    return [f"There: {'; '.join(parts)}.", ""]


# --- 0b. the summary (Task I1-23 Part B) ----------------------------------------------------

SUMMARY_TITLE = "## Summary"
MODE_WORDS = {"fwd": "forward only", "fwd_bwd": "forward and backward"}
# A ratio over a sweep "stays" when within this of its value at the default point.
SWEEP_BAND = 0.10
# Words no generated sentence may hold: the tool states numbers, the author judges
# (inbox 14, 15, 17). Tested.
FORBIDDEN = ("fastest", "slowest", "best", "worst", "wins", "optimum", "beats")
FINDINGS_AUTHOR = (
    "### Findings (written by the author of this result set; every statement names a "
    "number in the sections below)"
)
FINDINGS_GENERATED = (
    "Generated from the records by fixed templates, not written by an author (`report "
    "--summary FILE` prints an author's findings here instead): each sentence is a value, "
    "or a ratio to the baseline or to a memory budget, the entries in the op's order."
)


def _x(value: float | None) -> str:
    return f"{sig3(value)}×"


def _names(names: Sequence[str]) -> str:
    quoted = [f"`{n}`" for n in names]
    if len(quoted) < 3:
        return " and ".join(quoted)
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]


def where_text(row: Row, point: Mapping[str, Any]) -> str:
    """Where a row is, as the axes in which it differs from ``point``."""
    differ = [f"{k}={row.get(k)}" for k, v in point.items() if row.get(k) != v]
    return "the default point" if not differ else ", ".join(differ)


def _description(spec: Any, name: str) -> str:
    if spec is None:
        return ""
    try:
        return spec.implementation(name).description
    except KeyError:
        return ""


def _gib(value: int) -> str:
    return f"{value / 1024**3:.1f} GiB"


def gpus(recs: Sequence[Mapping[str, Any]]) -> list[str]:
    """Each GPU the records ran on, with its memory, from ``env.torch.devices``."""
    found = []
    for record in recs:
        device = str((record.get("case") or {}).get("device") or "")
        if not device.startswith("cuda"):
            continue
        index = int(device.partition(":")[2] or 0)
        for gpu in analysis._get(record, "env", "torch", "devices") or []:
            if gpu.get("index") == index:
                name = f"{gpu.get('name')}, {_gib(gpu.get('total_memory') or 0)}"
                if name not in found:
                    found.append(name)
    return found


def render_compared(
    recs: Sequence[Mapping[str, Any]],
    rows: Sequence[Row],
    variant_rows: Sequence[Row],
    point: Mapping[str, Any],
    baseline: str,
    spec: Any,
    *,
    dtypes: Sequence[str],
    modes: Sequence[str],
    strict: tuple[str, int] | None,
) -> list[str]:
    """What was compared: the op, the entries, the modes, the sweep and the variants."""
    op = recs[0].get("op")
    lines = ["### What was compared", ""]
    if spec is not None and spec.description:
        lines.append(f"The op `{op}`: {spec.description}.")
    else:
        lines.append(f"The op `{op}` (not described here: the op does not import).")
    lines += ["", f"The entries, in the op's order, against the baseline `{baseline}`:", ""]
    for name in analysis.implementations(rows, baseline):
        row = next(r for r in rows if r["implementation"] == name)
        what = _description(spec, name) or "not described"
        role = "the baseline, " if name == baseline else ""
        role += "a control, " if row.get("role") == "control" else ""
        runs = [m for m in modes if m in _modes_of(spec, name, rows)]
        only = f", {' and '.join(f'`{m}`' for m in runs)} only" if runs != list(modes) else ""
        lines.append(f"- `{name}` ({role}in `{row.get('env_name')}`{only}): {what}.")
    words = " and ".join(f"{MODE_WORDS.get(m, m)} (`{m}`)" for m in modes)
    lines += ["", f"Modes: {words}; dtype {', '.join(dtypes)}."]
    swept = {
        axis: analysis.axis_values(found, axis)
        for axis, found in analysis.sweeps(rows, point).items()
    }
    axes = []
    for name, value in point.items():
        described = ""
        if spec is not None:
            described = next((a.description for a in spec.axes if a.name == name), "")
        values = ", ".join(str(v) for v in swept.get(name) or [value])
        axes.append(f"{name}" + (f" ({described})" if described else "") + f": {values}")
    points = {tuple(r.get(name) for name in point) for r in rows}
    lines += [
        "",
        f"The sweep varies one axis at a time around the default point "
        f"{analysis.point_text(point)}: {'; '.join(axes)}. That is {len(points)} "
        f"point{'' if len(points) == 1 else 's'} and {len(recs)} records"
        + (f", {len(variant_rows)} of them of variants" if variant_rows else "")
        + ".",
    ]
    if strict is not None:
        lines.append(
            f"The reduction setting is compared at the default point, with the strict "
            f"run's {strict[1]} records in `{strict[0]}`."
        )
    if variant_rows:
        by_base: dict[str, list[str]] = {}
        for row in variant_rows:
            texts = by_base.setdefault(row["base"], [])
            if row.get("params") and row["params"] not in texts:
                texts.append(row["params"])
        listed = "; ".join(f"`{base}` with {', '.join(p)}" for base, p in by_base.items())
        lines.append(
            f'The variants, at the default point for "Time at equal peak" only (each '
            f"chunked entry's time within a memory budget): {listed}."
        )
    return lines + [""]


def _modes_of(spec: Any, name: str, rows: Sequence[Row]) -> set[str]:
    """The modes an entry runs in: the spec's (``Implementation.modes``) when the op
    imports and declares the entry, else those its records hold."""
    if spec is not None:
        try:
            return set(spec.implementation(name).modes)
        except KeyError:
            pass
    return {r["mode"] for r in rows if r["implementation"] == name}


def render_on_what(
    recs: Sequence[Mapping[str, Any]], sets: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[str]:
    """On what: the GPU, the hosts, the torch builds and counterparts per environment,
    the dates."""
    every = [r for found in sets.values() for r in found] or list(recs)
    lines = ["### On what", ""]
    on_gpu = any(str(r["case"].get("device", "")).startswith("cuda") for r in every)
    found = "; ".join(gpus(every)) or ("not recorded" if on_gpu else "none (the CPU)")
    lines.append(f"- GPU: {found}")
    hosts = sorted({h for h in (analysis.host_of(r) for r in every) if h})
    lines.append(f"- host: {', '.join(hosts) or MISSING}")
    torches: dict[str, list[str]] = defaultdict(list)
    packages: dict[str, set[str]] = defaultdict(set)
    for record in every:
        env = record.get("env") or {}
        version = (env.get("torch") or {}).get("version")
        if version and record["env_name"] not in torches[version]:
            torches[version].append(record["env_name"])
        text_ = analysis.packages_text(env.get("packages"))
        if text_:
            packages[record["env_name"]].update(text_.split(";"))
    lines.append(
        "- torch: "
        + "; ".join(
            f"{version} in {_names(sorted(envs))}" for version, envs in sorted(torches.items())
        )
    )
    if packages:
        lines.append(
            "- packages: "
            + "; ".join(
                f"{', '.join(sorted(found))} in `{env}`" for env, found in sorted(packages.items())
            )
        )
    stamps = sorted(r["timestamp"] for r in every if r.get("timestamp"))
    if stamps:
        first, last = stamps[0], stamps[-1]
        same = first[:10] == last[:10]
        lines.append(
            f"- date: {first[:10]}, {first[11:16]} to {last[11:16]} UTC"
            if same
            else f"- date: {first[:10]} {first[11:16]} to {last[:10]} {last[11:16]} UTC"
        )
    return lines + [""]


def parity_words(row: Row, notes: Sequence[Any], axes: Sequence[str]) -> str:
    """A row's parity in words: ok, fails (explained or not), not checked and why."""
    status = row.get("parity_status")
    if status == "ok":
        return "parity ok"
    if status == "skipped":
        return f"parity not checked ({cause('error', row.get('skip_reason'))[0]})"
    if status == "fail":
        found = analysis.explained(row, notes, axes)
        metrics = ", ".join(found)
        if all(found.values()):
            verdict = "explained"
        elif not any(found.values()):
            verdict = "unexplained"
        else:
            verdict = "; ".join(
                f"{m} {'explained' if n is not None else 'unexplained'}" for m, n in found.items()
            )
        return f"parity fails on {metrics}, {verdict} (see Parity failures)"
    return "parity not recorded"


def _ok(row: Row | None) -> bool:
    return row is not None and row.get("status") == "ok"


def _not_measured(row: Row) -> str:
    return f"{row.get('status')} ({cause(row.get('status') or '', row.get('reason'))[0]})"


def default_sentences(
    rows: Sequence[Row],
    point: Mapping[str, Any],
    baseline: str,
    notes: Sequence[Any],
    axes: Sequence[str],
    spec: Any,
    mentioned: set[str],
    *,
    dtype: str,
    mode: str,
    several_dtypes: bool,
) -> list[str]:
    """Findings 1: per entry at the default point, against the baseline; the
    controls in one sentence."""
    here = [r for r in analysis.at(rows, dtype=dtype, mode=mode) if _at_point(r, point)]
    if not here:
        return []
    by_name = {r["implementation"]: r for r in here}
    base = by_name.get(baseline)
    dtype_text = f", {dtype}" if several_dtypes else ""
    words = MODE_WORDS.get(mode, mode).capitalize()
    head = f"**{words} (`{mode}`){dtype_text}, at the default point.**"
    if _ok(base):
        head += (
            f" The baseline `{baseline}` takes {sig3(base.get('median_ms'))} ms and holds "
            f"{mib(base.get('peak_above_baseline'))} MiB above baseline; "
            f"{parity_words(base, notes, axes)}."
        )
    else:
        head += f" The baseline `{baseline}` has no `ok` record here: no ratios."
    lines = [head, ""]
    controls = []
    for name in analysis.implementations(here, baseline):
        row = by_name[name]
        if name == baseline:
            continue
        if row.get("role") == "control":
            controls.append(row)
            continue
        label = f"`{name}`"
        if name not in mentioned and _description(spec, name):
            label += f" ({_description(spec, name)})"
        mentioned.add(name)
        if not _ok(row):
            lines.append(f"- {label} was not measured here: {_not_measured(row)}.")
            continue
        if not _ok(base):
            lines.append(
                f"- {label} takes {sig3(row.get('median_ms'))} ms and holds "
                f"{mib(row.get('peak_above_baseline'))} MiB above baseline; "
                f"{parity_words(row, notes, axes)}."
            )
            continue
        time = ratio(row.get("median_ms"), base.get("median_ms"))
        peak = ratio(row.get("peak_above_baseline"), base.get("peak_above_baseline"))
        product = ratio(row.get("time_x_peak"), base.get("time_x_peak"))
        lines.append(
            f"- {label} takes {_x(time)} the baseline's time and {_x(peak)} its memory above "
            f"baseline; time × peak {_x(product)} the baseline's; "
            f"{parity_words(row, notes, axes)}."
        )
    if controls:
        checks = [
            c
            for c in environment_check(here, axes)
            if c["control"] in {r["implementation"] for r in controls}
        ]
        flagged = [c for c in checks if c["flags"]]
        what = (
            f"The controls ({_names([r['implementation'] for r in controls])}: "
            f"`{baseline}` run inside the {_names([r.get('env_name') for r in controls])} "
            f"environment{'s' if len(controls) > 1 else ''})"
        )
        if not flagged:
            lines.append(f"- {what} are within the environment check's bands here.")
        else:
            named = "; ".join(f"`{c['control']}`: {', '.join(c['flags'])}" for c in flagged)
            lines.append(f"- {what}: outside the environment check's bands here, {named}.")
    return lines + [""]


def _at_point(row: Row, point: Mapping[str, Any]) -> bool:
    return all(row.get(k) == v for k, v in point.items())


BUDGET_WORDS = {1.0: "1×", 0.75: "0.75×", 0.5: "0.5×"}


def equal_peak_sentences(
    rows: Sequence[Row],
    variant_rows: Sequence[Row],
    point: Mapping[str, Any],
    baseline: str,
    defaults: Mapping[str, str],
    *,
    dtypes: Sequence[str],
    modes: Sequence[str],
    several_dtypes: bool,
) -> list[str]:
    """Findings 2: per chunked entry, the point within each budget."""
    lines = []
    for found in analysis.equal_peak(rows, variant_rows, point, baseline):
        if found["dtype"] not in dtypes or found["mode"] not in modes:
            continue
        dtype_text = f", {found['dtype']}" if several_dtypes else ""
        budgets = found["budgets"]
        if budgets is None:
            lines += [
                f"**Time at equal peak (`{found['mode']}`{dtype_text}).** The baseline has "
                "no `ok` record with a peak at the default point: no budgets.",
                "",
            ]
            continue
        lines += [
            f"**Time at equal peak (`{found['mode']}`{dtype_text}).** The budgets are the "
            f"baseline's peak above baseline, {mib(budgets[0])} MiB, at "
            f"{', '.join(f'{f:g}×' for f in analysis.EQUAL_PEAK_BUDGETS)}.",
            "",
        ]
        for entry in found["entries"]:
            peaks = [r["peak_above_baseline"] for r in entry["points"]]
            lowest = f"lowest {mib(min(peaks))} MiB" if peaks else "none measured"
            cells = entry["cells"] or []
            if not any(cells):
                lines.append(f"- `{entry['base']}` has no point within any budget ({lowest}).")
                continue
            parts = []
            for factor, budget, row in zip(
                analysis.EQUAL_PEAK_BUDGETS, budgets, cells, strict=False
            ):
                within = f"within {factor:g}× ({mib(budget)} MiB)"
                if row is None:
                    parts.append(f"no point {within} ({lowest})")
                    continue
                if row.get("base") is not None:
                    what = f"with {row.get('params')}"
                else:
                    own = defaults.get(row["implementation"])
                    what = "with its default" + (f" ({own})" if own else "")
                fail = ", which fails parity" if row.get("parity_status") == "fail" else ""
                parts.append(f"{within}, {sig3(row.get('median_ms'))} ms {what}{fail}")
            lines.append(f"- `{entry['base']}`: {'; '.join(parts)}.")
        lines.append("")
    return lines


def _range_text(values: Sequence[tuple[Any, float]], axis: str, default: float | None) -> str:
    """A ratio over an axis: "within ±10 % of its default-point value" or its range."""
    low = min(values, key=lambda v: v[1])
    high = max(values, key=lambda v: v[1])
    if default is not None and all(
        abs(v / default - 1) <= SWEEP_BAND for _, v in values if default
    ):
        return f"within ±{SWEEP_BAND:.0%} of its {_x(default)} at the default point".replace(
            "%", " %"
        )
    return f"between {_x(low[1])} at {axis}={low[0]} and {_x(high[1])} at {axis}={high[0]}"


def sweep_sentences(
    rows: Sequence[Row],
    point: Mapping[str, Any],
    baseline: str,
    *,
    dtype: str,
    mode: str,
    several_dtypes: bool,
) -> list[str]:
    """Findings 3: per axis varied around the point, the baseline's median over it,
    and each entry's time / baseline and peak / baseline's: within the band of its
    default-point value, or the range."""
    selected = analysis.at(rows, dtype=dtype, mode=mode)
    lines = []
    dtype_text = f", {dtype}" if several_dtypes else ""
    for axis, swept in analysis.sweeps(selected, point).items():
        values = analysis.axis_values(swept, axis)
        if len(values) < 2:
            continue
        by_key = {(row[axis], row["implementation"]): row for row in swept}
        base = {v: by_key.get((v, baseline)) for v in values}
        first, last = values[0], values[-1]
        medians = [
            f"{sig3(base[v].get('median_ms'))} ms at {axis}={v}"
            for v in (first, last)
            if _ok(base[v])
        ]
        head = f"**Over {axis} (`{mode}`{dtype_text}).**"
        if len(medians) == 2:
            head += f" The baseline's median goes from {medians[0]} to {medians[1]}."
        lines += [head, ""]
        for name in analysis.implementations(swept, baseline):
            if name == baseline:
                continue
            if any(r.get("role") == "control" for r in swept if r["implementation"] == name):
                continue
            parts = []
            missing = []
            for label, key in (
                ("time / baseline", "median_ms"),
                ("peak above baseline / baseline's", "peak_above_baseline"),
            ):
                found = []
                for v in values:
                    row = by_key.get((v, name))
                    if _ok(row) and _ok(base[v]):
                        q = ratio(row.get(key), base[v].get(key))
                        if q is not None:
                            found.append((v, q))
                    elif key == "median_ms":
                        if row is None:
                            missing.append(f"{axis}={v}: no record")
                        elif not _ok(row):
                            missing.append(f"{axis}={v}: {row.get('status')}")
                        else:
                            missing.append(f"{axis}={v}: the baseline is not `ok`")
                if not found:
                    continue
                default = dict(found).get(point[axis])
                parts.append(f"{label} {_range_text(found, axis, default)}")
            if not parts and not missing:
                continue
            sentence = f"- `{name}`: {'; '.join(parts) or 'no ratio'}"
            if missing:
                sentence += f" (no ratio at {', '.join(missing)})"
            lines.append(sentence + ".")
        lines.append("")
    return lines


def repeat_sentences(rows: Sequence[Row], axes: Sequence[str], has_notes: bool) -> list[str]:
    """Findings 4: how closely the same code repeats across environments."""
    pairs = [c for c in environment_check(rows, axes) if c["measured"] == "both"]
    times = [c["time_ratio"] for c in pairs if c["time_ratio"] is not None]
    peaks = [c["peak_diff"] for c in pairs if c["peak_diff"] is not None]
    if not times:
        return []
    spread = max(abs(t - 1) for t in times)
    line = (
        f"**Repeatability.** The same code in different environments (the environment "
        f"check, {len(pairs)} pairs) repeats to within {spread:.1%} in time (ratios from "
        f"{sig3(min(times))} to {sig3(max(times))})".replace("%", " %")
    )
    if peaks:
        line += f" and to within {mib(max(abs(p) for p in peaks))} MiB in peak"
    line += "."
    if has_notes:
        line += " The op's notes on run-to-run variation are not restated here: see Notes."
    return [line, ""]


def headline_table(
    rows: Sequence[Row],
    variant_rows: Sequence[Row],
    point: Mapping[str, Any],
    baseline: str,
    defaults: Mapping[str, str],
    *,
    dtypes: Sequence[str],
    modes: Sequence[str],
) -> list[str]:
    """At the default point, every entry and mode, in the op's order: the ratios of
    the default-point tables and the equal-peak cell at the 1× budget."""
    cells_1x = {}
    for found in analysis.equal_peak(rows, variant_rows, point, baseline):
        for entry in found["entries"]:
            cells = entry["cells"]
            lowest = min((r["peak_above_baseline"] for r in entry["points"]), default=None)
            cell = MISSING if cells is None else _equal_peak_cell(cells[0], defaults, lowest)
            cells_1x[(found["dtype"], found["mode"], entry["base"])] = cell
    several = len(dtypes) > 1
    header = ["implementation", "mode", *(["dtype"] if several else [])]
    header += [
        "time / baseline",
        "peak above baseline / baseline's",
        "time × peak / baseline's",
        "≤ 1× budget: median ms",
    ]
    body = []
    for d in dtypes:
        for m in modes:
            here = [r for r in analysis.at(rows, dtype=d, mode=m) if _at_point(r, point)]
            by_name = {r["implementation"]: r for r in here}
            base = by_name.get(baseline)
            base = base if _ok(base) else None
            for name in analysis.implementations(here, baseline):
                row = by_name[name]
                cells = [impl_label(row, here), m, *([d] if several else [])]
                if not _ok(row):
                    cells += [text(row.get("status")), MISSING, MISSING]
                else:
                    cells += [
                        sig3(ratio(row.get(k), base and base.get(k)))
                        for k in ("median_ms", "peak_above_baseline", "time_x_peak")
                    ]
                cells.append(cells_1x.get((d, m, name), MISSING))
                body.append(cells)
    if not body:
        return []
    numeric = [False, False, *([False] if several else []), True, True, True, True]
    return [
        "#### Headline table",
        "",
        'The default point\'s numbers (the tables of "At the default point" and "Time at '
        "equal peak\"), every entry in the op's order.",
        "",
        table(header, body, numeric),
        "",
    ]


def caveats(
    rows_all: Sequence[Row],
    point: Mapping[str, Any],
    axes: Sequence[str],
    notes: Sequence[Any],
    *,
    unread: str | None,
    has_notes: bool,
) -> list[str]:
    """Caveats: what was not measured, where parity was not checked or failed, and
    the run-to-run band."""
    lines = ["### Caveats", ""]
    missing = [r for r in rows_all if r.get("status") in ("error", "unavailable")]
    groups: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in missing:
        groups[(row["status"], cause(row["status"], row.get("reason"))[0])].append(row)
    for (status, key), group in sorted(groups.items()):
        if status == "unavailable" and key.startswith("out of memory"):
            plain = "did not fit in the device's memory"
        elif status == "unavailable":
            plain = "could not run here"
        else:
            plain = "failed with an error nobody foresaw"
        lines.append(
            f"- {len(group)} record{'' if len(group) == 1 else 's'} {plain} "
            f"({status}: {key}): {_where_list(group, point)}."
        )
    skipped = [
        r for r in rows_all if r.get("status") == "ok" and r.get("parity_status") == "skipped"
    ]
    by_reason: dict[str, list[Row]] = defaultdict(list)
    for row in skipped:
        by_reason[cause("error", row.get("skip_reason"))[0]].append(row)
    for reason, group in sorted(by_reason.items()):
        counts = _counts(where_text(r, point) for r in group)
        where = ", ".join(f"{w} ({n})" for w, n in counts.items())
        lines.append(
            f"- Parity (the check against the op's reference) was not run in {len(group)} "
            f"record{'' if len(group) == 1 else 's'}, as the {reason}: at {where}."
        )
    total, explained_count = analysis.parity_failures(rows_all, notes, axes)
    if total:
        by_note: dict[Any, dict[str, int]] = {}
        entries: dict[Any, set[str]] = defaultdict(set)
        for row in rows_all:
            for metric, note in analysis.explained(row, notes, axes).items():
                counts_ = by_note.setdefault(note, {})
                counts_[metric] = counts_.get(metric, 0) + 1
                entries[note].add(row.get("base") or row["implementation"])
        lines.append(
            f"- {total} parity failure{'' if total == 1 else 's'}, {explained_count} explained:"
        )
        order = {id(note): i for i, note in enumerate(notes)}
        for note, counts_ in sorted(
            by_note.items(), key=lambda kv: order.get(id(kv[0]), len(order))
        ):
            metrics = ", ".join(f"{m} ({n})" for m, n in sorted(counts_.items()))
            who = _names(sorted(entries[note]))
            verdict = f"explained: {note.text}" if note is not None else "**unexplained**"
            lines.append(f"  - {who} on {metrics}, {verdict}")
    low, high = TIME_RATIO_RANGE
    lines.append(
        f"- A time ratio within the environment check's band [{low}, {high}] can be "
        "run-to-run variation, not a difference" + (" (see Notes)." if has_notes else ".")
    )
    if unread is not None:
        lines.append(
            f"- The op `{unread}` does not import here: its descriptions, definitions, "
            "parity notes and notes are not shown, and every parity failure counts as "
            "unexplained."
        )
    return lines + [""]


def _counts(items: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def _where_list(group: Sequence[Row], point: Mapping[str, Any]) -> str:
    """``impl`` (modes) at where, for the records of one cause."""
    found: dict[tuple[str, str], list[str]] = {}
    for row in group:
        modes_ = found.setdefault((row["implementation"], where_text(row, point)), [])
        if row["mode"] not in modes_:
            modes_.append(row["mode"])
    return "; ".join(
        f"`{name}` ({', '.join(modes_)}) at {where}" for (name, where), modes_ in found.items()
    )


def render_summary(
    recs: Sequence[Mapping[str, Any]],
    rows: Sequence[Row],
    variant_rows: Sequence[Row],
    point: Mapping[str, Any],
    baseline: str,
    spec: Any,
    *,
    dtypes: Sequence[str],
    modes: Sequence[str],
    sets: Mapping[str, Sequence[Mapping[str, Any]]],
    strict: tuple[str, int] | None,
    defaults: Mapping[str, str],
    notes: Sequence[Any],
    has_notes: bool,
    unread: str | None,
    findings: tuple[str, str] | None,
) -> str:
    """The section "Summary": what was compared, on what, the findings (generated,
    or the author's ``findings``, ``(file name, text)``) with the headline table, and
    the caveats."""
    axes = list(point)
    scope = [r for r in rows if r["dtype"] in dtypes and r["mode"] in modes]
    scope_all = scope + [r for r in variant_rows if r["dtype"] in dtypes and r["mode"] in modes]
    lines = [SUMMARY_TITLE, ""]
    lines += render_compared(
        recs, scope, variant_rows, point, baseline, spec, dtypes=dtypes, modes=modes, strict=strict
    )
    lines += render_on_what(recs, sets)
    if findings is not None:
        name, body = findings
        commits = sorted({r.get("tob_hash") for r in rows if r.get("tob_hash")})
        lines += [
            FINDINGS_AUTHOR,
            "",
            f"From `{name}`, for the records of torch-op-bench "
            f"{', '.join(c[:12] for c in commits) or MISSING}.",
            "",
            body.strip(),
            "",
        ]
    else:
        lines += ["### Findings", "", FINDINGS_GENERATED, ""]
        lines += generated_findings(
            scope,
            variant_rows,
            point,
            baseline,
            spec,
            notes,
            defaults,
            dtypes=dtypes,
            modes=modes,
            has_notes=has_notes,
        )
    lines += headline_table(
        scope, variant_rows, point, baseline, defaults, dtypes=dtypes, modes=modes
    )
    lines += caveats(scope_all, point, axes, notes, unread=unread, has_notes=has_notes)
    return "\n".join(lines).rstrip() + "\n"


def generated_findings(
    rows: Sequence[Row],
    variant_rows: Sequence[Row],
    point: Mapping[str, Any],
    baseline: str,
    spec: Any,
    notes: Sequence[Any],
    defaults: Mapping[str, str],
    *,
    dtypes: Sequence[str],
    modes: Sequence[str],
    has_notes: bool,
) -> list[str]:
    """The mechanical findings: the default point, time at equal peak, the sweeps,
    repeatability; fixed templates, every sentence a number against the baseline or
    a budget, in the op's order of entries."""
    axes = list(point)
    several = len(dtypes) > 1
    lines: list[str] = []
    mentioned: set[str] = set()
    for d in dtypes:
        for m in modes:
            lines += default_sentences(
                rows,
                point,
                baseline,
                notes,
                axes,
                spec,
                mentioned,
                dtype=d,
                mode=m,
                several_dtypes=several,
            )
    if variant_rows:
        lines += equal_peak_sentences(
            rows,
            variant_rows,
            point,
            baseline,
            defaults,
            dtypes=dtypes,
            modes=modes,
            several_dtypes=several,
        )
    for d in dtypes:
        for m in modes:
            lines += sweep_sentences(rows, point, baseline, dtype=d, mode=m, several_dtypes=several)
    lines += repeat_sentences(rows, axes, has_notes)
    return lines


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


def render_provenance(
    recs: Sequence[Mapping[str, Any]],
    rows: Sequence[Row],
    sets: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> str:
    """Where and with what the records were made: one row per environment build.
    ``sets``: the records of each directory read, by its name as given (``results``
    and ``--strict``'s), counted each and dated together; else ``recs``, unnamed."""
    every = [r for found in sets.values() for r in found] if sets else recs
    slugs = sorted({r["device_slug"] for r in every})
    hosts = sorted({h for h in (analysis.host_of(r) for r in every) if h})
    stamps = sorted(r["timestamp"] for r in every if r.get("timestamp"))
    lines = ["## Provenance", ""]
    lines.append(f"- device: {', '.join(slugs) or MISSING}")
    lines.append(f"- hosts: {', '.join(hosts) or MISSING}")
    if not sets:
        counted = f"{len(recs)} records"
    else:
        counted = ", ".join(
            f"{len(found)}{' records' if i == 0 else ''} in `{name}`"
            for i, (name, found) in enumerate(sets.items())
        )
    if stamps:
        lines.append(f"- records from {stamps[0]} to {stamps[-1]} ({counted})")
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


def throughput_columns(
    units: Sequence[Any] | None, rows: Sequence[Row]
) -> list[tuple[str, str, float]]:
    """The throughput columns, ``(header, row key, scale)``: one per work unit the op
    declares (``OpSpec.work_units``), headed by its ``rate_label``; ``units`` ``None``
    (the op does not import here): one per work key the records hold, ``<key>/s``,
    unscaled, which "How to read this report" says it cannot define."""
    if units is not None:
        return [(u.rate_label, u.key + analysis.RATE_SUFFIX, u.scale) for u in units]
    keys = sorted(
        {
            k.removesuffix(analysis.RATE_SUFFIX)
            for row in rows
            for k in row
            if k.endswith(analysis.RATE_SUFFIX) and row[k] is not None
        }
    )
    return [(f"{key}/s", key + analysis.RATE_SUFFIX, 1) for key in keys]


def _rate(row: Row, key: str, scale: float) -> float | None:
    value = row.get(key)
    return None if value is None else value / scale


def render_default(
    rows: Sequence[Row],
    point: Mapping[str, Any],
    baseline: str,
    *,
    title: str,
    notes: Sequence[Any] = (),
    units: Sequence[Any] | None = None,
) -> str:
    """One row per implementation at the default point, the baseline first and each
    control right after its original; a parity failure that one of ``notes`` (the
    op's ``parity_notes``) explains is marked, with the note under the table. The
    throughput columns are the op's ``units`` (``throughput_columns``)."""
    here = [row for row in rows if all(row.get(k) == v for k, v in point.items())]
    if not here:
        return f"### {title}\n\nNo records at the default point.\n"
    by_impl = {row["implementation"]: row for row in here}
    base = by_impl.get(baseline)
    base_ok = base if base is not None and base.get("status") == "ok" else None
    parity = _parity_columns(here)
    rates = throughput_columns(units, here)
    header = [
        "implementation",
        "env",
        "median ms (IQR)",
        "time / baseline",
        *(label for label, _, _ in rates),
        "peak above baseline, MiB",
        "peak above baseline / baseline's",
        "time × peak, ms·MiB",
        "time × peak / baseline's",
        "parity",
        *(p.removeprefix(analysis.PARITY_PREFIX) for p in parity),
        "override",
    ]
    numeric = [False, False, *[True] * (6 + len(rates)), False, *[True] * len(parity), False]
    body = []
    footnotes = Footnotes(notes, list(point))
    names = analysis.implementations(here, baseline)
    for name in names:
        row = by_impl[name]
        body.append(
            [
                impl_label(row, here),
                text(row.get("env_name")),
                _median_cell(row),
                sig3(ratio(row.get("median_ms"), base_ok and base_ok.get("median_ms"))),
                *(sig3(_rate(row, key, scale)) for _, key, scale in rates),
                mib(row.get("peak_above_baseline")),
                sig3(
                    ratio(
                        row.get("peak_above_baseline"),
                        base_ok and base_ok.get("peak_above_baseline"),
                    )
                ),
                sig3(row.get("time_x_peak")),
                sig3(ratio(row.get("time_x_peak"), base_ok and base_ok.get("time_x_peak"))),
                parity_text(row, footnotes),
                *(sig3(row.get(p)) for p in parity),
                override_text(row),
            ]
        )
    lines = [f"### {title}", "", f"At {analysis.point_text(point)}.", ""]
    if base_ok is None:
        lines += [f"The baseline `{baseline}` has no `ok` record here: no ratios.", ""]
    lines.append(table(header, body, numeric))
    lines += footnotes.lines()
    lines += dispatch_only_note([by_impl[name] for name in names], here)
    lines += entry_details([by_impl[name] for name in names], here)
    return "\n".join(lines) + "\n"


def _median_cell(row: Row) -> str:
    """The median and, in parentheses, the IQR; the status when not ``ok``."""
    if row.get("status") != "ok":
        return text(row.get("status"))
    return f"{sig3(row.get('median_ms'))} ({sig3(row.get('iqr_ms'))})"


def entry_details(entries: Sequence[Row], labels: Sequence[Row]) -> list[str]:
    """The table below the default-point one: per entry, in one cell, the ``info``
    fields that differ between records (``analysis.varying_info``; a field the entry
    lacks is not listed), so a difference cannot hide in ``info`` (R1-30 F6: as
    columns they made the table 64 wide); one field per line (I1-19: the torch LCE
    entries' ``op_decisions.*`` made one line of some 1500 characters)."""
    columns = sorted(
        c for c in _info_columns(entries) if any(r.get(c) is not None for r in entries)
    )
    if not columns:
        return []
    body = [
        [
            impl_label(row, labels),
            "<br>".join(
                f"{c.removeprefix(analysis.INFO_PREFIX)}={text(row[c], INFO_WIDTH)}"
                for c in columns
                if row.get(c) is not None
            )
            or MISSING,
        ]
        for row in entries
    ]
    return [
        "",
        "Entry details: the `info` fields that differ between records.",
        "",
        table(["implementation", "info"], body, [False, False]),
    ]


# --- 3. the sweeps --------------------------------------------------------------------


def _cell(
    row: Row | None, what: str, base: Row | None = None, footnotes: Footnotes | None = None
) -> str:
    if row is None:
        return MISSING
    if what == "parity":
        if row.get("status") != "ok":
            return text(row.get("status"))
        return parity_text(row, footnotes)
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
    rows: Sequence[Row],
    point: Mapping[str, Any],
    baseline: str,
    *,
    title: str,
    notes: Sequence[Any] = (),
) -> str:
    """For each axis that varies around the default point: median, peak, time × peak
    (with its ratio to the baseline's) and parity, one row per axis value and one
    column per implementation; the parity tables mark the failures ``notes``
    explain, as ``render_default``."""
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
            footnotes = Footnotes(notes, list(point))
            body = [
                [
                    str(value),
                    *(
                        _cell(
                            by_key.get((value, name)),
                            what,
                            by_key.get((value, baseline)),
                            footnotes,
                        )
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
                *footnotes.lines(),
                "",
            ]
    if not parts:
        return ""
    return "\n".join(parts)


def render_per_case(
    rows: Sequence[Row],
    axes: Sequence[str],
    baseline: str,
    dtype: str,
    mode: str,
    notes: Sequence[Any] = (),
    units: Sequence[Any] | None = None,
) -> str:
    """The default-point table at every case present, for records whose cases do not
    form a sweep (one case, or cases that differ in more than one axis)."""
    cases = sorted({analysis.case_of(row, axes)[1:] for row in rows}, key=_case_order)
    parts = []
    for values in cases:
        here = dict(zip(axes, values, strict=True))
        title = f"{mode}, {dtype}: {analysis.point_text(here)}"
        parts.append(render_default(rows, here, baseline, title=title, notes=notes, units=units))
    return "\n".join(parts)


def _case_order(values: tuple) -> tuple:
    # Numbers before text, each ascending, so a case order never depends on the files.
    return tuple((0, v, "") if isinstance(v, (int, float)) else (1, 0, str(v)) for v in values)


# --- 4. time at equal peak --------------------------------------------------------------

EQUAL_PEAK_TITLE = "## Time at equal peak"


def _equal_peak_cell(
    row: Row | None, defaults: Mapping[str, str], lowest: float | None = None
) -> str:
    """The median of the point within a budget, and in small print what produced it:
    the variant's parameter, or ``default`` (with its own value when recorded). None
    within it: ``—``, with the entry's ``lowest`` measured peak (bytes) when it has
    one, so a cell shows how far the entry misses (R1-39 F1)."""
    if row is None:
        return MISSING if lowest is None else f"{MISSING} <sub>lowest {mib(lowest)} MiB</sub>"
    if row.get("base") is not None:
        label = text(row.get("params"))
    else:
        own = defaults.get(row["implementation"])
        label = "default" + (f" ({text(own)})" if own else "")
    if row.get("parity_status") == "fail":
        label += ", parity fail"
    return f"{sig3(row.get('median_ms'))} <sub>{label}</sub>"


def _default_cell(row: Row | None) -> str:
    if row is None:
        return MISSING
    if row.get("status") != "ok":
        return text(row.get("status"))
    return f"{sig3(row.get('median_ms'))} ms, {mib(row.get('peak_above_baseline'))} MiB"


def same_point_note(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    """A sentence under an equal-peak table on the rows where every budget picks the
    same point (I1-19's fwd table): there the budgets do not separate the entry's
    points. Computed from the cells; nothing when no row, or one budget only."""
    same = [
        e["base"]
        for e in entries
        if len(e["cells"] or []) > 1
        and e["cells"][0] is not None
        and all(c is e["cells"][0] for c in e["cells"])
    ]
    if not same:
        return []
    where = (
        "In every row"
        if len(same) == len(entries)
        else "In the row"
        + ("s" if len(same) > 1 else "")
        + " of "
        + ", ".join(f"`{name}`" for name in same)
    )
    return [
        "",
        f"{where}, every budget admits the same point: none of the entry's points has "
        "its peak between the smallest budget and the largest, so the budgets do not "
        "separate them here.",
    ]


def render_equal_peak(
    found: Mapping[str, Any],
    point: Mapping[str, Any],
    baseline: str,
    variant_rows: Sequence[Row],
    defaults: Mapping[str, str],
) -> str:
    """One table of ``analysis.equal_peak``'s: a row per base entry with variants, a
    column per budget, and the default's own time and peak."""
    title = f"### {found['mode']}, {found['dtype']}"
    lines = [title, "", f"At {analysis.point_text(point)}.", ""]
    budgets = found["budgets"]
    if budgets is None:
        lines += [f"The baseline `{baseline}` has no `ok` record with a peak here: no budgets.", ""]
        budgets = []
    header = ["implementation"]
    header += [
        f"≤ {f:g}× ({mib(b)} MiB)"
        for f, b in zip(analysis.EQUAL_PEAK_BUDGETS, budgets, strict=False)
    ]
    header.append("default: median ms, peak MiB")
    body = []
    for entry in found["entries"]:
        cells = entry["cells"] or []
        peaks = [r["peak_above_baseline"] for r in entry["points"]]
        lowest = min(peaks, default=None)
        body.append(
            [
                entry["base"],
                *(_equal_peak_cell(row, defaults, lowest) for row in cells),
                _default_cell(entry["default"]),
            ]
        )
    lines.append(table(header, body, [False] + [True] * (len(header) - 1)))
    lines += same_point_note(found["entries"])
    names = {e["base"] for e in found["entries"]}
    unmeasured = [
        f"`{r['implementation']}` ({text(r.get('status'))})"
        for r in analysis.at(variant_rows, dtype=found["dtype"], mode=found["mode"])
        if r.get("base") in names and all(r.get(k) == v for k, v in point.items())
        if r.get("status") != "ok"
    ]
    if unmeasured:
        lines += ["", f"Not `ok` here, so in no cell: {', '.join(unmeasured)}."]
    return "\n".join(lines) + "\n"


def render_equal_peak_section(
    rows: Sequence[Row],
    variant_rows: Sequence[Row],
    point: Mapping[str, Any],
    baseline: str,
    defaults: Mapping[str, str],
    *,
    dtypes: Sequence[str],
    modes: Sequence[str],
) -> str:
    """The section "Time at equal peak" (Statement §6 step 6f): per dtype and mode
    with variants at ``point``, ``render_equal_peak``."""
    factors = ", ".join(f"{f:g}×" for f in analysis.EQUAL_PEAK_BUDGETS)
    lines = [
        EQUAL_PEAK_TITLE,
        "",
        f"Each chunked entry's time within a peak budget at the default point. The "
        f"budgets are the baseline `{baseline}`'s peak above baseline there, at "
        f"{factors}, the same for every entry. A cell is the median time, in ms, of the "
        f"entry's default or of one of its variants (the entry with another value of its "
        f"own chunk parameter, in small print), the one whose peak above baseline is the "
        f"largest within the budget; {MISSING} when none is (measured; no point within the "
        f"budget), with the entry's lowest measured peak in small print. The variants exist "
        f"for this comparison only: no value of a parameter is ranked here.",
        "",
    ]
    tables = [
        render_equal_peak(found, point, baseline, variant_rows, defaults)
        for found in analysis.equal_peak(rows, variant_rows, point, baseline)
        if found["dtype"] in dtypes and found["mode"] in modes
    ]
    lines += tables or ["No variant has a record at the default point."]
    return "\n".join(lines).rstrip("\n") + "\n"


# --- 5. the reduction setting ------------------------------------------------------------

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
    # Both values of a metric in one cell, so four metrics keep the table within
    # MAX_COLUMNS.
    for name in metrics:
        header += [f"{name} default, strict", f"{name} default / strict"]
    header.append("parity moves")
    if decisions:
        header.append("op_decisions")
    numeric = [False, True, True, True, *[True] * (2 * len(metrics)), False]
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
            cells += [reason, *[MISSING] * (2 + 2 * len(metrics)), MISSING]
        else:
            cells += [sig3(e["median_default"]), sig3(e["median_strict"]), sig3(e["time_ratio"])]
            for name in metrics:
                value = e["parity"].get(name)
                if value is None:
                    cells += [MISSING] * 2
                else:
                    both = f"{sig3(value['default'])}, {sig3(value['strict'])}"
                    cells += [both, sig3(value["ratio"])]
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


# --- 6. the environment check ------------------------------------------------------------


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
        f"{count['both']} pair{'' if count['both'] == 1 else 's'} compared; "
        f"{count['one']} with one side not measured"
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


# --- 7. parity failures --------------------------------------------------------------------

PARITY_FAILURES_TITLE = "## Parity failures"


def render_parity_failures(
    rows: Sequence[Row], axes: Sequence[str], notes: Sequence[Any], *, unread: str | None = None
) -> str:
    """Every ``fail`` of ``rows``, by entry, note and metric: explained by a note of
    the op, or unexplained; and the count the "final" check reads. ``unread`` is the
    ``op_ref`` of an op that does not import here, whose notes were not read: a
    sentence says so, so the count does not read as a real failure (R1-38 R1)."""
    # Per entry, per note (None: unexplained), per metric, the records.
    groups: dict[str, dict[Any, dict[str, list[Row]]]] = defaultdict(dict)
    for row in rows:
        for metric, note in analysis.explained(row, notes, axes).items():
            by_metric = groups[row["implementation"]].setdefault(note, defaultdict(list))
            by_metric[metric].append(row)
    total, explained = analysis.parity_failures(rows, notes, axes)
    lines = [PARITY_FAILURES_TITLE, ""]
    if not total:
        lines.append("No record fails parity.")
    else:
        lines += [
            "Every record that fails parity, by entry, note and metric: explained when one "
            "of the op's parity notes names the cause there, else unexplained. A record "
            "counts as explained when every metric it fails is.",
            "",
        ]
        # Per entry, a note's text once (R1-38 R3), the explained groups first, in
        # the notes' order; under it, per metric, the records.
        order = {id(note): i for i, note in enumerate(notes)}
        for name in sorted(groups):
            verdicts = sorted(
                groups[name], key=lambda n: len(order) if n is None else order.get(id(n), 0)
            )
            for note in verdicts:
                verdict = f"explained: {note.text}" if note is not None else "**unexplained**"
                lines.append(f"- `{name}`, {verdict}")
                for metric in sorted(groups[name][note]):
                    found = groups[name][note][metric]
                    lines.append(f"  - {metric} ({len(found)})")
                    for row in found:
                        lines.append(
                            f"    - {row['mode']}, {row.get('dtype')} {case_text(row, axes)}"
                        )
    lines += ["", f"{total} parity failure{'' if total == 1 else 's'}, {explained} explained."]
    if unread is not None:
        lines.append(
            f"The op `{unread}` does not import here: its parity notes are not read, and "
            "every failure counts as unexplained."
        )
    return "\n".join(lines) + "\n"


# --- 8. not measured -----------------------------------------------------------------------


REASON_WIDTH = 200
# An error's cause ends where its first number starts: what follows (sizes, free
# memory, indices) differs from record to record of one cause. A number is a token
# of its own ("4.00", "cuda:0", "4096x1024"), not digits in a word ("mat1", "sm_80",
# "2D", "BFloat16"), which name the cause (R1-31 F1).
FIRST_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:[.x]\d+)*(?![A-Za-z0-9_])")
OOM_SPECIFICS = re.compile(
    r"Tried to allocate (?P<tried>[\d.]+ \w+)\..*? of which (?P<free>[\d.]+ \w+) is free"
)


def cause(status: str, reason: str | None) -> tuple[str, str]:
    """The group of a not-measured record: its cause, and what of its reason's first
    line the cause leaves out. For an ``error``, the cause is the message up to its
    first number (``FIRST_NUMBER``), with the unfinished sentence there dropped
    ("OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB. …" is
    "OutOfMemoryError: CUDA out of memory."); for ``unavailable``, whose reasons are
    written to be stable, the whole first line, and the second line (an out-of-memory
    point's figures) is what it leaves out."""
    lines = (reason or "no reason recorded").strip().splitlines()
    line = lines[0]
    if status != "error":
        return line[:REASON_WIDTH], lines[1].strip() if len(lines) > 1 else ""
    found = FIRST_NUMBER.search(line)
    if found is None:
        return line[:REASON_WIDTH], ""
    head = line[: found.start()]
    # "…: CUDA out of memory. Tried to allocate " -> "…: CUDA out of memory."
    sentence = head.rfind(". ")
    key = head[: sentence + 1] if sentence != -1 else head.rstrip(" ([{:,=")
    name, colon, message = head.partition(":")
    if not (message if colon and " " not in name else head).strip(" :([{,="):
        # The message starts with a number: no cause to cut it at.
        return line[:REASON_WIDTH], ""
    oom = OOM_SPECIFICS.search(line)
    if oom:
        rest = f"tried to allocate {oom['tried']}, {oom['free']} free"
    else:
        rest = line[len(key) :].strip(" :,=")
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


# --- 9. the op's notes ----------------------------------------------------------------------

NOTES_TITLE = "## Notes"


def render_notes(op_name: str, notes: Sequence[str]) -> str:
    """The op's ``notes``, verbatim; nothing when it has none (or does not import)."""
    if not notes:
        return ""
    lines = [
        NOTES_TITLE,
        "",
        f"The notes of the op `{op_name}`, as its author wrote them (`OpSpec.notes`); "
        "the report does not compute them.",
        "",
    ]
    for note in notes:
        lines += [f"- {note}", ""]
    return "\n".join(lines)


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
    findings: str | Path | None = None,
) -> str:
    """The markdown report of a results directory; with ``strict``, a strict run's
    results directory, the "Reduction setting" section too; with ``findings``, a
    markdown file an author wrote for these records, its text as the summary's
    findings in place of the generated sentences."""
    written = None
    if findings is not None:
        written = (Path(findings).name, Path(findings).read_text(encoding="utf-8"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        recs = analysis.load(results_dir, op=op, device_slug=device_slug)
        if not recs:
            raise ValueError(f"no records under {results_dir}")
        # The variants are in "Time at equal peak", "Parity failures" and "Not
        # measured" only.
        base_recs, variant_recs = analysis.split_variants(recs)
        op_name = _one({r["op"] for r in recs}, "op", "choose one with --op")
        slug = _one(
            {r["device_slug"] for r in recs}, "device slug", "choose one with --device-slug"
        )
        # The strict set of the same op and device as RESULTS.
        recs_strict = [] if strict is None else analysis.load(strict, op=op_name, device_slug=slug)
        all_strict = recs_strict
        recs_strict = analysis.split_variants(recs_strict)[0]
    rows = analysis.to_rows(base_recs)
    variant_rows = analysis.to_rows(variant_recs)
    axes = analysis.axes_of(recs)
    notes = analysis.parity_notes(recs)
    spec = analysis.spec_of(recs[0])
    units = None if spec is None else list(spec.work_units)
    point = analysis.default_point(base_recs or recs, point=point)
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
    sets = {str(results_dir): recs}
    if strict is not None:
        sets[str(strict)] = all_strict
    unread = None if spec else recs[0].get("op_ref") or recs[0]["op"]
    defaults = analysis.default_params(base_recs, variant_recs)
    op_notes = analysis.op_notes(recs)
    lines.append(
        render_summary(
            recs,
            rows,
            variant_rows,
            point["params"],
            baseline,
            spec,
            dtypes=dtypes,
            modes=modes,
            sets=sets,
            strict=None if strict is None else (str(strict), len(all_strict)),
            defaults=defaults,
            notes=notes,
            has_notes=bool(op_notes),
            unread=unread,
            findings=written,
        )
    )
    work_keys = sorted({k for r in recs for k in (r.get("work") or {})})
    lines.append(render_definitions(recs, baseline, spec, unread or op_name, work_keys))
    lines.append(render_provenance(recs, rows, sets))
    source = SOURCES.get(point["source"], point["source"])
    lines += ["## At the default point", "", f"The default point is {source}.", ""]
    lines += axes_line(spec, point["params"])
    for d in dtypes:
        for m in modes:
            selected = analysis.at(rows, dtype=d, mode=m)
            lines.append(
                render_default(
                    selected, point["params"], baseline, title=f"{m}, {d}", notes=notes, units=units
                )
            )
    lines += ["## Sweeps", ""]
    # Per (dtype, mode), as plot decides: the sweeps around the point, or, when no
    # axis varies there, the default-point table at every case present.
    pairs = []
    for d in dtypes:
        for m in modes:
            selected = analysis.at(rows, dtype=d, mode=m)
            if selected:
                swept = render_sweeps(
                    selected, point["params"], baseline, title=f"{m}, {d}", notes=notes
                )
                pairs.append((d, m, selected, swept))
    if not any(swept for *_, swept in pairs):
        # No pair has a sweep: one line for all of them.
        lines += [f"{NO_SWEEP} in these records: one table per case instead.", ""]
        lines += [
            render_per_case(sel, axes, baseline, d, m, notes, units) for d, m, sel, _ in pairs
        ]
    else:
        for d, m, selected, swept in pairs:
            if swept:
                lines.append(swept)
                continue
            lines += [f"**{m}, {d}:** {NO_SWEEP.lower()}: one table per case instead.", ""]
            lines.append(render_per_case(selected, axes, baseline, d, m, notes, units))
    if variant_rows:
        lines.append(
            render_equal_peak_section(
                rows,
                variant_rows,
                point["params"],
                baseline,
                defaults,
                dtypes=dtypes,
                modes=modes,
            )
        )
    if strict is not None:
        lines.append(
            render_reduction_section(
                base_recs,
                recs_strict,
                point["params"],
                baseline,
                dtypes=dtypes,
                modes=modes,
                strict_dir=strict,
            )
        )
    scope = [row for row in rows if row["dtype"] in dtypes and row["mode"] in modes]
    scope_all = scope + [r for r in variant_rows if r["dtype"] in dtypes and r["mode"] in modes]
    lines.append(render_parity_failures(scope_all, axes, notes, unread=unread))
    lines.append(render_environment_check(scope, axes))
    lines.append(render_not_measured(scope_all, axes))
    lines.append(render_notes(op_name, op_notes))
    return "\n".join(lines).rstrip() + "\n"
