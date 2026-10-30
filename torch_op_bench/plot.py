"""``torch-op-bench plot``: the figures of the write-up (R8).

Three figures per mode (and dtype, when the results hold several): the grid
(metrics × swept axes), the ratio against the baseline, and, with a second results
directory, the overlay of both. Next to every PNG, a CSV holds exactly the plotted
points, so the write-up and the tests read what was drawn. matplotlib is imported
lazily, with the Agg backend set in code: no display is needed.
"""

from __future__ import annotations

import csv
import itertools
import math
import textwrap
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import analysis
from .analysis import Row
from .spec import MODES

DPI = 150
# Inches per panel; the bands above and below the panels are sized to their text.
PANEL_WIDTH, PANEL_HEIGHT = 3.8, 2.8
MIB = 1024**2
# The title ("large") wraps at this many characters per inch of figure width.
TITLE_CHARS_PER_INCH = 10
# The layout, in inches: the margin above the title and below the footer, and the
# gap between title and legend. Between the legend (or the title) and the first
# panel, and between the last panel and the footer, the gap is tight_layout's pad,
# LAYOUT_PAD times the font size (10 pt: 3 pt). Between two stacked panels (tick and
# axis labels included) it is tight_layout's h_pad, PANEL_PAD times the font size: the
# same 3 pt, so the figure reads evenly. Every such gap is at most a capital letter of
# the axis font (the tick and axis labels, 10 pt): about 7.3 pt, PANEL_PAD 0.73.
MARGIN = 0.05
GAP = 3 / 72
LAYOUT_PAD = 0.3
PANEL_PAD = LAYOUT_PAD
# tight_layout passes at most, until the panels stop moving (see _layout).
LAYOUT_PASSES = 4

# The line styles, by role. The baseline's is fixed here, in every figure, and owes
# nothing to matplotlib's defaults; a control takes its original's colour.
STYLE: dict[str, dict[str, Any]] = {
    "baseline": {"color": "black", "linestyle": "--", "linewidth": 2.5, "marker": "o", "size": 6},
    "counterpart": {"linestyle": "-", "linewidth": 1.5, "marker": "o", "size": 5},
    "control": {"linestyle": ":", "linewidth": 1.0, "marker": "s", "size": 4},
}
# matplotlib's tab10, a colour-blind-safe cycle, spelled out so it cannot change.
COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)
# A control's marker, one per environment, so two controls of one original (the
# same colour and line) stay apart: the first environment in sorted order gets the
# first. "s" is STYLE["control"]'s own.
CONTROL_MARKERS = ("s", "^", "D", "v", "P", "X", "<", ">")
# The overlay tells its two result sets apart by line style.
OVERLAY_LINESTYLE = ("-", "--")
# Two series of a panel coincide when they have values at the same x and, at every
# one, their points lie within COINCIDENT_MARKER_FRACTION of the marker size of each
# other, in display points, after the layout (P1, I1-14 round 2): the upper marker
# would cover the lower. A control and its original never do: the control has its own
# style by design; nor do two controls of one original (R1-34 R1), which the
# environment check compares each with the original. Series that coincide, directly
# or through another, form a stack: each is drawn a level above those drawn before it
# (the baseline is drawn first).
# Levels above the bottom are dashed, at COINCIDENT_SCALE of their line width and
# marker size; from level 2 on each is also shifted sideways, alternately left and
# right, by COINCIDENT_DODGE of the bottom's marker size per step, so the small markers
# sit side by side and every colour shows at a shared point (F7 of I1-12, F1 of
# R1-33).
COINCIDENT_MARKER_FRACTION = 0.5
COINCIDENT_SCALE = 0.5
COINCIDENT_DODGE = 0.5
# Layouts tried until the coincident series found on one are those its footer names.
COINCIDENT_PASSES = 3

METRIC_LABELS = {
    "median_ms": "median (ms)",
    "min_ms": "min (ms)",
    "mean_ms": "mean (ms)",
    "iqr_ms": "IQR (ms)",
    "tokens_per_s": "tokens/s",
    "tflops": "TFLOP/s",
    "peak_above_baseline": "peak above baseline (MiB)",
    "peak_bytes": "peak (MiB)",
    "time_x_peak": "time × peak (ms·MiB)",
    "speedup": "speedup (baseline / impl time)",
    "memory_ratio": "memory ratio (impl / baseline peak)",
    "throughput_ratio": "throughput ratio (impl / baseline)",
    "time_x_peak_ratio": "time × peak ratio (impl / baseline)",
}
# A linear y axis whose values reach above SCALE_ABOVE, or stay below SCALE_BELOW, is
# drawn in a unit scaled by a power of 10³, which its label names ("time × peak
# (10⁶ ms·MiB)"): with its tick labels never in scientific notation or offset
# (ScalarFormatter), no panel has an offset text above it, which would make its row
# taller than the others (N2, R1-30). The power is chosen per metric of a figure, so
# a row of panels reads in one unit.
SCALE_ABOVE = 1e5
SCALE_BELOW = 1e-4
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
NUMERIC = ("median_ms", "min_ms", "mean_ms", "iqr_ms", "tokens_per_s", "tflops", "time_x_peak")
MEMORY = ("peak_above_baseline", "peak_bytes")
DEFAULT_PARITY = {"fwd_bwd": "parity_grad_hidden_rel_l2", "fwd": "parity_loss_rel_err"}

CSV_COLUMNS = [
    "figure",
    "set",
    "metric",
    "axis",
    "x",
    "implementation",
    "env_name",
    "role",
    "y",
    "parity_status",
    "parity_fail",
    "parity_fail_metric",
    "dtype",
    "mode",
    "point",
    "work",
    "baseline_work",
]


def pyplot() -> Any:
    """matplotlib's pyplot on the Agg backend, set before pyplot is imported."""
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError("plot needs matplotlib: install torch-op-bench[plot]") from exc
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


# --- what to draw ------------------------------------------------------------------------


def known_metrics(rows: Sequence[Row]) -> list[str]:
    parity = sorted(
        {
            k
            for row in rows
            for k, v in row.items()
            if k.startswith(analysis.PARITY_PREFIX)
            and k not in ("parity_status", "parity_failed")
            and v is not None
        }
    )
    return [*NUMERIC, *MEMORY, *parity]


def default_metrics(rows: Sequence[Row], mode: str) -> list[str]:
    """median, tokens/s when present, peak above baseline, and one parity metric: the
    op's ``grad_hidden`` (fwd_bwd) or ``loss`` (fwd) error when it has one, else its
    first parity metric."""

    def present(name: str) -> bool:
        return any(row.get(name) is not None for row in rows)

    metrics = ["median_ms"]
    if present("tokens_per_s"):
        metrics.append("tokens_per_s")
    metrics.append("peak_above_baseline")
    parity = [m for m in known_metrics(rows) if m.startswith(analysis.PARITY_PREFIX)]
    preferred = DEFAULT_PARITY[mode]
    if preferred in parity:
        metrics.append(preferred)
    elif parity:
        metrics.append(parity[0])
    return metrics


def role_of(row_or_rows: Row | Sequence[Row], name: str, baseline: str) -> str:
    rows = [row_or_rows] if isinstance(row_or_rows, Mapping) else row_or_rows
    if name == baseline:
        return "baseline"
    if any(r["implementation"] == name and r.get("role") == "control" for r in rows):
        return "control"
    return "counterpart"


def colors(
    rows: Sequence[Row], baseline: str, order: Sequence[str] | None = None
) -> dict[str, str]:
    """One colour per implementation, by its place in a stable order: ``order`` (the
    spec's implementations, controls left out) and then the sorted names it lacks,
    so another invocation, or a figure without some implementation, gives the same
    colours. A control takes its original's; the baseline is black."""
    control_of = {
        r["implementation"]: r.get("control_of") for r in rows if r.get("role") == "control"
    }
    present = sorted({r["implementation"] for r in rows})
    ranked = [*(order or ()), *(n for n in present if n not in (order or ()))]
    ranked = [n for n in ranked if n != baseline and n not in control_of]
    result: dict[str, str] = {
        name: COLORS[i % len(COLORS)] for i, name in enumerate(ranked) if name in present
    }
    if baseline in present:
        result[baseline] = STYLE["baseline"]["color"]
    for name, original in control_of.items():
        if original == baseline:  # drawn or not (a ratio figure omits it)
            result[name] = STYLE["baseline"]["color"]
        else:
            result[name] = result.get(original, COLORS[len(ranked) % len(COLORS)])
    return result


def stable_order(recs: Sequence[Mapping[str, Any]]) -> list[str] | None:
    """The spec's implementations, controls and variants left out, when the op
    resolves."""
    spec = analysis.spec_of(recs[0]) if recs else None
    if spec is None:
        return None
    return [
        i.name
        for i in spec.implementations
        if i.info.get("role") != "control" and i.variant_of is None
    ]


def control_markers(rows: Sequence[Row]) -> dict[str, str]:
    """One marker per control, by its environment: controls of one original in
    different environments look different."""
    envs = sorted({r.get("env_name") or "" for r in rows if r.get("role") == "control"})
    marker = {env: CONTROL_MARKERS[i % len(CONTROL_MARKERS)] for i, env in enumerate(envs)}
    return {
        r["implementation"]: marker[r.get("env_name") or ""]
        for r in rows
        if r.get("role") == "control"
    }


def swept_axes(rows: Sequence[Row], point: Mapping[str, Any]) -> list[str]:
    return [
        axis
        for axis, swept in analysis.sweeps(rows, point).items()
        if len(analysis.axis_values(swept, axis)) > 1
    ]


def failed_metrics(row: Row) -> set[str]:
    """The parity metrics a row failed on, from ``parity_failed``."""
    return set(filter(None, (row.get("parity_failed") or "").split(",")))


def parity_fail_metric(row: Row, metric: str) -> bool | None:
    """Whether the row failed on ``metric`` itself, when it is a parity metric
    (``parity_<name>``); ``None`` for any other metric."""
    if not metric.startswith(analysis.PARITY_PREFIX):
        return None
    return metric.removeprefix(analysis.PARITY_PREFIX) in failed_metrics(row)


def _value(row: Row, metric: str) -> float | None:
    value = row.get(metric)
    if value is None:
        return None
    return value / MIB if metric in MEMORY else value


def work_text(row: Row) -> str | None:
    """The work units of a row, as the CSV shows them."""
    parts = [f"{k}={row[k]}" for k in ("tokens", "flops") if row.get(k) is not None]
    return ",".join(parts) or None


def ratio_rows(rows: Sequence[Row], baseline: str, axes: Sequence[str]) -> list[Row]:
    """Each ok row with ``speedup``, ``memory_ratio``, ``throughput_ratio`` and
    ``time_x_peak_ratio`` against
    the baseline's ok row at the same point, and both rows' work (``work``,
    ``baseline_work``); the baseline's own rows left out."""
    base = {
        (row["mode"], analysis.case_of(row, axes)): row
        for row in rows
        if row["implementation"] == baseline and row.get("status") == "ok"
    }
    result = []
    for row in rows:
        if row["implementation"] == baseline:
            continue
        ref = base.get((row["mode"], analysis.case_of(row, axes)))
        if ref is None or row.get("status") != "ok":
            result.append(
                {
                    **row,
                    "speedup": None,
                    "memory_ratio": None,
                    "throughput_ratio": None,
                    "time_x_peak_ratio": None,
                    "baseline_work": None,
                }
            )
            continue
        result.append(
            {
                **row,
                "speedup": _div(ref.get("median_ms"), row.get("median_ms")),
                "memory_ratio": _div(
                    row.get("peak_above_baseline"), ref.get("peak_above_baseline")
                ),
                "throughput_ratio": _div(row.get("tokens_per_s"), ref.get("tokens_per_s")),
                "time_x_peak_ratio": _div(row.get("time_x_peak"), ref.get("time_x_peak")),
                "baseline_work": work_text(ref),
            }
        )
    return result


def work_differs(ratios: Sequence[Row]) -> bool:
    """Whether some implementation's work differs from the baseline's at a point:
    only then is the throughput ratio not the speedup again."""
    return any(
        r.get("baseline_work") is not None and work_text(r) != r["baseline_work"] for r in ratios
    )


def _div(a: Any, b: Any) -> float | None:
    return None if a is None or b is None or b == 0 else a / b


# --- drawing -----------------------------------------------------------------------------------


def provenance(rows: Sequence[Row], label: str | None = None) -> str:
    """One line: per environment, its torch and torch-op-bench commit; the hosts."""
    builds = sorted(
        {
            (r.get("env_name"), r.get("torch_version"), (r.get("tob_hash") or "")[:12])
            for r in rows
            if r.get("torch_version") or r.get("tob_hash")
        },
        key=lambda b: tuple(str(x) for x in b),
    )
    hosts = sorted({r["host"] for r in rows if r.get("host")})
    parts = [f"{env}: torch {version}, torch-op-bench {tob or '?'}" for env, version, tob in builds]
    text = "; ".join(parts) or "no provenance recorded"
    prefix = f"{label}: " if label else ""
    return f"{prefix}{text}; host {', '.join(hosts) or '?'}"


def left_out(
    rows: Sequence[Row],
    metrics: Sequence[str],
    notes: Sequence[Any] = (),
    axes: Sequence[str] = (),
) -> str:
    """How many of the figure's records are not drawn, and why; and how many fail
    parity (drawn hollow, in every panel when any metric failed) and how many of
    those the op's parity ``notes`` explain."""
    counts = Counter(row["status"] for row in rows if row.get("status") != "ok")
    skipped = sum(
        1
        for row in rows
        if row.get("status") == "ok" and row.get("parity_status") not in ("ok", "fail")
    )
    parts = [f"{n} {status}" for status, n in sorted(counts.items())]
    parity = any(m.startswith(analysis.PARITY_PREFIX) for m in metrics)
    if skipped and parity:
        parts.append(f"{skipped} without parity (parity rows only)")
    if not parts:
        line = "Every record of this figure is drawn."
    else:
        line = f"Left out: {', '.join(parts)} (gaps in the lines)."
    failed, explained = analysis.parity_failures(rows, notes, axes)
    if failed:
        line += f" Parity failures: {failed} ({explained} explained), drawn hollow."
    return line


def y_exponent(values: Sequence[float | None]) -> int:
    """The power of 10 (a multiple of 3) a linear axis of ``values`` is drawn in."""
    top = max((abs(v) for v in values if v is not None and math.isfinite(v)), default=0.0)
    if top == 0 or SCALE_BELOW <= top <= SCALE_ABOVE:
        return 0
    return 3 * math.floor(math.log10(top) / 3)


def scaled_label(label: str, exponent: int) -> str:
    """``label`` in the unit scaled by 10**exponent: "time × peak (10⁶ ms·MiB)";
    a label without a unit gains one, "tokens/s (×10³)"."""
    if not exponent:
        return label
    power = "10" + str(exponent).translate(SUPERSCRIPT)
    head, sep, tail = label.partition(" (")
    if sep and tail.endswith(")"):
        return f"{head} ({power} {tail}"
    return f"{label} (×{power})"


def _plain_ticks(ax: Any, axis: str = "y") -> None:
    """Tick labels of a linear axis (``y`` or ``x``) as plain numbers: no offset, no
    1eN beside it."""
    from matplotlib.ticker import ScalarFormatter

    formatter = ScalarFormatter(useOffset=False, useMathText=True)
    formatter.set_scientific(False)
    getattr(ax, f"{axis}axis").set_major_formatter(formatter)


def _positions(values: Sequence[Any]) -> tuple[list[float], bool]:
    """x positions: the values themselves (log scale) for numbers, else indices."""
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    if numeric and all(v > 0 for v in values):
        return [float(v) for v in values], True
    return [float(i) for i in range(len(values))], False


def _draw_panel(
    ax: Any,
    sets: Sequence[tuple[str | None, Sequence[Row], str | None]],
    metric: str,
    axis: str,
    point: Mapping[str, Any],
    names: Sequence[str],
    palette: Mapping[str, str],
    markers: Mapping[str, str],
    baseline: str,
    figure: str,
    handles: dict[str, Any],
    drawn: list[tuple[Any, str, str, list[dict[str, Any]]]],
    exponent: int = 0,
) -> list[dict[str, Any]]:
    """One metric against one axis, for every result set, its y drawn in units of
    10**exponent; returns the CSV rows (with the values, not scaled).
    (ax, metric, axis, series) is appended to ``drawn``: each series with its artists,
    for ``stack_coincident`` once the layout is final."""
    from matplotlib.lines import Line2D

    all_swept = [analysis.sweeps(rows, point)[axis] for _, rows, _ in sets]
    values = analysis.axis_values([r for swept in all_swept for r in swept], axis)
    xs, log = _positions(values)
    plotted = []
    series = []
    for index, ((label, _, linestyle), swept) in enumerate(zip(sets, all_swept, strict=True)):
        for name in names:
            role = role_of(swept, name, baseline)
            style = {**STYLE[role], "marker": markers.get(name, STYLE[role]["marker"])}
            color = palette.get(name, "#7f7f7f")
            by_x = {r[axis]: r for r in swept if r["implementation"] == name}
            if not by_x:
                continue
            ys, ok_pts, fail_pts = [], [], []
            for x, value in zip(xs, values, strict=True):
                row = by_x.get(value)
                y = _value(row, metric) if row is not None and row.get("status") == "ok" else None
                ys.append(math.nan if y is None else y)
                if y is None:
                    continue
                fail = row.get("parity_status") == "fail"
                # Hollow: in a parity panel, this metric failed; elsewhere, any did.
                own = parity_fail_metric(row, metric)
                (fail_pts if (fail if own is None else own) else ok_pts).append((x, y))
                plotted.append(
                    {
                        "figure": figure,
                        "set": label or "",
                        "metric": metric,
                        "axis": axis,
                        "x": value,
                        "implementation": name,
                        "env_name": row.get("env_name"),
                        "role": role,
                        "y": y,
                        "parity_status": row.get("parity_status"),
                        "parity_fail": fail,
                        "parity_fail_metric": own,
                        "dtype": row.get("dtype"),
                        "mode": row.get("mode"),
                        "point": ", ".join(f"{k}={row.get(k)}" for k in point),
                        "work": work_text(row),
                        "baseline_work": row.get("baseline_work"),
                    }
                )
            line = linestyle or style["linestyle"]
            series.append(
                {
                    "set": index,
                    "name": name,
                    "control_of": next(
                        (r.get("control_of") for r in by_x.values() if r.get("control_of")),
                        None,
                    ),
                    "role": role,
                    "style": style,
                    "color": color,
                    "line": line,
                    "ys": ys,
                    "points": (ok_pts, fail_pts),
                }
            )
            key = f"{name} [{label}]" if label else name
            handles.setdefault(
                key,
                Line2D(
                    [],
                    [],
                    color=color,
                    linestyle=line,
                    linewidth=style["linewidth"],
                    marker=style["marker"],
                    markersize=style["size"],
                    label=key,
                ),
            )
    unit = 10.0**exponent
    for s in series:
        style, color = s["style"], s["color"]
        # The baseline on top of an equal counterpart, which must not hide it.
        z = 3 if s["role"] == "baseline" else 2
        (line,) = ax.plot(
            xs,
            [y / unit for y in s["ys"]],
            color=color,
            linestyle=s["line"],
            linewidth=style["linewidth"],
            zorder=z,
        )
        s["artists"] = [(line, 0.0)]
        for pts, face in zip(s["points"], (color, "none"), strict=True):
            if pts:
                (markers_,) = ax.plot(
                    [p[0] for p in pts],
                    [p[1] / unit for p in pts],
                    linestyle="none",
                    marker=style["marker"],
                    markersize=style["size"] + (2 if face == "none" else 0),
                    color=color,
                    markerfacecolor=face,
                    markeredgecolor=color,
                    zorder=z,
                )
                s["artists"].append((markers_, 2.0 if face == "none" else 0.0))
    drawn.append((ax, metric, axis, series))
    if log:
        ax.set_xscale("log", base=2)
        ax.minorticks_off()
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in values])
    ax.set_xlabel(axis)
    label = METRIC_LABELS.get(metric, metric.removeprefix(analysis.PARITY_PREFIX))
    ax.set_ylabel(ylabel_text(scaled_label(label, exponent)))
    if metric.startswith(analysis.PARITY_PREFIX):
        ax.set_yscale("log")
    else:
        _plain_ticks(ax)
    ax.grid(True, alpha=0.3)
    return plotted


def _coincide(lower: Mapping[str, Any], upper: Mapping[str, Any]) -> bool:
    """Whether two series of a panel coincide (COINCIDENT_MARKER_FRACTION); see
    ``coincident_pairs``."""
    if lower["set"] != upper["set"]:
        return False
    if upper.get("control_of") == lower["name"] or lower.get("control_of") == upper["name"]:
        return False
    # Two controls of one original (R1-34 R1): see COINCIDENT_MARKER_FRACTION.
    if upper.get("control_of") and upper.get("control_of") == lower.get("control_of"):
        return False
    a, b = lower["display"], upper["display"]
    if [p is None for p in a] != [p is None for p in b]:
        return False
    limit = COINCIDENT_MARKER_FRACTION * lower["style"]["size"]
    pairs = [(p, q) for p, q in zip(a, b, strict=True) if p is not None]
    return bool(pairs) and all(p == q or math.dist(p, q) <= limit for p, q in pairs)


def coincident_pairs(series: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    """Of a panel's series (``set``, ``name``, ``control_of``, the marker ``size`` of
    its ``style``, ``display``: each value's (x, y) in display points, None for no
    value), in drawing order: each one that coincides with an earlier one of its set
    -> the first such one, the one the footer names."""
    on = {}
    for k, s in enumerate(series):
        for j, t in enumerate(series[:k]):
            if _coincide(t, s):
                on[k] = j
                break
    return on


def coincident_levels(series: Sequence[Mapping[str, Any]]) -> dict[int, tuple[int, int, int]]:
    """Each series of a panel (as for ``coincident_pairs``) -> (its level, the number
    of levels of its stack, its stack's bottom series). A stack is a series with every
    one that coincides with it, directly or through another; each member is a level
    above the members drawn before it, so no two share a level (R1-33 F1)."""
    root = list(range(len(series)))

    def find(k: int) -> int:
        while root[k] != k:
            k = root[k]
        return k

    for k, s in enumerate(series):
        for j, t in enumerate(series[:k]):
            if _coincide(t, s):
                a, b = find(j), find(k)
                root[max(a, b)] = min(a, b)
    level: dict[int, int] = {}
    height: dict[int, int] = {}
    for k in range(len(series)):
        bottom = find(k)
        level[k] = height.get(bottom, 0)
        height[bottom] = level[k] + 1
    return {k: (level[k], height[find(k)], find(k)) for k in range(len(series))}


def stack_coincident(
    fig: Any, drawn: Sequence[tuple[Any, str, str, Sequence[dict[str, Any]]]]
) -> dict[tuple[int, str, str], list[tuple[str, str]]]:
    """Restyle the coincident series of every drawn panel (``_draw_panel``), measured
    in display points on ``fig`` as it is laid out now; returns (set index, upper,
    lower) -> the (metric, axis) panels, for the footer."""
    from matplotlib.transforms import offset_copy

    coincident: dict[tuple[int, str, str], list[tuple[str, str]]] = {}
    to_points = 72.0 / fig.dpi
    for ax, metric, axis, series in drawn:
        ax.get_xlim(), ax.get_ylim()  # the limits autoscaled before the transform
        for s in series:
            line, _ = s["artists"][0]
            s["display"] = [
                None
                if math.isnan(y)
                else tuple(v * to_points for v in ax.transData.transform((x, y)))
                for x, y in zip(line.get_xdata(), line.get_ydata(), strict=True)
            ]
        on = coincident_pairs(series)
        for upper, lower in on.items():
            key = (series[upper]["set"], series[upper]["name"], series[lower]["name"])
            coincident.setdefault(key, []).append((metric, axis))
        levels = coincident_levels(series)
        for s, (level, _, bottom) in zip(series, levels.values(), strict=True):
            style = s["style"]
            scale = COINCIDENT_SCALE if level else 1.0
            z = (3 if s["role"] == "baseline" else 2) + 2 * level
            transform = ax.transData
            if level >= 2:
                step = (level // 2) * (1 if level % 2 else -1)
                dx = step * COINCIDENT_DODGE * series[bottom]["style"]["size"]
                transform = offset_copy(ax.transData, fig=fig, x=dx, y=0, units="points")
            line, _ = s["artists"][0]
            line.set_linestyle("--" if level else s["line"])
            line.set_linewidth(style["linewidth"] * scale)
            for artist, extra in s["artists"]:
                artist.set_zorder(z)
                artist.set_transform(transform)
                if artist is not line:
                    artist.set_markersize(style["size"] * scale + extra)
    return coincident


def coincident_text(
    coincident: Mapping[tuple[int, str, str], Sequence[tuple[str, str]]],
    index: int,
    axes: Sequence[str],
    modes: Sequence[str],
) -> str | None:
    """The footer's clause on the coincident series of set ``index``, one per pair,
    its panels grouped by metric (the axes named when not every one)."""
    clauses = []
    for (i, upper, lower), panels in coincident.items():
        if i != index:
            continue
        by_metric: dict[str, list[str]] = {}
        for metric, axis in panels:
            by_metric.setdefault(metric, []).append(axis)
        names = []
        for metric, on in by_metric.items():
            label = METRIC_LABELS.get(metric, metric.removeprefix(analysis.PARITY_PREFIX))
            name = label.partition(" (")[0]
            names.append(name if len(on) == len(axes) else f"{name} on {'/'.join(on)}")
        clauses.append(f"{upper} on {lower} ({'/'.join(modes)}: {', '.join(names)})")
    if not clauses:
        return None
    return f"Coincident, the first drawn dashed on the second: {'; '.join(clauses)}."


def _figure(
    path: Path,
    sets: Sequence[tuple[str | None, Sequence[Row], str | None]],
    metrics: Sequence[str],
    axes: Sequence[str],
    point: Mapping[str, Any],
    baseline: str,
    title: str,
    footer: Sequence[str],
    *,
    palette: Mapping[str, str] | None = None,
    markers: Mapping[str, str] | None = None,
    ratio_line: bool = False,
    panel: Callable[[Any, dict[str, Any]], list[dict[str, Any]]] | None = None,
    panel_width: float = PANEL_WIDTH,
) -> Path:
    """The figure of ``metrics`` (rows) against ``axes`` (columns), each panel drawn
    by ``_draw_panel``, or by ``panel(ax, handles)`` (which adds its legend entries
    to ``handles`` and returns its CSV rows); its bands, layout and CSV."""
    plt = pyplot()
    union = [row for _, rows, _ in sets for row in rows]
    names = analysis.implementations(union, baseline)
    palette = palette if palette is not None else colors(union, baseline)
    markers = markers if markers is not None else control_markers(union)
    width = panel_width * max(1, len(axes))
    fig, grid = plt.subplots(
        len(metrics), max(1, len(axes)), figsize=(width, PANEL_HEIGHT * len(metrics)), squeeze=False
    )
    handles: dict[str, Any] = {}
    drawn: list[tuple[Any, str, str, list[dict[str, Any]]]] = []
    plotted: list[dict[str, Any]] = []
    for i, metric in enumerate(metrics):
        if panel is not None:
            for ax in grid[i]:
                plotted += panel(ax, handles)
            continue
        exponent = 0
        if not metric.startswith(analysis.PARITY_PREFIX):
            exponent = y_exponent(
                [
                    _value(r, metric)
                    for _, rows, _ in sets
                    for axis in axes
                    for r in analysis.sweeps(rows, point)[axis]
                    if r.get("status") == "ok"
                ]
            )
        for j, axis in enumerate(axes):
            ax = grid[i][j]
            plotted += _draw_panel(
                ax,
                sets,
                metric,
                axis,
                point,
                names,
                palette,
                markers,
                baseline,
                path.stem,
                handles,
                drawn,
                exponent,
            )
            if ratio_line:
                base = STYLE["baseline"]
                ax.axhline(
                    1.0,
                    color=base["color"],
                    linestyle=base["linestyle"],
                    linewidth=base["linewidth"],
                )
    modes = [m for m in MODES if any(row.get("mode") == m for row in union)]
    # Bands in inches, top to bottom: title, legend, panels, footer. The title and the
    # footer wrap to the figure's width, so neither is cut off in a one-column figure.
    # Each band is as high as its rendered text, measured, and the bands are stacked
    # with GAP between them, so the white space between title, legend, panels and
    # footer is at most about the height of a capital letter of that text.
    columns = max(1, min(4, int(width // 2.2)))
    title_lines = textwrap.wrap(title, int(width * TITLE_CHARS_PER_INCH), break_on_hyphens=False)
    # fig.text, not fig.suptitle: tight_layout makes room for a suptitle on top of
    # its rect, which would count the title band twice.
    title_text = fig.text(
        0.5, 1, "\n".join(title_lines), ha="center", va="top", fontsize="large", gid="title"
    )
    legend = None
    if handles:
        legend = fig.legend(
            handles=list(handles.values()),
            loc="upper center",
            ncol=columns,
            bbox_to_anchor=(0.5, 1),
            borderaxespad=0,
            fontsize="small",
        )
    footer_text = fig.text(0.01, 0, "", ha="left", va="bottom", fontsize="x-small", gid="footer")
    renderer = fig.canvas.get_renderer()

    def inches(artist: Any) -> float:
        return artist.get_window_extent(renderer).height / fig.dpi if artist else 0.0

    def arrange(coincident: Mapping[tuple[int, str, str], Sequence[tuple[str, str]]]) -> None:
        # The footer's first lines are the sets' "Left out" lines: each gains its
        # set's coincident series.
        lines = list(footer)
        for index in range(len(sets)):
            clause = coincident_text(coincident, index, axes, modes)
            if clause:
                lines[index] = f"{lines[index]} {clause}"
        lines = [
            part
            for line in lines
            for part in textwrap.wrap(line, int(width * 17), break_on_hyphens=False)
        ]
        footer_text.set_text("\n".join(lines))
        title_h, legend_h, footer_h = inches(title_text), inches(legend), inches(footer_text)
        legend_band = GAP + legend_h if legend is not None else 0.0  # below the title
        # tight_layout keeps LAYOUT_PAD (a fraction of the font size) between its rect
        # and the panels' outer edges (tick labels, axis labels): that is the gap there.
        height = MARGIN + title_h + legend_band + PANEL_HEIGHT * len(metrics) + footer_h + MARGIN
        fig.set_size_inches(width, height)
        title_text.set_y(1 - MARGIN / height)
        if legend is not None:
            legend.set_bbox_to_anchor((0.5, 1 - (MARGIN + title_h + GAP) / height), fig.transFigure)
        footer_text.set_y(MARGIN / height)
        top = 1 - (MARGIN + title_h + legend_band) / height
        _layout(fig, (0, (MARGIN + footer_h) / height, 1, top))

    # Coincidence is measured in display points, so on the laid-out panels; the footer
    # it names changes the layout. A first guess from the panels as created, then
    # until the layout and what it shows agree.
    coincident = stack_coincident(fig, drawn)
    for _ in range(COINCIDENT_PASSES):
        arrange(coincident)
        found = stack_coincident(fig, drawn)
        if found == coincident:
            break
        coincident = found
    else:
        # Not settled (a pair at the threshold): the footer names what the last
        # layout showed.
        arrange(coincident)
        stack_coincident(fig, drawn)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(plotted)
    return path


def _layout(fig: Any, rect: tuple[float, float, float, float]) -> None:
    """``tight_layout`` into ``rect`` until the panels stop moving.

    One pass is not enough: tight_layout sizes the panels from the tick labels it
    sees, and a resized panel can get other tick labels ("0.95" for "1.0"), wider
    ones pushing the y label past the figure's edge. Each pass also wraps a y label
    longer than its panel, which tight_layout does not count along its length.
    """
    previous = None
    for _ in range(LAYOUT_PASSES):
        fig.tight_layout(rect=rect, pad=LAYOUT_PAD, h_pad=PANEL_PAD, w_pad=1.08)
        fig.canvas.draw()  # the tick labels of the new panel sizes
        wrapped = _fit_ylabels(fig)
        # In pixels; tight_layout's own rounding moves panels by a fraction of one.
        width, height = fig.bbox.width, fig.bbox.height
        positions = [
            value * size
            for ax in fig.axes
            for value, size in zip(ax.get_position().bounds, (width, height) * 2, strict=True)
        ]
        if not wrapped and previous is not None:
            if max(abs(a - b) for a, b in zip(positions, previous, strict=True)) < 0.5:
                return
        previous = positions
    # The passes ran out with a change (a label just wrapped) no tight_layout has seen.
    fig.tight_layout(rect=rect, pad=LAYOUT_PAD, h_pad=PANEL_PAD, w_pad=1.08)


def ylabel_text(label: str) -> str:
    """A y label with its part in parentheses on a second line, e.g. "speedup" over
    "(baseline / impl time)"; a label without one stays on one line (the user,
    inbox 23)."""
    head, sep, tail = label.partition(" (")
    if not sep or not tail.endswith(")"):
        return label
    return f"{head}\n({tail}"


def _fit_ylabels(fig: Any) -> bool:
    """Wrap further each y label still longer than its panel is high: its longest
    line in two, keeping the split before the parentheses; whether any changed."""
    renderer = fig.canvas.get_renderer()
    changed = False
    for ax in fig.axes:
        label = ax.yaxis.label
        text = label.get_text()
        if not text or label.get_window_extent(renderer).height <= ax.bbox.height:
            continue
        lines = text.split("\n")
        longest = max(range(len(lines)), key=lambda i: len(lines[i]))
        parts = textwrap.wrap(lines[longest], math.ceil(len(lines[longest]) / 2) + 1)
        if len(parts) < 2:
            continue  # one word: nothing left to wrap
        lines[longest : longest + 1] = parts
        label.set_text("\n".join(lines))
        changed = True
    return changed


# --- the reduction setting --------------------------------------------------------------------

# The strict bar of a pair: the default's colour, hatched.
STRICT_HATCH = "///"
# Each bar's value above it (F9, I1-12: a bar too short to see still reads), in
# points: its size below the tick labels', and its distance from the bar.
BAR_LABEL_SMALLER = 2
BAR_LABEL_PAD = 2
REDUCTION_SETS = ("default", "strict")


def reduction_parity(entries: Sequence[Mapping[str, Any]], mode: str) -> str | None:
    """The parity metric of the reduction figure: the mode's default one when an
    entry has it in both sets, else the first one some entry has; ``None``: none."""
    names = sorted({name for e in entries for name in e["parity"]})
    preferred = DEFAULT_PARITY[mode].removeprefix(analysis.PARITY_PREFIX)
    if preferred in names:
        return preferred
    return names[0] if names else None


def _reduction_value(entry: Mapping[str, Any], metric: str, which: str) -> float | None:
    if not entry["measured"]:
        return None
    if metric == "median_ms":
        return entry[f"median_{which}"]
    value = entry["parity"].get(metric)
    return None if value is None else value[which]


def reduction_figure(
    path: Path,
    entries: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    point: Mapping[str, Any],
    palette: Mapping[str, str],
    title: str,
    footer: Sequence[str],
    baseline: str,
) -> Path:
    """Per entry a pair of bars, default and strict, for each of ``metrics`` (median
    time, a parity metric) one panel above the other, each on a linear axis from
    zero; the CSV holds the bars."""
    plt = pyplot()
    from matplotlib.patches import Patch

    names = [e["implementation"] for e in entries]
    width = max(PANEL_WIDTH * 1.5, 0.6 * len(names) + 1.5)
    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(width, PANEL_HEIGHT * len(metrics)), squeeze=False, sharex=True
    )
    plotted = []
    labels: dict[Any, list[tuple[Any, float]]] = {}  # per panel: (label, bar value)
    size = _font_points(plt.rcParams["ytick.labelsize"]) - BAR_LABEL_SMALLER
    xs = list(range(len(names)))
    for ax, metric in zip((row[0] for row in axes), metrics, strict=True):
        parity = metric != "median_ms"
        exponent = y_exponent(
            [_reduction_value(e, metric, which) for e in entries for which in REDUCTION_SETS]
        )
        unit = 10.0**exponent
        for x, entry in zip(xs, entries, strict=True):
            color = palette.get(entry["implementation"], "#7f7f7f")
            for offset, which in zip((-0.2, 0.2), REDUCTION_SETS, strict=True):
                y = _reduction_value(entry, metric, which)
                if y is None:
                    continue  # nothing to draw
                strict_bar = which == "strict"
                bars = ax.bar(
                    x + offset,
                    y / unit,
                    width=0.4,
                    color="white" if strict_bar else color,
                    edgecolor=color,
                    hatch=STRICT_HATCH if strict_bar else None,
                )
                (text,) = ax.bar_label(
                    bars, labels=[f"{y:.3g}"], padding=BAR_LABEL_PAD, fontsize=size
                )
                labels.setdefault(ax, []).append((text, y / unit))
                row = entry[which]
                plotted.append(
                    {
                        "figure": path.stem,
                        "set": which,
                        "metric": metric if not parity else analysis.PARITY_PREFIX + metric,
                        "axis": "implementation",
                        "x": entry["implementation"],
                        "implementation": entry["implementation"],
                        "env_name": row.get("env_name"),
                        "role": role_of(row, entry["implementation"], baseline),
                        "y": y,
                        "parity_status": row.get("parity_status"),
                        "parity_fail": row.get("parity_status") == "fail",
                        "parity_fail_metric": parity_fail_metric(
                            row, analysis.PARITY_PREFIX + metric if parity else metric
                        ),
                        "dtype": entry["dtype"],
                        "mode": entry["mode"],
                        "point": analysis.point_text(point),
                        "work": work_text(row),
                        "baseline_work": None,
                    }
                )
        label = METRIC_LABELS.get(metric, metric) if not parity else metric
        ax.set_ylabel(ylabel_text(scaled_label(label, exponent)))
        _plain_ticks(ax)
        # Linear from zero (R1-26 F1): a bar's length is its value, so the pair's
        # lengths compare as the values do, and a zero parity error shows as zero.
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)
    bottom = axes[-1][0]
    bottom.set_xticks(xs)
    bottom.set_xticklabels(
        [
            n if e["measured"] else f"{n} (not measured)"
            for n, e in zip(names, entries, strict=True)
        ],
        rotation=45,
        ha="right",
    )
    handles = [
        Patch(facecolor="#7f7f7f", edgecolor="#7f7f7f", label="default"),
        Patch(facecolor="white", edgecolor="#7f7f7f", hatch=STRICT_HATCH, label="strict"),
    ]
    # Bands, top to bottom, as in _figure: title, legend, panels (with their tick
    # labels), footer, each as high as its text, measured.
    title_lines = textwrap.wrap(title, int(width * TITLE_CHARS_PER_INCH), break_on_hyphens=False)
    title_text = fig.text(
        0.5, 1, "\n".join(title_lines), ha="center", va="top", fontsize="large", gid="title"
    )
    legend = fig.legend(
        handles=handles, loc="upper center", ncol=2, borderaxespad=0, fontsize="small"
    )
    lines = [
        part
        for line in footer
        for part in textwrap.wrap(line, int(width * 17), break_on_hyphens=False)
    ]
    footer_text = fig.text(
        0.01, 0, "\n".join(lines), ha="left", va="bottom", fontsize="x-small", gid="footer"
    )
    renderer = fig.canvas.get_renderer()

    def inches(artist: Any) -> float:
        return artist.get_window_extent(renderer).height / fig.dpi

    title_h, legend_h, footer_h = inches(title_text), inches(legend), inches(footer_text)
    labels_h = max(inches(label) for label in bottom.get_xticklabels())
    height = (
        MARGIN + title_h + GAP + legend_h + PANEL_HEIGHT * len(metrics) + labels_h + footer_h
    ) + MARGIN
    fig.set_size_inches(width, height)
    title_text.set_y(1 - MARGIN / height)
    legend.set_bbox_to_anchor((0.5, 1 - (MARGIN + title_h + GAP) / height), fig.transFigure)
    footer_text.set_y(MARGIN / height)
    top = 1 - (MARGIN + title_h + GAP + legend_h) / height
    rect = (0, (MARGIN + footer_h) / height, 1, top)
    _layout(fig, rect)
    # The labels rotated when two would overlap, and room for them above the bars:
    # each change resizes the panels, so lay out again.
    for _ in range(LAYOUT_PASSES):
        if not _fit_bar_labels(fig, labels):
            break
        _layout(fig, rect)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(plotted)
    return path


def _font_points(size: Any) -> float:
    from matplotlib.font_manager import FontProperties

    return FontProperties(size=size).get_size_in_points()


def _fit_bar_labels(fig: Any, labels: Mapping[Any, Sequence[tuple[Any, float]]]) -> bool:
    """Rotate every bar label upright when two of a panel overlap side by side, and
    raise each panel's top until its labels fit inside it; whether anything changed."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    changed = False
    texts = [text for these in labels.values() for text, _ in these]
    if texts and texts[0].get_rotation() == 0:
        for these in labels.values():
            boxes = sorted(tuple(text.get_window_extent(renderer).intervalx) for text, _ in these)
            if any(b[0] < a[1] + 1 for a, b in itertools.pairwise(boxes)):
                for text in texts:
                    text.set_rotation(90)
                fig.canvas.draw()
                changed = True
                break
    pad = BAR_LABEL_PAD * fig.dpi / 72
    for ax, these in labels.items():
        height = ax.bbox.height
        bottom, top = ax.get_ylim()
        # A bar of value v reaches v / top of the panel; its label, pad and the
        # label's height above that.
        needed = max(
            v * height / max(height - pad - text.get_window_extent(renderer).height, 1)
            for text, v in these
        )
        if needed > top * 1.001:
            ax.set_ylim(bottom, needed * 1.02)
            changed = True
    return changed


def plot_reduction(
    recs: Sequence[Mapping[str, Any]],
    recs_strict: Sequence[Mapping[str, Any]],
    out: Path,
    point: Mapping[str, Any],
    baseline: str,
    palette: Mapping[str, str],
    *,
    dtypes: Sequence[str],
    modes: Sequence[str],
    impls: Sequence[str] | None = None,
) -> list[Path]:
    """``reduction-<mode>[-<dtype>].png`` per dtype and mode with entries."""
    entries = analysis.reduction_rows(recs, recs_strict, point, baseline=baseline)
    if impls:
        keep = set(impls) | {baseline}
        entries = [e for e in entries if e["implementation"] in keep]
    rows = analysis.to_rows(recs)
    rows_strict = analysis.to_rows(recs_strict)
    footer = [
        f"strict: no reduced-precision reduction in bf16/fp16 GEMMs (hatched); "
        f"at {analysis.point_text(point)}",
        provenance(rows, "default"),
        provenance(rows_strict, "strict"),
    ]
    for label, these, expected in (("default", recs, "default"), ("strict", recs_strict, "strict")):
        odd = analysis.reduction_mismatches(these, expected)
        if odd:
            footer.append(f"Warning: {len(odd)} record(s) of the {label} set have another setting.")
    written = []
    for d in dtypes:
        for m in modes:
            selected = [e for e in entries if e["dtype"] == d and e["mode"] == m]
            if not selected:
                continue
            parity = reduction_parity(selected, m)
            metrics = ["median_ms", *([parity] if parity else [])]
            suffix = f"{m}-{d}" if len(dtypes) > 1 else m
            title = f"{recs[0]['op']} on {recs[0]['device_slug']}: {m}, {d}: default against strict"
            written.append(
                reduction_figure(
                    out / f"reduction-{suffix}.png",
                    selected,
                    metrics,
                    point,
                    palette,
                    title,
                    footer,
                    baseline,
                )
            )
    return written


# --- time at equal peak ------------------------------------------------------------------------

# The budgets' vertical lines: thin, grey, dotted, under the points.
BUDGET_STYLE = {"color": "#7f7f7f", "linestyle": ":", "linewidth": 1.0}


def equal_peak_panel(
    found: Mapping[str, Any],
    baseline: str,
    palette: Mapping[str, str],
    figure: str,
    point: Mapping[str, Any],
) -> Callable[[Any, dict[str, Any]], list[dict[str, Any]]]:
    """The panel of one ``analysis.equal_peak`` result: time against peak above
    baseline, one line per base entry through its points (default and variants, by
    peak; hollow when the record fails parity), the baseline as a reference point and
    the budgets as vertical lines. No point is labelled with its parameter: the CSV
    names each one's entry."""
    from matplotlib.lines import Line2D

    def csv_row(row: Row, role: str, x: Any, y: Any) -> dict[str, Any]:
        if role == "budget":  # a line, not a record's point
            row = {"implementation": row["implementation"]}
        fail = row.get("parity_status") == "fail"
        return {
            "figure": figure,
            "set": "",
            "metric": "median_ms" if role != "budget" else "budget",
            "axis": "peak_above_baseline",
            "x": x,
            "implementation": row["implementation"],
            "env_name": row.get("env_name"),
            "role": role,
            "y": y,
            "parity_status": row.get("parity_status"),
            "parity_fail": fail if role != "budget" else None,
            "parity_fail_metric": None,
            "dtype": found["dtype"],
            "mode": found["mode"],
            "point": analysis.point_text(point),
            "work": work_text(row) if role != "budget" else None,
            "baseline_work": None,
        }

    def draw(ax: Any, handles: dict[str, Any]) -> list[dict[str, Any]]:
        plotted = []
        base = found["baseline"]
        factors = ", ".join(f"{f:g}×" for f in analysis.EQUAL_PEAK_BUDGETS)
        for factor, budget in zip(
            analysis.EQUAL_PEAK_BUDGETS, found["budgets"] or (), strict=False
        ):
            ax.axvline(budget / MIB, zorder=1, **BUDGET_STYLE)
            plotted.append(csv_row(base, "budget", budget / MIB, None))
            plotted[-1]["implementation"] = f"{factor:g}× {baseline}"
        series =[("baseline", baseline, [base] if analysis.peak_point(base) else [])]
        series += [("counterpart", e["base"], e["points"]) for e in found["entries"]]
        for role, name, points in series:
            if not points:
                continue
            style = STYLE[role]
            color = palette.get(name, "#7f7f7f")
            xs = [r["peak_above_baseline"] / MIB for r in points]
            ys = [r["median_ms"] for r in points]
            z = 3 if role == "baseline" else 2
            if role != "baseline":
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linestyle=style["linestyle"],
                    zorder=z,
                    linewidth=style["linewidth"],
                )
            for r, x, y in zip(points, xs, ys, strict=True):
                fail = r.get("parity_status") == "fail"
                ax.plot(
                    [x],
                    [y],
                    linestyle="none",
                    marker=style["marker"],
                    color=color,
                    markersize=style["size"] + (2 if fail else 0),
                    markerfacecolor="none" if fail else color,
                    markeredgecolor=color,
                    zorder=z,
                )
                plotted.append(csv_row(r, role, x, y))
            label = f"{name} (baseline)" if role == "baseline" else name
            handles.setdefault(
                label,
                Line2D(
                    [],
                    [],
                    color=color,
                    linestyle="none" if role == "baseline" else style["linestyle"],
                    linewidth=style["linewidth"],
                    marker=style["marker"],
                    markersize=style["size"],
                    label=label,
                ),
            )
        if found["budgets"]:  # last in the legend: baseline, entries, budgets
            label = f"budgets: {factors} {baseline}'s peak"
            handles.setdefault(label, Line2D([], [], label=label, **BUDGET_STYLE))
        ax.set_xlabel(METRIC_LABELS["peak_above_baseline"])
        ax.set_ylabel(ylabel_text(METRIC_LABELS["median_ms"]))
        _plain_ticks(ax)
        _plain_ticks(ax, "x")
        ax.grid(True, alpha=0.3)
        return plotted

    return draw


def plot_equal_peak(
    rows: Sequence[Row],
    variant_rows: Sequence[Row],
    out: Path,
    point: Mapping[str, Any],
    baseline: str,
    palette: Mapping[str, str],
    *,
    title: str,
    notes: Sequence[Any],
    axes: Sequence[str],
    dtypes: Sequence[str],
    modes: Sequence[str],
    impls: Sequence[str] | None = None,
) -> list[Path]:
    """``equal-peak-<mode>[-<dtype>].png`` per dtype and mode with variants at
    ``point`` (``analysis.equal_peak``, as the report's "Time at equal peak"); with
    ``impls``, the base entries named, or one of whose variants is."""
    written = []
    for found in analysis.equal_peak(rows, variant_rows, point, baseline):
        d, m = found["dtype"], found["mode"]
        if d not in dtypes or m not in modes:
            continue
        variants = analysis.at(variant_rows, dtype=d, mode=m)
        if impls:
            named = set(impls) | {r["base"] for r in variants if r["implementation"] in impls}
            found = {**found, "entries": [e for e in found["entries"] if e["base"] in named]}
        bases = {e["base"] for e in found["entries"]}
        here = [
            r
            for r in [found["baseline"], *(e["default"] for e in found["entries"]), *variants]
            if r is not None and (r.get("base") is None or r["base"] in bases)
            if all(r.get(k) == v for k, v in point.items())
        ]
        suffix = f"{m}-{d}" if len(dtypes) > 1 else m
        path = out / f"equal-peak-{suffix}.png"
        factors = ", ".join(f"{f:g}×" for f in analysis.EQUAL_PEAK_BUDGETS)
        footer = [
            left_out(here, ["median_ms"], notes, axes),
            f"Budgets (dotted): {factors} {baseline}'s peak above baseline; each line is "
            f"one entry's default and variants, at {analysis.point_text(point)}.",
            provenance(here),
        ]
        written.append(
            _figure(
                path,
                [(None, here, None)],
                ["median_ms"],
                ["peak_above_baseline"],
                point,
                baseline,
                f"{title}: {m}, {d}: time at equal peak",
                footer,
                palette=palette,
                panel=equal_peak_panel(found, baseline, palette, path.stem, point),
                panel_width=2 * PANEL_WIDTH,
            )
        )
    return written


# --- the command -----------------------------------------------------------------------------


def _check(names: Sequence[str] | None, known: Sequence[str], what: str) -> None:
    unknown = [n for n in names or () if n not in known]
    if unknown:
        raise ValueError(f"unknown {what} {unknown}; known: {', '.join(known) or 'none'}")


def _load(results_dir: str | Path, op: str | None, device_slug: str | None) -> list[Row]:
    with warnings.catch_warnings():
        warnings.simplefilter("default")  # the merge warning reaches stderr, once
        recs = analysis.load(results_dir, op=op, device_slug=device_slug)
    if not recs:
        raise ValueError(f"no records under {results_dir}")
    for what, values, flag in (
        ("op", {r["op"] for r in recs}, "--op"),
        ("device slug", {r["device_slug"] for r in recs}, "--device-slug"),
    ):
        if len(values) > 1:
            raise ValueError(
                f"the results hold several {what}s ({', '.join(sorted(values))}): {flag}"
            )
    return recs


def plot(
    results_dir: str | Path,
    out_dir: str | Path,
    *,
    op: str | None = None,
    device_slug: str | None = None,
    dtype: str | None = None,
    mode: str | None = None,
    baseline: str = "torch_eager",
    point: Mapping[str, str] | None = None,
    metrics: Sequence[str] | None = None,
    sweep: Sequence[str] | None = None,
    impls: Sequence[str] | None = None,
    overlay: str | Path | None = None,
    labels: tuple[str, str] = ("A", "B"),
    strict: str | Path | None = None,
) -> list[Path]:
    """Write the figures and their CSV sidecars under ``out_dir``; return the PNGs.
    With ``strict`` (a strict run's results), the reduction figures too."""
    recs = _load(results_dir, op, device_slug)
    recs_strict = [] if strict is None else _load(strict, recs[0]["op"], recs[0]["device_slug"])
    recs_strict = analysis.split_variants(recs_strict)[0]
    # The variants are drawn in the equal-peak figures, and in the others only when
    # --impl names them.
    base_recs, variant_recs = analysis.split_variants(recs)
    variant_rows = analysis.to_rows(variant_recs)
    named = [r for r in variant_recs if r["implementation"] in (impls or ())]
    rows = analysis.to_rows(base_recs + named)
    axes_all = analysis.axes_of(recs)
    notes = analysis.parity_notes(recs)
    point = analysis.default_point(base_recs or recs, point=point)["params"]
    rows_b: list[Row] = []
    if overlay is not None:
        recs_b = _load(overlay, op or recs[0]["op"], device_slug)
        rows_b = analysis.to_rows(recs_b)
    # One palette for every figure of this invocation, from a stable order, before
    # --impl narrows the rows: an implementation's colour does not depend on what
    # else is drawn.
    rows_strict = analysis.to_rows(recs_strict) if recs_strict else []
    palette = colors(rows + rows_b + rows_strict, baseline, stable_order(recs))
    # The controls' markers likewise, from every environment in the records: with
    # --impl, a control keeps the marker its environment has without it (R1-13 F3).
    markers = control_markers(rows + rows_b)
    base_rows = [r for r in rows if r.get("base") is None]
    if impls:
        known = rows + rows_b + variant_rows
        _check(impls, list(dict.fromkeys(r["implementation"] for r in known)), "implementation")
        keep = set(impls) | {baseline}
        rows = [r for r in rows if r["implementation"] in keep]
        rows_b = [r for r in rows_b if r["implementation"] in keep]
    dtypes = [dtype] if dtype else list(dict.fromkeys(row["dtype"] for row in rows))
    modes = [mode] if mode else [m for m in MODES if any(row["mode"] == m for row in rows)]
    _check(metrics, known_metrics(rows), "metric")
    _check(sweep, axes_all, "sweep axis")
    out = Path(out_dir)
    written = []
    for d in dtypes:
        for m in modes:
            selected = analysis.at(rows, dtype=d, mode=m)
            if not selected:
                continue
            axes = [a for a in swept_axes(selected, point) if not sweep or a in sweep]
            if not axes:
                warnings.warn(
                    f"{m}, {d}: no axis varies around the default point "
                    f"{analysis.point_text(point)}: no figures (report has a table per case)",
                    stacklevel=2,
                )
                continue
            # The rows the grid draws: those of the sweeps around the point.
            scope = [r for axis in axes for r in analysis.sweeps(selected, point)[axis]]
            scope = list({id(r): r for r in scope}.values())
            chosen = list(metrics) if metrics else default_metrics(scope, m)
            # --metric is checked against every row; a mode may lack a metric (fwd has
            # no gradient errors), or have it only away from the sweeps (R1-22 R1): left
            # out of that mode, said once per mode and dtype.
            present = set(known_metrics(scope))
            for metric in [c for c in chosen if c not in present]:
                warnings.warn(f"{m}, {d}: no values of {metric}: left out", stacklevel=2)
            chosen = [c for c in chosen if c in present]
            if not chosen:
                warnings.warn(f"{m}, {d}: none of the metrics has values: no grid", stacklevel=2)
            suffix = f"{m}-{d}" if len(dtypes) > 1 else m
            title = f"{recs[0]['op']} on {recs[0]['device_slug']}: {m}, {d}"
            footer = [left_out(scope, chosen, notes, axes_all), provenance(selected)]
            if chosen:
                written.append(
                    _figure(
                        out / f"grid-{suffix}.png",
                        [(None, selected, None)],
                        chosen,
                        axes,
                        point,
                        baseline,
                        title,
                        footer,
                        palette=palette,
                        markers=markers,
                    )
                )
            ratios = ratio_rows(selected, baseline, axes_all)
            ratio_metrics = ["speedup", "memory_ratio", "time_x_peak_ratio"]
            # Within a case every implementation usually has the same work, so the
            # throughput ratio would repeat the speedup (R1-12 R2).
            if work_differs(ratios):
                ratio_metrics.append("throughput_ratio")
            written.append(
                _figure(
                    out / f"ratio-{suffix}.png",
                    [(None, ratios, None)],
                    ratio_metrics,
                    axes,
                    point,
                    baseline,
                    f"{title}: against {baseline}",
                    [
                        left_out(
                            [r for r in scope if r["implementation"] != baseline],
                            [],
                            notes,
                            axes_all,
                        ),
                        provenance(selected),
                    ],
                    palette=palette,
                    markers=markers,
                    ratio_line=True,
                )
            )
            if overlay is not None and chosen:
                selected_b = analysis.at(rows_b, dtype=d, mode=m)
                scope_b = [r for axis in axes for r in analysis.sweeps(selected_b, point)[axis]]
                scope_b = list({id(r): r for r in scope_b}.values())
                a, b = labels
                written.append(
                    _figure(
                        out / f"overlay-{suffix}.png",
                        [
                            (a, selected, OVERLAY_LINESTYLE[0]),
                            (b, selected_b, OVERLAY_LINESTYLE[1]),
                        ],
                        chosen,
                        axes,
                        point,
                        baseline,
                        f"{title}: {a} (solid) against {b} (dashed)",
                        [
                            f"{a}: {left_out(scope, chosen, notes, axes_all)}",
                            f"{b}: {left_out(scope_b, chosen, notes, axes_all)}",
                            provenance(selected, a),
                            provenance(selected_b, b),
                        ],
                        palette=palette,
                        markers=markers,
                    )
                )
    written += plot_equal_peak(
        base_rows,
        variant_rows,
        out,
        point,
        baseline,
        palette,
        title=f"{recs[0]['op']} on {recs[0]['device_slug']}",
        notes=notes,
        axes=axes_all,
        dtypes=dtypes,
        modes=modes,
        impls=impls,
    )
    if strict is not None:
        written += plot_reduction(
            base_recs,
            recs_strict,
            out,
            point,
            baseline,
            palette,
            dtypes=dtypes,
            modes=modes,
            impls=impls,
        )
    return written
