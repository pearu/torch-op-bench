"""``torch-op-bench plot``: the figures of the write-up (R8).

Three figures per mode (and dtype, when the results hold several): the grid
(metrics × swept axes), the ratio against the baseline, and, with a second results
directory, the overlay of both. Next to every PNG, a CSV holds exactly the plotted
points, so the write-up and the tests read what was drawn. matplotlib is imported
lazily, with the Agg backend set in code: no display is needed.
"""

from __future__ import annotations

import csv
import math
import textwrap
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
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
# LAYOUT_PAD times the font size (10 pt: 3 pt). Every such gap is at most a capital
# letter of the axis font (the tick and axis labels, 10 pt): about 7.3 pt.
MARGIN = 0.05
GAP = 3 / 72
LAYOUT_PAD = 0.3
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
    """The spec's implementations, controls left out, when the op resolves."""
    spec = analysis.spec_of(recs[0]) if recs else None
    if spec is None:
        return None
    return [i.name for i in spec.implementations if i.info.get("role") != "control"]


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


def left_out(rows: Sequence[Row], metrics: Sequence[str]) -> str:
    """How many of the figure's records are not drawn, and why."""
    counts = Counter(row["status"] for row in rows if row.get("status") != "ok")
    skipped = sum(
        1
        for row in rows
        if row.get("status") == "ok" and row.get("parity_status") not in ("ok", "fail")
    )
    parts = [f"{n} {status}" for status, n in sorted(counts.items())]
    if skipped and any(m.startswith(analysis.PARITY_PREFIX) for m in metrics):
        parts.append(f"{skipped} without parity (parity rows only)")
    if not parts:
        return "Every record of this figure is drawn."
    return f"Left out: {', '.join(parts)} (gaps in the lines)."


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
) -> list[dict[str, Any]]:
    """One metric against one axis, for every result set; returns the CSV rows."""
    from matplotlib.lines import Line2D

    all_swept = [analysis.sweeps(rows, point)[axis] for _, rows, _ in sets]
    values = analysis.axis_values([r for swept in all_swept for r in swept], axis)
    xs, log = _positions(values)
    plotted = []
    for (label, _, linestyle), swept in zip(sets, all_swept, strict=True):
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
                (fail_pts if fail else ok_pts).append((x, y))
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
                        "dtype": row.get("dtype"),
                        "mode": row.get("mode"),
                        "point": ", ".join(f"{k}={row.get(k)}" for k in point),
                        "work": work_text(row),
                        "baseline_work": row.get("baseline_work"),
                    }
                )
            line = linestyle or style["linestyle"]
            # The baseline on top: a counterpart with equal values must not hide it.
            z = 3 if role == "baseline" else 2
            ax.plot(xs, ys, color=color, linestyle=line, linewidth=style["linewidth"], zorder=z)
            for pts, face in ((ok_pts, color), (fail_pts, "none")):
                if pts:
                    ax.plot(
                        [p[0] for p in pts],
                        [p[1] for p in pts],
                        linestyle="none",
                        marker=style["marker"],
                        markersize=style["size"] + (2 if face == "none" else 0),
                        color=color,
                        markerfacecolor=face,
                        markeredgecolor=color,
                        zorder=z,
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
    if log:
        ax.set_xscale("log", base=2)
        ax.minorticks_off()
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in values])
    ax.set_xlabel(axis)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric.removeprefix(analysis.PARITY_PREFIX)))
    if metric.startswith(analysis.PARITY_PREFIX):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    return plotted


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
) -> Path:
    plt = pyplot()
    union = [row for _, rows, _ in sets for row in rows]
    names = analysis.implementations(union, baseline)
    palette = palette if palette is not None else colors(union, baseline)
    markers = markers if markers is not None else control_markers(union)
    width = PANEL_WIDTH * max(1, len(axes))
    fig, grid = plt.subplots(
        len(metrics), max(1, len(axes)), figsize=(width, PANEL_HEIGHT * len(metrics)), squeeze=False
    )
    handles: dict[str, Any] = {}
    plotted: list[dict[str, Any]] = []
    for i, metric in enumerate(metrics):
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
            )
            if ratio_line:
                base = STYLE["baseline"]
                ax.axhline(
                    1.0,
                    color=base["color"],
                    linestyle=base["linestyle"],
                    linewidth=base["linewidth"],
                )
    # Bands in inches, top to bottom: title, legend, panels, footer. The title and the
    # footer wrap to the figure's width, so neither is cut off in a one-column figure.
    # Each band is as high as its rendered text, measured, and the bands are stacked
    # with GAP between them, so the white space between title, legend, panels and
    # footer is at most about the height of a capital letter of that text.
    columns = max(1, min(4, int(width // 2.2)))
    lines = [
        part
        for line in footer
        for part in textwrap.wrap(line, int(width * 17), break_on_hyphens=False)
    ]
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
    footer_text = fig.text(
        0.01, 0, "\n".join(lines), ha="left", va="bottom", fontsize="x-small", gid="footer"
    )
    renderer = fig.canvas.get_renderer()

    def inches(artist: Any) -> float:
        return artist.get_window_extent(renderer).height / fig.dpi if artist else 0.0

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
        fig.tight_layout(rect=rect, pad=LAYOUT_PAD, h_pad=1.08, w_pad=1.08)
        fig.canvas.draw()  # the tick labels of the new panel sizes
        wrapped = _fit_ylabels(fig)
        positions = [ax.get_position().bounds for ax in fig.axes]
        if not wrapped and positions == previous:
            return
        previous = positions


def _fit_ylabels(fig: Any) -> bool:
    """Wrap each y label that is longer than its panel is high; whether any changed."""
    renderer = fig.canvas.get_renderer()
    changed = False
    for ax in fig.axes:
        label = ax.yaxis.label
        text = label.get_text()
        if not text or label.get_window_extent(renderer).height <= ax.bbox.height:
            continue
        words = " ".join(text.split("\n"))
        lines = text.count("\n") + 2
        label.set_text("\n".join(textwrap.wrap(words, math.ceil(len(words) / lines) + 1)))
        changed = True
    return changed


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
) -> list[Path]:
    """Write the figures and their CSV sidecars under ``out_dir``; return the PNGs."""
    recs = _load(results_dir, op, device_slug)
    rows = analysis.to_rows(recs)
    axes_all = analysis.axes_of(recs)
    point = analysis.default_point(recs, point=point)["params"]
    rows_b: list[Row] = []
    if overlay is not None:
        recs_b = _load(overlay, op or recs[0]["op"], device_slug)
        rows_b = analysis.to_rows(recs_b)
    # One palette for every figure of this invocation, from a stable order, before
    # --impl narrows the rows: an implementation's colour does not depend on what
    # else is drawn.
    palette = colors(rows + rows_b, baseline, stable_order(recs))
    # The controls' markers likewise, from every environment in the records: with
    # --impl, a control keeps the marker its environment has without it (R1-13 F3).
    markers = control_markers(rows + rows_b)
    if impls:
        _check(
            impls, list(dict.fromkeys(r["implementation"] for r in rows + rows_b)), "implementation"
        )
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
            chosen = list(metrics) if metrics else default_metrics(selected, m)
            suffix = f"{m}-{d}" if len(dtypes) > 1 else m
            scope = [r for axis in axes for r in analysis.sweeps(selected, point)[axis]]
            scope = list({id(r): r for r in scope}.values())
            title = f"{recs[0]['op']} on {recs[0]['device_slug']}: {m}, {d}"
            footer = [left_out(scope, chosen), provenance(selected)]
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
                        left_out([r for r in scope if r["implementation"] != baseline], []),
                        provenance(selected),
                    ],
                    palette=palette,
                    markers=markers,
                    ratio_line=True,
                )
            )
            if overlay is not None:
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
                            f"{a}: {left_out(scope, chosen)}",
                            f"{b}: {left_out(scope_b, chosen)}",
                            provenance(selected, a),
                            provenance(selected_b, b),
                        ],
                        palette=palette,
                        markers=markers,
                    )
                )
    return written
