import csv
import itertools
import math
import re
import struct
import sys

import pytest
import synth_results as S

from torch_op_bench import analysis, cli, plot
from torch_op_bench.spec import ParityNote

pytest.importorskip("matplotlib")


@pytest.fixture(autouse=True)
def headless(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("MPLBACKEND", raising=False)


def png_size(path):
    data = path.read_bytes()[:24]
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def draw(synth, out, **kwargs):
    with pytest.warns(UserWarning, match="different torch-op-bench commits"):
        return plot.plot(synth, out, baseline="base", **kwargs)


def test_figures_and_backend(synth, tmp_path):
    written = draw(synth, tmp_path / "out", dtype="float32")
    names = [p.name for p in written]
    assert names == ["grid-fwd.png", "ratio-fwd.png", "grid-fwd_bwd.png", "ratio-fwd_bwd.png"]
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"
    # 4 swept axes x 3.8 in wide; 4 metrics x 2.8 in, plus the title, legend and
    # footer bands (measured: test_gaps_between_bands), high; at 150 dpi.
    width, height = png_size(tmp_path / "out" / "grid-fwd_bwd.png")
    assert width == 4 * 3.8 * 150
    assert 4 * 2.8 * 150 < height < (4 * 2.8 + 1.2) * 150
    for path in written:
        assert path.with_suffix(".csv").exists()


def test_grid_csv_is_what_was_drawn(synth, tmp_path):
    draw(synth, tmp_path, dtype="float32", mode="fwd_bwd")
    rows = read_csv(tmp_path / "grid-fwd_bwd.csv")
    metrics = list(dict.fromkeys(r["metric"] for r in rows))
    # The synthetic op has no grad_hidden metric: its first parity metric is drawn.
    assert metrics == ["median_ms", "tokens_per_s", "peak_above_baseline", "parity_grad_rel"]
    medians = [r for r in rows if r["metric"] == "median_ms"]
    for r in medians:
        params = {**S.DEFAULT, r["axis"]: type(S.DEFAULT[r["axis"]])(r["x"])}
        expected = S.median_ms(r["implementation"], "fwd_bwd", "float32", params)
        assert float(r["y"]) == pytest.approx(expected), r
    # 4 implementations x (3 + 2 + 2 + 2) points, the default point in each axis' panel;
    # less the unavailable and the missing point.
    assert len(medians) == 4 * 9 - 2
    peaks = [r for r in rows if r["metric"] == "peak_above_baseline"]
    for r in peaks:
        params = {**S.DEFAULT, r["axis"]: type(S.DEFAULT[r["axis"]])(r["x"])}
        assert float(r["y"]) == pytest.approx(S.peak_above(r["implementation"], params) / S.MIB)
    fails = [r for r in rows if r["parity_fail"] == "True"]
    assert {(r["implementation"], r["axis"], r["x"]) for r in fails} == {("chunked", "M", "256")}
    assert {r["parity_status"] for r in fails} == {"fail"}
    roles = {r["implementation"]: r["role"] for r in rows}
    assert roles == {
        "base": "baseline",
        "compiled": "counterpart",
        "chunked": "counterpart",
        "base_other": "control",
    }


def test_hollow_marker_per_parity_metric(synth, tmp_path, monkeypatch):
    # F8 (I1-12): in a parity panel a marker is hollow when that metric failed, not
    # when any did. chunked at M=256 fails only grad_rel: its marker is filled in the
    # loss_rel_err panel, hollow in the grad_rel panel and in the time panel.
    import matplotlib.pyplot as plt

    closed = []
    monkeypatch.setattr(plt, "close", closed.append)
    metrics = ["median_ms", "parity_loss_rel_err", "parity_grad_rel"]
    draw(synth, tmp_path, dtype="float32", mode="fwd_bwd", sweep=["M"], metrics=metrics)
    monkeypatch.undo()
    fig = closed[0]  # the grid
    color = plot.colors(analysis.to_rows(load_quiet(synth)), "base")["chunked"]

    def faces(ax):
        """The face of chunked's marker at M=256 in a panel."""
        return [
            line.get_markerfacecolor()
            for line in ax.lines
            if line.get_linestyle() == "None"
            and line.get_color() == color
            and 256 in list(line.get_xdata())
        ]

    time, loss, grad = (fig.axes[i] for i in range(3))
    assert faces(time) == ["none"]
    assert faces(grad) == ["none"]
    assert faces(loss) != ["none"]
    assert len(faces(loss)) == 1
    for f in closed:
        plt.close(f)
    rows = read_csv(tmp_path / "grid-fwd_bwd.csv")
    at = {
        r["metric"]: (r["parity_fail"], r["parity_fail_metric"])
        for r in rows
        if (r["implementation"], r["axis"], r["x"]) == ("chunked", "M", "256")
    }
    assert at == {
        "median_ms": ("True", ""),
        "parity_loss_rel_err": ("True", "False"),
        "parity_grad_rel": ("True", "True"),
    }
    others = [r for r in rows if r["metric"].startswith("parity_") and r["parity_fail"] == "False"]
    assert others and {r["parity_fail_metric"] for r in others} == {"False"}


def test_ratio_csv(synth, tmp_path):
    draw(synth, tmp_path, dtype="float32", mode="fwd_bwd")
    rows = read_csv(tmp_path / "ratio-fwd_bwd.csv")
    # Every implementation has the baseline's work at each point: no throughput row,
    # which would repeat the speedup (R1-12 R2).
    metrics = ["speedup", "memory_ratio", "time_x_peak_ratio"]
    assert list(dict.fromkeys(r["metric"] for r in rows)) == metrics
    assert all(r["work"] == r["baseline_work"] for r in rows)
    for r in rows:
        params = dict(item.split("=") for item in r["point"].split(", "))
        flops = 2 * int(params["M"]) * int(params["K"]) * int(params["N"])
        assert r["work"] == f"tokens={params['M']},flops={flops}"
    assert "base" not in {r["implementation"] for r in rows}  # the line at 1
    speedup = {
        (r["implementation"], r["axis"], r["x"]): float(r["y"])
        for r in rows
        if r["metric"] == "speedup"
    }
    assert speedup[("compiled", "M", "64")] == pytest.approx(2.0)
    assert speedup[("chunked", "K", "32")] == pytest.approx(0.5)
    assert speedup[("base_other", "M", "128")] == pytest.approx(1 / S.DEVIATION)
    memory = {
        (r["implementation"], r["axis"], r["x"]): float(r["y"])
        for r in rows
        if r["metric"] == "memory_ratio"
    }
    assert memory[("chunked", "M", "64")] == pytest.approx(0.25)
    expected = 1 + S.PEAK_DEVIATION / S.peak_above("base", {**S.DEFAULT, "M": 256})
    assert memory[("base_other", "M", "256")] == pytest.approx(expected)
    product = {
        (r["implementation"], r["axis"], r["x"]): float(r["y"])
        for r in rows
        if r["metric"] == "time_x_peak_ratio"
    }
    # chunked: twice the time at a quarter of the peak; compiled: half the time.
    assert product[("chunked", "N", "8")] == pytest.approx(2.0 * 0.25)
    assert product[("compiled", "M", "64")] == pytest.approx(0.5)
    assert product[("base_other", "M", "128")] == pytest.approx(S.DEVIATION)
    assert product[("base_other", "M", "256")] == pytest.approx(expected)


def test_dtypes_suffix_and_overlay(synth, tmp_path):
    other = tmp_path / "second"
    S.write(other)
    with pytest.warns(UserWarning):
        written = plot.plot(
            synth, tmp_path / "out", baseline="base", overlay=other, labels=("run1", "run2")
        )
    names = {p.name for p in written}
    assert "overlay-fwd_bwd-bfloat16.png" in names and "grid-fwd-float32.png" in names
    assert len(written) == 2 * 2 * 3
    rows = read_csv(tmp_path / "out" / "overlay-fwd_bwd-float32.csv")
    assert {r["set"] for r in rows} == {"run1", "run2"}
    a = {
        (r["implementation"], r["metric"], r["axis"], r["x"], r["y"])
        for r in rows
        if r["set"] == "run1"
    }
    b = {
        (r["implementation"], r["metric"], r["axis"], r["x"], r["y"])
        for r in rows
        if r["set"] == "run2"
    }
    assert a == b  # the same data twice


def test_restrictions(synth, tmp_path):
    draw(
        synth,
        tmp_path,
        dtype="float32",
        mode="fwd_bwd",
        metrics=["min_ms"],
        sweep=["M"],
        impls=["compiled"],
    )
    rows = read_csv(tmp_path / "grid-fwd_bwd.csv")
    assert {r["metric"] for r in rows} == {"min_ms"}
    assert {r["axis"] for r in rows} == {"M"}
    assert {r["implementation"] for r in rows} == {"base", "compiled"}  # the baseline always
    with pytest.raises(ValueError, match=r"unknown metric \['x'\]; known: median_ms, min_ms"):
        draw(synth, tmp_path, metrics=["x"])
    with pytest.raises(ValueError, match=r"unknown sweep axis \['Q'\]; known: M, K, N, layout"):
        draw(synth, tmp_path, sweep=["Q"])
    with pytest.raises(ValueError, match=r"unknown implementation \['nope'\]"):
        draw(synth, tmp_path, impls=["nope"])


def colours_near(fig, ax, x, y, colours, radius=6):
    """Which of ``colours`` the rendered raster holds within ``radius`` pixels of the
    data point (x, y) of ``ax``."""
    import numpy as np
    from matplotlib.colors import to_rgb

    fig.canvas.draw()
    raster = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)
    px, py = ax.transData.transform((x, y))
    row, col = round(raster.shape[0] - py), round(px)
    window = raster[row - radius : row + radius + 1, col - radius : col + radius + 1]
    found = set()
    for colour in colours:
        rgb = np.array([round(255 * c) for c in to_rgb(colour)])
        if (np.abs(window - rgb).max(axis=-1) <= 24).any():
            found.add(colour)
    return found


@pytest.mark.parametrize("twins", [1, 3])
@pytest.mark.parametrize("original", ["compiled", "base"])
def test_coincident_series_both_shown(synth, tmp_path, monkeypatch, original, twins):
    # F7 (I1-12): a series equal to another one in a panel is drawn on top of it,
    # dashed and smaller, so both colours show at a shared point; the footer says so.
    # A twin of a counterpart, and of the baseline (which keeps its style). Three
    # twins (R1-33 F1): each a level up on the one before, so all four colours show.
    import matplotlib.pyplot as plt

    rows = analysis.at(analysis.to_rows(load_quiet(synth)), dtype="float32", mode="fwd")
    names = ["twin", "twin2", "twin3"][:twins]
    rows += [
        {**r, "implementation": name}
        for name in names
        for r in rows
        if r["implementation"] == original
    ]
    palette = plot.colors(rows, "base")
    closed = []
    monkeypatch.setattr(plt, "close", closed.append)
    plot._figure(
        tmp_path / "grid.png",
        [(None, rows, None)],
        ["median_ms", "tokens_per_s"],
        ["M", "K"],
        S.DEFAULT,
        "base",
        "synth: fwd",
        [plot.left_out(rows, ["median_ms"]), "provenance"],
        palette=palette,
    )
    monkeypatch.undo()
    (fig,) = closed
    y = S.median_ms(original, "fwd", "float32", S.DEFAULT)
    stack = [original, *names]
    every = {palette[name] for name in stack}
    assert len(every) == len(stack)
    for ax in fig.axes[:2]:  # median against M and K
        assert colours_near(fig, ax, S.DEFAULT[ax.get_xlabel()], y, every) == every

    def lines_of(name):
        return [line for line in fig.axes[0].lines if line.get_color() == palette[name]]

    # The one below keeps its style (base_other, black too, is a dotted control).
    role = "baseline" if original == "base" else "counterpart"
    style = plot.STYLE[role]
    under = lines_of(original)
    lines = [line for line in under if line.get_linestyle() not in ("None", ":")]
    assert [(line.get_linestyle(), line.get_linewidth()) for line in lines] == [
        (style["linestyle"], style["linewidth"])
    ]
    # Each twin a level up, dashed at half its width and marker size; from level 2 on
    # shifted left, then right, by half the bottom's marker size per step.
    twin_style = plot.STYLE["counterpart"]
    scale = plot.COINCIDENT_SCALE
    ax = fig.axes[0]
    to_points = 72.0 / fig.dpi
    centre = ax.transData.transform((S.DEFAULT["M"], y))[0] * to_points
    for level, name in enumerate(names, start=1):
        upper = lines_of(name)
        assert min(line.get_zorder() for line in upper) > max(line.get_zorder() for line in under)
        drawn = [line for line in upper if line.get_linestyle() != "None"]
        assert [(line.get_linestyle(), line.get_linewidth()) for line in drawn] == [
            ("--", twin_style["linewidth"] * scale)
        ]
        markers = [line for line in upper if line.get_linestyle() == "None"]
        assert {line.get_markersize() for line in markers} == {twin_style["size"] * scale}
        step = {1: 0, 2: -1, 3: 1}[level]
        for line in markers:
            x = line.get_transform().transform((S.DEFAULT["M"], y))[0] * to_points
            assert x - centre == pytest.approx(step * plot.COINCIDENT_DODGE * style["size"])
        under = upper
    footer = next(t for t in fig.texts if t.get_gid() == "footer").get_text()
    for upper in names:  # the footer names the first one each lies on
        clause = f"{upper} on {original} (fwd: median, tokens/s)"
        assert clause in footer.replace("\n", " ")
    assert footer.startswith("Left out:")
    plt.close(fig)


def test_coincident_pairs():
    # In display points (P1, I1-14 round 2): within half the lower series' marker size
    # (5 pt here) at every x.
    size = 5
    limit = plot.COINCIDENT_MARKER_FRACTION * size
    assert limit == 2.5

    def series(ys, set_=0, name="s", control_of=None):
        display = [None if y is None else (10.0 * i, y) for i, y in enumerate(ys)]
        return {
            "set": set_,
            "name": name,
            "control_of": control_of,
            "style": {"size": size},
            "display": display,
        }

    drawn = [
        series([100.0, 200.0, None], name="a"),
        series([100.0 + 2.4, 200.0 - 2.4, None], name="b"),  # just inside: on a
        series([100.0, 200.0, 300.0]),  # another x set
        series([100.0, 200.0 + 2.6, None]),  # just outside at one x
        series([100.0, 200.0, None], set_=1),  # another result set
        series([None, None, None]),
        series([None, None, None]),  # nothing drawn: nothing coincides
        series([100.0, 200.0, None], name="c"),  # on a, the first
        series([100.0, 200.0, None], name="a_ctl", control_of="a"),  # on b, not on a
    ]
    assert plot.coincident_pairs(drawn) == {1: 0, 7: 0, 8: 1}
    # One stack of a, b, c, a_ctl: four levels, none shared, even where two lie on
    # the same one; the control is in it through b.
    levels = plot.coincident_levels(drawn)
    assert [levels[k] for k in (0, 1, 7, 8)] == [(k, 4, 0) for k in range(4)]
    assert levels[2] == (0, 1, 2) and levels[4] == (0, 1, 4)
    # A control alone on its original is not coincident (P1, I1-14 round 2).
    for pair in ([drawn[0], drawn[8]], [drawn[8], drawn[0]]):
        assert plot.coincident_pairs(pair) == {}
        assert plot.coincident_levels(pair) == {0: (0, 1, 0), 1: (0, 1, 1)}
    # Nor are two controls of one original, each in its own environment (R1-34 R1):
    # styled alike by design; their agreement is the environment check's.
    a_other = series([100.0, 200.0, None], name="a_other", control_of="a")
    assert plot.coincident_pairs([drawn[8], a_other]) == {}
    # ... while a control on another implementation's control is still coincident.
    b_ctl = series([100.0, 200.0, None], name="b_ctl", control_of="b")
    assert plot.coincident_pairs([drawn[8], b_ctl]) == {1: 0}
    # Two on one that are not on each other (2.4 pt either side of it, 4.8 pt apart):
    # still a level each, so the later does not cover the earlier.
    fan = [
        series([100.0, 200.0, None], name="a"),
        series([102.4, 200.0, None], name="b"),
        series([97.6, 200.0, None], name="c"),
    ]
    assert plot.coincident_pairs(fan) == {1: 0, 2: 0}
    assert plot.coincident_levels(fan) == {0: (0, 3, 0), 1: (1, 3, 0), 2: (2, 3, 0)}
    text = plot.coincident_text(
        {(0, "b", "a"): [("median_ms", "M"), ("median_ms", "K"), ("tokens_per_s", "M")]},
        0,
        ["M", "K"],
        ["fwd"],
    )
    assert (
        text
        == "Coincident, the first drawn dashed on the second: b on a (fwd: median, tokens/s on M)."
    )
    assert plot.coincident_text({(1, "b", "a"): [("median_ms", "M")]}, 0, ["M"], ["fwd"]) is None


def test_styles(synth):
    with pytest.warns(UserWarning):
        rows = analysis.to_rows(analysis.load(synth))
    # Without an order: by sorted name, whatever the records' order (R1-12 remark).
    palette = plot.colors(rows, "base")
    assert palette["base"] == "black" and palette["base_other"] == "black"
    assert palette["chunked"] == plot.COLORS[0] and palette["compiled"] == plot.COLORS[1]
    assert plot.colors(rows[::-1], "base") == palette
    # With the spec's order; a figure lacking an implementation keeps the others'.
    order = ["base", "compiled", "chunked"]
    palette = plot.colors(rows, "base", order)
    assert palette["compiled"] == plot.COLORS[0] and palette["chunked"] == plot.COLORS[1]
    no_compiled = [r for r in rows if r["implementation"] != "compiled"]
    assert plot.colors(no_compiled, "base", order)["chunked"] == plot.COLORS[1]
    # A control of a counterpart takes its colour; without the baseline drawn (the
    # ratio figure), a control of the baseline is still black.
    rows2 = [
        {**r, "control_of": "chunked"} if r["implementation"] == "base_other" else r for r in rows
    ]
    assert plot.colors(rows2, "base")["base_other"] == plot.COLORS[0]
    no_base = [r for r in rows if r["implementation"] != "base"]
    assert plot.colors(no_base, "base")["base_other"] == "black"
    assert plot.STYLE["baseline"]["linestyle"] == "--" and plot.STYLE["control"]["linestyle"] == ":"


def test_stable_order():
    from torch_op_bench.ops import lce

    order = plot.stable_order([{"op_ref": "lce"}])
    # Controls and variants left out: a variant takes no colour of its own, so the
    # base entries' colours are as without variants.
    assert order == [i.name for i in lce.IMPLEMENTATIONS if i.info.get("role") != "control"]
    assert "torch_eager_liger" not in order and order[0] == "torch_eager"
    assert plot.stable_order([{"op_ref": "no_such_module_xyz:SPEC"}]) is None
    assert plot.stable_order([]) is None


def test_one_palette_per_invocation(synth, tmp_path, monkeypatch):
    palettes = []
    real = plot._figure

    def spy(*args, palette=None, **kwargs):
        palettes.append(dict(palette))
        return real(*args, palette=palette, **kwargs)

    monkeypatch.setattr(plot, "_figure", spy)
    draw(synth, tmp_path, dtype="float32", metrics=["median_ms"], sweep=["M"])
    draw(synth, tmp_path, dtype="float32", metrics=["median_ms"], sweep=["M"], impls=["chunked"])
    # fwd has no chunked; --impl drops compiled: the colours stay.
    assert all(p == palettes[0] for p in palettes)
    assert palettes[0]["chunked"] == plot.COLORS[0]


def test_left_out_and_provenance():
    rows = [
        {"status": "error"},
        {"status": "unavailable"},
        {"status": "ok", "parity_status": "skipped"},
        {"status": "ok", "parity_status": "ok"},
    ]
    assert (
        plot.left_out(rows, ["median_ms"])
        == "Left out: 1 error, 1 unavailable (gaps in the lines)."
    )
    assert "1 without parity" in plot.left_out(rows, ["parity_x"])
    assert plot.left_out(rows[3:], ["median_ms"]) == "Every record of this figure is drawn."
    # Parity failures are drawn (hollow, in every panel): counted, with those a note
    # explains (R1-30 F1).
    note = ParityNote("x", ("e",), lambda case, mode, metrics, version: mode == "fwd", "why")
    failed = {"status": "ok", "parity_status": "fail", "parity_failed": "e", "dtype": "f"}
    failed |= {"device": "cpu", "M": 1}
    rows += [{**failed, "implementation": "x", "mode": m} for m in ("fwd", "fwd_bwd")]
    assert plot.left_out(rows[3:], ["parity_e"], [note], ["M"]) == (
        "Every record of this figure is drawn. Parity failures: 2 (1 explained), drawn hollow."
    )
    assert plot.left_out(rows[3:], [], [note], ["M"]).endswith("2 (1 explained), drawn hollow.")
    assert plot.left_out(rows[3:], ["median_ms"]).endswith("2 (0 explained), drawn hollow.")
    row = {"env_name": "e", "torch_version": "2.1", "tob_hash": "abcdef1234567890", "host": "h"}
    assert plot.provenance([row], "A") == "A: e: torch 2.1, torch-op-bench abcdef123456; host h"
    assert plot.provenance([{}]) == "no provenance recorded; host ?"


def test_without_matplotlib(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ImportError, match=r"install torch-op-bench\[plot\]"):
        plot.pyplot()


def test_cli(synth, tmp_path, capsys, monkeypatch):
    argv = ["plot", str(synth), "--out", str(tmp_path), "--baseline", "base"]
    with pytest.warns(UserWarning):
        assert (
            cli.main([*argv, "--dtype", "float32", "--mode", "fwd", "--metric", "median_ms"]) == 0
        )
    out = capsys.readouterr().out
    assert f"wrote {tmp_path / 'grid-fwd.png'} and grid-fwd.csv" in out
    assert {r["metric"] for r in read_csv(tmp_path / "grid-fwd.csv")} == {"median_ms"}
    with pytest.warns(UserWarning), pytest.raises(SystemExit):
        cli.main([*argv, "--metric", "x"])
    assert "unknown metric" in capsys.readouterr().err
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(SystemExit):
        cli.main(argv)
    assert "install torch-op-bench[plot]" in capsys.readouterr().err


def test_no_sweep(synth, tmp_path):
    # Only the default point's records: no axis varies, so no figure and a warning.
    from torch_op_bench import records

    only = tmp_path / "only"
    with pytest.warns(UserWarning):
        recs = analysis.load(synth)
    for r in recs:
        if all(r["case"]["params"][k] == v for k, v in S.DEFAULT.items()):
            r = {k: v for k, v in r.items() if k != analysis.SOURCE}
            records.append(only / "synth" / S.SLUG / "x-host.jsonl", r)
    with pytest.warns(UserWarning) as caught:
        assert plot.plot(only, tmp_path / "out", baseline="base") == []
    messages = [str(w.message) for w in caught]
    assert any("no axis varies around the default point" in m for m in messages)
    assert any("different torch-op-bench commits" in m for m in messages)


def test_without_the_default_point(synth, tmp_path, capsys):
    # Two cases that differ in two axes (I1-6 on ex): no sweep figure, said per mode
    # and dtype, and exit code 0.
    two = S.subset(synth, tmp_path / "two", {"M": 64}, {"K": 16})
    with pytest.warns(UserWarning) as caught:
        assert (
            cli.main(["plot", str(two), "--out", str(tmp_path / "out"), "--baseline", "base"]) == 0
        )
    messages = [str(w.message) for w in caught if "no axis varies" in str(w.message)]
    assert len(messages) == 2 * 2
    assert all(m.endswith("no figures (report has a table per case)") for m in messages)
    assert capsys.readouterr().out == ""
    # --point centres on another point: around K=16 the synthetic set varies K only.
    written = draw(synth, tmp_path / "point", dtype="float32", mode="fwd_bwd", point={"K": "16"})
    assert [p.name for p in written] == ["grid-fwd_bwd.png", "ratio-fwd_bwd.png"]
    rows = read_csv(tmp_path / "point" / "grid-fwd_bwd.csv")
    assert {r["axis"] for r in rows} == {"K"}
    assert {r["x"] for r in rows} == {"16", "32"}


def test_controls_of_one_original_differ():
    # Two controls of torch_eager in two environments: the same colour and line,
    # so their markers differ (R1-12 F1), one per environment, stable by name.
    rows = [
        {"implementation": "eager", "env_name": "torch", "role": "baseline"},
        {
            "implementation": "eager_liger",
            "env_name": "liger",
            "role": "control",
            "control_of": "eager",
        },
        {
            "implementation": "eager_titan",
            "env_name": "titan",
            "role": "control",
            "control_of": "eager",
        },
        {"implementation": "other", "env_name": "titan"},
    ]
    markers = plot.control_markers(rows)
    assert markers == {"eager_liger": "s", "eager_titan": "^"}
    assert plot.control_markers(rows[::-1]) == markers
    palette = plot.colors(rows, "eager")
    assert palette["eager_liger"] == palette["eager_titan"] == "black"


def test_controls_drawn_with_their_markers(synth, tmp_path, monkeypatch):
    # The legend handle of a control carries its environment's marker.
    seen = {}
    real = plot._draw_panel

    def spy(ax, sets, metric, axis, point, names, palette, markers, *rest):
        seen.update(markers)
        return real(ax, sets, metric, axis, point, names, palette, markers, *rest)

    monkeypatch.setattr(plot, "_draw_panel", spy)
    draw(synth, tmp_path, dtype="float32", mode="fwd", metrics=["median_ms"], sweep=["M"])
    assert seen == {"base_other": "s"}


def test_throughput_ratio_when_work_differs(synth, tmp_path):
    # An implementation whose work differs from the baseline's at a point: the
    # throughput ratio is no longer the speedup, so it is drawn, and the CSV shows
    # both works.
    from torch_op_bench import records

    changed = tmp_path / "changed"
    with pytest.warns(UserWarning):
        recs = analysis.load(synth)
    for r in recs:
        r = {k: v for k, v in r.items() if k != analysis.SOURCE}
        if r["implementation"] == "chunked":
            r["work"] = {**r["work"], "tokens": 2 * r["work"]["tokens"]}
        records.append(changed / "synth" / S.SLUG / "x-host.jsonl", r)
    with pytest.warns(UserWarning):
        plot.plot(changed, tmp_path / "out", baseline="base", dtype="float32", mode="fwd_bwd")
    rows = read_csv(tmp_path / "out" / "ratio-fwd_bwd.csv")
    assert "throughput_ratio" in {r["metric"] for r in rows}
    (row,) = [
        r
        for r in rows
        if (r["metric"], r["implementation"], r["axis"], r["x"])
        == ("throughput_ratio", "chunked", "M", "64")
    ]
    assert row["work"] == f"tokens=128,flops={2 * 64 * 32 * 16}"
    assert row["baseline_work"] == f"tokens=64,flops={2 * 64 * 32 * 16}"
    assert float(row["y"]) == pytest.approx(2 * 0.5)  # twice the tokens in twice the time


def test_overlay_footer_counts_both_sets(synth, tmp_path, monkeypatch):
    # B lacks what A has and has gaps of its own: the footer counts each, labelled.
    from torch_op_bench import records

    other = tmp_path / "b"
    with pytest.warns(UserWarning):
        recs = analysis.load(synth)
    for r in recs:
        r = {k: v for k, v in r.items() if k != analysis.SOURCE}
        if r["implementation"] == "chunked" and r["case"]["params"]["M"] == 64:
            r = {**r, "status": "error", "reason": "boom", "times_ms": None}
        records.append(other / "synth" / S.SLUG / "x-host.jsonl", r)
    footers = {}
    real = plot._figure

    def spy(path, sets, metrics, axes, point, baseline, title, footer, **kwargs):
        footers[path.name] = footer
        return real(path, sets, metrics, axes, point, baseline, title, footer, **kwargs)

    monkeypatch.setattr(plot, "_figure", spy)
    with pytest.warns(UserWarning):
        plot.plot(
            synth,
            tmp_path / "out",
            baseline="base",
            dtype="float32",
            mode="fwd_bwd",
            overlay=other,
            labels=("one", "two"),
        )
    footer = footers["overlay-fwd_bwd.png"]
    # chunked fails parity at M=256 in float32, which synth_op's note does not explain.
    hollow = " Parity failures: 1 (0 explained), drawn hollow."
    assert footer[0] == "one: Left out: 1 unavailable (gaps in the lines)." + hollow
    assert footer[1] == "two: Left out: 1 error, 1 unavailable (gaps in the lines)." + hollow
    assert footer[2].startswith("one: ") and footer[3].startswith("two: ")


def test_markers_independent_of_impl(synth, tmp_path, monkeypatch):
    # Two controls of one original: alpha's sorts first, so it has the square and
    # other's the triangle. Drawing only other's control must not hand it the
    # square (R1-13 F3): the markers come from all records, before --impl.
    from torch_op_bench import records

    both = tmp_path / "both"
    with pytest.warns(UserWarning):
        recs = analysis.load(synth)
    for r in recs:
        r = {k: v for k, v in r.items() if k != analysis.SOURCE}
        records.append(both / "synth" / S.SLUG / "x-host.jsonl", r)
        if r["implementation"] == "base_other":
            alpha = {**r, "implementation": "base_alpha", "env_name": "alpha"}
            records.append(both / "synth" / S.SLUG / "x-host.jsonl", alpha)
    seen = []
    real = plot._draw_panel

    def spy(ax, sets, metric, axis, point, names, palette, markers, *rest):
        seen.append({n: markers[n] for n in names if n in markers})
        return real(ax, sets, metric, axis, point, names, palette, markers, *rest)

    monkeypatch.setattr(plot, "_draw_panel", spy)
    kwargs = dict(baseline="base", dtype="float32", mode="fwd", metrics=["median_ms"], sweep=["M"])
    with pytest.warns(UserWarning):
        plot.plot(both, tmp_path / "all", **kwargs)
    with pytest.warns(UserWarning):
        plot.plot(both, tmp_path / "one", impls=["base_other"], **kwargs)
    assert seen[0] == {"base_alpha": "s", "base_other": "^"}  # all drawn
    assert seen[-1] == {"base_other": "^"}  # alone, the same marker


def test_long_title_wraps_in_one_column(synth, tmp_path, monkeypatch):
    # R1-19 F2: a one-column figure (3.8 in) used to clip its title at both ends.
    import matplotlib.pyplot as plt

    closed, close = [], plt.close
    monkeypatch.setattr(plt, "close", closed.append)
    rows = [r for r in analysis.to_rows(load_quiet(synth)) if r["mode"] == "fwd"]
    rows = [r for r in rows if r["dtype"] == "float32"]
    title = "synth on nvidia-geforce-rtx-5060-ti: fwd_bwd, bfloat16: against torch_eager"
    path = plot._figure(
        tmp_path / "one.png",
        [(None, rows, None)],
        ["median_ms"],
        ["M"],
        S.DEFAULT,
        "base",
        title,
        ["footer"],
    )
    assert png_size(path)[0] > 0 and path.with_suffix(".csv").exists()
    (fig,) = closed
    suptitle = title_artist(fig)
    assert suptitle.get_text().count("\n") >= 1
    assert suptitle.get_text().replace("\n", " ") == title
    renderer = fig.canvas.get_renderer()
    box, page = suptitle.get_window_extent(renderer), fig.bbox
    assert page.x0 <= box.x0 and box.x1 <= page.x1, (box, page)
    assert page.y0 <= box.y0 and box.y1 <= page.y1, (box, page)
    close(fig)
    # A short title stays on one line.
    closed.clear()
    plot._figure(
        tmp_path / "short.png",
        [(None, rows, None)],
        ["median_ms"],
        ["M"],
        S.DEFAULT,
        "base",
        "synth: fwd",
        ["footer"],
    )
    assert "\n" not in title_artist(closed[0]).get_text()
    close(closed[0])


def load_quiet(synth):
    with pytest.warns(UserWarning):
        return analysis.load(synth)


def title_artist(fig):
    # 8792191 drew the title with fig.suptitle; now it is the text with gid "title".
    found = [t for t in fig.texts if t.get_gid() == "title"]
    return found[0] if found else fig._suptitle


def cap_height(size):
    """A capital letter's height, in points, of text of ``size`` (DejaVu Sans: 0.73 em)."""
    from matplotlib.font_manager import FontProperties

    return 0.73 * FontProperties(size=size).get_size_in_points()


def panel_rows(fig, renderer):
    """The tight boxes (tick and axis labels included) of the figure's rows of panels,
    top to bottom: a row's is the union of its panels'."""
    from matplotlib.transforms import Bbox

    rows = {}
    for ax in fig.axes:
        rows.setdefault(round(ax.get_position().y1, 6), []).append(ax.get_tightbbox(renderer))
    return [Bbox.union(rows[top]) for top in sorted(rows, reverse=True)]


def rendered_gaps(fig):
    """The white space, in points, between stacked elements, with its bound: a capital
    letter of the axis font (the tick and axis labels), as the user set it (inbox 22);
    between two stacked rows of panels too (inbox 34): ``panels-<i>-<i+1>``."""
    import matplotlib

    renderer = fig.canvas.get_renderer()
    pt = 72 / fig.dpi
    title = title_artist(fig)
    footer = [t for t in fig.texts if t is not title]
    tops = {ax: ax.get_position().y1 for ax in fig.axes}
    bottoms = {ax: ax.get_position().y0 for ax in fig.axes}
    first = [ax for ax in fig.axes if tops[ax] == max(tops.values())]
    last = [ax for ax in fig.axes if bottoms[ax] == min(bottoms.values())]
    panels_top = max(ax.get_tightbbox(renderer).y1 for ax in first)
    panels_bottom = min(ax.get_tightbbox(renderer).y0 for ax in last)
    axis_font = max(
        FontPropertiesSize(matplotlib.rcParams[name])
        for name in ("xtick.labelsize", "ytick.labelsize", "axes.labelsize")
    )
    bound = cap_height(axis_font)
    box = title.get_window_extent(renderer)
    gaps = {}
    above = box.y0
    if fig.legends:
        lbox = fig.legends[0].get_window_extent(renderer)
        gaps["title-legend"] = ((box.y0 - lbox.y1) * pt, bound)
        above = lbox.y0
    gaps["above-panels"] = ((above - panels_top) * pt, bound)
    rows = panel_rows(fig, renderer)
    for i, (upper, lower) in enumerate(itertools.pairwise(rows)):
        gaps[f"panels-{i}-{i + 1}"] = ((upper.y0 - lower.y1) * pt, bound)
    footer_top = max(t.get_window_extent(renderer).y1 for t in footer)
    gaps["panels-footer"] = ((panels_bottom - footer_top) * pt, bound)
    return gaps


def panel_gaps(gaps):
    """The names of ``rendered_gaps``'s gaps between two rows of panels."""
    return [name for name in gaps if re.fullmatch(r"panels-\d+-\d+", name)]


def with_strict(kwargs, synth_strict, synth=None):
    """``kwargs`` for ``draw`` with ``strict`` the strict set when asked, and the kinds
    of figure written: the reduction figure (R1-26 F2) is under the same rules; so is
    the equal-peak figure, with ``variants`` (written into ``synth``)."""
    kwargs = dict(kwargs)
    kinds = {"grid", "ratio", "overlay"}
    if kwargs.pop("variants", False):
        S.write_variants(synth)
        kinds.add("equal")
    if not kwargs.pop("strict", False):
        return kwargs, kinds
    return {**kwargs, "strict": synth_strict}, kinds | {"reduction"}


def FontPropertiesSize(size):
    from matplotlib.font_manager import FontProperties

    return FontProperties(size=size).get_size_in_points()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sweep": ["M"]},  # one column: the title wraps
        {},  # four columns
        {"sweep": ["M", "K"], "impls": ["base"]},  # two columns, the baseline alone
        {"strict": True},  # and the reduction figures
        {"strict": True, "impls": ["base"], "mode": "fwd_bwd"},  # one pair of bars
        {"variants": True, "sweep": ["M"]},  # and the equal-peak figures
        {"variants": True, "impls": ["chunked"], "mode": "fwd_bwd"},  # one entry
    ],
    ids=[
        "one-column",
        "four-columns",
        "two-columns",
        "reduction",
        "reduction-one-entry",
        "equal-peak",
        "equal-peak-one-entry",
    ],
)
def test_gaps_between_bands(synth, synth_strict, tmp_path, monkeypatch, kwargs):
    # The user (inbox 21): the white space between title, legend and first panel,
    # and between last panel and footer, is at most about a capital letter's height
    # of the text around it, in every figure plot writes.
    import matplotlib.pyplot as plt

    closed = []
    monkeypatch.setattr(plt, "close", closed.append)
    other = tmp_path / "second"
    S.write(other)
    kwargs, kinds = with_strict(kwargs, synth_strict, synth)
    written = draw(synth, tmp_path / "out", dtype="float32", overlay=other, **kwargs)
    assert {p.name.split("-")[0] for p in written} == kinds
    assert len(closed) == len(written)
    stacked = set()  # the kinds drawn with two rows of panels or more
    for path, fig in zip(written, closed, strict=True):
        gaps = rendered_gaps(fig)
        for name, (gap, bound) in gaps.items():
            assert -0.5 <= gap <= bound, (path.name, name, gap, bound)
        if panel_gaps(gaps):
            stacked.add(path.name.split("-")[0])
    # No kind is measured on one row only, but the equal-peak figure, one panel by
    # design.
    assert stacked == kinds - {"equal"}
    if kwargs.get("sweep") == ["M"]:  # the ratio and overlay titles wrap at 3.8 in
        assert any("\n" in title_artist(fig).get_text() for fig in closed)
    monkeypatch.undo()
    for fig in closed:
        plt.close(fig)


def nightly_time_x_peak(synth):
    """fwd_bwd rows of the synthetic set with time × peak planted in the 10⁶–10⁷ range
    of I1-12's records (ms·MiB), where an offset text "1e7" stood above the panel."""
    rows = [r for r in analysis.to_rows(load_quiet(synth)) if r["mode"] == "fwd_bwd"]
    rows = [r for r in rows if r["dtype"] == "float32"]
    for i, r in enumerate(rows):
        if r.get("time_x_peak") is not None:
            r["time_x_peak"] = 1.4e6 + 9.7e5 * (i % 13)
    return rows


PANEL_GAP_CASES = {
    # tight_layout's former h_pad (1.08 font sizes, 10.8 pt).
    "h_pad": (
        lambda synth: [r for r in analysis.to_rows(load_quiet(synth)) if r["mode"] == "fwd"],
        ["median_ms", "tokens_per_s"],
        {"PANEL_PAD": 1.08},
    ),
    # R1-30 N2 (R1-28's "nightly time_x_peak"): five metrics, time × peak in the
    # 10⁶–10⁷ range, in plain numbers of the unit as it was (no scaled unit).
    "time_x_peak": (
        nightly_time_x_peak,
        ["median_ms", "time_x_peak", "parity_grad_rel", "tflops", "iqr_ms"],
        {"SCALE_ABOVE": math.inf, "_plain_ticks": lambda ax: None},
    ),
}


@pytest.mark.parametrize("case", PANEL_GAP_CASES)
def test_panel_gap_measured(synth, tmp_path, monkeypatch, case):
    # The bound on the gap between stacked panels is not met vacuously: with the
    # layout as it was (``broken``) the same measurement reports a gap above it, and
    # as it is, every gap is within it.
    import matplotlib.pyplot as plt

    select, metrics, broken = PANEL_GAP_CASES[case]
    rows = [r for r in select(synth) if r["dtype"] == "float32"]
    figures = {}
    for name, patch in (("broken", broken), ("now", {})):
        closed = []
        monkeypatch.setattr(plt, "close", closed.append)
        for attr, value in patch.items():
            monkeypatch.setattr(plot, attr, value)
        plot._figure(
            tmp_path / f"{name}.png",
            [(None, rows, None)],
            metrics,
            ["M"],
            S.DEFAULT,
            "base",
            "synth",
            ["footer"],
        )
        monkeypatch.undo()
        (figures[name],) = closed
    gaps = rendered_gaps(figures["broken"])
    assert panel_gaps(gaps) == [f"panels-{i}-{i + 1}" for i in range(len(metrics) - 1)]
    assert max(gaps[name][0] - gaps[name][1] for name in panel_gaps(gaps)) > 0, gaps
    for name, (gap, bound) in rendered_gaps(figures["now"]).items():
        assert -0.5 <= gap <= bound, (case, name, gap, bound)
    if case == "time_x_peak":
        broken_offsets = [ax.yaxis.get_offset_text().get_text() for ax in figures["broken"].axes]
        assert any(broken_offsets)  # the 1e6 above the panel
        figures["now"].canvas.draw()
        assert [ax.yaxis.get_offset_text().get_text() for ax in figures["now"].axes] == [""] * 5
        assert figures["now"].axes[1].get_ylabel() == "time × peak\n(10⁶ ms·MiB)"
    for fig in figures.values():
        plt.close(fig)


def test_y_exponent_and_scaled_label():
    assert plot.y_exponent([1.4e6, 9.9e6, None]) == 6
    assert plot.y_exponent([2e5]) == 3
    assert plot.y_exponent([1e5, 3.0]) == 0  # at most SCALE_ABOVE: plain
    assert plot.y_exponent([2.25e-6, 0.0]) == -6
    assert plot.y_exponent([0.0, None]) == 0
    assert plot.scaled_label("time × peak (ms·MiB)", 6) == "time × peak (10⁶ ms·MiB)"
    assert plot.scaled_label("tokens/s", 3) == "tokens/s (×10³)"
    assert plot.scaled_label("loss_rel_err", -6) == "loss_rel_err (×10⁻⁶)"
    assert plot.scaled_label("median (ms)", 0) == "median (ms)"


def drawn_texts(fig):
    """(text, window extent) of every non-empty text the figure draws: tick labels
    outside the view, which matplotlib keeps but does not draw, are not among them."""
    from matplotlib.text import Text

    drawn = []
    draw = Text.draw

    def recording(self, renderer):
        draw(self, renderer)
        if self.get_visible() and self.get_text().strip():
            drawn.append((self.get_text(), self.get_window_extent(renderer)))

    Text.draw = recording
    try:
        fig.canvas.draw()
    finally:
        Text.draw = draw
    return drawn


def assert_inside(fig, name):
    page = fig.bbox
    texts = drawn_texts(fig)
    assert texts, name
    for text, box in texts:
        inside = page.x0 <= box.x0 and box.x1 <= page.x1 and page.y0 <= box.y0 and box.y1 <= page.y1
        assert inside, (name, text, box, page)


def nightly_like_ratios(synth):
    """Ratio rows of the synthetic set with the nightly's ratios planted along M
    (I1-6b/nightly, against V): their tick labels widen once the panels are laid
    out, which pushed the memory ratio's y label past the left edge on dac3e47."""
    recs = load_quiet(synth)
    rows = analysis.at(analysis.to_rows(recs), dtype="float32", mode="fwd_bwd")
    ratios = plot.ratio_rows(rows, "base", analysis.axes_of(recs))
    planted = {
        "compiled": ([0.973, 0.869, 0.728], [0.50, 0.416, 0.375], [0.514, 0.48, 0.515]),
        "chunked": ([0.834, 0.760, 0.660], [0.552, 0.44, 0.386], [0.662, 0.579, 0.585]),
    }
    ratios = [r for r in ratios if r["implementation"] in planted]
    for r in ratios:
        if (r["K"], r["N"], r["layout"]) == (32, 16, "a"):
            i = [64, 128, 256].index(r["M"])
            speedup, memory, product = planted[r["implementation"]]
            r.update(speedup=speedup[i], memory_ratio=memory[i], time_x_peak_ratio=product[i])
    return ratios


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sweep": ["M"]},  # one column: wrapped titles
        {"sweep": ["M", "K"]},
        {},  # four columns
        {"sweep": ["M"], "metrics": ["peak_above_baseline"]},  # one row
        {"sweep": ["layout"], "metrics": ["parity_grad_rel", "tflops"], "mode": "fwd_bwd"},
        {"strict": True},  # and the reduction figures
        {"strict": True, "impls": ["chunked"], "mode": "fwd_bwd"},
        {"variants": True, "sweep": ["M"]},
        {"variants": True, "impls": ["compiled[mode=a]"], "mode": "fwd"},
    ],
    ids=[
        "one-column",
        "two-columns",
        "four-columns",
        "one-row",
        "str-axis",
        "reduction",
        "reduction-two-entries",
        "equal-peak",
        "equal-peak-named-variant",
    ],
)
def test_nothing_clipped(synth, synth_strict, tmp_path, monkeypatch, kwargs):
    # The user (inbox 22): nothing non-empty may be clipped, in any direction, in
    # any figure plot writes: every drawn text lies inside the figure.
    import matplotlib.pyplot as plt

    closed = []
    monkeypatch.setattr(plt, "close", closed.append)
    other = tmp_path / "second"
    S.write(other)
    kwargs, kinds = with_strict({"dtype": "float32", **kwargs}, synth_strict, synth)
    written = draw(synth, tmp_path / "out", overlay=other, **kwargs)
    assert {p.name.split("-")[0] for p in written} == kinds
    monkeypatch.undo()
    for path, fig in zip(written, closed, strict=True):
        assert_inside(fig, path.name)
        plt.close(fig)


def test_nothing_clipped_after_relayout(synth, tmp_path, monkeypatch):
    # The case that clipped on dac3e47: one column, ratios whose tick labels widen
    # after the first layout pass, long y labels.
    import matplotlib.pyplot as plt

    closed, close = [], plt.close
    monkeypatch.setattr(plt, "close", closed.append)
    path = plot._figure(
        tmp_path / "ratio.png",
        [(None, nightly_like_ratios(synth), None)],
        ["speedup", "memory_ratio", "time_x_peak_ratio"],
        ["M"],
        S.DEFAULT,
        "base",
        "synth on synthdev: fwd_bwd, float32: against base",
        ["footer"],
        ratio_line=True,
    )
    monkeypatch.undo()
    (fig,) = closed
    assert_inside(fig, path.name)
    for name, (gap, bound) in rendered_gaps(fig).items():
        assert -0.5 <= gap <= bound, (name, gap, bound)
    close(fig)


def test_long_ylabel_wraps(tmp_path, monkeypatch):
    # A y label longer than its panel is wrapped, so it neither runs into the
    # neighbouring bands nor past the figure.
    import matplotlib.pyplot as plt

    monkeypatch.setitem(plot.METRIC_LABELS, "median_ms", "a very long label " * 4)
    rows = [
        {
            "implementation": "base",
            "env_name": "torch",
            "role": "baseline",
            "M": m,
            "median_ms": float(m),
            "status": "ok",
            "parity_status": "ok",
            "dtype": "float32",
            "mode": "fwd",
            "K": 32,
            "N": 16,
            "layout": "a",
        }
        for m in (64, 128, 256)
    ]
    closed, close = [], plt.close
    monkeypatch.setattr(plt, "close", closed.append)
    plot._figure(
        tmp_path / "long.png",
        [(None, rows, None)],
        ["median_ms"],
        ["M"],
        S.DEFAULT,
        "base",
        "synth: fwd",
        ["footer"],
    )
    monkeypatch.undo()
    (fig,) = closed
    (ax,) = fig.axes
    renderer = fig.canvas.get_renderer()
    assert "\n" in ax.yaxis.label.get_text()
    assert ax.yaxis.label.get_window_extent(renderer).height <= ax.bbox.height
    assert_inside(fig, "long.png")
    close(fig)


def test_ylabel_text():
    assert plot.ylabel_text("speedup (baseline / impl time)") == "speedup\n(baseline / impl time)"
    assert plot.ylabel_text("memory ratio (impl / baseline peak)") == (
        "memory ratio\n(impl / baseline peak)"
    )
    assert plot.ylabel_text("time × peak ratio (impl / baseline)") == (
        "time × peak ratio\n(impl / baseline)"
    )
    assert plot.ylabel_text("median (ms)") == "median\n(ms)"
    assert plot.ylabel_text("tokens/s") == "tokens/s"
    assert plot.ylabel_text("grad_rel") == "grad_rel"


@pytest.mark.parametrize(
    "kwargs",
    [{"sweep": ["M"]}, {}, {"sweep": ["M"], "metrics": ["median_ms", "tokens_per_s", "tflops"]}],
    ids=["one-column", "four-columns", "more-metrics"],
)
def test_ylabels_split_before_parentheses(synth, tmp_path, monkeypatch, kwargs):
    # The user (inbox 23): every y label's part in parentheses is on a second line,
    # always; a label without one stays on one line.
    import matplotlib.pyplot as plt

    closed = []
    monkeypatch.setattr(plt, "close", closed.append)
    other = tmp_path / "second"
    S.write(other)
    written = draw(synth, tmp_path / "out", dtype="float32", overlay=other, **kwargs)
    assert {p.name.split("-")[0] for p in written} == {"grid", "ratio", "overlay"}
    monkeypatch.undo()
    # A label as in METRIC_LABELS, or in a scaled unit (synth's TFLOP/s are ~10⁻⁶).
    labels = {
        plot.scaled_label(label, k)
        for label in plot.METRIC_LABELS.values()
        for k in range(-12, 13, 3)
    }
    seen = set()
    for path, fig in zip(written, closed, strict=True):
        for ax in fig.axes:
            text = ax.yaxis.label.get_text()
            source = text.replace("\n", " ")
            if " (" in source:
                head, _, tail = source.partition(" (")
                assert text.split("\n") == [head, f"({tail}"], (path.name, text)
                seen.add(source)
            else:
                assert "\n" not in text, (path.name, text)
        assert_inside(fig, path.name)
        plt.close(fig)
    assert seen and seen <= labels


def test_metric_a_mode_lacks(synth, tmp_path, capsys):
    # R1-21 F1: --metric parity_grad_rel without --mode used to end in a usage error
    # (an empty log-scale fwd panel) and write nothing. fwd has no gradient errors:
    # its grid is left out with a warning, fwd_bwd's is drawn.
    out = tmp_path / "out"
    argv = ["plot", str(synth), "--out", str(out), "--baseline", "base"]
    with pytest.warns(UserWarning) as caught:
        assert cli.main([*argv, "--metric", "parity_grad_rel"]) == 0
    messages = [str(w.message) for w in caught]
    for dtype in S.DTYPES:
        assert f"fwd, {dtype}: no values of parity_grad_rel: left out" in messages
        assert f"fwd, {dtype}: none of the metrics has values: no grid" in messages
        assert not (out / f"grid-fwd-{dtype}.png").exists()
        assert (out / f"ratio-fwd-{dtype}.png").exists()
        grid = out / f"grid-fwd_bwd-{dtype}.png"
        assert png_size(grid)[0] > 0
        assert {r["metric"] for r in read_csv(grid.with_suffix(".csv"))} == {"parity_grad_rel"}
    assert not any("fwd_bwd" in m and "no values" in m for m in messages)
    # One metric fwd has, one it lacks: fwd's grid has the one, with a warning.
    out = tmp_path / "two"
    argv = ["plot", str(synth), "--out", str(out), "--baseline", "base", "--dtype", "float32"]
    with pytest.warns(UserWarning) as caught:
        assert cli.main([*argv, "--metric", "parity_grad_rel,median_ms"]) == 0
    assert "fwd, float32: no values of parity_grad_rel: left out" in [
        str(w.message) for w in caught
    ]
    assert {r["metric"] for r in read_csv(out / "grid-fwd.csv")} == {"median_ms"}
    assert {r["metric"] for r in read_csv(out / "grid-fwd_bwd.csv")} == {
        "parity_grad_rel",
        "median_ms",
    }


def test_reduction_figure(synth, synth_strict, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    out = tmp_path / "out"
    closed = []
    monkeypatch.setattr(plt, "close", closed.append)
    written = draw(synth, out, dtype="float32", strict=synth_strict)
    monkeypatch.undo()
    figures = dict(zip((p.name for p in written), closed, strict=True))
    names = [p.name for p in written]
    assert names[-2:] == ["reduction-fwd.png", "reduction-fwd_bwd.png"]
    assert "reduction-fwd.png" not in [p.name for p in draw(synth, tmp_path / "plain")]
    path = out / "reduction-fwd_bwd.png"
    assert min(png_size(path)) > 0
    rows = read_csv(path.with_suffix(".csv"))
    bars = {(r["metric"], r["implementation"], r["set"]): float(r["y"]) for r in rows}
    # Per entry a pair of bars, time and the mode's parity metric; compiled is absent
    # from the strict set: no bars, not a failure.
    assert {k[1] for k in bars} == {"base", "base_other", "chunked"}
    t = S.median_ms("base", "fwd_bwd", "float32", S.DEFAULT)
    assert bars[("median_ms", "base", "default")] == pytest.approx(t)
    assert bars[("median_ms", "base", "strict")] == pytest.approx(S.STRICT_TIME * t)
    assert bars[("parity_grad_rel", "base", "default")] == pytest.approx(2e-3)
    assert bars[("parity_grad_rel", "base", "strict")] == pytest.approx(1e-3)
    assert {r["metric"] for r in rows} == {"median_ms", "parity_grad_rel"}
    assert {r["figure"] for r in rows} == {"reduction-fwd_bwd"}
    roles = {r["implementation"]: r["role"] for r in rows}
    assert roles == {"base": "baseline", "base_other": "control", "chunked": "counterpart"}
    assert {r["point"] for r in rows} == {"M=128, K=32, N=16, layout=a"}
    # R1-26 F1: both panels linear from zero, so each bar is as long as its value.
    fig = figures["reduction-fwd_bwd.png"]
    for ax, metric in zip(fig.axes, ("median_ms", "parity_grad_rel"), strict=True):
        assert ax.get_yscale() == "linear"
        assert ax.get_ylim()[0] == 0
        drawn = sorted(round(b.get_height(), 12) for b in ax.patches)
        assert drawn == sorted(round(y, 12) for (m, _, _), y in bars.items() if m == metric)
        assert all(b.get_y() == 0 for b in ax.patches)
    for fig in closed:
        plt.close(fig)
    # fwd: the loss error; base_other is unavailable in the strict fwd float32.
    fwd = read_csv(out / "reduction-fwd.csv")
    assert {r["metric"] for r in fwd} == {"median_ms", "parity_loss_rel_err"}
    assert {r["implementation"] for r in fwd} == {"base"}


def test_reduction_figure_draws_a_zero(synth, synth_strict, tmp_path, monkeypatch):
    # A zero parity error is a result: a bar of length zero, and a row in the CSV.
    value = plot._reduction_value

    def zero_strict(entry, metric, which):
        y = value(entry, metric, which)
        return 0.0 if y is not None and which == "strict" and metric != "median_ms" else y

    monkeypatch.setattr(plot, "_reduction_value", zero_strict)
    out = tmp_path / "out"
    draw(synth, out, dtype="float32", mode="fwd_bwd", strict=synth_strict)
    rows = read_csv(out / "reduction-fwd_bwd.csv")
    zeros = {
        r["implementation"] for r in rows if r["set"] == "strict" and r["metric"] != "median_ms"
    }
    assert zeros == {"base", "base_other", "chunked"}
    assert all(
        float(r["y"]) == 0 for r in rows if r["set"] == "strict" and r["metric"] != "median_ms"
    )


@pytest.mark.parametrize("impls", [None, ["chunked"]], ids=["three-entries", "two-entries"])
def test_reduction_bars_carry_their_values(synth, synth_strict, tmp_path, monkeypatch, impls):
    # F9 (I1-12): torchtitan_ce's loss error of 2.25e-6 had no visible bar on the
    # linear axis. Every bar carries its value, inside its panel, and the layout
    # rules hold with the labels.
    import matplotlib.pyplot as plt

    value = plot._reduction_value

    def planted(entry, metric, which):
        y = value(entry, metric, which)
        tiny = entry["implementation"] == "chunked" and metric != "median_ms"
        return 2.25e-6 if y is not None and tiny else y

    monkeypatch.setattr(plot, "_reduction_value", planted)
    closed = []
    monkeypatch.setattr(plt, "close", closed.append)
    out = tmp_path / "out"
    draw(synth, out, dtype="float32", mode="fwd_bwd", strict=synth_strict, impls=impls)
    monkeypatch.undo()
    fig = closed[-1]
    rows = read_csv(out / "reduction-fwd_bwd.csv")
    for ax, metric in zip(fig.axes, ("median_ms", "parity_grad_rel"), strict=True):
        texts = [t for t in ax.texts if t.get_text()]
        ys = sorted(float(r["y"]) for r in rows if r["metric"] == metric)
        assert len(texts) == len(ax.patches) == len(ys)
        assert sorted(float(t.get_text()) for t in texts) == pytest.approx(ys, rel=5e-3)
        assert ax.get_ylim()[0] == 0  # still linear from zero
        renderer = fig.canvas.get_renderer()
        for t in texts:
            box = t.get_window_extent(renderer)
            assert all(ax.bbox.contains(x, y) for x, y in box.corners()), t.get_text()
    parity = [t.get_text() for t in fig.axes[1].texts]
    assert parity.count("2.25e-06") == 2
    assert_inside(fig, "reduction-fwd_bwd.png")
    for name, (gap, bound) in rendered_gaps(fig).items():
        assert -0.5 <= gap <= bound, (name, gap, bound)
    for f in closed:
        plt.close(f)


def test_reduction_figure_with_impl_and_cli(synth, synth_strict, tmp_path):
    out = tmp_path / "out"
    argv = ["plot", str(synth), "--out", str(out), "--baseline", "base", "--mode", "fwd_bwd"]
    argv += ["--dtype", "bfloat16", "--impl", "chunked", "--strict", str(synth_strict)]
    with pytest.warns(UserWarning, match="different torch-op-bench commits"):
        assert cli.main(argv) == 0
    rows = read_csv(out / "reduction-fwd_bwd.csv")
    assert {r["implementation"] for r in rows} == {"base", "chunked"}


def test_reduction_parity():
    def entry(*names):
        return {"parity": {name: {} for name in names}}

    assert plot.reduction_parity(
        [entry("loss_rel_err"), entry("grad_hidden_rel_l2")], "fwd_bwd"
    ) == ("grad_hidden_rel_l2")
    assert plot.reduction_parity([entry("b", "a")], "fwd_bwd") == "a"
    assert plot.reduction_parity([entry()], "fwd") is None


def test_metric_only_away_from_the_sweeps(synth, tmp_path):
    # R1-22 R1: a parity metric present only in rows the grid does not draw (away from
    # the sweeps around the point) is not a metric of the grid: no empty log panel.
    from torch_op_bench import records

    away = {**S.DEFAULT, "M": 64, "K": 16}  # two axes off the point: in no sweep
    recs = [r for r in records.read(synth) if r["status"] == "ok"]
    planted = next(r for r in recs if r["implementation"] == "base" and r["mode"] == "fwd")
    planted = {
        **planted,
        "case": {**planted["case"], "params": away},
        "case_key": "away",
        "parity": {**planted["parity"], "metrics": {"loss_rel_err": 1e-3, "away_err": 0.1}},
    }
    target = tmp_path / "results" / "synth" / S.SLUG / "x-host-a.jsonl"
    for record in recs:
        records.append(target, record)
    records.append(target, planted)
    with pytest.warns(UserWarning, match="different torch-op-bench commits"):
        rows = analysis.to_rows(analysis.load(tmp_path / "results"))
    assert "parity_away_err" in plot.known_metrics(rows)  # present in the records
    out = tmp_path / "out"
    with pytest.warns(UserWarning) as caught:
        plot.plot(
            tmp_path / "results",
            out,
            baseline="base",
            dtype=planted["case"]["dtype"],
            mode="fwd",
            metrics=["median_ms", "parity_away_err"],
        )
    messages = [str(w.message) for w in caught]
    assert f"fwd, {planted['case']['dtype']}: no values of parity_away_err: left out" in messages
    grid = read_csv(out / "grid-fwd.csv")
    assert {r["metric"] for r in grid} == {"median_ms"}


# --- time at equal peak (Task I1-7) ------------------------------------------------------


def test_equal_peak_figure(synth, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    S.write_variants(synth)
    closed = []
    monkeypatch.setattr(plt, "close", closed.append)
    written = draw(synth, tmp_path, dtype="bfloat16")
    monkeypatch.undo()
    names = [p.name for p in written]
    assert names[-2:] == ["equal-peak-fwd.png", "equal-peak-fwd_bwd.png"]
    rows = read_csv(tmp_path / "equal-peak-fwd_bwd.csv")
    point = "M=128, K=32, N=16, layout=a"
    assert {(r["metric"], r["axis"], r["dtype"], r["mode"], r["point"]) for r in rows} == {
        ("budget", "peak_above_baseline", "bfloat16", "fwd_bwd", point),
        ("median_ms", "peak_above_baseline", "bfloat16", "fwd_bwd", point),
    }
    drawn = [
        (r["implementation"], r["role"], float(r["x"]), r["y"] and float(r["y"]), r["parity_fail"])
        for r in rows
    ]
    # The budgets (64 MiB the baseline's peak), the baseline, then each entry's points
    # by peak: the planted numbers of synth_results.VARIANTS.
    assert drawn == [
        ("1× base", "budget", 64.0, "", ""),
        ("0.5× base", "budget", 32.0, "", ""),
        ("0.25× base", "budget", 16.0, "", ""),
        ("base", "baseline", 64.0, 64.0, "False"),
        ("compiled[mode=a]", "counterpart", 48.0, 38.4, "False"),
        ("compiled", "counterpart", 64.0, 32.0, "False"),
        ("chunked[num_chunks=8]", "counterpart", 8.0, 192.0, "False"),
        ("chunked", "counterpart", 16.0, 128.0, "False"),
        ("chunked[num_chunks=2]", "counterpart", 40.0, 102.4, "True"),
    ]
    fig = closed[names.index("equal-peak-fwd_bwd.png")]
    (ax,) = fig.axes
    # The entries' colours are those of every other figure: the variants take none.
    recs = load_quiet(synth)
    bases = analysis.split_variants(recs)[0]
    palette = plot.colors(analysis.to_rows(bases), "base", plot.stable_order(recs))
    texts = [t.get_text() for t in fig.legends[0].get_texts()]
    assert texts == [
        "base (baseline)",
        "compiled",
        "chunked",
        "budgets: 1×, 0.5×, 0.25× base's peak",
    ]
    # One line per entry through its points; the failing point hollow.
    lines = [
        line for line in ax.get_lines() if len(line.get_xdata()) > 1 and line.get_linestyle() == "-"
    ]
    assert [list(line.get_xdata()) for line in lines] == [[48.0, 64.0], [8.0, 16.0, 40.0]]
    assert [line.get_color() for line in lines] == [palette["compiled"], palette["chunked"]]
    hollow = [m for m in ax.get_lines() if m.get_markerfacecolor() == "none"]
    assert [(list(m.get_xdata()), list(m.get_ydata())) for m in hollow] == [([40.0], [102.4])]
    footer = [t for t in fig.texts if t.get_gid() == "footer"][0].get_text().replace("\n", " ")
    assert "Left out: 1 unavailable" in footer
    assert "Parity failures: 1 (1 explained), drawn hollow." in footer
    assert "Budgets (dotted): 1×, 0.5×, 0.25× base's peak above baseline" in footer
    assert ax.get_xlabel() == "peak above baseline (MiB)" and ax.get_ylabel() == "median\n(ms)"
    for fig in closed:
        plt.close(fig)


def test_variants_only_where_named(synth, tmp_path):
    # Without --impl the grid and ratio figures are as without the variants; naming a
    # variant draws it there (at the default point) and narrows the equal-peak figure
    # to its base.
    draw(synth, tmp_path / "before", dtype="float32", mode="fwd_bwd")
    S.write_variants(synth)
    draw(synth, tmp_path / "after", dtype="float32", mode="fwd_bwd")
    for name in ("grid-fwd_bwd.csv", "ratio-fwd_bwd.csv"):
        assert (tmp_path / "after" / name).read_text() == (tmp_path / "before" / name).read_text()
    written = draw(
        synth, tmp_path / "named", dtype="float32", mode="fwd_bwd", impls=["chunked[num_chunks=2]"]
    )
    assert [p.name for p in written] == [
        "grid-fwd_bwd.png",
        "ratio-fwd_bwd.png",
        "equal-peak-fwd_bwd.png",
    ]
    grid = read_csv(tmp_path / "named" / "grid-fwd_bwd.csv")
    assert {r["implementation"] for r in grid} == {"base", "chunked[num_chunks=2]"}
    assert {r["x"] for r in grid if r["axis"] == "M" and r["implementation"] != "base"} == {"128"}
    equal = read_csv(tmp_path / "named" / "equal-peak-fwd_bwd.csv")
    assert [r["implementation"] for r in equal if r["role"] == "counterpart"] == [
        "chunked[num_chunks=8]",
        "chunked",
        "chunked[num_chunks=2]",
    ]


def test_equal_peak_coincident_entries_stacked(synth, tmp_path, monkeypatch):
    # Two entries whose points coincide one by one (torch_lce and its no-override
    # twin in fwd on ey) are stacked and named in the footer, as in every figure.
    import matplotlib.pyplot as plt

    S.write_variants(synth)
    recs = load_quiet(synth)
    bases, variants = analysis.split_variants(recs)
    rows = analysis.at(analysis.to_rows(bases), dtype="bfloat16", mode="fwd_bwd")
    variant_rows = analysis.at(analysis.to_rows(variants), dtype="bfloat16", mode="fwd_bwd")
    twins = [
        {**r, "implementation": r["implementation"].replace("chunked", "twin"), "base": "twin"}
        for r in variant_rows
        if r["base"] == "chunked"
    ]
    rows += [{**r, "implementation": "twin"} for r in rows if r["implementation"] == "chunked"]
    closed = []
    monkeypatch.setattr(plt, "close", closed.append)
    palette = plot.colors(rows, "base")
    plot.plot_equal_peak(
        rows,
        variant_rows + twins,
        tmp_path,
        S.DEFAULT,
        "base",
        palette,
        title="synth",
        notes=[],
        axes=list(S.DEFAULT),
        dtypes=["bfloat16"],
        modes=["fwd_bwd"],
    )
    monkeypatch.undo()
    (fig,) = closed
    footer = [t for t in fig.texts if t.get_gid() == "footer"][0].get_text().replace("\n", " ")
    assert (
        "Coincident, the first drawn dashed on the second: twin on chunked (fwd_bwd: median)."
        in footer
    )
    (ax,) = fig.axes
    dashed = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
    assert [line.get_color() for line in dashed] == [palette["twin"]]
    assert "compiled on" not in footer and "on compiled" not in footer
    plt.close(fig)
