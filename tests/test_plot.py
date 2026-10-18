import csv
import struct
import sys

import pytest
import synth_results as S

from torch_op_bench import analysis, cli, plot

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
    # 4 swept axes x 3.8 in wide; 4 metrics x 2.8 in, plus the title, one legend row
    # and two footer lines, high; at 150 dpi.
    width, height = png_size(tmp_path / "out" / "grid-fwd_bwd.png")
    assert width == 4 * 3.8 * 150
    assert height == round((0.45 + 0.1 + 0.24 + 4 * 2.8 + 0.15 + 2 * 0.17) * 150)
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
    assert order == [i.name for i in lce.LCE.implementations if i.info.get("role") != "control"]
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
    assert footer[0] == "one: Left out: 1 unavailable (gaps in the lines)."
    assert footer[1] == "two: Left out: 1 error, 1 unavailable (gaps in the lines)."
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

    closed = []
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
    suptitle = fig._suptitle
    assert suptitle.get_text().count("\n") >= 1
    assert suptitle.get_text().replace("\n", " ") == title
    renderer = fig.canvas.get_renderer()
    box, page = suptitle.get_window_extent(renderer), fig.bbox
    assert page.x0 <= box.x0 and box.x1 <= page.x1, (box, page)
    assert page.y0 <= box.y0 and box.y1 <= page.y1, (box, page)
    plt.close(fig)
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
    assert "\n" not in closed[0]._suptitle.get_text()
    plt.close(closed[0])


def load_quiet(synth):
    with pytest.warns(UserWarning):
        return analysis.load(synth)
