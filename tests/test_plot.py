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
    assert list(dict.fromkeys(r["metric"] for r in rows)) == [
        "speedup",
        "memory_ratio",
        "throughput_ratio",
    ]
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
    palette = plot.colors(rows, "base")
    assert palette["base"] == "black" and palette["base_other"] == "black"
    assert palette["compiled"] == plot.COLORS[0] and palette["chunked"] == plot.COLORS[1]
    # A control of a counterpart takes its colour; without the baseline drawn (the
    # ratio figure), a control of the baseline is still black.
    rows2 = [
        {**r, "control_of": "chunked"} if r["implementation"] == "base_other" else r for r in rows
    ]
    assert plot.colors(rows2, "base")["base_other"] == plot.COLORS[1]
    no_base = [r for r in rows if r["implementation"] != "base"]
    assert plot.colors(no_base, "base")["base_other"] == "black"
    assert plot.STYLE["baseline"]["linestyle"] == "--" and plot.STYLE["control"]["linestyle"] == ":"


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
