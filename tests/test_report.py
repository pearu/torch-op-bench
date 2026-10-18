import json
import re

import pytest
import synth_results as S

from torch_op_bench import analysis, cli, report

MERGE_WARNING = "different torch-op-bench commits"


def make(synth, **kwargs):
    return report.report(synth, baseline="base", **kwargs)


def section(text, start, end="\n## "):
    begin = text.index(start)
    stop = text.find(end, begin + len(start))
    return text[begin : stop if stop != -1 else None]


def cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def table_rows(text):
    return [line for line in text.splitlines() if line.startswith("| ")]


def test_numbers():
    assert report.sig3(None) == "—"
    assert report.sig3(0.5) == "0.500"
    assert report.sig3(128.0) == "128"
    assert report.sig3(1234.5) == "1230"  # 3 significant digits, left of the point too
    assert report.sig3(99.96) == "100"
    assert report.sig3(0.0099996) == "0.0100"
    assert report.sig3(0.000123) == "1.23e-04"
    assert report.sig3(2.5e9) == "2.50e+09"
    assert report.sig3(0) == "0"
    assert report.sig3(True) == "True"
    assert report.mib(None) == "—" and report.mib(3 * 1024**2 / 2) == "1.5"
    assert report.text("a|b\nc") == "a\\|b c"
    assert report.text("x" * 50, 40) == "x" * 39 + "…"
    assert report.text("x" * 40, 40) == "x" * 40


def test_header_and_warning(synth):
    text = make(synth, command=["torch-op-bench", "report", "results", "--baseline", "base"])
    assert text.startswith("# Report: synth on synthdev\n")
    assert "Made by `torch-op-bench report results --baseline base`." in text
    assert f"**Warning:** records come from {MERGE_WARNING}" in text


def test_provenance(synth):
    text = section(make(synth), "## Provenance")
    assert "- hosts: host-a, host-b" in text
    rows = table_rows(text)
    assert len(rows) == 2 + 2  # header, alignment, one row per environment build
    assert "| other | 2.14.1+cu130 | 13.0 | 3.13.1 | counterpart=0123456789ab; helper=1.0 |" in text
    assert "| torch | 2.16.0a0+gitabc | 13.0 | 3.14.8 | — | aaaaaaaaaaaa |" in text
    assert "bf16_reduced=True" in text
    assert "**The records come from different torch-op-bench commits**" in text


def test_default_point(synth):
    text = make(synth)
    assert "The default point is the op's defaults." in text
    part = section(text, "### fwd_bwd, float32", "\n### ")
    names = [cells(row)[0] for row in table_rows(part)[2:]]
    assert names == ["base", "base_other (control of base)", "compiled", "chunked"]
    header = cells(table_rows(part)[0])
    by_name = {cells(row)[0]: cells(row) for row in table_rows(part)[2:]}
    col = header.index("time / baseline")
    for name in ["compiled", "chunked"]:
        expected = S.median_ms(name, "fwd_bwd", "float32", S.DEFAULT) / S.median_ms(
            "base", "fwd_bwd", "float32", S.DEFAULT
        )
        assert by_name[name][col] == report.sig3(expected)
    assert by_name["chunked"][col] == "2.00" and by_name["compiled"][col] == "0.500"
    assert by_name["base_other (control of base)"][col] == "1.10"
    # The peak above the inputs, not the absolute peak (R1-12 F2), so labelled.
    assert "peak MiB" not in header and "peak / baseline" not in header
    peak = header.index("peak above baseline, MiB")
    assert by_name["chunked"][peak] == report.mib(S.peak_above("chunked", S.DEFAULT))
    ratio = header.index("peak above baseline / baseline's")
    assert by_name["chunked"][ratio] == "0.250"
    # Elapsed time x peak memory, the user's comparison, and its ratio (Part B).
    product = header.index("time × peak, ms·MiB")
    t = S.median_ms("chunked", "fwd_bwd", "float32", S.DEFAULT)
    expected = t * S.peak_above("chunked", S.DEFAULT) / S.MIB
    assert by_name["chunked"][product] == report.sig3(expected)
    assert header[product + 1] == "time × peak / baseline's"
    assert by_name["chunked"][product + 1] == "0.500"  # 2x the time, 1/4 of the peak
    assert by_name["compiled"][product + 1] == "0.500"
    assert by_name["base_other (control of base)"][product + 1] == "1.10"
    assert by_name["base"][product + 1] == "1.00"
    assert "api.name" in header and "num_chunks" in header  # varying info
    # fwd: chunked has no fwd records, and no info column is empty there
    part = section(text, "### fwd, float32", "\n### ")
    assert "chunked" not in part and "api.name" not in part


def test_sweeps(synth):
    text = make(synth, dtype="float32", mode="fwd_bwd")
    part = section(text, "#### fwd_bwd, float32: parity against M", "\n####")
    assert "| 256 | ok | ok | ok | fail: grad_rel |" in part
    part = section(text, "#### fwd_bwd, float32: median ms against N", "\n####")
    assert "| 8 | 64.0 | 64.0 | — | 128 |" in part  # compiled has no record at N=8
    part = section(text, "#### fwd_bwd, float32: median ms against K", "\n####")
    assert "| 16 | 64.0 | 64.0 | 32.0 | unavailable |" in part
    part = section(
        text, "#### fwd_bwd, float32: time × peak, ms·MiB (/ baseline's) against M", "\n####"
    )
    base = S.median_ms("base", "fwd_bwd", "float32", S.DEFAULT) * S.peak_above("base", S.DEFAULT)
    base /= S.MIB
    assert f"| 128 | {report.sig3(base)} (1.00) | " in part
    assert f"(1.10) | {report.sig3(base / 2)} (0.500) | {report.sig3(base / 2)} (0.500) |" in part
    part = section(text, "#### fwd_bwd, float32: median ms against layout", "\n####")
    assert "| :--- |" in part.splitlines()[3]  # a str axis is not right-aligned
    assert "### fwd, " not in text and "bfloat16" not in text.split("## Environment check")[0]


def test_environment_check(synth):
    text = section(make(synth), "## Environment check")
    flagged = [line for line in text.splitlines() if line.startswith("- **base_other**")]
    # time and peak in both modes and dtypes; grad_rel in fwd_bwd, both dtypes
    assert len(flagged) == 2 * 2 * 2 + 2
    assert any(
        "fwd_bwd float32 at M=128, K=32, N=16, layout=a: time ratio 1.10 outside [0.95, 1.05]"
        in line
        for line in flagged
    )
    assert any(
        "at M=256, K=32, N=16, layout=a: peak difference 100.0 MiB" in line for line in flagged
    )
    # The parity of the same code moved by the environment (R1-12 R1).
    assert any(
        "fwd_bwd float32 at M=128, K=32, N=8, layout=a: grad_rel 0.00250 against 0.00200 "
        "(ratio 1.25) outside [0.9, 1.1]" in line
        for line in flagged
    )
    header = cells(table_rows(text)[0])
    assert header[-3:] == ["grad_rel ratio", "loss_rel_err ratio", "flag"]
    rows = [cells(r) for r in table_rows(text)[2:]]
    assert len(rows) == 6 * 2 * 2  # every point both have, both modes, both dtypes
    assert sum(r[-1] == "" for r in rows) == len(rows) - 10
    at_n8 = [r for r in rows if r[1] == "base" and "N=8" in r[4] and r[2] == "fwd_bwd"]
    assert [r[-3:] for r in at_n8] == [["1.25", "1.00", "grad_rel"]] * 2
    fwd = [r for r in rows if r[2] == "fwd"]
    assert {r[-3] for r in fwd} == {"—"}  # no grad_rel in fwd: neither side has it


def test_parity_ratios():
    control = {"parity_a": 2.0, "parity_b": 1.0, "parity_c": 3.0, "parity_z": 0.0}
    original = {"parity_a": 1.0, "parity_b": 1.0, "parity_d": 4.0, "parity_z": 0.0}
    assert report.parity_ratios(control, original) == {
        "a": {"control": 2.0, "original": 1.0, "ratio": 2.0, "side": None},
        "b": {"control": 1.0, "original": 1.0, "ratio": 1.0, "side": None},
        "c": {"control": 3.0, "original": None, "ratio": None, "side": "control only"},
        "d": {"control": None, "original": 4.0, "ratio": None, "side": "original only"},
        "z": {"control": 0.0, "original": 0.0, "ratio": 1.0, "side": None},
    }
    base = {"implementation": "e", "mode": "fwd", "status": "ok", "median_ms": 1.0}
    rows = [
        {**base, **original},
        {
            **base,
            **control,
            "implementation": "c",
            "role": "control",
            "control_of": "e",
            "env_name": "x",
        },
    ]
    text = report.render_environment_check(rows, [])
    assert "a 2.00 against 1.00 (ratio 2.00) outside [0.9, 1.1]" in text
    assert "c on the control only" in text and "d on the original only" in text
    assert "| control only | original only |" in text
    # Only one side: reported, not flagged as the environment's difference.
    one_sided = [
        {**base, "parity_d": 1.0},
        {**rows[1], "parity_a": None, "parity_b": None, "parity_c": 1.0, "parity_z": None},
    ]
    text = report.render_environment_check(one_sided, [])
    assert "c on the control only and d on the original only." in text
    assert "differs from the baseline's" not in text
    # Zero against nonzero: no ratio, both values shown, flagged.
    zero = [
        {**base, "parity_a": 0.0},
        {**rows[1], "parity_a": 1.0, "parity_b": None, "parity_c": None, "parity_z": None},
    ]
    checks = report.environment_check(zero, [])
    assert checks[0]["flags"] == ["a"]
    assert "| 1.00 / 0 |" in report.render_environment_check(zero, [])


def test_environment_check_without_controls():
    rows = [{"implementation": "a", "mode": "fwd", "status": "ok", "role": None}]
    assert "No control has a record" in report.render_environment_check(rows, [])
    control = {
        "implementation": "c",
        "mode": "fwd",
        "status": "ok",
        "role": "control",
        "control_of": "a",
        "median_ms": 1.0,
        "peak_above_baseline": 0,
    }
    original = {
        "implementation": "a",
        "mode": "fwd",
        "status": "ok",
        "median_ms": 1.01,
        "peak_above_baseline": 0,
    }
    text = report.render_environment_check([original, control], [])
    assert "Every control is within the limits" in text


def test_not_measured(synth):
    text = section(make(synth), "## Not measured")
    assert (
        "- error: RuntimeError: boom\n  - compiled, fwd, float32 M=64, K=32, N=16, layout=a" in text
    )
    assert "second line" not in text  # the reason's first line only
    assert "- unavailable: needs a GPU\n  - chunked, fwd_bwd, float32 M=128, K=16" in text
    rows = [{"status": "ok"}]
    assert "Every record is `ok`." in report.render_not_measured(rows, [])


def test_missing_baseline_and_point(synth):
    text = report.report(synth, baseline="no_such", dtype="float32", mode="fwd")
    assert "The baseline `no_such` has no `ok` record here: no ratios." in text
    assert "No records at the default point." in report.render_default(
        [], {"M": 1}, "base", title="t"
    )


def test_inferred_default_point(synth, tmp_path):
    # An op_ref that does not resolve: the default point comes from the records.
    out = tmp_path / "other"
    for path in synth.rglob("*.jsonl"):
        target = out / path.relative_to(synth)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        target.write_text(
            "".join(json.dumps({**r, "op_ref": "no_such_module_xyz:SPEC"}) + "\n" for r in lines)
        )
    text = report.report(out, baseline="base", dtype="float32", mode="fwd")
    assert "The default point is inferred from the records." in text


def test_several_ops(synth):
    extra = synth / "synth2" / S.SLUG / "20261017T130000Z-host-a.jsonl"
    with pytest.warns(UserWarning, match=MERGE_WARNING):
        record = analysis.load(synth)[0]
    record = {k: v for k, v in record.items() if k != analysis.SOURCE}
    from torch_op_bench import records

    records.append(extra, {**record, "op": "synth2"})
    with pytest.raises(ValueError, match="several ops .*: choose one with --op"):
        report.report(synth)
    assert report.report(synth, op="synth", baseline="base").startswith("# Report: synth on")
    with pytest.raises(ValueError, match="no records under"):
        empty = synth.parent / "empty"
        empty.mkdir()
        report.report(empty)


def test_cli(synth, tmp_path, capsys):
    assert cli.main(["report", str(synth), "--baseline", "base", "--dtype", "float32"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Report: synth on synthdev")
    assert f"Made by `torch-op-bench report {synth} --baseline base --dtype float32`." in out
    target = tmp_path / "reports" / "r.md"
    assert cli.main(["report", str(synth), "--baseline", "base", "--out", str(target)]) == 0
    assert capsys.readouterr().out == f"wrote {target}\n"
    assert target.read_text().startswith("# Report: synth on synthdev")
    with pytest.raises(SystemExit):
        cli.main(["report", str(tmp_path / "nothing")])
    assert "no results at" in capsys.readouterr().err
    argv = ["report", str(synth), "--baseline", "base", "--point", "M=64", "--mode", "fwd"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "The default point is set by --point." in out and "At M=64, K=32, N=16" in out
    with pytest.raises(SystemExit):
        cli.main(["report", str(synth), "--point", "M"])
    assert "--point: expected AXIS=VALUE" in capsys.readouterr().err


def test_no_sweep(synth, tmp_path):
    # Only the default point's records: nothing varies around it.
    from torch_op_bench import records

    only = tmp_path / "only"
    with pytest.warns(UserWarning, match=MERGE_WARNING):
        recs = analysis.load(synth)
    for r in recs:
        if all(r["case"]["params"][k] == v for k, v in S.DEFAULT.items()):
            r = {k: v for k, v in r.items() if k != analysis.SOURCE}
            records.append(only / "synth" / S.SLUG / "x-host.jsonl", r)
    text = report.report(only, baseline="base")
    assert "## Sweeps\n\nNo axis varies around the default point in these records: one table" in (
        text
    )
    assert "### fwd_bwd, float32" in text  # the default point is still reported
    # ... and, in place of the sweeps, one table per case: here the one case.
    assert "### fwd_bwd, float32: M=128, K=32, N=16, layout=a" in text
    assert "No axis varies" not in make(synth)


def test_without_the_default_point(synth, tmp_path):
    # Two cases that differ in two axes and miss the spec's default point, as the
    # I1-6 run on ex did: an inferred point, its source, and a table per case.
    two = S.subset(synth, tmp_path / "two", {"M": 64}, {"K": 16})
    text = report.report(two, baseline="base")
    spec_point = "M=128, K=32, N=16, layout=a"
    assert (
        f"The default point is inferred (the spec's default point {spec_point} has no records)."
        in text
    )
    with pytest.warns(UserWarning, match=MERGE_WARNING):
        point = analysis.default_point(analysis.load(two))
    assert point["params"] in [{**S.DEFAULT, "M": 64}, {**S.DEFAULT, "K": 16}]
    assert "No records at the default point." not in text
    part = section(text, "## Sweeps")
    assert "No axis varies around the default point in these records: one table" in part
    for case in ["M=64, K=32, N=16, layout=a", "M=128, K=16, N=16, layout=a"]:
        for mode in ["fwd", "fwd_bwd"]:
            for dtype in ["float32", "bfloat16"]:
                assert f"### {mode}, {dtype}: {case}\n\nAt {case}.\n" in part
    # The rows of the default-point table, its ratios included.
    table = section(part, "### fwd_bwd, float32: M=64,", "\n### ")
    header = cells(table_rows(table)[0])
    by_name = {cells(row)[0]: cells(row) for row in table_rows(table)[2:]}
    assert by_name["chunked"][header.index("time / baseline")] == "2.00"
    assert by_name["chunked"][header.index("time × peak / baseline's")] == "0.500"
    # A partial --point names one axis; the rest stay the spec's.
    text = report.report(two, baseline="base", point={"M": "64"})
    assert "The default point is set by --point." in text
    assert "At M=64, K=32, N=16, layout=a." in section(text, "## At the default point")


def test_point(synth):
    text = make(synth, dtype="float32", mode="fwd_bwd", point={"K": "16"})
    assert "The default point is set by --point." in text
    assert "At M=128, K=16, N=16, layout=a." in text
    # The sweeps are around that point now: the synthetic set varies only K there.
    part = section(text, "#### fwd_bwd, float32: median ms against K", "\n####")
    base = S.median_ms("base", "fwd_bwd", "float32", {**S.DEFAULT, "K": 16})
    assert f"| 16 | {report.sig3(base)} |" in part
    assert "median ms against M" not in text
    with pytest.raises(ValueError, match="--point: unknown axis 'X'; known: M, K, N, layout"):
        make(synth, point={"X": "1"})
    assert analysis.parse_point(["M=64,K=16", "layout=b"]) == {"M": "64", "K": "16", "layout": "b"}
    assert analysis.parse_point(None) is None
    with pytest.raises(ValueError, match="expected AXIS=VALUE, got 'M'"):
        analysis.parse_point("M")


def test_provenance_builds():
    def rec(env_name, version, tob):
        return {
            "device_slug": "d",
            "env_name": env_name,
            "timestamp": "t",
            "env": {
                "hostname": "h",
                "python": "3.14",
                "torch": {"version": version, "cuda": "13.0"},
                "torch_op_bench": {"hash": tob},
                "packages": {"p": "1"},
            },
        }

    # Another torch-op-bench commit alone is not another build of the environment.
    text = report.render_provenance([rec("e", "2.1", "a" * 40), rec("e", "2.1", "b" * 40)], [])
    assert "different torch-op-bench commits" in text and "different builds" not in text
    text = report.render_provenance([rec("e", "2.1", "a" * 40), rec("e", "2.2", "a" * 40)], [])
    assert "**Environment `e` has 2 different builds**" in text
    assert "different torch-op-bench commits" not in text
    runner_only = {"device_slug": "d", "env_name": "e", "timestamp": "t", "env": None}
    assert "| e |" not in report.render_provenance([runner_only], [])


def test_per_case_per_dtype_and_mode(synth, tmp_path):
    # R1-19 F1: float32 sweeps around the point, bfloat16 has only two cases that
    # differ in two axes. Both appear under Sweeps, each in its own way.
    from torch_op_bench import records

    with pytest.warns(UserWarning, match=MERGE_WARNING):
        recs = analysis.load(synth)
    cases = [{**S.DEFAULT, "M": 64}, {**S.DEFAULT, "K": 16}]
    target = tmp_path / "mixed" / "synth" / S.SLUG / "x-host.jsonl"
    for r in recs:
        if r["case"]["dtype"] == "float32" or r["case"]["params"] in cases:
            records.append(target, {k: v for k, v in r.items() if k != analysis.SOURCE})
    text = report.report(tmp_path / "mixed", baseline="base")
    part = section(text, "## Sweeps")
    assert "#### fwd_bwd, float32: median ms against M" in part
    assert "#### fwd, float32: median ms against K" in part
    assert "### fwd_bwd, float32: M=" not in part  # float32 has sweeps, no per-case
    for mode in ["fwd", "fwd_bwd"]:
        note = f"**{mode}, bfloat16:** no axis varies around the default point: one table"
        assert note in part
        for case in ["M=64, K=32, N=16, layout=a", "M=128, K=16, N=16, layout=a"]:
            assert f"### {mode}, bfloat16: {case}\n\nAt {case}.\n" in part
    # The all-pairs line is for when no pair has a sweep.
    assert "No axis varies around the default point in these records" not in part


def test_reduction_section(synth, synth_strict):
    text = make(synth, strict=synth_strict)
    assert "## Reduction setting" not in make(synth)  # only with --strict
    part = section(text, "## Reduction setting")
    # The check: the one strict record with the default setting is named.
    impl, mode, dtype = S.STRICT_ODD
    assert (
        f"**Warning:** 1 record(s) of RESULTS_STRICT do not have the strict setting in "
        f"`env.torch.matmul`: {impl} {mode} {dtype} M=128, K=32, N=16, layout=a (default)."
    ) in part
    assert "of RESULTS do not" not in part
    bwd = section(part, "### fwd_bwd, float32", "\n### ")
    header, _, *body = table_rows(bwd)
    header = cells(header)
    rows = {cells(line)[0]: dict(zip(header, cells(line), strict=True)) for line in body}
    assert list(rows) == ["base", "base_other (control of base)", "compiled", "chunked"]
    base = rows["base"]
    assert base["time strict / default"] == "1.20"
    assert base["grad_rel default"] == "0.00200" and base["grad_rel strict"] == "0.00100"
    assert base["grad_rel default / strict"] == "2.00"
    assert base["parity moves"] == "yes: grad_rel, loss_rel_err"
    other = rows["base_other (control of base)"]
    assert other["grad_rel default / strict"] == "0.952" and other["parity moves"] == "no"
    assert rows["chunked"]["parity moves"] == "no"
    # An entry in one directory only, or unavailable in one: "not measured".
    assert rows["compiled"]["median ms default"] == "not measured (default: ok, strict: absent)"
    assert rows["chunked"]["op_decisions"] == "default: —; strict: _mm_accumulates_in_fp32=True"
    assert base["op_decisions"] == "—"
    # The reading, computed.
    assert (
        "Under the defaults, `base_other`, `chunked` have the strict parity (fp32 "
        "accumulation or no small-M GEMMs); `base` do not. Not compared (not measured in "
        "both, or no parity): `compiled`."
    ) in bwd
    assert "torch LCE: `chunked`'s default parity equals its strict parity." in bwd
    fwd = section(part, "### fwd, float32", "\n### ")
    impl, mode, dtype = S.STRICT_UNAVAILABLE
    assert "not measured (default: ok, strict: unavailable)" in fwd
    assert "`base_other`, `compiled`." in fwd and "op_decisions" not in fwd
    assert "torch LCE" not in fwd  # no entry with op_decisions in fwd


def test_reduction_section_warns_about_the_default_set(synth, synth_strict):
    # The strict set as RESULTS and the default one as --strict: every record warned.
    text = report.report(synth_strict, baseline="base", strict=synth, mode="fwd_bwd")
    part = section(text, "## Reduction setting")
    assert "record(s) of RESULTS do not have the default setting" in part
    assert "record(s) of RESULTS_STRICT do not have the strict setting" in part
    assert re.search(r" and \d+ more\.", part)  # the names are capped
    assert "### fwd_bwd, float32" in part  # the section still renders


def test_reduction_cli(synth, synth_strict, capsys):
    argv = ["report", str(synth), "--baseline", "base", "--strict", str(synth_strict)]
    assert cli.main([*argv, "--dtype", "bfloat16", "--mode", "fwd_bwd"]) == 0
    out = capsys.readouterr().out
    assert "## Reduction setting" in out and "### fwd_bwd, bfloat16" in out
    assert "### fwd_bwd, float32" not in section(out, "## Reduction setting")


def test_decisions_text():
    assert report.decisions_text([None, None]) == "—"
    pr = {"source": "m", "_mm_fp32_out_available": True, "_mm_accumulates_in_fp32": False}
    both = "_mm_fp32_out_available=True, _mm_accumulates_in_fp32=False"
    assert report.decisions_text([pr, dict(pr)]) == both
    strict = {**pr, "_mm_accumulates_in_fp32": True}
    assert report.decisions_text([pr, strict]) == (
        f"default: {both}; strict: _mm_fp32_out_available=True, _mm_accumulates_in_fp32=True"
    )
    ctx = {"source": "c", "is_cuda": True, "use_acc_dtype": True, "alloc_x": False}
    assert report.decisions_text([ctx, ctx]) == "is_cuda=True, use_acc_dtype=True"
    gone = {"unavailable": "no module"}
    assert report.decisions_text([gone, gone]) == "unavailable: no module"
