import json
import re
import warnings

import pytest
import synth_op
import synth_results as S
import toy_op

from torch_op_bench import analysis, cli, report
from torch_op_bench.spec import ParityNote

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


def tables(text):
    """Each markdown table of ``text``, as its lines."""
    found = [[]]
    for line in text.splitlines():
        if line.startswith("| "):
            found[-1].append(line)
        elif found[-1]:
            found.append([])
    return [t for t in found if t]


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


def test_provenance_counts_each_directory(synth, synth_strict):
    # I1-19 stumble 6: the records of RESULTS and of --strict's directory, each named.
    with pytest.warns(UserWarning, match=MERGE_WARNING):
        n = len(analysis.load(synth))
    with pytest.warns(UserWarning, match=MERGE_WARNING):
        n_strict = len(analysis.load(synth_strict))
    line = next(x for x in make(synth).splitlines() if x.startswith("- records from"))
    assert line.endswith(f" ({n} records in `{synth}`)")
    line = next(
        x for x in make(synth, strict=synth_strict).splitlines() if x.startswith("- records from")
    )
    assert line.endswith(f" ({n} records in `{synth}`, {n_strict} in `{synth_strict}`)")
    assert n_strict and n_strict != n


def test_default_point(synth):
    text = make(synth)
    assert "The default point is the op's defaults." in text
    part = section(text, "### fwd_bwd, float32", "\n### ")
    names = [cells(row)[0] for row in tables(part)[0][2:]]
    assert names == ["base", "base_other (control of base)", "compiled", "chunked"]
    header = cells(tables(part)[0][0])
    by_name = {cells(row)[0]: cells(row) for row in tables(part)[0][2:]}
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
    assert header[2] == "median ms (IQR)" and "status" not in header
    assert by_name["chunked"][2] == "256 (25.6)"
    # The info fields that differ between records: a table of their own (R1-30 F6).
    details = tables(part)[1]
    assert cells(details[0]) == ["implementation", "info"]
    by_name = {cells(row)[0]: cells(row)[1] for row in details[2:]}
    assert list(by_name) == names
    assert by_name["chunked"] == "api.name=v2<br>num_chunks=4"  # one per line
    assert by_name["compiled"] == "compile=fullgraph" and by_name["base"] == "—"
    assert "Entry details: the `info` fields that differ between records." in part
    # fwd: chunked has no fwd records, and no info field of its is there
    part = section(text, "### fwd, float32", "\n### ")
    assert "chunked" not in part and "api.name" not in part
    # Not ok: the status in the median's place.
    text = make(synth, dtype="float32", mode="fwd", point={"M": "64"})
    part = section(text, "### fwd, float32", "\n## ")
    by_name = {cells(row)[0]: cells(row) for row in tables(part)[0][2:]}
    assert by_name["compiled"][2] == "error"


def test_override_column(synth):
    # Whether an override ran: its dispatch and its own kernel (R1-30 N1), from the
    # verify fields; the synth set records neither, so they are planted on its rows.
    with pytest.warns(UserWarning, match=MERGE_WARNING):
        rows = analysis.to_rows(analysis.load(synth))
    planted = {
        "base": (False, False),
        "base_other": (True, True),
        "compiled": (True, False),
    }
    here = [dict(r) for r in analysis.at(rows, dtype="float32", mode="fwd")]
    for row in here:
        ran, kernel = planted[row["implementation"]]
        row.update(override_ran=ran, override_kernel_ran=kernel)
    part = report.render_default(here, S.DEFAULT, "base", title="fwd, float32")
    header = cells(tables(part)[0][0])
    assert "override_ran" not in header and header[-1] == "override"
    by_name = {cells(row)[0]: cells(row)[-1] for row in tables(part)[0][2:]}
    assert by_name == {
        "base": "no",
        "base_other (control of base)": "kernel",
        "compiled": "dispatch only",
    }
    note = (
        "`compiled` in fwd dispatches to the override, which runs no kernel of its own "
        "on this path (`override_kernel_ran` false): its fwd time is not the override "
        "kernel's."
    )
    assert note in part
    assert part.index(note) < part.index("Entry details")  # under the table it is about
    # Two such entries: one sentence naming both.
    for row in here:
        if row["implementation"] == "base_other":
            row["override_kernel_ran"] = False
    part = report.render_default(here, S.DEFAULT, "base", title="fwd, float32")
    assert (
        "`base_other (control of base)`, `compiled` in fwd dispatch to the override, "
        "which runs no kernel of its own on this path (`override_kernel_ran` false): "
        "their fwd time is not the override kernel's."
    ) in part
    # None dispatch only: no sentence.
    for row in here:
        row["override_kernel_ran"] = row["override_ran"]
    assert "dispatch" not in report.render_default(here, S.DEFAULT, "base", title="t")
    # A record older than override_kernel_ran; an entry recording neither.
    assert report.override_text({"override_ran": True, "override_kernel_ran": None}) == ("dispatch")
    assert report.override_text({"override_ran": None}) == "—"
    assert report.override_text({}) == "—"
    # The report itself: the column, and no sentence on the synth set as it is.
    text = make(synth)
    part = section(text, "### fwd, float32", "\n### ")
    assert cells(tables(part)[0][0])[-1] == "override"
    assert {cells(row)[-1] for row in tables(part)[0][2:]} == {"—"}
    assert "dispatches to the override" not in text


def widen(root, out):
    """``root``'s records in the shape of I1-12's: 11 entries in fwd_bwd (copies of
    ``compiled``), four parity metrics, and two entries whose ``info`` keys are
    disjoint, 40 in all."""
    for path in root.rglob("*.jsonl"):
        lines = []
        for line in path.read_text().splitlines():
            record = json.loads(line)
            copies = [record]
            if record["implementation"] == "compiled":
                record["info"].update({f"a{i:02}": i for i in range(20)})
                for k in range(7):
                    copy = json.loads(line)
                    copy["implementation"] = f"compiled_{k}"
                    copies.append(copy)
            if record["implementation"] == "chunked":
                record["info"].update({f"b{i:02}": "x" * i for i in range(20)})
            for r in copies:
                metrics = (r.get("parity") or {}).get("metrics")
                if metrics:
                    metrics.update(m3=1e-3, m4=2e-3)
            lines += [json.dumps(r) for r in copies]
        target = out / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n")
    return out


def test_tables_within_max_columns(synth, synth_strict, tmp_path):
    # R1-30 F6: no table of the report is wider than MAX_COLUMNS, on the synthetic set
    # and on one of I1-12's width; and every row has the header's cells.
    wide = widen(synth, tmp_path / "wide")
    wide_strict = widen(synth_strict, tmp_path / "wide-strict")
    for text in [make(synth, strict=synth_strict), make(wide, strict=wide_strict)]:
        found = tables(text)
        assert len(found) > 10
        for lines in found:
            width = len(cells(lines[0]))
            assert width <= report.MAX_COLUMNS, lines[0]
            assert {len(cells(line)) for line in lines} == {width}, lines[0]
    part = section(text, "### fwd_bwd, float32", "\n### ")
    default, details = tables(part)
    assert len(cells(default[0])) == report.MAX_COLUMNS  # 4 parity metrics
    assert len(default) == 2 + 11
    by_name = {cells(row)[0]: cells(row)[1] for row in details[2:]}
    assert by_name["compiled"].startswith("a00=0<br>a01=1<br>") and "b00" not in by_name["compiled"]
    assert by_name["chunked"].startswith("api.name=v2<br>b00=<br>b01=x<br>")
    assert by_name["chunked"].count("=") == 22  # api.name, b00..b19, num_chunks


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


def test_parity_notes(synth):
    # R1-30 F1: synth_op's note explains chunked's grad_rel failure at M=256 in
    # bfloat16, not in float32; its other note names a metric chunked never fails.
    text = make(synth, mode="fwd_bwd")
    footnote = f"¹ `chunked`, grad_rel: {synth_op.NOTE.text}"
    part = section(text, "#### fwd_bwd, bfloat16: parity against M", "\n####")
    assert "| 256 | ok | ok | ok | fail: grad_rel¹ |" in part
    assert footnote in part.splitlines()
    part = section(text, "#### fwd_bwd, float32: parity against M", "\n####")
    assert "| 256 | ok | ok | ok | fail: grad_rel |" in part and "¹" not in part
    # Only the parity table of the failure carries it.
    assert text.count(footnote) == 1 and synth_op.NEVER.text not in text
    part = section(text, "## Parity failures")
    assert text.index("## Parity failures") < text.index("## Environment check")
    lines = part.splitlines()
    i = lines.index(f"- `chunked`, explained: {synth_op.NOTE.text}")
    assert lines[i + 1 : i + 6] == [
        "  - grad_rel (1)",
        "    - fwd_bwd, bfloat16 M=256, K=32, N=16, layout=a",
        "- `chunked`, **unexplained**",
        "  - grad_rel (1)",
        "    - fwd_bwd, float32 M=256, K=32, N=16, layout=a",
    ]
    assert "2 parity failures, 1 explained." in lines
    # One mode with no failure.
    part = section(make(synth, mode="fwd"), "## Parity failures")
    assert "No record fails parity." in part and "0 parity failures, 0 explained." in part


def test_parity_notes_without_the_op(synth, monkeypatch):
    # The op does not import here: no note, `fail` as it was, every failure unexplained.
    def unresolvable(ref):
        raise ModuleNotFoundError(ref)

    monkeypatch.setattr(analysis, "_SPECS", {})
    monkeypatch.setattr("torch_op_bench.spec.resolve_op", unresolvable)
    text = make(synth, mode="fwd_bwd")
    part = section(text, "#### fwd_bwd, bfloat16: parity against M", "\n####")
    assert "| 256 | ok | ok | ok | fail: grad_rel |" in part
    assert "¹" not in text and synth_op.NOTE.text not in text
    part = section(text, "## Parity failures")
    assert part.rstrip().splitlines()[-2:] == [
        "2 parity failures, 0 explained.",
        f"The op `{S.OP_REF}` does not import here: its parity notes are not read, and "
        "every failure counts as unexplained.",
    ]
    # Where the op imports, no such sentence (R1-38 R1).
    monkeypatch.undo()
    assert "does not import here" not in make(synth, mode="fwd_bwd")


def toy_rows(failed="y_rel_err"):
    """The toy op's ``chunked`` failing parity in both modes at its default point,
    beside ``plain`` passing."""
    point = {"M": 16, "K": 4, "N": 4, "layout": "contiguous"}
    rows = []
    for mode in ("fwd", "fwd_bwd"):
        for name, status in (("plain", "ok"), ("chunked", "fail")):
            rows.append(
                {
                    "implementation": name,
                    "mode": mode,
                    "dtype": "float32",
                    "device": "cpu",
                    "env_name": "torch",
                    **point,
                    "status": "ok",
                    "median_ms": 1.0,
                    "parity_status": status,
                    "parity_failed": failed if status == "fail" else None,
                    "parity_y_rel_err": 1e-3,
                }
            )
    return point, rows


def test_parity_notes_of_the_toy_op():
    # toy_op's note on chunked applies in fwd_bwd only: the footnote is under the
    # fwd_bwd table, and the fwd failure stays unexplained.
    notes = toy_op.TOY.parity_notes
    point, rows = toy_rows()
    fwd = report.render_default(
        analysis.at(rows, mode="fwd"), point, "plain", title="f", notes=notes
    )
    assert "| fail: y_rel_err |" in fwd and "¹" not in fwd
    both = report.render_default(
        analysis.at(rows, mode="fwd_bwd"), point, "plain", title="b", notes=notes
    )
    assert "| fail: y_rel_err¹ |" in both
    assert f"¹ `chunked`, any metric: {notes[0].text}" in both.splitlines()
    text = report.render_parity_failures(rows, list(point), notes)
    assert f"- `chunked`, explained: {notes[0].text}\n  - y_rel_err (1)\n" in text
    assert "- `chunked`, **unexplained**\n  - y_rel_err (1)\n" in text
    assert "2 parity failures, 1 explained." in text
    assert "1 parity failure, 0 explained." in report.render_parity_failures(
        rows[:2], list(point), notes
    )


def test_footnotes_numbered_per_table():
    # Distinct notes numbered in the order first marked, one footnote each per table;
    # a cell whose metrics two notes explain carries both marks.
    point, rows = toy_rows("y_rel_err,grad_a_rel_l2")
    rows = analysis.at(rows, mode="fwd_bwd")
    rows.append({**rows[1], "implementation": "plain2", "parity_failed": "grad_a_rel_l2"})
    always = lambda *args: True  # noqa: E731
    notes = [
        ParityNote("chunked", ("y_rel_err",), always, "first"),
        ParityNote("chunked", ("grad_a_rel_l2",), always, "second"),
        ParityNote("plain2", None, always, "third"),
    ]
    text = report.render_default(rows, point, "plain", title="b", notes=notes)
    assert "| fail: y_rel_err¹, grad_a_rel_l2² |" in text
    assert "| fail: grad_a_rel_l2³ |" in text
    lines = text.splitlines()
    footnotes = [line for line in lines if line and line[0] in "¹²³"]
    assert footnotes == [
        "¹ `chunked`, y_rel_err: first",
        "² `chunked`, grad_a_rel_l2: second",
        "³ `plain2`, any metric: third",
    ]


def test_partly_explained_cell():
    # R1-38 R2: a record failing two metrics, one of them explained, carries the mark
    # on that metric only; "Parity failures" lists the other as unexplained.
    point, rows = toy_rows("y_rel_err,grad_a_rel_l2")
    rows = analysis.at(rows, mode="fwd_bwd")
    always = lambda *args: True  # noqa: E731
    notes = [ParityNote("chunked", ("y_rel_err",), always, "the cause of y")]
    text = report.render_default(rows, point, "plain", title="b", notes=notes)
    assert "| fail: y_rel_err¹, grad_a_rel_l2 |" in text
    assert "fail¹" not in text and "¹ `chunked`, y_rel_err: the cause of y" in text
    failures = report.render_parity_failures(rows, list(point), notes)
    assert failures.splitlines()[4:] == [
        "- `chunked`, explained: the cause of y",
        "  - y_rel_err (1)",
        "    - fwd_bwd, float32 M=16, K=4, N=4, layout=contiguous",
        "- `chunked`, **unexplained**",
        "  - grad_a_rel_l2 (1)",
        "    - fwd_bwd, float32 M=16, K=4, N=4, layout=contiguous",
        "",
        "1 parity failure, 0 explained.",
    ]
    # Without the notes: every metric unmarked.
    assert "| fail: y_rel_err, grad_a_rel_l2 |" in report.render_default(
        rows, point, "plain", title="b"
    )


def test_parity_failures_cite_a_note_once():
    # R1-38 R3: one note covering two metrics of an entry in two modes: its text once,
    # the (metric, case) groups under it; the count unchanged.
    point, rows = toy_rows("y_rel_err,grad_a_rel_l2")
    always = lambda *args: True  # noqa: E731
    note = ParityNote("chunked", None, always, "one long cause")
    text = report.render_parity_failures(rows, list(point), [note])
    assert text.count("one long cause") == 1
    assert text.splitlines()[4:] == [
        "- `chunked`, explained: one long cause",
        "  - grad_a_rel_l2 (2)",
        "    - fwd, float32 M=16, K=4, N=4, layout=contiguous",
        "    - fwd_bwd, float32 M=16, K=4, N=4, layout=contiguous",
        "  - y_rel_err (2)",
        "    - fwd, float32 M=16, K=4, N=4, layout=contiguous",
        "    - fwd_bwd, float32 M=16, K=4, N=4, layout=contiguous",
        "",
        "2 parity failures, 2 explained.",
    ]


def test_op_notes(synth, monkeypatch):
    # R1-42 N1, N2: the op's notes verbatim, last; none when it has none, or when the
    # op does not import here.
    assert "## Notes" not in make(synth)
    spec = analysis.spec_of({"op_ref": S.OP_REF})
    notes = ["2026-10-31: first, with `code` | a bar.", "2026-10-31: second."]
    monkeypatch.setattr(spec, "notes", notes)
    text = make(synth)
    part = text[text.index("## Notes") :]
    assert text.index("## Not measured") < text.index("## Notes")
    assert part.splitlines() == [
        "## Notes",
        "",
        "The notes of the op `synth`, as its author wrote them (`OpSpec.notes`); the "
        "report does not compute them.",
        "",
        f"- {notes[0]}",
        "",
        f"- {notes[1]}",
    ]

    def unresolvable(ref):
        raise ModuleNotFoundError(ref)

    monkeypatch.setattr(analysis, "_SPECS", {})
    monkeypatch.setattr("torch_op_bench.spec.resolve_op", unresolvable)
    assert "## Notes" not in make(synth)
    assert report.render_notes("x", []) == ""


def test_lce_notes():
    # Each LCE note dated and sourced by a run a reader can repeat, no path, probe or
    # review id a reader of the report cannot reach (R1-43 R1); the op's text.
    from torch_op_bench.ops import lce

    assert len(lce.LCE.notes) == 2
    for note in lce.LCE.notes:
        assert re.match(r"\d{4}-\d{2}-\d{2}: ", note) and "reproduced by `run lce --case" in note
        assert not re.search(r"\.asb|\.py\b|Review|R1-\d", note), note
    assert "`run lce --case N=8192 --impl torch_eager`" in lce.LCE.notes[0]
    # The paths and commands a reader follows, as they are (R1-45 F1).
    assert "`parity.metrics.grad_hidden_rel_l2`" in lce.LCE.notes[0]
    assert "with `--env` for the 2.14.1 environments" in lce.LCE.notes[1]
    assert "`ey`, torch 2.16.0.dev20261017" in lce.LCE.notes[0]
    assert "on one card" in lce.LCE.notes[1]


def flat_info(info, prefix=analysis.INFO_PREFIX):
    """``info`` as ``to_rows`` flattens it: nested keys joined by dots."""
    out = {}
    for key, value in info.items():
        if isinstance(value, dict):
            out.update(flat_info(value, f"{prefix}{key}."))
        else:
            out[prefix + key] = value
    return out


# The info of torch_lce at I1-19's default point, fwd_bwd (`ey`, 9ce07bc), less the
# fields every entry shares.
LCE_INFO = {
    "lce_source_files": ["_native/ops/linear_cross_entropy/__init__.py"],
    "lce_source_md5": "64bcb6b82ab92123fb60fff7a4cfdb97",
    "num_chunks": 32,
    "op_decisions": {
        "acc_dtype": "float32",
        "alloc_input_grad_acc_buf": False,
        "alloc_linear_bias_grad_chunk": False,
        "alloc_logits_acc_buf": False,
        "alloc_prob_target_buf": False,
        "alloc_weight_grad_chunk": False,
        "dtype": "bfloat16",
        "forward_uses_acc_input": False,
        "forward_uses_cuda_out_dtype": True,
        "grad_input_dtype": "bfloat16",
        "is_cuda": True,
        "is_prob_target": False,
        "linear_weight_cast_dtype": "bfloat16",
        "source": "torch.nn.modules.linear_cross_entropy._ChunkContext.build",
        "use_acc_dtype": True,
        "weight_chunk_dtype": "float32",
        "weight_grad_mm_same_dtype": True,
        "weight_grad_uses_logits_buf_temp": True,
    },
    "options": {
        "acc_dtype": "float32",
        "acc_policy": "compact",
        "batch_chunk_size": 128,
        "chunking_method": "aspect_ratio",
        "resolved": True,
    },
}


def test_entry_details_one_field_per_line():
    # I1-19 stumble 7: the torch LCE row's 26 fields, one per line in their cell; the
    # table stays two columns wide.
    rows = [
        {"implementation": "torch_eager"},
        {"implementation": "torch_lce", **flat_info(LCE_INFO)},
        {"implementation": "torch_compiled", **flat_info({"compile": "torch.compile"})},
    ]
    # As to_rows makes them: every row has every column.
    columns = dict.fromkeys(k for row in rows for k in row)
    rows = [{**dict.fromkeys(columns), **row} for row in rows]
    lines = report.entry_details(rows, rows)
    body = {cells(r)[0]: cells(r) for r in tables("\n".join(lines))[0][2:]}
    assert all(len(c) == 2 for c in body.values())
    pairs = body["torch_lce"][1].split("<br>")
    assert len(pairs) == 3 + 18 + 5  # lce_source_files is one field
    assert pairs[3] == "op_decisions.acc_dtype=float32"
    assert pairs[-1] == "options.resolved=True"
    assert all(", " not in pair or pair.startswith("lce_source_files") for pair in pairs)
    assert body["torch_compiled"][1] == "compile=torch.compile"
    assert body["torch_eager"][1] == "—"


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


def test_environment_check_status_mismatch():
    # R1-30 F2: a control out of memory where its original fits is the environment's
    # difference, not a pair to leave out.
    def row(name, n, status, **more):
        found = {"implementation": name, "mode": "fwd_bwd", "dtype": "bf16", "N": n}
        found.update(status=status, median_ms=1.0, peak_above_baseline=0, **more)
        if status != "ok":
            found.update(median_ms=None, peak_above_baseline=None)
        return found

    control = {"role": "control", "control_of": "a", "env_name": "x"}
    rows = [
        row("a", 1, "ok"),
        row("c", 1, "ok", **control),
        row("a", 2, "ok"),
        row("c", 2, "error", reason=OOM.format(tried="4.00", free="2.22"), **control),
        row("a", 3, "error", reason="RuntimeError: boom"),
        row("c", 3, "ok", **control),
        row("a", 4, "unavailable", reason="needs 2 GPUs"),
        row("c", 4, "error", reason="RuntimeError: boom", **control),
    ]
    checks = report.environment_check(rows, ["N"])
    assert [(c["measured"], c["flags"]) for c in checks] == [
        ("both", []),
        ("one", ["status"]),
        ("one", ["status"]),
        ("neither", []),
    ]
    text = report.render_environment_check(rows, ["N"])
    assert "1 pair compared; 2 with one side not measured (flagged below); 1 with neither" in text
    body = [cells(r) for r in table_rows(text)[2:]]
    assert [r[4] for r in body] == ["N=1", "N=2", "N=3"]  # neither side: no row
    assert body[1][5:] == ["error / ok", "—", "status"]
    assert body[2][5:] == ["ok / error", "—", "status"]
    assert (
        "- **c** (`x`) against a, fwd_bwd bf16 at N=2: control `error` (OutOfMemoryError: "
        "CUDA out of memory.), original `ok`: the environment makes a difference there."
    ) in text
    assert "control `ok`, original `error` (RuntimeError: boom)" in text
    assert "Every control is within the limits" not in text
    # Only the pair of both ok: within the limits.
    text = report.render_environment_check(rows[:2], ["N"])
    assert "1 pair compared; 0 with one side not measured; 0 with neither" in text
    assert "Every control is within the limits" in text
    # Neither side measured anywhere: counted, no table.
    text = report.render_environment_check(rows[6:], ["N"])
    assert "0 pairs compared; 0 with one side not measured; 1 with neither" in text
    assert "| control |" not in text and "Every control" not in text


def test_not_measured(synth):
    text = section(make(synth), "## Not measured")
    assert (
        "- error: RuntimeError: boom (1)\n  - compiled, fwd, float32 M=64, K=32, N=16, layout=a\n"
        in text
    )
    assert "second line" not in text  # the reason's first line only
    assert "- unavailable: needs a GPU (1)\n  - chunked, fwd_bwd, float32 M=128, K=16" in text
    rows = [{"status": "ok"}]
    assert "Every record is `ok`." in report.render_not_measured(rows, [])


# The reasons of I1-12's six out-of-memory errors, cut where the figures end.
OOM = (
    "OutOfMemoryError: CUDA out of memory. Tried to allocate {tried} GiB. GPU 0 has a "
    "total capacity of 15.57 GiB of which {free} GiB is free. Including non-PyTorch "
    "memory, this process has 13.28 GiB memory in use."
)


def test_cause():
    # One cause, whatever its figures (R1-30 F5); the figures go to the row.
    key = "OutOfMemoryError: CUDA out of memory."
    assert report.cause("error", OOM.format(tried="4.00", free="2.28")) == (
        key,
        "tried to allocate 4.00 GiB, 2.28 GiB free",
    )
    assert report.cause("error", OOM.format(tried="8.00", free="1.31"))[0] == key
    # Two different RuntimeErrors stay two causes.
    shape = report.cause("error", "RuntimeError: shape mismatch: 3 against 4")
    illegal = report.cause("error", "RuntimeError: CUDA error: an illegal memory access")
    assert shape == ("RuntimeError: shape mismatch", "3 against 4")
    assert illegal == ("RuntimeError: CUDA error: an illegal memory access", "")
    # Digits in a word name the cause; a number token ends it (R1-31 F1): no two of
    # these six causes share a group.
    lines = [
        "RuntimeError: mat1 and mat2 shapes cannot be multiplied (4096x1024 and 2048x131072)",
        "RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float",
        "RuntimeError: Expected 2D tensor for input, got 3D",
        "RuntimeError: Expected 3 arguments, got 2",
        "ValueError: sm_80 or newer is required",
        "ValueError: sm_90 kernels are not built for this wheel",
    ]
    keys = [report.cause("error", line)[0] for line in lines]
    assert len(set(keys)) == 6
    assert keys[0] == "RuntimeError: mat1 and mat2 shapes cannot be multiplied"
    assert keys[1:3] + keys[4:] == lines[1:3] + lines[4:]  # no number: the whole line
    assert keys[3] == "RuntimeError: Expected"
    # ... and other shapes of the same cause stay in its group.
    other = "RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x16 and 32x64)"
    assert report.cause("error", other) == (keys[0], "(8x16 and 32x64)")
    assert report.cause("error", "TimeoutError: worker exceeded 600.0 s") == (
        "TimeoutError: worker exceeded",
        "600.0 s",
    )
    # A message that starts with a number has nothing to cut at: the whole line.
    assert report.cause("error", "ValueError: 3 is odd")[0] == "ValueError: 3 is odd"
    # unavailable: the first line, numbers and all; the second is what it leaves out.
    assert report.cause("unavailable", "needs 2 GPUs") == ("needs 2 GPUs", "")
    assert report.cause("unavailable", "needs 2 GPUs\nmore") == ("needs 2 GPUs", "more")
    assert report.cause("error", None) == ("no reason recorded", "")


def test_not_measured_groups_by_cause():
    figures = [("4.00", "2.28"), ("4.00", "2.22"), ("4.00", "2.22"), ("8.00", "2.23")]
    rows = [
        {
            "implementation": f"e{i}",
            "mode": "fwd_bwd",
            "dtype": "bfloat16",
            "N": 16384,
            "status": "error",
            "reason": OOM.format(tried=tried, free=free),
        }
        for i, (tried, free) in enumerate(figures)
    ]
    rows.append({**rows[0], "implementation": "x", "reason": "RuntimeError: boom 7"})
    text = report.render_not_measured(rows, ["N"])
    assert text.count("\n- ") == 2
    assert "- error: OutOfMemoryError: CUDA out of memory. (4)\n" in text
    assert "  - e3, fwd_bwd, bfloat16 N=16384: tried to allocate 8.00 GiB, 2.23 GiB free\n" in text
    assert "- error: RuntimeError: boom (1)\n  - x, fwd_bwd, bfloat16 N=16384: 7\n" in text


def test_not_measured_out_of_memory_unavailable():
    # Task I1-16: points that do not fit on the device are one group, by the reason's
    # first line; each row has its figures from the second.
    device = "out of memory on NVIDIA GeForce RTX 5060 Ti (15.6 GiB)"
    figures = [
        "tried to allocate 4.00 GiB, 2.28 GiB free; 9.13 GiB allocated, 3.97 GiB reserved "
        "but unallocated; phase: parity",
        "tried to allocate 8.00 GiB, 2.23 GiB free; 13.16 GiB allocated, 1.87 MiB reserved "
        "but unallocated; phase: timing",
    ]
    rows = [
        {
            "implementation": "torchtitan_ce",
            "mode": mode,
            "dtype": "bfloat16",
            "N": 16384,
            "status": "unavailable",
            "reason": f"{device}\n{line}",
        }
        for mode, line in zip(("fwd", "fwd_bwd"), figures, strict=True)
    ]
    text = report.render_not_measured(rows, ["N"])
    assert text.count("\n- ") == 1
    assert f"- unavailable: {device} (2)\n" in text
    assert f"  - torchtitan_ce, fwd, bfloat16 N=16384: {figures[0]}\n" in text
    assert f"  - torchtitan_ce, fwd_bwd, bfloat16 N=16384: {figures[1]}\n" in text


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
    header = cells(tables(table)[0][0])
    by_name = {cells(row)[0]: cells(row) for row in tables(table)[0][2:]}
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
    assert base["grad_rel default, strict"] == "0.00200, 0.00100"
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


# --- time at equal peak (Task I1-7) ------------------------------------------------------


def test_equal_peak_section(synth):
    S.write_variants(synth)
    text = section(make(synth), "## Time at equal peak")
    assert (
        "The budgets are the baseline `base`'s peak above baseline there, at 1×, 0.75×, 0.5×"
        in text
    )
    assert "no value of a parameter is ranked here" in text
    # I1-19 stumble 2: an empty cell is measured, not missing.
    assert "— when none is (measured; no point within the budget), with the entry's" in text
    header = (
        "| implementation | ≤ 1× (64.0 MiB) | ≤ 0.75× (48.0 MiB) | ≤ 0.5× (32.0 MiB) "
        "| default: median ms, peak MiB |"
    )
    part = section(text, "### fwd_bwd, bfloat16", "\n### ")
    assert tables(part)[0] == [
        header,
        "| :--- | ---: | ---: | ---: | ---: |",
        # compiled: its 64 MiB default fits 1x exactly; nothing fits below, and the
        # cells show its lowest peak, a variant's (R1-39 F1); its 80 MiB variant is
        # unavailable here.
        "| compiled | 32.0 <sub>default</sub> | — <sub>lowest 56.0 MiB</sub> "
        "| — <sub>lowest 56.0 MiB</sub> | 32.0 ms, 64.0 MiB |",
        # chunked: the 40 MiB variant (failing parity) within 1x and 0.75x; its 16 MiB
        # default within 0.5x; the 8 MiB variant is never the largest.
        "| chunked | 102 <sub>num_chunks=2, parity fail</sub> "
        "| 102 <sub>num_chunks=2, parity fail</sub> | 128 <sub>default (num_chunks=4)</sub> "
        "| 128 ms, 16.0 MiB |",
    ]
    assert "Not `ok` here, so in no cell: `compiled[mode=b]` (unavailable)." in part
    part = section(text, "### fwd_bwd, float32", "\n### ")
    assert tables(part)[0][2:] == [
        "| compiled | 64.0 <sub>default</sub> | — <sub>lowest 56.0 MiB</sub> "
        "| — <sub>lowest 56.0 MiB</sub> | 64.0 ms, 64.0 MiB |",
        "| chunked | 205 <sub>num_chunks=2</sub> | 205 <sub>num_chunks=2</sub> "
        "| 256 <sub>default (num_chunks=4)</sub> | 256 ms, 16.0 MiB |",
    ]
    assert "Not `ok`" not in part
    # fwd: chunked runs fwd_bwd only, so only compiled has a row.
    assert [cells(r)[0] for r in tables(section(text, "### fwd, float32", "\n### "))[0][2:]] == [
        "compiled"
    ]
    assert [line for line in text.splitlines() if line.startswith("### ")] == [
        "### fwd, float32",
        "### fwd_bwd, float32",
        "### fwd, bfloat16",
        "### fwd_bwd, bfloat16",
    ]


def test_equal_peak_same_point_note():
    # I1-19 stumble 3: a row whose budgets all pick one point is named under the
    # table; a row with no point, or one budget, is not.
    a, b = {"median_ms": 1.0}, {"median_ms": 2.0}
    same = {"base": "same", "cells": [a, a, a]}
    moves = {"base": "moves", "cells": [b, a, a]}
    empty = {"base": "empty", "cells": [None, None, None]}
    assert report.same_point_note([moves, empty]) == []
    assert report.same_point_note([{"base": "one", "cells": [a]}]) == []
    assert report.same_point_note([{"base": "none", "cells": None}]) == []
    tail = (
        ", every budget admits the same point: none of the entry's points has its peak "
        "between the smallest budget and the largest, so the budgets do not separate "
        "them here."
    )
    assert report.same_point_note([same, same]) == ["", "In every row" + tail]
    assert report.same_point_note([same, moves]) == ["", "In the row of `same`" + tail]
    other = {**same, "base": "other"}
    assert report.same_point_note([same, empty, other]) == [
        "",
        "In the rows of `same`, `other`" + tail,
    ]


def test_equal_peak_without_a_baseline(synth):
    S.write_variants(synth)
    text = section(
        report.report(synth, baseline="nope", dtype="float32", mode="fwd_bwd"),
        "## Time at equal peak",
    )
    assert "The baseline `nope` has no `ok` record with a peak here: no budgets." in text
    assert tables(text)[0] == [
        "| implementation | default: median ms, peak MiB |",
        "| :--- | ---: |",
        "| compiled | 64.0 ms, 64.0 MiB |",
        "| chunked | 256 ms, 16.0 MiB |",
    ]


def test_variants_change_nothing_else(synth, synth_strict):
    # The variants are in "Time at equal peak", "Parity failures" and "Not measured"
    # only: every other table reads as without them (no row, no column, no info).
    before = make(synth, strict=synth_strict)
    assert "## Time at equal peak" not in before
    S.write_variants(synth)
    after = make(synth, strict=synth_strict)
    for title in (
        "## At the default point",
        "## Sweeps",
        "## Reduction setting",
        "## Environment check",
    ):
        assert section(after, title) == section(before, title), title
    assert "[" not in section(after, "## At the default point")
    assert (
        section(after, "## Sweeps", "## Time at equal peak").rstrip()
        == section(before, "## Sweeps").rstrip()
    )
    failures = section(after, "## Parity failures")
    assert f"- `chunked[num_chunks=2]`, explained: {synth_op.NOTE.text}\n  - grad_rel (1)\n" in (
        failures
    )
    assert (
        "3 parity failures, 2 explained." in failures
        and "2 parity failures, 1 explained." in before
    )
    missing = section(after, "## Not measured")
    assert "- unavailable: out of memory on cuda:0 (16.0 GiB) (1)" in missing
    assert (
        "  - compiled[mode=b], fwd_bwd, bfloat16 M=128, K=32, N=16, layout=a: tried 1.0 GiB"
        in missing
    )


def test_within_and_default_params(synth):
    def row(peak, ms):
        return {"peak_above_baseline": peak, "median_ms": ms}

    points = [row(10, 5.0), row(20, 4.0), row(20, 3.0), row(40, 2.0)]
    assert analysis.within(points, 20) is points[2]  # the largest peak within; the faster
    assert analysis.within(points, 39) is points[2]
    assert analysis.within(points, 40) is points[3]
    assert analysis.within(points, 9) is None
    S.write_variants(synth)
    with pytest.warns(UserWarning):
        recs = analysis.load(synth)
    bases, variants = analysis.split_variants(recs)
    assert {r["implementation"] for r in variants} == set(S.VARIANTS)
    assert not any("[" in r["implementation"] for r in bases)
    # chunked records its num_chunks; compiled has no field named mode.
    assert analysis.default_params(bases, variants) == {"chunked": "num_chunks=4"}
    row = analysis.flatten(variants[0])
    assert (row["base"], row["params"]) == ("chunked", "num_chunks=2")
    assert analysis.flatten(bases[0])["base"] is None
    assert not any(k.startswith("params.") or k == "base" for k in analysis.varying_info(recs))


# --- the summary and the definitions (Task I1-23) ------------------------------------------


def headers(text):
    """The header cells of every table of ``text``."""
    return [h for lines in tables(text) for h in cells(lines[0])]


def entry_labels(text):
    """The entries' labels as the report's tables write them."""
    names = set()
    for lines in tables(text):
        if cells(lines[0])[0] == "implementation":
            names.update(cells(line)[0] for line in lines[2:])
    return names


def assert_every_header_defined(text, results_dir):
    """Every header of every table maps to a definition the report shows."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recs = analysis.load(results_dir)
    spec = analysis.spec_of(recs[0])
    rows = analysis.to_rows(recs)
    rates = [label for label, _, _ in report.throughput_columns(spec and spec.work_units, rows)]
    metrics = {
        k.removeprefix(analysis.PARITY_PREFIX)
        for k in rows[0]
        if k.startswith(analysis.PARITY_PREFIX) and k not in analysis.RESULTS
    }
    axes = analysis.axes_of(recs)
    how = section(text, report.READ_TITLE)
    for header in set(headers(text)):
        term = report.header_term(
            header, rates=rates, metrics=metrics, axes=axes, entries=entry_labels(text)
        )
        assert term is not None, header
        if term == "implementation" or (term == "axis" and spec is None):
            continue  # an entry is described in the Summary; an axis by the op only
        if header in [*rates, *metrics, *axes]:
            if spec is not None:
                assert f"**{header}**" in how, header  # the op's term, under its name
            continue
        assert f"- **{term}**: " in how, (header, term)


def test_definitions_cover_every_header(synth, synth_strict, tmp_path):
    # Part A: every header the report emits has a definition (so a new column cannot
    # appear undefined), on the synthetic set with its strict run and variants.
    S.write_variants(synth)
    text = make(synth, strict=synth_strict, command=["torch-op-bench", "report", str(synth)])
    assert report.header_term("a new column") is None
    assert len(set(headers(text))) > 30
    assert_every_header_defined(text, synth)
    how = section(text, report.READ_TITLE)
    terms = [term for term, _ in report.DEFINITIONS]
    assert [line.split("**")[1] for line in how.splitlines() if line.startswith("- **")][
        : len(terms)
    ] == terms
    # The settings as recorded, the baseline named, the bands.
    assert "of 3 timed calls, in milliseconds, after 1 untimed calls" in how
    assert "the baseline `base`'s" in how and "[0.95, 1.05]" in how and "64 MiB" in how
    # The op's words: its units, metrics with tolerances, axes with descriptions.
    assert "  - **TFLOP/s**: FLOP per second, in units of 1e+12; FLOP: 2·M·K·N." in how
    assert "  - **grad_rel**: the planted gradient error; tolerance 0.01." in how
    assert "  The tolerances are planted." in how
    assert "  - **layout**: the layout; values a, b; default a." in how
    # Order: summary, definitions, provenance; the axes under the default point.
    assert text.index("## Summary") < text.index(report.READ_TITLE) < text.index("## Provenance")
    assert (
        "There: M=128 (the rows); K=32 (the inner size); N=16 (the columns); layout=a (the layout)."
        in text
    )


def test_throughput_columns_from_work_units(synth, monkeypatch):
    # Part A: the throughput columns are the op's declared units, never the report's.
    def default_header(text):
        return cells(tables(section(text, "### fwd_bwd, float32", "\n### "))[0][0])

    assert default_header(make(synth))[4:6] == ["tokens/s", "TFLOP/s"]
    spec = analysis.spec_of({"op_ref": S.OP_REF})
    monkeypatch.setattr(spec, "work_units", [])
    header = default_header(make(synth))
    assert "tokens/s" not in header and "TFLOP/s" not in header
    assert header[3:5] == ["time / baseline", "peak above baseline, MiB"]
    assert "- **Units of work**: none declared, so no throughput column." in make(synth)
    # The toy op's one unit, with its own label and scale.
    monkeypatch.setattr(spec, "work_units", toy_op.TOY.work_units)
    text = make(synth)
    header = default_header(text)
    assert header[4] == "toy MFLOP/s" and "tokens/s" not in header
    part = section(text, "### fwd_bwd, float32", "\n### ")
    row = next(cells(r) for r in tables(part)[0][2:] if cells(r)[0] == "chunked")
    t = S.median_ms("chunked", "fwd_bwd", "float32", S.DEFAULT) / 1e3
    assert row[4] == report.sig3(2 * 128 * 32 * 16 / t / 1e6)
    assert report.throughput_columns(toy_op.TOY.work_units, []) == [
        ("toy MFLOP/s", "flops_per_s", 1e6)
    ]


def plant_skipped(synth):
    """``compiled`` fwd_bwd float32 at the default point, its parity skipped (a later
    record of the key, which merge keeps)."""
    from torch_op_bench import records

    with pytest.warns(UserWarning, match=MERGE_WARNING):
        recs = analysis.load(synth)
    (record,) = [
        r
        for r in recs
        if (r["implementation"], r["mode"], r["case"]["dtype"])
        == ("compiled", "fwd_bwd", "float32")
        and r["case"]["params"] == S.DEFAULT
    ]
    record = {k: v for k, v in record.items() if k != analysis.SOURCE}
    record["timestamp"] = "2026-10-18T12:00:00+00:00"
    reason = "reference does not fit: 13.8 GiB needed, 11.3 GiB free (80 % usable)"
    record["parity"] = {"status": "skipped", "reason": reason, "metrics": {}, "failed": []}
    records.append(synth / "synth" / S.SLUG / "20261018T120000Z-host-a.jsonl", record)


def test_summary_generated(synth):
    # Part B: what was compared, on what, the generated findings and the caveats, on
    # the synthetic set with a planted skip, error, unavailable and parity failures.
    plant_skipped(synth)
    text = make(synth)
    summary = section(text, "## Summary")
    lines = summary.splitlines()
    assert "The op `synth`: a synthetic product, never run." in lines
    assert [line for line in lines if line.startswith("- `")][:4] == [
        "- `base` (the baseline, in `torch`): the plain product.",
        "- `base_other` (a control, in `other`): not described.",
        "- `compiled` (in `torch`): not described.",
        "- `chunked` (in `other`): the product in chunks.",
    ]
    assert (
        "Modes: forward only (`fwd`) and forward and backward (`fwd_bwd`); dtype float32, bfloat16."
        in lines
    )
    assert (
        "The sweep varies one axis at a time around the default point M=128, K=32, N=16, "
        "layout=a: M (the rows): 64, 128, 256; K (the inner size): 16, 32; N (the columns): "
        "8, 16; layout (the layout): a, b. That is 6 points and "
    ) in summary
    on_what = section(summary, "### On what", "\n### ")
    assert "- GPU: not recorded" in on_what and "- host: host-a, host-b" in on_what
    assert "- torch: 2.14.1+cu130 in `other`; 2.16.0a0+gitabc in `torch`" in on_what
    assert "- packages: counterpart=0123456789ab, helper=1.0 in `other`" in on_what
    assert "- date: 2026-10-17 12:00 to 2026-10-18 12:00 UTC" in on_what
    findings = section(summary, "### Findings", "\n### ")
    assert report.FINDINGS_GENERATED in findings
    # Exact sentences for planted numbers: compiled at half the baseline's time.
    assert (
        "**Forward and backward (`fwd_bwd`), float32, at the default point.** The baseline "
        f"`base` takes {report.sig3(S.median_ms('base', 'fwd_bwd', 'float32', S.DEFAULT))} ms "
        f"and holds {report.mib(S.peak_above('base', S.DEFAULT))} MiB above baseline; parity ok."
    ) in findings
    assert (
        "- `compiled` takes 0.500× the baseline's time and 1.00× its memory above baseline; "
        "time × peak 0.500× the baseline's; parity not checked (reference does not fit)."
    ) in findings
    assert (
        "- `chunked` (the product in chunks) takes 2.00× the baseline's time and 0.250× its "
        "memory above baseline; time × peak 0.500× the baseline's; parity ok."
    ) in findings
    # The planted control: 10 % slower at the default point, outside the time band.
    assert (
        "- The controls (`base_other`: `base` run inside the `other` environment): outside "
        "the environment check's bands here, `base_other`: time."
    ) in findings
    assert "**Over M (`fwd_bwd`, float32).** The baseline's median goes from " in findings
    assert (
        "- `compiled`: time / baseline within ±10 % of its 0.500× at the default point" in findings
    )
    caveats = section(summary, "### Caveats", "\n### ")
    assert (
        "- 1 record failed with an error nobody foresaw (error: RuntimeError: boom): "
        "`compiled` (fwd) at M=64."
    ) in caveats
    assert (
        "- 1 record could not run here (unavailable: needs a GPU): `chunked` (fwd_bwd) at K=16."
    ) in caveats
    assert (
        "- Parity (the check against the op's reference) was not run in 1 record, as the "
        "reference does not fit: at the default point (1)."
    ) in caveats
    found = caveats.splitlines()
    at = found.index("- 2 parity failures, 1 explained:")
    assert found[at + 1 : at + 3] == [
        f"  - `chunked` on grad_rel (1), explained: {synth_op.NOTE.text}",
        "  - `chunked` on grad_rel (1), **unexplained**",
    ]
    assert "### Findings (written" not in text


def test_summary_failure_sentences(synth):
    # At M=256 chunked fails parity: explained in bfloat16, not in float32.
    text = make(synth, point={"M": "256"}, mode="fwd_bwd")
    findings = section(text, "### Findings", "\n### ")
    assert (
        "- `chunked` (the product in chunks) takes 2.00× the baseline's time and 0.250× its "
        "memory above baseline; time × peak 0.500× the baseline's; parity fails on "
        "grad_rel, unexplained (see Parity failures)."
    ) in findings
    assert "parity fails on grad_rel, explained (see Parity failures)." in findings
    # The planted control's 100 MiB at M=256.
    assert "outside the environment check's bands here, `base_other`: peak." in findings


def test_summary_forbidden_words(synth, synth_strict):
    S.write_variants(synth)
    text = make(synth, strict=synth_strict)
    words = set(re.findall(r"[a-z]+", section(text, "## Summary").lower()))
    assert not words & set(report.FORBIDDEN)
    assert report.FORBIDDEN == ("fastest", "slowest", "best", "worst", "wins", "optimum", "beats")


def test_summary_equal_peak_sentences(synth):
    # Planted: budgets 64, 48, 32 MiB; chunked's variant at 40 MiB fails parity in
    # bfloat16; compiled's lowest point 56 MiB misses the 0.75x and 0.5x budgets.
    S.write_variants(synth)
    text = make(synth, mode="fwd_bwd", dtype="float32")
    findings = section(text, "### Findings", "\n### ")
    t = S.median_ms("chunked", "fwd_bwd", "float32", S.DEFAULT)
    c = S.median_ms("compiled", "fwd_bwd", "float32", S.DEFAULT)
    assert (
        "**Time at equal peak (`fwd_bwd`).** The budgets are the baseline's peak above "
        "baseline, 64.0 MiB, at 1×, 0.75×, 0.5×."
    ) in findings
    assert (
        f"- `compiled`: within 1× (64.0 MiB), {report.sig3(c)} ms with its default; no point "
        "within 0.75× (48.0 MiB) (lowest 56.0 MiB); no point within 0.5× (32.0 MiB) (lowest "
        "56.0 MiB)."
    ) in findings
    assert (
        f"- `chunked`: within 1× (64.0 MiB), {report.sig3(0.8 * t)} ms with num_chunks=2; "
        f"within 0.75× (48.0 MiB), {report.sig3(0.8 * t)} ms with num_chunks=2; within 0.5× "
        f"(32.0 MiB), {report.sig3(t)} ms with its default (num_chunks=4)."
    ) in findings
    text = make(synth, mode="fwd_bwd", dtype="bfloat16")
    assert "ms with num_chunks=2, which fails parity" in text


def test_summary_findings_file(synth, tmp_path, capsys):
    # With --summary, the author's paragraphs in place of the generated sentences;
    # the headline table stays.
    findings = tmp_path / "findings-x.md"
    findings.write_text("PLACEHOLDER: one.\n\nPLACEHOLDER: two, with `code`.\n")
    text = make(synth, findings=findings)
    part = section(text, "### Findings (written", "\n### Caveats")
    assert part.splitlines()[:7] == [
        report.FINDINGS_AUTHOR,
        "",
        "From `findings-x.md`, for the records of torch-op-bench aaaaaaaaaaaa, bbbbbbbbbbbb.",
        "",
        "PLACEHOLDER: one.",
        "",
        "PLACEHOLDER: two, with `code`.",
    ]
    assert report.FINDINGS_GENERATED not in text and "takes 0.500×" not in text
    assert "#### Headline table" in part
    assert cli.main(["report", str(synth), "--baseline", "base", "--summary", str(findings)]) == 0
    assert "PLACEHOLDER: two" in capsys.readouterr().out


def test_summary_without_the_op(synth, monkeypatch):
    # The op does not import: the summary from the records alone, and what it cannot show.
    def unresolvable(ref):
        raise ModuleNotFoundError(ref)

    monkeypatch.setattr(analysis, "_SPECS", {})
    monkeypatch.setattr("torch_op_bench.spec.resolve_op", unresolvable)
    text = make(synth)
    summary = section(text, "## Summary")
    assert "The op `synth` (not described here: the op does not import)." in summary
    assert "- `base` (the baseline, in `torch`): not described." in summary
    assert (
        f"- The op `{S.OP_REF}` does not import here: its descriptions, definitions, parity "
        "notes and notes are not shown, and every parity failure counts as unexplained."
    ) in summary
    how = section(text, report.READ_TITLE)
    assert (
        f"The op `{S.OP_REF}` does not import here: its definitions are not shown; the "
        "throughput columns (flops/s, tokens/s) are the records' work over the median, in "
        "units the op defines."
    ) in how
    header = cells(tables(section(text, "### fwd_bwd, float32", "\n### "))[0][0])
    assert header[4:6] == ["flops/s", "tokens/s"]
    assert "There: " not in text
    assert_every_header_defined(text, synth)


def test_headline_table_equals_the_default_point(synth):
    S.write_variants(synth)
    text = make(synth)
    head = tables(section(text, "#### Headline table", "\n### "))[0]
    assert cells(head[0]) == [
        "implementation",
        "mode",
        "dtype",
        "time / baseline",
        "peak above baseline / baseline's",
        "time × peak / baseline's",
        "≤ 1× budget: median ms",
    ]
    rows = [cells(r) for r in head[2:]]
    assert len(rows) == 2 * (3 + 4)  # fwd: base, control, compiled; fwd_bwd: + chunked
    for name, mode, dtype, *ratios, cell in rows:
        part = section(text, f"### {mode}, {dtype}\n\nAt M=128", "\n### ")
        default = {cells(r)[0]: cells(r) for r in tables(part)[0][2:]}[name]
        assert ratios == [default[3], default[7], default[9]], (name, mode, dtype)
        peak = section(section(text, report.EQUAL_PEAK_TITLE), f"### {mode}, {dtype}", "\n### ")
        equal = {cells(r)[0]: cells(r) for r in tables(peak)[0][2:]}
        assert cell == (equal[name][1] if name in equal else "—"), (name, mode, dtype)
        assert (name in equal) == (name in ("chunked", "compiled"))


def lce_set(root):
    """Records in the structure of the `ey` set (I1-19): the real `lce` op, every base
    entry in each of its modes at the default point and at N=1024, bfloat16, and the
    variants of the chunked entries at the default point; numbers planted."""
    from torch_op_bench import records, worker
    from torch_op_bench.ops import lce
    from torch_op_bench.spec import Case, to_json

    spec = lce.LCE
    path = root / "lce" / "gpu" / "20261101T000000Z-ey.jsonl"
    points = [dict(spec.default_point), {**spec.default_point, "N": 1024}]
    for i, impl in enumerate(spec.implementations):
        for mode in [m for m in ("fwd", "fwd_bwd") if m in impl.modes]:
            for params in points[:1] if impl.variant_of else points:
                case = Case(params, "bfloat16", "cuda:0")
                payload = {
                    "op": "lce",
                    "op_name": "lce",
                    "case": case.to_dict(),
                    "implementation": impl.name,
                    "mode": mode,
                    "settings": {"warmup": 10, "repeat": 50, "flush_l2": False},
                    "env_name": impl.env_name,
                    "device_slug": "gpu",
                    "work": spec.work(case, mode),
                    "info": to_json(impl.info),
                }
                record = worker.base_record(payload, "/envs/x/bin/python")
                t = (100 + 7 * i) * (3 if mode == "fwd_bwd" else 1) * params["N"] / 4096
                metrics = {"loss_abs_err": 1e-2, "loss_rel_err": 2e-3}
                if mode == "fwd_bwd":
                    metrics.update(grad_hidden_rel_l2=2e-3, grad_weight_rel_l2=1e-3)
                record.update(
                    timestamp="2026-11-01T00:00:00+00:00",
                    status="ok",
                    times_ms=[t, t, t],
                    memory={"baseline_bytes": 2**30, "peak_bytes": 2**30 + (64 + 32 * i) * 2**20},
                    parity={"status": "ok", "failed": [], "metrics": metrics, "reason": None},
                    env={
                        "hostname": "ey",
                        "torch": {
                            "version": "2.16.0",
                            "devices": [{"index": 0, "name": "GPU", "total_memory": 16 * 2**30}],
                        },
                        "torch_op_bench": {"hash": "c" * 40},
                    },
                )
                records.append(path, record)
    return root


def test_generated_findings_on_the_lce_structure(tmp_path):
    # Part B on the structure of the `ey` set: every entry and mode gets its sentence,
    # the controls theirs, no forbidden word; every header defined (Part A).
    from torch_op_bench.ops import lce

    root = lce_set(tmp_path / "results")
    text = report.report(root)
    summary = section(text, "## Summary")
    findings = section(summary, "### Findings", "\n### ")
    for mode in ("fwd", "fwd_bwd"):
        part = section(findings, f"(`{mode}`), at the default point.**", "\n**")
        named = re.findall(r"^- `([^`]+)`", part, flags=re.M)
        expected = [
            impl.name
            for impl in lce.LCE.implementations
            if mode in impl.modes
            and impl.variant_of is None
            and impl.name != "torch_eager"
            and impl.info.get("role") != "control"
        ]
        assert named == expected, mode
        assert "- The controls (`torch_eager_liger` and `torch_eager_torchtitan`: " in part
    assert "- `liger` (Liger-Kernel's fused kernel, accumulating in fp32) takes " in findings
    assert "**Time at equal peak (`fwd_bwd`).**" in findings and "**Over N (`fwd`).**" in findings
    assert not set(re.findall(r"[a-z]+", summary.lower())) & set(report.FORBIDDEN)
    assert "- GPU: GPU, 16.0 GiB" in summary
    assert_every_header_defined(text, root)
    how = section(text, report.READ_TITLE)
    assert "2·N·H·V in `fwd` (the logits), 6·N·H·V in `fwd_bwd`" in how
    assert "  - **N**: rows of hidden, the tokens of a batch; values " in how
    # The throughput as before: TFLOP/s is the row's tflops, now a third in fwd.
    part = section(text, "### fwd, bfloat16", "\n### ")
    row = next(cells(r) for r in tables(part)[0][2:] if cells(r)[0] == "torch_eager")
    flops = 2 * 4096 * 4096 * 131072
    assert row[4:6] == [report.sig3(4096 / 0.1), report.sig3(flops / 0.1 / 1e12)]
