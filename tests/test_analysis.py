import json
import statistics

import pytest
import synth_results as S

from torch_op_bench import analysis, records

MERGE_WARNING = "different torch-op-bench commits"


def load(root, **kwargs):
    with pytest.warns(UserWarning, match=MERGE_WARNING):
        return analysis.load(root, **kwargs)


def find(rows, **where):
    (row,) = analysis.at(rows, **where)
    return row


def test_load_merges_and_warns(synth):
    recs = load(synth)
    # 6 points x 2 dtypes x 7 (implementation, mode) pairs, one missing; the second
    # host's stale duplicate dropped.
    assert len(recs) == 6 * 2 * 7 - 1
    assert {r[analysis.SOURCE] for r in recs} == {
        "20261017T120000Z-host-a.jsonl",
        "20261016T120000Z-host-b.jsonl",
    }
    rows = analysis.to_rows(recs)
    base = find(
        analysis.at(rows, **S.DEFAULT), implementation="base", mode="fwd_bwd", dtype="float32"
    )
    assert base["median_ms"] == S.median_ms("base", "fwd_bwd", "float32", S.DEFAULT)
    assert base["host"] == "host-a"
    assert len(load(synth, op="synth", device_slug=S.SLUG)) == len(recs)
    assert len(load(synth, op="synth")) == len(recs)
    with pytest.raises(FileNotFoundError, match="no results at"):
        analysis.load(synth, op="nope")


def test_flatten_columns(synth):
    rows = analysis.to_rows(load(synth))
    assert list(rows[0]) == [
        *analysis.HEAD,
        "M",
        "K",
        "N",
        "layout",  # the op's axis order, not sorted
        *analysis.RESULTS,
        "parity_grad_rel",
        "parity_loss_rel_err",
        *analysis.TAIL,
        "info.api.name",
        "info.compile",
        "info.num_chunks",
    ]
    assert all(list(row) == list(rows[0]) for row in rows)


def test_flatten_values(synth):
    rows = analysis.to_rows(load(synth))
    params = {**S.DEFAULT, "M": 256}
    row = find(analysis.at(rows, **params), implementation="chunked", dtype="float32")
    t = S.median_ms("chunked", "fwd_bwd", "float32", params)
    samples = [0.9 * t, t, 1.1 * t]
    q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive")
    expected = {
        "op": "synth",
        "device_slug": S.SLUG,
        "host": "host-a",
        "timestamp": "2026-10-17T12:00:00+00:00",
        "mode": "fwd_bwd",
        "env_name": "other",
        "status": "ok",
        "reason": None,
        "n": 3,
        "median_ms": t,
        "min_ms": 0.9 * t,
        "iqr_ms": q3 - q1,
        "peak_bytes": S.BASELINE_BYTES + S.peak_above("chunked", params),
        "baseline_bytes": S.BASELINE_BYTES,
        "peak_above_baseline": S.peak_above("chunked", params),
        "tokens": 256,
        "flops": 2 * 256 * 32 * 16,
        "tokens_per_s": 256 / (t / 1e3),
        "tflops": 2 * 256 * 32 * 16 / (t / 1e3) / 1e12,
        "parity_status": "fail",
        "parity_failed": "grad_rel",
        "parity_grad_rel": 0.5,
        "parity_loss_rel_err": 1e-3,
        "role": None,
        "control_of": None,
        "compiled": None,
        "override_ran": None,
        "native_events": 0,
        "torch_version": "2.14.1+cu130",
        "torch_cuda": "13.0",
        "torch_git_version": S.ENVS["other"]["git_version"],
        "matmul_bf16_reduced": True,
        "packages": "counterpart=0123456789ab;helper=1.0",
        "tob_hash": "b" * 40,
        "info.api.name": "v2",
        "info.compile": None,
        "info.num_chunks": 4,
    }
    assert {k: row[k] for k in expected} == expected
    assert row["mean_ms"] == pytest.approx(t)
    control = find(
        analysis.at(rows, **S.DEFAULT), implementation="base_other", mode="fwd", dtype="float32"
    )
    assert (control["role"], control["control_of"]) == ("control", "base")
    compiled = find(
        analysis.at(rows, **S.DEFAULT), implementation="compiled", mode="fwd", dtype="float32"
    )
    assert compiled["native_events"] == 1 and compiled["info.compile"] == "fullgraph"


def test_flatten_error_record(synth):
    rows = analysis.to_rows(load(synth))
    (row,) = [r for r in rows if r["status"] == "error"]
    assert row["host"] == "host-b"  # no env: from the file name
    assert row["reason"].startswith("RuntimeError: boom")
    numbers = ["n", "median_ms", "peak_above_baseline", "tokens_per_s", "parity_status"]
    assert all(row[k] is None for k in [*numbers, "torch_version", "packages", "tob_hash"])
    assert row["tokens"] == 64  # the work is known before the run
    (row,) = [r for r in rows if r["status"] == "unavailable"]
    assert row["reason"] == "needs a GPU" and row["median_ms"] is None


def test_flatten_missing_fields():
    row = analysis.flatten({})
    assert row["op"] is None and row["median_ms"] is None and row["host"] is None
    row = analysis.flatten(
        {
            "case": {"params": {"M": 1}, "dtype": "float32", "device": "cpu"},
            "times_ms": [2.0],
            "memory": {"peak_bytes": 10, "baseline_bytes": 4},
            "env": {"torch": {"version": "2"}, "packages": {"x": None}},
            "info": {"a": {"b": [1, 2]}},
        },
        info_keys=["a.b", "absent"],
    )
    assert row["M"] == 1 and row["median_ms"] == 2.0 and row["iqr_ms"] == 0.0
    assert row["peak_above_baseline"] is None  # on CPU
    assert row["tokens_per_s"] is None  # no work unit
    assert row["matmul_bf16_reduced"] is None  # a record made before it was recorded
    assert row["packages"] == "x=not installed"
    assert row["info.a.b"] == json.dumps([1, 2]) and row["info.absent"] is None


def test_host_of():
    assert analysis.host_of({"env": {"hostname": "h"}}) == "h"
    assert analysis.host_of({analysis.SOURCE: "20261017T120000Z-my-host.jsonl"}) == "my-host"
    assert analysis.host_of({analysis.SOURCE: "odd.jsonl"}) is None
    assert analysis.host_of({}) is None


def test_varying_info(synth):
    recs = load(synth)
    assert analysis.varying_info(recs) == ["api.name", "compile", "num_chunks"]
    same = [{"status": "ok", "info": {"x": 1, "role": "a"}}, {"status": "ok", "info": {"x": 1}}]
    assert analysis.varying_info(same) == []  # role is a column of its own
    within = [{"status": "ok", "info": {"o": {"n": 1}}}, {"status": "ok", "info": {"o": {"n": 2}}}]
    assert analysis.varying_info(within) == ["o.n"]
    errors = [{"status": "ok", "info": {"x": 1}}, {"status": "error", "info": {"x": 2}}]
    assert analysis.varying_info(errors) == []  # ok records only


def test_default_point(synth):
    recs = load(synth)
    assert analysis.default_point(recs) == {"source": "spec", "params": S.DEFAULT}
    unresolved = [{**r, "op_ref": "no_such_module_xyz:SPEC"} for r in recs]
    assert analysis.default_point(unresolved) == {"source": "inferred", "params": S.DEFAULT}
    assert analysis.axes_of(unresolved) == ["K", "M", "N", "layout"]  # sorted


def test_sweeps(synth):
    rows = analysis.to_rows(load(synth))
    base = analysis.at(rows, implementation="base", mode="fwd_bwd", dtype="float32")
    swept = analysis.sweeps(base, S.DEFAULT)
    assert {axis: analysis.axis_values(rs, axis) for axis, rs in swept.items()} == S.SWEEP
    for rows_of_axis in swept.values():
        assert any(all(r[k] == v for k, v in S.DEFAULT.items()) for r in rows_of_axis)
    assert analysis.axis_values([{"a": "y"}, {"a": "x"}, {"a": "y"}], "a") == ["y", "x"]


def test_implementations_order(synth):
    rows = analysis.to_rows(load(synth))
    assert analysis.implementations(rows, "base") == ["base", "base_other", "compiled", "chunked"]
    # Without a baseline, first appearance (here the second host's file sorts first),
    # and a control still follows its original.
    order = analysis.implementations(rows, None)
    assert order == ["compiled", "base", "base_other", "chunked"]
    rows = [r for r in rows if r["implementation"] != "base"]
    assert analysis.implementations(rows, "base") == ["compiled", "chunked", "base_other"]


def test_records_unchanged_by_load(synth):
    # load reads through records.read: the files are not rewritten.
    before = sorted(p.read_text() for p in synth.rglob("*.jsonl"))
    load(synth)
    assert sorted(p.read_text() for p in synth.rglob("*.jsonl")) == before
    assert records.read(synth)
