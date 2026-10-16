import datetime
import json

import pytest

from torch_op_bench import records


def rec(impl="plain", status="ok", timestamp="2026-10-16T10:00:00+00:00", git="a" * 40, **kw):
    record = {
        "schema": records.SCHEMA,
        "op": "toy",
        "device_slug": "cpu",
        "case_key": "M=8|float32|cpu",
        "implementation": impl,
        "mode": "fwd",
        "status": status,
        "timestamp": timestamp,
        "env": {"torch_op_bench": {"hash": git}},
    }
    record.update(kw)
    return record


def test_slugify():
    assert records.slugify("NVIDIA GeForce RTX 5060 Ti") == "nvidia-geforce-rtx-5060-ti"
    assert records.slugify("--") == "unknown"


def test_result_path():
    ts = datetime.datetime(2026, 10, 16, 9, 5, 7, tzinfo=datetime.timezone.utc)
    path = records.result_path("results", "toy", "cpu", timestamp=ts, host="Box.Local")
    assert path.as_posix() == "results/toy/cpu/20261016T090507Z-box-local.jsonl"
    assert records.result_path("r", "toy", "cpu").parent.as_posix() == "r/toy/cpu"


def test_write_read_file_and_tree(tmp_path):
    a = tmp_path / "toy" / "cpu" / "1-a.jsonl"
    b = tmp_path / "toy" / "gpu" / "2-b.jsonl"
    records.append(a, rec("plain"))
    records.append(a, rec("chunked"))
    records.append(b, rec("plain"))
    with a.open("a") as f:
        f.write("\n")
    assert [r["implementation"] for r in records.read(a)] == ["plain", "chunked"]
    assert len(records.read(tmp_path)) == 3
    assert json.loads(a.read_text().splitlines()[0]) == rec("plain")


def test_read_rejects_other_schema(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps({"schema": 99}) + "\n")
    with pytest.raises(ValueError, match="schema 99"):
        records.read(path)


def test_key_and_completed():
    ok, err = rec("plain"), rec("chunked", status="error")
    assert records.key(ok) == ("toy", "cpu", "M=8|float32|cpu", "plain", "fwd")
    assert records.completed([ok, err, rec("x", status="unavailable")]) == {records.key(ok)}


def test_merge_keeps_latest_ok():
    older = rec(timestamp="2026-10-16T09:00:00+00:00", tag="older")
    newer = rec(timestamp="2026-10-16T11:00:00+00:00", tag="newer")
    failed_later = rec(status="error", timestamp="2026-10-16T12:00:00+00:00")
    other = rec("chunked", status="error", tag="e1")
    other_later = rec("chunked", status="error", tag="e2")
    merged = records.merge([newer, other, older, failed_later, other_later])
    assert [r.get("tag") for r in merged] == ["newer", "e2"]


def test_merge_warns_on_git_mismatch():
    with pytest.warns(UserWarning, match="different torch-op-bench commits"):
        records.merge([rec("plain"), rec("chunked", git="b" * 40)])


def test_stats():
    assert records.stats([]) == {"n": 0, "median": None, "min": None, "mean": None, "iqr": None}
    assert records.stats([2.0]) == {"n": 1, "median": 2.0, "min": 2.0, "mean": 2.0, "iqr": 0.0}
    s = records.stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s == {"n": 5, "median": 3.0, "min": 1.0, "mean": 3.0, "iqr": 2.0}
