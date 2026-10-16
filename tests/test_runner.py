import io
import subprocess
import sys

import pytest
import toy_op

from torch_op_bench import records, runner
from torch_op_bench.spec import Case

SPEC = toy_op.TOY
FAST = runner.Settings(warmup=1, repeat=2, timeout=120)


def test_settings_and_env_map():
    assert runner.Settings().to_dict() == {
        "warmup": 10,
        "repeat": 50,
        "flush_l2": False,
        "seed": 0,
        "timeout": 600.0,
    }
    assert runner.parse_env_map() == {"torch": sys.executable}
    env_map = runner.parse_env_map(["other=/x/python", "torch=/y/python"])
    assert env_map == {"torch": "/y/python", "other": "/x/python"}
    for bad in ["other", "=x", "other="]:
        with pytest.raises(ValueError, match="NAME=PYTHON"):
            runner.parse_env_map([bad])


def test_device_slug(monkeypatch):
    assert runner.device_slug("cpu") == "cpu"
    with pytest.raises(ValueError, match="cannot get the name of cuda:1 .*--device-slug"):
        runner.device_slug("cuda:1", python="/no/such/python")
    with pytest.raises(ValueError, match=r"\(exit code 1\); pass --device-slug"):
        runner.device_slug("cuda", python="false")

    def fake_run(cmd, **kwargs):
        assert "get_device_name(1)" in cmd[-1]
        return subprocess.CompletedProcess(cmd, 0, stdout="warning\nNVIDIA H100 80GB HBM3\n")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.device_slug("cuda:1") == "nvidia-h100-80gb-hbm3"


def test_enumerate_one_axis_at_a_time():
    cases = runner.enumerate_cases(SPEC, device="cpu", dtypes=["float32"])
    points = [(c.params["M"], c.params["K"], c.params["N"], c.params["layout"]) for c in cases]
    assert points == [
        (8, 4, 4, "contiguous"),
        (16, 4, 4, "contiguous"),  # the default point, once
        (32, 4, 4, "contiguous"),
        (16, 8, 4, "contiguous"),
        (16, 4, 4, "transposed"),
    ]
    assert all(c.dtype == "float32" and c.device == "cpu" for c in cases)


def test_enumerate_sweep_values_dtypes_and_explicit_cases():
    cases = runner.enumerate_cases(SPEC, device="cpu", sweep=["K"], values={"K": [2, 3]})
    assert [(c.params["K"], c.dtype) for c in cases] == [
        (2, "float32"),
        (3, "float32"),
        (2, "float64"),
        (3, "float64"),
    ]
    cases = runner.enumerate_cases(
        SPEC, device="cpu", dtypes=["float64"], cases=[{"M": 5}, {"M": 6, "layout": "transposed"}]
    )
    assert [c.params for c in cases] == [
        {"M": 5, "K": 4, "N": 4, "layout": "contiguous"},
        {"M": 6, "K": 4, "N": 4, "layout": "transposed"},
    ]
    with pytest.raises(ValueError, match="dtypes"):
        runner.enumerate_cases(SPEC, device="cpu", dtypes=["int8"])
    with pytest.raises(KeyError):
        runner.enumerate_cases(SPEC, device="cpu", cases=[{"Z": 1}])
    with pytest.raises(KeyError):
        runner.enumerate_cases(SPEC, device="cpu", sweep=["Z"])
    with pytest.raises(ValueError, match=r"not swept: \['K'\]"):
        runner.enumerate_cases(SPEC, device="cpu", sweep=["M"], values={"K": [99]})
    with pytest.raises(ValueError, match="not to explicit cases"):
        runner.enumerate_cases(SPEC, device="cpu", cases=[{"M": 5}], values={"M": [1]})


def test_enumerate_uses_default_dtypes(monkeypatch):
    monkeypatch.setattr(SPEC, "default_dtypes", ["float64"])
    cases = runner.enumerate_cases(SPEC, device="cpu", sweep=["K"])
    assert {c.dtype for c in cases} == {"float64"}
    cases = runner.enumerate_cases(SPEC, device="cpu", sweep=["K"], dtypes=["float32"])
    assert {c.dtype for c in cases} == {"float32"}


def test_enumerate_runs_respects_modes():
    case = Case(SPEC.default_point, "float32", "cpu")
    runs = runner.enumerate_runs(SPEC, [case])
    pairs = [(impl.name, mode) for _, impl, mode in runs]
    assert ("broken", "fwd") in pairs and ("broken", "fwd_bwd") not in pairs
    assert len(pairs) == 14
    runs = runner.enumerate_runs(SPEC, [case], ["plain", "broken"], ["fwd_bwd"])
    assert [(impl.name, mode) for _, impl, mode in runs] == [("plain", "fwd_bwd")]
    with pytest.raises(KeyError):
        runner.enumerate_runs(SPEC, [case], ["nope"])
    with pytest.raises(ValueError, match="unknown modes \\['bwd'\\]"):
        runner.enumerate_runs(SPEC, [case], modes=["bwd"])


def sweep(tmp_path, tests_dir, impls, modes=("fwd",), env_map=None, settings=FAST, **kw):
    cases = runner.enumerate_cases(SPEC, device="cpu", dtypes=["float32"], sweep=["K"])
    out = io.StringIO()
    written = runner.run_sweep(
        SPEC,
        "toy_op:TOY",
        runner.enumerate_runs(SPEC, cases, impls, modes),
        settings=settings,
        results_dir=tmp_path,
        slug="cpu",
        env_map=env_map or runner.parse_env_map(),
        pythonpath=[str(tests_dir)],
        out=out,
        **kw,
    )
    return written, out.getvalue().splitlines()


def test_run_sweep_resume_and_force(tmp_path, tests_dir):
    written, lines = sweep(tmp_path, tests_dir, ["plain", "missing"])
    assert [r["status"] for r in written] == ["ok", "unavailable", "ok", "unavailable"]
    assert lines[0].startswith("[1/4] plain fwd K=4,M=16,N=4,layout=contiguous|float32|cpu: ok")
    assert "unavailable: toy package not installed" in lines[1]
    files = list((tmp_path / "toy" / "cpu").glob("*.jsonl"))
    assert len(files) == 1
    assert records.read(tmp_path) == written
    record = written[0]
    assert record["settings"] == FAST.to_dict()
    assert record["case"] == {"params": SPEC.default_point, "dtype": "float32", "device": "cpu"}
    assert record["env"]["torch"]["version"] and record["env_from"] == "worker"

    # Resume: ok records are skipped, unavailable ones run again.
    written, lines = sweep(tmp_path, tests_dir, ["plain", "missing"])
    assert [r["implementation"] for r in written] == ["missing", "missing"]
    assert lines[0].endswith("recorded, skipped")

    written, _ = sweep(tmp_path, tests_dir, ["plain"], force=True)
    assert [r["status"] for r in written] == ["ok", "ok"]
    assert len(records.merge(records.read(tmp_path))) == 4


def test_run_sweep_environments(tmp_path, tests_dir):
    written, lines = sweep(tmp_path, tests_dir, ["elsewhere"])
    assert {r["status"] for r in written} == {"error"}
    assert "no interpreter for environment 'other'" in written[0]["reason"]
    assert written[0]["env_from"] == "runner" and written[0]["env"]["torch"] is None
    assert written[0]["python"] is None

    env_map = runner.parse_env_map([f"other={sys.executable}"])
    written, _ = sweep(tmp_path, tests_dir, ["elsewhere"], env_map=env_map)
    assert [r["status"] for r in written] == ["ok", "ok"]
    assert written[0]["env_name"] == "other" and written[0]["python"] == sys.executable
    assert written[0]["env_from"] == "worker"


def test_run_sweep_error_and_parity_lines(tmp_path, tests_dir, monkeypatch):
    written, lines = sweep(tmp_path, tests_dir, ["broken"])
    assert written[0]["status"] == "error"
    assert "error: RuntimeError: toy failure" in lines[0]
    monkeypatch.setitem(SPEC.tolerances, "y_rel_err", -1.0)
    record = written[0] | {
        "status": "ok",
        "times_ms": [1.0],
        "parity": {"status": "fail", "failed": ["y_rel_err"]},
    }
    assert runner._describe(record) == "ok, median 1 ms, parity fail (y_rel_err)"
    record["parity"] = {"status": "skipped", "failed": []}
    assert runner._describe(record) == "ok, median 1 ms, parity skipped"


def test_launch_timeout(tmp_path, tests_dir):
    settings = runner.Settings(warmup=1, repeat=2, timeout=0.01)
    written, lines = sweep(tmp_path, tests_dir, ["plain"], settings=settings)
    assert written[0]["status"] == "error"
    assert written[0]["reason"] == "timeout after 0.01 s"
    assert "timeout" in lines[0]


def test_launch_failures(tests_dir, tmp_path):
    case = Case(SPEC.default_point, "float32", "cpu")
    payload = runner.make_payload(
        SPEC, "toy_op:TOY", case, SPEC.implementation("plain"), "fwd", FAST, slug="cpu"
    )
    record = runner.launch(payload, str(tmp_path / "no-python"), 10)
    assert record["status"] == "error" and record["reason"].startswith("cannot launch")

    fake = tmp_path / "fake-python"
    fake.write_text("#!/bin/sh\necho broken >&2\nexit 3\n")
    fake.chmod(0o755)
    record = runner.launch(payload, str(fake), 10)
    assert record["reason"] == "worker exited with code 3 without a record"
    assert record["stderr"] == "broken\n"


def test_worker_error_carries_stderr(tests_dir):
    case = Case(SPEC.default_point, "float32", "cpu")
    payload = runner.make_payload(
        SPEC,
        "toy_op:TOY",
        case,
        SPEC.implementation("broken"),
        "fwd",
        FAST,
        slug="cpu",
        pythonpath=[str(tests_dir)],
    )
    record = runner.launch(payload, sys.executable, 120)
    assert record["status"] == "error" and record["env_from"] == "worker"
    assert "toy: about to fail" in record["stderr"]


def test_tail():
    assert runner._tail(None) is None
    assert runner._tail(b"x\xff") == "x�"
    assert len(runner._tail("y" * (runner.STDERR_LIMIT + 5))) == runner.STDERR_LIMIT


def _alive(pid: int) -> bool:
    # A killed orphan may linger as a zombie until it is reaped: that counts as dead.
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _wait_dead(pid: int, seconds: float = 5.0) -> bool:
    import time

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _spawner_payload(tests_dir):
    case = Case(SPEC.default_point, "float32", "cpu")
    return runner.make_payload(
        SPEC,
        "toy_op:TOY",
        case,
        SPEC.implementation("spawner"),
        "fwd",
        FAST,
        slug="cpu",
        pythonpath=[str(tests_dir)],
    )


def test_launch_kills_children_of_a_finished_worker(tests_dir, tmp_path, monkeypatch):
    import time

    pidfile = tmp_path / "pid"
    monkeypatch.setenv("TOB_TOY_PIDFILE", str(pidfile))
    begin = time.monotonic()
    record = runner.launch(_spawner_payload(tests_dir), sys.executable, 60)
    # The surviving child keeps the worker's stdout and stderr open: the runner must
    # not wait for them, and must not turn a finished run into a timeout.
    assert record["status"] == "ok", record["reason"]
    assert time.monotonic() - begin < 30
    assert _wait_dead(int(pidfile.read_text()))


def test_launch_timeout_kills_the_process_group(tests_dir, tmp_path, monkeypatch):
    pidfile = tmp_path / "pid"
    monkeypatch.setenv("TOB_TOY_PIDFILE", str(pidfile))
    monkeypatch.setenv("TOB_TOY_HANG", "1")
    record = runner.launch(_spawner_payload(tests_dir), sys.executable, 8)
    assert record["status"] == "error" and record["reason"] == "timeout after 8 s"
    assert _wait_dead(int(pidfile.read_text()))


def test_alive_helper():
    import os

    assert _alive(os.getpid())
    assert not _alive(2**22 + 12345)
