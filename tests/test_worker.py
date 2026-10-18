import json
import os
import subprocess
import sys

import pytest

from torch_op_bench import worker

SETTINGS = {"warmup": 1, "repeat": 3, "flush_l2": False, "seed": 0, "timeout": 60}


def payload(tests_dir, impl="plain", mode="fwd", device="cpu", dtype="float32", **params):
    point = {"M": 8, "K": 4, "N": 4, "layout": "contiguous", **params}
    return {
        "op": "toy_op:TOY",
        "op_name": "toy",
        "case": {"params": point, "dtype": dtype, "device": device},
        "implementation": impl,
        "mode": mode,
        "settings": SETTINGS,
        "pythonpath": [str(tests_dir)],
        "env_name": "torch",
        "env_vars": {"TOB_TOY_FLAG": "on"} if impl == "flagged" else {},
        "packages": ["pytest"],
        "device_slug": device,
        "work": {"flops": 256.0},
        "info": {},
    }


def run_subprocess(data, env=None):
    proc = subprocess.run(
        [sys.executable, "-m", "torch_op_bench.worker"],
        input=json.dumps(data),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    lines = proc.stdout.splitlines()
    assert len(lines) == 1, (proc.stdout, proc.stderr)
    return json.loads(lines[0]), proc


@pytest.mark.parametrize("mode", ["fwd", "fwd_bwd"])
@pytest.mark.parametrize("impl", ["plain", "chunked", "backward"])
def test_run_payload_ok(tests_dir, impl, mode):
    record = worker.run_payload(payload(tests_dir, impl, mode, layout="transposed"))
    assert record["status"] == "ok", record["traceback"]
    assert len(record["times_ms"]) == SETTINGS["repeat"]
    assert record["memory"] == {"peak_bytes": None, "baseline_bytes": None}
    assert record["parity"]["status"] == "ok"
    expected = {"y_abs_err", "y_rel_err"}
    if mode == "fwd_bwd":
        expected |= {"grad_a_rel_l2", "grad_b_rel_l2"}
    assert set(record["parity"]["metrics"]) == expected
    assert record["case_key"] == "K=4,M=8,N=4,layout=transposed|float32|cpu"
    assert record["env_from"] == "worker" and record["env"]["packages"]["pytest"]
    assert record["work"] == {"flops": 256.0}
    json.dumps(record)


def test_run_payload_unavailable(tests_dir):
    record = worker.run_payload(payload(tests_dir, "missing"))
    assert record["status"] == "unavailable"
    assert record["reason"] == "toy package not installed"
    assert record["times_ms"] is None


def test_setup_raises_unavailable(tests_dir, monkeypatch):
    import toy_op

    from torch_op_bench.spec import Unavailable

    def unsupported(case):
        raise Unavailable("toy needs a device it does not have")

    timed = []
    monkeypatch.setattr(toy_op.TOY.implementation("plain"), "setup", unsupported)
    monkeypatch.setattr(worker, "_parity", lambda *args: timed.append(args))
    data = payload(tests_dir, "plain")
    record = worker.run_payload(data)
    assert record["status"] == "unavailable"
    assert record["reason"] == "toy needs a device it does not have"
    assert record["traceback"] is None
    assert timed == []
    for field in ("times_ms", "memory", "parity", "verify"):
        assert record[field] is None, field
    assert record["info"] == data["info"]
    assert record["env_from"] == "worker" and record["env"]["packages"]["pytest"]


def test_run_payload_error(tests_dir):
    record = worker.run_payload(payload(tests_dir, "broken"))
    assert record["status"] == "error"
    assert record["reason"] == "RuntimeError: toy failure"
    assert "run_broken" in record["traceback"]
    assert record["env"] is not None


def test_run_payload_unsupported_mode(tests_dir):
    record = worker.run_payload(payload(tests_dir, "broken", mode="fwd_bwd"))
    assert record["status"] == "error" and "does not support mode" in record["reason"]


def test_parity_fail_and_skips(tests_dir, monkeypatch):
    import toy_op

    monkeypatch.setitem(toy_op.TOY.tolerances, "y_rel_err", -1.0)
    record = worker.run_payload(payload(tests_dir))
    assert record["status"] == "ok"
    assert record["parity"]["status"] == "fail" and record["parity"]["failed"] == ["y_rel_err"]
    monkeypatch.setattr(toy_op.TOY, "reference_fits", lambda case: "too big")
    record = worker.run_payload(payload(tests_dir))
    assert record["parity"]["status"] == "skipped" and record["parity"]["reason"] == "too big"
    monkeypatch.setattr(toy_op.TOY, "reference_fits", lambda case: False)
    record = worker.run_payload(payload(tests_dir))
    assert record["parity"]["reason"] == "reference does not fit"
    monkeypatch.setattr(toy_op.TOY, "reference_fits", lambda case: True)
    monkeypatch.setattr(toy_op.TOY.reference, "modes", {"fwd_bwd"})
    record = worker.run_payload(payload(tests_dir))
    assert record["parity"]["status"] == "skipped" and "mode 'fwd'" in record["parity"]["reason"]


def test_backward_style_parity_against_backward_style_reference(tests_dir, monkeypatch):
    # A reference that also calls backward() on the shared inputs (and keeps copies of
    # its results): without clearing gradients between calls, the implementation's
    # would include the reference's.
    import toy_op

    def reference(inputs, mode):
        return tuple(t.detach().clone() for t in toy_op.run_backward(inputs, mode))

    monkeypatch.setattr(toy_op.TOY.reference, "run", reference)
    record = worker.run_payload(payload(tests_dir, "backward", "fwd_bwd", dtype="float64"))
    assert record["status"] == "ok", record["traceback"]
    assert record["parity"]["status"] == "ok"
    assert record["parity"]["metrics"]["grad_a_rel_l2"] < 1e-12


def test_env_failure_keeps_the_record(tests_dir, monkeypatch):
    def broken(packages=()):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(worker.env, "collect", broken)
    record = worker.run_payload(payload(tests_dir))
    assert record["status"] == "ok" and len(record["times_ms"]) == SETTINGS["repeat"]
    assert record["env"] is None
    assert (
        record["env_error"] == "RuntimeError: CUDA error: an illegal memory access was encountered"
    )


def test_setup_dict_merged_into_info(tests_dir, monkeypatch):
    data = payload(tests_dir, "described")
    data["info"] = {"chunk": "rows of 4", "resolved_chunk": None}
    record = worker.run_payload(data)
    assert record["status"] == "ok", record["traceback"]
    assert record["info"] == {"chunk": "rows of 4", "resolved_chunk": 4, "M_seen": 8}

    import toy_op

    monkeypatch.setattr(toy_op.TOY.implementation("described"), "setup", lambda case: [1])
    record = worker.run_payload(data)
    assert record["status"] == "error"
    assert record["reason"] == "TypeError: setup returned a list, not a dict or None"
    assert record["info"] == data["info"]


def test_unavailable_without_reason(tests_dir, monkeypatch):
    import toy_op

    monkeypatch.setattr(toy_op.TOY.implementation("missing"), "available", lambda: False)
    record = worker.run_payload(payload(tests_dir, "missing"))
    assert record["status"] == "unavailable" and record["reason"] == "not available"


def test_setup_runs(tests_dir, monkeypatch):
    import toy_op

    seen = []
    monkeypatch.setattr(toy_op.TOY.implementation("plain"), "setup", seen.append)
    worker.run_payload(payload(tests_dir))
    assert [case.params["M"] for case in seen] == [8]


def test_subprocess_ok_and_stdout_kept_clean(tests_dir, monkeypatch):
    # The flagged implementation's verify checks that TOB_TOY_FLAG was set before torch
    # was imported; it also prints, which must not reach the record stream.
    env = {k: v for k, v in os.environ.items() if k != "TOB_TOY_FLAG"}
    record, proc = run_subprocess(payload(tests_dir, "flagged", "fwd_bwd"), env=env)
    assert record["status"] == "ok", record["traceback"]
    assert record["verify"] == {"flag": "on"}
    assert record["python"] == sys.executable


def test_subprocess_unavailable_and_error(tests_dir):
    record, _ = run_subprocess(payload(tests_dir, "missing"))
    assert record["status"] == "unavailable"
    record, _ = run_subprocess(payload(tests_dir, "broken"))
    assert record["status"] == "error" and "toy failure" in record["traceback"]


def test_flag_missing_fails_verify(tests_dir):
    data = payload(tests_dir, "flagged")
    data["env_vars"] = {}
    env = {k: v for k, v in os.environ.items() if k != "TOB_TOY_FLAG"}
    record, _ = run_subprocess(data, env=env)
    assert record["status"] == "error" and "TOB_TOY_FLAG not set" in record["reason"]


@pytest.mark.cuda
def test_subprocess_cuda_memory_backward_style_matches(tests_dir):
    plain, _ = run_subprocess(payload(tests_dir, "plain", "fwd_bwd", device="cuda", M=256, K=256))
    back, _ = run_subprocess(payload(tests_dir, "backward", "fwd_bwd", device="cuda", M=256, K=256))
    assert back["status"] == "ok", back["traceback"]
    assert back["memory"]["baseline_bytes"] == plain["memory"]["baseline_bytes"]
    assert back["parity"]["status"] == "ok"


@pytest.mark.cuda
def test_subprocess_cuda_memory(tests_dir):
    record, _ = run_subprocess(payload(tests_dir, "plain", "fwd_bwd", device="cuda", M=256, K=256))
    assert record["status"] == "ok", record["traceback"]
    assert record["memory"]["peak_bytes"] > record["memory"]["baseline_bytes"]
    assert record["parity"]["status"] == "ok"
    assert record["env"]["torch"]["devices"]
