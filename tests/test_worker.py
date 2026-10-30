import json
import os
import re
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


# torch's CUDA out-of-memory message, as I1-12 recorded it.
CUDA_OOM = (
    "CUDA out of memory. Tried to allocate 4.00 GiB. GPU 0 has a total capacity of 15.57 "
    "GiB of which 2.22 GiB is free. Including non-PyTorch memory, this process has 13.34 "
    "GiB memory in use. Of the allocated memory 9.19 GiB is allocated by PyTorch, and 3.97 "
    "GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is "
    "large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid "
    "fragmentation."
)
CUDA_FIGURES = (
    "tried to allocate 4.00 GiB, 2.22 GiB free; 9.19 GiB allocated, 3.97 GiB reserved "
    "but unallocated"
)


def out_of_memory_at(calls):
    """A ``run`` that runs the toy's plain run, and runs out of memory at call ``calls``."""
    import torch
    import toy_op

    count = []

    def run(inputs, mode):
        count.append(mode)
        if len(count) == calls:
            raise torch.OutOfMemoryError(CUDA_OOM)
        return toy_op.run_plain(inputs, mode)

    return run


# On CPU, with peak_memory calling fn once: the plain run's calls are parity (1),
# memory (2), warmup (3, SETTINGS' one) and timing (4 to 6).
@pytest.mark.parametrize(
    ("phase", "calls"), [("parity", 1), ("memory", 2), ("warmup", 3), ("timing", 5)]
)
def test_out_of_memory_is_unavailable(tests_dir, monkeypatch, phase, calls):
    # Task I1-16: a point that does not fit is foreseen, not an error.
    import torch
    import toy_op

    from torch_op_bench import measure

    monkeypatch.setattr(measure, "peak_memory", lambda fn, device, reset: fn() and {})
    monkeypatch.setattr(toy_op.TOY.implementation("plain"), "run", out_of_memory_at(calls))
    record = worker.run_payload(payload(tests_dir))
    assert record["status"] == "unavailable", record["traceback"]
    first, second = record["reason"].split("\n")
    assert first == f"out of memory on {worker.device_memory(torch.device('cpu'))}"
    assert second == f"{CUDA_FIGURES}; phase: {phase}"
    assert record["traceback"] is None
    for field in ("times_ms", "memory", "parity", "verify"):
        assert record[field] is None, field
    assert record["env"]["packages"]["pytest"]


@pytest.mark.parametrize("phase", ["setup", "verify"])
def test_out_of_memory_in_setup_and_verify(tests_dir, monkeypatch, phase):
    import toy_op

    def oom(*args):
        raise MemoryError()

    monkeypatch.setattr(toy_op.TOY.implementation("plain"), phase, oom)
    record = worker.run_payload(payload(tests_dir))
    assert record["status"] == "unavailable", record["traceback"]
    assert record["reason"].split("\n")[1] == f"MemoryError; phase: {phase}"


def test_out_of_memory_of_the_cpu_allocator(tests_dir, monkeypatch):
    # The CPU allocator raises a RuntimeError, not torch.OutOfMemoryError.
    import torch
    import toy_op

    def run(inputs, mode):
        return torch.empty(1 << 52, dtype=torch.uint8)  # 4 PiB

    monkeypatch.setattr(toy_op.TOY.implementation("plain"), "run", run)
    record = worker.run_payload(payload(tests_dir))
    assert record["status"] == "unavailable", record["traceback"]
    assert record["reason"].endswith("\ntried to allocate 4194304.00 GiB; phase: parity")


def test_out_of_memory_reason_parts(monkeypatch):
    import torch

    class Props:
        name = "NVIDIA GeForce RTX 5060 Ti"
        total_memory = 16_718_757_888  # ey's: 15.57 GiB

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: Props)
    assert worker.out_of_memory_reason(
        torch.device("cuda:0"), torch.OutOfMemoryError(CUDA_OOM), "timing"
    ) == (f"out of memory on NVIDIA GeForce RTX 5060 Ti (15.6 GiB)\n{CUDA_FIGURES}; phase: timing")
    # A message without the figures: its first line.
    assert worker.memory_figures(torch.OutOfMemoryError("odd\nmore")) == "OutOfMemoryError: odd"
    assert not worker.is_out_of_memory(RuntimeError("toy failure"))


def test_reference_out_of_memory_skips_parity(tests_dir, monkeypatch):
    # The reference not fitting (reference_fits was wrong) leaves the measurement
    # sound: ok, parity skipped.
    import torch
    import toy_op

    def oom(*args):
        raise torch.OutOfMemoryError(CUDA_OOM)

    monkeypatch.setattr(toy_op.TOY.reference, "run", oom)
    record = worker.run_payload(payload(tests_dir))
    assert record["status"] == "ok", record["traceback"]
    assert len(record["times_ms"]) == SETTINGS["repeat"]
    assert record["parity"]["status"] == "skipped"
    assert record["parity"]["reason"] == f"reference out of memory: {CUDA_FIGURES}"
    assert record["parity"]["metrics"] == {}
    monkeypatch.undo()
    monkeypatch.setattr(toy_op.TOY, "compare", oom)
    record = worker.run_payload(payload(tests_dir))
    assert record["status"] == "ok", record["traceback"]
    assert record["parity"]["reason"] == f"comparison out of memory: {CUDA_FIGURES}"
    # Any other failure of the reference is still an error.
    monkeypatch.undo()
    monkeypatch.setattr(toy_op.TOY.reference, "run", toy_op.run_broken)
    assert worker.run_payload(payload(tests_dir))["status"] == "error"


@pytest.mark.cuda
def test_subprocess_cuda_out_of_memory(tests_dir):
    # torch's own message on this device: its figures are read.
    record, _ = run_subprocess(payload(tests_dir, device="cuda", M=1 << 20, K=1 << 13))
    assert record["status"] == "unavailable", record["traceback"]
    first, second = record["reason"].split("\n")
    assert first.startswith("out of memory on ") and first.endswith(" GiB)")
    assert re.fullmatch(
        r"tried to allocate 32\.00 GiB, [\d.]+ \w+ free; [\d.]+ \w+ allocated, [\d.]+ \w+ "
        r"reserved but unallocated; phase: setup",
        second,
    ), second


@pytest.mark.parametrize("pairs", [True, False])
def test_apply_matmul_reduction(monkeypatch, pairs):
    import fake_matmul
    import torch

    flags = fake_matmul.PairFlags() if pairs else fake_matmul.old_flags()
    monkeypatch.setattr(torch.backends.cuda, "matmul", flags)
    worker.apply_matmul_reduction("default")  # touches nothing
    assert flags.allow_bf16_reduced_precision_reduction is True
    assert flags.allow_fp16_reduced_precision_reduction is True
    worker.apply_matmul_reduction("strict")
    for name in fake_matmul.PAIR_FLAGS:
        if pairs:
            assert flags.pairs[name] == (False, True)  # split-K kept: no cuBLASLt needed
        else:
            assert getattr(flags, name) is False
    with pytest.raises(ValueError, match="matmul_reduction must be one of"):
        worker.apply_matmul_reduction("loose")


@pytest.mark.parametrize("pairs", [True, False])
def test_run_payload_strict(tests_dir, monkeypatch, pairs):
    import fake_matmul
    import torch

    flags = fake_matmul.PairFlags() if pairs else fake_matmul.old_flags()
    monkeypatch.setattr(torch.backends.cuda, "matmul", flags)
    seen = {}
    real_measure = worker._measure

    def measure(*args):  # where the implementation's setup runs
        seen["at_setup"] = flags.allow_bf16_reduced_precision_reduction
        return real_measure(*args)

    monkeypatch.setattr(worker, "_measure", measure)
    data = payload(tests_dir)
    data["settings"] = {**SETTINGS, "matmul_reduction": "strict"}
    record = worker.run_payload(data)
    assert record["status"] == "ok", record["traceback"]
    assert seen["at_setup"] is False  # set before the implementation's setup
    assert record["settings"]["matmul_reduction"] == "strict"
    matmul = record["env"]["torch"]["matmul"]
    assert matmul["allow_bf16_reduced_precision_reduction"] is False
    assert matmul["allow_fp16_reduced_precision_reduction"] is False
    split_k = matmul["allow_bf16_reduced_precision_reduction_split_k"]
    assert split_k is (True if pairs else None)


def test_run_payload_default_touches_no_flag(tests_dir, monkeypatch):
    import fake_matmul
    import torch

    flags = fake_matmul.PairFlags()
    flags.allow_fp16_reduced_precision_reduction = (True, False)  # not the default
    monkeypatch.setattr(torch.backends.cuda, "matmul", flags)
    record = worker.run_payload(payload(tests_dir))  # no matmul_reduction: default
    assert record["status"] == "ok", record["traceback"]
    assert flags.pairs == {
        "allow_bf16_reduced_precision_reduction": (True, True),
        "allow_fp16_reduced_precision_reduction": (True, False),
    }
