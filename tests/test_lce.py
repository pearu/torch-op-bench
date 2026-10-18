import sys

import pytest
import torch

from torch_op_bench import cli, records, runner, worker
from torch_op_bench.measure import clear_grads
from torch_op_bench.ops import lce
from torch_op_bench.spec import Case, resolve_op

SPEC = lce.LCE
IMPLS = ["torch_eager", "torch_compiled", "torch_lce", "torch_lce_no_override"]
LIGER_IMPLS = ["liger", "liger_native_accum", "torch_eager_liger"]
TITAN = ["torchtitan_chunked", "torchtitan_chunked_compiled", "torchtitan_ce"]
TITAN_IMPLS = [*TITAN, "torch_eager_torchtitan"]
TINY = {"N": 64, "H": 32, "V": 96}
# Whether this torch has the torch._native LCE override (nightlies do, 2.14.1 not).
OVERRIDE = lce.lce_override_present()


def case(dtype="float32", ignore_frac="0", device="cpu", **params):
    return Case({**TINY, "ignore_frac": ignore_frac, **params}, dtype, device)


def inputs(c, seed=0):
    generator = torch.Generator(device=c.device)
    generator.manual_seed(seed)
    return SPEC.make_inputs(c, torch.device(c.device), generator)


@pytest.fixture(autouse=True)
def restore_overrides():
    # torch_lce_no_override's setup disables the override process-wide; undo it.
    yield
    if hasattr(torch.backends, "python_native"):
        torch.backends.python_native.enable_operations(*lce.OVERRIDE_OPS)


def test_registered():
    assert resolve_op("lce") is SPEC
    assert [impl.name for impl in SPEC.implementations] == IMPLS + LIGER_IMPLS + TITAN_IMPLS
    assert SPEC.default_point == {"N": 4096, "H": 4096, "V": 131072, "ignore_frac": "0"}
    assert SPEC.default_dtypes == ["bfloat16"]
    for impl in [*SPEC.implementations, SPEC.reference]:
        assert {k: impl.info[k] for k in lce.CONTRACT} == lce.CONTRACT
        if impl.name in LIGER_IMPLS + TITAN_IMPLS:
            continue
        assert impl.env_name == "torch" and impl.modes == {"fwd", "fwd_bwd"}
    eager, control = SPEC.implementation("torch_eager"), SPEC.implementation("torch_eager_liger")
    assert control.env_name == "liger" and control.run is eager.run
    assert control.info == {**eager.info, "role": "control", "control_of": "torch_eager"}
    assert control.modes == {"fwd", "fwd_bwd"}
    for name, accum_dtype in [("liger", "float32"), ("liger_native_accum", None)]:
        impl = SPEC.implementation(name)
        assert impl.env_name == "liger" and impl.modes == {"fwd_bwd"}
        assert impl.packages == ("liger-kernel", "triton")
        assert impl.info == {**lce.CONTRACT, "accum_dtype": accum_dtype, "backend": "liger"}
    control = SPEC.implementation("torch_eager_torchtitan")
    assert control.env_name == "torchtitan" and control.run is eager.run
    assert control.info == {**eager.info, "role": "control", "control_of": "torch_eager"}
    assert control.modes == {"fwd", "fwd_bwd"} and control.verify is lce.verify_native
    for name in TITAN:
        impl = SPEC.implementation(name)
        assert impl.env_name == "torchtitan"
        assert impl.modes == ({"fwd", "fwd_bwd"} if name == "torchtitan_ce" else {"fwd_bwd"})
        assert impl.packages == ("torchtitan", "spmd_types")
        assert impl.info == {**lce.CONTRACT, "backend": "torchtitan"}
    compiled = SPEC.implementation("torchtitan_chunked_compiled")
    assert compiled.verify is lce.verify_torchtitan_compiled
    assert compiled.available is lce.torchtitan_compiled_available
    for name in ["torchtitan_chunked", "torchtitan_ce"]:
        impl = SPEC.implementation(name)
        assert impl.verify is lce.verify_native and impl.available is lce.torchtitan_available


@pytest.mark.parametrize("ignore_frac, ignored", [("0", 0), ("0.25", 16), ("1", 64)])
def test_inputs_deterministic(ignore_frac, ignored):
    c = case("bfloat16", ignore_frac)
    hidden, weight, labels = inputs(c)
    again = inputs(c)
    for x, y in zip((hidden, weight, labels), again, strict=True):
        assert torch.equal(x, y)
    assert not torch.equal(hidden, inputs(c, seed=1)[0])
    assert hidden.shape == (64, 32) and weight.shape == (96, 32) and labels.shape == (64,)
    assert hidden.dtype == weight.dtype == torch.bfloat16 and labels.dtype == torch.int64
    assert hidden.requires_grad and weight.requires_grad and not labels.requires_grad
    assert int((labels == -100).sum()) == ignored
    assert bool((labels < 96).all())
    assert weight.float().std().item() == pytest.approx(32**-0.5, rel=0.2)


def test_inputs_reject_bad_fraction():
    with pytest.raises(ValueError, match="ignore_frac"):
        inputs(case(ignore_frac="1.5"))


def _impl_outputs(name, c, mode):
    impl = SPEC.implementation(name)
    if impl.setup is not None:
        impl.setup(c)
    x = inputs(c)
    return impl.run(x, mode), x


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("ignore_frac", ["0", "0.25"])
@pytest.mark.parametrize("name", IMPLS)
def test_parity_and_modes(name, dtype, ignore_frac):
    c = case(dtype, ignore_frac)
    outputs, x = _impl_outputs(name, c, "fwd_bwd")
    assert outputs[1].shape == (64, 32) and outputs[2].shape == (96, 32)
    ref = SPEC.reference.run(x, "fwd_bwd")
    assert ref[0].dtype == torch.float32
    metrics = SPEC.compare(outputs, ref, c)
    assert set(metrics) == {
        "loss_abs_err",
        "loss_rel_err",
        "grad_hidden_rel_l2",
        "grad_weight_rel_l2",
    }
    for metric, tolerance in SPEC.tolerances.items():
        assert metrics[metric] <= tolerance, (metric, metrics)
    clear_grads(x)
    (loss_fwd,) = SPEC.implementation(name).run(x, "fwd")
    assert not loss_fwd.requires_grad
    torch.testing.assert_close(loss_fwd, outputs[0])
    assert set(SPEC.compare((loss_fwd,), ref[:1], c)) == {"loss_abs_err", "loss_rel_err"}


@pytest.mark.parametrize("name", IMPLS)
def test_repeated_calls_identical(name):
    c = case("float32", "0.25")
    impl = SPEC.implementation(name)
    if impl.setup is not None:
        impl.setup(c)
    x = inputs(c)
    first = [t.clone() for t in impl.run(x, "fwd_bwd")]
    clear_grads(x)
    second = impl.run(x, "fwd_bwd")
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_setup_records_options_and_source_hash():
    c = case("bfloat16")
    info = lce.setup_lce(c)
    assert info["compiled"] is False
    assert set(info["options"]) == {
        "acc_policy",
        "acc_dtype",
        "chunking_method",
        "batch_chunk_size",
        "resolved",
    }
    assert info["options"]["resolved"] is True
    assert info["options"]["chunking_method"] == "aspect_ratio"
    assert info["options"]["acc_dtype"] == "float32"
    md5 = info["lce_source_md5"]
    assert isinstance(md5, str) and len(md5) == 32 and int(md5, 16) >= 0
    assert "nn/modules/linear_cross_entropy.py" in info["lce_source_files"]
    no_override = lce.setup_lce_no_override(c)
    assert no_override["override_disabled"] == list(lce.OVERRIDE_OPS)
    assert no_override["override_present"] is OVERRIDE
    assert info["override_registration"] == lce.override_registration()
    assert no_override["lce_source_md5"] == md5
    assert lce.setup_eager(c) == {"compiled": False}
    assert lce.setup_compiled(c) == {"compiled": True}


def test_op_decisions_of_the_build():
    # This build's _ChunkContext, for the case: CPU, bf16 accumulated in fp32.
    decisions = lce.op_decisions(case("bfloat16"))
    assert decisions["source"] == f"{lce.CHUNK_CONTEXT}.build"
    assert decisions["is_cuda"] is False and decisions["use_acc_dtype"] is True
    assert decisions["forward_uses_cuda_out_dtype"] is False
    assert decisions["acc_dtype"] == "float32" and decisions["dtype"] == "bfloat16"
    assert not any(name.startswith("compute_") for name in decisions)
    assert lce.op_decisions(case("float32"))["use_acc_dtype"] is False
    info = lce.setup_lce(case("bfloat16"))
    assert info["op_decisions"] == decisions
    assert lce.setup_lce_no_override(case("bfloat16"))["op_decisions"] == decisions


@pytest.mark.cuda
def test_op_decisions_on_cuda():
    decisions = lce.op_decisions(case("bfloat16", device="cuda"))
    assert decisions["is_cuda"] is True and decisions["forward_uses_cuda_out_dtype"] is True


def _fake_lce_module(monkeypatch, **attrs):
    """A fake torch.nn.modules.linear_cross_entropy holding ``attrs``."""
    import types

    module = types.ModuleType(lce.CHUNK_CONTEXT.rpartition(".")[0])
    for name, value in attrs.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_op_decisions_of_a_fake_context(monkeypatch):
    import dataclasses
    import types

    calls = []

    @dataclasses.dataclass
    class _ChunkContext:
        is_cuda: bool
        acc_dtype: torch.dtype
        compute_input_grad: bool
        alloc_weight_grad_chunk: bool  # depends on the gradients asked: left out
        batch_chunk_size: int  # not a decision: left out

        @classmethod
        def build(cls, **kwargs):
            calls.append(kwargs)
            grads = kwargs["compute_input_grad"]
            return cls(False, kwargs["acc_dtype"], grads, grads, kwargs["batch_chunk_size"])

    class Options:
        def _adjust(self, **kwargs):
            return types.SimpleNamespace(acc_policy="compact", acc_dtype=torch.float32)

    _fake_lce_module(monkeypatch, _ChunkContext=_ChunkContext)
    monkeypatch.setattr(lce, "_lce_options", Options)
    c = case("bfloat16", N=4096, H=4096, V=131072)
    assert lce.op_decisions(c) == {
        "source": f"{lce.CHUNK_CONTEXT}.build",
        "is_cuda": False,
        "acc_dtype": "float32",
    }
    assert [k["compute_input_grad"] for k in calls] == [False, True]
    hidden, weight = calls[0]["input"], calls[0]["linear_weight"]
    assert hidden.shape == (4096, 4096) and weight.shape == (131072, 4096)
    assert hidden.stride() == (0, 0) and weight.stride() == (0, 0)  # nothing allocated
    assert calls[0]["batch_chunk_size"] == 1 and calls[0]["acc_policy"] == "compact"


def test_op_decisions_of_the_pr_predicates(monkeypatch):
    seen = []

    def fp32_out(dtype, device):
        seen.append((dtype, device))
        return True

    _fake_lce_module(
        monkeypatch,
        _ChunkContext=object,
        _mm_fp32_out_available=fp32_out,
        _mm_accumulates_in_fp32=lambda dtype, device: dtype == torch.float32,
    )
    assert lce.op_decisions(case("bfloat16")) == {
        "source": lce.CHUNK_CONTEXT.rpartition(".")[0],
        "_mm_fp32_out_available": True,
        "_mm_accumulates_in_fp32": False,
    }
    assert seen == [(torch.bfloat16, torch.device("cpu"))]
    # Only one of them: not the PR's API, so the context is read instead (and fails).
    _fake_lce_module(monkeypatch, _mm_fp32_out_available=fp32_out)
    assert "unavailable" in lce.op_decisions(case("bfloat16"))


def test_op_decisions_unavailable(monkeypatch):
    _fake_lce_module(monkeypatch)
    found = lce.op_decisions(case())
    assert found == {
        "unavailable": f"{lce.CHUNK_CONTEXT.rpartition('.')[0]} has neither "
        f"{', '.join(lce.MM_PREDICATES)} nor _ChunkContext"
    }

    class Broken:
        @classmethod
        def build(cls, **kwargs):
            raise TypeError("unexpected keyword argument 'linear_bias'")

    _fake_lce_module(monkeypatch, _ChunkContext=Broken)
    found = lce.op_decisions(case("bfloat16"))
    assert found["unavailable"].startswith(f"{lce.CHUNK_CONTEXT}.build: TypeError")

    def raising(dtype, device):
        raise RuntimeError("no device")

    _fake_lce_module(monkeypatch, _mm_fp32_out_available=raising, _mm_accumulates_in_fp32=raising)
    assert "RuntimeError: no device" in lce.op_decisions(case())["unavailable"]
    monkeypatch.setitem(sys.modules, lce.CHUNK_CONTEXT.rpartition(".")[0], None)
    assert "ModuleNotFoundError" in lce.op_decisions(case())["unavailable"]


def test_options_unresolved_without_adjust(monkeypatch):
    class Plain:
        acc_policy, acc_dtype, chunking_method, batch_chunk_size = "auto", None, "auto", None

    monkeypatch.setattr(lce, "_lce_options", lambda: Plain())
    assert lce.resolved_options(case()) == {
        "acc_policy": "auto",
        "acc_dtype": None,
        "chunking_method": "auto",
        "batch_chunk_size": None,
        "resolved": False,
    }


def test_source_hash_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(torch, "__path__", [str(tmp_path)])
    assert lce.lce_source_md5() == {"lce_source_md5": None, "lce_source_files": []}


def test_verify_on_cpu_records_no_override():
    c = case("bfloat16")
    outputs = []
    for name in ["torch_lce", "torch_lce_no_override"]:
        impl = SPEC.implementation(name)
        impl.setup(c)
        x = inputs(c)
        result = impl.verify(lambda impl=impl, x=x: outputs.append(impl.run(x, "fwd_bwd")))
        assert result == {"override_ran": False, "override_events": [], "native_events": []}
    assert len(outputs) == 2


def test_verify_raises_when_eligible_but_absent(monkeypatch):
    c = case("bfloat16", device="cuda")
    monkeypatch.setitem(lce._STATE, "case", c)
    monkeypatch.setitem(lce._STATE, "options", {"acc_policy": "compact", "acc_dtype": "float32"})
    monkeypatch.setitem(lce._STATE, "registration", {"registered": True, "reason": None})
    monkeypatch.setattr(lce, "override_eligible", lambda case, options: True)
    scatter = "_native::scatter_add__cutedsl_10"
    monkeypatch.setattr(lce, "native_events", lambda fn: [scatter])
    with pytest.raises(AssertionError, match="although eligible .*; it is registered, yet did not"):
        lce.verify_lce(lambda: None)
    reason = "disabled by TORCH_DISABLE_NATIVE_JIT=1"
    monkeypatch.setitem(lce._STATE, "registration", {"registered": False, "reason": reason})
    with pytest.raises(AssertionError, match=f"; not registered: {reason}; the timing"):
        lce.verify_lce(lambda: None)
    unknown = {"registered": None, "reason": "cannot tell: RuntimeError: boom"}
    monkeypatch.setitem(lce._STATE, "registration", unknown)
    with pytest.raises(AssertionError, match="; registration unknown: RuntimeError: boom; the"):
        lce.verify_lce(lambda: None)
    names = ["_native::_linear_cross_entropy_batch_chunked_cutedsl_2"]
    monkeypatch.setattr(lce, "native_events", lambda fn: [*names, scatter])
    assert lce.verify_lce(lambda: None) == {
        "override_ran": True,
        "override_events": names,
        "native_events": [*names, scatter],
    }
    with pytest.raises(AssertionError, match="ran although disabled"):
        lce.verify_lce_no_override(lambda: None)
    # Other torch._native overrides are recorded, not mistaken for LCE's.
    monkeypatch.setattr(lce, "native_events", lambda fn: [scatter])
    assert lce.verify_lce_no_override(lambda: None) == {
        "override_ran": False,
        "override_events": [],
        "native_events": [scatter],
    }


def test_override_registration(monkeypatch):
    from torch._native import cutedsl_utils as cu

    absent = {
        "present": False,
        "registered": False,
        "reason": "the build has no torch._native LCE override",
    }
    expected = {"present": True, "registered": True, "reason": None}
    if not OVERRIDE:  # e.g. torch 2.14.1, the CPU wheel of CI
        expected = absent
    elif not cu.runtime_available():
        expected = {
            **expected,
            "registered": False,
            "reason": lce.override_registration()["reason"],
        }
        assert expected["reason"].startswith("CuTeDSL runtime not available")
    assert lce.override_registration() == expected

    monkeypatch.setattr(lce, "lce_override_present", lambda: False)
    monkeypatch.setattr(cu, "runtime_available", lambda: True)
    assert lce.override_registration() == absent
    monkeypatch.setattr(lce, "lce_override_present", lambda: True)
    monkeypatch.setattr(cu, "runtime_available", lambda: True)
    monkeypatch.setattr(cu, "check_native_jit_disabled", lambda: False)
    monkeypatch.setattr(cu, "_version_is_ok", lambda: True)
    assert lce.override_registration() == {"present": True, "registered": True, "reason": None}
    monkeypatch.setattr(cu, "_version_is_ok", lambda: False)
    monkeypatch.setattr(cu, "runtime_version", lambda: "4.5.0")
    assert lce.override_registration()["reason"] == (
        "nvidia-cutlass-dsl 4.5.0 is not a known-good version "
        "(TORCH_NATIVE_SKIP_VERSION_CHECK=1 overrides)"
    )
    monkeypatch.setattr(cu, "check_native_jit_disabled", lambda: True)
    assert lce.override_registration() == {
        "present": True,
        "registered": False,
        "reason": "disabled by TORCH_DISABLE_NATIVE_JIT=1",
    }
    monkeypatch.setattr(cu, "runtime_available", lambda: False)
    assert lce.override_registration()["reason"].startswith("CuTeDSL runtime not available")

    def broken():
        raise RuntimeError("boom")

    monkeypatch.setattr(cu, "runtime_available", broken)
    assert lce.override_registration() == {
        "present": True,
        "registered": None,
        "reason": "cannot tell: RuntimeError: boom",
    }
    import torch._native

    monkeypatch.delattr(torch._native, "cutedsl_utils")
    monkeypatch.setitem(sys.modules, "torch._native.cutedsl_utils", None)
    result = lce.override_registration()
    assert result["present"] is True
    assert result["registered"] is None
    assert result["reason"].startswith(
        "cannot tell: torch._native.cutedsl_utils: ModuleNotFoundError"
    )


def test_override_eligible(monkeypatch):
    compact = {"acc_policy": "compact", "acc_dtype": "float32"}
    assert not lce.override_eligible(case("bfloat16"), compact)
    assert not lce.override_eligible(case("float32", device="cuda"), compact)
    for capability, expected in [((7, 5), False), ((8, 0), True), ((12, 0), True)]:
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d, c=capability: c)
        assert lce.override_eligible(case("bfloat16", device="cuda"), compact) is expected
        assert lce.override_eligible(case("float16", device="cuda"), compact) is expected
    accurate = {"acc_policy": "accurate", "acc_dtype": "float32"}
    assert not lce.override_eligible(case("bfloat16", device="cuda"), accurate)


def test_override_present(monkeypatch):
    import importlib.util
    from pathlib import Path

    on_disk = any(
        (Path(root) / "_native" / "ops" / "linear_cross_entropy" / "__init__.py").exists()
        for root in torch.__path__
    )
    assert OVERRIDE is on_disk
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert lce.lce_override_present() is False

    def no_parent(name):
        raise ModuleNotFoundError("No module named 'torch._native'")

    monkeypatch.setattr(importlib.util, "find_spec", no_parent)
    assert lce.lce_override_present() is False
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert lce.lce_override_present() is True


SM75 = "torch's linear_cross_entropy CuTeDSL path needs sm_80 or newer; Tesla T4 is sm_75"


def _plant_device(monkeypatch, capability, name="Tesla T4"):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: name)


@pytest.mark.parametrize("name", ["torch_lce", "torch_lce_no_override"])
def test_setup_below_sm80_is_unavailable(monkeypatch, name):
    from torch_op_bench.spec import Unavailable

    disabled = []
    monkeypatch.setattr(
        torch.backends.python_native, "disable_operations", lambda *ops: disabled.append(ops)
    )
    setup = SPEC.implementation(name).setup
    _plant_device(monkeypatch, (7, 5))
    with pytest.raises(Unavailable) as raised:
        setup(case("bfloat16", device="cuda"))
    assert raised.value.reason == SM75 == str(raised.value)
    assert disabled == []  # refused before the process-wide disabling
    # One check per setup (R1-19 O3), the one before the disabling.
    calls = []
    real = lce.check_device
    monkeypatch.setattr(lce, "check_device", lambda c: calls.append(c) or real(c))
    setup(case("bfloat16"))
    assert len(calls) == 1
    # CPU and sm_80+ pass the check (the rest of setup is tested elsewhere).
    lce.check_device(case("bfloat16"))
    _plant_device(monkeypatch, (8, 0), "NVIDIA A100")
    lce.check_device(case("bfloat16", device="cuda"))
    # Availability of the entry as such is not device-bound.
    assert lce.lce_no_override_available() is True


def test_availability(monkeypatch):
    assert lce.lce_available() is True
    assert lce.lce_no_override_available() is True
    absent = (
        f"torch {torch.__version__} has no torch._native LCE override "
        "(torch/_native/ops/linear_cross_entropy)"
    )
    assert lce.lce_override_available() == (True if OVERRIDE else absent)
    assert SPEC.implementation("torch_lce").available is lce.lce_override_available
    monkeypatch.setattr(lce, "lce_override_present", lambda: False)
    assert lce.lce_override_available() == absent
    assert lce.lce_no_override_available() is True
    monkeypatch.setattr(lce, "lce_override_present", lambda: True)
    assert lce.lce_override_available() is True
    assert lce.compile_available() is True
    monkeypatch.delattr(torch.nn.functional, "linear_cross_entropy")
    reason = lce.lce_available()
    assert reason.startswith(f"torch {torch.__version__} lacks")
    assert "torch.nn.functional.linear_cross_entropy" in reason
    assert lce.lce_no_override_available() == reason
    assert lce.lce_override_available() == reason
    monkeypatch.undo()
    import types

    monkeypatch.setattr(torch, "backends", types.SimpleNamespace())
    assert "python_native" in lce.lce_no_override_available()
    monkeypatch.undo()
    import importlib

    dynamo = importlib.import_module("torch._dynamo")
    monkeypatch.setattr(dynamo, "is_dynamo_supported", lambda: False)
    assert "not supported" in lce.compile_available()
    monkeypatch.delattr(torch, "compile")
    assert "has no torch.compile" in lce.compile_available()


def test_work():
    assert SPEC.work(case()) == {"tokens": 64, "flops": 6 * 64 * 32 * 96}


def test_reference_fits(monkeypatch):
    assert SPEC.reference_fits(case()) is True
    c = case(device="cuda", N=4096, H=4096, V=131072)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1024**3, 16 * 1024**3))
    reason = SPEC.reference_fits(c)
    assert reason == "reference does not fit: 10.1 GiB needed, 1.0 GiB free (80 % usable)"
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (80 * 1024**3, 0))
    assert SPEC.reference_fits(c) is True


def test_through_worker_and_runner(tmp_path):
    cases = runner.enumerate_cases(
        SPEC, device="cpu", dtypes=["float32"], cases=[{**TINY, "ignore_frac": "0.25"}]
    )
    runs = runner.enumerate_runs(SPEC, cases, IMPLS)
    assert len(runs) == 8
    written = runner.run_sweep(
        SPEC,
        "lce",
        runs,
        settings=runner.Settings(warmup=1, repeat=2, timeout=600),
        results_dir=tmp_path,
        slug="cpu",
        env_map=runner.parse_env_map(),
    )
    assert [(r["implementation"], r["mode"]) for r in written] == [
        (impl, mode) for impl in IMPLS for mode in ("fwd", "fwd_bwd")
    ]
    for record in written:
        assert record["work"] == {"tokens": 64, "flops": 6 * 64 * 32 * 96}
        assert record["info"]["ignore_index"] == -100
        if record["implementation"] == "torch_lce" and not OVERRIDE:
            assert record["status"] == "unavailable"
            assert "has no torch._native LCE override" in record["reason"]
            continue
        assert record["status"] == "ok", (record["implementation"], record["reason"])
        assert record["parity"]["status"] == "ok", record["parity"]
    by_impl = {r["implementation"]: r for r in written if r["mode"] == "fwd_bwd"}
    assert by_impl["torch_compiled"]["info"]["compiled"] is True
    no_override = by_impl["torch_lce_no_override"]["info"]
    assert no_override["override_disabled"] == list(lce.OVERRIDE_OPS)
    assert no_override["override_present"] is OVERRIDE
    assert no_override["override_registration"]["present"] is OVERRIDE
    names = ["torch_lce_no_override"]
    if OVERRIDE:
        names.append("torch_lce")
        assert by_impl["torch_lce"]["verify"] == {
            "override_ran": False,
            "override_events": [],
            "native_events": [],
        }
        assert len(by_impl["torch_lce"]["info"]["lce_source_md5"]) == 32
    for name in names:
        assert set(by_impl[name]["env"]["packages"]) == {"nvidia-cutlass-dsl", "apache-tvm-ffi"}
        assert "override_registration" in by_impl[name]["info"]
    assert len(records.read(tmp_path / "lce" / "cpu")) == 8


def test_worker_payload_in_process(monkeypatch):
    # In process, so the override's presence can be planted: the eager path of
    # torch_lce runs on CPU whether or not the build has it.
    monkeypatch.setattr(lce, "lce_override_present", lambda: True)
    c = case("bfloat16", "0.25")
    payload = runner.make_payload(
        SPEC,
        "lce",
        c,
        SPEC.implementation("torch_lce"),
        "fwd_bwd",
        runner.Settings(1, 2),
        slug="cpu",
    )
    record = worker.run_payload(payload)
    assert record["status"] == "ok", record["traceback"]
    assert record["parity"]["status"] == "ok"
    assert record["info"]["options"]["resolved"] is True


def test_list(capsys):
    assert cli.main(["list", "lce"]) == 0
    out = capsys.readouterr().out
    for name in IMPLS:
        if name == "torch_lce" and not OVERRIDE:
            assert f"  {name}  modes=fwd,fwd_bwd  env=torch  unavailable: torch " in out
            assert "has no torch._native LCE override" in out
        else:
            assert f"  {name}  modes=fwd,fwd_bwd  env=torch  available" in out
    assert "default dtypes: bfloat16" in out
    not_checked = "not checked (environment 'liger')"
    assert f"  liger  modes=fwd_bwd  env=liger  {not_checked}" in out
    assert f"  liger_native_accum  modes=fwd_bwd  env=liger  {not_checked}" in out
    assert (
        f"  torch_eager_liger (control of torch_eager)  modes=fwd,fwd_bwd  env=liger  {not_checked}"
    ) in out
    not_checked = "not checked (environment 'torchtitan')"
    for name in ["torchtitan_chunked", "torchtitan_chunked_compiled"]:
        assert f"  {name}  modes=fwd_bwd  env=torchtitan  {not_checked}" in out
    assert f"  torchtitan_ce  modes=fwd,fwd_bwd  env=torchtitan  {not_checked}" in out
    control = "torch_eager_torchtitan (control of torch_eager)"
    assert f"  {control}  modes=fwd,fwd_bwd  env=torchtitan  {not_checked}" in out


@pytest.mark.parametrize(
    "N, H, V, expected",
    [
        (4096, 4096, 131072, (32, 128, 32)),  # the contract's default point
        (1024, 1024, 32768, (32, 32, 32)),
        (4096, 4096, 65536, (16, 256, 16)),
        (1000, 4096, 2048, (1, 1000, 1)),  # V < H: one chunk of all tokens
        (100, 64, 640, (10, 16, 7)),  # rounded up to a power of two
        (100, 64, 192, (3, 64, 2)),  # the last chunk is shorter
    ],
)
def test_liger_chunks(N, H, V, expected):
    chunks = lce.liger_chunks(N, H, V)
    assert (chunks["inc_factor"], chunks["chunk_size"], chunks["num_chunks"]) == expected


def test_liger_chunks_mem_const():
    assert lce.liger_chunks(4096, 4096, 131072, chunk_mem_const=4) == {
        "inc_factor": 8,
        "chunk_size": 512,
        "num_chunks": 8,
    }


def test_liger_setup(monkeypatch):
    monkeypatch.setattr(lce, "liger_version", lambda: "0.8.4")
    c = case("bfloat16", N=4096, H=4096, V=131072)
    expected = {
        "chunk_mem_const": 1,
        "inc_factor": 32,
        "chunk_size": 128,
        "num_chunks": 32,
        "liger_kernel": "0.8.4",
        "ce_impl": "triton",
    }
    assert SPEC.implementation("liger").setup(c) == {**expected, "accum_dtype": "float32"}
    setup = SPEC.implementation("liger_native_accum").setup
    assert setup(c) == {**expected, "accum_dtype": None}


def test_liger_version(monkeypatch):
    def missing(name):
        raise lce.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(lce.importlib.metadata, "version", missing)
    assert lce.liger_version() is None
    monkeypatch.setattr(lce.importlib.metadata, "version", lambda name: f"{name} 1.0")
    assert lce.liger_version() == "liger-kernel 1.0"


def _fake_liger(monkeypatch, fn):
    import types

    for name in ["liger_kernel", "liger_kernel.transformers"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    functional = types.ModuleType("liger_kernel.transformers.functional")
    functional.liger_fused_linear_cross_entropy = fn
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers.functional", functional)
    return functional


def test_liger_available(monkeypatch):
    monkeypatch.setattr(lce, "liger_version", lambda: None)
    monkeypatch.setitem(sys.modules, "liger_kernel", None)
    reason = lce.liger_available()
    assert reason.startswith(
        f"liger_kernel.transformers.functional cannot be imported "
        f"(liger-kernel None, torch {torch.__version__}): "
    )

    def with_accum(input, weight, target, ignore_index=-100, accum_dtype=None):
        return None

    def without_accum(input, weight, target, ignore_index=-100):
        return None

    monkeypatch.setattr(lce, "liger_version", lambda: "0.1.0")
    _fake_liger(monkeypatch, without_accum)
    assert lce.liger_available() == (
        "liger-kernel 0.1.0: liger_fused_linear_cross_entropy takes no accum_dtype"
    )
    _fake_liger(monkeypatch, with_accum)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert lce.liger_available() == "liger-kernel needs a CUDA device"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert lce.liger_available() is True


def test_liger_run_passes_the_contract(monkeypatch):
    calls = []

    def fake(input, weight, target, ignore_index, reduction, accum_dtype):
        calls.append((ignore_index, reduction, accum_dtype))
        return lce._eager_loss(input, weight, target)

    _fake_liger(monkeypatch, fake)
    c = case("float32", "0.25")
    x = inputs(c)
    for name in ["liger", "liger_native_accum"]:
        outputs = SPEC.implementation(name).run(x, "fwd_bwd")
        reference = SPEC.reference.run(x, "fwd_bwd")
        assert SPEC.compare(outputs, reference, c)["loss_rel_err"] < 1e-6
        clear_grads(x)
    assert calls == [(-100, "mean", torch.float32), (-100, "mean", None)]


def test_liger_verify(monkeypatch):
    seen = []
    events = [
        ("liger_cross_entropy_kernel", True),
        ("aten::mm", False),
        ("_liger_grad_kernel", True),
        ("LigerFusedLinearCrossEntropyFunction", False),  # CPU side: not a kernel
        ("element_mul_kernel", True),
        ("cross_entropy_loss_forward_kernel", True),
        ("_native::scatter_add__cutedsl_10", False),
    ]

    def fake_run(fn):
        seen.append("profiled")
        fn()
        return events

    monkeypatch.setattr(lce, "profile_run", fake_run)
    assert lce.verify_liger(lambda: seen.append("ran")) == {
        "liger_events": ["_liger_grad_kernel", "liger_cross_entropy_kernel"],
        "native_events": ["_native::scatter_add__cutedsl_10"],
    }
    assert seen == ["profiled", "ran"]  # one profiled run serves both filters
    # A non-Liger cross_entropy kernel alone does not pass.
    other = [("cross_entropy_loss_forward_kernel", True), ("nll_loss_forward_kernel_2d", True)]
    monkeypatch.setattr(lce, "profile_run", lambda fn: other)
    with pytest.raises(AssertionError, match="no Liger kernel .* ran"):
        lce.verify_liger(lambda: None)
    monkeypatch.setattr(lce, "profile_run", lambda fn: [])
    with pytest.raises(AssertionError, match="no Liger kernel .* ran"):
        lce.verify_liger(lambda: None)


def test_select():
    events = [("a::x", False), ("b_kernel", True), ("a::y", True), ("a::x", True)]
    assert lce.select(events, lambda name: name.startswith("a::")) == ["a::x", "a::y"]
    assert lce.select(events, lambda name: name.startswith("a::"), kernels_only=True) == [
        "a::x",
        "a::y",
    ]
    assert lce.select(events, lambda name: "kernel" in name, kernels_only=True) == ["b_kernel"]
    assert lce.select(events[:1], bool, kernels_only=True) == []


def test_verify_native(monkeypatch):
    monkeypatch.setattr(lce, "profile_run", lambda fn: [("_native::op_cutedsl_1", False)])
    assert lce.verify_native(lambda: None) == {"native_events": ["_native::op_cutedsl_1"]}
    monkeypatch.setattr(lce, "profile_run", lambda fn: [("aten::mm", False)])
    assert lce.verify_native(lambda: None) == {"native_events": []}  # empty is no error


@pytest.mark.parametrize("name", ["torch_eager", "torch_compiled", "torch_eager_liger"])
def test_baselines_verify_native_events(name):
    impl = SPEC.implementation(name)
    assert impl.verify is lce.verify_native
    c = case("float32", "0.25")
    payload = runner.make_payload(
        SPEC, "lce", c, impl, "fwd_bwd", runner.Settings(1, 2), slug="cpu"
    )
    record = worker.run_payload(payload)
    assert record["status"] == "ok", record["traceback"]
    native = record["verify"]["native_events"]
    assert isinstance(native, list) and all(e.startswith("_native::") for e in native)


def test_profiled_events_on_cpu():
    c = case()
    x = inputs(c)
    events = lce.profiled_events(lambda: lce.run_eager(x, "fwd"), lambda name: "aten::" in name)
    assert "aten::linear" in events or "aten::mm" in events
    assert events == sorted(set(events))
    assert lce.profiled_events(lambda: lce.run_eager(x, "fwd"), bool, kernels_only=True) == []


def test_liger_entries_through_runner_on_cpu(tmp_path):
    cases = runner.enumerate_cases(
        SPEC, device="cpu", dtypes=["float32"], cases=[{**TINY, "ignore_frac": "0.25"}]
    )
    written = runner.run_sweep(
        SPEC,
        "lce",
        runner.enumerate_runs(SPEC, cases, LIGER_IMPLS, ["fwd_bwd"]),
        settings=runner.Settings(warmup=1, repeat=2, timeout=600),
        results_dir=tmp_path,
        slug="cpu",
        env_map=runner.parse_env_map([f"liger={sys.executable}"]),
    )
    by_impl = {r["implementation"]: r for r in written}
    assert list(by_impl) == LIGER_IMPLS
    for name in ["liger", "liger_native_accum"]:
        record = by_impl[name]
        assert record["status"] == "unavailable" and record["env_name"] == "liger"
        try:
            import liger_kernel  # noqa: F401
        except ImportError:
            assert "liger_kernel.transformers.functional cannot be imported" in record["reason"]
        else:  # a torch environment with Liger installed (§4.2)
            assert record["reason"] == "liger-kernel needs a CUDA device"
    control = by_impl["torch_eager_liger"]
    assert control["status"] == "ok", control["reason"]
    assert control["env_name"] == "liger" and control["parity"]["status"] == "ok"
    assert control["info"]["role"] == "control"
    assert control["info"]["control_of"] == "torch_eager"


def _fake_torchtitan(monkeypatch, api="compile_config"):
    """A stand-in for torchtitan's losses with either API: the plain loss over
    ``pred[..., V]`` (summed, divided by the count), the wrapper over its lm_head;
    what they were given is recorded in ``calls``."""
    import dataclasses
    import types

    import torch.nn.functional as F

    calls = {"compile": [], "pred_shapes": [], "lm_heads": []}

    def loss(pred, labels, count):
        calls["pred_shapes"].append(tuple(pred.shape))
        total = F.cross_entropy(
            pred.flatten(0, -2).float(), labels.flatten(), reduction="sum", ignore_index=-100
        )
        return total / count, {}

    @dataclasses.dataclass
    class CEConfig:
        global_vocab_size: int | None = None

    @dataclasses.dataclass
    class ChunkedConfig:
        num_chunks: int = 8

    def init(self, config, **kwargs):
        if api == "compile_config":
            calls["compile"].append(kwargs.get("compile_config"))
        else:
            assert not kwargs
        self.config = config

    def set_lm_head(self, module):
        calls["lm_heads"].append(module)
        self.lm_head = module

    def chunked_call(self, pred, labels, count):
        return loss(self.lm_head(pred), labels, count)

    if api == "compile_config":
        init_signature = lambda self, config, *, compile_config=None: None  # noqa: E731
    else:
        init_signature = lambda self, config: None  # noqa: E731

    def cls(name, config, call, **extra):
        def __init__(self, config, **kwargs):
            init(self, config, **kwargs)

        __init__.__signature__ = lce.inspect.signature(init_signature)
        return type(name, (), {"Config": config, "__init__": __init__, "__call__": call, **extra})

    module = types.ModuleType("torchtitan.components.loss")
    module.CrossEntropyLoss = cls(
        "CrossEntropyLoss", CEConfig, lambda self, pred, labels, count: loss(pred, labels, count)
    )
    module.ChunkedLossWrapper = cls(
        "ChunkedLossWrapper", ChunkedConfig, chunked_call, set_lm_head=set_lm_head
    )
    modules = {"torchtitan.components.loss": module}
    for name in ["torchtitan", "torchtitan.components", "torchtitan.distributed"]:
        modules[name] = types.ModuleType(name)
    if api == "compile_config":
        config = types.ModuleType("torchtitan.config")

        @dataclasses.dataclass
        class CompileConfig:
            enable: bool = False
            components: list = dataclasses.field(default_factory=lambda: ["model", "loss"])
            backend: str = "inductor"

        config.CompileConfig = CompileConfig
        utils = types.ModuleType("torchtitan.distributed.utils")
        utils.get_spmd_backend = lambda: "spmd_types"
        modules.update({"torchtitan.config": config, "torchtitan.distributed.utils": utils})
    else:
        local = types.ModuleType("torchtitan.distributed.local_compile")
        local.apply_local_compile = calls["compile"].append
        modules["torchtitan.distributed.local_compile"] = local
        modules["torchtitan.distributed.utils"] = None  # main: no get_spmd_backend
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(lce, "torchtitan_version", lambda: "0.3.0")
    return calls


def test_torchtitan_version(monkeypatch):
    def missing(name):
        raise lce.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(lce.importlib.metadata, "version", missing)
    assert lce.torchtitan_version() is None
    monkeypatch.setattr(lce.importlib.metadata, "version", lambda name: f"{name} 1.0")
    assert lce.torchtitan_version() == "torchtitan 1.0"


def test_torchtitan_available(monkeypatch):
    monkeypatch.setattr(lce, "torchtitan_version", lambda: None)
    monkeypatch.setitem(sys.modules, "torchtitan", None)
    reason = lce.torchtitan_available()
    assert reason.startswith(
        f"torchtitan.components.loss cannot be imported "
        f"(torchtitan None, torch {torch.__version__}): "
    )
    assert lce.torchtitan_compiled_available() == reason
    _fake_torchtitan(monkeypatch)
    assert lce.torchtitan_available() is True
    assert lce.torchtitan_compiled_available() is True
    import importlib

    dynamo = importlib.import_module("torch._dynamo")
    monkeypatch.setattr(dynamo, "is_dynamo_supported", lambda: False)
    assert "not supported" in lce.torchtitan_compiled_available()
    assert lce.torchtitan_available() is True


def test_torchtitan_api(monkeypatch):
    _fake_torchtitan(monkeypatch, "compile_config")
    assert lce.torchtitan_api() == "compile_config"
    _fake_torchtitan(monkeypatch, "local_compile")
    assert lce.torchtitan_api() == "local_compile"


@pytest.mark.parametrize("api", ["compile_config", "local_compile"])
def test_torchtitan_setup(monkeypatch, api):
    calls = _fake_torchtitan(monkeypatch, api)
    c = case("bfloat16", N=4096, H=4096, V=131072)
    common = {
        "torchtitan": "0.3.0",
        "torchtitan_api": api,
        "spmd_backend": "spmd_types" if api == "compile_config" else None,
        "logits_dtype": "float32",
        "loss_dtype": "float32",
    }
    chunks = {"num_chunks": 8, "chunk_tokens": 512, "grad_accum_dtype": "float32"}
    assert SPEC.implementation("torchtitan_chunked").setup(c) == {
        **common,
        **chunks,
        "compiled": False,
    }
    compile = (
        "torch.compile(backend='inductor')"
        if api == "compile_config"
        else "torch.compile(fullgraph=True)"
    )
    assert SPEC.implementation("torchtitan_chunked_compiled").setup(c) == {
        **common,
        **chunks,
        "compiled": True,
        "compile_scope": "inner loss_fn",
        "compile": compile,
    }
    assert SPEC.implementation("torchtitan_ce").setup(c) == {**common, "compiled": False}
    if api == "compile_config":
        assert calls["compile"][0] is None and calls["compile"][2] is None
        assert calls["compile"][1].enable and calls["compile"][1].components == ["loss"]
    else:
        assert calls["compile"] == [[], ["loss"], []]
    assert type(lce._STATE["titan_loss"]).__name__ == "CrossEntropyLoss"


@pytest.mark.parametrize("name", ["torchtitan_chunked", "torchtitan_chunked_compiled"])
def test_torchtitan_setup_rejects_uneven_chunks(monkeypatch, name):
    _fake_torchtitan(monkeypatch)
    with pytest.raises(ValueError, match="N=60 is not divisible by num_chunks=8"):
        SPEC.implementation(name).setup(case(N=60))
    assert SPEC.implementation("torchtitan_ce").setup(case(N=60))["compiled"] is False


@pytest.mark.parametrize("api, layout", [("compile_config", (1, 64)), ("local_compile", (64,))])
@pytest.mark.parametrize("name", TITAN)
def test_torchtitan_run_passes_the_contract(monkeypatch, api, layout, name):
    calls = _fake_torchtitan(monkeypatch, api)
    c = case("float32", "0.25")
    impl = SPEC.implementation(name)
    impl.setup(c)
    x = inputs(c)
    outputs = impl.run(x, "fwd_bwd")
    reference = SPEC.reference.run(x, "fwd_bwd")
    metrics = SPEC.compare(outputs, reference, c)
    assert max(metrics.values()) < 1e-5, metrics  # the mean over non-ignored tokens
    # torchtitan's layout: the wrapper gets the hidden states, the plain loss the logits.
    assert calls["pred_shapes"] == [(*layout, 96)]
    if name != "torchtitan_ce":
        # The lm_head computes over the case's own weight, which is not copied.
        (head,) = calls["lm_heads"]
        assert head.weight is x[1] and not list(head.parameters())
        clear_grads(x)
        impl.run(x, "fwd_bwd")
        assert calls["lm_heads"][1] is head  # cached per inputs
        impl.run(inputs(c), "fwd_bwd")
        assert calls["lm_heads"][2] is not head
    else:
        clear_grads(x)
        (loss,) = impl.run(x, "fwd")
        assert not loss.requires_grad
        torch.testing.assert_close(loss, outputs[0])


def test_verify_torchtitan_compiled(monkeypatch):
    monkeypatch.setattr(lce, "profile_run", lambda fn: [("aten::mm", False)])
    with pytest.raises(AssertionError, match="no event shows the compiled loss ran"):
        lce.verify_torchtitan_compiled(lambda: None)
    # A triton_ name counts only as a device kernel.
    monkeypatch.setattr(lce, "profile_run", lambda fn: [("triton_poi_fused_0", False)])
    with pytest.raises(AssertionError, match="no event shows the compiled loss ran"):
        lce.verify_torchtitan_compiled(lambda: None)
    events = [
        ("CompiledFunction", False),
        ("CompiledFunctionBackward", False),
        ("triton_red_fused_1", True),
        ("aten::mm", False),
        ("_native::op_cutedsl_1", False),
    ]
    monkeypatch.setattr(lce, "profile_run", lambda fn: events)
    assert lce.verify_torchtitan_compiled(lambda: None) == {
        "compiled_events": ["CompiledFunction", "CompiledFunctionBackward", "triton_red_fused_1"],
        "native_events": ["_native::op_cutedsl_1"],
    }


def test_torchtitan_entries_through_runner_on_cpu(tmp_path):
    cases = runner.enumerate_cases(
        SPEC, device="cpu", dtypes=["float32"], cases=[{**TINY, "ignore_frac": "0.25"}]
    )
    written = runner.run_sweep(
        SPEC,
        "lce",
        runner.enumerate_runs(SPEC, cases, TITAN_IMPLS, ["fwd_bwd"]),
        settings=runner.Settings(warmup=1, repeat=2, timeout=600),
        results_dir=tmp_path,
        slug="cpu",
        env_map=runner.parse_env_map([f"torchtitan={sys.executable}"]),
    )
    by_impl = {r["implementation"]: r for r in written}
    assert list(by_impl) == TITAN_IMPLS
    try:
        import torchtitan.components.loss  # noqa: F401
    except ImportError:
        for name in TITAN:
            record = by_impl[name]
            assert record["status"] == "unavailable" and record["env_name"] == "torchtitan"
            assert "torchtitan.components.loss cannot be imported" in record["reason"]
    else:  # run from a torchtitan environment
        for name in TITAN:
            assert by_impl[name]["status"] == "ok", by_impl[name]["reason"]
    control = by_impl["torch_eager_torchtitan"]
    assert control["status"] == "ok", control["reason"]
    assert control["env_name"] == "torchtitan" and control["parity"]["status"] == "ok"
    assert control["info"]["role"] == "control"
    assert control["info"]["control_of"] == "torch_eager"
    assert isinstance(control["verify"]["native_events"], list)


def test_parent_does_not_import_torch():
    import subprocess

    code = (
        "import sys; from torch_op_bench.spec import resolve_op; resolve_op('lce');"
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.cuda
def test_cuda_small_case_through_runner(tmp_path):
    cases = runner.enumerate_cases(
        SPEC, device="cuda", cases=[{"N": 1024, "H": 1024, "V": 32768, "ignore_frac": "0"}]
    )
    assert [c.dtype for c in cases] == ["bfloat16"]
    written = runner.run_sweep(
        SPEC,
        "lce",
        runner.enumerate_runs(SPEC, cases, IMPLS),
        settings=runner.Settings(warmup=2, repeat=5, timeout=900),
        results_dir=tmp_path,
        slug="gpu",
        env_map=runner.parse_env_map(),
    )
    assert len(written) == 8
    eligible = torch.cuda.get_device_capability() >= (8, 0)
    for record in written:
        name, parity = record["implementation"], record["parity"]
        assert record["status"] == "ok", (name, record["reason"])
        if name == "torch_lce_no_override":
            # A finding, not a defect of the spec: with the override disabled, the eager
            # accumulator's bf16 loss is ~2e-2 off the fp32 reference on CUDA (its
            # gradients are not); see Handoff I1-2. Only the loss may fail.
            assert parity["failed"] in ([], ["loss_rel_err"]), parity
        else:
            assert parity["status"] == "ok", (name, parity)
        assert record["memory"]["peak_bytes"] > record["memory"]["baseline_bytes"]
    fwd_bwd = {r["implementation"]: r for r in written if r["mode"] == "fwd_bwd"}
    assert fwd_bwd["torch_lce"]["verify"]["override_ran"] is eligible
    assert fwd_bwd["torch_lce_no_override"]["verify"]["override_ran"] is False
    for name in ["torch_lce", "torch_lce_no_override"]:
        verify = fwd_bwd[name]["verify"]
        assert set(verify["override_events"]) <= set(verify["native_events"])
        assert all(event.startswith("_native::") for event in verify["native_events"])
    peak = {name: r["memory"]["peak_bytes"] for name, r in fwd_bwd.items()}
    assert peak["torch_lce"] < peak["torch_eager"], peak


@pytest.mark.cuda
@pytest.mark.parametrize("name", ["torch_lce", "torch_lce_no_override"])
def test_cuda_below_sm80_record(monkeypatch, name):
    # ey has no sub-sm_80 device: the capability is planted, in process. CUDA is
    # initialised first: its lazy init reads the capability and warns about the
    # planted one, which -W error::UserWarning turns into an error when this test
    # runs alone.
    torch.cuda.init()
    monkeypatch.setattr(lce, "lce_override_present", lambda: True)
    _plant_device(monkeypatch, (7, 5))
    c = Case({"N": 1024, "H": 1024, "V": 32768, "ignore_frac": "0"}, "bfloat16", "cuda")
    payload = runner.make_payload(
        SPEC, "lce", c, SPEC.implementation(name), "fwd_bwd", runner.Settings(1, 2), slug="gpu"
    )
    record = worker.run_payload(payload)
    assert (record["status"], record["reason"], record["traceback"]) == ("unavailable", SM75, None)
    assert record["times_ms"] is None and record["parity"] is None


@pytest.mark.cuda
def test_cuda_override_not_registered_is_named(monkeypatch):
    # The override registers at import; TORCH_DISABLE_NATIVE_JIT=1 keeps it out, as a
    # missing or wrong nvidia-cutlass-dsl would on a fresh host.
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("the override is not eligible below sm_80")
    monkeypatch.setenv("TORCH_DISABLE_NATIVE_JIT", "1")
    c = Case({"N": 1024, "H": 1024, "V": 32768, "ignore_frac": "0"}, "bfloat16", "cuda")
    payload = runner.make_payload(
        SPEC, "lce", c, SPEC.implementation("torch_lce"), "fwd", runner.Settings(1, 2), slug="gpu"
    )
    record = runner.launch(payload, sys.executable, 600)
    assert record["status"] == "error"
    assert "not registered: disabled by TORCH_DISABLE_NATIVE_JIT=1" in record["reason"]
    assert record["env"]["packages"]["nvidia-cutlass-dsl"]


def _liger_environments():
    """(id, python) of the Liger environments present: ``TOB_LIGER_PYTHON``, or
    those scripts/setup-env.sh made under envs/.prefix of this checkout."""
    import os
    from pathlib import Path

    from test_envs import ROOT

    found = []
    if os.environ.get("TOB_LIGER_PYTHON"):
        found.append(pytest.param(os.environ["TOB_LIGER_PYTHON"], id="TOB_LIGER_PYTHON"))
    for name in ["liger", "liger-git"]:
        python = ROOT / "envs" / ".prefix" / name / "bin" / "python"
        if Path(python).exists():
            found.append(pytest.param(str(python), id=name))
    if not found:
        reason = "no Liger environment: TOB_LIGER_PYTHON unset, envs/.prefix/liger* absent"
        found.append(pytest.param(None, id="none", marks=pytest.mark.skip(reason=reason)))
    return found


@pytest.mark.cuda
@pytest.mark.parametrize("liger_python", _liger_environments())
def test_cuda_liger_through_runner(tmp_path, liger_python):
    from test_envs import liger_git_commit

    params = {"N": 1024, "H": 1024, "V": 32768, "ignore_frac": "0.1"}
    cases = runner.enumerate_cases(SPEC, device="cuda", cases=[params])
    names = ["liger", "liger_native_accum", "torch_eager_liger", "torch_eager"]
    written = runner.run_sweep(
        SPEC,
        "lce",
        runner.enumerate_runs(SPEC, cases, names),
        settings=runner.Settings(warmup=2, repeat=5, timeout=900),
        results_dir=tmp_path,
        slug="gpu",
        env_map=runner.parse_env_map([f"liger={liger_python}"]),
    )
    assert len(written) == 6
    for record in written:
        name = record["implementation"]
        assert record["status"] == "ok", (name, record["reason"])
        assert record["env_name"] == ("torch" if name == "torch_eager" else "liger")
    fwd_bwd = {r["implementation"]: r for r in written if r["mode"] == "fwd_bwd"}
    torch_build = fwd_bwd["torch_eager"]["env"]["torch"]["version"]
    chunks = lce.liger_chunks(1024, 1024, 32768)
    for name in ["liger", "liger_native_accum"]:
        record = fwd_bwd[name]
        assert "liger_cross_entropy_kernel" in record["verify"]["liger_events"]
        assert record["env"]["torch"]["version"] != torch_build  # the PyPI wheel
        assert {k: record["info"][k] for k in chunks} == chunks
        package = record["env"]["packages"]["liger-kernel"]
        if isinstance(package, dict):  # envs/liger-git.yml
            assert package["commit"] == liger_git_commit()
            assert "linkedin/Liger-Kernel" in package["url"]
        else:  # envs/liger.yml
            assert package == "0.8.4"
    assert fwd_bwd["liger"]["parity"]["status"] == "ok", fwd_bwd["liger"]["parity"]
    for record in written:
        if record["implementation"] == "torch_eager_liger":
            assert record["parity"]["status"] == "ok", record["parity"]
            assert record["info"]["role"] == "control"
    peak = {name: r["memory"]["peak_bytes"] for name, r in fwd_bwd.items()}
    # At this small N, liger's float32 weight-gradient accumulator (V*H*4 bytes)
    # outweighs eager's N*V logits; see Handoff I1-3. Its native accumulation does not.
    assert peak["liger_native_accum"] < peak["torch_eager_liger"], peak


def _torchtitan_environments():
    """(id, python) of the torchtitan environments present: ``TOB_TORCHTITAN_PYTHON``,
    or those scripts/setup-env.sh made under envs/.prefix of this checkout."""
    import os
    from pathlib import Path

    from test_envs import ROOT

    found = []
    if os.environ.get("TOB_TORCHTITAN_PYTHON"):
        python = os.environ["TOB_TORCHTITAN_PYTHON"]
        found.append(pytest.param(python, id="TOB_TORCHTITAN_PYTHON"))
    for name in ["torchtitan", "torchtitan-git"]:
        python = ROOT / "envs" / ".prefix" / name / "bin" / "python"
        if Path(python).exists():
            found.append(pytest.param(str(python), id=name))
    if not found:
        reason = (
            "no torchtitan environment: TOB_TORCHTITAN_PYTHON unset, "
            "envs/.prefix/torchtitan* absent"
        )
        found.append(pytest.param(None, id="none", marks=pytest.mark.skip(reason=reason)))
    return found


def _run_torchtitan(tmp_path, python, device, params, names, dtypes=None):
    cases = runner.enumerate_cases(SPEC, device=device, dtypes=dtypes, cases=[params])
    return runner.run_sweep(
        SPEC,
        "lce",
        runner.enumerate_runs(SPEC, cases, names),
        settings=runner.Settings(warmup=2, repeat=5, timeout=900),
        results_dir=tmp_path,
        slug=device,
        env_map=runner.parse_env_map([f"torchtitan={python}"]),
    )


@pytest.mark.parametrize("titan_python", _torchtitan_environments())
def test_torchtitan_on_cpu_through_runner(tmp_path, titan_python):
    params = {**TINY, "ignore_frac": "0.25"}
    written = _run_torchtitan(tmp_path, titan_python, "cpu", params, TITAN_IMPLS, ["float32"])
    # fwd for torchtitan_ce and the control, fwd_bwd for all four.
    assert len(written) == 6
    for record in written:
        name = record["implementation"]
        assert record["status"] == "ok", (name, record["reason"])
        assert record["env_name"] == "torchtitan"
        assert record["parity"]["status"] == "ok", (name, record["parity"])
        assert isinstance(record["verify"]["native_events"], list)
    fwd_bwd = {r["implementation"]: r for r in written if r["mode"] == "fwd_bwd"}
    for name in ["torchtitan_chunked", "torchtitan_chunked_compiled"]:
        assert fwd_bwd[name]["info"]["num_chunks"] == 8
        assert fwd_bwd[name]["info"]["chunk_tokens"] == 8
    assert fwd_bwd["torchtitan_chunked_compiled"]["verify"]["compiled_events"]


@pytest.mark.cuda
@pytest.mark.parametrize("titan_python", _torchtitan_environments())
def test_cuda_torchtitan_through_runner(tmp_path, titan_python):
    from test_envs import torchtitan_git_commit

    params = {"N": 1024, "H": 1024, "V": 32768, "ignore_frac": "0.1"}
    written = _run_torchtitan(tmp_path, titan_python, "cuda", params, TITAN_IMPLS)
    assert len(written) == 6
    for record in written:
        assert record["status"] == "ok", (record["implementation"], record["reason"])
    fwd_bwd = {r["implementation"]: r for r in written if r["mode"] == "fwd_bwd"}
    for name in TITAN:
        package = fwd_bwd[name]["env"]["packages"]["torchtitan"]
        if isinstance(package, dict):  # envs/torchtitan-git.yml
            assert package["commit"] == torchtitan_git_commit()
            assert "pytorch/torchtitan" in package["url"]
        else:  # envs/torchtitan.yml
            assert package == "0.3.0"
    assert fwd_bwd["torchtitan_chunked_compiled"]["verify"]["compiled_events"]
    for name in ["torchtitan_chunked", "torchtitan_chunked_compiled", "torchtitan_ce"]:
        # fp32 logits and sums: the loss is closer to the fp32 reference than eager bf16's.
        loss_err = fwd_bwd[name]["parity"]["metrics"]["loss_rel_err"]
        assert loss_err < fwd_bwd["torch_eager_torchtitan"]["parity"]["metrics"]["loss_rel_err"]
        # grad_weight accumulates in bf16 across the chunk backwards: recorded and
        # reported in the Handoff against the tolerance, not asserted.
        failed = fwd_bwd[name]["parity"]["failed"]
        assert set(failed) <= {"grad_weight_rel_l2"}, (name, fwd_bwd[name]["parity"])
    assert fwd_bwd["torch_eager_torchtitan"]["parity"]["status"] == "ok"
    peak = {name: r["memory"]["peak_bytes"] for name, r in fwd_bwd.items()}
    # Eight chunks of fp32 logits against the full bf16 logits plus their fp32 gradient.
    assert peak["torchtitan_chunked"] < peak["torch_eager_torchtitan"], peak
