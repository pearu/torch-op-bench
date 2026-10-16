import sys

import pytest
import torch

from torch_op_bench import cli, records, runner, worker
from torch_op_bench.measure import clear_grads
from torch_op_bench.ops import lce
from torch_op_bench.spec import Case, resolve_op

SPEC = lce.LCE
IMPLS = ["torch_eager", "torch_compiled", "torch_lce", "torch_lce_no_override"]
TINY = {"N": 64, "H": 32, "V": 96}


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
    assert [impl.name for impl in SPEC.implementations] == IMPLS
    assert SPEC.default_point == {"N": 4096, "H": 4096, "V": 131072, "ignore_frac": "0"}
    assert SPEC.default_dtypes == ["bfloat16"]
    for impl in [*SPEC.implementations, SPEC.reference]:
        assert impl.env_name == "torch" and impl.modes == {"fwd", "fwd_bwd"}
        assert {k: impl.info[k] for k in lce.CONTRACT} == lce.CONTRACT


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
    assert no_override["lce_source_md5"] == md5
    assert lce.setup_eager(c) == {"compiled": False}
    assert lce.setup_compiled(c) == {"compiled": True}


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
        assert result == {"override_ran": False, "override_events": []}
    assert len(outputs) == 2


def test_verify_raises_when_eligible_but_absent(monkeypatch):
    c = case("bfloat16", device="cuda")
    monkeypatch.setitem(lce._STATE, "case", c)
    monkeypatch.setitem(lce._STATE, "options", {"acc_policy": "compact", "acc_dtype": "float32"})
    monkeypatch.setattr(lce, "override_eligible", lambda case, options: True)
    monkeypatch.setattr(lce, "override_events", lambda fn: [])
    with pytest.raises(AssertionError, match="did not run although eligible"):
        lce.verify_lce(lambda: None)
    names = ["_native::_linear_cross_entropy_batch_chunked_cutedsl_2"]
    monkeypatch.setattr(lce, "override_events", lambda fn: names)
    assert lce.verify_lce(lambda: None) == {"override_ran": True, "override_events": names}
    with pytest.raises(AssertionError, match="ran although disabled"):
        lce.verify_lce_no_override(lambda: None)


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


def test_availability(monkeypatch):
    assert lce.lce_available() is True
    assert lce.lce_no_override_available() is True
    assert lce.compile_available() is True
    monkeypatch.delattr(torch.nn.functional, "linear_cross_entropy")
    reason = lce.lce_available()
    assert reason.startswith(f"torch {torch.__version__} lacks")
    assert "torch.nn.functional.linear_cross_entropy" in reason
    assert lce.lce_no_override_available() == reason
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
    runs = runner.enumerate_runs(SPEC, cases)
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
        assert record["status"] == "ok", (record["implementation"], record["reason"])
        assert record["parity"]["status"] == "ok", record["parity"]
        assert record["work"] == {"tokens": 64, "flops": 6 * 64 * 32 * 96}
        assert record["info"]["ignore_index"] == -100
    by_impl = {r["implementation"]: r for r in written if r["mode"] == "fwd_bwd"}
    assert by_impl["torch_compiled"]["info"]["compiled"] is True
    assert by_impl["torch_lce"]["verify"] == {"override_ran": False, "override_events": []}
    assert len(by_impl["torch_lce"]["info"]["lce_source_md5"]) == 32
    assert by_impl["torch_lce_no_override"]["info"]["override_disabled"] == list(lce.OVERRIDE_OPS)
    assert len(records.read(tmp_path / "lce" / "cpu")) == 8


def test_worker_payload_in_process():
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
        assert f"  {name}  modes=fwd,fwd_bwd  env=torch  available" in out
    assert "default dtypes: bfloat16" in out


def test_parent_does_not_import_torch():
    import subprocess

    code = (
        "import sys; from torch_op_bench.spec import resolve_op; resolve_op('lce');"
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
