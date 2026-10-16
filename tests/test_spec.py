import pytest
import toy_op

from torch_op_bench import ops
from torch_op_bench.spec import Axis, Case, Implementation, OpSpec, resolve_op, to_json


def test_axis_parse_and_validation():
    assert Axis("M", [1, 2]).parse("8") == 8
    assert Axis("layout", ["a"], kind="str").parse("8") == "8"
    with pytest.raises(ValueError, match="kind"):
        Axis("M", [1], kind="float")
    with pytest.raises(ValueError, match="no values"):
        Axis("M", [])


def test_case_key_slug_roundtrip():
    case = Case({"N": 4, "M": 8, "layout": "a b"}, "float32", "cpu")
    same = Case({"M": 8, "layout": "a b", "N": 4}, "float32", "cpu")
    assert case.key == same.key == "M=8,N=4,layout=a b|float32|cpu"
    assert case.slug == "M8-N4-layouta-b-float32-cpu"
    assert Case.from_dict(case.to_dict()) == case
    assert Case({"M": 9, "N": 4, "layout": "a b"}, "float32", "cpu").key != case.key


def test_implementation_defaults_and_modes():
    impl = Implementation("x", lambda inputs, mode: None)
    assert impl.modes == {"fwd", "fwd_bwd"}
    assert impl.env_name == "torch" and impl.env_vars == {}
    assert impl.available() is True
    with pytest.raises(ValueError, match="unknown modes"):
        Implementation("x", lambda inputs, mode: None, modes={"bwd"})


def test_opspec_lookup_and_defaults():
    spec = toy_op.TOY
    assert spec.axis("K").values == [4, 8]
    with pytest.raises(KeyError):
        spec.axis("Z")
    assert spec.implementation("plain").name == "plain"
    assert spec.implementation("reference") is spec.reference
    with pytest.raises(KeyError):
        spec.implementation("nope")
    assert spec.default_point == {"M": 16, "K": 4, "N": 4, "layout": "contiguous"}
    assert spec.work(Case(spec.default_point, "float32", "cpu")) == {"flops": 512.0}
    assert spec.reference_fits(Case(spec.default_point, "float32", "cpu")) is True


def test_opspec_validation():
    impl = Implementation("x", lambda inputs, mode: None)
    kwargs = dict(
        name="op",
        axes=[Axis("M", [1])],
        dtypes=["float32"],
        make_inputs=None,
        reference=impl,
        compare=None,
    )
    with pytest.raises(ValueError, match="duplicate"):
        OpSpec(implementations=[impl, impl], **kwargs)
    with pytest.raises(ValueError, match="unknown axes"):
        OpSpec(implementations=[impl], defaults={"Q": 1}, **kwargs)
    assert OpSpec(implementations=[impl], **kwargs).work(None) == {}


def test_resolve_dotted_path():
    assert resolve_op("toy_op:TOY") is toy_op.TOY
    with pytest.raises(TypeError, match="not an OpSpec"):
        resolve_op("toy_op:compare")
    with pytest.raises(ModuleNotFoundError):
        resolve_op("no_such_module:X")


def test_resolve_by_name(monkeypatch):
    assert ops.REGISTRY == {}
    with pytest.raises(KeyError, match="in-package ops: none"):
        resolve_op("toy")
    monkeypatch.setitem(ops.REGISTRY, "toy", "toy_op:TOY")
    assert resolve_op("toy") is toy_op.TOY


def test_to_json():
    assert to_json({"a": {3, 1}, "b": object}) == {"a": [1, 3], "b": str(object)}
