"""The op contract: what an op declares so that the core can benchmark it.

Nothing here imports torch: the parent process enumerates cases from a spec, and
only the worker process touches tensors.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

MODES = ("fwd", "fwd_bwd")
AXIS_KINDS = ("int", "str")


@dataclass(frozen=True)
class Axis:
    """A named shape parameter of an op and the values a sweep takes."""

    name: str
    values: list
    kind: str = "int"

    def __post_init__(self) -> None:
        if self.kind not in AXIS_KINDS:
            raise ValueError(f"axis {self.name!r}: kind must be one of {AXIS_KINDS}")
        if not self.values:
            raise ValueError(f"axis {self.name!r}: no values")

    def parse(self, text: str) -> int | str:
        """Convert a command-line value to this axis' kind."""
        return int(text) if self.kind == "int" else text


@dataclass(frozen=True)
class Case:
    """One point of an op's sweep: a value per axis, a dtype and a device."""

    params: Mapping[str, Any]
    dtype: str
    device: str

    @property
    def key(self) -> str:
        """A stable string identifying the case, used for resume."""
        params = ",".join(f"{k}={self.params[k]}" for k in sorted(self.params))
        return f"{params}|{self.dtype}|{self.device}"

    @property
    def slug(self) -> str:
        """A short string usable in file names."""
        parts = [f"{k}{self.params[k]}" for k in sorted(self.params)]
        parts += [self.dtype, self.device]
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", "-".join(parts))

    def to_dict(self) -> dict:
        return {"params": dict(self.params), "dtype": self.dtype, "device": self.device}

    @classmethod
    def from_dict(cls, data: Mapping) -> Case:
        return cls(params=dict(data["params"]), dtype=data["dtype"], device=data["device"])


def _always_available() -> bool | str:
    return True


def _always_fits(case: Case) -> bool | str:
    return True


@dataclass
class Implementation:
    """One way of computing an op.

    ``run(inputs, mode)`` computes the op on the inputs from ``OpSpec.make_inputs``
    in ``mode`` (one of ``modes``) and returns its outputs. ``available()`` returns
    ``True`` or a string saying why the implementation cannot run; it must import
    what it needs itself, so that the spec stays importable without it.
    ``env_vars`` are set in the worker before torch is imported; ``env_name`` names
    the environment (interpreter) the worker runs in. ``setup(case)`` runs once in
    the worker before anything is measured. ``verify(fn)`` receives the timed
    zero-argument callable, may run it (e.g. under a profiler) and returns a
    JSON-serialisable dict for the record or ``None``; it raises if the intended
    code path did not run. ``packages`` are the distributions whose versions the
    record carries; ``info`` is free-form, recorded as is.
    """

    name: str
    run: Callable[[Any, str], Any]
    modes: set[str] = field(default_factory=lambda: set(MODES))
    env_name: str = "torch"
    env_vars: dict[str, str] = field(default_factory=dict)
    setup: Callable[[Case], None] | None = None
    verify: Callable[[Callable[[], Any]], dict | None] | None = None
    available: Callable[[], bool | str] = _always_available
    packages: tuple[str, ...] = ()
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.modes) - set(MODES)
        if unknown:
            raise ValueError(f"implementation {self.name!r}: unknown modes {sorted(unknown)}")


@dataclass
class OpSpec:
    """An op: its axes, inputs, implementations, reference and metrics.

    ``make_inputs(case, device, generator)`` builds the inputs, seeded by the
    ``torch.Generator``. ``compare(outputs, ref_outputs, case)`` returns named error
    metrics, checked against ``tolerances`` (a metric without a tolerance is
    recorded, not checked). ``work(case)`` returns the work units of a case (e.g.
    ``{"flops": ...}``) for throughput. ``defaults`` is the point a sweep varies one
    axis around (an axis missing from it defaults to its first value).
    ``reference_fits(case)`` is the memory guard of the parity check: ``True``, or a
    string saying why the reference is skipped.
    """

    name: str
    axes: list[Axis]
    dtypes: list[str]
    make_inputs: Callable[[Case, Any, Any], Any]
    implementations: list[Implementation]
    reference: Implementation
    compare: Callable[[Any, Any, Case], dict[str, float]]
    tolerances: dict[str, float] = field(default_factory=dict)
    work: Callable[[Case], dict[str, float]] = lambda case: {}
    defaults: dict[str, Any] = field(default_factory=dict)
    reference_fits: Callable[[Case], bool | str] = _always_fits

    def __post_init__(self) -> None:
        names = [impl.name for impl in self.implementations]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"op {self.name!r}: duplicate implementations {duplicates}")
        unknown = set(self.defaults) - {axis.name for axis in self.axes}
        if unknown:
            raise ValueError(f"op {self.name!r}: defaults for unknown axes {sorted(unknown)}")

    def axis(self, name: str) -> Axis:
        for axis in self.axes:
            if axis.name == name:
                return axis
        raise KeyError(f"op {self.name!r} has no axis {name!r}")

    def implementation(self, name: str) -> Implementation:
        if name == self.reference.name:
            return self.reference
        for impl in self.implementations:
            if impl.name == name:
                return impl
        raise KeyError(f"op {self.name!r} has no implementation {name!r}")

    @property
    def default_point(self) -> dict[str, Any]:
        return {axis.name: self.defaults.get(axis.name, axis.values[0]) for axis in self.axes}


def resolve_op(ref: str) -> OpSpec:
    """Return the op spec named ``ref``.

    ``ref`` is either the short name of an in-package op (``torch_op_bench.ops``) or
    a dotted path ``package.module:OBJECT``.
    """
    if ":" in ref:
        module_name, _, attr = ref.partition(":")
    else:
        from . import ops

        if ref not in ops.REGISTRY:
            known = ", ".join(sorted(ops.REGISTRY)) or "none"
            raise KeyError(f"unknown op {ref!r} (in-package ops: {known}; or use module:OBJECT)")
        module_name, _, attr = ops.REGISTRY[ref].partition(":")
    spec = getattr(importlib.import_module(module_name), attr)
    if not isinstance(spec, OpSpec):
        raise TypeError(f"{ref!r} is a {type(spec).__name__}, not an OpSpec")
    return spec


def to_json(obj: Any) -> Any:
    """Make ``obj`` JSON-serialisable (sets become sorted lists, unknowns strings)."""
    return json.loads(json.dumps(obj, default=_json_default))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)
