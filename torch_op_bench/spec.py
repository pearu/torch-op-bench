"""The op contract: what an op declares so that the core can benchmark it.

Nothing here imports torch: the parent process enumerates cases from a spec, and
only the worker process touches tensors.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

MODES = ("fwd", "fwd_bwd")
AXIS_KINDS = ("int", "str")
# Characters that structure ``Case.key``; axis names and str values must not hold them.
KEY_SEPARATORS = ",=|"


def _check_key_text(text: str, what: str) -> None:
    bad = sorted(set(text) & set(KEY_SEPARATORS))
    if bad:
        raise ValueError(f"{what} {text!r} contains {''.join(bad)!r}, reserved for case keys")


def normalize_device(device: str) -> str:
    """``cpu`` stays; ``cuda`` becomes ``cuda:0``, so both name the same case."""
    if not re.fullmatch(r"[a-z]+(:[0-9]+)?", device):
        raise ValueError(f"invalid device {device!r} (expected e.g. cpu, cuda, cuda:1)")
    if device == "cpu" or ":" in device:
        return device
    return f"{device}:0"


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
        _check_key_text(self.name, "axis name")
        if self.kind == "str":
            for value in self.values:
                _check_key_text(value, f"axis {self.name!r} value")

    def parse(self, text: str) -> int | str:
        """Convert a command-line value to this axis' kind."""
        if self.kind == "int":
            return int(text)
        _check_key_text(text, f"axis {self.name!r} value")
        return text


@dataclass(frozen=True)
class Case:
    """One point of an op's sweep: a value per axis, a dtype and a device."""

    params: Mapping[str, Any]
    dtype: str
    device: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", normalize_device(self.device))
        for name, value in self.params.items():
            _check_key_text(name, "axis name")
            if isinstance(value, str):
                _check_key_text(value, f"axis {name!r} value")

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


class Unavailable(Exception):
    """Raised by an implementation's ``setup`` when it cannot run the case here.

    E.g. a device its op does not support. The worker records ``unavailable`` with
    the exception's message as the reason, as for an implementation whose
    ``available()`` says no; ``error`` stays for failures nobody foresaw.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _always_available() -> bool | str:
    return True


def _always_fits(case: Case) -> bool | str:
    return True


@dataclass(frozen=True)
class ParityNote:
    """A known cause of an implementation's parity failure (R1-30 F1).

    ``implementation`` fails ``metrics`` (``None``: any of its metrics) for a cause
    outside the tool, named in ``text``: one or two sentences with the cause, its
    source (file and line, the torch version it was verified on) and a reference (an
    issue or PR URL). ``applies(case, mode, metrics, torch_version)`` says whether
    the cause holds for a record: its case, mode, parity metrics and the worker's
    torch version (``None`` when not recorded). The note lives with the op, not in
    the records: a cause found after a run needs no rerun.
    """

    implementation: str
    metrics: tuple[str, ...] | None
    applies: Callable[[Case, str, Mapping[str, float], str | None], bool]
    text: str


@dataclass
class Implementation:
    """One way of computing an op.

    ``run(inputs, mode)`` computes the op on the inputs from ``OpSpec.make_inputs``
    in ``mode`` (one of ``modes``) and returns its outputs; it is called many times
    on the same inputs, and the worker clears the inputs' ``.grad`` before each call
    (outside what is measured). ``available()`` returns
    ``True`` or a string saying why the implementation cannot run; it must import
    what it needs itself, so that the spec stays importable without it.
    ``env_vars`` are set in the worker before torch is imported; ``env_name`` names
    the environment (interpreter) the worker runs in. ``setup(case)`` runs once in
    the worker before anything is measured; it raises ``Unavailable`` when the case
    cannot run here (e.g. on a device the op does not support), and may return a
    dict of what the implementation only knows at run time (resolved options, a source hash, whether
    it compiled), merged over ``info`` into the record's ``info``. ``verify(fn)`` receives the timed
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
    setup: Callable[[Case], dict[str, Any] | None] | None = None
    verify: Callable[[Callable[[], Any]], dict | None] | None = None
    available: Callable[[], bool | str] = _always_available
    packages: tuple[str, ...] = ()
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.modes) - set(MODES)
        if unknown:
            raise ValueError(f"implementation {self.name!r}: unknown modes {sorted(unknown)}")

    def as_control(self, env_name: str) -> Implementation:
        """This implementation run in environment ``env_name``, as a control.

        The copy is named ``<name>_<env_name>`` and shares everything but its name,
        environment and ``info`` (``role: "control"``, ``control_of: <name>``). It
        is how a baseline enters a counterpart's environment, so that a comparison
        across environments can tell the counterpart from the torch build.
        """
        return dataclasses.replace(
            self,
            name=f"{self.name}_{env_name}",
            env_name=env_name,
            info={**self.info, "role": "control", "control_of": self.name},
        )

    def with_params(self, **params: Any) -> Implementation:
        """This implementation with ``params`` passed to its ``setup``, as a variant.

        The copy is named ``<name>[k=v,...]`` (keys sorted, values as given), its
        ``info`` gains ``base: <name>`` and ``params``, and its ``setup`` becomes
        ``setup(case, **params)``: the op's setup takes the parameters as keyword
        arguments and keeps them where its ``run`` reads them; an implementation without
        ``setup`` has no variants (``ValueError``). A variant exists for
        the time-at-equal-peak comparison (Statement §6 step 6f): ``run`` leaves it
        out unless ``--impl`` names it.
        """
        if not params:
            raise ValueError(f"implementation {self.name!r}: a variant needs parameters")
        if self.setup is None:
            # Its parameters would reach nothing: the base under another name (R1-39 R1).
            raise ValueError(
                f"implementation {self.name!r} has no setup to take the parameters {params}"
            )
        name = f"{self.name}[{','.join(f'{k}={params[k]}' for k in sorted(params))}]"
        base_setup = self.setup
        return dataclasses.replace(
            self,
            name=name,
            info={**self.info, "base": self.name, "params": dict(params)},
            setup=lambda case: base_setup(case, **params),
        )

    @property
    def variant_of(self) -> str | None:
        """The implementation this one is a variant of (``with_params``), or ``None``."""
        return self.info.get("base") if "params" in self.info else None


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
    string saying why the reference is skipped. ``default_dtypes`` are the dtypes a
    run uses when none are given (``None``: all of ``dtypes``). ``parity_notes``
    explain parity failures with a known cause (``ParityNote``).
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
    default_dtypes: list[str] | None = None
    parity_notes: list[ParityNote] = field(default_factory=list)

    def __post_init__(self) -> None:
        names = [impl.name for impl in self.implementations]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"op {self.name!r}: duplicate implementations {duplicates}")
        unknown = sorted({n.implementation for n in self.parity_notes} - set(names))
        if unknown:
            raise ValueError(
                f"op {self.name!r}: parity notes for unknown implementations {unknown}"
            )
        unknown = set(self.defaults) - {axis.name for axis in self.axes}
        if unknown:
            raise ValueError(f"op {self.name!r}: defaults for unknown axes {sorted(unknown)}")
        if self.default_dtypes is not None:
            unknown = sorted(set(self.default_dtypes) - set(self.dtypes))
            if unknown or not self.default_dtypes:
                raise ValueError(
                    f"op {self.name!r}: default_dtypes must be a non-empty subset of "
                    f"dtypes, got {self.default_dtypes}"
                )

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
