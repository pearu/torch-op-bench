"""The worker: one (case, implementation, mode) per process (R2).

``python -m torch_op_bench.worker`` (or the runner's bootstrap of it, see
``runner.worker_command``) reads one JSON payload on stdin and prints exactly one
JSON record on stdout. Whatever the op prints goes to stderr. The
payload carries everything the record describes, so the parent and the worker need
not share a Python or a torch version; the module imports torch only after it has
applied the implementation's environment variables.

Payload keys: ``op`` (the reference resolved by ``spec.resolve_op``), ``op_name``,
``case`` (``Case.to_dict``), ``implementation``, ``mode``, ``settings`` (``warmup``,
``repeat``, ``flush_l2``, ``seed``, ``timeout``, ``matmul_reduction``), ``pythonpath``
(entries prepended to ``sys.path``), ``env_name``, ``env_vars``, ``packages``,
``device_slug``, ``work``, ``info``.
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
import traceback
from collections.abc import Callable
from typing import Any

from . import env, records
from .spec import Case, Implementation, OpSpec, Unavailable, resolve_op, to_json

# ``--matmul-reduction``: ``default`` leaves torch's matmul settings alone; ``strict``
# turns off the reduced-precision reduction of bf16 and fp16 GEMMs.
MATMUL_REDUCTIONS = ("default", "strict")
REDUCED_PRECISION_FLAGS = (
    "allow_bf16_reduced_precision_reduction",
    "allow_fp16_reduced_precision_reduction",
)

# An out-of-memory error's figures. CUDA's message: "Tried to allocate 4.00 GiB. GPU 0
# has a total capacity of 15.57 GiB of which 2.28 GiB is free. … Of the allocated
# memory 9.13 GiB is allocated by PyTorch, and 3.97 GiB is reserved by PyTorch but
# unallocated." The CPU allocator's: "… you tried to allocate 4503599627370496 bytes."
CUDA_OOM_FIGURES = re.compile(
    r"Tried to allocate (?P<tried>[\d.]+ \w+)\..*? of which (?P<free>[\d.]+ \w+) is free"
    r".*? (?P<allocated>[\d.]+ \w+) is allocated by PyTorch, and (?P<reserved>[\d.]+ \w+) is"
    r" reserved by PyTorch but unallocated",
    re.DOTALL,
)
CPU_OOM = "DefaultCPUAllocator: can't allocate memory"
CPU_OOM_TRIED = re.compile(r"tried to allocate (?P<bytes>\d+) bytes")
GIB = 1 << 30


def apply_matmul_reduction(setting: str) -> None:
    """Set the process-wide matmul reduction for ``setting``; ``default`` touches
    nothing. ``strict`` sets each reduced-precision flag to ``(False, True)`` (no
    reduced-precision reduction, split-K still allowed) on builds whose flags are
    pairs, else to ``False``. ``(False, False)`` is not offered: it needs cuBLASLt."""
    if setting not in MATMUL_REDUCTIONS:
        raise ValueError(f"matmul_reduction must be one of {MATMUL_REDUCTIONS}, not {setting!r}")
    if setting == "default":
        return
    import torch

    flags = torch.backends.cuda.matmul
    for name in REDUCED_PRECISION_FLAGS:
        pairs = hasattr(flags, f"{name}_split_k")
        setattr(flags, name, (False, True) if pairs else False)


def base_record(payload: dict[str, Any], python: str) -> dict[str, Any]:
    """A record with every field, describing the payload; results are ``None``."""
    case = Case.from_dict(payload["case"])
    return {
        "schema": records.SCHEMA,
        "timestamp": env.utc_timestamp(),
        "op": payload["op_name"],
        "op_ref": payload["op"],
        "device_slug": payload["device_slug"],
        "case": case.to_dict(),
        "case_key": case.key,
        "implementation": payload["implementation"],
        "mode": payload["mode"],
        "env_name": payload["env_name"],
        "python": python,
        "settings": payload["settings"],
        "work": payload.get("work", {}),
        "info": payload.get("info", {}),
        "status": None,
        "reason": None,
        "traceback": None,
        "stderr": None,
        "times_ms": None,
        "memory": None,
        "parity": None,
        "verify": None,
        "env": None,
        "env_from": None,
        "env_error": None,
    }


def run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply the payload's environment, run it, and return its record."""
    os.environ.update(payload.get("env_vars", {}))
    for entry in reversed(payload.get("pythonpath", [])):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    record = base_record(payload, sys.executable)
    try:
        spec = resolve_op(payload["op"])
        impl = spec.implementation(payload["implementation"])
        case = Case.from_dict(payload["case"])
        mode = payload["mode"]
        if mode not in impl.modes:
            raise ValueError(f"implementation {impl.name!r} does not support mode {mode!r}")
        # Before anything of the implementation runs; read back into env.torch.matmul.
        apply_matmul_reduction(payload["settings"].get("matmul_reduction", "default"))
        available = impl.available()
        if available is True:
            try:
                measured = _measure(spec, impl, case, mode, payload["settings"], record["info"])
            except Unavailable as exc:  # from setup, before anything is timed
                record["status"] = "unavailable"
                record["reason"] = exc.reason
            else:
                record.update(measured)
                record["status"] = "ok"
        else:
            record["status"] = "unavailable"
            record["reason"] = available if isinstance(available, str) else "not available"
    except Exception as exc:
        record["status"] = "error"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    record["env_from"] = "worker"
    try:
        record["env"] = env.collect(payload.get("packages", ()))
    except Exception as exc:  # e.g. CUDA in a bad state: keep the measurement anyway
        record["env_error"] = f"{type(exc).__name__}: {exc}"
    return to_json(record)


def is_out_of_memory(exc: BaseException) -> bool:
    """Whether ``exc`` says the device's memory ran out: ``torch.OutOfMemoryError``
    (CUDA), ``MemoryError``, or the CPU allocator's ``RuntimeError``."""
    import torch

    if isinstance(exc, (torch.OutOfMemoryError, MemoryError)):
        return True
    return isinstance(exc, RuntimeError) and CPU_OOM in str(exc)


def memory_figures(exc: BaseException) -> str:
    """What an out-of-memory error says of the memory, in one line: the size tried,
    what was free, allocated and reserved but unallocated; else its message's first
    line."""
    message = str(exc)
    found = CUDA_OOM_FIGURES.search(message)
    if found:
        return (
            f"tried to allocate {found['tried']}, {found['free']} free; "
            f"{found['allocated']} allocated, {found['reserved']} reserved but unallocated"
        )
    found = CPU_OOM_TRIED.search(message)
    if found:
        return f"tried to allocate {int(found['bytes']) / GIB:.2f} GiB"
    line = message.strip().splitlines()[0] if message.strip() else ""
    return f"{type(exc).__name__}: {line}" if line else type(exc).__name__


def device_memory(device: Any) -> str:
    """The device and its total memory: "NVIDIA GeForce RTX 5060 Ti (15.6 GiB)"."""
    import torch

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        name, total = props.name, props.total_memory
    else:
        name, total = device.type, os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return f"{name} ({total / GIB:.1f} GiB)"


def out_of_memory_reason(device: Any, exc: BaseException, phase: str) -> str:
    """The reason of a record whose point does not fit: a first line stable for the
    device (``report`` groups by it), a second with the figures and the phase."""
    return f"out of memory on {device_memory(device)}\n{memory_figures(exc)}; phase: {phase}"


def _measure(
    spec: OpSpec,
    impl: Implementation,
    case: Case,
    mode: str,
    settings: dict[str, Any],
    info: dict[str, Any],
) -> dict[str, Any]:
    """Measure the case; a point that does not fit in the device's memory, wherever
    it runs out, raises ``Unavailable`` with ``out_of_memory_reason``."""
    import torch

    device = torch.device(case.device)
    at = {"phase": "setup"}
    try:
        return _phases(spec, impl, case, mode, settings, info, device, at)
    except Exception as exc:
        if not is_out_of_memory(exc):
            raise
        reason = out_of_memory_reason(device, exc, at["phase"])
    # Outside the ``except``: the error's traceback, and the tensors its frames hold,
    # are released before the record's environment is collected.
    _release(device)
    raise Unavailable(reason)


def _release(device: Any) -> None:
    import torch

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _phases(
    spec: OpSpec,
    impl: Implementation,
    case: Case,
    mode: str,
    settings: dict[str, Any],
    info: dict[str, Any],
    device: Any,
    at: dict[str, str],
) -> dict[str, Any]:
    """``_measure``'s work, phase by phase; ``at["phase"]`` names the one running:
    ``setup``, ``parity``, ``verify``, ``memory``, ``warmup``, ``timing``."""
    import torch

    from .measure import clear_grads, peak_memory, time_callable

    generator = torch.Generator(device=device)
    generator.manual_seed(settings["seed"])
    inputs = spec.make_inputs(case, device, generator)
    setup_info = impl.setup(case) if impl.setup is not None else None
    if setup_info is not None and not isinstance(setup_info, dict):
        raise TypeError(f"setup returned a {type(setup_info).__name__}, not a dict or None")
    info = {**info, **(setup_info or {})}

    # Every call starts from the inputs' first state: gradients a ``backward()``
    # left on them are cleared before each call, outside what is measured.
    def reset() -> None:
        clear_grads(inputs)

    def fn() -> Any:
        return impl.run(inputs, mode)

    def fresh_fn() -> Any:
        reset()
        return fn()

    at["phase"] = "parity"
    parity = _parity(spec, impl, case, inputs, mode, reset)
    _release(device)
    at["phase"] = "verify"
    verify = impl.verify(fresh_fn) if impl.verify is not None else None
    at["phase"] = "memory"
    memory = peak_memory(fn, device, reset=reset)
    times = time_callable(
        fn,
        device=device,
        warmup=settings["warmup"],
        repeat=settings["repeat"],
        flush_l2=settings["flush_l2"],
        reset=reset,
        on_phase=lambda phase: at.update(phase=phase),
    )
    return {
        "info": info,
        "parity": parity,
        "verify": verify,
        "memory": memory,
        "times_ms": times,
    }


def _parity(
    spec: OpSpec,
    impl: Implementation,
    case: Case,
    inputs: Any,
    mode: str,
    reset: Callable[[], None],
) -> dict:
    parity: dict[str, Any] = {
        "status": None,
        "reason": None,
        "metrics": {},
        "tolerances": dict(spec.tolerances),
        "failed": [],
    }
    fits = spec.reference_fits(case)
    if fits is not True:
        parity["status"] = "skipped"
        parity["reason"] = fits if isinstance(fits, str) else "reference does not fit"
        return parity
    if mode not in spec.reference.modes:
        parity["status"] = "skipped"
        parity["reason"] = f"reference does not support mode {mode!r}"
        return parity
    # The implementation first, its outputs kept in host memory while the reference
    # runs: on the device they would pin the allocator segments its temporaries used,
    # and the fp32 reference would not have the memory ``reference_fits`` saw (I1-17).
    reset()
    outputs = _move(impl.run(inputs, mode), "cpu")
    # The reference running out of memory (``reference_fits`` was wrong) skips
    # parity: the measurement stays sound. The implementation running out of memory
    # here is the point not fitting, as anywhere in ``_measure``.
    reset()
    _release_cached(case)
    try:
        ref_outputs = spec.reference.run(inputs, mode)
    except Exception as exc:
        if not is_out_of_memory(exc):
            raise
        parity["reason"] = f"reference out of memory: {memory_figures(exc)}"
    else:
        reset()
        outputs = _move(outputs, case.device)
        try:
            metrics = {
                name: float(value)
                for name, value in spec.compare(outputs, ref_outputs, case).items()
            }
        except Exception as exc:
            if not is_out_of_memory(exc):
                raise
            parity["reason"] = f"comparison out of memory: {memory_figures(exc)}"
        del ref_outputs
    del outputs
    if parity["reason"] is not None:
        parity["status"] = "skipped"
        return parity
    parity["metrics"] = metrics
    # ``not value <= tol`` also fails NaN.
    parity["failed"] = sorted(
        name
        for name, tol in spec.tolerances.items()
        if name in metrics and not metrics[name] <= tol
    )
    parity["status"] = "fail" if parity["failed"] else "ok"
    return parity


def _move(outputs: Any, device: str) -> Any:
    """``outputs`` (a tuple) with each tensor moved to ``device``."""
    import torch

    return tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in outputs)


def _release_cached(case: Case) -> None:
    """Return the memory the caching allocator holds unused to the device, for the
    reference to allocate in full."""
    gc.collect()
    if case.device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()


def main() -> None:
    # Keep the real stdout for the record; anything else printed goes to stderr.
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    payload = json.load(sys.stdin)
    record = run_payload(payload)
    out.write(json.dumps(record) + "\n")
    out.flush()


if __name__ == "__main__":
    main()
