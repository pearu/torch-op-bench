"""The worker: one (case, implementation, mode) per process (R2).

``python -m torch_op_bench.worker`` reads one JSON payload on stdin and prints
exactly one JSON record on stdout. Whatever the op prints goes to stderr. The
payload carries everything the record describes, so the parent and the worker need
not share a Python or a torch version; the module imports torch only after it has
applied the implementation's environment variables.

Payload keys: ``op`` (the reference resolved by ``spec.resolve_op``), ``op_name``,
``case`` (``Case.to_dict``), ``implementation``, ``mode``, ``settings`` (``warmup``,
``repeat``, ``flush_l2``, ``seed``, ``timeout``), ``pythonpath`` (entries prepended
to ``sys.path``), ``env_name``, ``env_vars``, ``packages``, ``device_slug``,
``work``, ``info``.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import traceback
from typing import Any

from . import env, records
from .spec import Case, Implementation, OpSpec, resolve_op, to_json


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
        available = impl.available()
        if available is True:
            record.update(_measure(spec, impl, case, mode, payload["settings"]))
            record["status"] = "ok"
        else:
            record["status"] = "unavailable"
            record["reason"] = available if isinstance(available, str) else "not available"
    except Exception as exc:
        record["status"] = "error"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    record["env"] = env.collect(payload.get("packages", ()))
    record["env_from"] = "worker"
    return to_json(record)


def _measure(
    spec: OpSpec, impl: Implementation, case: Case, mode: str, settings: dict[str, Any]
) -> dict[str, Any]:
    import torch

    from .measure import peak_memory, time_callable

    device = torch.device(case.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(settings["seed"])
    inputs = spec.make_inputs(case, device, generator)
    if impl.setup is not None:
        impl.setup(case)

    def fn() -> Any:
        return impl.run(inputs, mode)

    parity = _parity(spec, impl, case, inputs, mode)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    verify = impl.verify(fn) if impl.verify is not None else None
    memory = peak_memory(fn, device)
    times = time_callable(
        fn,
        device=device,
        warmup=settings["warmup"],
        repeat=settings["repeat"],
        flush_l2=settings["flush_l2"],
    )
    return {"parity": parity, "verify": verify, "memory": memory, "times_ms": times}


def _parity(spec: OpSpec, impl: Implementation, case: Case, inputs: Any, mode: str) -> dict:
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
    ref_outputs = spec.reference.run(inputs, mode)
    outputs = impl.run(inputs, mode)
    metrics = {
        name: float(value) for name, value in spec.compare(outputs, ref_outputs, case).items()
    }
    del ref_outputs, outputs
    parity["metrics"] = metrics
    # ``not value <= tol`` also fails NaN.
    parity["failed"] = sorted(
        name
        for name, tol in spec.tolerances.items()
        if name in metrics and not metrics[name] <= tol
    )
    parity["status"] = "fail" if parity["failed"] else "ok"
    return parity


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
