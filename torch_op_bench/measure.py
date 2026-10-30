"""Timing and peak memory of a zero-argument callable (R3). Runs in the worker."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

L2_FLUSH_BYTES = 256 * 1024**2


def clear_grads(inputs: Any) -> None:
    """Set ``.grad`` to ``None`` on every tensor in ``inputs``.

    Walks tensors in tuples, lists and dicts, so that an implementation that calls
    ``backward()`` starts each call from the same state as the first: no gradient
    accumulation in its time, no leftover gradients in its memory baseline.
    """
    if isinstance(inputs, torch.Tensor):
        if inputs.grad is not None:
            inputs.grad = None
    elif isinstance(inputs, (tuple, list)):
        for item in inputs:
            clear_grads(item)
    elif isinstance(inputs, dict):
        for item in inputs.values():
            clear_grads(item)


def time_callable(
    fn: Callable[[], Any],
    *,
    device: str | torch.device,
    warmup: int,
    repeat: int,
    flush_l2: bool,
    flush_bytes: int = L2_FLUSH_BYTES,
    reset: Callable[[], Any] | None = None,
    on_phase: Callable[[str], Any] | None = None,
) -> list[float]:
    """Run ``fn`` ``warmup`` times untimed, then ``repeat`` times timed.

    Returns all ``repeat`` samples in milliseconds. On CUDA each repetition is
    bracketed by a pre-created pair of events and the host synchronises once, after
    the loop. ``reset`` (e.g. clearing gradients) runs before every call, and with
    ``flush_l2`` a ``flush_bytes`` buffer is overwritten before each repetition; both
    outside the timed region. ``on_phase`` is told ``"warmup"`` before the warmup and
    ``"timing"`` before the timed repetitions.
    """
    device = torch.device(device)
    if on_phase is None:
        on_phase = _nothing
    on_phase("warmup")
    flush = torch.empty(flush_bytes, dtype=torch.uint8, device=device) if flush_l2 else None
    if reset is None:
        reset = _nothing
    for _ in range(warmup):
        reset()
        fn()
    on_phase("timing")
    if device.type == "cuda":
        with torch.cuda.device(device):
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
            torch.cuda.synchronize()
            for start, end in zip(starts, ends, strict=True):
                reset()
                if flush is not None:
                    flush.zero_()
                start.record()
                fn()
                end.record()
            torch.cuda.synchronize()
            return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]
    samples = []
    for _ in range(repeat):
        reset()
        if flush is not None:
            flush.zero_()
        begin = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - begin) * 1e3)
    return samples


def peak_memory(
    fn: Callable[[], Any],
    device: str | torch.device,
    reset: Callable[[], Any] | None = None,
) -> dict[str, int | None]:
    """Peak allocated memory of one call of ``fn``, from a clean slate.

    ``reset`` runs first. ``baseline_bytes`` is what is allocated when ``fn`` starts
    (the inputs); ``peak_bytes`` the absolute peak during the call. Not measured on
    CPU (``None``).
    """
    device = torch.device(device)
    if device.type != "cuda":
        return {"peak_bytes": None, "baseline_bytes": None}
    with torch.cuda.device(device):
        if reset is not None:
            reset()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
    return {"peak_bytes": peak, "baseline_bytes": baseline}


def _nothing(*args: Any) -> None:
    pass
