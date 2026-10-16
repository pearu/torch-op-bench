"""Timing and peak memory of a zero-argument callable (R3). Runs in the worker."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

L2_FLUSH_BYTES = 256 * 1024**2


def time_callable(
    fn: Callable[[], Any],
    *,
    device: str | torch.device,
    warmup: int,
    repeat: int,
    flush_l2: bool,
    flush_bytes: int = L2_FLUSH_BYTES,
) -> list[float]:
    """Run ``fn`` ``warmup`` times untimed, then ``repeat`` times timed.

    Returns all ``repeat`` samples in milliseconds. On CUDA each repetition is
    bracketed by a pre-created pair of events and the host synchronises once, after
    the loop. With ``flush_l2`` a ``flush_bytes`` buffer is overwritten before each
    repetition, outside the timed region.
    """
    device = torch.device(device)
    flush = torch.empty(flush_bytes, dtype=torch.uint8, device=device) if flush_l2 else None
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        with torch.cuda.device(device):
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
            torch.cuda.synchronize()
            for start, end in zip(starts, ends, strict=True):
                if flush is not None:
                    flush.zero_()
                start.record()
                fn()
                end.record()
            torch.cuda.synchronize()
            return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]
    samples = []
    for _ in range(repeat):
        if flush is not None:
            flush.zero_()
        begin = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - begin) * 1e3)
    return samples


def peak_memory(fn: Callable[[], Any], device: str | torch.device) -> dict[str, int | None]:
    """Peak allocated memory of one call of ``fn``, from a clean slate.

    ``baseline_bytes`` is what is allocated when ``fn`` starts (the inputs);
    ``peak_bytes`` the absolute peak during the call. Not measured on CPU (``None``).
    """
    device = torch.device(device)
    if device.type != "cuda":
        return {"peak_bytes": None, "baseline_bytes": None}
    with torch.cuda.device(device):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
    return {"peak_bytes": peak, "baseline_bytes": baseline}
