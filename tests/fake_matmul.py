"""Fakes of ``torch.backends.cuda.matmul`` for the tests of the matmul reduction
setting: a build whose reduced-precision flags are pairs, and one whose are not."""

from __future__ import annotations

from types import SimpleNamespace

PAIR_FLAGS = (
    "allow_bf16_reduced_precision_reduction",
    "allow_fp16_reduced_precision_reduction",
)


class PairFlags:
    """torch main: each flag is a pair ``(reduced_precision, split_k)``, set as a bool
    or a tuple, read back as the flag (its first element) and ``<flag>_split_k``."""

    def __init__(self) -> None:
        object.__setattr__(self, "pairs", {name: (True, True) for name in PAIR_FLAGS})
        object.__setattr__(self, "allow_tf32", False)
        object.__setattr__(self, "fp32_precision", "ieee")

    def __getattr__(self, name: str):
        pairs = object.__getattribute__(self, "pairs")
        if name in pairs:
            return pairs[name][0]
        if name.removesuffix("_split_k") in pairs:
            return pairs[name.removesuffix("_split_k")][1]
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name not in self.pairs:
            raise AttributeError(name)
        self.pairs[name] = (value, True) if isinstance(value, bool) else tuple(value)


def old_flags() -> SimpleNamespace:
    """An older build: plain bool flags, no ``_split_k`` and no ``fp32_precision``."""
    return SimpleNamespace(allow_tf32=False, **{name: True for name in PAIR_FLAGS})
