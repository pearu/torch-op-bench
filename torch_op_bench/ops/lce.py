"""Linear cross entropy (LCE): ``cross_entropy(linear(hidden, weight), labels)``.

The common contract of the counterparts (Statement 1, §5): inputs ``hidden[N,H]``
and ``weight[V,H]`` with ``requires_grad``, int64 ``labels[N]`` with a fraction
``ignore_frac`` of them set to ``ignore_index=-100``; no bias, no label smoothing;
the output is the mean loss over the non-ignored tokens. Mode ``fwd_bwd`` runs the
forward and ``loss.backward()`` and returns ``(loss, grad_hidden, grad_weight)``;
mode ``fwd`` runs the forward under ``no_grad`` and returns ``(loss,)``.

torch is imported lazily, inside the callables, so the parent process needs no torch
and the worker imports it only after an implementation's environment is set.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
from typing import Any

from ..spec import Axis, Case, Implementation, OpSpec

IGNORE_INDEX = -100

# Constants of the contract, recorded in every implementation's info.
CONTRACT: dict[str, Any] = {
    "ignore_index": IGNORE_INDEX,
    "reduction": "mean",
    "bias": False,
    "label_smoothing": 0.0,
}

# The ops the torch._native CuTeDSL override replaces; disabling them selects the
# eager accumulator of F.linear_cross_entropy's chunked path.
OVERRIDE_OPS = (
    "_linear_cross_entropy_batch_chunked",
    "_linear_cross_entropy_batch_chunked_no_reduction",
)

# Source files whose Python-only edits the torch build hash would miss (R4).
LCE_SOURCES = (
    "nn/modules/linear_cross_entropy*.py",
    "_native/ops/linear_cross_entropy/*.py",
)

# The distributions the override runs on (R4: versions of what an implementation
# uses); torch._native checks both before registering the override.
DSL_PACKAGES = ("nvidia-cutlass-dsl", "apache-tvm-ffi")

# Liger's fused LCE runs Triton kernels: the versions of both are recorded.
LIGER_PACKAGES = ("liger-kernel", "triton")

# torchtitan's losses import spmd_types, which torchtitan pins: both are recorded.
TORCHTITAN_PACKAGES = ("torchtitan", "spmd_types")

# ChunkedLossWrapper.Config's default number of chunks, which torchtitan trains with.
TORCHTITAN_CHUNKS = 8

# What setup learnt about the case this worker runs; verify reads it. A worker
# runs one case, so module state is per case.
_STATE: dict[str, Any] = {}


def _shape(case: Case) -> tuple[int, int, int]:
    return case.params["N"], case.params["H"], case.params["V"]


def make_inputs(case: Case, device: Any, generator: Any) -> tuple[Any, Any, Any]:
    import torch

    dtype = getattr(torch, case.dtype)
    n, h, v = _shape(case)
    frac = float(case.params["ignore_frac"])
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"ignore_frac must be in [0, 1], got {frac}")
    hidden = torch.randn(n, h, dtype=dtype, device=device, generator=generator)
    weight = torch.randn(v, h, dtype=dtype, device=device, generator=generator) * h**-0.5
    labels = torch.randint(0, v, (n,), device=device, generator=generator)
    ignored = round(frac * n)
    if ignored:
        positions = torch.randperm(n, device=device, generator=generator)[:ignored]
        labels[positions] = IGNORE_INDEX
    return hidden.requires_grad_(), weight.requires_grad_(), labels


def _call(loss_fn: Any, inputs: tuple[Any, Any, Any], mode: str) -> tuple:
    import torch

    hidden, weight, labels = inputs
    if mode == "fwd":
        with torch.no_grad():
            return (loss_fn(hidden, weight, labels),)
    loss = loss_fn(hidden, weight, labels)
    loss.backward()
    return (loss.detach(), hidden.grad, weight.grad)


def _eager_loss(hidden: Any, weight: Any, labels: Any) -> Any:
    import torch.nn.functional as F

    return F.cross_entropy(F.linear(hidden, weight), labels, ignore_index=IGNORE_INDEX)


@functools.cache
def _compiled_loss() -> Any:
    import torch

    return torch.compile(_eager_loss, fullgraph=True)


@functools.cache
def _lce_options() -> Any:
    import torch

    return torch.nn.LinearCrossEntropyOptions()


def _lce_loss(hidden: Any, weight: Any, labels: Any) -> Any:
    import torch.nn.functional as F

    return F.linear_cross_entropy(
        hidden, weight, labels, ignore_index=IGNORE_INDEX, options=_lce_options()
    )


def run_eager(inputs: Any, mode: str) -> tuple:
    return _call(_eager_loss, inputs, mode)


def run_compiled(inputs: Any, mode: str) -> tuple:
    return _call(_compiled_loss(), inputs, mode)


def run_lce(inputs: Any, mode: str) -> tuple:
    return _call(_lce_loss, inputs, mode)


def _liger_loss(hidden: Any, weight: Any, labels: Any, accum_dtype: str | None) -> Any:
    """Liger's functional with the contract's arguments only; the rest at its defaults."""
    import torch
    from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy

    return liger_fused_linear_cross_entropy(
        hidden,
        weight,
        labels,
        ignore_index=IGNORE_INDEX,
        reduction="mean",
        accum_dtype=None if accum_dtype is None else getattr(torch, accum_dtype),
    )


def run_liger(inputs: Any, mode: str) -> tuple:
    return _call(functools.partial(_liger_loss, accum_dtype="float32"), inputs, mode)


def run_liger_native_accum(inputs: Any, mode: str) -> tuple:
    return _call(functools.partial(_liger_loss, accum_dtype=None), inputs, mode)


def _titan_layout(pred: Any, labels: Any) -> tuple[Any, Any]:
    """``pred[N, ...]`` and ``labels[N]`` in the layout the torchtitan losses take:
    ``[1, N, ...]`` and ``[1, N]`` (0.3.0, chunked along dim 1), as they are for
    ``main``'s ``[T, ...]`` and ``[T]`` (chunked along dim 0)."""
    if _STATE["titan_api"] == "compile_config":
        return pred.unsqueeze(0), labels.unsqueeze(0)
    return pred, labels


def _lm_head(weight: Any) -> Any:
    """A module computing ``F.linear(x, weight)`` over the case's own ``weight``.

    ``weight`` is a plain attribute, not an ``nn.Parameter`` copy, so the chunk
    backwards accumulate into the benchmark's ``weight.grad``. Cached per inputs:
    the module is remade only for another ``weight`` tensor.
    """
    cached = _STATE.get("lm_head")
    if cached is not None and cached.weight is weight:
        return cached
    import torch.nn as nn
    import torch.nn.functional as F

    class LMHead(nn.Module):
        def __init__(self, weight: Any) -> None:
            super().__init__()
            self.weight = weight  # a Tensor, not a Parameter: a plain attribute

        def forward(self, x: Any) -> Any:
            return F.linear(x, self.weight)

    _STATE["lm_head"] = LMHead(weight)
    return _STATE["lm_head"]


def _valid_tokens(labels: Any) -> Any:
    # The divisor torchtitan's losses take for the mean over non-ignored tokens.
    return (labels != IGNORE_INDEX).sum()


def _torchtitan_chunked_loss(hidden: Any, weight: Any, labels: Any) -> Any:
    """torchtitan's ChunkedLossWrapper: with ``hidden`` requiring grad, its call has
    already run the chunk backwards; ``loss.backward()`` hands the accumulated
    gradient to ``hidden``."""
    wrapper = _STATE["titan_loss"]
    wrapper.set_lm_head(_lm_head(weight))
    pred, target = _titan_layout(hidden, labels)
    # Positional: 0.3.0 names it global_valid_tokens, main global_loss_token_counts.
    loss, _ = wrapper(pred, target, _valid_tokens(labels))
    return loss


def _torchtitan_ce_loss(hidden: Any, weight: Any, labels: Any) -> Any:
    import torch.nn.functional as F

    pred, target = _titan_layout(F.linear(hidden, weight), labels)
    loss, _ = _STATE["titan_loss"](pred, target, _valid_tokens(labels))
    return loss


def run_torchtitan_chunked(inputs: Any, mode: str) -> tuple:
    return _call(_torchtitan_chunked_loss, inputs, mode)


def run_torchtitan_ce(inputs: Any, mode: str) -> tuple:
    return _call(_torchtitan_ce_loss, inputs, mode)


def run_reference(inputs: Any, mode: str) -> tuple:
    """fp32 eager on fresh fp32 copies of the inputs, and their gradients."""
    hidden, weight, labels = inputs
    hidden = hidden.detach().float().requires_grad_()
    weight = weight.detach().float().requires_grad_()
    return _call(_eager_loss, (hidden, weight, labels), mode)


def compare(outputs: tuple, ref_outputs: tuple, case: Case) -> dict[str, float]:
    loss, loss_ref = outputs[0].float(), ref_outputs[0].float()
    abs_err = (loss - loss_ref).abs().item()
    metrics = {
        "loss_abs_err": abs_err,
        "loss_rel_err": abs_err / max(abs(loss_ref.item()), 1e-6),
    }
    names = ("grad_hidden_rel_l2", "grad_weight_rel_l2")
    for name, grad, grad_ref in zip(names, outputs[1:], ref_outputs[1:], strict=False):
        grad_ref = grad_ref.float()
        metrics[name] = ((grad.float() - grad_ref).norm() / grad_ref.norm()).item()
    return metrics


def reference_fits(case: Case) -> bool | str:
    """On CUDA: whether the fp32 reference fits in 80 % of the free memory.

    Estimate: fp32 logits, their gradient and the softmax temporaries (3 N V), and
    fp32 copies of the inputs and their gradients (2 (N H + V H)).
    """
    if not case.device.startswith("cuda"):
        return True
    import torch

    n, h, v = _shape(case)
    needed = 4 * (3 * n * v + 2 * (n * h + v * h))
    free = torch.cuda.mem_get_info(torch.device(case.device))[0]
    if needed <= 0.8 * free:
        return True
    gib = 1024**3
    return (
        f"reference does not fit: {needed / gib:.1f} GiB needed, "
        f"{free / gib:.1f} GiB free (80 % usable)"
    )


def work(case: Case) -> dict[str, float]:
    n, h, v = _shape(case)
    # Forward GEMM plus the two backward GEMMs; recomputation is an
    # implementation's own cost.
    return {"tokens": n, "flops": 6 * n * h * v}


# --- availability -------------------------------------------------------------


def _torch_version() -> str:
    import torch

    return torch.__version__


def compile_available() -> bool | str:
    import torch

    if not hasattr(torch, "compile"):
        return f"torch {_torch_version()} has no torch.compile"
    import torch._dynamo

    if not torch._dynamo.is_dynamo_supported():
        return f"torch.compile is not supported here (torch {_torch_version()})"
    return True


def lce_available() -> bool | str:
    import torch
    import torch.nn.functional as F

    missing = [
        name
        for name, present in (
            ("torch.nn.functional.linear_cross_entropy", hasattr(F, "linear_cross_entropy")),
            ("torch.nn.LinearCrossEntropyOptions", hasattr(torch.nn, "LinearCrossEntropyOptions")),
        )
        if not present
    ]
    if missing:
        return f"torch {_torch_version()} lacks {', '.join(missing)}"
    return True


def lce_no_override_available() -> bool | str:
    available = lce_available()
    if available is not True:
        return available
    import torch

    if not hasattr(torch.backends, "python_native"):
        return f"torch {_torch_version()} lacks torch.backends.python_native"
    return True


def liger_version() -> str | None:
    try:
        return importlib.metadata.version("liger-kernel")
    except importlib.metadata.PackageNotFoundError:
        return None


def liger_available() -> bool | str:
    """Whether this environment has the Liger API the entries call, and a GPU."""
    try:
        from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy
    except ImportError as exc:
        return (
            f"liger_kernel.transformers.functional cannot be imported "
            f"(liger-kernel {liger_version()}, torch {_torch_version()}): {exc}"
        )
    if "accum_dtype" not in inspect.signature(liger_fused_linear_cross_entropy).parameters:
        return (
            f"liger-kernel {liger_version()}: liger_fused_linear_cross_entropy takes no accum_dtype"
        )
    import torch

    if not torch.cuda.is_available():
        return "liger-kernel needs a CUDA device"
    return True


def torchtitan_version() -> str | None:
    try:
        return importlib.metadata.version("torchtitan")
    except importlib.metadata.PackageNotFoundError:
        return None


def torchtitan_available() -> bool | str:
    """Whether torchtitan's losses import here. No CUDA requirement: with no mesh
    they are plain torch, which runs on CPU."""
    try:
        import torchtitan.components.loss  # noqa: F401
    except ImportError as exc:
        return (
            f"torchtitan.components.loss cannot be imported "
            f"(torchtitan {torchtitan_version()}, torch {_torch_version()}): {exc}"
        )
    return True


def torchtitan_compiled_available() -> bool | str:
    available = torchtitan_available()
    return compile_available() if available is True else available


# --- setup: what is only known at run time --------------------------------------


def lce_source_md5() -> dict[str, Any]:
    """md5 over the LCE source files of the torch that is imported.

    Files are looked up under each entry of ``torch.__path__``, the first entry
    holding a relative path winning, as for imports; the hash covers each file's
    relative path and contents, in sorted order. ``None`` when there are none.
    """
    import torch

    files: dict[str, Path] = {}
    for root in map(Path, torch.__path__):
        for pattern in LCE_SOURCES:
            for path in root.glob(pattern):
                files.setdefault(path.relative_to(root).as_posix(), path)
    if not files:
        return {"lce_source_md5": None, "lce_source_files": []}
    digest = hashlib.md5()
    for name in sorted(files):
        digest.update(name.encode() + b"\0" + files[name].read_bytes() + b"\0")
    return {"lce_source_md5": digest.hexdigest(), "lce_source_files": sorted(files)}


def resolved_options(case: Case) -> dict[str, Any]:
    """LinearCrossEntropyOptions() as the build resolves it for the case."""
    import torch

    options = _lce_options()
    fields = ("acc_policy", "acc_dtype", "chunking_method", "batch_chunk_size")
    resolved = False
    if hasattr(options, "_adjust"):
        n, h, v = _shape(case)
        options = options._adjust(
            num_batches=n,
            in_features=h,
            num_classes=v,
            dtype=getattr(torch, case.dtype),
            device=torch.device(case.device),
        )
        resolved = True
    info = {name: getattr(options, name, None) for name in fields}
    if isinstance(info["acc_dtype"], torch.dtype):
        info["acc_dtype"] = str(info["acc_dtype"]).removeprefix("torch.")
    return {**info, "resolved": resolved}


def setup_eager(case: Case) -> dict[str, Any]:
    return {"compiled": False}


def setup_compiled(case: Case) -> dict[str, Any]:
    _compiled_loss()
    return {"compiled": True}


def override_registration() -> dict[str, Any]:
    """Whether this build registers the torch._native LCE override, and if not why.

    Mirrors the checks of ``register_linear_cross_entropy_overrides``, through
    ``torch._native.cutedsl_utils``, a private module: ``registered`` is ``None``
    when it cannot be asked.
    """
    try:
        from torch._native import cutedsl_utils as cu
    except Exception as exc:
        reason = f"cannot tell: torch._native.cutedsl_utils: {type(exc).__name__}: {exc}"
        return {"registered": None, "reason": reason}
    try:
        if not cu.runtime_available():
            reason = (
                "CuTeDSL runtime not available (nvidia-cutlass-dsl or apache-tvm-ffi "
                "missing, or a build without CUDA)"
            )
        elif cu.check_native_jit_disabled():
            reason = "disabled by TORCH_DISABLE_NATIVE_JIT=1"
        elif not cu._version_is_ok():
            reason = (
                f"nvidia-cutlass-dsl {cu.runtime_version()} is not a known-good version "
                "(TORCH_NATIVE_SKIP_VERSION_CHECK=1 overrides)"
            )
        else:
            reason = None
    except Exception as exc:
        return {"registered": None, "reason": f"cannot tell: {type(exc).__name__}: {exc}"}
    return {"registered": reason is None, "reason": reason}


def setup_lce(case: Case) -> dict[str, Any]:
    options = resolved_options(case)
    registration = override_registration()
    _STATE.update(case=case, options=options, registration=registration)
    return {
        "compiled": False,
        "options": options,
        "override_registration": registration,
        **lce_source_md5(),
    }


def setup_lce_no_override(case: Case) -> dict[str, Any]:
    import torch

    torch.backends.python_native.disable_operations(*OVERRIDE_OPS)
    return {**setup_lce(case), "override_disabled": list(OVERRIDE_OPS)}


def liger_chunks(N: int, H: int, V: int, chunk_mem_const: int = 1) -> dict[str, int]:
    """How Liger's fused LCE chunks the N tokens (``fused_linear_cross_entropy_forward``
    of liger-kernel 0.8.4): each chunk's logits are bounded by ``chunk_mem_const * N * H``.
    """
    inc_factor = -(-V // (chunk_mem_const * H))
    rows = -(-N // inc_factor)
    chunk_size = min(1 << (rows - 1).bit_length(), N)
    return {"inc_factor": inc_factor, "chunk_size": chunk_size, "num_chunks": -(-N // chunk_size)}


def _setup_liger(case: Case, accum_dtype: str | None) -> dict[str, Any]:
    # With ce_impl=None (the public default) Liger calls its Triton
    # liger_cross_entropy_kernel directly, not its backend dispatcher, so the
    # LIGER_KERNEL_IMPL* environment variables do not apply.
    return {
        "accum_dtype": accum_dtype,
        "chunk_mem_const": 1,
        **liger_chunks(*_shape(case)),
        "liger_kernel": liger_version(),
        "ce_impl": "triton",
    }


def setup_liger(case: Case) -> dict[str, Any]:
    return _setup_liger(case, "float32")


def setup_liger_native_accum(case: Case) -> dict[str, Any]:
    return _setup_liger(case, None)


def torchtitan_api() -> str:
    """How this torchtitan compiles its loss: ``compile_config`` (0.3.0, a
    ``CompileConfig`` passed to the loss's constructor; ``[B, L, ...]`` inputs) or
    ``local_compile`` (``main`` since 0.3.0: the process-wide
    ``apply_local_compile(["loss"])``; ``[T, ...]`` inputs)."""
    from torchtitan.components.loss import ChunkedLossWrapper

    params = inspect.signature(ChunkedLossWrapper.__init__).parameters
    return "compile_config" if "compile_config" in params else "local_compile"


def _spmd_backend() -> str | None:
    """torchtitan's SPMD backend; ``None`` for a torchtitan without the choice."""
    try:
        from torchtitan.distributed.utils import get_spmd_backend
    except ImportError:
        return None
    return get_spmd_backend()


def _build_torchtitan(chunked: bool, compiled: bool) -> tuple[Any, str]:
    """torchtitan's loss as its trainer builds it, and how it is compiled."""
    from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss

    api = torchtitan_api()
    cls = ChunkedLossWrapper if chunked else CrossEntropyLoss
    config = cls.Config(num_chunks=TORCHTITAN_CHUNKS) if chunked else cls.Config()
    if api == "compile_config":
        if not compiled:
            return cls(config), ""
        from torchtitan.config import CompileConfig

        compile_config = CompileConfig(enable=True, components=["loss"])
        compile = f"torch.compile(backend={compile_config.backend!r})"
        return cls(config, compile_config=compile_config), compile
    from torchtitan.distributed.local_compile import apply_local_compile

    # Process-wide; a worker runs one implementation, so it is this entry's.
    apply_local_compile(["loss"] if compiled else [])
    return cls(config), "torch.compile(fullgraph=True)" if compiled else ""


def _setup_torchtitan(case: Case, chunked: bool, compiled: bool) -> dict[str, Any]:
    n = case.params["N"]
    if chunked and n % TORCHTITAN_CHUNKS:
        raise ValueError(
            f"N={n} is not divisible by num_chunks={TORCHTITAN_CHUNKS}: torchtitan's "
            "ChunkedLossWrapper splits the tokens into equal chunks"
        )
    loss, compile = _build_torchtitan(chunked, compiled)
    api = torchtitan_api()
    _STATE.update(titan_loss=loss, titan_api=api, lm_head=None)
    info: dict[str, Any] = {
        "torchtitan": torchtitan_version(),
        "torchtitan_api": api,
        "spmd_backend": _spmd_backend(),
        # Upcast with .float() before the cross-entropy: per chunk for the
        # wrapper (chunk_tokens x V x 4 bytes), the full logits for the plain loss.
        "logits_dtype": "float32",
        "loss_dtype": "float32",
        "compiled": compiled,
    }
    if chunked:
        info.update(
            num_chunks=TORCHTITAN_CHUNKS,
            chunk_tokens=n // TORCHTITAN_CHUNKS,
            # The GradAccumulator buffer of grad_hidden; grad_weight accumulates
            # in weight's dtype across the chunk backwards.
            grad_accum_dtype="float32",
        )
    if compiled:
        info.update(compile_scope="inner loss_fn", compile=compile)
    return info


def setup_torchtitan_chunked(case: Case) -> dict[str, Any]:
    return _setup_torchtitan(case, chunked=True, compiled=False)


def setup_torchtitan_chunked_compiled(case: Case) -> dict[str, Any]:
    return _setup_torchtitan(case, chunked=True, compiled=True)


def setup_torchtitan_ce(case: Case) -> dict[str, Any]:
    return _setup_torchtitan(case, chunked=False, compiled=False)


# --- verify: did the intended code path run? ----------------------------------------------


def profile_run(fn: Any) -> list[tuple[str, bool]]:
    """Run ``fn`` once under the profiler; its distinct events as sorted
    ``(name, on_device)`` pairs, ``on_device`` for CUDA kernel and memcpy events.

    One run serves several filters (``select``), so a verify that records more than
    one kind of event still costs one profiled call.
    """
    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return sorted({(e.name, e.device_type == DeviceType.CUDA) for e in prof.events()})


def select(events: list[tuple[str, bool]], keep: Any, kernels_only: bool = False) -> list[str]:
    """The sorted distinct names of ``events`` that ``keep`` accepts; with
    ``kernels_only``, of device events only."""
    return sorted(
        {name for name, on_device in events if keep(name) and (on_device or not kernels_only)}
    )


def profiled_events(fn: Any, keep: Any, kernels_only: bool = False) -> list[str]:
    """Run ``fn`` once under the profiler; the sorted names of the events ``keep``s."""
    return select(profile_run(fn), keep, kernels_only)


def _is_native_event(name: str) -> bool:
    return "_native::" in name


def native_events(fn: Any) -> list[str]:
    """The names of all torch._native events of one run of ``fn``.

    Not only LCE's: with the LCE override disabled, other torch._native overrides
    (e.g. ``_native::scatter_add__cutedsl_*`` in the backward) may still run, and
    whoever reads "no override" numbers should see them. Empty when none ran, or
    when the torch has no torch._native: that is the fact, not an error.
    """
    return profiled_events(fn, _is_native_event)


def verify_native(fn: Any) -> dict[str, Any]:
    """The torch._native overrides one run of a baseline (or its control) ran."""
    return {"native_events": native_events(fn)}


def override_eligible(case: Case, options: dict[str, Any]) -> bool:
    """Whether the build's override takes this case: CUDA sm_80+, bf16 or fp16, and
    the options resolved to what the kernel implements (compact, fp32 accumulation).
    """
    if not case.device.startswith("cuda") or case.dtype not in ("bfloat16", "float16"):
        return False
    import torch

    if torch.cuda.get_device_capability(torch.device(case.device)) < (8, 0):
        return False
    return options.get("acc_policy") == "compact" and options.get("acc_dtype") == "float32"


def _lce_events(names: list[str]) -> list[str]:
    return [name for name in names if "linear_cross_entropy" in name]


def verify_lce(fn: Any) -> dict[str, Any]:
    native = native_events(fn)
    events = _lce_events(native)
    case, options = _STATE["case"], _STATE["options"]
    if not events and override_eligible(case, options):
        registration = _STATE["registration"]
        if registration["registered"]:
            cause = "it is registered, yet did not run"
        elif registration["registered"] is None:
            cause = f"registration unknown: {registration['reason'].removeprefix('cannot tell: ')}"
        else:
            cause = f"not registered: {registration['reason']}"
        raise AssertionError(
            f"the torch._native LCE override did not run although eligible "
            f"({case.device}, {case.dtype}, acc_policy={options['acc_policy']}, "
            f"acc_dtype={options['acc_dtype']}); {cause}; the timing would be mislabelled"
        )
    return {"override_ran": bool(events), "override_events": events, "native_events": native}


def verify_lce_no_override(fn: Any) -> dict[str, Any]:
    native = native_events(fn)
    events = _lce_events(native)
    if events:
        raise AssertionError(f"the torch._native LCE override ran although disabled: {events}")
    return {"override_ran": False, "override_events": [], "native_events": native}


def _is_liger_event(name: str) -> bool:
    # Liger's own kernels only: a non-Liger cross_entropy kernel does not count.
    return "liger" in name.lower()


def verify_liger(fn: Any) -> dict[str, Any]:
    """The Liger (Triton) kernels of one run; none means the fused path did not run.

    Kernels only: the names of Liger's autograd function appear on the CPU side
    whether or not its kernels ran. The torch._native events come from the same run.
    """
    events = profile_run(fn)
    liger = select(events, _is_liger_event, kernels_only=True)
    if not liger:
        raise AssertionError(
            "no Liger kernel (a name containing 'liger') ran; the timing would be mislabelled"
        )
    return {"liger_events": liger, "native_events": select(events, _is_native_event)}


def verify_torchtitan_compiled(fn: Any) -> dict[str, Any]:
    """The events showing the compiled loss ran: its ``CompiledFunction`` (the
    autograd function of a compiled graph) or, on CUDA, an Inductor ``triton_``
    kernel. None means torchtitan ran its loss eagerly."""
    events = profile_run(fn)
    compiled = sorted(
        {
            *select(events, lambda name: "CompiledFunction" in name),
            *select(events, lambda name: name.startswith("triton_"), kernels_only=True),
        }
    )
    if not compiled:
        raise AssertionError(
            "no event shows the compiled loss ran (no 'CompiledFunction', no triton_ "
            "kernel); the timing would be mislabelled"
        )
    return {"compiled_events": compiled, "native_events": select(events, _is_native_event)}


# --- the spec -------------------------------------------------------------------

BOTH = {"fwd", "fwd_bwd"}

TORCH_EAGER = Implementation(
    "torch_eager",
    run_eager,
    modes=BOTH,
    setup=setup_eager,
    verify=verify_native,
    info={**CONTRACT, "role": "baseline"},
)


def _liger(name: str, run: Any, setup: Any, accum_dtype: str | None) -> Implementation:
    # fwd_bwd only: Liger computes the gradients inside its forward whenever the
    # inputs require grad, so a fwd number would include the backward work.
    return Implementation(
        name,
        run,
        modes={"fwd_bwd"},
        env_name="liger",
        setup=setup,
        verify=verify_liger,
        available=liger_available,
        packages=LIGER_PACKAGES,
        info={**CONTRACT, "accum_dtype": accum_dtype, "backend": "liger"},
    )


def _torchtitan(
    name: str, run: Any, setup: Any, modes: set[str], verify: Any = verify_native
) -> Implementation:
    return Implementation(
        name,
        run,
        modes=modes,
        env_name="torchtitan",
        setup=setup,
        verify=verify,
        available=(
            torchtitan_compiled_available
            if verify is verify_torchtitan_compiled
            else torchtitan_available
        ),
        packages=TORCHTITAN_PACKAGES,
        info={**CONTRACT, "backend": "torchtitan"},
    )


# fwd_bwd only for the chunked wrapper: it runs the chunk backwards whenever the
# hidden states require grad, which a leaf keeps under no_grad, so a fwd number
# would include the backward work.
TORCHTITAN = [
    _torchtitan(
        "torchtitan_chunked", run_torchtitan_chunked, setup_torchtitan_chunked, {"fwd_bwd"}
    ),
    _torchtitan(
        "torchtitan_chunked_compiled",
        run_torchtitan_chunked,
        setup_torchtitan_chunked_compiled,
        {"fwd_bwd"},
        verify=verify_torchtitan_compiled,
    ),
    _torchtitan("torchtitan_ce", run_torchtitan_ce, setup_torchtitan_ce, BOTH),
    TORCH_EAGER.as_control("torchtitan"),
]


LCE = OpSpec(
    name="lce",
    axes=[
        Axis("N", [1024, 2048, 4096, 8192, 16384]),
        Axis("H", [1024, 2048, 4096, 8192]),
        Axis("V", [32768, 65536, 131072, 262144]),
        Axis("ignore_frac", ["0", "0.1"], kind="str"),
    ],
    defaults={"N": 4096, "H": 4096, "V": 131072, "ignore_frac": "0"},
    dtypes=["bfloat16", "float16", "float32"],
    default_dtypes=["bfloat16"],
    make_inputs=make_inputs,
    implementations=[
        TORCH_EAGER,
        Implementation(
            "torch_compiled",
            run_compiled,
            modes=BOTH,
            setup=setup_compiled,
            verify=verify_native,
            available=compile_available,
            info={**CONTRACT, "compile": "torch.compile(fullgraph=True)"},
        ),
        Implementation(
            "torch_lce",
            run_lce,
            modes=BOTH,
            setup=setup_lce,
            verify=verify_lce,
            available=lce_available,
            packages=DSL_PACKAGES,
            info={**CONTRACT, "options": "LinearCrossEntropyOptions()"},
        ),
        Implementation(
            "torch_lce_no_override",
            run_lce,
            modes=BOTH,
            setup=setup_lce_no_override,
            verify=verify_lce_no_override,
            available=lce_no_override_available,
            packages=DSL_PACKAGES,
            info={**CONTRACT, "options": "LinearCrossEntropyOptions()"},
        ),
        _liger("liger", run_liger, setup_liger, "float32"),
        _liger("liger_native_accum", run_liger_native_accum, setup_liger_native_accum, None),
        TORCH_EAGER.as_control("liger"),
        *TORCHTITAN,
    ],
    reference=Implementation(
        "reference_fp32", run_reference, modes=BOTH, info={**CONTRACT, "dtype": "float32"}
    ),
    compare=compare,
    # 4x the largest error of torch_eager in bfloat16 against the fp32 reference,
    # measured on CUDA (RTX 5060 Ti, sm_120, torch 2.16.0a0, 2026-10-16) at
    # (N, H, V) = (4096, 4096, 131072) and (1024, 1024, 32768), ignore_frac 0 and 0.1:
    #   loss_rel_err       max 2.964e-03 -> 1.2e-02
    #   grad_hidden_rel_l2 max 2.450e-03 -> 9.8e-03
    #   grad_weight_rel_l2 max 2.373e-03 -> 9.5e-03
    # loss_abs_err is recorded, not checked: it scales with the loss.
    tolerances={
        "loss_rel_err": 1.2e-2,
        "grad_hidden_rel_l2": 9.8e-3,
        "grad_weight_rel_l2": 9.5e-3,
    },
    work=work,
    reference_fits=reference_fits,
)
