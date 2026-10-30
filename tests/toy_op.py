"""A toy op for testing the core: ``y = (a @ b).sum()``.

Lives in the tests, not in the package. Each implementation exercises one path of
the worker: ``plain`` and ``chunked`` run, ``backward`` leaves its gradients on
the inputs, ``missing`` is unavailable, ``broken``
raises, ``flagged`` checks that its ``env_vars`` were set before torch was imported,
``elsewhere`` runs in the environment named ``other``, ``spawner`` leaves a child
process behind, ``described`` adds run-time facts to its ``info`` from ``setup``.
"""

import os
import sys

# Captured before torch is imported by this module: the worker must have applied the
# implementation's environment variables before importing anything that imports torch.
TORCH_PREIMPORTED = "torch" in sys.modules
FLAG_AT_IMPORT = os.environ.get("TOB_TOY_FLAG")

import torch  # noqa: E402

from torch_op_bench.spec import Axis, Case, Implementation, OpSpec, ParityNote  # noqa: E402


def make_inputs(case: Case, device, generator):
    dtype = getattr(torch, case.dtype)
    m, k, n = case.params["M"], case.params["K"], case.params["N"]
    a = torch.randn(m, k, dtype=dtype, device=device, generator=generator)
    if case.params["layout"] == "transposed":
        b = torch.randn(n, k, dtype=dtype, device=device, generator=generator).t()
    else:
        b = torch.randn(k, n, dtype=dtype, device=device, generator=generator)
    return a.requires_grad_(), b.requires_grad_()


def _finish(y, inputs, mode):
    if mode == "fwd":
        return (y,)
    return (y, *torch.autograd.grad(y, inputs))


def run_plain(inputs, mode):
    a, b = inputs
    return _finish((a @ b).sum(), inputs, mode)


def run_chunked(inputs, mode):
    a, b = inputs
    y = sum((chunk @ b).sum() for chunk in a.split(4))
    return _finish(y, inputs, mode)


def run_backward(inputs, mode):
    # The ``loss.backward()`` style: gradients land on the inputs' ``.grad``.
    a, b = inputs
    y = (a @ b).sum()
    if mode == "fwd":
        return (y,)
    y.backward()
    return (y, a.grad, b.grad)


def run_reference(inputs, mode):
    a, b = (x.detach().double().requires_grad_() for x in inputs)
    return _finish((a @ b).sum(), (a, b), mode)


def run_broken(inputs, mode):
    print("toy: about to fail", file=sys.stderr)
    raise RuntimeError("toy failure")


def verify_flag(fn):
    if TORCH_PREIMPORTED or FLAG_AT_IMPORT != "on":
        raise AssertionError(
            f"TOB_TOY_FLAG not set before torch import ({TORCH_PREIMPORTED=}, {FLAG_AT_IMPORT=})"
        )
    print("verify_flag: noise on stdout")
    fn()
    return {"flag": FLAG_AT_IMPORT}


def setup_spawner(case):
    # Leaves a child process behind; TOB_TOY_PIDFILE receives its pid. With
    # TOB_TOY_HANG set, the worker then hangs until it is killed.
    import subprocess
    import time

    child = subprocess.Popen(["sleep", "60"])
    with open(os.environ["TOB_TOY_PIDFILE"], "w") as f:
        f.write(str(child.pid))
    if os.environ.get("TOB_TOY_HANG"):
        time.sleep(60)


def setup_info(case):
    return {"resolved_chunk": 4, "M_seen": case.params["M"]}


def compare(outputs, ref_outputs, case):
    y, y_ref = outputs[0].double(), ref_outputs[0]
    metrics = {"y_abs_err": (y - y_ref).abs().item()}
    metrics["y_rel_err"] = metrics["y_abs_err"] / max(y_ref.abs().item(), 1e-30)
    for name, g, g_ref in zip(("a", "b"), outputs[1:], ref_outputs[1:], strict=False):
        metrics[f"grad_{name}_rel_l2"] = ((g.double() - g_ref).norm() / g_ref.norm()).item()
    return metrics


PLAIN = Implementation("plain", run_plain)

TOY = OpSpec(
    name="toy",
    axes=[
        Axis("M", [8, 16, 32]),
        Axis("K", [4, 8]),
        Axis("N", [4]),
        Axis("layout", ["contiguous", "transposed"], kind="str"),
    ],
    dtypes=["float32", "float64"],
    make_inputs=make_inputs,
    implementations=[
        PLAIN,
        PLAIN.as_control("other"),
        Implementation("chunked", run_chunked),
        Implementation(
            "described", run_plain, modes={"fwd"}, setup=setup_info, info={"chunk": "rows of 4"}
        ),
        Implementation("backward", run_backward),
        Implementation("missing", run_plain, available=lambda: "toy package not installed"),
        Implementation("broken", run_broken, modes={"fwd"}),
        Implementation("flagged", run_plain, env_vars={"TOB_TOY_FLAG": "on"}, verify=verify_flag),
        Implementation("elsewhere", run_plain, modes={"fwd"}, env_name="other"),
        Implementation("spawner", run_plain, modes={"fwd"}, setup=setup_spawner),
    ],
    reference=Implementation("reference", run_reference),
    compare=compare,
    tolerances={"y_rel_err": 1e-4, "grad_a_rel_l2": 1e-4, "grad_b_rel_l2": 1e-4},
    work=lambda case: {"flops": 2.0 * case.params["M"] * case.params["K"] * case.params["N"]},
    defaults={"M": 16},
    # A cause that holds in fwd_bwd only: a failure in fwd stays unexplained.
    parity_notes=[
        ParityNote(
            "chunked",
            None,
            lambda case, mode, metrics, torch_version: mode == "fwd_bwd",
            "chunked sums its chunks in another order in fwd_bwd (toy#1).",
        )
    ],
)
