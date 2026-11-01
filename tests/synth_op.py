"""A synthetic op for the report and plot tests: a spec only, never run.

Its records are written by ``synth_results``; nothing here imports torch.
"""

from torch_op_bench.spec import Axis, Implementation, OpSpec, ParityNote, WorkUnit


def _never(*args, **kwargs):
    raise AssertionError("the synthetic op is never run")


# chunked's planted parity failure (synth_results.PARITY_FAIL, both dtypes) has a cause
# in bfloat16 only; its float32 failure stays unexplained. The second note names a
# metric chunked never fails, so it is never shown.
NOTE = ParityNote(
    "chunked",
    ("grad_rel",),
    lambda case, mode, metrics, torch_version: case.dtype == "bfloat16",
    "a planted cause, in bfloat16 only (synth#1).",
)
NEVER = ParityNote(
    "chunked", ("loss_rel_err",), lambda *args: True, "chunked never fails loss_rel_err."
)


SYNTH = OpSpec(
    name="synth",
    axes=[
        Axis("M", [64, 128, 256], description="the rows"),
        Axis("K", [16, 32], description="the inner size"),
        Axis("N", [8, 16], description="the columns"),
        Axis("layout", ["a", "b"], kind="str", description="the layout"),
    ],
    defaults={"M": 128, "K": 32, "N": 16, "layout": "a"},
    dtypes=["float32", "bfloat16"],
    make_inputs=_never,
    implementations=[
        Implementation("base", _never, info={"role": "baseline"}, description="the plain product"),
        Implementation("chunked", _never, description="the product in chunks"),
    ],
    reference=Implementation("reference", _never),
    compare=_never,
    tolerances={"loss_rel_err": 1e-2, "grad_rel": 1e-2},
    # As synth_results writes it.
    work=lambda case, mode: {
        "tokens": case.params["M"],
        "flops": 2 * case.params["M"] * case.params["K"] * case.params["N"],
    },
    parity_notes=[NOTE, NEVER],
    description="a synthetic product, never run",
    work_units=[
        WorkUnit("tokens", "tokens", "the rows, M", "tokens/s", 1),
        WorkUnit("flops", "FLOP", "2·M·K·N", "TFLOP/s", 1e12),
    ],
    metric_definitions={
        "loss_rel_err": "the planted loss error",
        "grad_rel": "the planted gradient error",
    },
    tolerance_basis="The tolerances are planted.",
)
