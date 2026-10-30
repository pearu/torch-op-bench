"""A synthetic op for the report and plot tests: a spec only, never run.

Its records are written by ``synth_results``; nothing here imports torch.
"""

from torch_op_bench.spec import Axis, Implementation, OpSpec, ParityNote


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
        Axis("M", [64, 128, 256]),
        Axis("K", [16, 32]),
        Axis("N", [8, 16]),
        Axis("layout", ["a", "b"], kind="str"),
    ],
    defaults={"M": 128, "K": 32, "N": 16, "layout": "a"},
    dtypes=["float32", "bfloat16"],
    make_inputs=_never,
    implementations=[
        Implementation("base", _never, info={"role": "baseline"}),
        Implementation("chunked", _never),
    ],
    reference=Implementation("reference", _never),
    compare=_never,
    parity_notes=[NOTE, NEVER],
)
