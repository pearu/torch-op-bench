"""A synthetic op for the report and plot tests: a spec only, never run.

Its records are written by ``synth_results``; nothing here imports torch.
"""

from torch_op_bench.spec import Axis, Implementation, OpSpec


def _never(*args, **kwargs):
    raise AssertionError("the synthetic op is never run")


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
    implementations=[Implementation("base", _never, info={"role": "baseline"})],
    reference=Implementation("reference", _never),
    compare=_never,
)
