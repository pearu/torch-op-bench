import pytest
import torch

from torch_op_bench.measure import peak_memory, time_callable


class Counter:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1


@pytest.mark.parametrize("flush_l2", [False, True])
def test_time_callable_cpu(flush_l2):
    fn = Counter()
    samples = time_callable(
        fn, device="cpu", warmup=3, repeat=5, flush_l2=flush_l2, flush_bytes=1024
    )
    assert len(samples) == 5
    assert fn.calls == 8
    assert all(isinstance(s, float) and s >= 0 for s in samples)


def test_time_callable_measures_ms():
    import time

    samples = time_callable(
        lambda: time.sleep(0.01), device="cpu", warmup=0, repeat=2, flush_l2=False
    )
    assert all(9 <= s < 1000 for s in samples)


def test_peak_memory_cpu_is_null():
    fn = Counter()
    assert peak_memory(fn, "cpu") == {"peak_bytes": None, "baseline_bytes": None}
    assert fn.calls == 0


@pytest.mark.cuda
@pytest.mark.parametrize("flush_l2", [False, True])
def test_time_callable_cuda(flush_l2):
    x = torch.randn(256, 256, device="cuda")
    fn = Counter()

    def work():
        fn()
        x @ x

    samples = time_callable(
        work, device="cuda", warmup=2, repeat=4, flush_l2=flush_l2, flush_bytes=1 << 20
    )
    assert len(samples) == 4 and fn.calls == 6
    assert all(s > 0 for s in samples)


@pytest.mark.cuda
def test_peak_memory_cuda():
    x = torch.ones(1024, 1024, device="cuda")
    result = peak_memory(lambda: x * 2, "cuda")
    assert result["baseline_bytes"] >= x.nbytes
    assert result["peak_bytes"] >= result["baseline_bytes"] + x.nbytes


def test_clear_grads_walks_containers():
    from torch_op_bench.measure import clear_grads

    tensors = [torch.ones(2, requires_grad=True) for _ in range(4)]
    for t in tensors:
        t.sum().backward()
    clear_grads((tensors[0], [tensors[1], {"x": tensors[2]}], "not a tensor", 3))
    assert [t.grad is None for t in tensors] == [True, True, True, False]


def test_reset_runs_before_every_call_outside_timing():
    events = []
    samples = time_callable(
        lambda: events.append("fn"),
        device="cpu",
        warmup=1,
        repeat=2,
        flush_l2=False,
        reset=lambda: events.append("reset"),
    )
    assert len(samples) == 2
    assert events == ["reset", "fn"] * 3


def test_on_phase_names_warmup_and_timing():
    events = []
    time_callable(
        lambda: events.append("fn"),
        device="cpu",
        warmup=2,
        repeat=1,
        flush_l2=False,
        on_phase=events.append,
    )
    assert events == ["warmup", "fn", "fn", "timing", "fn"]


def test_backward_style_does_not_accumulate():
    import toy_op

    a = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    _, ga, gb = toy_op.run_plain((a, b), "fwd_bwd")
    from torch_op_bench.measure import clear_grads

    time_callable(
        lambda: toy_op.run_backward((a, b), "fwd_bwd"),
        device="cpu",
        warmup=2,
        repeat=3,
        flush_l2=False,
        reset=lambda: clear_grads((a, b)),
    )
    torch.testing.assert_close(a.grad, ga)
    torch.testing.assert_close(b.grad, gb)


@pytest.mark.cuda
def test_peak_memory_cuda_reset_first():
    x = torch.ones(1024, 1024, device="cuda", requires_grad=True)
    x.grad = torch.ones_like(x)

    def reset():
        x.grad = None

    result = peak_memory(lambda: x * 2, "cuda", reset=reset)
    assert x.grad is None
    assert result["baseline_bytes"] < 2 * x.nbytes
