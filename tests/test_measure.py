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
