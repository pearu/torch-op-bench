import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


def _has_cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def pytest_collection_modifyitems(config, items):
    if _has_cuda():
        return
    skip = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def tests_dir() -> Path:
    return TESTS_DIR


@pytest.fixture
def synth(tmp_path) -> Path:
    """A synthetic results directory (see ``synth_results``); returns its root."""
    import synth_results

    root = tmp_path / "results"
    synth_results.write(root)
    return root


@pytest.fixture
def synth_strict(tmp_path) -> Path:
    """The strict counterpart of ``synth`` (see ``synth_results.write_strict``)."""
    import synth_results

    root = tmp_path / "results-strict"
    synth_results.write_strict(root)
    return root
