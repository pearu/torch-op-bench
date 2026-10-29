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


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    # With a device, the cuda tests go to one xdist worker (--dist loadgroup, in
    # pyproject.toml): the peak-memory and subprocess-CUDA-memory tests measure the
    # device, and a second worker on it would perturb them. Tests use the first GPU
    # only. tryfirst: xdist's own hook on the worker, which turns the mark into the
    # node id's "@cuda" suffix the scheduler groups by, runs before this one otherwise.
    mark = pytest.mark.xdist_group("cuda") if _has_cuda() else pytest.mark.skip("no CUDA device")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(mark)


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    # On an xdist worker, xdist's own hook has now sent the scheduler the ids with
    # "@cuda", which it groups by. From here on the tests run, are reported and are
    # recorded by pytest-testmon under their plain ids, as in a serial run, so that
    # testmon keeps one record per test whichever way the suite ran.
    for item in session.items:
        if item.nodeid.endswith("@cuda"):
            item._nodeid = item.nodeid.removesuffix("@cuda")


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
