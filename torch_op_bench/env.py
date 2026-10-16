"""Provenance: what produced a record (R4).

Everything returned is JSON-serialisable. A missing torch, a torch without CUDA, a
missing ``nvidia-smi`` or a package not installed as a git checkout give ``null``
fields, not errors.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch_op_bench


def utc_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def driver_version() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return out.splitlines()[0].strip() if out else None


def git_info(root: Path | None = None) -> dict[str, Any]:
    """The git hash and dirty flag of the checkout the package is imported from."""
    if root is None:
        root = Path(torch_op_bench.__file__).resolve().parent.parent
    if not (root / ".git").exists() or shutil.which("git") is None:
        return {"hash": None, "dirty": None}
    commit = _run(["git", "rev-parse", "HEAD"], cwd=root)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root)
    return {"hash": commit, "dirty": None if status is None else bool(status)}


def package_versions(packages: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"version": None, "git_version": None, "cuda": None, "cudnn": None, "devices": []}
    info: dict[str, Any] = {
        "version": torch.__version__,
        "git_version": getattr(torch.version, "git_version", None),
        "cuda": torch.version.cuda,
        "cudnn": None,
        "devices": [],
    }
    if not torch.cuda.is_available():
        return info
    try:
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception:  # cuDNN missing or broken: not ours to fail on
        info["cudnn"] = None
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        info["devices"].append(
            {
                "index": index,
                "name": props.name,
                "capability": f"{props.major}.{props.minor}",
                "sm_count": props.multi_processor_count,
                "total_memory": props.total_memory,
            }
        )
    return info


def collect(packages: Iterable[str] = (), *, include_torch: bool = True) -> dict[str, Any]:
    """The provenance record of this Python process.

    With ``include_torch=False`` the ``torch`` field is ``None`` and torch is not
    touched (no CUDA context is created).
    """
    return {
        "hostname": socket.gethostname(),
        "timestamp": utc_timestamp(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch_info() if include_torch else None,
        "driver": driver_version(),
        "torch_op_bench": {"version": torch_op_bench.__version__, **git_info()},
        "packages": package_versions(packages),
    }
