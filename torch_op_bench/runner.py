"""Sweeps: enumerate (case, implementation, mode) triples and run each in a worker."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

from . import env, records
from .spec import MODES, Case, Implementation, OpSpec, to_json
from .worker import base_record

STDERR_LIMIT = 20_000


@dataclass
class Settings:
    warmup: int = 10
    repeat: int = 50
    flush_l2: bool = False
    seed: int = 0
    timeout: float = 600.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_env_map(items: Iterable[str] = ()) -> dict[str, str]:
    """``NAME=PYTHON`` items to a map; ``torch`` defaults to this interpreter."""
    env_map = {"torch": sys.executable}
    for item in items:
        name, sep, python = item.partition("=")
        if not sep or not name or not python:
            raise ValueError(f"expected NAME=PYTHON, got {item!r}")
        env_map[name] = python
    return env_map


def device_slug(device: str, python: str = sys.executable) -> str:
    """``cpu``, or the slugified name of the CUDA device as ``python``'s torch sees it.

    The name is asked of a separate interpreter so that this process never holds a
    CUDA context while workers measure.
    """
    if device == "cpu":
        return "cpu"
    index = int(device.partition(":")[2] or 0)
    code = f"import torch; print(torch.cuda.get_device_name({index}))"
    try:
        proc = subprocess.run(
            [python, "-c", code], capture_output=True, text=True, timeout=120, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return records.slugify(device)
    return records.slugify(proc.stdout.strip().splitlines()[-1])


def enumerate_cases(
    spec: OpSpec,
    *,
    device: str,
    dtypes: Sequence[str] | None = None,
    sweep: Sequence[str] | None = None,
    values: Mapping[str, Sequence[Any]] | None = None,
    cases: Sequence[Mapping[str, Any]] | None = None,
) -> list[Case]:
    """The cases of a sweep, for each dtype.

    With ``cases``, each given point (completed from the op's default point) is one
    case. Otherwise each axis in ``sweep`` (default: all) is varied over its values
    (or ``values[axis]``) one at a time, the other axes held at the default point;
    the default point appears once. Values for an axis that is not swept, or
    together with ``cases``, are an error rather than silently ignored.
    """
    dtypes = list(dtypes or spec.dtypes)
    unknown = set(dtypes) - set(spec.dtypes)
    if unknown:
        raise ValueError(f"op {spec.name!r} does not accept dtypes {sorted(unknown)}")
    values = dict(values or {})
    default = spec.default_point
    if cases:
        if values:
            raise ValueError("axis values apply to a sweep, not to explicit cases")
        points = [{**default, **case} for case in cases]
    else:
        swept = list(sweep or [axis.name for axis in spec.axes])
        unswept = sorted(set(values) - set(swept))
        if unswept:
            raise ValueError(f"values given for axes that are not swept: {unswept}")
        points = []
        for name in swept:
            for value in values.get(name, spec.axis(name).values):
                points.append({**default, name: value})
    for point in points:
        for name in point:
            spec.axis(name)
    result: list[Case] = []
    seen: set[str] = set()
    for dtype in dtypes:
        for point in points:
            case = Case(point, dtype, device)
            if case.key not in seen:
                seen.add(case.key)
                result.append(case)
    return result


def enumerate_runs(
    spec: OpSpec,
    cases: Sequence[Case],
    implementations: Sequence[str] | None = None,
    modes: Sequence[str] | None = None,
) -> list[tuple[Case, Implementation, str]]:
    """(case, implementation, mode) triples; an implementation runs only its modes."""
    impls = (
        [spec.implementation(name) for name in implementations]
        if implementations
        else list(spec.implementations)
    )
    modes = list(modes or MODES)
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise ValueError(f"unknown modes {unknown} (known: {', '.join(MODES)})")
    return [
        (case, impl, mode)
        for case in cases
        for impl in impls
        for mode in modes
        if mode in impl.modes
    ]


def make_payload(
    spec: OpSpec,
    op_ref: str,
    case: Case,
    impl: Implementation,
    mode: str,
    settings: Settings,
    *,
    slug: str,
    pythonpath: Sequence[str] = (),
) -> dict[str, Any]:
    return to_json(
        {
            "op": op_ref,
            "op_name": spec.name,
            "case": case.to_dict(),
            "implementation": impl.name,
            "mode": mode,
            "settings": settings.to_dict(),
            "pythonpath": [str(p) for p in pythonpath],
            "env_name": impl.env_name,
            "env_vars": dict(impl.env_vars),
            "packages": list(impl.packages),
            "device_slug": slug,
            "work": spec.work(case),
            "info": impl.info,
        }
    )


def failure_record(
    payload: dict[str, Any], python: str | None, reason: str, stderr: str | None = None
) -> dict[str, Any]:
    """An error record made by the runner when the worker gave none.

    Its ``env`` is this (the parent's) process', without torch so that the parent
    never creates a CUDA context, marked ``env_from: "runner"``.
    """
    record = base_record(payload, python)
    record.update(
        status="error",
        reason=reason,
        stderr=_tail(stderr),
        env=env.collect(payload["packages"], include_torch=False),
        env_from="runner",
    )
    return to_json(record)


def _tail(text: str | bytes | None) -> str | None:
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    if not text:
        return None
    return text[-STDERR_LIMIT:]


def launch(payload: dict[str, Any], python: str, timeout: float) -> dict[str, Any]:
    """Run one worker under ``python`` and return its record.

    The worker runs in a session (process group) of its own, and the whole group is
    killed when the worker exits or times out, so a process it spawned cannot outlive
    it to hold GPU memory during later measurements. Its output goes to temporary
    files, not pipes: the runner waits for the worker, not for the end of a stream
    that a surviving child may keep open.
    """
    child_env = dict(os.environ)
    child_env.update(payload["env_vars"])
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        try:
            proc = subprocess.Popen(
                [python, "-m", "torch_op_bench.worker"],
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=err,
                text=True,
                env=child_env,
                start_new_session=True,
            )
        except OSError as exc:
            return failure_record(payload, python, f"cannot launch {python!r}: {exc}")
        timed_out = False
        try:
            try:
                proc.stdin.write(json.dumps(payload))
                proc.stdin.close()
            except BrokenPipeError:  # the worker died before reading its payload
                pass
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            _kill_group(proc.pid)
            proc.wait()
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
    if timed_out:
        return failure_record(payload, python, f"timeout after {timeout:g} s", stderr)
    lines = stdout.strip().splitlines()
    try:
        record = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        reason = f"worker exited with code {proc.returncode} without a record"
        return failure_record(payload, python, reason, stderr)
    if record["status"] == "error":
        record["stderr"] = _tail(stderr)
    return record


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):  # the group is already gone
        pass


def _describe(record: dict[str, Any]) -> str:
    if record["status"] == "ok":
        median = records.stats(record["times_ms"])["median"]
        text = f"ok, median {median:.4g} ms"
        parity = record["parity"]
        if parity["status"] != "ok":
            text += f", parity {parity['status']}"
            if parity["failed"]:
                text += f" ({', '.join(parity['failed'])})"
        return text
    return f"{record['status']}: {record['reason']}"


def run_sweep(
    spec: OpSpec,
    op_ref: str,
    runs: Sequence[tuple[Case, Implementation, str]],
    *,
    settings: Settings,
    results_dir: str | Path,
    slug: str,
    env_map: Mapping[str, str],
    pythonpath: Sequence[str] = (),
    force: bool = False,
    out: TextIO = sys.stdout,
) -> list[dict[str, Any]]:
    """Run each triple in a worker, appending records to one new results file.

    Triples already recorded ``ok`` under ``<results>/<op>/<slug>/`` are skipped
    unless ``force``. Prints one line per triple. Returns the new records.
    """
    device_dir = Path(results_dir) / spec.name / slug
    done = (
        set() if force or not device_dir.exists() else records.completed(records.read(device_dir))
    )
    path = records.result_path(results_dir, spec.name, slug)
    written = []
    for index, (case, impl, mode) in enumerate(runs, 1):
        prefix = f"[{index}/{len(runs)}] {impl.name} {mode} {case.key}"
        payload = make_payload(
            spec, op_ref, case, impl, mode, settings, slug=slug, pythonpath=pythonpath
        )
        if records.key(base_record(payload, None)) in done:
            print(f"{prefix}: recorded, skipped", file=out, flush=True)
            continue
        python = env_map.get(impl.env_name)
        if python is None:
            reason = (
                f"no interpreter for environment {impl.env_name!r} "
                f"(use --env {impl.env_name}=PYTHON)"
            )
            record = failure_record(payload, None, reason)
        else:
            record = launch(payload, python, settings.timeout)
        records.append(path, record)
        written.append(record)
        print(f"{prefix}: {_describe(record)}", file=out, flush=True)
    return written
