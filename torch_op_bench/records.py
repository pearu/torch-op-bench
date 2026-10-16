"""Results as JSON Lines records (R5).

One record per (case, implementation, mode); files under
``<results>/<op>/<device-slug>/<timestamp>-<host>.jsonl``, one per ``run``
invocation, append-only. Files from several hosts merge by concatenation.
"""

from __future__ import annotations

import datetime
import json
import re
import socket
import statistics
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SCHEMA = 1
STATUSES = ("ok", "unavailable", "error")

Key = tuple[str, str, str, str, str]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "unknown"


def key(record: dict[str, Any]) -> Key:
    """(op, device slug, case key, implementation, mode): what resume and merge match on."""
    return (
        record["op"],
        record["device_slug"],
        record["case_key"],
        record["implementation"],
        record["mode"],
    )


def result_path(
    results_dir: str | Path,
    op: str,
    device_slug: str,
    *,
    timestamp: datetime.datetime | None = None,
    host: str | None = None,
) -> Path:
    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc)
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    host = slugify(host or socket.gethostname())
    return Path(results_dir) / op / device_slug / f"{stamp}-{host}.jsonl"


def append(path: str | Path, record: dict[str, Any]) -> None:
    """Append one record to ``path``, creating it and its directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def read(path: str | Path) -> list[dict[str, Any]]:
    """Read the records of a ``.jsonl`` file, or of every one under a directory."""
    path = Path(path)
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    records = []
    for file in files:
        with file.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("schema") != SCHEMA:
                    raise ValueError(
                        f"{file}:{lineno}: schema {record.get('schema')!r}, expected {SCHEMA}"
                    )
                records.append(record)
    return records


def completed(records: Iterable[dict[str, Any]]) -> set[Key]:
    """Keys that have an ``ok`` record: a resumed sweep skips them."""
    return {key(r) for r in records if r["status"] == "ok"}


def merge(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """De-duplicate by key: the latest ``ok`` record, else the latest record.

    "Latest" is by the record's ``timestamp``, then by position. Warns when the kept
    records come from different torch-op-bench git hashes.
    """
    chosen: dict[Key, tuple[tuple, dict[str, Any]]] = {}
    for position, record in enumerate(records):
        rank = (record["status"] == "ok", record.get("timestamp") or "", position)
        k = key(record)
        if k not in chosen or rank > chosen[k][0]:
            chosen[k] = (rank, record)
    merged = [record for _, record in sorted(chosen.values(), key=lambda item: item[0][2])]
    hashes = {((r.get("env") or {}).get("torch_op_bench") or {}).get("hash") for r in merged} - {
        None
    }
    if len(hashes) > 1:
        warnings.warn(
            f"records come from different torch-op-bench commits: {sorted(hashes)}",
            stacklevel=2,
        )
    return merged


def stats(samples: Sequence[float]) -> dict[str, float | int | None]:
    """Median, min, mean and inter-quartile range of timing samples."""
    if not samples:
        return {"n": 0, "median": None, "min": None, "mean": None, "iqr": None}
    if len(samples) > 1:
        q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive")
        iqr = q3 - q1
    else:
        iqr = 0.0
    return {
        "n": len(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "mean": statistics.fmean(samples),
        "iqr": iqr,
    }
