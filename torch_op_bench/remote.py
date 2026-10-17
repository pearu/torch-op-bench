"""Helpers of ``scripts/remote-run.sh`` that run on the remote host.

``python -m torch_op_bench.remote count RESULTS`` prints how many records each
(op, device slug) has per status: what ``remote-run.sh status`` shows of a run in
progress. It reads the files line by line and skips a line it cannot parse (the
one a running worker is still appending), so it never fails on a live directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .records import STATUSES


def count(results_dir: str | Path) -> dict[tuple[str, str], Counter]:
    """Records per status, for each (op, device slug), over every ``.jsonl`` file."""
    counts: dict[tuple[str, str], Counter] = {}
    for path in sorted(Path(results_dir).rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue  # empty, or still being written
            if not isinstance(record, dict) or "status" not in record:
                continue
            key = (str(record.get("op")), str(record.get("device_slug")))
            counts.setdefault(key, Counter())[record["status"]] += 1
    return counts


def format_counts(counts: dict[tuple[str, str], Counter]) -> str:
    if not counts:
        return "no records yet"
    lines = []
    for (op, slug), statuses in sorted(counts.items()):
        known = [s for s in STATUSES if statuses[s]] + sorted(set(statuses) - set(STATUSES))
        parts = ", ".join(f"{statuses[s]} {s}" for s in known)
        lines.append(f"{op}/{slug}: {sum(statuses.values())} records ({parts})")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m torch_op_bench.remote")
    sub = parser.add_subparsers(dest="command", required=True)
    count_cmd = sub.add_parser("count", help="records per status under a results directory")
    count_cmd.add_argument("results")
    args = parser.parse_args(argv)
    if not Path(args.results).is_dir():
        print(f"no results directory at {args.results}")
        return 0
    print(format_counts(count(args.results)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
