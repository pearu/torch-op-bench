"""Command line: ``torch-op-bench run|env|list|report``."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import env, runner
from .spec import MODES, OpSpec, resolve_op


def _split(items: Sequence[str] | None) -> list[str]:
    return [part for item in items or () for part in item.split(",") if part]


def _assignments(text: str) -> list[tuple[str, str]]:
    pairs = []
    for item in text.split(","):
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise ValueError(f"expected AXIS=VALUE, got {item!r}")
        pairs.append((name, value))
    return pairs


def _parse_values(spec: OpSpec, items: Sequence[str]) -> dict[str, list[Any]]:
    values = {}
    for item in items:
        name, sep, text = item.partition("=")
        if not sep:
            raise ValueError(f"expected AXIS=V1,V2,..., got {item!r}")
        axis = spec.axis(name)
        values[name] = [axis.parse(v) for v in text.split(",") if v]
    return values


def _parse_cases(spec: OpSpec, items: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {name: spec.axis(name).parse(value) for name, value in _assignments(item)} for item in items
    ]


def _load(op: str, pythonpath: Sequence[str]) -> OpSpec:
    for entry in reversed(pythonpath):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return resolve_op(op)


def cmd_run(args: argparse.Namespace) -> int:
    spec = _load(args.op, args.pythonpath)
    env_map = runner.parse_env_map(args.env)
    cases = runner.enumerate_cases(
        spec,
        device=args.device,
        dtypes=_split(args.dtype),
        sweep=_split(args.sweep),
        values=_parse_values(spec, args.values),
        cases=_parse_cases(spec, args.case),
    )
    runs = runner.enumerate_runs(spec, cases, _split(args.impl), _split(args.mode))
    settings = runner.Settings(
        warmup=args.warmup,
        repeat=args.repeat,
        flush_l2=args.flush_l2,
        seed=args.seed,
        timeout=args.timeout,
    )
    slug = args.device_slug or runner.device_slug(args.device, env_map["torch"])
    written = runner.run_sweep(
        spec,
        args.op,
        runs,
        settings=settings,
        results_dir=args.results,
        slug=slug,
        env_map=env_map,
        pythonpath=args.pythonpath,
        force=args.force,
    )
    counts = Counter(record["status"] for record in written)
    summary = ", ".join(f"{counts[s]} {s}" for s in ("ok", "unavailable", "error") if counts[s])
    print(
        f"{len(written)} new records ({summary or 'none'}) under {args.results}/{spec.name}/{slug}/"
    )
    return 0


def cmd_env(args: argparse.Namespace) -> int:
    print(json.dumps(env.collect(_split(args.package)), indent=2))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    spec = _load(args.op, args.pythonpath)
    default = spec.default_point
    print(f"op: {spec.name}")
    print("axes:")
    for axis in spec.axes:
        values = " ".join(str(v) for v in axis.values)
        print(f"  {axis.name} ({axis.kind}): {values}  [default {default[axis.name]}]")
    print(f"dtypes: {' '.join(spec.dtypes)}")
    print(f"default dtypes: {' '.join(spec.default_dtypes or spec.dtypes)}")
    print("implementations (availability checked in this interpreter, environment 'torch'):")
    for impl in [*spec.implementations, spec.reference]:
        if impl.env_name != "torch":
            status = f"not checked (environment {impl.env_name!r})"
        else:
            available = impl.available()
            status = "available" if available is True else f"unavailable: {available}"
            if impl.env_vars:
                # The worker sets them before importing torch; here they are not set.
                status += f" (checked without {', '.join(sorted(impl.env_vars))})"
        role = " (reference)" if impl is spec.reference else ""
        if impl.info.get("role") == "control":
            role = f" (control of {impl.info['control_of']})"
        modes = ",".join(m for m in MODES if m in impl.modes)
        print(f"  {impl.name}{role}  modes={modes}  env={impl.env_name}  {status}")
    tolerances = ", ".join(f"{k}<={v:g}" for k, v in spec.tolerances.items()) or "none"
    print(f"tolerances: {tolerances}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from . import report

    text = report.report(
        args.results,
        op=args.op,
        device_slug=args.device_slug,
        dtype=args.dtype,
        mode=args.mode,
        baseline=args.baseline,
        command=["torch-op-bench", *args.argv],
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


def _selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("results", help="results directory")
    parser.add_argument("--op", help="the op, when the results hold several")
    parser.add_argument("--device-slug", help="the device slug, when the results hold several")
    parser.add_argument("--dtype", help="one dtype (default: each in turn)")
    parser.add_argument("--mode", choices=MODES, help="one mode (default: each in turn)")
    parser.add_argument(
        "--baseline", default="torch_eager", help="the implementation others are compared with"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torch-op-bench", description="Benchmark torch ops and their counterparts."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    op_help = "in-package op name, or module:OBJECT"
    path_help = "directory prepended to sys.path, here and in workers (repeatable)"

    run = sub.add_parser("run", help="run a sweep, appending records under --results")
    run.add_argument("op", help=op_help)
    run.add_argument("--device", default="cpu", help="cpu, cuda or cuda:N (default: cpu)")
    run.add_argument("--device-slug", help="results subdirectory (default: from the device name)")
    run.add_argument(
        "--dtype",
        action="append",
        help="dtype(s), comma-separated (default: the op's default dtypes)",
    )
    run.add_argument("--sweep", action="append", help="axes to vary one at a time (default: all)")
    run.add_argument(
        "--values", action="append", default=[], metavar="AXIS=V1,V2", help="override axis values"
    )
    run.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="AXIS=V,AXIS=V",
        help="run this point (completed from the default point) instead of a sweep; repeatable",
    )
    run.add_argument("--impl", action="append", help="implementation(s) (default: all)")
    run.add_argument(
        "--mode", action="append", help=f"mode(s) of {', '.join(MODES)} (default: all)"
    )
    run.add_argument("--warmup", type=int, default=runner.Settings.warmup)
    run.add_argument("--repeat", type=int, default=runner.Settings.repeat)
    run.add_argument("--flush-l2", action="store_true", help="flush L2 between repetitions")
    run.add_argument("--seed", type=int, default=runner.Settings.seed)
    run.add_argument(
        "--timeout", type=float, default=runner.Settings.timeout, help="seconds per worker"
    )
    run.add_argument("--results", default="results", help="results directory (default: results)")
    run.add_argument("--force", action="store_true", help="rerun triples already recorded ok")
    run.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME=PYTHON",
        help="interpreter of an environment (repeatable; torch defaults to this one)",
    )
    run.add_argument("--pythonpath", action="append", default=[], help=path_help)
    run.set_defaults(func=cmd_run)

    env_cmd = sub.add_parser("env", help="print the provenance record of this interpreter")
    env_cmd.add_argument("--package", action="append", help="also report these package versions")
    env_cmd.set_defaults(func=cmd_env)

    list_cmd = sub.add_parser("list", help="show an op's axes, dtypes and implementations")
    list_cmd.add_argument("op", help=op_help)
    list_cmd.add_argument("--pythonpath", action="append", default=[], help=path_help)
    list_cmd.set_defaults(func=cmd_list)

    report_cmd = sub.add_parser("report", help="write the markdown tables of a results directory")
    _selection(report_cmd)
    report_cmd.add_argument("--out", help="markdown file to write (default: stdout)")
    report_cmd.set_defaults(func=cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    args.argv = argv
    try:
        return args.func(args)
    except (KeyError, ValueError, TypeError, ImportError, FileNotFoundError) as exc:
        parser.error(str(exc))
        return 2  # unreachable: parser.error exits


if __name__ == "__main__":
    sys.exit(main())
