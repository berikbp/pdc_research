"""Command line entry points. Run from the repository root."""

import argparse
import json
import subprocess

from .baseline import run_baseline
from .common import write_json
from .config import experiment_plan, load_configs, load_experiment
from .hardware import hardware_report
from .runner import run
from .workloads import prepare

DEFAULT_EXPERIMENT = "configs/experiments/pilot.json"
DEFAULT_MANIFEST = "data/workloads/pilot.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Record visible hardware and installed tools")
    doctor.add_argument("--output", default="runs/hardware.json")
    prep = commands.add_parser("prepare", help="Prepare aligned FLORES sentence groups")
    prep.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    prep.add_argument("--source", required=True, help="Extracted FLORES root containing dev/ and devtest/")
    prep.add_argument("--output", default=DEFAULT_MANIFEST)
    for name in ("plan", "run", "baseline"):
        sub = commands.add_parser(name)
        sub.add_argument("--machine", required=True)
        sub.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
        if name == "run":
            sub.add_argument("--manifest", default=DEFAULT_MANIFEST)
        if name in ("run", "baseline"):
            sub.add_argument(
                "--output", required=True, help="New output directory; existing directories refused"
            )
    return parser


def dispatch(args: argparse.Namespace) -> None:
    if args.command == "doctor":
        report = hardware_report()
        write_json(args.output, report)
        print(json.dumps(report, indent=2))
        return
    if args.command == "prepare":
        prepare(load_experiment(args.experiment), args.source, args.output)
        print(f"Created {args.output}")
        return
    machine, experiment = load_configs(args.machine, args.experiment)
    if args.command == "plan":
        print(json.dumps(experiment_plan(machine, experiment), indent=2))
    elif args.command == "run":
        run(machine, experiment, args.manifest, args.output)
    elif args.command == "baseline":
        run_baseline(machine, experiment, args.output)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dispatch(args)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Error: {exc}\n")
