"""Command line for the dataset quality gate; exits 1 when the gate fails."""

import argparse
import logging
import sys
from pathlib import Path

from agentwatch_validation.gate import (
    PASSED,
    GateError,
    freeze_schema,
    load_release,
    validate,
    write_schema,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentwatch_validation", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("validate", "check a release against the frozen schema"),
        ("freeze-schema", "write the frozen schema from a release and its contract"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--release", type=Path, required=True)
        command.add_argument("--schema", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        if args.command == "freeze-schema":
            release = load_release(args.release)
            write_schema(freeze_schema(release), args.schema, release.contract["schema_version"])
            print(f"schema: {args.schema}")
            return 0
        outcome = validate(args.release, args.schema)
    except GateError as error:
        print(f"Quality gate could not run: {error}", file=sys.stderr)
        return 1
    print(f"{outcome.status}: {outcome.result['release']} ({outcome.result_path})")
    print(f"result: {outcome.result_path}")  # read by the orchestrated validate stage
    for anomaly in outcome.result["anomalies"]:
        print(f"  anomaly {anomaly['feature']}: {anomaly['description']}", file=sys.stderr)
    for drift in outcome.result["drift"]:
        print(f"  drift {drift['feature']}: {drift['description']}")
    return 0 if outcome.status == PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())
