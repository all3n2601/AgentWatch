"""Print the newest published dataset release, for scripts and later pipeline steps.

python -m agentwatch_data.releases --output .data/datasets
"""

import argparse
import sys
from pathlib import Path

from agentwatch_data.build import published_releases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(".data/datasets"))
    args = parser.parse_args(argv)
    releases = published_releases(args.output)
    if not releases:
        print(f"No published release in {args.output}; run make dataset first", file=sys.stderr)
        return 1
    print(releases[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
