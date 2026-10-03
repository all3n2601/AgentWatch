"""Import existing experiment artifacts without rerunning the workload."""

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def publish(output: Path, kind: str, api_url: str) -> dict:
    body = {
        "kind": kind,
        "result": json.loads((output / "results.json").read_text()),
        "spans": [],
        "samples": [],
    }
    for field in ("spans", "samples"):
        path = output / f"{field}.jsonl"
        if path.exists():
            body[field] = [
                json.loads(line) for line in path.read_text().splitlines() if line.strip()
            ]
    request = urllib.request.Request(
        api_url.rstrip("/") + "/v1/runs",
        data=json.dumps(body, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["coding", "cpu"], required=True)
    parser.add_argument(
        "--api-url", default=os.environ.get("AGENTWATCH_API_URL", "http://127.0.0.1:8090")
    )
    args = parser.parse_args()
    try:
        result = publish(args.output, args.kind, args.api_url)
    except (urllib.error.URLError, ValueError, OSError) as error:
        parser.exit(1, f"Import failed: {error}. Local evidence remains in {args.output}.\n")
    print(
        f"Run {result['run_id']}: {'saved' if result['created'] else 'already saved'}; "
        f"added {result['added']['spans']} spans and {result['added']['samples']} samples."
    )


if __name__ == "__main__":
    main()
