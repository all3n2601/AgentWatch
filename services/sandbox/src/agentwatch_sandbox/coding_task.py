"""A trusted local test tool fixture; this is not a security sandbox."""

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time


def summarize(values: list[int]) -> dict:
    return {
        "count": len(values),
        "total": sum(values),
        "mean": sum(values) / len(values) if values else None,
    }


def run_tests(iterations: int) -> dict:
    """Test empty, ordinary, and CPU-heavy inputs with a repeatable output."""
    assert summarize([]) == {"count": 0, "total": 0, "mean": None}
    assert summarize([1, 2, 3]) == {"count": 3, "total": 6, "mean": 2}
    value = b"agentwatch-coding-test"
    for _ in range(iterations):
        value = hashlib.sha256(value).digest()
    assert len(value) == 32
    return {"tests_passed": 3, "checksum": value.hex()}


def coding_task(iterations: int) -> dict:
    """Execute the fixed project's tests in a separate interpreter without a shell."""
    started_ns = time.time_ns()
    parent_start = time.process_time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    completed = subprocess.run(
        [sys.executable, "-m", "agentwatch_sandbox.coding_task", "--iterations", str(iterations)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = json.loads(completed.stdout)
    if result["tests_passed"] != 3:
        raise RuntimeError("Coding task did not pass all three tests")
    return {
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
        "cpu_seconds": (
            time.process_time()
            - parent_start
            + after.ru_utime
            + after.ru_stime
            - before.ru_utime
            - before.ru_stime
        ),
        "checksum": result["checksum"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    print(json.dumps(run_tests(args.iterations)))
