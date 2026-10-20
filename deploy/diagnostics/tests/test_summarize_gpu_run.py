import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "summarize_gpu_run.py"
spec = importlib.util.spec_from_file_location("summarize_gpu_run", SCRIPT)
summarize_gpu_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize_gpu_run)


def event(step_id, state, observed_at, parent="agent-run"):
    return {
        "kind": "record",
        "payload": {
            "record_type": "step_event",
            "step_id": step_id,
            "state": state,
            "parent_step_id": parent,
            "observed_at": observed_at,
        },
    }


def write_journal(probe, entries):
    run = probe / "agent-runs/fix-mean-empty-20261020T000000Z-abc123"
    run.mkdir(parents=True)
    (run / "live-journal.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n{broken\n"
    )


def test_journal_windows_cover_executed_non_root_steps(tmp_path):
    write_journal(
        tmp_path,
        [
            {"kind": "registration", "payload": {}},
            event("agent-run", "running", "2026-10-20T00:00:00+00:00", parent=None),
            event("model-call-1", "queued", "2026-10-20T00:00:01+00:00"),
            event("model-call-1", "running", "2026-10-20T00:00:01.5+00:00"),
            event("model-call-1", "completed", "2026-10-20T00:00:04+00:00"),
            event("tool-2-rejected", "queued", "2026-10-20T00:00:05+00:00"),
            event("tool-2-rejected", "failed", "2026-10-20T00:00:05+00:00"),
            event("tool-3-run_tests", "running", "2026-10-20T00:00:06+00:00"),
            event("tool-3-run_tests", "cancelled", "2026-10-20T00:00:07+00:00"),
            {"kind": "record", "payload": {"record_type": "resource_sample"}},
        ],
    )
    windows, malformed, source = summarize_gpu_run.find_windows(tmp_path)
    assert source == "agent steps"
    assert malformed == 1
    assert [(name, (end - start).total_seconds()) for name, start, end in windows] == [
        ("model-call-1", 2.5),
        ("tool-3-run_tests", 1.0),
    ]


def test_naive_timestamps_are_malformed_not_guessed(tmp_path):
    write_journal(tmp_path, [event("model-call-1", "running", "2026-10-20T00:00:01")])
    windows, malformed, _ = summarize_gpu_run.find_windows(tmp_path)
    assert (windows, malformed) == ([], 2)


def test_pre_runner_probes_still_read_demo_spans(tmp_path):
    spans = tmp_path / "llm-demo"
    spans.mkdir()
    (spans / "agent-spans.jsonl").write_text(
        json.dumps(
            {
                "name": "llm.chat",
                "start_time": "2026-10-13T22:19:00+00:00",
                "end_time": "2026-10-13T22:19:02+00:00",
            }
        )
        + "\n"
    )
    windows, malformed, source = summarize_gpu_run.find_windows(tmp_path)
    assert (source, malformed, [name for name, *_ in windows]) == ("agent spans", 0, ["llm.chat"])


def test_gpu_probe_preflight_uses_an_endpoint_the_api_serves():
    """Regression: the probe once checked /health, which does not exist (PR #27)."""
    import re

    from agentwatch_worker.app import app

    script = (SCRIPT.parent / "gpu-sampled-llm-run.sh").read_text()
    paths = set(re.findall(r'"\$api_url(/[\w/]+)"', script))
    served = {getattr(route, "path", None) for route in app.routes}
    assert paths and paths <= served, f"{paths - served} are not served by the worker API"
