import assert from "node:assert/strict";
import test from "node:test";
import { agentSchema } from "../lib/agent-schema";
import { experimentSchema } from "../lib/experiment-schema";

function evidence() {
  return {
    model: "qwen3:1.7b",
    status: "passed",
    messages: [
      {
        role: "assistant",
        content: "",
        tool_calls: [{ function: { name: "run_tests", arguments: {} } }],
      },
      { role: "tool", tool_name: "run_tests", content: '{"passed":true}' },
    ],
    inference: [
      {
        request_index: 0,
        wall_seconds: 1,
        metrics: { eval_count: 0, load_duration: 0 },
      },
    ],
  };
}

test("preserves real tool calls and distinguishes zero from missing server metrics", () => {
  const agent = agentSchema.parse(evidence());
  assert.deepEqual(agent.messages[0].tool_calls?.[0].function.arguments, {});
  assert.equal(agent.inference[0].metrics.eval_count, 0);
  assert.equal(agent.inference[0].metrics.eval_duration, undefined);
  assert.equal(agent.spans, undefined);
});

test("rejects negative or nonfinite inference measurements and unknown roles", () => {
  for (const wall_seconds of [-1, NaN, Infinity]) {
    const data = evidence();
    data.inference[0].wall_seconds = wall_seconds;
    assert.equal(agentSchema.safeParse(data).success, false);
  }
  assert.equal(
    agentSchema.safeParse({
      ...evidence(),
      messages: [{ role: "invented", content: "" }],
    }).success,
    false,
  );
});

test("span timestamps must be valid and ordered", () => {
  const span = {
    name: "llm.chat",
    start_time: "2026-10-02T12:00:01Z",
    end_time: "2026-10-02T12:00:02Z",
    context: { trace_id: "trace", span_id: "span" },
  };
  assert.equal(
    agentSchema.safeParse({ ...evidence(), spans: [span] }).success,
    true,
  );
  assert.equal(
    agentSchema.safeParse({
      ...evidence(),
      spans: [{ ...span, end_time: "2026-10-02T12:00:00Z" }],
    }).success,
    false,
  );
});

test("invalid optional agent evidence does not hide the measured coding run", () => {
  const step = {
    step_id: "clean",
    submitted_ns: 0,
    started_ns: 1e6,
    finished_ns: 1e9,
    duration_seconds: 1,
    queue_seconds: 0.001,
    execution_seconds: 0.999,
    cpu_seconds: 0.9,
    queue_fraction: 0.001,
    resource: "none",
    checksum: "abc",
    evidence: "timestamps",
    rule_version: "v1",
    threshold_seconds: 0.01,
  };
  const run = {
    run_id: "run",
    experiment: {
      worker_capacity: 1,
      iterations: 1,
      baseline_repeats: 3,
      workload: "tests",
      tests_per_step: 3,
    },
    baseline: {
      execution_cv: 0.01,
      queue_threshold_seconds: 0.01,
      steps: [step, step, step],
    },
    clean: step,
    contended: step,
    checks: { passed: true },
    limitations: [],
    agent: { model: "broken" },
  };
  const parsed = experimentSchema.parse(run);
  assert.equal(parsed.agent, undefined);
  assert.equal(parsed.checks.passed, true);
});
