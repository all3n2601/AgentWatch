import assert from "node:assert/strict";
import test from "node:test";
import {
  askAssistant,
  citedRecords,
  groundAnswer,
} from "../lib/assistant-chat";
import { runTool, type AssistantSources } from "../lib/assistant-tools";
import type { DatasetRun } from "../lib/dataset-runs";
import type { InsightConfig } from "../lib/insights";
import type { Ready } from "../lib/overview-model";
import type { RunView } from "../lib/unified-runs";

// Synthetic runs and endpoint replies for tests only; they are not recorded-run evidence.
const FAILED = "aaaaaaaa-1111-4111-8111-111111111111";
const PASSED = "bbbbbbbb-2222-4222-8222-222222222222";
const CLEAN = "cccccccc-3333-4333-8333-333333333333";

const run = (runId: string, overrides: Partial<DatasetRun>): DatasetRun => ({
  runId,
  kind: "agent_replay",
  workload: "replay",
  taskGroup: "fix-window-count",
  experiment: "exp-1",
  replayCondition: "clean",
  replayIndex: 1,
  injectionMethod: "none",
  agentModel: "qwen3:1.7b",
  recordedAt: "2026-01-02T00:00:00.000Z",
  passed: true,
  steps: 4,
  durationSeconds: 2,
  queueSeconds: 0.1,
  executionSeconds: 1.9,
  excessSteps: 0,
  labels: { none: 4 },
  stepTypes: { model_call: 3, tool: 1 },
  excessByStepType: {},
  topLabel: "none",
  telemetryPresent: 1,
  ...overrides,
});

const runs = [
  run(FAILED, {
    replayCondition: "mixed",
    passed: false,
    durationSeconds: 20,
    excessSteps: 2,
  }),
  run(PASSED, { replayCondition: "cpu", durationSeconds: 4 }),
  run(CLEAN, {}),
];

const sources = (
  overrides: Partial<AssistantSources> = {},
): AssistantSources => ({
  dataset: async () => ({
    ok: true,
    value: {
      status: "ready",
      runs,
      manifest: { dataset_version: "steps-test+abc" },
    } as unknown as Ready,
  }),
  importMap: async () => ({}),
  liveRuns: async () => ({
    ok: false,
    message: "The live API could not be reached.",
  }),
  savedRuns: async () => ({ ok: true, value: [] }),
  runView: async () => null,
  ...overrides,
});

const config: InsightConfig = {
  baseUrl: "http://gateway.test/v1",
  apiKey: "",
  model: "test-model",
};

/** A fake chat endpoint that replays scripted responses and records each request body. */
function fakeEndpoint(replies: { status?: number; body?: unknown }[]) {
  const requests: Record<string, unknown>[] = [];
  const fetchImpl = (async (_url: string, init?: RequestInit) => {
    requests.push(JSON.parse(String(init?.body)));
    const reply = replies[Math.min(requests.length - 1, replies.length - 1)];
    return new Response(JSON.stringify(reply.body ?? {}), {
      status: reply.status ?? 200,
    });
  }) as typeof fetch;
  return { fetchImpl, requests };
}
const answer = (content: string) => ({
  body: { choices: [{ message: { content } }] },
});
const toolCall = (name: string, args: object) => ({
  body: {
    choices: [
      {
        message: {
          content: null,
          tool_calls: [
            {
              id: "call-1",
              type: "function",
              function: { name, arguments: JSON.stringify(args) },
            },
          ],
        },
      },
    ],
  },
});

test("runTool refuses unknown tools, malformed JSON and out-of-range or extra arguments", async () => {
  for (const [name, args] of [
    ["drop_tables", "{}"],
    ["search_runs", "{not json"],
    ["search_runs", '{"limit":100}'],
    ["search_runs", '{"sql":"select 1"}'],
    ["get_run", '{"run_id":"../../etc"}'],
  ]) {
    const output = await runTool(name, args, sources());
    assert.match(output.content, /^\{"error"/, `${name} ${args}`);
    assert.deepEqual(output.records, []);
  }
});

test("search_runs returns failed dataset runs with their slowdown and reports an unreachable store", async () => {
  const output = await runTool(
    "search_runs",
    '{"outcome":"failed"}',
    sources(),
  );
  const body = JSON.parse(output.content);
  assert.equal(body.total_matches, 1);
  assert.equal(body.runs[0].short_id, "aaaaaaaa");
  // 20 s against the experiment's clean median of 2 s.
  assert.equal(body.runs[0].slowdown_vs_clean, "10.0×");
  assert.deepEqual(body.notes, [
    "Live runs: The live API could not be reached.",
  ]);
  assert.equal(output.records[0].href, null);
});

test("get_run resolves a short id and links an imported run to its live record", async () => {
  const live = "dddddddd-4444-4444-8444-444444444444";
  const output = await runTool(
    "get_run",
    '{"run_id":"AAAAAAAA"}',
    sources({ importMap: async () => ({ [FAILED]: live }) }),
  );
  const body = JSON.parse(output.content);
  assert.equal(body.dataset.run_id, FAILED);
  assert.equal(body.dataset.outcome, "failed");
  assert.equal(output.records[0].href, `/runs/${live}`);
  // The imported live run's steps are labelled as the same run, not a different one.
  const withSteps = await runTool(
    "get_run",
    '{"run_id":"aaaaaaaa"}',
    sources({
      importMap: async () => ({ [FAILED]: live }),
      runView: async (id) =>
        id === live
          ? ({
              source: "live",
              snapshot: {
                run_id: live,
                state: "completed",
                registration: { resources: [] },
                steps: [
                  {
                    event: {
                      step_id: "model-1",
                      step_type: "model_call",
                      state: "completed",
                      resource_id: null,
                    },
                    timings: {
                      duration_seconds: 2,
                      queue_seconds: 0.5,
                      execution_seconds: 1.5,
                      queue_fraction: 0.25,
                    },
                  },
                ],
              },
            } as unknown as RunView)
          : null,
    }),
  );
  assert.match(
    JSON.parse(withSteps.content).stored.relation,
    /^Live import of dataset run aaaaaaaa/,
  );
  assert.match(
    (await runTool("get_run", '{"run_id":"eeeeeeee"}', sources())).content,
    /No run matches/,
  );
});

test("groundAnswer drops sentences with figures or run ids the evidence lacks", () => {
  const evidence =
    '{"runs":[{"short_id":"aaaaaaaa","duration_s":20}],"failed":1}';
  const { text, removed } = groundAnswer(
    "Run aaaaaaaa took 20 s. It used 95% of the GPU.\n- Run ffff9999 also failed.\nOne run failed.",
    evidence,
  );
  assert.equal(text, "Run aaaaaaaa took 20 s.\nOne run failed.");
  assert.equal(removed, 2);
});

test("citedRecords keeps only runs the answer names, once each", () => {
  const record = {
    runId: FAILED,
    shortId: "aaaaaaaa",
    source: "dataset" as const,
    title: "t",
    status: "failed",
    href: null,
  };
  assert.deepEqual(
    citedRecords("See aaaaaaaa and AAAAAAAA.", [record, record]),
    [record],
  );
  assert.deepEqual(citedRecords("No ids here.", [record]), []);
});

test("askAssistant runs the requested tool, sends its result back and returns a grounded answer", async () => {
  const { fetchImpl, requests } = fakeEndpoint([
    toolCall("search_runs", { outcome: "failed" }),
    answer(
      "1 run failed: aaaaaaaa, 10.0× slower than clean. It is 99% certain to be GPU bound.",
    ),
  ]);
  const reply = await askAssistant(
    config,
    [{ role: "user", content: "Which runs failed?" }],
    sources(),
    fetchImpl,
  );
  assert.ok(reply.ok);
  if (!reply.ok) return;
  assert.equal(reply.text, "1 run failed: aaaaaaaa, 10.0× slower than clean.");
  assert.equal(reply.removed, 1);
  assert.deepEqual(
    reply.records.map((r) => r.shortId),
    ["aaaaaaaa"],
  );
  assert.deepEqual(
    reply.tools.map((t) => [t.name, t.ok]),
    [["search_runs", true]],
  );
  const second = requests[1].messages as {
    role: string;
    tool_call_id?: string;
  }[];
  assert.equal(second.at(-1)?.role, "tool");
  assert.equal(second.at(-1)?.tool_call_id, "call-1");
  assert.ok(Array.isArray(requests[0].tools));
  // The gateway must answer with one JSON completion, not an event stream.
  assert.equal(requests[0].stream, false);
});

test("askAssistant falls back to fixed context when the endpoint rejects tool definitions", async () => {
  const { fetchImpl, requests } = fakeEndpoint([
    { status: 400, body: { error: "tools unsupported" } },
    answer("1 run failed."),
  ]);
  const reply = await askAssistant(
    config,
    [{ role: "user", content: "Any failures?" }],
    sources(),
    fetchImpl,
  );
  assert.ok(reply.ok);
  assert.equal(requests[1].tools, undefined);
  if (reply.ok)
    assert.deepEqual(
      reply.tools.map((t) => t.name),
      ["get_overview", "search_runs"],
    );
});

test("askAssistant stops a model that only calls tools, and reports endpoint failures safely", async () => {
  const looping = fakeEndpoint([toolCall("get_overview", {})]);
  const stuck = await askAssistant(
    config,
    [{ role: "user", content: "Hi" }],
    sources(),
    looping.fetchImpl,
  );
  assert.equal(stuck.ok, false);
  // Tools are offered for at most four rounds; the fifth request must answer.
  assert.equal(looping.requests.length, 5);
  assert.equal(looping.requests[4].tools, undefined);

  const denied = fakeEndpoint([
    { status: 401, body: { error: "secret detail" } },
  ]);
  const failure = await askAssistant(
    config,
    [{ role: "user", content: "Hi" }],
    sources(),
    denied.fetchImpl,
  );
  assert.equal(failure.ok, false);
  if (!failure.ok) assert.doesNotMatch(failure.message, /secret detail/);
});
