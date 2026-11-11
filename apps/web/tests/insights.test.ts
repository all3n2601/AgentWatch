import assert from "node:assert/strict";
import test from "node:test";
import type { DatasetRun } from "../lib/dataset-runs";
import type { Manifest } from "../lib/dataset-schema";
import {
  generateInsight,
  groundInsight,
  insightConfig,
  overviewFacts,
  requestCompletion,
  runFacts,
  userPrompt,
  type Fact,
} from "../lib/insights";
import type { DiagnosisStep } from "../lib/unified-runs";

// Synthetic values for tests only; they are not recorded-run evidence.
const totals = {
  runs: 56,
  steps: 392,
  passed: 44,
  queue: 80.21,
  execution: 119.96,
  queueFraction: 0.401,
  attributed: 56,
};

const manifest = {
  rows: 5953,
  label_resource: { none: 4904, inference: 326 },
  telemetry_status: { present: 777, empty: 5176 },
} as unknown as Manifest;

const replay = (
  condition: string,
  seconds: number,
  experiment = "exp-1",
): DatasetRun =>
  ({
    runId: `${condition}-${seconds}`,
    experiment,
    taskGroup: "task-a",
    replayCondition: condition,
    durationSeconds: seconds,
    recordedAt: "2026-11-05T10:00:00.000Z",
  }) as DatasetRun;

const facts: Fact[] = [
  {
    id: "F1",
    text: "Queue delay is 40.1% of profiled step time (80.21 s queued).",
  },
  { id: "F2", text: "There are 5,953 dataset steps." },
];

test("overview facts are numbered, include the caveats, and omit dataset facts when absent", () => {
  const without = overviewFacts(totals);
  assert.deepEqual(
    without.map((f) => f.id),
    ["F1", "F2", "F3", "F4"],
  );
  assert.match(without[1].text, /40\.1%/);
  assert.match(without.at(-1)!.text, /No accuracy or confidence figure/);

  const withDataset = overviewFacts(totals, {
    manifest,
    release: "steps-test",
    runs: [replay("clean", 2), replay("clean", 2), replay("inference", 10)],
  });
  const text = withDataset.map((f) => f.text).join("\n");
  assert.match(text, /5,953 labelled steps/);
  assert.match(text, /1,049 of 5,953 dataset steps \(18%\)/);
  assert.match(text, /not a controlled comparison/);
  assert.match(text, /×5\.0/);
});

test("run facts never describe a live step's resource as an attribution", () => {
  const steps: DiagnosisStep[] = [
    {
      stepId: "tool-1",
      type: "tool",
      state: "completed",
      durationSeconds: 1,
      queueSeconds: 0.5,
      executionSeconds: 0.5,
      waitingShare: 0.5,
      resource: "cpu_sandbox",
      basis: "ran_on",
    },
    {
      stepId: "never-started",
      type: "tool",
      state: "failed",
      durationSeconds: null,
      queueSeconds: null,
      executionSeconds: null,
      waitingShare: null,
      resource: null,
      basis: "none",
    },
  ];
  const live = runFacts(steps, "live")
    .map((f) => f.text)
    .join("\n");
  assert.match(live, /not an attribution of waiting/);
  assert.match(live, /Live runs store no per-step attribution/);
  assert.match(live, /1 have a measured queue delay and 1 do not/);
  const saved = runFacts(
    steps.map((s) => ({
      ...s,
      basis: s.basis === "ran_on" ? "attributed" : s.basis,
    })),
    "saved",
  )
    .map((f) => f.text)
    .join("\n");
  assert.match(saved, /the saved rule attributed it to cpu sandbox/);
});

test("grounding keeps cited, supported points and drops the rest", () => {
  const insight = groundInsight(
    JSON.stringify({
      summary: "Queueing is 40.1% of step time.",
      points: [
        { text: "Queue delay is 40.1% of step time.", facts: ["F1"] },
        { text: "Queue delay is 62% of step time.", facts: ["F1"] },
        { text: "There are 5,953 steps.", facts: ["F9"] },
        { text: "No citation at all.", facts: [] },
        { text: "5953 dataset steps exist.", facts: ["F2"] },
      ],
      caveats: [
        "Figures come from 80.21 s of queueing.",
        "Invented 99 caveat.",
      ],
    }),
    facts,
  );
  assert.deepEqual(
    insight.points.map((p) => p.facts),
    [["F1"], ["F2"]],
  );
  assert.equal(insight.summary, "Queueing is 40.1% of step time.");
  assert.deepEqual(insight.caveats, ["Figures come from 80.21 s of queueing."]);
  assert.equal(insight.dropped, 4);
});

test("a summary with an unsupported figure is withheld, not shown", () => {
  const insight = groundInsight(
    JSON.stringify({
      summary: "Waiting is 90% of time.",
      points: [],
      caveats: [],
    }),
    facts,
  );
  assert.equal(insight.summary, null);
  assert.equal(insight.dropped, 1);
});

test("replies wrapped in prose or fences parse; non-JSON and wrong shapes are errors", () => {
  const wrapped =
    'Here you go:\n```json\n{"points":[{"text":"There are 5,953 steps.","facts":["F2"]}]}\n```';
  assert.equal(groundInsight(wrapped, facts).points.length, 1);
  assert.throws(
    () => groundInsight("no json here", facts),
    /did not return JSON/,
  );
  assert.throws(
    () => groundInsight('{"points":"not an array"}', facts),
    /expected format/,
  );
});

test("config needs a URL and model, trims the slash, and rejects non-http URLs", () => {
  assert.equal(insightConfig({}), null);
  assert.equal(insightConfig({ AGENTWATCH_INSIGHTS_URL: "http://x/v1" }), null);
  assert.equal(
    insightConfig({
      AGENTWATCH_INSIGHTS_URL: "file:///etc",
      AGENTWATCH_INSIGHTS_MODEL: "m",
    }),
    null,
  );
  assert.deepEqual(
    insightConfig({
      AGENTWATCH_INSIGHTS_URL: " http://localhost:20128/v1/ ",
      AGENTWATCH_INSIGHTS_MODEL: "my-model",
    }),
    { baseUrl: "http://localhost:20128/v1", apiKey: "", model: "my-model" },
  );
});

const config = {
  baseUrl: "http://gateway.test/v1",
  apiKey: "secret-key",
  model: "m",
};
const completion = (content: string) =>
  new Response(JSON.stringify({ choices: [{ message: { content } }] }), {
    status: 200,
  });

test("the request carries the facts and the key, and nothing else about the environment", async () => {
  let seen: { url: string; init: RequestInit } | undefined;
  const result = await requestCompletion(config, facts, async (url, init) => {
    seen = { url: String(url), init: init! };
    return completion("{}");
  });
  assert.equal(result.ok, true);
  assert.equal(seen!.url, "http://gateway.test/v1/chat/completions");
  assert.equal(
    (seen!.init.headers as Record<string, string>).Authorization,
    "Bearer secret-key",
  );
  const body = JSON.parse(String(seen!.init.body));
  assert.equal(body.model, "m");
  assert.equal(body.messages[1].content, userPrompt(facts));
  assert.doesNotMatch(String(seen!.init.body), /secret-key/);
});

test("no Authorization header is sent when no key is configured", async () => {
  let headers: Record<string, string> = {};
  await requestCompletion(
    { ...config, apiKey: "" },
    facts,
    async (_url, init) => {
      headers = init!.headers as Record<string, string>;
      return completion("{}");
    },
  );
  assert.equal("Authorization" in headers, false);
});

test("failures report a safe message: no key, no endpoint body", async () => {
  const rejected = await requestCompletion(
    config,
    facts,
    async () => new Response("internal detail secret-key", { status: 401 }),
  );
  assert.equal(rejected.ok, false);
  if (!rejected.ok) {
    assert.equal(rejected.kind, "rejected");
    assert.match(rejected.message, /401/);
    assert.doesNotMatch(rejected.message, /secret-key|internal detail/);
  }
  const down = await requestCompletion(config, facts, async () => {
    throw new TypeError("fetch failed: secret-key");
  });
  assert.equal(down.ok, false);
  if (!down.ok) {
    assert.equal(down.kind, "unavailable");
    assert.doesNotMatch(down.message, /secret-key/);
  }
  const odd = await requestCompletion(
    config,
    facts,
    async () => new Response('{"unexpected":true}', { status: 200 }),
  );
  assert.equal(odd.ok, false);
  if (!odd.ok) assert.equal(odd.kind, "invalid_response");
});

test("generateInsight returns grounded output with the facts it used", async () => {
  const result = await generateInsight(config, facts, async () =>
    completion(
      JSON.stringify({
        summary: "Queue delay is 40.1% of profiled step time.",
        points: [{ text: "There are 5,953 dataset steps.", facts: ["F2"] }],
      }),
    ),
  );
  assert.equal(result.ok, true);
  if (result.ok) {
    assert.deepEqual(result.facts, facts);
    assert.equal(result.insight.points.length, 1);
    assert.equal(result.model, "m");
  }
  const garbled = await generateInsight(config, facts, async () =>
    completion("sorry"),
  );
  assert.equal(garbled.ok, false);
  if (!garbled.ok) assert.equal(garbled.kind, "invalid_response");
});
