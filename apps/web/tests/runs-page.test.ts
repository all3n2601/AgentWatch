import assert from "node:assert/strict";
import test from "node:test";
import type { Experiment, Step } from "../lib/experiment-schema";
import {
  compareSelection,
  contendedOverClean,
  DEFAULT_FILTERS,
  filterOptions,
  filterRows,
  keepOfferedFilters,
  liveRow,
  mergeOlderSaved,
  parseOlderSaved,
  parseRunsQuery,
  parseRunsSelection,
  parseRunsView,
  runsSelectionQuery,
  writeRunsQuery,
  savedRow,
  summarizeRows,
  toggleCompare,
  UNATTRIBUTED,
} from "../lib/runs-page";
import { buildComparison } from "../lib/compare-page";

// Synthetic runs for tests only; they are not recorded-run evidence.
const telemetry = (status: string) => ({
  sample_count: 4,
  valid_pressure_samples: 4,
  max_cpu_pressure_avg10_percent: 1.5,
  status,
});

function step(overrides: Partial<Step> = {}): Step {
  return {
    step_id: "baseline",
    submitted_ns: 0,
    started_ns: 1e9,
    finished_ns: 2e9,
    duration_seconds: 2,
    queue_seconds: 0.5,
    execution_seconds: 1.5,
    cpu_seconds: 1.2,
    queue_fraction: 0.25,
    resource: "none",
    checksum: "abc",
    evidence: "timestamps",
    rule_version: "cpu-queue-v1",
    threshold_seconds: 0.01,
    telemetry: telemetry("available"),
    ...overrides,
  };
}

function experiment(
  id: string,
  finishedMs: number,
  overrides: Partial<Experiment> = {},
): Experiment {
  return {
    run_id: id,
    kind: "coding",
    persisted: true,
    experiment: {
      worker_capacity: 1,
      iterations: 3,
      baseline_repeats: 3,
      workload: "fixed Python project: run three tests",
      tests_per_step: 3,
    },
    baseline: {
      execution_cv: 0.01,
      queue_threshold_seconds: 0.01,
      steps: [step(), step(), step()],
    },
    clean: step({ step_id: "clean", duration_seconds: 2 }),
    contended: step({
      step_id: "contended",
      // Starts at 1 ns so any positive finish time keeps the timing order valid.
      started_ns: 1,
      finished_ns: finishedMs * 1e6,
      duration_seconds: 8,
      queue_seconds: 6,
      execution_seconds: 2,
      queue_fraction: 0.75,
      resource: "cpu_sandbox",
    }),
    checks: { passed: true, queue_increased: true, stable_baseline: true },
    limitations: [],
    ...overrides,
  };
}

const live = (id: string, createdAt: string) =>
  liveRow({
    runId: id,
    taskId: "fix-window-count-r1",
    state: "running",
    createdAt,
  });

test("a saved row sums every step and keeps telemetry coverage distinct from zero", () => {
  const row = savedRow(experiment("a", 1_000));
  assert.equal(row.key, "saved:a");
  assert.ok(row.evidence);
  assert.equal(row.evidence.stepCount, 5);
  assert.equal(row.evidence.queueSeconds, 0.5 * 4 + 6);
  assert.equal(row.evidence.executionSeconds, 1.5 * 4 + 2);
  assert.equal(row.evidence.attributedSteps, 1);
  assert.deepEqual(row.evidence.ruleVersions, ["cpu-queue-v1"]);
  assert.equal(row.evidence.telemetry, "full");
  assert.deepEqual(row.evidence.checks, [
    { id: "queue_increased", passed: true },
    { id: "stable_baseline", passed: true },
  ]);
  assert.equal(row.evidence.evidenceHref.includes("run_id=a"), true);

  const partial = savedRow(
    experiment("b", 1_000, {
      clean: step({
        step_id: "clean",
        telemetry: telemetry("unavailable_or_no_samples"),
      }),
    }),
  );
  assert.equal(partial.evidence?.telemetry, "limited");
  assert.equal(partial.evidence?.telemetrySteps, 4);

  const steps = [step({ telemetry: undefined })];
  const none = savedRow(
    experiment("c", 1_000, {
      baseline: { execution_cv: 0, queue_threshold_seconds: 0, steps },
      clean: step({ telemetry: undefined }),
      contended: step({ telemetry: undefined }),
    }),
  );
  assert.equal(none.evidence?.telemetry, "none");
});

test("a live row has no evidence, so its timings stay unavailable", () => {
  const row = live(
    "11111111-1111-4111-8111-111111111111",
    "2026-11-05T17:00:00Z",
  );
  assert.equal(row.evidence, null);
  assert.equal(row.waitingShare, null);
  assert.equal(row.steps, null);
  assert.equal(row.key.startsWith("live:"), true);
});

test("the contended-over-clean ratio is a ratio of the two recorded steps", () => {
  const row = savedRow(experiment("a", 1_000));
  assert.ok(row.evidence);
  assert.equal(contendedOverClean(row.evidence), 4);
});

test("search covers workload and model; the resource filter never matches live rows", () => {
  const agent = savedRow(
    experiment("agent-run", 2_000, {
      agent: {
        model: "qwen3:1.7b",
        status: "passed",
        messages: [],
        limitations: [],
        inference: [],
      },
    }),
  );
  const unattributed = savedRow(
    experiment("plain", 1_000, {
      contended: step({ step_id: "contended", resource: "none" }),
    }),
  );
  const running = live("live-1", "2026-11-05T17:00:00Z");
  const rows = [agent, unattributed, running];

  const byModel = filterRows(rows, { ...DEFAULT_FILTERS, query: "QWEN3" });
  assert.deepEqual(
    byModel.map((row) => row.key),
    ["saved:agent-run"],
  );
  const byWorkload = filterRows(rows, {
    ...DEFAULT_FILTERS,
    query: "three tests",
  });
  assert.equal(byWorkload.length, 2);

  const cpu = filterRows(rows, { ...DEFAULT_FILTERS, resource: "cpu_sandbox" });
  assert.deepEqual(
    cpu.map((row) => row.key),
    ["saved:agent-run"],
  );
  const none = filterRows(rows, { ...DEFAULT_FILTERS, resource: UNATTRIBUTED });
  assert.deepEqual(
    none.map((row) => row.key),
    ["saved:plain"],
  );

  const liveOnly = filterRows(rows, { ...DEFAULT_FILTERS, source: "live" });
  assert.deepEqual(liveOnly, [running]);

  assert.deepEqual(filterOptions(rows), {
    kinds: ["agent", "coding"],
    resources: ["cpu_sandbox"],
    unattributed: true,
  });
});

test("summaries count each state group once", () => {
  const failing = savedRow(
    experiment("x", 1_000, { checks: { passed: false } }),
  );
  const summary = summarizeRows([
    savedRow(experiment("a", 1_000)),
    failing,
    live("l", "2026-11-05T17:00:00Z"),
  ]);
  assert.deepEqual(summary, {
    total: 3,
    done: 1,
    needsAttention: 1,
    inProgress: 1,
  });
});

test("comparison needs two saved runs and names differing configuration", () => {
  const olderRun = experiment("older", 1_000);
  const newerRun = experiment("newer", 5_000, {
    experiment: { ...olderRun.experiment, worker_capacity: 2 },
  });
  const older = savedRow(olderRun);
  const newer = savedRow(newerRun);
  assert.deepEqual(compareSelection([older]), {
    status: "incomplete",
    selected: 1,
  });

  const ready = compareSelection([newer, older]);
  assert.equal(ready.status, "ready");
  if (ready.status === "ready") {
    assert.equal(ready.baseline.id, "older");
    assert.equal(ready.candidate.id, "newer");
    assert.equal(ready.compatibility.verdict, "differs");
    assert.deepEqual(ready.compatibility.differing, ["Worker capacity"]);
    // The Runs selection and a direct Compare URL apply one rule to the pair.
    assert.deepEqual(
      ready.compatibility,
      buildComparison(olderRun, newerRun).compatibility,
    );
  }

  // A different kind is a recorded mismatch that Compare lists, not a hidden pair.
  const cpu = compareSelection([
    older,
    savedRow(experiment("cpu", 2_000, { kind: "cpu" })),
  ]);
  assert.equal(cpu.status, "ready");
  if (cpu.status === "ready")
    assert.ok(cpu.compatibility.differing.includes("Workload kind"));
  assert.equal(
    compareSelection([older, live("l", "2026-11-05T17:00:00Z")]).status,
    "incompatible",
  );
});

test("matching recorded settings stay unverified while the policy is unknown", () => {
  const selection = compareSelection([
    savedRow(experiment("a", 1_000)),
    savedRow(experiment("b", 2_000)),
  ]);
  assert.equal(selection.status, "ready");
  if (selection.status === "ready") {
    assert.equal(selection.compatibility.verdict, "unverified");
    assert.ok(selection.compatibility.unknown.includes("Concurrency policy"));
  }
});

test("comparison selection keeps at most two runs", () => {
  assert.deepEqual(toggleCompare([], "a"), ["a"]);
  assert.deepEqual(toggleCompare(["a", "b"], "c"), ["a", "b"]);
  assert.deepEqual(toggleCompare(["a", "b"], "a"), ["b"]);
});

test("the query keeps search text and rejects malformed live cursors", () => {
  assert.deepEqual(parseRunsQuery({ q: "fix-mean" }), {
    query: "fix-mean",
    liveBefore: null,
  });
  assert.deepEqual(parseRunsQuery({ q: ["a", "b"], liveBefore: "550" }), {
    query: "a",
    liveBefore: 550,
  });
  for (const bad of ["abc", "0", "-3", "1.5"])
    assert.equal(
      parseRunsQuery({ liveBefore: bad }).liveBeforeError,
      "Invalid live run cursor.",
    );
  assert.equal(parseRunsQuery({ q: "x".repeat(500) }).query.length, 200);
});

test("the view parameter selects a known tab and falls back to the list", () => {
  assert.equal(parseRunsView({}), "all");
  assert.equal(parseRunsView({ view: "compare" }), "compare");
  assert.equal(parseRunsView({ view: ["compare", "all"] }), "compare");
  for (const bad of ["", "Compare", "live", "../compare"])
    assert.equal(parseRunsView({ view: bad }), "all");
});

test("older saved pages are validated and merged without duplicates", () => {
  assert.deepEqual(parseOlderSaved({ experiments: [{ run_id: "x" }] }), {
    ok: false,
  });
  assert.deepEqual(parseOlderSaved(null), { ok: false });

  const body = JSON.parse(
    JSON.stringify({
      experiments: [experiment("a", 1_000), experiment("b", 900)],
      nextCursor: null,
    }),
  );
  const parsed = parseOlderSaved(body);
  assert.equal(parsed.ok, true);
  if (parsed.ok) {
    assert.equal(parsed.nextCursor, null);
    const merged = mergeOlderSaved(
      [savedRow(experiment("a", 1_000))],
      parsed.rows,
    );
    assert.deepEqual(
      merged.map((row) => row.key),
      ["saved:a", "saved:b"],
    );
  }
});

const SAVED_KEY = "saved:127d9be5-079e-4122-96fb-d5995bf71752";
const LIVE_KEY = "live:0b9e1f52-5c1d-4a4e-9d55-4c0f6a0f3a11";

test("the selection is read from the URL and invalid values fall back to the defaults", () => {
  assert.deepEqual(parseRunsSelection({}), {
    filters: DEFAULT_FILTERS,
    page: 0,
    previewKey: null,
    compareKeys: [],
  });
  assert.deepEqual(
    parseRunsSelection({
      q: "fix",
      source: "saved",
      state: "needs_attention",
      kind: "coding",
      resource: UNATTRIBUTED,
      page: "3",
      preview: SAVED_KEY,
      compare: `${SAVED_KEY},${LIVE_KEY}`,
    }),
    {
      filters: {
        query: "fix",
        source: "saved",
        group: "needs_attention",
        kind: "coding",
        resource: UNATTRIBUTED,
      },
      page: 2,
      previewKey: SAVED_KEY,
      compareKeys: [SAVED_KEY, LIVE_KEY],
    },
  );
  const bad = parseRunsSelection({
    source: "both",
    state: "Done",
    kind: "<script>",
    resource: "x".repeat(65),
    page: "0",
    preview: "saved:a b",
    compare: `${SAVED_KEY},${SAVED_KEY},bogus,${LIVE_KEY},live:third`,
  });
  assert.deepEqual(bad.filters, DEFAULT_FILTERS);
  assert.equal(bad.previewKey, null);
  assert.equal(bad.page, 0);
  // Duplicates and malformed keys are dropped; at most two runs are kept.
  assert.deepEqual(bad.compareKeys, [SAVED_KEY, LIVE_KEY]);
  for (const page of ["-1", "1.5", "abc", "100001"])
    assert.equal(parseRunsSelection({ page }).page, 0);
});

test("only non-default selection values are written, and they read back unchanged", () => {
  assert.deepEqual(
    runsSelectionQuery({
      filters: DEFAULT_FILTERS,
      page: 0,
      previewKey: null,
      compareKeys: [],
    }),
    {},
  );
  const selection = {
    filters: {
      query: "fix window",
      source: "live" as const,
      group: "done" as const,
      kind: "coding",
      resource: "cpu_sandbox",
    },
    page: 4,
    previewKey: LIVE_KEY,
    compareKeys: [SAVED_KEY, LIVE_KEY],
  };
  const query = runsSelectionQuery(selection);
  assert.deepEqual(query, {
    q: "fix window",
    source: "live",
    state: "done",
    kind: "coding",
    resource: "cpu_sandbox",
    page: "5",
    preview: LIVE_KEY,
    compare: `${SAVED_KEY},${LIVE_KEY}`,
  });
  assert.deepEqual(parseRunsSelection(query), selection);
});

test("writing the selection keeps unrelated parameters and clears defaults", () => {
  const url = new URL(
    "http://local/runs?view=compare&liveBefore=40&source=saved&page=3&q=old",
  );
  const selection = {
    filters: { ...DEFAULT_FILTERS, kind: "cpu" },
    page: 0,
    previewKey: null,
    compareKeys: [],
  };
  writeRunsQuery(url, new URLSearchParams(runsSelectionQuery(selection)));
  assert.equal(url.search, "?view=compare&liveBefore=40&kind=cpu");
});

test("kind and resource filters no loaded run offers fall back to all", () => {
  const options = {
    kinds: ["coding"],
    resources: ["cpu_sandbox"],
    unattributed: false,
  };
  const kept = {
    ...DEFAULT_FILTERS,
    kind: "coding",
    resource: "cpu_sandbox",
    source: "saved" as const,
  };
  assert.deepEqual(keepOfferedFilters(kept, options), kept);
  assert.deepEqual(
    keepOfferedFilters(
      { ...kept, kind: "cpu", resource: UNATTRIBUTED },
      options,
    ),
    { ...kept, kind: "all", resource: "all" },
  );
  assert.equal(
    keepOfferedFilters(
      { ...kept, resource: UNATTRIBUTED },
      { ...options, unattributed: true },
    ).resource,
    UNATTRIBUTED,
  );
});

test("evidence links name the run, whether it is in history or a local artifact", () => {
  const id = "127d9be5-079e-4122-96fb-d5995bf71752";
  const saved = savedRow(experiment(id, 1_000)).evidence;
  assert.equal(
    saved?.evidenceHref,
    `/api/experiments/coding/artifact?file=results.json&run_id=${id}`,
  );
  // A local artifact is bound to the run the page showed, never "whatever is on disk now".
  const local = savedRow(experiment(id, 1_000, { persisted: false })).evidence;
  assert.equal(
    local?.evidenceHref,
    `/api/experiments/coding/artifact?file=results.json&local_run_id=${id}`,
  );
});
