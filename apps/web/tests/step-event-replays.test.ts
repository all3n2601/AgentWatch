import assert from "node:assert/strict";
import test from "node:test";
import {
  findReplayScenario,
  replayScenarios,
  SYNTHETIC_RUN_ID,
} from "../fixtures/step-event-replays";
import {
  activeSteps,
  failedSteps,
  feedFreshness,
  runOutcome,
  stepTotals,
} from "../lib/live-status";
import { readStepEvents } from "../lib/step-events";

function scenario(id: string) {
  const replay = findReplayScenario(id);
  assert.ok(replay, `missing scenario ${id}`);
  const view = readStepEvents(SYNTHETIC_RUN_ID, replay.records);
  return {
    replay,
    view,
    totals: stepTotals(view, replay.run),
    freshness: feedFreshness({
      ageMs: replay.receivedAgoMs,
      outcome: runOutcome(replay.run),
    }),
  };
}

type Wire = Record<string, unknown>;
// Mirrors the contract's per-state field rules so fixtures stay realistic;
// enforcing these on real records is the B01 service's responsibility.
function contractShapeError(record: Wire) {
  const has = (field: string) => record[field] !== null;
  switch (record.state) {
    case "queued":
      if (has("started_tick_ns") || has("finished_tick_ns") || has("reason"))
        return "queued has execution fields";
      break;
    case "running":
      if (!has("started_tick_ns") || has("finished_tick_ns") || has("reason"))
        return "running needs a start and no finish";
      break;
    case "completed":
      if (!has("started_tick_ns") || !has("finished_tick_ns") || has("reason"))
        return "completed needs start, finish and no reason";
      break;
    default:
      if (!has("finished_tick_ns") || !has("reason"))
        return "terminal failure needs finish and reason";
  }
  if (has("cpu_seconds") !== has("cpu_scope")) return "CPU scope mismatch";
  if (
    has("cpu_seconds") &&
    !["completed", "failed", "cancelled"].includes(String(record.state))
  )
    return "nonterminal CPU";
  if (has("cpu_seconds") && !has("started_tick_ns")) return "CPU without start";
  return null;
}

test("every scenario is readable, contract-shaped and uniquely named", () => {
  assert.equal(
    new Set(replayScenarios.map((s) => s.id)).size,
    replayScenarios.length,
  );
  for (const replay of replayScenarios) {
    const view = readStepEvents(SYNTHETIC_RUN_ID, replay.records);
    assert.equal(view.unreadableRecords, 0, replay.id);
    assert.equal(view.otherRunRecords, 0, replay.id);
    for (const record of replay.records)
      assert.equal(contractShapeError(record as Wire), null, replay.id);
  }
});

test("active-steps shows running and queued steps with partial totals", () => {
  const { view, totals, freshness } = scenario("active-steps");
  const active = activeSteps(view);
  assert.deepEqual(
    active.map((s) => [s.stepId, s.state]),
    [
      ["run-tests", "running"],
      ["summarize", "queued"],
    ],
  );
  assert.equal(active[0].startedObservedAt, "2026-10-14T12:00:01.500Z");
  assert.ok(active.every((s) => s.timings === null));
  assert.equal(totals.final, false);
  assert.equal(freshness, "fresh");
});

test("completed-run has final totals", () => {
  const { totals, freshness } = scenario("completed-run");
  assert.equal(totals.final, true);
  assert.ok(Math.abs(totals.duration.seconds - 4.2) < 1e-9);
  assert.ok(Math.abs((totals.cpu.seconds ?? NaN) - 1.2) < 1e-9);
  assert.equal(freshness, "closed");
});

test("failed-run reports the failed run and its failed step reason", () => {
  const { replay, view, totals } = scenario("failed-run");
  assert.equal(runOutcome(replay.run), "failed");
  assert.deepEqual(
    failedSteps(view).map((s) => [s.stepId, s.reason]),
    [["run-tests", "Tool timed out; cleanup status is recorded separately"]],
  );
  assert.equal(totals.final, true);
});

test("cancelled-before-start has duration but no queue, execution or CPU", () => {
  const { view, totals } = scenario("cancelled-before-start");
  const step = view.steps.find((s) => s.stepId === "run-tests")!;
  assert.equal(step.state, "cancelled");
  assert.equal(step.timings?.executionSeconds, null);
  assert.equal(step.cpuSeconds, null);
  assert.equal(totals.execution.unmeasuredSteps, 1);
  assert.equal(totals.cpu.seconds, null);
});

test("out-of-order holds the early completion behind a gap", () => {
  const { view, totals } = scenario("out-of-order");
  assert.equal(
    view.steps.find((s) => s.stepId === "run-tests")?.state,
    "queued",
  );
  assert.deepEqual(view.gaps, [
    { stepId: "run-tests", missingSequence: 1, pendingRevisions: 1 },
  ]);
  assert.equal(totals.final, false);
});

test("no-records has no steps, no receipt and non-final totals", () => {
  const { replay, view, totals, freshness } = scenario("no-records");
  assert.equal(replay.receivedAgoMs, null);
  assert.deepEqual(view.steps, []);
  assert.equal(view.lastObservedAt, null);
  assert.equal(freshness, "never");
  assert.equal(totals.final, false);
  assert.ok(totals.reasons.includes("No steps have been reported"));
});

test("stale-feed is stale while the run is open", () => {
  assert.equal(scenario("stale-feed").freshness, "stale");
});

test("missing-cpu leaves CPU unmeasured while timing totals are final", () => {
  const { totals } = scenario("missing-cpu");
  assert.equal(totals.final, true);
  assert.equal(totals.cpu.seconds, null);
  assert.equal(totals.cpu.unmeasuredSteps, 1);
});

test("zero-duration keeps a measured zero total", () => {
  const { totals } = scenario("zero-duration");
  assert.equal(totals.final, true);
  assert.equal(totals.duration.seconds, 0);
  assert.equal(totals.duration.measuredSteps, 1);
  assert.equal(totals.cpu.seconds, 0);
});

test("unknown scenario ids resolve to nothing", () => {
  assert.equal(findReplayScenario("missing"), undefined);
  assert.equal(findReplayScenario(undefined), undefined);
});
