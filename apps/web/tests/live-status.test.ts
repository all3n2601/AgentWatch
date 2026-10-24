import assert from "node:assert/strict";
import test from "node:test";
import {
  SYNTHETIC_RUN_ID,
  syntheticStepEvent as event,
} from "../fixtures/step-event-replays";
import {
  activeSteps,
  failedSteps,
  feedFreshness,
  formatReceiptAge,
  LIVE_FEED_STALE_AFTER_MS,
  receiptAgeMs,
  runOutcome,
  stepTotals,
} from "../lib/live-status";
import { readStepEvents } from "../lib/step-events";

const at = (second: number) =>
  `2026-10-14T12:00:${String(second).padStart(2, "0")}Z`;
const queued = (step_id: string) =>
  event({ step_id, sequence: 0, state: "queued", observed_at: at(0) });
const running = (step_id: string, started_s = 1) =>
  event({
    step_id,
    sequence: 1,
    state: "running",
    observed_at: at(1),
    started_s,
  });
const completed = (
  step_id: string,
  started_s: number,
  finished_s: number,
  cpu_seconds: number | null = null,
) =>
  event({
    step_id,
    sequence: 2,
    state: "completed",
    observed_at: at(5),
    started_s,
    finished_s,
    cpu_seconds,
    cpu_scope: cpu_seconds === null ? null : "process",
  });
const view = (records: unknown[]) => readStepEvents(SYNTHETIC_RUN_ID, records);
const done = { state: "completed" as const, reason: null };

test("active and failed steps are selected by applied state", () => {
  const live = view([
    queued("q"),
    queued("r"),
    running("r"),
    queued("f"),
    event({
      step_id: "f",
      sequence: 1,
      state: "failed",
      observed_at: at(2),
      finished_s: 1,
      reason: "Tool timed out",
    }),
  ]);
  assert.deepEqual(
    activeSteps(live).map((s) => s.stepId),
    ["q", "r"],
  );
  assert.deepEqual(
    failedSteps(live).map((s) => s.stepId),
    ["f"],
  );
});

test("run outcome comes from the reported run state, not step states", () => {
  assert.equal(runOutcome(null), "unknown");
  assert.equal(runOutcome({ state: "queued", reason: null }), "active");
  assert.equal(runOutcome({ state: "running", reason: null }), "active");
  assert.equal(runOutcome({ state: "failed", reason: "x" }), "failed");
  assert.equal(runOutcome({ state: "cancelled", reason: "x" }), "cancelled");
  assert.equal(runOutcome(done), "completed");
});

test("feed freshness separates never-received, fresh, stale and closed", () => {
  assert.equal(feedFreshness({ ageMs: null, outcome: "active" }), "never");
  assert.equal(
    feedFreshness({ ageMs: LIVE_FEED_STALE_AFTER_MS, outcome: "active" }),
    "fresh",
  );
  assert.equal(
    feedFreshness({ ageMs: LIVE_FEED_STALE_AFTER_MS + 1, outcome: "unknown" }),
    "stale",
  );
  assert.equal(
    feedFreshness({ ageMs: 10 * LIVE_FEED_STALE_AFTER_MS, outcome: "failed" }),
    "closed",
  );
});

test("receipt age advances only with the browser's monotonic clock", () => {
  const receipt = { ageAtAnchorMs: 2_000, anchorMonotonicMs: 50_000 };
  assert.equal(receiptAgeMs(null, 60_000), null);
  assert.equal(receiptAgeMs(receipt, 50_000), 2_000);
  assert.equal(receiptAgeMs(receipt, 57_500), 9_500);
  // A monotonic reading before the anchor never makes the record younger.
  assert.equal(receiptAgeMs(receipt, 49_000), 2_000);
});

test("receipt age is shown as coarse relative text", () => {
  assert.equal(formatReceiptAge(0), "Just now");
  assert.equal(formatReceiptAge(4_999), "Just now");
  assert.equal(formatReceiptAge(42_000), "42 s ago");
  assert.equal(formatReceiptAge(120_000), "2 min ago");
  assert.equal(formatReceiptAge(2 * 3_600_000 + 1), "2 h ago");
});

test("totals are final only for a closed run with every step terminal", () => {
  const live = view([
    queued("a"),
    running("a", 1),
    completed("a", 1, 3, 1.5),
    queued("b"),
    running("b", 0.5),
    completed("b", 0.5, 1.5, 0.25),
  ]);
  const totals = stepTotals(live, done);
  assert.equal(totals.final, true);
  assert.deepEqual(totals.reasons, []);
  assert.deepEqual(totals.queue, {
    seconds: 1.5,
    measuredSteps: 2,
    unmeasuredSteps: 0,
  });
  assert.equal(totals.execution.seconds, 3);
  // Both steps are submitted at tick 0, so their windows overlap: [0, 3] and [0, 1.5].
  assert.equal(totals.duration.seconds, 3);
  assert.deepEqual(totals.cpu, {
    seconds: 1.75,
    scope: "process",
    measuredSteps: 2,
    unmeasuredSteps: 0,
  });
});

test("an active step keeps totals partial and excludes it from sums", () => {
  const live = view([
    queued("a"),
    running("a", 1),
    completed("a", 1, 3),
    queued("b"),
    running("b"),
  ]);
  const totals = stepTotals(live, { state: "running", reason: null });
  assert.equal(totals.final, false);
  assert.deepEqual(totals.reasons, [
    "Run is still active",
    "1 step is still queued or running",
  ]);
  assert.equal(totals.duration.seconds, 3);
  assert.equal(totals.duration.measuredSteps, 1);
});

test("gaps, unread records and unknown run state block final totals", () => {
  const live = view([
    queued("a"),
    running("a", 1),
    completed("a", 1, 3),
    { record_type: "step_event" },
    event({
      step_id: "b",
      sequence: 1,
      state: "running",
      observed_at: at(1),
      started_s: 1,
    }),
  ]);
  assert.deepEqual(stepTotals(live, null).reasons, [
    "Run state has not been reported",
    "1 step is waiting for a missing revision",
    "1 record could not be read",
  ]);
});

test("a closed run with unfinished steps or no steps is not final", () => {
  assert.deepEqual(
    stepTotals(view([queued("a")]), {
      state: "failed",
      reason: "Run aborted",
    }).reasons,
    ["1 step is still queued or running"],
  );
  assert.deepEqual(stepTotals(view([]), done).reasons, [
    "No steps have been reported",
  ]);
});

test("missing CPU or mixed CPU scopes leave CPU totals unmeasured", () => {
  const missing = stepTotals(
    view([
      queued("a"),
      running("a", 1),
      completed("a", 1, 2, 0.5),
      queued("b"),
      running("b", 1),
      completed("b", 1, 2),
    ]),
    done,
  );
  assert.equal(missing.final, true);
  assert.deepEqual(missing.cpu, {
    seconds: null,
    scope: null,
    measuredSteps: 1,
    unmeasuredSteps: 1,
  });

  const mixed = stepTotals(
    view([
      queued("a"),
      running("a", 1),
      completed("a", 1, 2, 0.5),
      queued("b"),
      running("b", 1),
      {
        ...completed("b", 1, 2, 0.5),
        cpu_scope: "process_and_children",
      },
    ]),
    done,
  );
  assert.equal(mixed.cpu.seconds, null);
  assert.equal(mixed.cpu.measuredSteps, 2);
});

test("never-started steps count toward duration but not queue or execution", () => {
  const totals = stepTotals(
    view([
      queued("a"),
      event({
        step_id: "a",
        sequence: 1,
        state: "cancelled",
        observed_at: at(1),
        finished_s: 0.5,
        reason: "Cancelled before a worker started",
      }),
    ]),
    { state: "cancelled", reason: "Operator cancelled the run" },
  );
  assert.equal(totals.final, true);
  assert.deepEqual(totals.queue, {
    seconds: 0,
    measuredSteps: 0,
    unmeasuredSteps: 1,
  });
  assert.equal(totals.duration.seconds, 0.5);
});
