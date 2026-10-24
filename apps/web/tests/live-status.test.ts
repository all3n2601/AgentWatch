import assert from "node:assert/strict";
import test from "node:test";
import { runSnapshotSchema } from "../lib/live-api-schema";
import { readRunSnapshot } from "../lib/live-snapshot";
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
import {
  OTHER_RUN_ID,
  runSnapshot,
  savedRecord,
  stepEvent,
  stepView,
} from "./live-fixtures";

type Event = ReturnType<typeof stepEvent>;
const queued = (stepId: string) => stepEvent(stepId, 0, "queued");
const running = (stepId: string, started = 1) =>
  stepEvent(stepId, 1, "running", { started });
const completed = (
  stepId: string,
  started: number,
  finished: number,
  cpu: number | null = null,
  cpuScope?: "process" | "process_and_children",
) => stepEvent(stepId, 2, "completed", { started, finished, cpu, cpuScope });

/** The view of a snapshot whose applied state is each step's highest listed revision. */
function view(events: Event[]) {
  const applied = new Map<string, Event>();
  for (const event of events)
    if ((applied.get(event.step_id)?.sequence ?? -1) < event.sequence)
      applied.set(event.step_id, event);
  const snapshot = runSnapshotSchema.parse(
    runSnapshot({
      steps: [...applied.values()].map(stepView),
      records: events.map((event, index) => savedRecord(event, index + 1)),
    }),
  );
  return readRunSnapshot(snapshot, snapshot.records);
}
const done = { state: "completed" as const, reason: null };

test("active and failed steps are selected by applied state", () => {
  const live = view([
    queued("q"),
    queued("r"),
    running("r"),
    queued("f"),
    stepEvent("f", 1, "failed", { finished: 1, reason: "Tool timed out" }),
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

test("gaps, other runs' records and unknown run state block final totals", () => {
  const early = stepEvent("b", 1, "running", { started: 1 });
  const snapshot = runSnapshotSchema.parse(
    runSnapshot({
      steps: [stepView(completed("a", 1, 3))],
      gaps: [
        {
          step_id: "b",
          missing_sequence: 0,
          first_pending_sequence: 1,
          expired: false,
        },
      ],
      records: [
        savedRecord(completed("a", 1, 3), 1),
        savedRecord(early, 2, "pending"),
        savedRecord(stepEvent("x", 0, "queued", { runId: OTHER_RUN_ID }), 3),
      ],
    }),
  );
  const live = readRunSnapshot(snapshot, snapshot.records);
  assert.deepEqual(stepTotals(live, null).reasons, [
    "Run state has not been reported",
    "1 step is waiting for a missing revision",
    "1 record belongs to another run",
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
      completed("b", 1, 2, 0.5, "process_and_children"),
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
      stepEvent("a", 1, "cancelled", {
        finished: 0.5,
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
