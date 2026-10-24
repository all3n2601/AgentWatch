import assert from "node:assert/strict";
import test from "node:test";
import { runSnapshotSchema } from "../lib/live-api-schema";
import { readRunSnapshot, runStatusOf } from "../lib/live-snapshot";
import { stepTotals } from "../lib/live-status";
import {
  OTHER_RUN_ID,
  runSnapshot,
  savedRecord,
  stepEvent,
  stepView,
} from "./live-fixtures";

/** Parses through the response schema, as the live routes do. */
const snapshotOf = (overrides: Record<string, unknown>) =>
  runSnapshotSchema.parse(runSnapshot(overrides));

test("pending, then expired, then a filled prefix never applies the expired revision", () => {
  const queued = stepEvent("run-tests", 0, "queued");
  const running = stepEvent("run-tests", 1, "running", { started: 1 });
  const completed = stepEvent("run-tests", 2, "completed", {
    started: 1,
    finished: 3,
  });

  // 1. Revision 2 arrives before revision 1 and is held as pending.
  const pending = snapshotOf({
    steps: [stepView(queued)],
    gaps: [
      {
        step_id: "run-tests",
        missing_sequence: 1,
        first_pending_sequence: 2,
        expired: false,
      },
    ],
    records: [savedRecord(queued, 1), savedRecord(completed, 2, "pending")],
  });
  let view = readRunSnapshot(pending, pending.records);
  assert.equal(view.steps[0].state, "queued");
  assert.equal(view.steps[0].appliedSequence, 0);
  assert.deepEqual(view.gaps, [
    {
      stepId: "run-tests",
      missingSequence: 1,
      firstPendingSequence: 2,
      expired: false,
      pendingRevisions: 1,
    },
  ]);
  assert.ok(
    stepTotals(view, runStatusOf(pending)).reasons.includes(
      "1 step is waiting for a missing revision",
    ),
  );

  // 2. The service expires it after 300 s without a filled prefix.
  const expiredRecords = [
    savedRecord(queued, 1),
    savedRecord(completed, 2, "expired"),
  ];
  const expired = snapshotOf({
    steps: [stepView(queued)],
    gaps: [
      {
        step_id: "run-tests",
        missing_sequence: 1,
        first_pending_sequence: 2,
        expired: true,
      },
    ],
    records: expiredRecords,
  });
  view = readRunSnapshot(expired, expired.records);
  assert.equal(view.gaps[0].expired, true);
  assert.ok(
    stepTotals(view, runStatusOf(expired)).reasons.includes(
      "1 step has expired revisions that will never be applied",
    ),
  );

  // 3. Revision 1 finally arrives. The prefix advances to running, and the expired
  //    completion stays unapplied even though its own record says "completed".
  const filled = snapshotOf({
    state: "running",
    steps: [stepView(running)],
    gaps: [
      {
        step_id: "run-tests",
        missing_sequence: 2,
        first_pending_sequence: 2,
        expired: true,
      },
    ],
    records: [savedRecord(running, 3)],
  });
  view = readRunSnapshot(filled, [...expired.records, ...filled.records]);
  const [step] = view.steps;
  assert.equal(step.state, "running");
  assert.equal(step.appliedSequence, 1);
  assert.equal(step.timings, null, "the expired completion supplies no timing");
  assert.equal(step.startedObservedAt, running.observed_at);
  assert.deepEqual(view.gaps, [
    {
      stepId: "run-tests",
      missingSequence: 2,
      firstPendingSequence: 2,
      expired: true,
      pendingRevisions: 1,
    },
  ]);
  const totals = stepTotals(view, runStatusOf(filled));
  assert.equal(totals.final, false);
  assert.equal(totals.duration.measuredSteps, 0);
});

test("a step with only pending revisions is a gap, not an applied step", () => {
  const early = stepEvent("late", 1, "running", { started: 1 });
  const snapshot = snapshotOf({
    gaps: [
      {
        step_id: "late",
        missing_sequence: 0,
        first_pending_sequence: 1,
        expired: false,
      },
    ],
    records: [savedRecord(early, 1, "pending")],
  });
  const view = readRunSnapshot(snapshot, snapshot.records);
  assert.deepEqual(view.steps, []);
  assert.equal(view.gaps[0].missingSequence, 0);
  assert.equal(view.gaps[0].pendingRevisions, 1);
});

test("terminal steps take the service's timings; unfinished steps have none", () => {
  const done = stepEvent("tool-1", 2, "completed", {
    submitted: 1,
    started: 1.5,
    finished: 4,
    cpu: 0.25,
  });
  const unstarted = stepEvent("tool-2", 1, "failed", {
    submitted: 2,
    finished: 2.5,
    reason: "Rejected tool call",
  });
  const active = stepEvent("model-call-1", 1, "running", {
    submitted: 3,
    started: 3.25,
    type: "model_call",
  });
  const snapshot = snapshotOf({
    steps: [stepView(done), stepView(unstarted), stepView(active)],
  });
  const view = readRunSnapshot(snapshot, []);
  const byId = new Map(view.steps.map((step) => [step.stepId, step]));
  assert.deepEqual(byId.get("tool-1")!.timings, {
    queueSeconds: 0.5,
    executionSeconds: 2.5,
    durationSeconds: 3,
  });
  assert.equal(byId.get("tool-1")!.cpuSeconds, 0.25);

  const failed = byId.get("tool-2")!;
  assert.equal(failed.started, false);
  assert.equal(failed.timings!.queueSeconds, null);
  assert.equal(failed.timings!.executionSeconds, null);
  assert.equal(failed.timings!.durationSeconds, 0.5);
  assert.equal(failed.reason, "Rejected tool call");

  const running = byId.get("model-call-1")!;
  assert.equal(running.timings, null);
  assert.equal(running.started, true);
  assert.equal(
    running.startedObservedAt,
    null,
    "a start whose running record was not received is not invented",
  );
});

test("steps are ordered by submission within a clock domain, not by name", () => {
  const snapshot = snapshotOf({
    steps: [
      stepView(stepEvent("model-call-10", 0, "queued", { submitted: 10 })),
      stepView(stepEvent("model-call-2", 0, "queued", { submitted: 2 })),
    ],
  });
  assert.deepEqual(
    readRunSnapshot(snapshot, []).steps.map((step) => step.stepId),
    ["model-call-2", "model-call-10"],
  );
});

test("records from another run are counted and never applied", () => {
  const ours = stepEvent("tool-1", 1, "running", { started: 1 });
  const theirs = stepEvent("tool-1", 1, "running", {
    started: 9,
    second: 9,
    runId: OTHER_RUN_ID,
  });
  const snapshot = snapshotOf({
    steps: [stepView(ours)],
    records: [savedRecord(ours, 1), savedRecord(theirs, 2)],
  });
  const view = readRunSnapshot(snapshot, snapshot.records);
  assert.equal(view.otherRunRecords, 1);
  assert.equal(view.steps[0].startedObservedAt, ours.observed_at);
});
