import assert from "node:assert/strict";
import test from "node:test";
import {
  SYNTHETIC_RUN_ID,
  syntheticStepEvent as event,
} from "../fixtures/step-event-replays";
import { readStepEvents } from "../lib/step-events";

const at = (second: number) =>
  `2026-10-14T12:00:${String(second).padStart(2, "0")}Z`;

test("applies the contiguous revision prefix of each step", () => {
  const view = readStepEvents(SYNTHETIC_RUN_ID, [
    event({ step_id: "a", sequence: 0, state: "queued", observed_at: at(0) }),
    event({
      step_id: "a",
      sequence: 1,
      state: "running",
      observed_at: at(1),
      started_s: 0.5,
    }),
    event({
      step_id: "a",
      sequence: 2,
      state: "completed",
      observed_at: at(3),
      started_s: 0.5,
      finished_s: 2.5,
      cpu_seconds: 1.5,
      cpu_scope: "process",
    }),
  ]);
  assert.equal(view.steps.length, 1);
  const [step] = view.steps;
  assert.equal(step.state, "completed");
  assert.equal(step.appliedSequence, 2);
  assert.equal(step.startedObservedAt, at(1));
  assert.deepEqual(step.timings, {
    queueSeconds: 0.5,
    executionSeconds: 2,
    durationSeconds: 2.5,
  });
  assert.equal(step.cpuSeconds, 1.5);
  assert.equal(step.cpuScope, "process");
  assert.deepEqual(view.gaps, []);
  assert.equal(view.lastObservedAt, at(3));
});

test("holds a future terminal revision as pending behind a gap", () => {
  const view = readStepEvents(SYNTHETIC_RUN_ID, [
    event({ step_id: "a", sequence: 0, state: "queued", observed_at: at(0) }),
    event({
      step_id: "a",
      sequence: 2,
      state: "completed",
      observed_at: at(3),
      started_s: 1,
      finished_s: 2,
    }),
  ]);
  const [step] = view.steps;
  assert.equal(step.state, "queued");
  assert.equal(step.timings, null);
  assert.deepEqual(view.gaps, [
    { stepId: "a", missingSequence: 1, pendingRevisions: 1 },
  ]);
});

test("reports a step whose queued revision has not arrived as a gap only", () => {
  const view = readStepEvents(SYNTHETIC_RUN_ID, [
    event({
      step_id: "late",
      sequence: 1,
      state: "running",
      observed_at: at(1),
      started_s: 1,
    }),
  ]);
  assert.deepEqual(view.steps, []);
  assert.deepEqual(view.gaps, [
    { stepId: "late", missingSequence: 0, pendingRevisions: 1 },
  ]);
});

test("running steps carry a start time but no computed duration", () => {
  const view = readStepEvents(SYNTHETIC_RUN_ID, [
    event({ step_id: "a", sequence: 0, state: "queued", observed_at: at(0) }),
    event({
      step_id: "a",
      sequence: 1,
      state: "running",
      observed_at: at(2),
      started_s: 1,
    }),
  ]);
  const [step] = view.steps;
  assert.equal(step.state, "running");
  assert.equal(step.startedObservedAt, at(2));
  assert.equal(step.timings, null);
});

test("a step that fails before starting has total duration but no queue or execution time", () => {
  const view = readStepEvents(SYNTHETIC_RUN_ID, [
    event({ step_id: "a", sequence: 0, state: "queued", observed_at: at(0) }),
    event({
      step_id: "a",
      sequence: 1,
      state: "cancelled",
      observed_at: at(1),
      finished_s: 0.75,
      reason: "Cancelled before a worker started",
    }),
  ]);
  const [step] = view.steps;
  assert.equal(step.startedObservedAt, null);
  assert.deepEqual(step.timings, {
    queueSeconds: null,
    executionSeconds: null,
    durationSeconds: 0.75,
  });
  assert.equal(step.cpuSeconds, null);
  assert.equal(step.reason, "Cancelled before a worker started");
});

test("keeps zero-duration terminal steps as measured zero", () => {
  const view = readStepEvents(SYNTHETIC_RUN_ID, [
    event({ step_id: "a", sequence: 0, state: "queued", observed_at: at(0) }),
    event({
      step_id: "a",
      sequence: 1,
      state: "running",
      observed_at: at(0),
      started_s: 0,
    }),
    event({
      step_id: "a",
      sequence: 2,
      state: "completed",
      observed_at: at(0),
      started_s: 0,
      finished_s: 0,
    }),
  ]);
  assert.deepEqual(view.steps[0].timings, {
    queueSeconds: 0,
    executionSeconds: 0,
    durationSeconds: 0,
  });
});

test("counts unreadable records and ignores records from other runs", () => {
  const view = readStepEvents(SYNTHETIC_RUN_ID, [
    { record_type: "step_event", state: "running" },
    {
      ...event({
        step_id: "a",
        sequence: 0,
        state: "queued",
        observed_at: at(0),
      }),
      observed_at: "2026-10-14 12:00:00",
    },
    {
      ...event({
        step_id: "a",
        sequence: 1,
        state: "running",
        observed_at: at(1),
        started_s: 1,
      }),
      submitted_tick_ns: 2e9,
    },
    event({
      step_id: "b",
      sequence: 0,
      state: "queued",
      observed_at: at(0),
      run_id: "00000000-0000-4000-8000-00000000f002",
    }),
  ]);
  assert.equal(view.unreadableRecords, 3);
  assert.equal(view.otherRunRecords, 1);
  assert.deepEqual(view.steps, []);
});

test("an exact retry of an applied revision is not counted twice", () => {
  const queued = event({
    step_id: "a",
    sequence: 0,
    state: "queued",
    observed_at: at(0),
  });
  const view = readStepEvents(SYNTHETIC_RUN_ID, [queued, queued]);
  assert.equal(view.steps.length, 1);
  assert.deepEqual(view.gaps, []);
});
