import assert from "node:assert/strict";
import test from "node:test";
import {
  NOT_CHARTED_SHOWN,
  liveTimelineSteps,
  openStepLabel,
  type StepState,
} from "../lib/live-view";

// Synthetic steps for tests only; they are not recorded-run evidence.
const steps = (count: number, state: StepState = "completed") =>
  Array.from({ length: count }, (_, index) => ({
    stepId: `step-${index}`,
    state,
  }));

test("step states and the selected step are looked up by id", () => {
  const all = [...steps(3), { stepId: "tool-x", state: "running" as const }];
  const model = liveTimelineSteps(all, [
    "step-0",
    "step-1",
    "step-2",
    "tool-x",
  ]);
  assert.equal(model.stateById.get("tool-x"), "running");
  assert.equal(model.stateById.get("missing"), undefined);
  assert.equal(model.select("step-1")?.stepId, "step-1");
  // No choice, or a choice that is gone, falls back to the running step, then the last.
  assert.equal(model.select(null)?.stepId, "tool-x");
  assert.equal(model.select("gone")?.stepId, "tool-x");
  assert.equal(liveTimelineSteps(steps(2), []).select(null)?.stepId, "step-1");
  assert.equal(liveTimelineSteps([], []).select(null), undefined);
});

test("steps not on the chart are named up to a bound and the rest are counted", () => {
  const few = liveTimelineSteps(steps(3, "queued"), ["step-1"]);
  assert.deepEqual(few.notCharted, { shown: ["step-0", "step-2"], more: 0 });

  const many = liveTimelineSteps(steps(5_000, "queued"), []);
  assert.equal(many.notCharted.shown.length, NOT_CHARTED_SHOWN);
  assert.equal(many.notCharted.shown[0], "step-0");
  assert.equal(many.notCharted.more, 5_000 - NOT_CHARTED_SHOWN);
});

test("a large run is indexed in linear time", () => {
  // 20,000 steps would take seconds with a per-row search; the index takes milliseconds.
  const large = steps(20_000);
  const started = performance.now();
  const model = liveTimelineSteps(
    large,
    large.map((step) => step.stepId),
  );
  for (const step of large) model.stateById.get(step.stepId);
  assert.ok(performance.now() - started < 1_000);
  assert.equal(model.notCharted.more, 0);
});

test("an open step's timeline value names its state; a queued step is never Running", () => {
  assert.equal(openStepLabel("queued"), "Queued");
  assert.equal(openStepLabel("running"), "Running");
  // No applied state yet (only unapplied events): say so rather than guess.
  assert.equal(openStepLabel(null), "In progress");
});
