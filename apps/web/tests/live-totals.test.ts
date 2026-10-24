import assert from "node:assert/strict";
import test from "node:test";
import { runSnapshotSchema } from "../lib/live-api-schema";
import { readRunSnapshot } from "../lib/live-snapshot";
import { stepTotals, type RunStatus } from "../lib/live-status";
import { runSnapshot, stepEvent, stepView } from "./live-fixtures";

const closed: RunStatus = { state: "completed", reason: null };
type Window = { submitted: number; started: number; finished: number };

function totalsOf(
  steps: [string, string | null, Window, string?][],
  run: RunStatus = closed,
) {
  const snapshot = runSnapshotSchema.parse(
    runSnapshot({
      state: run?.state,
      steps: steps.map(([id, parent, window, clock]) =>
        stepView(
          stepEvent(id, 2, "completed", { ...window, parent, clock, cpu: 0.1 }),
        ),
      ),
    }),
  );
  return stepTotals(readRunSnapshot(snapshot, []), run);
}

const near = (actual: number | null, expected: number) =>
  assert.ok(
    actual !== null && Math.abs(actual - expected) < 1e-9,
    `${actual} ≈ ${expected}`,
  );

test("a parent step and its children are not summed into double the run's time", () => {
  // Shape of a recorded agent run: the root step spans its sequential children.
  const totals = totalsOf([
    ["agent-run", null, { submitted: 0, started: 0, finished: 5.44 }],
    [
      "model-call-1",
      "agent-run",
      { submitted: 0.1, started: 0.2, finished: 2 },
    ],
    [
      "tool-1-run_tests",
      "agent-run",
      { submitted: 2, started: 2.1, finished: 2.6 },
    ],
    [
      "model-call-2",
      "agent-run",
      { submitted: 2.6, started: 2.7, finished: 5.4 },
    ],
  ]);
  assert.equal(totals.final, true);
  near(totals.duration.seconds, 5.44);
  assert.equal(totals.duration.measuredSteps, 4);
  // Leaf sums only: the parent's queue and execution windows contain its children's.
  assert.equal(totals.parentSteps, 1);
  near(totals.queue.seconds, 0.1 + 0.1 + 0.1);
  near(totals.execution.seconds, 1.8 + 0.5 + 2.7);
  assert.equal(totals.execution.measuredSteps, 3);
  near(totals.cpu.seconds, 0.3);
});

test("concurrent leaf steps count once in the span but add up in execution time", () => {
  const totals = totalsOf([
    ["tool-a", null, { submitted: 0, started: 0, finished: 2 }],
    ["tool-b", null, { submitted: 1, started: 1, finished: 3 }],
    ["tool-c", null, { submitted: 5, started: 5, finished: 6 }],
  ]);
  near(totals.duration.seconds, 3 + 1);
  near(totals.execution.seconds, 2 + 2 + 1);
  assert.equal(totals.parentSteps, 0);
});

test("steps from different clock domains are never combined into one span", () => {
  const totals = totalsOf([
    ["runner-step", null, { submitted: 0, started: 0, finished: 2 }, "clock-a"],
    [
      "service-step",
      null,
      { submitted: 100, started: 100, finished: 101 },
      "clock-b",
    ],
  ]);
  assert.equal(totals.duration.seconds, null);
  assert.equal(totals.duration.clockDomains, 2);
  assert.equal(totals.duration.measuredSteps, 2);
  // Per-step execution times are each within one domain, so their sum stays valid.
  near(totals.execution.seconds, 3);
});
