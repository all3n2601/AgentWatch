import type { FixtureRun } from "../fixture-control";
import { fixtureRun, snapshot, stepEvent } from "../live-runs";

// SYNTHETIC browser-test fixtures: scripted live runs served by the fixture API. None of these
// values were measured; they replace the former ?replay= development scenarios and must never
// be shown or exported as recorded evidence.

type StepType = "model_call" | "tool" | "retrieval" | "other";

function completedStep(
  runId: string,
  stepId: string,
  [submitted, started, finished]: [number, number, number],
  { type = "tool", cpu = 0.4 }: { type?: StepType; cpu?: number | null } = {},
) {
  const base = { runId, type, submitted };
  return [
    stepEvent(stepId, 0, "queued", { ...base, second: submitted }),
    stepEvent(stepId, 1, "running", { ...base, started, second: started }),
    stepEvent(stepId, 2, "completed", {
      ...base,
      started,
      finished,
      cpu,
      second: finished,
    }),
  ];
}

const lastOf = <T>(items: T[]) => items[items.length - 1];

export type LiveScenario = {
  id: string;
  description: string;
  build: (runId: string) => FixtureRun;
  /** Text the live view must show for this scenario. */
  shows: (string | RegExp)[];
};

export const liveScenarios: LiveScenario[] = [
  {
    id: "active-steps",
    description: "One step finished, one running and one still queued.",
    build(runId) {
      const plan = completedStep(runId, "plan", [0, 0.1, 1.2], {
        type: "model_call",
      });
      const tests = completedStep(runId, "run-tests", [1.3, 1.5, 3.5]).slice(
        0,
        2,
      );
      const summarize = stepEvent("summarize", 0, "queued", {
        runId,
        type: "model_call",
        submitted: 3.6,
      });
      return fixtureRun(runId, [
        snapshot(runId, {
          steps: [lastOf(plan), lastOf(tests), summarize],
          records: [...plan, ...tests, summarize],
        }),
      ]);
    },
    shows: [
      "Partial step totals",
      "2 active · 0 failed or cancelled",
      "Run is still active",
    ],
  },
  {
    id: "completed-run",
    description:
      "Every step finished and the run is closed, so totals are final.",
    build(runId) {
      const steps = [
        completedStep(runId, "plan", [0, 0.1, 1.2], { type: "model_call" }),
        completedStep(runId, "run-tests", [1.3, 1.5, 3.5]),
        completedStep(runId, "summarize", [3.6, 3.7, 4.4], {
          type: "model_call",
        }),
      ];
      return fixtureRun(
        runId,
        [
          snapshot(runId, {
            state: "completed",
            steps: steps.map(lastOf),
            records: steps.flat(),
          }),
        ],
        { state: "completed" },
      );
    },
    shows: ["Final step totals", "3 finished of 3 reported", "Polling stopped"],
  },
  {
    id: "failed-run",
    description: "A tool step timed out and the run closed as failed.",
    build(runId) {
      const plan = completedStep(runId, "plan", [0, 0.1, 1.2], {
        type: "model_call",
      });
      const started = completedStep(runId, "run-tests", [1.3, 1.5, 3.5]).slice(
        0,
        2,
      );
      const failed = stepEvent("run-tests", 2, "failed", {
        runId,
        submitted: 1.3,
        started: 1.5,
        finished: 5.5,
        reason: "Tool timed out; cleanup status is recorded separately",
      });
      return fixtureRun(
        runId,
        [
          snapshot(runId, {
            state: "failed",
            reason: "Tool step run-tests failed",
            steps: [lastOf(plan), failed],
            records: [...plan, ...started, failed],
          }),
        ],
        { state: "failed" },
      );
    },
    shows: [
      "Tool step run-tests failed",
      "Reason: Tool timed out; cleanup status is recorded separately",
      "0 active · 1 failed or cancelled",
    ],
  },
  {
    id: "cancelled-before-start",
    description:
      "A queued step was cancelled before any worker started it; it has no execution time or CPU.",
    build(runId) {
      const plan = completedStep(runId, "plan", [0, 0.1, 1.2], {
        type: "model_call",
      });
      const queued = stepEvent("run-tests", 0, "queued", {
        runId,
        submitted: 1.3,
      });
      const cancelled = stepEvent("run-tests", 1, "cancelled", {
        runId,
        submitted: 1.3,
        finished: 2.1,
        reason: "Cancelled before a worker started",
      });
      return fixtureRun(
        runId,
        [
          snapshot(runId, {
            state: "cancelled",
            reason: "Operator cancelled the run",
            steps: [lastOf(plan), cancelled],
            records: [...plan, queued, cancelled],
          }),
        ],
        { state: "cancelled" },
      );
    },
    shows: ["Operator cancelled the run", "Never started", "Final step totals"],
  },
  {
    id: "out-of-order",
    description:
      "A completion arrived before its running revision; it is held as pending, not shown as completed.",
    build(runId) {
      const plan = completedStep(runId, "plan", [0, 0.1, 1.2], {
        type: "model_call",
      });
      const [queued, , done] = completedStep(
        runId,
        "run-tests",
        [1.3, 1.5, 3.5],
      );
      return fixtureRun(runId, [
        snapshot(runId, {
          steps: [lastOf(plan), queued],
          records: [...plan, queued, [done, "pending"]],
          gaps: [
            {
              step_id: "run-tests",
              missing_sequence: 1,
              first_pending_sequence: 2,
              expired: false,
            },
          ],
        }),
      ]);
    },
    shows: [
      /run-tests is applied through revision 0 and is waiting for revision 1/,
    ],
  },
  {
    id: "no-records",
    description:
      "The run is registered and queued, but no step record has been received.",
    build: (runId) =>
      fixtureRun(runId, [snapshot(runId, { state: "queued" })], {
        state: "queued",
      }),
    shows: ["No step has an applied state yet.", "None yet", "No records"],
  },
  {
    id: "missing-cpu",
    description:
      "The run completed but one step has no CPU measurement, so CPU time is not totalled.",
    build(runId) {
      const steps = [
        completedStep(runId, "plan", [0, 0.1, 1.2], { type: "model_call" }),
        completedStep(runId, "run-tests", [1.3, 1.5, 3.5], { cpu: null }),
      ];
      return fixtureRun(
        runId,
        [
          snapshot(runId, {
            state: "completed",
            steps: steps.map(lastOf),
            records: steps.flat(),
          }),
        ],
        { state: "completed" },
      );
    },
    shows: ["1 of 2 finished steps measured"],
  },
  {
    id: "zero-duration",
    description:
      "A step finished in the same tick it started: a measured zero.",
    build(runId) {
      const step = completedStep(runId, "cache-hit", [0, 0, 0], {
        type: "retrieval",
        cpu: 0,
      });
      return fixtureRun(
        runId,
        [
          snapshot(runId, {
            state: "completed",
            steps: [lastOf(step)],
            records: step,
          }),
        ],
        { state: "completed" },
      );
    },
    shows: ["Final step totals", "0 ms"],
  },
];
