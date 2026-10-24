import { randomUUID } from "node:crypto";
import {
  runSnapshot,
  runSummary,
  savedRecord,
  stepEvent,
  stepView,
} from "../tests/live-fixtures";
import type { FixtureRun, FixtureState } from "./fixture-control";

// Builders for synthetic live runs served by the fixture API in browser tests.

export { resourceSample } from "../tests/live-fixtures";
export { stepEvent };
export const newRunId = () => randomUUID();

type Payload = { record_type: string };
type StepPayload = ReturnType<typeof stepEvent>;

/**
 * A snapshot whose records are numbered 1..n in the given order. `steps` lists each step's
 * applied revision; `pending` and `expired` records are reported but never applied.
 */
export function snapshot(
  runId: string,
  {
    state = "running",
    reason = null,
    steps = [],
    records = [],
    gaps = [],
    unfinished = [],
  }: {
    state?: string;
    reason?: string | null;
    steps?: StepPayload[];
    records?: (Payload | [Payload, "pending" | "expired"])[];
    gaps?: unknown[];
    unfinished?: string[];
  },
): FixtureState {
  return {
    snapshot: runSnapshot(
      {
        state,
        reason,
        revision: state === "queued" ? 0 : state === "running" ? 1 : 2,
        unfinished_steps: unfinished,
        steps: steps.map(stepView),
        records: records.map((record, index) =>
          Array.isArray(record)
            ? savedRecord(record[0], index + 1, record[1])
            : savedRecord(record, index + 1),
        ),
        gaps,
      },
      runId,
    ),
  };
}

/** A run listed at history position `sequence` (higher is newer). */
export function fixtureRun(
  runId: string,
  states: FixtureState[],
  {
    sequence = 1,
    state = "running",
  }: { sequence?: number; state?: string } = {},
): FixtureRun {
  return { summary: runSummary(runId, sequence, state), states };
}
