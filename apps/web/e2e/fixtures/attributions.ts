import { attributionPage, modelStatus } from "../../tests/attribution-fixtures";
import type { AttributionState, FixtureRun } from "../fixture-control";
import { fixtureRun, snapshot } from "../live-runs";

// Synthetic live runs with scripted B03 attributions, for browser tests only.

export * from "../../tests/attribution-fixtures";

type Row = Record<string, unknown> & { sequence: number };

/** One attributions response holding `rows`; the fixture API pages it by cursor. */
export function attributionState(
  runId: string,
  rows: Row[],
  { model = modelStatus() }: { model?: Record<string, unknown> } = {},
): AttributionState {
  return { page: attributionPage(runId, rows, { model }) };
}

/** A live run (open unless `state` says otherwise) whose attributions follow `states`. */
export function attributedRun(
  runId: string,
  states: AttributionState[],
  { state = "running" }: { state?: string } = {},
): FixtureRun {
  return {
    ...fixtureRun(runId, [snapshot(runId, { state })], { state }),
    attributions: states,
  };
}
