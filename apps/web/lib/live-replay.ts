import {
  findReplayScenario,
  replayScenarios,
  SYNTHETIC_RUN_ID,
} from "../fixtures/step-event-replays";
import type { RunStatus } from "./live-status";
import { readStepEvents, type LiveRunView } from "./step-events";

/** A synthetic scenario already read through the StepEvent adapter. */
export type LiveReplay = {
  id: string;
  title: string;
  description: string;
  run: RunStatus;
  /** Browser-clock time the scripted records count as received. */
  receivedAtMs: number;
  view: LiveRunView;
};

export type LivePageData = {
  replay: LiveReplay | null;
  /** Requested scenario id, kept so an unknown id can be reported. */
  requestedReplay: string | null;
  scenarios: { id: string; title: string }[];
  renderedAtMs: number;
};

/**
 * Builds the live page's data. Until the F02 live consumer exists the only
 * source is a development replay, which callers must disable in production.
 */
export function liveReplayData(
  requested: string | undefined,
  {
    replayEnabled,
    nowMs = Date.now(),
  }: { replayEnabled: boolean; nowMs?: number },
): LivePageData {
  if (!replayEnabled)
    return {
      replay: null,
      requestedReplay: null,
      scenarios: [],
      renderedAtMs: nowMs,
    };
  const scenario = findReplayScenario(requested);
  return {
    replay: scenario
      ? {
          id: scenario.id,
          title: scenario.title,
          description: scenario.description,
          run: scenario.run,
          receivedAtMs: nowMs - scenario.receivedAgoMs,
          view: readStepEvents(SYNTHETIC_RUN_ID, scenario.records),
        }
      : null,
    requestedReplay: requested ?? null,
    scenarios: replayScenarios.map(({ id, title }) => ({ id, title })),
    renderedAtMs: nowMs,
  };
}
