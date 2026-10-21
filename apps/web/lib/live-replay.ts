import {
  findReplayScenario,
  replayScenarios,
  SYNTHETIC_RUN_ID,
} from "../fixtures/step-event-replays";
import type { LiveFeedData } from "./live-feed-data";
import type { RunStatus } from "./live-status";
import { readStepEvents, type LiveRunView } from "./step-events";

/** A synthetic scenario already read through the StepEvent adapter. */
export type LiveReplay = {
  id: string;
  title: string;
  description: string;
  run: RunStatus;
  /**
   * Scripted age of the latest record when the browser receives the page, or
   * null when no record has been received.
   */
  receivedAgoMs: number | null;
  view: LiveRunView;
};

export type LivePageData = {
  replay: LiveReplay | null;
  /** Requested scenario id, kept so an unknown id can be reported. */
  requestedReplay: string | null;
  scenarios: { id: string; title: string }[];
  /** Recorded runs from the live API; set only when no replay was requested. */
  feed?: LiveFeedData;
};

/**
 * Builds the live page's data. Until the F02 live consumer exists the only
 * source is a development replay, which callers must disable in production.
 * It carries no server wall-clock time: the browser anchors receipt itself.
 */
export function liveReplayData(
  requested: string | undefined,
  { replayEnabled }: { replayEnabled: boolean },
): LivePageData {
  if (!replayEnabled)
    return {
      replay: null,
      requestedReplay: null,
      scenarios: [],
    };
  const scenario = findReplayScenario(requested);
  return {
    replay: scenario
      ? {
          id: scenario.id,
          title: scenario.title,
          description: scenario.description,
          run: scenario.run,
          receivedAgoMs: scenario.receivedAgoMs,
          view: readStepEvents(SYNTHETIC_RUN_ID, scenario.records),
        }
      : null,
    requestedReplay: requested ?? null,
    scenarios: replayScenarios.map(({ id, title }) => ({ id, title })),
  };
}
