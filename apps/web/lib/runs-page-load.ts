import "server-only";
import { liveRunsUrl, loadExperiments } from "./experiments";
import { createLiveApi } from "./live-api";
import { liveRunSummaries } from "./live-feed-data";
import {
  liveRow,
  savedRow,
  type LiveRunsResult,
  type RunsPageData,
  type RunsQuery,
} from "./runs-page";

async function loadLiveRows(query: RunsQuery): Promise<LiveRunsResult> {
  if (query.liveBeforeError)
    return { status: "unavailable", reason: query.liveBeforeError };
  const result = await createLiveApi(liveRunsUrl()).runPage(
    query.liveBefore ?? undefined,
  );
  return result.ok
    ? {
        status: "ready",
        rows: liveRunSummaries(result.data).map(liveRow),
        nextCursor: result.data.next_cursor,
      }
    : { status: "unavailable", reason: result.message };
}

/** The first page of saved history and one page of live runs, read together. */
export async function loadRunsPage(query: RunsQuery): Promise<RunsPageData> {
  const [history, live] = await Promise.all([
    loadExperiments(),
    loadLiveRows(query),
  ]);
  return {
    saved: {
      rows: history.experiments.map(savedRow),
      source: history.source ?? "local",
      warnings: history.warnings,
      nextCursor: history.nextCursor ?? null,
    },
    live,
    liveBefore: query.liveBefore,
  };
}
