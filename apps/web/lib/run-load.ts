import { liveRunsUrl, loadExperiments, loadRequestedRuns } from "./experiments";
import { createLiveApi, isLiveRunId } from "./live-api";
import type { DashboardData } from "./experiment-schema";
import type { RunView } from "./unified-runs";

/**
 * One run for its own page or summary, whichever store holds it: the live API is asked first,
 * then saved history. Null means neither has it; no other run is substituted.
 */
export async function loadRunView(
  runId: string,
): Promise<{ view: RunView; data: DashboardData } | null> {
  if (!isLiveRunId(runId)) return null;
  const live = await createLiveApi(liveRunsUrl()).runSnapshot(runId, 0, {
    limit: 1,
  });
  // A live run needs no saved-run lookup, and a failed lookup would add a misleading warning.
  const data = live.ok
    ? await loadExperiments()
    : await loadRequestedRuns([runId]);
  if (live.ok) return { view: { source: "live", snapshot: live.data }, data };
  const saved = data.experiments.find((run) => run.run_id === runId);
  return saved ? { view: { source: "saved", run: saved }, data } : null;
}
