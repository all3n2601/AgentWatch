import "server-only";
import type { AssistantSources, Loaded } from "./assistant-tools";
import { loadDatasetSnapshot } from "./dataset-load";
import { liveRunsUrl, loadExperiments } from "./experiments";
import { loadImportMap } from "./import-map";
import { createLiveApi } from "./live-api";
import type { Ready } from "./overview-model";
import { loadRunView } from "./run-load";
import { fromLivePage, fromSaved, type UnifiedRun } from "./unified-runs";

/**
 * The assistant's data for one request: the dataset release, the import map,
 * and the ingestion database's live and saved runs. Each source is read at
 * most once per request; a failure is reported to the model, not hidden.
 */
export function requestSources(): AssistantSources {
  const once = <T>(load: () => Promise<T>) => {
    let pending: Promise<T> | undefined;
    return () => (pending ??= load());
  };

  const dataset = once(async (): Promise<Loaded<Ready>> => {
    const snapshot = await loadDatasetSnapshot();
    return snapshot.status === "ready"
      ? { ok: true, value: snapshot }
      : { ok: false, message: snapshot.reason };
  });

  const liveRuns = once(async (): Promise<Loaded<UnifiedRun[]>> => {
    const page = await createLiveApi(liveRunsUrl()).runPage();
    return page.ok
      ? { ok: true, value: fromLivePage(page.data) }
      : { ok: false, message: page.message };
  });

  const savedRuns = once(async (): Promise<Loaded<UnifiedRun[]>> => {
    const data = await loadExperiments();
    // Local demo artifacts are not database history; do not present them as such.
    return data.source === "database"
      ? { ok: true, value: data.experiments.map(fromSaved) }
      : {
          ok: false,
          message: "Saved-run history is unavailable from the database.",
        };
  });

  return {
    dataset,
    importMap: once(() => loadImportMap()),
    liveRuns,
    savedRuns,
    runView: async (runId) => (await loadRunView(runId))?.view ?? null,
  };
}
