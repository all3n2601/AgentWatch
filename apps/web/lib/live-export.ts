import type { LiveApi, LiveApiResult } from "./live-api";
import type { LiveSavedRecord } from "./live-api-schema";

// A JSON download of one live run's records, read from the B01 live API for the selected UUID
// only. It is not saved evidence: live records are kept apart from saved run history (ADR 0001)
// and agent runs report only to the live API (ADR 0002), so a live run has no saved-run file.

/** B01 retains at most 100,000 records per run, read 256 per page. */
export const MAX_EXPORT_PAGES = Math.ceil(100_000 / 256);

export const LIVE_EXPORT_KIND = "agentwatch-live-records-v1";

export type LiveRecordsExport = {
  export_kind: typeof LIVE_EXPORT_KIND;
  description: string;
  run_id: string;
  /** The web server's clock when the first and last pages were requested. */
  read_started_at: string;
  read_finished_at: string;
  pages_read: number;
  /** Run state, steps, gaps and changes as of the last page read. */
  run: Record<string, unknown>;
  /** Every record in sequence order, with its disposition exactly as the service reported it. */
  records: LiveSavedRecord[];
};

const DESCRIPTION =
  "Live records read from the AgentWatch live API (GET /v1/live/runs/{run_id}) for this run " +
  "only. This is not saved evidence: live runs have no saved-run counterpart. Records, gaps " +
  "and pending or expired dispositions are as the service reported them; nothing is filled " +
  "in. Run state, steps and gaps come from the last page read and can change between page " +
  "reads, so they describe the run as of read_finished_at.";

/** Reads every record page for `runId`. Any failed page fails the export; nothing partial. */
export async function readLiveRecordsExport(
  api: LiveApi,
  runId: string,
  now: () => Date = () => new Date(),
): Promise<LiveApiResult<LiveRecordsExport>> {
  const startedAt = now().toISOString();
  const records: LiveSavedRecord[] = [];
  let after = 0;
  for (let page = 1; page <= MAX_EXPORT_PAGES; page += 1) {
    const result = await api.runSnapshot(runId, after);
    if (!result.ok) return result;
    const { records: pageRecords, next_cursor, ...run } = result.data;
    if (run.run_id !== runId)
      return {
        ok: false,
        kind: "invalid_response",
        message: "The live API answered with a different run.",
      };
    records.push(...pageRecords);
    if (next_cursor === null)
      return {
        ok: true,
        data: {
          export_kind: LIVE_EXPORT_KIND,
          description: DESCRIPTION,
          run_id: runId,
          read_started_at: startedAt,
          read_finished_at: now().toISOString(),
          pages_read: page,
          run,
          records,
        },
      };
    after = next_cursor;
  }
  return {
    ok: false,
    kind: "invalid_response",
    message: `The run has more than ${MAX_EXPORT_PAGES} record pages, beyond the service's retention bound.`,
  };
}

export const liveExportFilename = (runId: string) =>
  `agentwatch-live-records-${runId}.json`;
