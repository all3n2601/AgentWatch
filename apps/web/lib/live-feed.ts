import "server-only";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { apiUrl, dataDirectory } from "./experiments";
import { createLiveApi, type LiveApi } from "./live-api";
import type { LiveSavedRecord } from "./live-api-schema";
import {
  latestReceiptAgeMs,
  liveSampleSeries,
  parseGpuSamples,
  wallSteps,
  type GpuEvidence,
  type LiveFeedData,
  type LiveRunSummary,
} from "./live-feed-data";
import { readRunSnapshot, runStatusOf } from "./live-snapshot";

// Prototype reader for recorded live runs: the B01 read API plus, when present, the A01
// nvidia-smi diagnostic files under the local data directory.

const MAX_RECORD_PAGES = 40;

class LiveReadError extends Error {
  constructor(
    message: string,
    readonly notFound: boolean,
  ) {
    super(message);
  }
}

async function loadRunRecords(api: LiveApi, runId: string) {
  let after = 0;
  const records: LiveSavedRecord[] = [];
  for (let page = 0; page < MAX_RECORD_PAGES; page += 1) {
    const result = await api.runSnapshot(runId, after);
    if (!result.ok)
      throw new LiveReadError(result.message, result.kind === "not_found");
    records.push(...result.data.records);
    if (result.data.next_cursor === null)
      return { detail: result.data, records };
    after = result.data.next_cursor;
  }
  throw new Error(`more than ${MAX_RECORD_PAGES} record pages`);
}

async function listOrEmpty(directory: string) {
  try {
    return await readdir(directory);
  } catch {
    return [];
  }
}

// Probes write .data/<name>/agent-runs/<run>/; collections nest them as .data/<c>/agent/<name>/.
const MAX_PROBE_DEPTH = 3;

async function probeDirectories(root: string) {
  const found: string[] = [];
  let level = [root];
  for (let depth = 1; depth <= MAX_PROBE_DEPTH; depth += 1) {
    const next: string[] = [];
    for (const parent of level)
      for (const name of await listOrEmpty(parent))
        if (!name.startsWith(".")) next.push(path.join(parent, name));
    found.push(...next);
    level = next;
  }
  return found;
}

/** The A01 probe directory whose agent evidence names this run, if one exists locally. */
async function findGpuProbe(runId: string): Promise<string | null> {
  for (const probe of await probeDirectories(dataDirectory())) {
    const runsDir = path.join(probe, "agent-runs");
    for (const runDir of await listOrEmpty(runsDir)) {
      try {
        const evidence: unknown = JSON.parse(
          await readFile(
            path.join(runsDir, runDir, "agent-evidence.json"),
            "utf8",
          ),
        );
        if (
          typeof evidence === "object" &&
          evidence !== null &&
          "run_id" in evidence &&
          evidence.run_id === runId
        )
          return probe;
      } catch {
        // Unreadable or partial evidence belongs to some other run; keep looking.
      }
    }
  }
  return null;
}

async function loadGpuEvidence(runId: string): Promise<GpuEvidence> {
  const probe = await findGpuProbe(runId);
  if (!probe) return { missing: "No local GPU probe recorded this run." };
  try {
    const [csv, meta] = await Promise.all([
      readFile(path.join(probe, "gpu-samples.csv"), "utf8"),
      readFile(path.join(probe, "probe-meta.txt"), "utf8"),
    ]);
    return parseGpuSamples(csv, meta, path.basename(probe));
  } catch {
    return {
      missing: `${path.basename(probe)} has no readable gpu-samples.csv or probe-meta.txt.`,
    };
  }
}

const message = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

export async function loadLiveFeed(requested?: string): Promise<LiveFeedData> {
  const base = { requested: requested ?? null, selected: null };
  const api = createLiveApi(apiUrl());
  const page = await api.runPage();
  if (!page.ok)
    return {
      ...base,
      runs: [],
      error: `Live API unavailable: ${page.message}`,
    };
  const runs: LiveRunSummary[] = page.data.runs.map((run) => ({
    runId: run.run_id,
    taskId: run.registration.task_id,
    state: run.state,
    createdAt: run.created_at,
  }));
  // A requested UUID is read directly, so runs older than the newest page stay selectable.
  // A missing one is reported, never substituted.
  const runId = requested ?? runs[0]?.runId;
  if (!runId) return { ...base, runs, error: null };
  try {
    const { detail, records } = await loadRunRecords(api, runId);
    // Only applied revisions are drawn; pending and expired ones are reported as gaps.
    const applied = records.flatMap((r) =>
      r.disposition === "applied" ? [r.payload] : [],
    );
    const steps = applied.filter((p) => p.record_type === "step_event");
    const samples = applied.filter((p) => p.record_type === "resource_sample");
    return {
      ...base,
      runs,
      error: null,
      selected: {
        replay: {
          id: runId,
          title: detail.registration.task_id,
          description: "",
          run: runStatusOf(detail),
          receivedAgoMs: latestReceiptAgeMs(records, Date.now()),
          view: readRunSnapshot(detail, records),
        },
        resources: detail.registration.resources.map((r) => ({
          resourceId: r.resource_id,
          resourceClass: r.resource_class,
        })),
        wallSteps: wallSteps(steps),
        liveSeries: liveSampleSeries(samples),
        gpu: await loadGpuEvidence(runId),
        recordCount: records.length,
      },
    };
  } catch (error) {
    if (error instanceof LiveReadError && error.notFound)
      return { ...base, runs, error: null };
    return {
      ...base,
      runs,
      error: `Could not read live run ${runId}: ${message(error)}`,
    };
  }
}
