import "server-only";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { apiUrl, dataDirectory } from "./experiments";
import {
  cpuSeries,
  latestReceiptAgeMs,
  liveDetailSchema,
  liveListSchema,
  parseGpuSamples,
  wallSteps,
  type GpuEvidence,
  type LiveFeedData,
  type LiveRecord,
  type LiveRunSummary,
} from "./live-feed-data";
import { readStepEvents } from "./step-events";

// Prototype reader for recorded live runs: the B01 read API plus, when present, the A01
// nvidia-smi diagnostic files under the local data directory.

const API_TIMEOUT_MS = 3000;
const RUN_LIST_LIMIT = 50;
const RECORD_PAGE_LIMIT = 256;
const MAX_RECORD_PAGES = 40;

async function getJson(url: string): Promise<unknown> {
  const response = await fetch(url, {
    cache: "no-store",
    signal: AbortSignal.timeout(API_TIMEOUT_MS),
  });
  if (!response.ok) throw new Error(`${response.status} from ${url}`);
  return response.json();
}

async function loadRunRecords(runId: string) {
  let after = 0;
  const records: LiveRecord[] = [];
  for (let page = 0; page < MAX_RECORD_PAGES; page += 1) {
    const detail = liveDetailSchema.parse(
      await getJson(
        `${apiUrl()}/v1/live/runs/${encodeURIComponent(runId)}?after=${after}&limit=${RECORD_PAGE_LIMIT}`,
      ),
    );
    records.push(...detail.records);
    if (detail.next_cursor === null) return { detail, records };
    after = detail.next_cursor;
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
  let runs: LiveRunSummary[];
  try {
    runs = liveListSchema
      .parse(await getJson(`${apiUrl()}/v1/live/runs?limit=${RUN_LIST_LIMIT}`))
      .runs.map((run) => ({
        runId: run.run_id,
        taskId: run.registration.task_id,
        state: run.state,
        createdAt: run.created_at,
      }));
  } catch (error) {
    return {
      ...base,
      runs: [],
      error: `Live API unavailable: ${message(error)}`,
    };
  }
  // Show only the selected UUID; a missing selection is reported, never substituted.
  const runId = requested ?? runs[0]?.runId;
  if (!runId || (requested && !runs.some((r) => r.runId === requested)))
    return { ...base, runs, error: null };
  try {
    const { detail, records } = await loadRunRecords(runId);
    const payloads = records.map((r) => r.payload);
    const steps = payloads.filter((p) => p.record_type === "step_event");
    const samples = payloads.filter((p) => p.record_type === "resource_sample");
    return {
      ...base,
      runs,
      error: null,
      selected: {
        replay: {
          id: runId,
          title: detail.registration.task_id,
          description: "",
          run: { state: detail.state, reason: detail.reason },
          receivedAgoMs: latestReceiptAgeMs(records, Date.now()),
          view: readStepEvents(runId, steps),
        },
        resources: detail.registration.resources.map((r) => ({
          resourceId: r.resource_id,
          resourceClass: r.resource_class,
        })),
        wallSteps: wallSteps(steps),
        cpuSeries: cpuSeries(samples),
        gpu: await loadGpuEvidence(runId),
        recordCount: records.length,
      },
    };
  } catch (error) {
    return {
      ...base,
      runs,
      error: `Could not read live run ${runId}: ${message(error)}`,
    };
  }
}
