import "server-only";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import { experimentSchema, type DashboardData } from "./experiment-schema";

/** Run ids are UUIDs in both run history and the worker's local artifacts. */
export const isRunId = (value: string) => z.uuid().safeParse(value).success;

/** Why a local evidence file was not served for the run a page showed. */
export type LocalArtifactRead =
  | { status: "ok"; content: string }
  /** The kind has no local artifact, or the file is missing or unreadable. */
  | { status: "unavailable" }
  /** results.json now records another run. */
  | { status: "other-run" }
  /** The file is newer than results.json: a newer run is writing it and has not finished. */
  | { status: "unfinished" };

const resultsRunId = (content: string) => {
  try {
    const parsed = z
      .object({ run_id: z.string() })
      .safeParse(JSON.parse(content));
    return parsed.success ? parsed.data.run_id : null;
  } catch {
    return null;
  }
};

/**
 * Reads one local evidence file of `kind` only if it still belongs to `runId`. The worker
 * demo (services/worker/src/agentwatch_worker/demo.py) writes spans.jsonl and samples.jsonl
 * during a run and results.json last, so a file belongs to the run in results.json only
 * while it is no newer than results.json. The checks run after the file is read, so a file
 * rewritten between the check and the read is refused too.
 */
export async function readLocalArtifact(
  kind: string,
  file: string,
  runId: string,
): Promise<LocalArtifactRead> {
  const directory = path.join(dataDirectory(), `${kind}-demo`);
  try {
    const content = await readFile(path.join(directory, file), "utf8");
    const [fileStat, resultsStat, results] = await Promise.all([
      stat(path.join(directory, file)),
      stat(path.join(directory, "results.json")),
      readFile(path.join(directory, "results.json"), "utf8"),
    ]);
    if (fileStat.mtimeMs > resultsStat.mtimeMs) return { status: "unfinished" };
    const recorded = resultsRunId(results);
    if (recorded === null) return { status: "unavailable" };
    return recorded === runId
      ? { status: "ok", content }
      : { status: "other-run" };
  } catch {
    return { status: "unavailable" };
  }
}

export function dataDirectory() {
  return (
    process.env.AGENTWATCH_DATA_DIR ||
    path.resolve(process.cwd(), "../../.data")
  );
}
export function apiUrl() {
  return process.env.AGENTWATCH_API_URL || "http://127.0.0.1:8090";
}
/** The B01 live runs collection; live runs are not part of saved run history. */
export const liveRunsUrl = () => `${apiUrl()}/v1/live/runs`;
/** The O02 concurrency policy of the only supported target, `coding-agent`. */
export const policyUrl = () =>
  `${apiUrl()}/v1/policies/concurrency/coding-agent`;
const historySchema = z.object({
  runs: z.array(
    z.object({
      kind: z.enum(["coding", "cpu"]),
      result: experimentSchema,
      created_at: z.string(),
    }),
  ),
  next_cursor: z.number().nullable(),
});
export async function loadExperiments(before?: number): Promise<DashboardData> {
  try {
    const response = await fetch(
      `${apiUrl()}/v1/runs?limit=100${before ? `&before=${before}` : ""}`,
      { cache: "no-store", signal: AbortSignal.timeout(3000) },
    );
    if (!response.ok) throw new Error("History unavailable");
    const history = historySchema.parse(await response.json());
    return {
      experiments: history.runs.map((run) => ({
        ...run.result,
        kind: run.kind,
        persisted: true,
        stored_at: run.created_at,
      })),
      warnings: [],
      source: "database",
      nextCursor: history.next_cursor,
      loadedAt: new Date().toISOString(),
    };
  } catch {
    if (before) throw new Error("Older history could not be loaded");
  }
  const data: DashboardData = {
    experiments: [],
    warnings: [
      "Run history is unavailable. Showing the latest local artifacts.",
    ],
    source: "local",
    loadedAt: new Date().toISOString(),
  };
  for (const kind of ["coding", "cpu"] as const) {
    try {
      const raw = await readFile(
        path.join(dataDirectory(), `${kind}-demo/results.json`),
        "utf8",
      );
      data.experiments.push({
        ...experimentSchema.parse(JSON.parse(raw)),
        kind,
      });
    } catch (error) {
      data.warnings.push(
        (error as NodeJS.ErrnoException).code === "ENOENT"
          ? `${kind} experiment has no recorded run yet.`
          : `${kind} experiment could not be read.`,
      );
    }
  }
  return data;
}

export async function loadRequestedRuns(
  ids: (string | undefined)[],
): Promise<DashboardData> {
  const data = await loadExperiments();
  await Promise.all(
    [...new Set(ids.filter((id): id is string => !!id))].map(async (id) => {
      if (data.experiments.some((run) => run.run_id === id)) return;
      try {
        if (!isRunId(id)) throw new Error("Invalid run ID");
        const response = await fetch(`${apiUrl()}/v1/runs/${id}`, {
          cache: "no-store",
          signal: AbortSignal.timeout(3000),
        });
        if (!response.ok) throw new Error("Run unavailable");
        const saved = historySchema.shape.runs.element.parse(
          await response.json(),
        );
        data.experiments.push({
          ...saved.result,
          kind: saved.kind,
          persisted: true,
          stored_at: saved.created_at,
        });
      } catch {
        data.warnings.push(
          `Requested run ${id.slice(0, 8)} could not be loaded. Select another recorded run.`,
        );
      }
    }),
  );
  return data;
}
