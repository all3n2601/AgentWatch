import "server-only";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import { experimentSchema, type DashboardData } from "./experiment-schema";

export function dataDirectory() {
  return (
    process.env.AGENTWATCH_DATA_DIR ||
    path.resolve(process.cwd(), "../../.data")
  );
}
export function apiUrl() {
  return process.env.AGENTWATCH_API_URL || "http://127.0.0.1:8090";
}
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
        if (!z.uuid().safeParse(id).success) throw new Error("Invalid run ID");
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
