import "server-only";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { parquetReadObjects } from "hyparquet";
import { z } from "zod";
import { datasetDirectory, loadDatasetSnapshot } from "./dataset-load";
import {
  buildExperimentsPage,
  type BuildReport,
  type ExperimentsPage,
  type ReplayStep,
} from "./experiments-page";

export type ExperimentsPageState =
  | { status: "ready"; page: ExperimentsPage }
  | { status: "unavailable"; reason: string }
  | { status: "empty"; release: string };

/** Parquet int64 columns arrive as bigint; replay positions are small. */
const replayIndex = z.preprocess(
  (value) => (typeof value === "bigint" ? Number(value) : value),
  z.number().int().positive().nullable(),
);

/** The `steps.parquet` columns behind replay roles and label outcomes. */
const REPLAY_STEP_COLUMNS = [
  "kind",
  "experiment",
  "replay_index",
  "replay_role",
  "label_basis",
] as const;

const replayStepRowSchema = z.object({
  kind: z.string(),
  experiment: z.string().nullable(),
  replay_index: replayIndex,
  replay_role: z.string().nullable(),
  label_basis: z.string(),
});

const buildReportSchema = z.object({
  build_id: z.string(),
  dataset_version: z.string(),
  experiments: z.number().int().nonnegative().optional(),
  rejections: z.array(
    z.object({ origin: z.string(), reason: z.string(), stage: z.string() }),
  ),
});

/**
 * Replay steps' protocol role and label outcome. These columns are not part of
 * the shared run summary, so they are read here; null when they cannot be read
 * or do not match the expected types, never an empty list.
 */
async function readReplaySteps(
  root: string,
  release: string,
): Promise<ReplayStep[] | null> {
  try {
    const bytes = await readFile(path.join(root, release, "steps.parquet"));
    const file = bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength,
    );
    const rows = await parquetReadObjects({
      file,
      columns: [...REPLAY_STEP_COLUMNS],
    });
    return rows.flatMap((raw) => {
      const row = replayStepRowSchema.parse(raw);
      return row.kind === "agent_replay" &&
        row.experiment !== null &&
        row.replay_index !== null &&
        row.replay_role !== null
        ? [
            {
              experiment: row.experiment,
              replayIndex: row.replay_index,
              replayRole: row.replay_role,
              labelBasis: row.label_basis,
            },
          ]
        : [];
    });
  } catch {
    return null;
  }
}

/** The newest build report naming `release`; null when none is readable. */
async function readBuildReport(
  root: string,
  release: string,
): Promise<BuildReport | null> {
  const builds = path.join(root, "builds");
  let names: string[];
  try {
    names = (await readdir(builds))
      .filter((name) => name.endsWith(".json"))
      .sort()
      .reverse();
  } catch {
    return null;
  }
  for (const name of names) {
    let raw: unknown;
    try {
      raw = JSON.parse(await readFile(path.join(builds, name), "utf8"));
    } catch {
      continue; // An unreadable report for another release must not hide this one's.
    }
    const parsed = buildReportSchema.safeParse(raw);
    if (parsed.success && parsed.data.dataset_version === release)
      return {
        buildId: parsed.data.build_id,
        experiments: parsed.data.experiments ?? null,
        rejections: parsed.data.rejections,
      };
  }
  return null;
}

/** The Experiments page for the current dataset release. */
export async function loadExperimentsPage(
  root = datasetDirectory(),
): Promise<ExperimentsPageState> {
  const snapshot = await loadDatasetSnapshot(root);
  if (snapshot.status !== "ready")
    return { status: "unavailable", reason: snapshot.reason };
  const release = snapshot.manifest.dataset_version;
  const [replaySteps, build] = await Promise.all([
    readReplaySteps(root, release),
    readBuildReport(root, release),
  ]);
  const page = buildExperimentsPage({
    release,
    runs: snapshot.runs,
    validation: {
      status: snapshot.validation.status,
      validatedAt: snapshot.validation.validated_at,
      gateVersion: snapshot.validation.gate_version,
    },
    replaySteps,
    build,
  });
  return page ? { status: "ready", page } : { status: "empty", release };
}
