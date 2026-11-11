import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { parquetReadObjects } from "hyparquet";
import { z } from "zod";
import {
  DATASET_COLUMNS,
  parseStepRows,
  summarizeRuns,
  type DatasetRun,
  type StepRow,
} from "./dataset-runs";
import {
  manifestSchema,
  splitSchema,
  validationSchema,
  type DatasetSnapshot,
} from "./dataset-schema";

const buildReportSchema = z.object({ dataset_version: z.string() });
class DatasetMismatchError extends Error {}
const FETCH_HINT = "Run `make pull-dataset` to fetch the DVC-tracked dataset.";

export function datasetDirectory() {
  return (
    process.env.AGENTWATCH_DATASET_DIR ||
    path.resolve(process.cwd(), "../../data/datasets")
  );
}

async function readJson<T>(file: string, schema: z.ZodType<T>): Promise<T> {
  return schema.parse(JSON.parse(await readFile(file, "utf8")));
}

async function lastEntry(directory: string, accept: (name: string) => boolean) {
  const names = (await readdir(directory)).filter(accept).sort();
  const name = names.at(-1);
  if (!name) throw new Error(`Nothing found in ${path.basename(directory)}`);
  return name;
}

/** The release named by the newest build report; build report names sort by build time. */
export async function currentRelease(root: string) {
  const builds = path.join(root, "builds");
  const names = (await readdir(builds))
    .filter((name) => name.endsWith(".json"))
    .sort()
    .reverse();
  for (const name of names) {
    const parsed = buildReportSchema.safeParse(
      JSON.parse(await readFile(path.join(builds, name), "utf8")),
    );
    if (parsed.success) return parsed.data.dataset_version;
  }
  throw new Error("No build report names a dataset release");
}

/** The release's step rows, validated; a column that changed type fails loudly, not silently. */
export async function readStepRows(
  root: string,
  release: string,
): Promise<StepRow[]> {
  const bytes = await readFile(path.join(root, release, "steps.parquet"));
  const file = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
  return parseStepRows(
    await parquetReadObjects({ file, columns: [...DATASET_COLUMNS] }),
  );
}

/**
 * Reads the current dataset release's manifest, `split-v2` assignment and quality-gate result
 * from the local DVC checkout. Missing or malformed files are reported as unavailable, never as
 * an empty dataset.
 */
export async function loadDatasetSnapshot(
  root = datasetDirectory(),
): Promise<DatasetSnapshot> {
  try {
    const release = await currentRelease(root);
    const manifest = await readJson(
      path.join(root, release, "manifest.json"),
      manifestSchema,
    );
    const splitDir = path.join(root, "splits", release);
    const split = await readJson(
      path.join(
        splitDir,
        await lastEntry(splitDir, (name) => name.endsWith(".json")),
      ),
      splitSchema,
    );
    const validationDir = path.join(root, "validations", release);
    const attempt = await lastEntry(validationDir, () => true);
    const validation = await readJson(
      path.join(validationDir, attempt, "result.json"),
      validationSchema,
    );
    const rows = await readStepRows(root, release);
    const runs: DatasetRun[] = summarizeRuns(rows);
    if (rows.length !== manifest.rows || runs.length !== manifest.runs)
      throw new DatasetMismatchError();
    return { status: "ready", manifest, split, validation, runs };
  } catch (error) {
    const detail =
      error instanceof DatasetMismatchError
        ? "The step file and manifest disagree about the row or run count."
        : error instanceof z.ZodError
          ? "A dataset file does not match the expected format."
          : error instanceof Error && "code" in error && error.code === "ENOENT"
            ? "Dataset files were not found."
            : "The dataset could not be read.";
    return { status: "unavailable", reason: `${detail} ${FETCH_HINT}` };
  }
}
