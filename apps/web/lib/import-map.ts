import { readFile } from "node:fs/promises";
import path from "node:path";
import { z } from "zod";

const mapSchema = z.record(z.uuid(), z.uuid());

/**
 * The recorded-run-to-live-run UUID map written by `make agent-import MAP=...`. Optional: without
 * it dataset runs simply do not link to their live run. Malformed content is ignored, not trusted.
 */
export async function loadImportMap(
  file = process.env.AGENTWATCH_IMPORT_MAP ||
    path.resolve(process.cwd(), "../../.data/live-import-map.json"),
): Promise<Record<string, string>> {
  try {
    const parsed = mapSchema.safeParse(
      JSON.parse(await readFile(file, "utf8")),
    );
    return parsed.success ? parsed.data : {};
  } catch {
    return {};
  }
}
