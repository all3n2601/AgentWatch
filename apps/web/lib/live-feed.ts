import "server-only";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { dataDirectory } from "./experiments";
import { parseGpuSamples, type GpuEvidence } from "./live-feed-data";

// The A01 nvidia-smi diagnostic for a live run, when a local probe recorded one under the
// data directory. A local-development convenience: it is not ingested evidence.

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

export async function loadGpuEvidence(runId: string): Promise<GpuEvidence> {
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
