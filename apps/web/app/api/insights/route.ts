import { createHash } from "node:crypto";
import { z } from "zod";
import { aggregate } from "@/lib/dashboard-data";
import { loadDatasetSnapshot } from "@/lib/dataset-load";
import { loadExperiments } from "@/lib/experiments";
import {
  generateInsight,
  insightConfig,
  overviewFacts,
  runFacts,
  type Fact,
} from "@/lib/insights";
import { loadRunView } from "@/lib/run-load";
import { diagnoseView } from "@/lib/unified-runs";
export const dynamic = "force-dynamic";

const requestSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("overview") }),
  z.object({ kind: z.literal("run"), runId: z.uuid() }),
]);

const COOLDOWN_MS = 3_000;
const CACHE_MS = 5 * 60_000;
const CACHE_LIMIT = 50;
let lastStart = 0;
const cache = new Map<string, { at: number; body: unknown }>();

const STATUS = { unavailable: 503, rejected: 502, invalid_response: 502 };

/** Facts are always rebuilt here from the server's own data; the browser only names what to summarize. */
async function factsFor(
  request: z.infer<typeof requestSchema>,
): Promise<Fact[] | null> {
  if (request.kind === "run") {
    const loaded = await loadRunView(request.runId);
    return loaded
      ? runFacts(diagnoseView(loaded.view), loaded.view.source)
      : null;
  }
  const [data, dataset] = await Promise.all([
    loadExperiments(),
    loadDatasetSnapshot(),
  ]);
  return overviewFacts(
    aggregate(data.experiments),
    dataset.status === "ready"
      ? {
          manifest: dataset.manifest,
          runs: dataset.runs,
          release: dataset.manifest.dataset_version,
        }
      : undefined,
  );
}

export async function POST(request: Request) {
  const config = insightConfig(process.env);
  if (!config)
    return Response.json(
      { error: "AI summaries are not configured on this server." },
      { status: 404 },
    );
  const parsed = requestSchema.safeParse(
    await request.json().catch(() => null),
  );
  if (!parsed.success)
    return Response.json(
      { error: "Invalid summary request." },
      { status: 400 },
    );
  let facts: Fact[] | null;
  try {
    facts = await factsFor(parsed.data);
  } catch {
    return Response.json(
      { error: "The data to summarize could not be read." },
      { status: 503 },
    );
  }
  if (!facts)
    return Response.json({ error: "That run was not found." }, { status: 404 });

  const key = createHash("sha256")
    .update(`${config.model}\n${JSON.stringify(facts)}`)
    .digest("hex");
  const hit = cache.get(key);
  if (hit && Date.now() - hit.at < CACHE_MS)
    return Response.json({ ...(hit.body as object), cached: true });
  if (Date.now() - lastStart < COOLDOWN_MS)
    return Response.json(
      { error: "Wait a few seconds before asking again." },
      { status: 429 },
    );
  lastStart = Date.now();

  const result = await generateInsight(config, facts);
  if (!result.ok)
    return Response.json(
      { error: result.message },
      { status: STATUS[result.kind] },
    );
  const body = {
    facts: result.facts,
    insight: result.insight,
    model: result.model,
    generatedAt: new Date().toISOString(),
  };
  if (cache.size >= CACHE_LIMIT) cache.delete(cache.keys().next().value!);
  cache.set(key, { at: Date.now(), body });
  return Response.json({ ...body, cached: false });
}
