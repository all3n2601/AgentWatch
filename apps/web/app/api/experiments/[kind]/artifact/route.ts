import { readFile } from "node:fs/promises";
import path from "node:path";
import {
  apiUrl,
  dataDirectory,
  isRunId,
  localArtifactRunId,
} from "@/lib/experiments";
export const dynamic = "force-dynamic";

/**
 * Downloads one evidence file. `run_id` reads that run from run history. `local_run_id`
 * reads the local artifact only if it still holds that run, so a file rewritten after the
 * page loaded is refused rather than served as the run the page showed. Without either,
 * the kind's local artifact is served as before; the UI always names a run.
 */
export async function GET(
  request: Request,
  context: { params: Promise<{ kind: string }> },
) {
  const { kind } = await context.params;
  const params = new URL(request.url).searchParams;
  const file = params.get("file") || "results.json";
  if (
    !["cpu", "coding"].includes(kind) ||
    !["results.json", "spans.jsonl", "samples.jsonl"].includes(file)
  ) {
    return Response.json({ error: "Unknown artifact" }, { status: 404 });
  }
  const runId = params.get("run_id");
  const localRunId = params.get("local_run_id");
  if (runId !== null) {
    if (!isRunId(runId))
      return Response.json({ error: "Invalid run ID" }, { status: 400 });
    try {
      const response = await fetch(
        `${apiUrl()}/v1/runs/${runId}/artifacts/${file}`,
        { cache: "no-store", signal: AbortSignal.timeout(5000) },
      );
      if (!response.ok)
        return Response.json(
          { error: "Saved artifact unavailable" },
          { status: response.status },
        );
      return new Response(await response.text(), {
        headers: {
          "Content-Type":
            response.headers.get("Content-Type") || "application/json",
          "Content-Disposition": `attachment; filename="agentwatch-${runId}-${file}"`,
          "Cache-Control": "no-store",
        },
      });
    } catch {
      return Response.json(
        { error: "Run history unavailable" },
        { status: 503 },
      );
    }
  }
  if (localRunId !== null) {
    if (!isRunId(localRunId))
      return Response.json({ error: "Invalid run ID" }, { status: 400 });
    const recorded = await localArtifactRunId(kind);
    if (recorded === null)
      return Response.json({ error: "Artifact unavailable" }, { status: 404 });
    if (recorded !== localRunId)
      return Response.json(
        {
          error:
            "The local artifact files now hold a different run. Reload the page to see it.",
        },
        { status: 409 },
      );
  }
  try {
    const content = await readFile(
      path.join(dataDirectory(), `${kind}-demo`, file),
      "utf8",
    );
    return new Response(content, {
      headers: {
        "Content-Type": file.endsWith("jsonl")
          ? "application/x-ndjson"
          : "application/json",
        "Content-Disposition": `attachment; filename="agentwatch-${localRunId ?? kind}-${file}"`,
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return Response.json({ error: "Artifact unavailable" }, { status: 404 });
  }
}
