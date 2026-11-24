import { readFile } from "node:fs/promises";
import path from "node:path";
import {
  apiUrl,
  dataDirectory,
  isRunId,
  readLocalArtifact,
} from "@/lib/experiments";
export const dynamic = "force-dynamic";

const contentType = (file: string) =>
  file.endsWith("jsonl") ? "application/x-ndjson" : "application/json";

/**
 * Downloads one evidence file. `run_id` reads that run from run history. `local_run_id`
 * reads the local artifact only while it still holds that run (see `readLocalArtifact`):
 * a file rewritten after the page loaded, or still being rewritten by a newer run, is
 * refused (409) rather than served as the run the page showed. Without either parameter
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
    const local = await readLocalArtifact(kind, file, localRunId);
    if (local.status === "unavailable")
      return Response.json({ error: "Artifact unavailable" }, { status: 404 });
    if (local.status !== "ok")
      return Response.json(
        {
          error:
            local.status === "other-run"
              ? "The local artifact files now hold a different run. Reload the page to see it."
              : "A newer run is rewriting the local artifact files. Reload the page once it finishes.",
        },
        { status: 409 },
      );
    return new Response(local.content, {
      headers: {
        "Content-Type": contentType(file),
        "Content-Disposition": `attachment; filename="agentwatch-${localRunId}-${file}"`,
        "Cache-Control": "no-store",
      },
    });
  }
  try {
    const content = await readFile(
      path.join(dataDirectory(), `${kind}-demo`, file),
      "utf8",
    );
    return new Response(content, {
      headers: {
        "Content-Type": contentType(file),
        "Content-Disposition": `attachment; filename="agentwatch-${kind}-${file}"`,
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return Response.json({ error: "Artifact unavailable" }, { status: 404 });
  }
}
