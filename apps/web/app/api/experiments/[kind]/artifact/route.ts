import { readFile } from "node:fs/promises";
import path from "node:path";
import { apiUrl, dataDirectory } from "@/lib/experiments";
export const dynamic = "force-dynamic";
export async function GET(
  request: Request,
  context: { params: Promise<{ kind: string }> },
) {
  const { kind } = await context.params;
  const file = new URL(request.url).searchParams.get("file") || "results.json";
  if (
    !["cpu", "coding"].includes(kind) ||
    !["results.json", "spans.jsonl", "samples.jsonl"].includes(file)
  ) {
    return Response.json({ error: "Unknown artifact" }, { status: 404 });
  }
  const runId = new URL(request.url).searchParams.get("run_id");
  if (runId) {
    if (!/^[0-9a-f-]{36}$/i.test(runId))
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
        "Content-Disposition": `attachment; filename="agentwatch-${kind}-${file}"`,
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return Response.json({ error: "Artifact unavailable" }, { status: 404 });
  }
}
