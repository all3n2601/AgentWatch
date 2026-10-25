import { isLiveRunId } from "@/lib/live-api";
import { loadGpuEvidence } from "@/lib/live-feed";
export const dynamic = "force-dynamic";

/** The local A01 nvidia-smi diagnostic for a live run, or why there is none. */
export async function GET(
  _request: Request,
  context: { params: Promise<{ runId: string }> },
) {
  const { runId } = await context.params;
  if (!isLiveRunId(runId))
    return Response.json({ error: "Invalid live run ID" }, { status: 400 });
  return Response.json(await loadGpuEvidence(runId), {
    headers: { "Cache-Control": "no-store" },
  });
}
