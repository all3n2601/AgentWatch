import { liveRunsUrl } from "@/lib/experiments";
import { createLiveApi, failureStatus, isLiveRunId } from "@/lib/live-api";
import { liveExportFilename, readLiveRecordsExport } from "@/lib/live-export";
export const dynamic = "force-dynamic";

/**
 * Downloads the selected live run's records as JSON. Only that UUID is read; there is no
 * fallback to a local file or another run when it cannot be read.
 */
export async function GET(
  _request: Request,
  context: { params: Promise<{ runId: string }> },
) {
  const { runId } = await context.params;
  if (!isLiveRunId(runId))
    return Response.json({ error: "Invalid live run ID" }, { status: 400 });
  const result = await readLiveRecordsExport(
    createLiveApi(liveRunsUrl()),
    runId,
  );
  if (!result.ok)
    return Response.json(
      { error: result.message, kind: result.kind },
      { status: failureStatus[result.kind] },
    );
  return new Response(JSON.stringify(result.data, null, 2), {
    headers: {
      "Content-Type": "application/json",
      "Content-Disposition": `attachment; filename="${liveExportFilename(runId)}"`,
      "Cache-Control": "no-store",
    },
  });
}
