import { liveRunsUrl } from "@/lib/experiments";
import {
  createLiveApi,
  failureStatus,
  isLiveRunId,
  parseAttributionQuery,
} from "@/lib/live-api";
export const dynamic = "force-dynamic";

/**
 * One page of the selected live run's step attributions (B03), read by UUID. An unknown run
 * is a 404; no other run's rows are ever substituted.
 */
export async function GET(
  request: Request,
  context: { params: Promise<{ runId: string }> },
) {
  const { runId } = await context.params;
  if (!isLiveRunId(runId))
    return Response.json({ error: "Invalid live run ID" }, { status: 400 });
  const query = parseAttributionQuery(new URL(request.url).searchParams);
  if (!query.success)
    return Response.json(
      { error: "Invalid attribution cursor or page size" },
      { status: 400 },
    );
  const result = await createLiveApi(liveRunsUrl()).attributions(
    runId,
    query.data.after,
    { limit: query.data.limit },
  );
  if (!result.ok)
    return Response.json(
      { error: result.message, kind: result.kind },
      { status: failureStatus[result.kind] },
    );
  return Response.json(result.data, {
    headers: { "Cache-Control": "no-store" },
  });
}
