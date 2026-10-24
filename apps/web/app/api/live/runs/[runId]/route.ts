import { apiUrl } from "@/lib/experiments";
import {
  createLiveApi,
  failureStatus,
  isLiveRunId,
  parseSnapshotQuery,
} from "@/lib/live-api";
export const dynamic = "force-dynamic";

/**
 * The selected live run's snapshot and one record page, fetched by UUID directly so any
 * recorded run can be selected, not only those on the newest history page.
 */
export async function GET(
  request: Request,
  context: { params: Promise<{ runId: string }> },
) {
  const { runId } = await context.params;
  if (!isLiveRunId(runId))
    return Response.json({ error: "Invalid live run ID" }, { status: 400 });
  const query = parseSnapshotQuery(new URL(request.url).searchParams);
  if (!query.success)
    return Response.json(
      { error: "Invalid record cursor or page size" },
      { status: 400 },
    );
  const result = await createLiveApi(apiUrl()).runSnapshot(
    runId,
    query.data.after,
    query.data.limit,
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
