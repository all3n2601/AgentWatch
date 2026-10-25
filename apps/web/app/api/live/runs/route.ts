import { liveRunsUrl } from "@/lib/experiments";
import { createLiveApi, failureStatus, parseListQuery } from "@/lib/live-api";
export const dynamic = "force-dynamic";

/** One page of live runs from the B01 read API, validated before it reaches the browser. */
export async function GET(request: Request) {
  const query = parseListQuery(new URL(request.url).searchParams);
  if (!query.success)
    return Response.json({ error: "Invalid history cursor" }, { status: 400 });
  const result = await createLiveApi(liveRunsUrl()).runPage(query.data.before);
  if (!result.ok)
    return Response.json(
      { error: result.message, kind: result.kind },
      { status: failureStatus[result.kind] },
    );
  return Response.json(result.data, {
    headers: { "Cache-Control": "no-store" },
  });
}
