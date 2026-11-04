import { liveRunsUrl, policyUrl } from "@/lib/experiments";
import { isLiveRunId } from "@/lib/live-api";
import { createPolicyApi, policyFailureResponse } from "@/lib/policy-api";
export const dynamic = "force-dynamic";

/** The concurrency policy versions that were active while the selected live run was open. */
export async function GET(
  _request: Request,
  context: { params: Promise<{ runId: string }> },
) {
  const { runId } = await context.params;
  if (!isLiveRunId(runId))
    return policyFailureResponse({
      kind: "invalid_request",
      message: "Invalid live run ID.",
    });
  const result = await createPolicyApi({
    policy: policyUrl(),
    liveRuns: liveRunsUrl(),
  }).runPolicy(runId);
  if (!result.ok) return policyFailureResponse(result);
  return Response.json(result.data, {
    headers: { "Cache-Control": "no-store" },
  });
}
