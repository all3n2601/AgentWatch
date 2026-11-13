import { liveRunsUrl, policyUrl } from "@/lib/experiments";
import { createPolicyApi, policyFailureResponse } from "@/lib/policy-api";
export const dynamic = "force-dynamic";

/**
 * The active coding-agent policy and only the newest history entry. Views that show the
 * current policy beside other evidence read this instead of a full history page; the entry
 * lets them check that both reads saw the same version (`controlsConsistent`).
 */
export async function GET() {
  const policy = createPolicyApi({
    policy: policyUrl(),
    liveRuns: liveRunsUrl(),
  });
  const [active, history] = await Promise.all([
    policy.active(),
    policy.history(undefined, undefined, 1),
  ]);
  if (!active.ok) return policyFailureResponse(active);
  if (!history.ok) return policyFailureResponse(history);
  return Response.json(
    { active: active.data, history: history.data },
    { headers: { "Cache-Control": "no-store" } },
  );
}
