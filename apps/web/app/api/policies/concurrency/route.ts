import { liveRunsUrl, policyUrl } from "@/lib/experiments";
import { parseListQuery } from "@/lib/live-api";
import {
  createPolicyApi,
  MAX_POLICY_BODY_CHARS,
  policyFailureResponse,
} from "@/lib/policy-api";
import { policyChangeSchema } from "@/lib/policy-api-schema";
export const dynamic = "force-dynamic";

const api = () =>
  createPolicyApi({ policy: policyUrl(), liveRuns: liveRunsUrl() });

/** The active coding-agent policy and one page of its versions, newest first. */
export async function GET(request: Request) {
  const query = parseListQuery(new URL(request.url).searchParams);
  if (!query.success)
    return policyFailureResponse({
      kind: "invalid_request",
      message: "Invalid history cursor.",
    });
  const policy = api();
  const [active, history] = await Promise.all([
    policy.active(),
    policy.history(query.data.before),
  ]);
  if (!active.ok) return policyFailureResponse(active);
  if (!history.ok) return policyFailureResponse(history);
  return Response.json(
    { active: active.data, history: history.data },
    { headers: { "Cache-Control": "no-store" } },
  );
}

/** Records a validated policy change; 201 when stored, 200 when the request was a replay. */
export async function POST(request: Request) {
  const text = await request.text();
  if (text.length > MAX_POLICY_BODY_CHARS)
    return policyFailureResponse({
      kind: "invalid_request",
      message: "The policy change is too large.",
    });
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return policyFailureResponse({
      kind: "invalid_request",
      message: "The policy change is not JSON.",
    });
  }
  const change = policyChangeSchema.safeParse(body);
  if (!change.success)
    return policyFailureResponse({
      kind: "invalid_request",
      message: `The policy change is invalid: ${change.error.issues
        .map((issue) => `${issue.path.join(".") || "body"}: ${issue.message}`)
        .join("; ")}`,
    });
  const result = await api().change(change.data);
  if (!result.ok) return policyFailureResponse(result);
  return Response.json(result.data, {
    status: result.data.created ? 201 : 200,
    headers: { "Cache-Control": "no-store" },
  });
}
