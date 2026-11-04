import { liveRunsUrl, policyUrl } from "@/lib/experiments";
import { parseListQuery } from "@/lib/live-api";
import {
  createPolicyApi,
  MAX_POLICY_BODY_BYTES,
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
  const origin = request.headers.get("origin");
  // Next can normalize the URL to its listening hostname. Host is the authority the client used.
  const url = new URL(request.url);
  const requestOrigin = `${url.protocol}//${request.headers.get("host") ?? url.host}`;
  if (
    request.headers.get("sec-fetch-site") === "cross-site" ||
    (origin !== null && origin !== requestOrigin)
  ) {
    return Response.json(
      {
        kind: "invalid_request",
        error: "Cross-origin policy changes are forbidden.",
      },
      { status: 403 },
    );
  }
  const mediaType = request.headers
    .get("content-type")
    ?.split(";")[0]
    .trim()
    .toLowerCase();
  if (mediaType !== "application/json") {
    return Response.json(
      {
        kind: "invalid_request",
        error: "Policy changes require application/json.",
      },
      { status: 415 },
    );
  }

  const reader = request.body?.getReader();
  const bytes = new Uint8Array(MAX_POLICY_BODY_BYTES);
  let length = 0;
  if (reader) {
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (length + value.byteLength > bytes.length) {
          await reader.cancel();
          return Response.json(
            {
              kind: "invalid_request",
              error: "The policy change is too large.",
            },
            { status: 413 },
          );
        }
        bytes.set(value, length);
        length += value.byteLength;
      }
    } finally {
      reader.releaseLock();
    }
  }
  let body: unknown;
  try {
    body = JSON.parse(
      new TextDecoder("utf-8", { fatal: true }).decode(
        bytes.subarray(0, length),
      ),
    );
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
