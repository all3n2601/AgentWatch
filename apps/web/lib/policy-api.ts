import { z } from "zod";
import {
  failureStatus,
  LIVE_API_TIMEOUT_MS,
  type LiveApiFailure,
} from "./live-api";
import {
  policyChangeResultSchema,
  policyControlsSchema,
  policyPageSchema,
  policyVersionSchema,
  runPolicySchema,
  type PolicyChangeRequest,
  type PolicyChangeResultView,
  type PolicyControls,
  type PolicyPageView,
  type PolicyVersionView,
  type RunPolicyView,
} from "./policy-api-schema";

// Clients for the O02 concurrency policy API. `createPolicyApi` calls the worker from route
// handlers; `createPolicyBrowserApi` calls this app's routes, which return worker bodies or
// `{error, kind}` failures. Both share the live API result shape and timeout.

/** Versions per history page; the worker accepts 1–100. */
export const POLICY_HISTORY_PAGE_LIMIT = 20;
/** Policy change bodies are a few hundred bytes; anything far larger is not a change. */
export const MAX_POLICY_BODY_CHARS = 16_384;

export type PolicyFailure = {
  kind: LiveApiFailure["kind"] | "conflict";
  message: string;
};
export type PolicyResult<T> =
  { ok: true; data: T } | ({ ok: false } & PolicyFailure);

/** HTTP status a web route returns for each failure kind. */
export const policyFailureStatus: Record<PolicyFailure["kind"], number> = {
  ...failureStatus,
  conflict: 409,
};

const failureKinds = Object.keys(
  policyFailureStatus,
) as PolicyFailure["kind"][];
const routeErrorSchema = z.object({
  error: z.string(),
  kind: z.enum(failureKinds),
});
const serviceErrorSchema = z.object({
  detail: z.union([
    z.string(),
    z.array(
      z.looseObject({
        loc: z.array(z.union([z.string(), z.number()])),
        msg: z.string(),
      }),
    ),
  ]),
});

/** FastAPI validation locations start with "body"; the rest names the rejected field. */
function detailText(detail: z.infer<typeof serviceErrorSchema>["detail"]) {
  if (typeof detail === "string") return detail;
  return detail
    .map(({ loc, msg }) => {
      const field = loc.filter((part) => part !== "body").join(".");
      return field ? `${field}: ${msg}` : msg;
    })
    .join("; ");
}

function statusKind(status: number): PolicyFailure["kind"] {
  if (status === 404) return "not_found";
  if (status === 409) return "conflict";
  if (status === 400 || status === 413 || status === 422)
    return "invalid_request";
  return "unavailable";
}

/** Prefers a web route's classification, then the service's `detail`, then the status. */
async function failureOf(response: Response): Promise<PolicyFailure> {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    // An unreadable error body is classified by its status alone.
  }
  const routed = routeErrorSchema.safeParse(body);
  if (routed.success)
    return { kind: routed.data.kind, message: routed.data.error };
  const service = serviceErrorSchema.safeParse(body);
  return {
    kind: statusKind(response.status),
    message: service.success
      ? detailText(service.data.detail)
      : `The policy service responded with HTTP ${response.status}.`,
  };
}

type Transport = { fetchImpl?: typeof fetch; timeoutMs?: number };

function requester({
  fetchImpl = fetch,
  timeoutMs = LIVE_API_TIMEOUT_MS,
}: Transport) {
  return async function request<T>(
    url: string,
    schema: z.ZodType<T>,
    { body, signal }: { body?: unknown; signal?: AbortSignal } = {},
  ): Promise<PolicyResult<T>> {
    const timeout = AbortSignal.timeout(timeoutMs);
    let response: Response;
    try {
      response = await fetchImpl(url, {
        cache: "no-store",
        signal: signal ? AbortSignal.any([timeout, signal]) : timeout,
        ...(body !== undefined && {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
      });
    } catch {
      return {
        ok: false,
        kind: "unavailable",
        message: timeout.aborted
          ? `The policy service did not respond within ${timeoutMs / 1000} s.`
          : "The policy service could not be reached.",
      };
    }
    if (!response.ok) return { ok: false, ...(await failureOf(response)) };
    let data: unknown;
    try {
      data = await response.json();
    } catch {
      return {
        ok: false,
        kind: "invalid_response",
        message: "The policy service returned a body that is not JSON.",
      };
    }
    const parsed = schema.safeParse(data);
    return parsed.success
      ? { ok: true, data: parsed.data }
      : {
          ok: false,
          kind: "invalid_response",
          message: `The policy service response does not match the policy contract (${parsed.error.issues.length} issue${parsed.error.issues.length === 1 ? "" : "s"}).`,
        };
  };
}

/**
 * The worker's policy API for one target (`urls.policy`, e.g. `/v1/policies/concurrency/
 * coding-agent`) and the live runs collection that reports each run's policy.
 */
export function createPolicyApi(
  urls: { policy: string; liveRuns: string },
  transport: Transport = {},
) {
  const request = requester(transport);
  return {
    active(signal?: AbortSignal): Promise<PolicyResult<PolicyVersionView>> {
      return request(urls.policy, policyVersionSchema, { signal });
    },
    /** One page of versions, newest first; `before` continues from `next_before`. */
    history(
      before?: number,
      signal?: AbortSignal,
    ): Promise<PolicyResult<PolicyPageView>> {
      const query = new URLSearchParams({
        limit: String(POLICY_HISTORY_PAGE_LIMIT),
      });
      if (before !== undefined) query.set("before", String(before));
      return request(`${urls.policy}/versions?${query}`, policyPageSchema, {
        signal,
      });
    },
    change(
      body: PolicyChangeRequest,
      signal?: AbortSignal,
    ): Promise<PolicyResult<PolicyChangeResultView>> {
      return request(`${urls.policy}/versions`, policyChangeResultSchema, {
        body,
        signal,
      });
    },
    runPolicy(
      runId: string,
      signal?: AbortSignal,
    ): Promise<PolicyResult<RunPolicyView>> {
      return request(
        `${urls.liveRuns}/${encodeURIComponent(runId)}/policy`,
        runPolicySchema,
        { signal },
      );
    },
  };
}

/** This app's policy routes, as called from the browser. */
export function createPolicyBrowserApi({
  baseUrl = "",
  ...transport
}: Transport & { baseUrl?: string } = {}) {
  const request = requester(transport);
  const controlsUrl = `${baseUrl}/api/policies/concurrency`;
  return {
    controls(
      before?: number,
      signal?: AbortSignal,
    ): Promise<PolicyResult<PolicyControls>> {
      const query = before === undefined ? "" : `?before=${before}`;
      return request(`${controlsUrl}${query}`, policyControlsSchema, {
        signal,
      });
    },
    change(
      body: PolicyChangeRequest,
    ): Promise<PolicyResult<PolicyChangeResultView>> {
      return request(controlsUrl, policyChangeResultSchema, { body });
    },
    runPolicy(
      runId: string,
      signal?: AbortSignal,
    ): Promise<PolicyResult<RunPolicyView>> {
      return request(
        `${baseUrl}/api/live/runs/${encodeURIComponent(runId)}/policy`,
        runPolicySchema,
        { signal },
      );
    },
  };
}

/** A route response for a failed policy call, keeping its classification for the browser. */
export function policyFailureResponse(failure: PolicyFailure) {
  return Response.json(
    { error: failure.message, kind: failure.kind },
    { status: policyFailureStatus[failure.kind] },
  );
}
