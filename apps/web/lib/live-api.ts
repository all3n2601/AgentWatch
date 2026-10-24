import { z } from "zod";
import {
  MAX_LIVE_CURSOR,
  runPageSchema,
  runSnapshotSchema,
  type LiveRunPage,
  type LiveRunSnapshot,
} from "./live-api-schema";

// Reads the B01 live read API. The base URL and fetch are passed in so route handlers keep
// the server-only configuration and tests can substitute the transport.

/** Covers the service's 3 s lock and 5 s statement timeouts without hanging a poll forever. */
export const LIVE_API_TIMEOUT_MS = 8_000;
export const LIVE_RUN_LIST_LIMIT = 50;
export const LIVE_RECORD_PAGE_LIMIT = 256;

export type LiveApiFailure = {
  kind: "not_found" | "invalid_request" | "unavailable" | "invalid_response";
  message: string;
};
export type LiveApiResult<T> =
  { ok: true; data: T } | ({ ok: false } & LiveApiFailure);

/** HTTP status a web route returns for each failure kind. */
export const failureStatus: Record<LiveApiFailure["kind"], number> = {
  not_found: 404,
  invalid_request: 422,
  unavailable: 503,
  invalid_response: 502,
};

const cursor = z.coerce.number().int().nonnegative().max(MAX_LIVE_CURSOR);
const snapshotQuerySchema = z.object({
  after: cursor.default(0),
  limit: z.coerce
    .number()
    .int()
    .min(1)
    .max(LIVE_RECORD_PAGE_LIMIT)
    .default(LIVE_RECORD_PAGE_LIMIT),
});
const listQuerySchema = z.object({ before: cursor.positive().optional() });

const present = (params: URLSearchParams, names: string[]) =>
  Object.fromEntries(
    names.flatMap((name) => {
      const value = params.get(name);
      return value === null || value === "" ? [] : [[name, value]];
    }),
  );

export function parseSnapshotQuery(params: URLSearchParams) {
  return snapshotQuerySchema.safeParse(present(params, ["after", "limit"]));
}
export function parseListQuery(params: URLSearchParams) {
  return listQuerySchema.safeParse(present(params, ["before"]));
}
export const isLiveRunId = (value: string) => z.uuid().safeParse(value).success;

const failureKinds = Object.keys(failureStatus) as LiveApiFailure["kind"][];
const routeErrorSchema = z.object({
  error: z.string(),
  kind: z.enum(failureKinds),
});

function statusFailure(status: number): LiveApiFailure {
  if (status === 404)
    return { kind: "not_found", message: "The live API has no such run." };
  if (status === 400 || status === 422)
    return {
      kind: "invalid_request",
      message: "The live API rejected the request parameters.",
    };
  return {
    kind: "unavailable",
    message: `The live API responded with HTTP ${status}.`,
  };
}

/** Prefers the failure a web route already classified; B01 errors are mapped by status. */
async function failureOf(response: Response): Promise<LiveApiFailure> {
  try {
    const routed = routeErrorSchema.safeParse(await response.json());
    if (routed.success)
      return { kind: routed.data.kind, message: routed.data.error };
  } catch {
    // An unreadable error body is classified by its status alone.
  }
  return statusFailure(response.status);
}

/**
 * A client for the live runs collection at `runsUrl`: B01's `/v1/live/runs` on the server,
 * or this app's `/api/live/runs` routes in the browser, which return the same bodies.
 */
export function createLiveApi(
  runsUrl: string,
  {
    fetchImpl = fetch,
    timeoutMs = LIVE_API_TIMEOUT_MS,
  }: { fetchImpl?: typeof fetch; timeoutMs?: number } = {},
) {
  async function get<T>(
    path: string,
    schema: z.ZodType<T>,
    signal?: AbortSignal,
  ): Promise<LiveApiResult<T>> {
    const timeout = AbortSignal.timeout(timeoutMs);
    let response: Response;
    try {
      response = await fetchImpl(`${runsUrl}${path}`, {
        cache: "no-store",
        signal: signal ? AbortSignal.any([timeout, signal]) : timeout,
      });
    } catch {
      return {
        ok: false,
        kind: "unavailable",
        message: timeout.aborted
          ? `The live API did not respond within ${timeoutMs / 1000} s.`
          : "The live API could not be reached.",
      };
    }
    if (!response.ok) return { ok: false, ...(await failureOf(response)) };
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      return {
        ok: false,
        kind: "invalid_response",
        message: "The live API returned a body that is not JSON.",
      };
    }
    const parsed = schema.safeParse(body);
    return parsed.success
      ? { ok: true, data: parsed.data }
      : {
          ok: false,
          kind: "invalid_response",
          message: `The live API response does not match live-contracts-v1 (${parsed.error.issues.length} issue${parsed.error.issues.length === 1 ? "" : "s"}).`,
        };
  }

  return {
    /** One page of live runs, newest first; `before` continues from a previous page. */
    runPage(
      before?: number,
      signal?: AbortSignal,
    ): Promise<LiveApiResult<LiveRunPage>> {
      const query = new URLSearchParams({ limit: String(LIVE_RUN_LIST_LIMIT) });
      if (before !== undefined) query.set("before", String(before));
      return get(`?${query}`, runPageSchema, signal);
    },
    /** The run's current snapshot with one page of records after `after`. */
    runSnapshot(
      runId: string,
      after = 0,
      {
        limit = LIVE_RECORD_PAGE_LIMIT,
        signal,
      }: { limit?: number; signal?: AbortSignal } = {},
    ): Promise<LiveApiResult<LiveRunSnapshot>> {
      const query = new URLSearchParams({
        after: String(after),
        limit: String(limit),
      });
      return get(
        `/${encodeURIComponent(runId)}?${query}`,
        runSnapshotSchema,
        signal,
      );
    },
  };
}
export type LiveApi = ReturnType<typeof createLiveApi>;
