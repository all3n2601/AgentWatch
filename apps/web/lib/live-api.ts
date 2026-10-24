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

function upstreamFailure(status: number): LiveApiFailure {
  if (status === 404)
    return { kind: "not_found", message: "The live API has no such run." };
  if (status === 422)
    return {
      kind: "invalid_request",
      message: "The live API rejected the request parameters.",
    };
  return {
    kind: "unavailable",
    message: `The live API responded with HTTP ${status}.`,
  };
}

export function createLiveApi(
  baseUrl: string,
  fetchImpl: typeof fetch = fetch,
) {
  async function get<T>(
    path: string,
    schema: z.ZodType<T>,
  ): Promise<LiveApiResult<T>> {
    let response: Response;
    try {
      response = await fetchImpl(`${baseUrl}${path}`, {
        cache: "no-store",
        signal: AbortSignal.timeout(LIVE_API_TIMEOUT_MS),
      });
    } catch (error) {
      const timedOut =
        error instanceof DOMException && error.name === "TimeoutError";
      return {
        ok: false,
        kind: "unavailable",
        message: timedOut
          ? `The live API did not respond within ${LIVE_API_TIMEOUT_MS / 1000} s.`
          : "The live API could not be reached.",
      };
    }
    if (!response.ok) return { ok: false, ...upstreamFailure(response.status) };
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
    runPage(before?: number): Promise<LiveApiResult<LiveRunPage>> {
      const query = new URLSearchParams({ limit: String(LIVE_RUN_LIST_LIMIT) });
      if (before !== undefined) query.set("before", String(before));
      return get(`/v1/live/runs?${query}`, runPageSchema);
    },
    /** The run's current snapshot with one page of records after `after`. */
    runSnapshot(
      runId: string,
      after = 0,
      limit = LIVE_RECORD_PAGE_LIMIT,
    ): Promise<LiveApiResult<LiveRunSnapshot>> {
      const query = new URLSearchParams({
        after: String(after),
        limit: String(limit),
      });
      return get(
        `/v1/live/runs/${encodeURIComponent(runId)}?${query}`,
        runSnapshotSchema,
      );
    },
  };
}
export type LiveApi = ReturnType<typeof createLiveApi>;
