import type { LiveApiFailure } from "./live-api";
import type {
  LiveRunPage,
  LiveRunSnapshot,
  LiveSavedRecord,
} from "./live-api-schema";
import type { FeedReceipt } from "./live-status";
import { terminalStates } from "./step-events";

// Polling state for one selected live run, kept free of timers and fetch so it can be tested.
// B01 offers polling only (docs/live-ingestion.md): read the snapshot after the largest record
// sequence received, replace steps and gaps from every response, and drain record pages.

/** Poll interval while the selected run is queued or running. */
export const LIVE_POLL_INTERVAL_MS = 2_000;
/** Poll interval of the run list. */
export const LIVE_RUN_LIST_INTERVAL_MS = 10_000;
export const RETRY_BASE_DELAY_MS = 1_000;
export const RETRY_MAX_DELAY_MS = 30_000;
/** Retries are spread by up to ±20% so reconnecting viewers do not poll in step. */
export const RETRY_JITTER_FRACTION = 0.2;

export type FeedConnection =
  | { status: "connecting" }
  | { status: "connected" }
  | {
      status: "retrying";
      /** Consecutive failed reads since the last success. */
      attempt: number;
      retryInMs: number;
      message: string;
    }
  /** Reading this run again cannot succeed; the last valid data stays visible. */
  | {
      status: "stopped";
      kind: "not_found" | "invalid_request";
      message: string;
    };

export type RunFeed = {
  runId: string;
  /**
   * The latest valid snapshot, with its record page moved into `records`; null before the
   * first one. Its steps and gaps are the service's current state.
   */
  snapshot: LiveRunSnapshot | null;
  /** Every record received for the run, in sequence order. */
  records: LiveSavedRecord[];
  /** Largest record sequence received; the next read continues after it. */
  cursor: number;
  /** The browser's monotonic time when new records last arrived; never a server clock. */
  receipt: FeedReceipt | null;
  connection: FeedConnection;
  /** More record pages were waiting when the last response arrived. */
  draining: boolean;
};

export const initialRunFeed = (runId: string): RunFeed => ({
  runId,
  snapshot: null,
  records: [],
  cursor: 0,
  receipt: null,
  connection: { status: "connecting" },
  draining: false,
});

/** Exponential backoff with jitter, capped at RETRY_MAX_DELAY_MS. `random` is in [0, 1). */
export function retryDelayMs(attempt: number, random: number) {
  const exponential = RETRY_BASE_DELAY_MS * 2 ** Math.max(0, attempt - 1);
  const jitter = 1 + RETRY_JITTER_FRACTION * (2 * random - 1);
  return Math.round(
    Math.min(
      RETRY_MAX_DELAY_MS,
      Math.min(exponential, RETRY_MAX_DELAY_MS) * jitter,
    ),
  );
}

/** Applies one valid response. `nowMonotonicMs` is the browser's performance.now(). */
export function applySnapshot(
  feed: RunFeed,
  snapshot: LiveRunSnapshot,
  nowMonotonicMs: number,
): RunFeed {
  if (snapshot.run_id !== feed.runId)
    return applyFailure(
      feed,
      {
        kind: "invalid_response",
        message: "The live API answered with a different run.",
      },
      0.5,
    );
  const page = snapshot.records;
  // Records are immutable arrivals; a retried page can repeat ones already received.
  const added = page
    .filter((record) => record.sequence > feed.cursor)
    .sort((a, b) => a.sequence - b.sequence);
  const cursor = added.length ? added[added.length - 1].sequence : feed.cursor;
  return {
    ...feed,
    snapshot: { ...snapshot, records: [] },
    records: added.length ? [...feed.records, ...added] : feed.records,
    cursor,
    receipt: added.length
      ? { ageAtAnchorMs: 0, anchorMonotonicMs: nowMonotonicMs }
      : feed.receipt,
    connection: { status: "connected" },
    draining: snapshot.next_cursor !== null,
  };
}

/** The connection after a failed read; retryable failures back off with jitter. */
export function connectionAfterFailure(
  previous: FeedConnection,
  failure: LiveApiFailure,
  random: number,
): FeedConnection {
  if (failure.kind === "not_found" || failure.kind === "invalid_request")
    return { status: "stopped", kind: failure.kind, message: failure.message };
  const attempt = previous.status === "retrying" ? previous.attempt + 1 : 1;
  return {
    status: "retrying",
    attempt,
    retryInMs: retryDelayMs(attempt, random),
    message: failure.message,
  };
}

/** Records a failed read. Everything received before it stays as it was. */
export function applyFailure(
  feed: RunFeed,
  failure: LiveApiFailure,
  random: number,
): RunFeed {
  const connection = connectionAfterFailure(feed.connection, failure, random);
  return {
    ...feed,
    connection,
    draining: connection.status === "stopped" ? false : feed.draining,
  };
}

export const runClosed = (feed: RunFeed) =>
  feed.snapshot !== null && terminalStates.includes(feed.snapshot.state);

/**
 * Delay before the next read, or null to stop. A closed run stops once its records are
 * drained; late evidence can still arrive, so viewers can check again explicitly.
 */
export function nextPollDelayMs(feed: RunFeed): number | null {
  switch (feed.connection.status) {
    case "stopped":
      return null;
    case "retrying":
      return feed.connection.retryInMs;
    case "connecting":
      return 0;
    case "connected":
      if (feed.draining) return 0;
      return runClosed(feed) ? null : LIVE_POLL_INTERVAL_MS;
  }
}

/** The newest page of live runs; independent of the selected run's feed. */
export type RunListFeed = {
  page: LiveRunPage | null;
  connection: FeedConnection;
};

export const initialRunList: RunListFeed = {
  page: null,
  connection: { status: "connecting" },
};

export function nextListPollDelayMs(list: RunListFeed): number | null {
  switch (list.connection.status) {
    case "stopped":
      return null;
    case "retrying":
      return list.connection.retryInMs;
    case "connecting":
      return 0;
    case "connected":
      return LIVE_RUN_LIST_INTERVAL_MS;
  }
}
