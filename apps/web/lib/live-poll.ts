import type { LiveApiFailure } from "./live-api";
import type {
  LiveRunPage,
  LiveRunSnapshot,
  LiveSavedRecord,
} from "./live-api-schema";
import type { FeedReceipt } from "./live-status";
import { terminalStates } from "./live-view";

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
  /** A viewer's explicit check of a closed run for late evidence; null before any. */
  lateCheck: LateCheck | null;
};

/**
 * `recordsBefore` is the record count when the check started, so the result counts only
 * records that arrived because of it. A failed check keeps it: a later automatic retry that
 * succeeds settles the same check.
 */
export type LateCheck =
  | { status: "checking"; recordsBefore: number }
  | { status: "checked"; checkedAtWallMs: number; newRecords: number }
  | {
      status: "failed";
      checkedAtWallMs: number;
      message: string;
      recordsBefore: number;
    };

export const initialRunFeed = (runId: string): RunFeed => ({
  runId,
  snapshot: null,
  records: [],
  cursor: 0,
  receipt: null,
  connection: { status: "connecting" },
  draining: false,
  lateCheck: null,
});

/**
 * Exponential backoff with jitter, kept within [RETRY_BASE_DELAY_MS, RETRY_MAX_DELAY_MS].
 * `random` is in [0, 1).
 */
export function retryDelayMs(attempt: number, random: number) {
  const exponential = RETRY_BASE_DELAY_MS * 2 ** Math.max(0, attempt - 1);
  const jitter = 1 + RETRY_JITTER_FRACTION * (2 * random - 1);
  const jittered = Math.min(exponential, RETRY_MAX_DELAY_MS) * jitter;
  return Math.round(
    Math.min(RETRY_MAX_DELAY_MS, Math.max(RETRY_BASE_DELAY_MS, jittered)),
  );
}

/**
 * Marks cached step records applied once the snapshot's step state has reached them. `after`
 * filters immutable arrivals, so a pending record that the service applies later is never
 * delivered again; only the snapshot shows the change. The service applies each step's
 * revisions as a contiguous prefix, so every revision up to the step's current one is applied.
 */
function reconcileDispositions(
  records: LiveSavedRecord[],
  snapshot: LiveRunSnapshot,
): LiveSavedRecord[] {
  if (records.every((record) => record.disposition === "applied"))
    return records;
  const appliedThrough = new Map(
    snapshot.steps.map(({ event }) => [event.step_id, event.sequence]),
  );
  let changed = false;
  const reconciled = records.map((record) => {
    const { payload } = record;
    if (
      record.disposition === "applied" ||
      payload.record_type !== "step_event"
    )
      return record;
    const current = appliedThrough.get(payload.step_id);
    if (current === undefined || payload.sequence > current) return record;
    changed = true;
    return { ...record, disposition: "applied" as const };
  });
  return changed ? reconciled : records;
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
    records: reconcileDispositions(
      added.length ? [...feed.records, ...added] : feed.records,
      snapshot,
    ),
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

/** Starts a late-evidence check; a check already in progress keeps its starting count. */
export const beginLateCheck = (feed: RunFeed): RunFeed =>
  feed.lateCheck?.status === "checking"
    ? feed
    : {
        ...feed,
        lateCheck: { status: "checking", recordsBefore: feed.records.length },
      };

/**
 * Settles a pending late-evidence check after a read, once its pages are drained.
 * `nowWallMs` is the browser's wall clock, used only to tell the viewer when it was checked.
 */
export function settleLateCheck(
  feed: RunFeed,
  failure: LiveApiFailure | null,
  nowWallMs: number,
): RunFeed {
  const check = feed.lateCheck;
  if (check === null || check.status === "checked") return feed;
  if (failure)
    return check.status === "failed"
      ? feed
      : {
          ...feed,
          lateCheck: {
            status: "failed",
            checkedAtWallMs: nowWallMs,
            message: failure.message,
            recordsBefore: check.recordsBefore,
          },
        };
  if (feed.draining) return feed;
  return {
    ...feed,
    lateCheck: {
      status: "checked",
      checkedAtWallMs: nowWallMs,
      newRecords: feed.records.length - check.recordsBefore,
    },
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
