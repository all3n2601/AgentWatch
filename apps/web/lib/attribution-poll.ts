import type { LiveApiFailure } from "./live-api";
import type {
  AttributionPageData,
  ModelStatus,
  ReceivedItem,
} from "./attribution-schema";
import {
  connectionAfterFailure,
  LIVE_POLL_INTERVAL_MS,
  type FeedConnection,
} from "./live-poll";

// Polling state for one run's attributions while its Attribution tab is open, free of timers
// and fetch so it can be tested. Incremental reads continue after the largest `sequence`
// received and drain `next_cursor` (docs/live-ingestion.md, Attribution). `evidence_changed`
// can turn true after a row was served, and an incremental read never returns that row again,
// so the whole run is re-read from `after=0` on a slower schedule and replaces the rows.

/** Re-read from `after=0` this often, to show `evidence_changed` set after a row was served. */
export const ATTRIBUTION_REREAD_INTERVAL_MS = 30_000;

export type AttributionFeed = {
  runId: string;
  /** The serving model's status from the latest page; null before the first. */
  model: ModelStatus | null;
  /** Every row received, by ascending `sequence`; unique by `sequence`. */
  items: ReceivedItem[];
  /** Largest `sequence` received; incremental reads continue after it. */
  cursor: number;
  connection: FeedConnection;
  /** More pages were waiting when the last response arrived. */
  draining: boolean;
  /** A re-read from `after=0` in progress: the rows read so far and its own cursor. */
  reread: { cursor: number; items: ReceivedItem[] } | null;
  /** Browser monotonic time the last full read started; the first read is a full read. */
  fullReadStartedMs: number | null;
  /** A full read has completed at least once, so every row up to it has its current flag. */
  loaded: boolean;
};

export const initialAttributionFeed = (runId: string): AttributionFeed => ({
  runId,
  model: null,
  items: [],
  cursor: 0,
  connection: { status: "connecting" },
  draining: false,
  reread: null,
  fullReadStartedMs: null,
  loaded: false,
});

/** The `after` cursor of the next read. */
export const nextReadAfter = (feed: AttributionFeed) =>
  feed.reread ? feed.reread.cursor : feed.cursor;

/**
 * Starts a re-read from `after=0` when one is due and none is in progress. The first read of
 * a feed is itself a full read. `nowMs` is the browser's performance.now().
 */
export function beginDueReread(
  feed: AttributionFeed,
  nowMs: number,
): AttributionFeed {
  if (feed.reread || feed.draining) return feed;
  if (feed.fullReadStartedMs === null)
    return {
      ...feed,
      reread: { cursor: 0, items: [] },
      fullReadStartedMs: nowMs,
    };
  if (nowMs - feed.fullReadStartedMs < ATTRIBUTION_REREAD_INTERVAL_MS)
    return feed;
  return {
    ...feed,
    reread: { cursor: 0, items: [] },
    fullReadStartedMs: nowMs,
  };
}

/** Appends rows after `existing`'s last sequence; a repeated sequence is ignored. */
function appendAfter(existing: ReceivedItem[], page: ReceivedItem[]) {
  const last = existing.at(-1)?.sequence ?? 0;
  const added = page
    .filter((item) => item.sequence > last)
    .sort((a, b) => a.sequence - b.sequence);
  const unique = added.filter(
    (item, index) => index === 0 || item.sequence !== added[index - 1].sequence,
  );
  return unique.length ? [...existing, ...unique] : existing;
}

const lastSequence = (items: ReceivedItem[], fallback: number) =>
  items.at(-1)?.sequence ?? fallback;

/**
 * Applies one valid page. A page for another run is a contract failure: it is never shown
 * under the selected run.
 */
export function applyAttributionPage(
  feed: AttributionFeed,
  page: AttributionPageData,
  random: number,
): AttributionFeed {
  if (page.run_id !== feed.runId)
    return applyAttributionFailure(
      feed,
      {
        kind: "invalid_response",
        message: "The live API answered with another run's attributions.",
      },
      random,
    );
  // A continuation cursor on an empty page would be followed at once, forever; treat it as a
  // contract failure so it backs off like any other.
  if (page.attributions.length === 0 && page.next_cursor !== null)
    return applyAttributionFailure(
      feed,
      {
        kind: "invalid_response",
        message:
          "The live API returned an empty attributions page with a continuation cursor.",
      },
      random,
    );
  const draining = page.next_cursor !== null;
  if (feed.reread) {
    const items = appendAfter(feed.reread.items, page.attributions);
    const cursor = lastSequence(items, 0);
    if (draining)
      return {
        ...feed,
        model: page.model,
        reread: { cursor, items },
        connection: { status: "connected" },
        draining,
      };
    // The full read replaces every row, so each shows its current `evidence_changed`.
    return {
      ...feed,
      model: page.model,
      items,
      cursor: Math.max(feed.cursor, cursor),
      reread: null,
      connection: { status: "connected" },
      draining: false,
      loaded: true,
    };
  }
  const items = appendAfter(feed.items, page.attributions);
  return {
    ...feed,
    model: page.model,
    items,
    cursor: lastSequence(items, feed.cursor),
    connection: { status: "connected" },
    draining,
  };
}

/** Records a failed read; rows already received stay, and a re-read resumes where it was. */
export function applyAttributionFailure(
  feed: AttributionFeed,
  failure: LiveApiFailure,
  random: number,
): AttributionFeed {
  const connection = connectionAfterFailure(feed.connection, failure, random);
  return {
    ...feed,
    connection,
    draining: connection.status === "stopped" ? false : feed.draining,
  };
}

/**
 * Delay before the next read, or null to stop. While the run is open, rows are read every
 * live poll interval; once it is closed only the slower re-read continues, which also picks up
 * rows written by the pass that runs at the close.
 */
export function nextAttributionDelayMs(
  feed: AttributionFeed,
  runOpen: boolean,
  nowMs: number,
): number | null {
  switch (feed.connection.status) {
    case "stopped":
      return null;
    case "retrying":
      return feed.connection.retryInMs;
    case "connecting":
      return 0;
    case "connected": {
      if (feed.draining) return 0;
      // Deliberate trade-off: re-reads continue after the run has closed and settled. The
      // worker sets `evidence_changed` in the audit pass after the close, and the API does not
      // say when that audit has run (`evidence_audited` is not exposed), so the client cannot
      // know when a re-read can no longer change anything. The re-read is slow and runs only
      // while the Attribution tab is open.
      const sinceFull =
        feed.fullReadStartedMs === null
          ? Infinity
          : nowMs - feed.fullReadStartedMs;
      const untilReread = Math.max(
        0,
        ATTRIBUTION_REREAD_INTERVAL_MS - sinceFull,
      );
      return runOpen
        ? Math.min(LIVE_POLL_INTERVAL_MS, untilReread)
        : untilReread;
    }
  }
}
