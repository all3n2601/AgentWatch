import assert from "node:assert/strict";
import test from "node:test";
import { attributionPageSchema } from "../lib/attribution-schema";
import {
  applyAttributionFailure,
  applyAttributionPage,
  ATTRIBUTION_REREAD_INTERVAL_MS,
  beginDueReread,
  initialAttributionFeed,
  nextAttributionDelayMs,
  nextReadAfter,
  type AttributionFeed,
} from "../lib/attribution-poll";
import { LIVE_POLL_INTERVAL_MS } from "../lib/live-poll";
import {
  attributionItem,
  attributionPage,
  attributionRecord,
} from "./attribution-fixtures";
import { OTHER_RUN_ID, RUN_ID } from "./live-fixtures";

const item = (sequence: number, evidenceChanged = false) =>
  attributionItem(sequence, attributionRecord(RUN_ID, `step-${sequence}`), {
    evidenceChanged,
  });
const page = (
  sequences: number[],
  nextCursor: number | null = null,
  changed: number[] = [],
  runId = RUN_ID,
) =>
  attributionPageSchema.parse(
    attributionPage(
      runId,
      sequences.map((sequence) => item(sequence, changed.includes(sequence))),
      { nextCursor },
    ),
  );

/** Runs one read as the hook does: start a due re-read, then apply the page. */
function read(
  feed: AttributionFeed,
  nowMs: number,
  response: ReturnType<typeof page>,
) {
  const started = beginDueReread(feed, nowMs);
  return {
    after: nextReadAfter(started),
    feed: applyAttributionPage(started, response, 0.5),
  };
}

test("the first read is a full read from 0 that drains every page", () => {
  let feed = initialAttributionFeed(RUN_ID);
  const first = read(feed, 0, page([1, 2], 2));
  assert.equal(first.after, 0);
  feed = first.feed;
  assert.equal(feed.loaded, false);
  assert.equal(nextAttributionDelayMs(feed, true, 0), 0);
  const second = read(feed, 10, page([3]));
  assert.equal(second.after, 2);
  feed = second.feed;
  assert.equal(feed.loaded, true);
  assert.deepEqual(
    feed.items.map((row) => row.sequence),
    [1, 2, 3],
  );
  assert.equal(feed.cursor, 3);
});

test("incremental reads continue after the largest sequence and ignore repeats", () => {
  let feed = read(initialAttributionFeed(RUN_ID), 0, page([1])).feed;
  const next = read(feed, LIVE_POLL_INTERVAL_MS, page([1, 2]));
  assert.equal(next.after, 1);
  feed = next.feed;
  assert.deepEqual(
    feed.items.map((row) => row.sequence),
    [1, 2],
  );
});

test("a re-read from 0 replaces rows, so evidence_changed set after serving is shown", () => {
  let feed = read(initialAttributionFeed(RUN_ID), 0, page([1, 2])).feed;
  feed = read(feed, LIVE_POLL_INTERVAL_MS, page([])).feed;
  const due = read(
    feed,
    ATTRIBUTION_REREAD_INTERVAL_MS,
    page([1, 2, 3], null, [1]),
  );
  assert.equal(due.after, 0);
  feed = due.feed;
  assert.deepEqual(
    feed.items.map((row) => [
      row.sequence,
      (row as { evidence_changed?: boolean }).evidence_changed,
    ]),
    [
      [1, true],
      [2, false],
      [3, false],
    ],
  );
  assert.equal(feed.cursor, 3);
});

test("rows read during a re-read appear only once the re-read is complete", () => {
  let feed = read(initialAttributionFeed(RUN_ID), 0, page([1, 2])).feed;
  feed = read(feed, ATTRIBUTION_REREAD_INTERVAL_MS, page([1], 1, [1])).feed;
  assert.ok(feed.reread);
  assert.equal(
    (feed.items[0] as { evidence_changed?: boolean }).evidence_changed,
    false,
  );
  const rest = read(feed, ATTRIBUTION_REREAD_INTERVAL_MS + 1, page([2]));
  assert.equal(rest.after, 1);
  assert.equal(rest.feed.reread, null);
  assert.equal(
    (rest.feed.items[0] as { evidence_changed?: boolean }).evidence_changed,
    true,
  );
});

test("an open run is polled every live interval; a closed run only on the re-read schedule", () => {
  const feed = read(initialAttributionFeed(RUN_ID), 0, page([1])).feed;
  assert.equal(nextAttributionDelayMs(feed, true, 0), LIVE_POLL_INTERVAL_MS);
  assert.equal(
    nextAttributionDelayMs(feed, false, 0),
    ATTRIBUTION_REREAD_INTERVAL_MS,
  );
  assert.equal(
    nextAttributionDelayMs(feed, false, ATTRIBUTION_REREAD_INTERVAL_MS - 5),
    5,
  );
  assert.equal(
    nextAttributionDelayMs(feed, true, ATTRIBUTION_REREAD_INTERVAL_MS - 5),
    5,
  );
});

test("another run's page is refused and nothing of it is kept", () => {
  const feed = read(
    initialAttributionFeed(RUN_ID),
    0,
    page([1], null, [], OTHER_RUN_ID),
  ).feed;
  assert.equal(feed.items.length, 0);
  assert.equal(feed.connection.status, "retrying");
});

test("a failed read keeps rows; not found stops polling; a re-read resumes at its cursor", () => {
  let feed = read(initialAttributionFeed(RUN_ID), 0, page([1, 2])).feed;
  feed = read(feed, ATTRIBUTION_REREAD_INTERVAL_MS, page([1], 1)).feed;
  feed = applyAttributionFailure(
    feed,
    { kind: "unavailable", message: "down" },
    0.5,
  );
  assert.equal(feed.items.length, 2);
  assert.equal(nextReadAfter(feed), 1);
  assert.ok((nextAttributionDelayMs(feed, true, 0) ?? 0) > 0);

  const gone = applyAttributionFailure(
    feed,
    { kind: "not_found", message: "gone" },
    0.5,
  );
  assert.equal(nextAttributionDelayMs(gone, true, 0), null);
});
