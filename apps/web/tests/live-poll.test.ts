import assert from "node:assert/strict";
import test from "node:test";
import { runSnapshotSchema } from "../lib/live-api-schema";
import {
  applyFailure,
  applySnapshot,
  beginLateCheck,
  initialRunFeed,
  initialRunList,
  LIVE_POLL_INTERVAL_MS,
  LIVE_RUN_LIST_INTERVAL_MS,
  nextListPollDelayMs,
  nextPollDelayMs,
  retryDelayMs,
  RETRY_MAX_DELAY_MS,
  settleLateCheck,
  type RunFeed,
} from "../lib/live-poll";
import { feedDisplay, LIVE_FEED_STALE_AFTER_MS } from "../lib/live-status";
import {
  OTHER_RUN_ID,
  RUN_ID,
  runSnapshot,
  savedRecord,
  stepEvent,
  stepView,
} from "./live-fixtures";

const unavailable = {
  kind: "unavailable" as const,
  message: "The live API responded with HTTP 503.",
};

function page(
  sequences: number[],
  overrides: Record<string, unknown> = {},
  runId = RUN_ID,
) {
  const event = stepEvent("tool-1", 0, "queued", { runId });
  return runSnapshotSchema.parse(
    runSnapshot(
      {
        steps: [stepView(event)],
        records: sequences.map((sequence) =>
          savedRecord({ ...event, event_id: `e-${sequence}` }, sequence),
        ),
        ...overrides,
      },
      runId,
    ),
  );
}

const sequencesOf = (feed: RunFeed) => feed.records.map((r) => r.sequence);

test("a failed read keeps the last data and retries until a read recovers", () => {
  let feed = applySnapshot(initialRunFeed(RUN_ID), page([1, 2]), 1_000);
  assert.equal(nextPollDelayMs(feed), LIVE_POLL_INTERVAL_MS);

  feed = applyFailure(feed, unavailable, 0.5);
  assert.deepEqual(feed.connection, {
    status: "retrying",
    attempt: 1,
    retryInMs: 1_000,
    message: unavailable.message,
  });
  assert.equal(nextPollDelayMs(feed), 1_000);
  assert.deepEqual(sequencesOf(feed), [1, 2], "received records are kept");
  assert.equal(feed.snapshot?.state, "running", "the last snapshot is kept");
  assert.equal(feed.runId, RUN_ID, "the selection is kept");

  feed = applyFailure(feed, unavailable, 0.5);
  assert.equal(feed.connection.status, "retrying");
  assert.equal(nextPollDelayMs(feed), 2_000);

  // The retried page repeats record 2; the cursor continues after it.
  feed = applySnapshot(feed, page([2, 3]), 9_000);
  assert.deepEqual(feed.connection, { status: "connected" });
  assert.deepEqual(sequencesOf(feed), [1, 2, 3]);
  assert.equal(feed.cursor, 3);
  assert.equal(nextPollDelayMs(feed), LIVE_POLL_INTERVAL_MS);

  // A later failure starts backing off from the first attempt again.
  feed = applyFailure(feed, unavailable, 0.5);
  assert.equal(
    feed.connection.status === "retrying" && feed.connection.attempt,
    1,
  );
});

test("retry delays grow exponentially with bounded jitter and a 30 s cap", () => {
  // Jitter never takes the first retry below the 1 s floor.
  assert.equal(retryDelayMs(1, 0), 1_000);
  assert.equal(retryDelayMs(1, 0.5), 1_000);
  assert.equal(retryDelayMs(2, 0), 1_600);
  assert.ok(retryDelayMs(1, 0.9999) <= 1_200);
  assert.equal(retryDelayMs(3, 0.5), 4_000);
  for (const attempt of [6, 10, 100, 2_000])
    for (const random of [0, 0.5, 0.9999]) {
      const delay = retryDelayMs(attempt, random);
      assert.ok(delay <= RETRY_MAX_DELAY_MS, `${attempt}: ${delay}`);
      assert.ok(delay >= RETRY_MAX_DELAY_MS * 0.8, `${attempt}: ${delay}`);
    }
});

test("a missing run or rejected request stops polling without discarding data", () => {
  const received = applySnapshot(initialRunFeed(RUN_ID), page([1]), 1_000);
  for (const kind of ["not_found", "invalid_request"] as const) {
    const feed = applyFailure(received, { kind, message: "x" }, 0.5);
    assert.equal(feed.connection.status, "stopped");
    assert.equal(nextPollDelayMs(feed), null);
    assert.deepEqual(sequencesOf(feed), [1]);
  }
});

test("record pages are drained immediately and a closed run stops once drained", () => {
  let feed = applySnapshot(
    initialRunFeed(RUN_ID),
    page([1, 2], { next_cursor: 2 }),
    1_000,
  );
  assert.equal(feed.draining, true);
  assert.equal(nextPollDelayMs(feed), 0);

  feed = applySnapshot(feed, page([3], { state: "completed" }), 1_100);
  assert.equal(feed.draining, false);
  assert.equal(nextPollDelayMs(feed), null);

  const closedButDraining = applySnapshot(
    initialRunFeed(RUN_ID),
    page([1], { state: "failed", reason: "boom", next_cursor: 1 }),
    1_000,
  );
  assert.equal(nextPollDelayMs(closedButDraining), 0);
});

test("a response for another run is treated as a failed read, not applied", () => {
  const received = applySnapshot(initialRunFeed(RUN_ID), page([1]), 1_000);
  const feed = applySnapshot(received, page([7], {}, OTHER_RUN_ID), 2_000);
  assert.equal(feed.connection.status, "retrying");
  assert.equal(feed.snapshot?.run_id, RUN_ID);
  assert.deepEqual(sequencesOf(feed), [1]);
});

test("receipt age is anchored to the browser clock, whatever the server clock says", () => {
  const browserWallMs = Date.UTC(2026, 9, 24, 12, 0, 0);
  for (const skewMs of [-60_000, 0, 60_000]) {
    const skewed = page([1]);
    skewed.records[0].received_at = new Date(
      browserWallMs + skewMs,
    ).toISOString();
    const feed = applySnapshot(initialRunFeed(RUN_ID), skewed, 5_000);
    assert.deepEqual(feed.receipt, {
      ageAtAnchorMs: 0,
      anchorMonotonicMs: 5_000,
    });

    const display = feedDisplay({
      receivedAgoMs: feed.receipt!.ageAtAnchorMs,
      outcome: "active",
      clock: {
        anchorMonotonicMs: feed.receipt!.anchorMonotonicMs,
        nowMonotonicMs: 8_000,
        nowWallMs: browserWallMs + 3_000,
      },
    });
    assert.equal(display.ageMs, 3_000, `skew ${skewMs}`);
    assert.equal(display.freshness, "fresh", `skew ${skewMs}`);
    assert.equal(display.receivedAtWallMs, browserWallMs, `skew ${skewMs}`);
  }
});

test("a poll without new records keeps the receipt, so the feed can go stale", () => {
  let feed = applySnapshot(initialRunFeed(RUN_ID), page([1]), 5_000);
  feed = applySnapshot(feed, page([1]), 5_000 + LIVE_FEED_STALE_AFTER_MS);
  assert.deepEqual(feed.receipt, {
    ageAtAnchorMs: 0,
    anchorMonotonicMs: 5_000,
  });
  const display = feedDisplay({
    receivedAgoMs: 0,
    outcome: "active",
    clock: {
      anchorMonotonicMs: feed.receipt!.anchorMonotonicMs,
      nowMonotonicMs: 5_001 + LIVE_FEED_STALE_AFTER_MS,
      nowWallMs: 0,
    },
  });
  assert.equal(display.freshness, "stale");
});

test("the run list polls on its own interval and backs off after failures", () => {
  assert.equal(nextListPollDelayMs(initialRunList), 0);
  assert.equal(
    nextListPollDelayMs({ page: null, connection: { status: "connected" } }),
    LIVE_RUN_LIST_INTERVAL_MS,
  );
  assert.equal(
    nextListPollDelayMs({
      page: null,
      connection: {
        status: "retrying",
        attempt: 2,
        retryInMs: 2_000,
        message: "x",
      },
    }),
    2_000,
  );
});

test("a late-evidence check reports how many records it found, once drained", () => {
  const CHECKED_AT = Date.UTC(2026, 9, 24, 18, 0, 0);
  const closed = applySnapshot(
    initialRunFeed(RUN_ID),
    page([1, 2], { state: "completed" }),
    1_000,
  );
  assert.equal(
    settleLateCheck(closed, null, CHECKED_AT),
    closed,
    "no check, no change",
  );

  // No new records.
  let feed = beginLateCheck(closed);
  assert.deepEqual(feed.lateCheck, { status: "checking", recordsBefore: 2 });
  feed = settleLateCheck(
    applySnapshot(feed, page([1, 2], { state: "completed" }), 2_000),
    null,
    CHECKED_AT,
  );
  assert.deepEqual(feed.lateCheck, {
    status: "checked",
    checkedAtWallMs: CHECKED_AT,
    newRecords: 0,
  });

  // New records over two pages: still checking until the last page arrives.
  feed = beginLateCheck(feed);
  feed = settleLateCheck(
    applySnapshot(
      feed,
      page([3], { state: "completed", next_cursor: 3 }),
      3_000,
    ),
    null,
    CHECKED_AT,
  );
  assert.equal(feed.lateCheck?.status, "checking");
  feed = settleLateCheck(
    applySnapshot(feed, page([4], { state: "completed" }), 3_100),
    null,
    CHECKED_AT,
  );
  assert.deepEqual(feed.lateCheck, {
    status: "checked",
    checkedAtWallMs: CHECKED_AT,
    newRecords: 2,
  });
});

test("a failed late-evidence check reports the error; a later successful retry settles it", () => {
  const closed = applySnapshot(
    initialRunFeed(RUN_ID),
    page([1], { state: "completed" }),
    1_000,
  );
  let feed = beginLateCheck(closed);
  feed = settleLateCheck(applyFailure(feed, unavailable, 0.5), unavailable, 10);
  assert.deepEqual(feed.lateCheck, {
    status: "failed",
    checkedAtWallMs: 10,
    message: unavailable.message,
    recordsBefore: 1,
  });
  // The automatic retry succeeds with one new record.
  feed = settleLateCheck(
    applySnapshot(feed, page([1, 2], { state: "completed" }), 2_000),
    null,
    20,
  );
  assert.deepEqual(feed.lateCheck, {
    status: "checked",
    checkedAtWallMs: 20,
    newRecords: 1,
  });
});

test("starting a check while one is in progress keeps its starting count", () => {
  const closed = applySnapshot(
    initialRunFeed(RUN_ID),
    page([1], { state: "completed" }),
    1_000,
  );
  const checking = beginLateCheck(closed);
  assert.equal(beginLateCheck(checking), checking);
});

test("a pending record becomes applied when the snapshot applies it without redelivery", () => {
  const queued = stepEvent("tool-1", 0, "queued");
  const running = stepEvent("tool-1", 1, "running", { started: 1 });
  const completed = stepEvent("tool-1", 2, "completed", {
    started: 1,
    finished: 2,
  });
  const snapshot = (records: unknown[], current: typeof queued) =>
    runSnapshotSchema.parse(
      runSnapshot({ steps: [stepView(current)], records }),
    );
  // The completion arrives before the running revision, so it waits as pending.
  let feed = applySnapshot(
    initialRunFeed(RUN_ID),
    snapshot(
      [savedRecord(queued, 1), savedRecord(completed, 2, "pending")],
      queued,
    ),
    1_000,
  );
  assert.deepEqual(
    feed.records.map((r) => r.disposition),
    ["applied", "pending"],
  );
  // The running revision fills the gap; the completion is not delivered again.
  feed = applySnapshot(
    feed,
    snapshot([savedRecord(running, 3)], completed),
    2_000,
  );
  assert.deepEqual(
    feed.records.map((r) => [r.sequence, r.disposition]),
    [
      [1, "applied"],
      [2, "applied"],
      [3, "applied"],
    ],
  );
});

test("records beyond the snapshot's applied step state stay pending", () => {
  const queued = stepEvent("tool-1", 0, "queued");
  const completed = stepEvent("tool-1", 2, "completed", { finished: 2 });
  const feed = applySnapshot(
    initialRunFeed(RUN_ID),
    runSnapshotSchema.parse(
      runSnapshot({
        steps: [stepView(queued)],
        records: [savedRecord(queued, 1), savedRecord(completed, 2, "pending")],
      }),
    ),
    1_000,
  );
  assert.equal(feed.records[1].disposition, "pending");
});
