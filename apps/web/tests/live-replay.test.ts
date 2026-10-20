import assert from "node:assert/strict";
import test from "node:test";
import { replayScenarios } from "../fixtures/step-event-replays";
import { liveReplayData } from "../lib/live-replay";
import {
  feedClockTicks,
  feedDisplay,
  LIVE_FEED_STALE_AFTER_MS,
  runOutcome,
} from "../lib/live-status";

test("replay is never offered when it is disabled", () => {
  assert.deepEqual(liveReplayData("failed-run", { replayEnabled: false }), {
    replay: null,
    requestedReplay: null,
    scenarios: [],
  });
});

test("an enabled replay reads the scenario through the adapter", () => {
  const data = liveReplayData("stale-feed", { replayEnabled: true });
  assert.equal(data.replay?.id, "stale-feed");
  assert.equal(data.replay?.receivedAgoMs, 120_000);
  assert.equal(data.replay?.view.steps[0].state, "running");
  assert.equal(data.scenarios.length, replayScenarios.length);
});

test("an unknown or absent scenario keeps the request without a replay", () => {
  const unknown = liveReplayData("nope", { replayEnabled: true });
  assert.equal(unknown.replay, null);
  assert.equal(unknown.requestedReplay, "nope");
  const absent = liveReplayData(undefined, { replayEnabled: true });
  assert.equal(absent.requestedReplay, null);
  assert.equal(absent.scenarios.length, replayScenarios.length);
});

test("server page data carries no wall-clock receipt or render time", () => {
  const data = liveReplayData("active-steps", { replayEnabled: true });
  assert.deepEqual(Object.keys(data).sort(), [
    "replay",
    "requestedReplay",
    "scenarios",
  ]);
  assert.ok(data.replay && !("receivedAtMs" in data.replay));
});

// The server's wall clock reads SERVER_WALL_MS when it renders the page. The
// browser's wall clock is offset from it; its monotonic clock reads
// HYDRATION_MONOTONIC_MS when the replay view mounts.
const SERVER_WALL_MS = Date.parse("2026-10-14T12:10:00Z");
const HYDRATION_MONOTONIC_MS = 4_321;

function browserView(scenarioId: string, offsetMs: number, elapsedMs: number) {
  const replay = liveReplayData(scenarioId, { replayEnabled: true }).replay!;
  const browserWallAtHydrationMs = SERVER_WALL_MS + offsetMs;
  const display = feedDisplay({
    receivedAgoMs: replay.receivedAgoMs,
    outcome: runOutcome(replay.run),
    clock: {
      anchorMonotonicMs: HYDRATION_MONOTONIC_MS,
      nowMonotonicMs: HYDRATION_MONOTONIC_MS + elapsedMs,
      nowWallMs: browserWallAtHydrationMs + elapsedMs,
    },
  });
  return {
    age: display.ageMs,
    freshness: display.freshness,
    receivedAtWallMs: display.receivedAtWallMs,
    browserWallAtHydrationMs,
  };
}

for (const [label, offsetMs] of [
  ["60 s ahead of", 60_000],
  ["60 s behind", -60_000],
] as const) {
  test(`freshness is unaffected when the browser clock is ${label} the server`, () => {
    const justLoaded = browserView("active-steps", offsetMs, 10_000);
    assert.equal(justLoaded.age, 10_000);
    assert.equal(justLoaded.freshness, "fresh");
    // "Last record received" is the browser's own load time, never server time.
    assert.equal(
      justLoaded.receivedAtWallMs,
      justLoaded.browserWallAtHydrationMs,
    );

    const quiet = browserView(
      "active-steps",
      offsetMs,
      LIVE_FEED_STALE_AFTER_MS + 1,
    );
    assert.equal(quiet.freshness, "stale");

    const stale = browserView("stale-feed", offsetMs, 0);
    assert.equal(stale.freshness, "stale");
    assert.equal(
      stale.receivedAtWallMs,
      stale.browserWallAtHydrationMs - 120_000,
    );
  });
}

test("a terminal replay's receipt age keeps advancing while freshness stays closed", () => {
  const replay = liveReplayData("completed-run", {
    replayEnabled: true,
  }).replay!;
  const outcome = runOutcome(replay.run);
  assert.equal(feedClockTicks(replay.receivedAgoMs), true);
  const at = (elapsedMs: number) =>
    feedDisplay({
      receivedAgoMs: replay.receivedAgoMs,
      outcome,
      clock: {
        anchorMonotonicMs: HYDRATION_MONOTONIC_MS,
        nowMonotonicMs: HYDRATION_MONOTONIC_MS + elapsedMs,
        nowWallMs: SERVER_WALL_MS + elapsedMs,
      },
    });
  assert.deepEqual(at(0), {
    freshness: "closed",
    ageMs: 0,
    ageText: "Just now",
    receivedAtWallMs: SERVER_WALL_MS,
  });
  assert.deepEqual(at(65_000), {
    freshness: "closed",
    ageMs: 65_000,
    ageText: "1 min ago",
    receivedAtWallMs: SERVER_WALL_MS,
  });
});

test("before the browser clock is read, the display uses the scripted age", () => {
  assert.deepEqual(
    feedDisplay({ receivedAgoMs: 120_000, outcome: "active", clock: null }),
    {
      freshness: "stale",
      ageMs: 120_000,
      ageText: "2 min ago",
      receivedAtWallMs: null,
    },
  );
});

test("no-records replay shows no receipt age and needs no ticking clock", () => {
  const replay = liveReplayData("no-records", { replayEnabled: true }).replay!;
  assert.equal(replay.receivedAgoMs, null);
  assert.deepEqual(replay.view.steps, []);
  assert.equal(feedClockTicks(replay.receivedAgoMs), false);
  const display = feedDisplay({
    receivedAgoMs: replay.receivedAgoMs,
    outcome: runOutcome(replay.run),
    clock: {
      anchorMonotonicMs: HYDRATION_MONOTONIC_MS,
      nowMonotonicMs: HYDRATION_MONOTONIC_MS + 600_000,
      nowWallMs: SERVER_WALL_MS,
    },
  });
  assert.deepEqual(display, {
    freshness: "never",
    ageMs: null,
    ageText: null,
    receivedAtWallMs: null,
  });
});
