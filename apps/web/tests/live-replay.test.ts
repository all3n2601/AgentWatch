import assert from "node:assert/strict";
import test from "node:test";
import { replayScenarios } from "../fixtures/step-event-replays";
import { liveReplayData } from "../lib/live-replay";

const NOW_MS = Date.parse("2026-10-14T12:10:00Z");

test("replay is never offered when it is disabled", () => {
  assert.deepEqual(
    liveReplayData("failed-run", { replayEnabled: false, nowMs: NOW_MS }),
    {
      replay: null,
      requestedReplay: null,
      scenarios: [],
      renderedAtMs: NOW_MS,
    },
  );
});

test("an enabled replay reads the scenario through the adapter", () => {
  const data = liveReplayData("stale-feed", {
    replayEnabled: true,
    nowMs: NOW_MS,
  });
  assert.equal(data.replay?.id, "stale-feed");
  assert.equal(data.replay?.receivedAtMs, NOW_MS - 120_000);
  assert.equal(data.replay?.view.steps[0].state, "running");
  assert.equal(data.scenarios.length, replayScenarios.length);
});

test("an unknown or absent scenario keeps the request without a replay", () => {
  const unknown = liveReplayData("nope", {
    replayEnabled: true,
    nowMs: NOW_MS,
  });
  assert.equal(unknown.replay, null);
  assert.equal(unknown.requestedReplay, "nope");
  const absent = liveReplayData(undefined, {
    replayEnabled: true,
    nowMs: NOW_MS,
  });
  assert.equal(absent.requestedReplay, null);
  assert.equal(absent.scenarios.length, replayScenarios.length);
});
