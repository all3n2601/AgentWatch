import type { FixtureRun, SavedRun } from "./fixture-control";
import { fixtureRun, resourceSample, snapshot, stepEvent } from "./live-runs";
import { savedRun, step } from "./saved-runs";

// Large synthetic runs for the F04 large-run budgets (large-runs.spec.ts). Generated at
// runtime from a fixed seed, so every run of the suite sees the same data. Synthetic
// browser-test fixtures only; never recorded evidence.

export const LARGE_RUN_SEED = 20_261_124;

/** mulberry32: a small deterministic generator, enough for fixture variety. */
export function seededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
  };
}
export type Random = ReturnType<typeof seededRandom>;

/** A version-4 UUID drawn from `random`. */
export function seededUuid(random: Random) {
  const hex = Array.from({ length: 32 }, () =>
    Math.floor(random() * 16).toString(16),
  );
  hex[12] = "4";
  hex[16] = ((parseInt(hex[16], 16) & 0x3) | 0x8).toString(16);
  const h = hex.join("");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

/** `count` saved coding runs, newest last; about a third failed their checks. */
export function largeSavedHistory(random: Random, count: number): SavedRun[] {
  return Array.from({ length: count }, (_, index) =>
    savedRun(seededUuid(random), {
      sequence: index + 1,
      passed: random() > 0.33,
      cleanSeconds: 2 + random() * 3,
    }),
  );
}

/** `count` live runs with one queued step each, listed by sequence. */
export function largeLiveList(random: Random, count: number): FixtureRun[] {
  return Array.from({ length: count }, (_, index) => {
    const runId = seededUuid(random);
    const queued = stepEvent("tool-1", 0, "queued", { runId });
    return fixtureRun(
      runId,
      [snapshot(runId, { steps: [queued], records: [queued] })],
      { sequence: index + 1 },
    );
  });
}

export const LARGE_LIVE_STEPS = 2_000;
export const LARGE_LIVE_SAMPLES = 5_000;
/** 1,000 completed steps × 3 events + 1,000 running steps × 2 events + the samples. */
export const LARGE_LIVE_RECORDS = 10_000;
/** Every tenth sample is unavailable: 10%, exactly. */
const UNAVAILABLE_EVERY = 10;

/**
 * An open live run with 2,000 tool steps and 5,000 CPU samples: 10,000 records. The first
 * 1,000 steps completed (queued, running, completed); the rest are running (queued,
 * running). Records are numbered in observed order.
 */
export function largeLiveRun(
  runId: string,
  random: Random,
  sequence: number,
): FixtureRun {
  type Timed = { at: number; payload: { record_type: string } };
  const timed: Timed[] = [];
  const applied = [];
  for (let index = 0; index < LARGE_LIVE_STEPS; index += 1) {
    const stepId = `step-${String(index).padStart(4, "0")}`;
    const submitted = index * 0.5;
    const started = submitted + random() * 0.4;
    const queued = stepEvent(stepId, 0, "queued", {
      runId,
      submitted,
      second: submitted,
    });
    const running = stepEvent(stepId, 1, "running", {
      runId,
      submitted,
      started,
      second: started,
    });
    timed.push({ at: submitted, payload: queued });
    timed.push({ at: started, payload: running });
    if (index < LARGE_LIVE_STEPS / 2) {
      const finished = started + 0.2 + random() * 2;
      const completed = stepEvent(stepId, 2, "completed", {
        runId,
        submitted,
        started,
        finished,
        cpu: (finished - started) * random(),
        second: finished,
      });
      timed.push({ at: finished, payload: completed });
      applied.push(completed);
    } else applied.push(running);
  }
  for (let index = 0; index < LARGE_LIVE_SAMPLES; index += 1) {
    const second = index * 0.2;
    const unavailable = index % UNAVAILABLE_EVERY === 0;
    timed.push({
      at: second,
      payload: resourceSample(`cpu-${index}`, {
        runId,
        second,
        value: unavailable ? null : 10 + random() * 80,
        availability: unavailable ? "unavailable" : "measured",
      }),
    });
  }
  timed.sort((a, b) => a.at - b.at);
  return fixtureRun(
    runId,
    [
      snapshot(runId, {
        steps: applied,
        records: timed.map((entry) => entry.payload),
      }),
    ],
    { sequence },
  );
}

export const LARGE_SAVED_STEPS = 1_000;
export const LARGE_AGENT_MESSAGES = 500;

/** A saved agent run with 1,000 baseline steps and a 500-message conversation. */
export function largeSavedRun(
  runId: string,
  random: Random,
  sequence: number,
): SavedRun {
  const endMs = Date.parse("2026-11-01T12:00:00Z") + sequence * 3_600_000;
  const roles = ["user", "assistant", "tool", "assistant"] as const;
  return savedRun(runId, {
    sequence,
    model: "qwen3:1.7b",
    baselineSteps: Array.from({ length: LARGE_SAVED_STEPS }, (_, index) =>
      step(
        `baseline-${index}`,
        endMs - 3_600_000 + index * 3_000,
        0.1 + random() * 0.5,
        1 + random() * 1.5,
      ),
    ),
    conversation: {
      messages: Array.from({ length: LARGE_AGENT_MESSAGES }, (_, index) => {
        const role = roles[index % roles.length];
        return {
          role,
          content: `Synthetic browser-test message ${index}: ${"lorem ".repeat(1 + Math.floor(random() * 30))}`,
          ...(role === "tool" && { tool_name: "run_tests" }),
        };
      }),
      inference: Array.from(
        { length: LARGE_AGENT_MESSAGES / 2 },
        (_, index) => ({
          request_index: index,
          wall_seconds: 0.5 + random() * 3,
          metrics: { eval_count: Math.floor(50 + random() * 400) },
        }),
      ),
    },
  });
}
