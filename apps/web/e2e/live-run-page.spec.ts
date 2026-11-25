import { expect, test, type Locator, type Page } from "@playwright/test";
import { servePolicy } from "./audit";
import {
  advanceRun,
  attributionReads,
  registerRun,
  resetFixtureApi,
  snapshotRequests,
} from "./fixture-control";
import {
  attributedRun,
  attributionItem,
  attributionRecord,
  attributionState,
} from "./fixtures/attributions";
import { fixtureRun, newRunId, snapshot, stepEvent } from "./live-runs";

// F04 phase 2: a live run's own page (/runs/{id}) while it is polled. The browser's clock is
// controlled (page.clock), so poll timing is deterministic and no test sleeps. Every run is
// synthetic fixture data.

/** LIVE_POLL_INTERVAL_MS in lib/live-poll.ts. */
const POLL_MS = 2_000;
/** LIVE_FEED_STALE_AFTER_MS in lib/live-status.ts. */
const STALE_AFTER_MS = 30_000;
/** The longest jittered retry delay (lib/live-poll.ts: 30 s cap, +20%). */
const MAX_RETRY_MS = 36_000;
/** Fake time advanced per attempt while waiting for a retry to land. */
const CLOCK_STEP_MS = 500;

test.beforeEach(async ({ page, request }) => {
  await resetFixtureApi(request);
  await servePolicy(page);
  await page.clock.install();
});

/** One tool step queued, then running, then completed, for `runId`. */
function toolSteps(runId: string) {
  const queued = stepEvent("tool-1", 0, "queued", { runId });
  const running = stepEvent("tool-1", 1, "running", { runId, started: 1 });
  const completed = stepEvent("tool-1", 2, "completed", {
    runId,
    started: 1,
    finished: 3,
    cpu: 0.5,
  });
  return { queued, running, completed };
}

const connection = (page: Page) =>
  page.getByRole("status").filter({ hasText: "Live API connection" });
const feedTile = (page: Page) =>
  page.getByRole("status").filter({ hasText: "Feed" }).first();

/**
 * Advances the page clock in short steps until `locator` shows `text`, within the longest
 * retry delay. One large jump would also fire the page's 10 s request timeout (a fake-clock
 * timer) while a real response is still on its way, and the retry would fail again.
 */
async function advanceUntil(page: Page, locator: Locator, text: string) {
  let advancedMs = 0;
  await expect(async () => {
    const shown = (await locator.textContent()) ?? "";
    if (shown.includes(text)) return;
    expect(advancedMs, `"${text}" within the retry delay`).toBeLessThan(
      MAX_RETRY_MS,
    );
    await page.clock.runFor(CLOCK_STEP_MS);
    advancedMs += CLOCK_STEP_MS;
    throw new Error(`waiting for "${text}"`);
  }).toPass({ timeout: 60_000, intervals: [250] });
}

/** Hides or shows the page as a background browser tab does. */
async function setHidden(page: Page, hidden: boolean) {
  await page.evaluate((value) => {
    Object.defineProperty(document, "hidden", {
      configurable: true,
      get: () => value,
    });
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      get: () => (value ? "hidden" : "visible"),
    });
    document.dispatchEvent(new Event("visibilitychange"));
  }, hidden);
}

test("a failed read on the run page keeps the last data and recovers", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  const { queued, running } = toolSteps(runId);
  const other = stepEvent("tool-2", 0, "queued", { runId, second: 2 });
  await registerRun(
    request,
    fixtureRun(runId, [
      snapshot(runId, { steps: [running], records: [queued, running] }),
      { status: 503 },
      snapshot(runId, {
        steps: [running, other],
        records: [queued, running, other],
      }),
    ]),
  );
  await page.goto(`/runs/${runId}?tab=status`);
  await expect(connection(page)).toContainText("Connected");

  await advanceRun(request, runId);
  await page.clock.runFor(POLL_MS);
  await expect(connection(page)).toContainText("Reconnecting (attempt 1)");
  await expect(connection(page)).toContainText(
    "the last data received stays shown",
  );
  // The last valid snapshot is still on screen.
  await page.getByRole("tab", { name: "Timeline" }).click();
  await expect(page.getByRole("button", { name: /tool-1/ })).toBeVisible();

  await advanceRun(request, runId);
  await page.getByRole("tab", { name: "Live status" }).click();
  await advanceUntil(page, connection(page), "Connected");
  await page.getByRole("tab", { name: "Timeline" }).click();
  await expect(page.getByRole("button", { name: /tool-2/ })).toBeVisible();
});

test("every tab of the run page shares one poll of the live API", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  const { queued, running } = toolSteps(runId);
  await registerRun(
    request,
    fixtureRun(runId, [
      snapshot(runId, { steps: [running], records: [queued, running] }),
    ]),
  );
  await page.goto(`/runs/${runId}?tab=status`);
  await expect(connection(page)).toContainText("Connected");
  for (const tab of ["Timeline", "Resources", "Evidence", "Live status"]) {
    await page.getByRole("tab", { name: tab, exact: true }).click();
    await expect(
      page.getByRole("tab", { name: tab, exact: true }),
    ).toHaveAttribute("aria-selected", "true");
  }

  const before = (await snapshotRequests(request, runId)).length;
  const polls = 5;
  for (let index = 0; index < polls; index += 1) {
    await page.clock.runFor(POLL_MS);
    await expect
      .poll(async () => (await snapshotRequests(request, runId)).length)
      .toBeGreaterThanOrEqual(before + index + 1);
  }
  // One poller: one read per interval, not one per tab that was opened.
  expect((await snapshotRequests(request, runId)).length - before).toBe(polls);
});

test("polling pauses while the page is hidden and reads at once when it is shown", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  const { queued, running } = toolSteps(runId);
  await registerRun(
    request,
    fixtureRun(runId, [
      snapshot(runId, { steps: [running], records: [queued, running] }),
    ]),
  );
  await page.goto(`/runs/${runId}?tab=status`);
  await expect(connection(page)).toContainText("Connected");
  await page.clock.runFor(POLL_MS);
  await expect
    .poll(async () => (await snapshotRequests(request, runId)).length)
    .toBeGreaterThanOrEqual(2);

  await setHidden(page, true);
  // The read that was already scheduled finds the page hidden and does not run.
  const hiddenFrom = (await snapshotRequests(request, runId)).length;
  await page.clock.runFor(POLL_MS * 10);
  expect((await snapshotRequests(request, runId)).length).toBe(hiddenFrom);

  // Shown again: exactly one read, straight away. A read made while hidden would have
  // arrived before it and pushed the count past this.
  await setHidden(page, false);
  await expect
    .poll(async () => (await snapshotRequests(request, runId)).length)
    .toBe(hiddenFrom + 1);
});

test("a reload keeps the live run and the tab in the address", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  const { queued, running } = toolSteps(runId);
  await registerRun(
    request,
    fixtureRun(runId, [
      snapshot(runId, { steps: [running], records: [queued, running] }),
    ]),
  );
  for (const [path, tab] of [
    [`/runs/${runId}?tab=resources`, "Resources"],
    [`/live?liveRun=${runId}&tab=evidence`, "Evidence"],
  ]) {
    await page.goto(path);
    await expect(page.getByRole("tab", { name: tab })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    await page.reload();
    await expect(page).toHaveURL(new RegExp(runId));
    await expect(page.getByRole("tab", { name: tab })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  }
  await expect(
    page.getByRole("link", { name: new RegExp(runId.slice(0, 8)) }),
  ).toHaveAttribute("aria-current", "page");
});

test("offline, the last data stays shown and is labelled stale until the API is back", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  const { queued, running } = toolSteps(runId);
  await registerRun(
    request,
    fixtureRun(runId, [
      snapshot(runId, { steps: [running], records: [queued, running] }),
    ]),
  );
  await page.goto(`/runs/${runId}?tab=status`);
  await expect(connection(page)).toContainText("Connected");
  await expect(feedTile(page)).toContainText("Receiving");

  await page.context().setOffline(true);
  await page.clock.runFor(POLL_MS);
  await expect(connection(page)).toContainText("Reconnecting");
  await page.clock.runFor(STALE_AFTER_MS);
  await expect(feedTile(page)).toContainText("Stale");
  // The run's last state is still shown, not replaced by an empty or zero state.
  const runState = page.getByText("Run state", { exact: true }).locator("..");
  await expect(runState).toContainText("Running");

  await page.context().setOffline(false);
  await advanceUntil(page, connection(page), "Connected");
});

test("a step that is only queued is labelled Queued on the timeline, never Running", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  const { queued, running } = toolSteps(runId);
  const waiting = stepEvent("tool-2", 0, "queued", { runId, second: 2 });
  await registerRun(
    request,
    fixtureRun(runId, [
      snapshot(runId, {
        steps: [running, waiting],
        records: [queued, running, waiting],
      }),
    ]),
  );
  await page.goto(`/runs/${runId}?tab=timeline`);
  const step = (id: string) =>
    page.getByRole("list", { name: "Step timeline" }).getByRole("button", {
      name: new RegExp(`\\b${id}\\b`),
    });
  await expect(step("tool-2")).toContainText("Queued");
  await expect(step("tool-2")).not.toContainText("Running");
  await expect(step("tool-1")).toContainText("Running");
});

test("the Timeline stays mounted while another tab is open, keeping the chosen step", async ({
  page,
  request,
}) => {
  // Rebuilding a 2,000-step timeline on every return to the tab missed the 1 s tab-switch
  // budget (large-runs.spec.ts), so the run page keeps it mounted, as /live does.
  const runId = newRunId();
  const { queued, running, completed } = toolSteps(runId);
  const waiting = stepEvent("tool-2", 0, "queued", { runId, second: 4 });
  const started = stepEvent("tool-2", 1, "running", {
    runId,
    started: 5,
    second: 5,
  });
  await registerRun(
    request,
    fixtureRun(runId, [
      snapshot(runId, {
        steps: [completed, started],
        records: [queued, running, completed, waiting, started],
      }),
    ]),
  );
  await page.goto(`/runs/${runId}?tab=timeline`);
  const timeline = page.locator("[aria-label='Step timeline']");
  const step = (id: string) =>
    timeline.getByRole("button", { name: new RegExp(`\\b${id}\\b`) });
  // Without a choice the running step is selected; choose the completed one instead.
  await expect(step("tool-2")).toHaveAttribute("aria-pressed", "true");
  await step("tool-1").click();
  await expect(step("tool-1")).toHaveAttribute("aria-pressed", "true");

  await page.getByRole("tab", { name: "Live status" }).click();
  await expect(timeline).toBeAttached();
  await expect(timeline).toBeHidden();
  await page.getByRole("tab", { name: "Timeline" }).click();
  // A remounted timeline would have dropped the choice and selected tool-2 again.
  await expect(step("tool-1")).toHaveAttribute("aria-pressed", "true");
});

test("the Attribution tab polls only while it is open", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  await registerRun(
    request,
    attributedRun(runId, [
      attributionState(runId, [
        attributionItem(1, attributionRecord(runId, "tool-1")),
      ]),
    ]),
  );
  await page.goto(`/runs/${runId}?tab=attribution`);
  await expect(
    page.getByRole("region", { name: "Step attributions table" }),
  ).toBeVisible();
  await page.clock.runFor(POLL_MS * 2);
  await expect
    .poll(async () => (await attributionReads(request, runId)).length)
    .toBeGreaterThanOrEqual(2);

  await page.getByRole("tab", { name: "Timeline" }).click();
  const closedAt = (await attributionReads(request, runId)).length;
  await page.clock.runFor(POLL_MS * 10);
  // The live snapshot keeps being read for the open tab, so time did pass for polling.
  await expect
    .poll(async () => (await snapshotRequests(request, runId)).length)
    .toBeGreaterThan(5);
  expect((await attributionReads(request, runId)).length).toBe(closedAt);
});
