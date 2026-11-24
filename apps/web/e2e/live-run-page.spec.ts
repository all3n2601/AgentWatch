import { expect, test, type Page } from "@playwright/test";
import { servePolicy } from "./audit";
import {
  advanceRun,
  registerRun,
  resetFixtureApi,
  snapshotRequests,
} from "./fixture-control";
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
  await page.clock.runFor(MAX_RETRY_MS);
  await page.getByRole("tab", { name: "Live status" }).click();
  await expect(connection(page)).toContainText("Connected");
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
  await page.clock.runFor(MAX_RETRY_MS);
  await expect(connection(page)).toContainText("Connected");
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
