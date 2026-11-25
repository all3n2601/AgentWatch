import { expect, test, type Page, type TestInfo } from "@playwright/test";
import { pageOverflow, runRowButton, servePolicy } from "./audit";
import {
  registerRun,
  registerSavedRun,
  resetFixtureApi,
  snapshotRequests,
} from "./fixture-control";
import {
  LARGE_AGENT_MESSAGES,
  LARGE_LIVE_RECORDS,
  LARGE_LIVE_SAMPLES,
  LARGE_LIVE_STEPS,
  LARGE_RUN_SEED,
  LARGE_SAVED_STEPS,
  largeLiveList,
  largeLiveRun,
  largeSavedHistory,
  largeSavedRun,
  seededRandom,
  seededUuid,
} from "./large-runs";

// F04 phase 2: large runs against budgets declared on 2026-11-24, before the first
// measurement, and committed before this spec was first run. A budget is never loosened to
// pass; a missed budget is reported as a finding. Every measurement is printed and attached,
// passing or not. Synthetic fixture data only (e2e/large-runs.ts, seed LARGE_RUN_SEED).

const BUDGETS = {
  /** Navigation start until the selected tab shows its data. */
  firstTabVisibleMs: 5_000,
  /** A tab click until the new tab shows its data. */
  tabSwitchMs: 1_000,
  /** The longest main-thread task while the open run is polled (after it has loaded). */
  longTaskWhilePollingMs: 500,
} as const;

/** Polls observed for the long-task budget (LIVE_POLL_INTERVAL_MS is 2 s). */
const POLLS_OBSERVED = 3;
/** Generous waits, so a slow page is measured rather than cut off by a timeout. */
const MEASURE_TIMEOUT_MS = 60_000;

test.describe.configure({ timeout: 180_000 });

type Measurement = {
  page: string;
  what: string;
  ms: number;
  budgetMs: number | null;
};

async function record(testInfo: TestInfo, measurements: Measurement[]) {
  const lines = measurements.map(
    (m) =>
      `${m.page} | ${m.what} | ${Math.round(m.ms)} ms | ${m.budgetMs === null ? "no budget" : `budget ${m.budgetMs} ms: ${m.ms <= m.budgetMs ? "met" : "MISSED"}`}`,
  );
  console.log(`\n${testInfo.title}\n${lines.join("\n")}`);
  await testInfo.attach("measurements.json", {
    body: JSON.stringify(measurements, null, 2),
    contentType: "application/json",
  });
  for (const m of measurements)
    if (m.budgetMs !== null)
      expect
        .soft(m.ms, `${m.page}: ${m.what} (budget ${m.budgetMs} ms)`)
        .toBeLessThanOrEqual(m.budgetMs);
}

/** Milliseconds until `ready` resolves, started just before `action`. */
async function timed(
  action: () => Promise<unknown>,
  ready: () => Promise<unknown>,
) {
  const started = performance.now();
  await action();
  await ready();
  return performance.now() - started;
}

type LongTask = { start: number; duration: number };
declare global {
  interface Window {
    /** Filled by observeLongTasks in the page under test. */
    __longTasks?: LongTask[];
  }
}

/** Collects main-thread long tasks from page load on (PerformanceObserver "longtask"). */
async function observeLongTasks(page: Page) {
  await page.addInitScript(() => {
    const tasks: LongTask[] = [];
    window.__longTasks = tasks;
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries())
        tasks.push({ start: entry.startTime, duration: entry.duration });
    }).observe({ type: "longtask", buffered: true });
  });
}

/** The longest task that started after `sinceMs` (page time), in milliseconds. */
const longestTaskSince = (page: Page, sinceMs: number) =>
  page.evaluate((since) => {
    return Math.max(
      0,
      ...(window.__longTasks ?? [])
        .filter((task) => task.start >= since)
        .map((task) => task.duration),
    );
  }, sinceMs);

/** Waits (in real time) for `polls` more snapshot reads of `runId`; returns the longest task. */
async function longestTaskWhilePolling(
  page: Page,
  request: Parameters<typeof snapshotRequests>[0],
  runId: string,
) {
  const since = await page.evaluate(() => performance.now());
  const before = (await snapshotRequests(request, runId)).length;
  await expect
    .poll(async () => (await snapshotRequests(request, runId)).length, {
      timeout: MEASURE_TIMEOUT_MS,
    })
    .toBeGreaterThanOrEqual(before + POLLS_OBSERVED);
  return longestTaskSince(page, since);
}

async function expectNoPageOverflow(page: Page, where: string) {
  const { scrollWidth, clientWidth } = await pageOverflow(page);
  expect
    .soft(scrollWidth, `${where}: page scrolls sideways`)
    .toBeLessThanOrEqual(clientWidth);
}

test.beforeEach(async ({ page, request }) => {
  await resetFixtureApi(request);
  await servePolicy(page);
  await observeLongTasks(page);
});

test("the Runs list with 300 saved and 120 live runs", async ({
  page,
  request,
}, testInfo) => {
  const random = seededRandom(LARGE_RUN_SEED);
  const saved = largeSavedHistory(random, 300);
  for (const run of saved) await registerSavedRun(request, run);
  for (const run of largeLiveList(random, 120)) await registerRun(request, run);
  const newest = saved.at(-1)!.run_id;
  const measurements: Measurement[] = [];
  const at = (what: string, ms: number, budgetMs: number | null) =>
    measurements.push({ page: "/runs", what, ms, budgetMs });

  at(
    "first tab visible",
    await timed(
      () => page.goto("/runs"),
      () =>
        expect(runRowButton(page, newest)).toBeVisible({
          timeout: MEASURE_TIMEOUT_MS,
        }),
    ),
    BUDGETS.firstTabVisibleMs,
  );
  const pager = page.getByRole("navigation", { name: "Run list pages" });
  // 100 saved (the first history page) and 50 live (the newest live page) are loaded.
  await expect(pager).toContainText("Page 1 of 6");
  at(
    "next list page",
    await timed(
      () => pager.getByRole("button", { name: "Next" }).click(),
      () => expect(pager).toContainText("Page 2 of 6"),
    ),
    null,
  );
  at(
    "load older saved runs",
    await timed(
      () => page.getByRole("button", { name: "Load older saved runs" }).click(),
      // The list stays on the page it was on; older runs add pages.
      () => expect(pager).toContainText("Page 2 of 10"),
    ),
    null,
  );
  at(
    "switch to Compare selection tab",
    await timed(
      () => page.getByRole("tab", { name: /Compare selection/ }).click(),
      () =>
        expect(page.getByText("No runs selected for comparison")).toBeVisible(),
    ),
    BUDGETS.tabSwitchMs,
  );
  await expectNoPageOverflow(page, "/runs");
  await record(testInfo, measurements);
});

test(`/live with a ${LARGE_LIVE_STEPS}-step, 10,000-record run selected`, async ({
  page,
  request,
}, testInfo) => {
  const random = seededRandom(LARGE_RUN_SEED + 1);
  for (const run of largeLiveList(random, 119)) await registerRun(request, run);
  const runId = seededUuid(random);
  await registerRun(request, largeLiveRun(runId, random, 120));
  const measurements: Measurement[] = [];
  const at = (what: string, ms: number, budgetMs: number | null) =>
    measurements.push({ page: "/live", what, ms, budgetMs });

  at(
    "first tab visible (all records read)",
    await timed(
      () => page.goto(`/live?liveRun=${runId}`),
      () =>
        expect(
          page.getByText(`${LARGE_LIVE_RECORDS} records received`).first(),
        ).toBeVisible({ timeout: MEASURE_TIMEOUT_MS }),
    ),
    BUDGETS.firstTabVisibleMs,
  );
  for (const [tab, ready] of [
    ["Timeline", () => page.getByText(`${LARGE_LIVE_STEPS} steps in a`)],
    [
      "Resources",
      () => page.getByRole("heading", { name: /Resource/ }).first(),
    ],
    [
      "Evidence",
      () => page.getByRole("link", { name: "Download live records (JSON)" }),
    ],
    ["Status", () => page.getByText("Run state", { exact: true })],
  ] as const) {
    at(
      `switch to ${tab}`,
      await timed(
        () => page.getByRole("tab", { name: tab, exact: true }).click(),
        () => expect(ready()).toBeVisible({ timeout: MEASURE_TIMEOUT_MS }),
      ),
      BUDGETS.tabSwitchMs,
    );
  }
  // No budget: shows the long-task observer works, so a 0 ms result while polling is real.
  at("longest task while loading", await longestTaskSince(page, 0), null);
  at(
    `longest task over ${POLLS_OBSERVED} polls (Status tab)`,
    await longestTaskWhilePolling(page, request, runId),
    BUDGETS.longTaskWhilePollingMs,
  );
  await page.getByRole("tab", { name: "Timeline", exact: true }).click();
  at(
    `longest task over ${POLLS_OBSERVED} polls (Timeline tab)`,
    await longestTaskWhilePolling(page, request, runId),
    BUDGETS.longTaskWhilePollingMs,
  );
  await expectNoPageOverflow(page, "/live");
  await record(testInfo, measurements);
});

test(`a live run's page with ${LARGE_LIVE_STEPS} steps and ${LARGE_LIVE_SAMPLES} samples`, async ({
  page,
  request,
}, testInfo) => {
  const random = seededRandom(LARGE_RUN_SEED + 2);
  const runId = seededUuid(random);
  await registerRun(request, largeLiveRun(runId, random, 1));
  const measurements: Measurement[] = [];
  const at = (what: string, ms: number, budgetMs: number | null) =>
    measurements.push({ page: "/runs/{live}", what, ms, budgetMs });
  const timelineReady = () => page.getByText(`${LARGE_LIVE_STEPS} steps in a`);

  at(
    "first tab visible (Timeline, all records read)",
    await timed(
      () => page.goto(`/runs/${runId}?tab=timeline`),
      () =>
        expect(timelineReady()).toBeVisible({ timeout: MEASURE_TIMEOUT_MS }),
    ),
    BUDGETS.firstTabVisibleMs,
  );
  for (const [tab, ready] of [
    [
      "Resources",
      () => page.getByRole("heading", { name: /Resource/ }).first(),
    ],
    ["Live status", () => page.getByText("Run state", { exact: true })],
    ["Timeline", timelineReady],
  ] as const) {
    at(
      `switch to ${tab}`,
      await timed(
        () => page.getByRole("tab", { name: tab, exact: true }).click(),
        () => expect(ready()).toBeVisible({ timeout: MEASURE_TIMEOUT_MS }),
      ),
      BUDGETS.tabSwitchMs,
    );
  }
  at("longest task while loading", await longestTaskSince(page, 0), null);
  at(
    `longest task over ${POLLS_OBSERVED} polls (Timeline tab)`,
    await longestTaskWhilePolling(page, request, runId),
    BUDGETS.longTaskWhilePollingMs,
  );
  // 10% of the samples are unavailable and are reported as such, never as zero.
  await page.getByRole("tab", { name: "Resources", exact: true }).click();
  await expect(page.getByText(/unavailable/i).first()).toBeVisible();
  await expectNoPageOverflow(page, "/runs/{live}");
  await record(testInfo, measurements);
});

test(`a saved run with ${LARGE_SAVED_STEPS} steps and ${LARGE_AGENT_MESSAGES} agent messages`, async ({
  page,
  request,
}, testInfo) => {
  const random = seededRandom(LARGE_RUN_SEED + 3);
  const runId = seededUuid(random);
  await registerSavedRun(request, largeSavedRun(runId, random, 1));
  const measurements: Measurement[] = [];
  const at = (what: string, ms: number, budgetMs: number | null) =>
    measurements.push({ page: "/runs/{saved}", what, ms, budgetMs });

  at(
    "first tab visible (Diagnosis)",
    await timed(
      () => page.goto(`/runs/${runId}`),
      () =>
        expect(
          page.getByRole("tab", { name: "Diagnosis", selected: true }),
        ).toBeVisible({ timeout: MEASURE_TIMEOUT_MS }),
    ),
    BUDGETS.firstTabVisibleMs,
  );
  for (const [tab, ready] of [
    ["Timeline", () => page.getByText(`${LARGE_SAVED_STEPS + 2} spans in a`)],
    [
      "Agent",
      () => page.getByText("Synthetic browser-test message 499").first(),
    ],
    [
      "Resources",
      () => page.getByRole("tab", { name: "Resources", selected: true }),
    ],
    [
      "Evidence",
      () => page.getByRole("link", { name: "Download results.json" }),
    ],
  ] as const) {
    at(
      `switch to ${tab}`,
      await timed(
        () => page.getByRole("tab", { name: tab, exact: true }).click(),
        () => expect(ready()).toBeAttached({ timeout: MEASURE_TIMEOUT_MS }),
      ),
      BUDGETS.tabSwitchMs,
    );
  }
  await expectNoPageOverflow(page, "/runs/{saved}");
  await record(testInfo, measurements);
});
