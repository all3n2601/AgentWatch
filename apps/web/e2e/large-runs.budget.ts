import { expect, test, type Page, type TestInfo } from "@playwright/test";
import { servePolicy } from "./audit";
import { resetFixtureApi, snapshotRequests } from "./fixture-control";
import {
  expectLiveRunPageResources,
  largeLiveTimelineReady,
  LARGE_RUN_TIMEOUT_MS,
  liveRunPageFlow,
  liveViewFlow,
  runsListFlow,
  savedRunPageFlow,
  type RunStep,
  type StepBudget,
} from "./large-run-flows";
import {
  LARGE_AGENT_MESSAGES,
  LARGE_LIVE_SAMPLES,
  LARGE_LIVE_STEPS,
  LARGE_SAVED_STEPS,
} from "./large-runs";

// F04 phase 2: large-run budgets, run with `pnpm test:budgets` (playwright.budgets.config.ts),
// outside the required browser suite. The budgets were declared on 2026-11-24, before the
// first measurement, and committed before this spec was first run. They are never loosened
// to pass; a missed budget is a finding. CI at 6b80a42 missed five of them (recorded in
// docs/project-checklist.md, F04), so they were moved out of the required check unchanged on
// 2026-11-26. The same flows run in the required suite (large-runs.spec.ts) without timing.
// Every measurement is printed and attached, passing or not.

const BUDGETS = {
  /** Navigation start until the selected tab shows its data. */
  firstTabVisibleMs: 5_000,
  /** A tab click until the new tab shows its data. */
  tabSwitchMs: 1_000,
  /** The longest main-thread task while the open run is polled (after it has loaded). */
  longTaskWhilePollingMs: 500,
} as const;

const BUDGET_FOR_STEP: Record<Exclude<StepBudget, null>, number> = {
  firstTabVisible: BUDGETS.firstTabVisibleMs,
  tabSwitch: BUDGETS.tabSwitchMs,
};

/** Polls observed for the long-task budget (LIVE_POLL_INTERVAL_MS is 2 s). */
const POLLS_OBSERVED = 3;

test.describe.configure({ timeout: 180_000 });

type Measurement = {
  page: string;
  what: string;
  ms: number;
  budgetMs: number | null;
};

/** A RunStep that times each step, from just before its action until it is ready. */
function timedSteps(pageName: string, measurements: Measurement[]): RunStep {
  return async ({ what, budget, action, ready }) => {
    const started = performance.now();
    await action();
    await ready();
    measurements.push({
      page: pageName,
      what,
      ms: performance.now() - started,
      budgetMs: budget === null ? null : BUDGET_FOR_STEP[budget],
    });
  };
}

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
      timeout: LARGE_RUN_TIMEOUT_MS,
    })
    .toBeGreaterThanOrEqual(before + POLLS_OBSERVED);
  return longestTaskSince(page, since);
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
  const measurements: Measurement[] = [];
  await runsListFlow(page, request, timedSteps("/runs", measurements));
  await record(testInfo, measurements);
});

test(`/live with a ${LARGE_LIVE_STEPS}-step, 10,000-record run selected`, async ({
  page,
  request,
}, testInfo) => {
  const measurements: Measurement[] = [];
  const at = (what: string, ms: number, budgetMs: number | null) =>
    measurements.push({ page: "/live", what, ms, budgetMs });
  const runId = await liveViewFlow(
    page,
    request,
    timedSteps("/live", measurements),
  );
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
  await record(testInfo, measurements);
});

test(`a live run's page with ${LARGE_LIVE_STEPS} steps and ${LARGE_LIVE_SAMPLES} samples`, async ({
  page,
  request,
}, testInfo) => {
  const measurements: Measurement[] = [];
  const at = (what: string, ms: number, budgetMs: number | null) =>
    measurements.push({ page: "/runs/{live}", what, ms, budgetMs });
  const runId = await liveRunPageFlow(
    page,
    request,
    timedSteps("/runs/{live}", measurements),
  );
  await expect(largeLiveTimelineReady(page)).toBeVisible();
  at("longest task while loading", await longestTaskSince(page, 0), null);
  at(
    `longest task over ${POLLS_OBSERVED} polls (Timeline tab)`,
    await longestTaskWhilePolling(page, request, runId),
    BUDGETS.longTaskWhilePollingMs,
  );
  // The Timeline panel stays mounted (hidden) while another tab is open (run-tabs.tsx).
  await page.getByRole("tab", { name: "Live status", exact: true }).click();
  at(
    `longest task over ${POLLS_OBSERVED} polls (Live status tab, Timeline hidden)`,
    await longestTaskWhilePolling(page, request, runId),
    BUDGETS.longTaskWhilePollingMs,
  );
  await expectLiveRunPageResources(page);
  await record(testInfo, measurements);
});

test(`a saved run with ${LARGE_SAVED_STEPS} steps and ${LARGE_AGENT_MESSAGES} agent messages`, async ({
  page,
  request,
}, testInfo) => {
  const measurements: Measurement[] = [];
  await savedRunPageFlow(
    page,
    request,
    timedSteps("/runs/{saved}", measurements),
  );
  await record(testInfo, measurements);
});
