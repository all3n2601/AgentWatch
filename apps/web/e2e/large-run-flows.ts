import {
  expect,
  type APIRequestContext,
  type Locator,
  type Page,
} from "@playwright/test";
import { pageOverflow, runRowButton } from "./audit";
import { registerRun, registerSavedRun } from "./fixture-control";
import {
  LARGE_LIVE_RECORDS,
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

// The large-run flows, shared by two suites:
// - large-runs.spec.ts (the required browser suite) runs each flow and checks what it shows;
// - large-runs.budget.ts (`pnpm test:budgets`) runs the same flows and times each step
//   against the declared budgets.
// Synthetic fixture data only (e2e/large-runs.ts, seed LARGE_RUN_SEED).

/**
 * Generous waits, so a slow page is still checked (functional suite) or measured (budget
 * suite) rather than cut off. Speed is judged only by the budget suite.
 */
export const LARGE_RUN_TIMEOUT_MS = 60_000;

/** The declared budget a step is measured against in the budget suite, if any. */
export type StepBudget = "firstTabVisible" | "tabSwitch" | null;

/** One user step: `action`, then wait until `ready` passes. */
export type FlowStep = {
  what: string;
  budget: StepBudget;
  action: () => Promise<unknown>;
  ready: () => Promise<unknown>;
};

/** Runs a step. The budget suite's version also times it. */
export type RunStep = (step: FlowStep) => Promise<void>;

/** Runs a step without timing it, for the functional suite. */
export const runStep: RunStep = async ({ action, ready }) => {
  await action();
  await ready();
};

async function expectNoPageOverflow(page: Page, where: string) {
  const { scrollWidth, clientWidth } = await pageOverflow(page);
  expect
    .soft(scrollWidth, `${where}: page scrolls sideways`)
    .toBeLessThanOrEqual(clientWidth);
}

/** Visible within LARGE_RUN_TIMEOUT_MS. */
const shown = (locator: Locator) => () =>
  expect(locator).toBeVisible({ timeout: LARGE_RUN_TIMEOUT_MS });

/** The Runs list with 300 saved and 120 live runs: paging, older history, Compare tab. */
export async function runsListFlow(
  page: Page,
  request: APIRequestContext,
  run: RunStep,
) {
  const random = seededRandom(LARGE_RUN_SEED);
  const saved = largeSavedHistory(random, 300);
  for (const savedRun of saved) await registerSavedRun(request, savedRun);
  for (const liveRun of largeLiveList(random, 120))
    await registerRun(request, liveRun);
  const newest = saved.at(-1)!.run_id;
  const pager = page.getByRole("navigation", { name: "Run list pages" });

  await run({
    what: "first tab visible",
    budget: "firstTabVisible",
    action: () => page.goto("/runs"),
    ready: shown(runRowButton(page, newest)),
  });
  // 100 saved (the first history page) and 50 live (the newest live page) are loaded.
  await expect(pager).toContainText("Page 1 of 6");
  await run({
    what: "next list page",
    budget: null,
    action: () => pager.getByRole("button", { name: "Next" }).click(),
    ready: () => expect(pager).toContainText("Page 2 of 6"),
  });
  await run({
    what: "load older saved runs",
    budget: null,
    action: () =>
      page.getByRole("button", { name: "Load older saved runs" }).click(),
    // The list stays on the page it was on; older runs add pages.
    ready: () => expect(pager).toContainText("Page 2 of 10"),
  });
  await run({
    what: "switch to Compare selection tab",
    budget: "tabSwitch",
    action: () => page.getByRole("tab", { name: /Compare selection/ }).click(),
    ready: shown(page.getByText("No runs selected for comparison")),
  });
  await expectNoPageOverflow(page, "/runs");
}

/**
 * /live with a 2,000-step, 10,000-record run selected among 120 live runs: every record
 * read, then each tab. Ends on the Status tab. Returns the large run's id.
 */
export async function liveViewFlow(
  page: Page,
  request: APIRequestContext,
  run: RunStep,
) {
  const random = seededRandom(LARGE_RUN_SEED + 1);
  for (const liveRun of largeLiveList(random, 119))
    await registerRun(request, liveRun);
  const runId = seededUuid(random);
  await registerRun(request, largeLiveRun(runId, random, 120));

  await run({
    what: "first tab visible (all records read)",
    budget: "firstTabVisible",
    action: () => page.goto(`/live?liveRun=${runId}`),
    ready: shown(
      page.getByText(`${LARGE_LIVE_RECORDS} records received`).first(),
    ),
  });
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
  ] as const)
    await run({
      what: `switch to ${tab}`,
      budget: "tabSwitch",
      action: () => page.getByRole("tab", { name: tab, exact: true }).click(),
      ready: shown(ready()),
    });
  await expectNoPageOverflow(page, "/live");
  return runId;
}

/** The Timeline tab's summary on a large live run's page or on /live. */
export const largeLiveTimelineReady = (page: Page) =>
  page.getByText(`${LARGE_LIVE_STEPS} steps in a`);

/**
 * A live run's own page with 2,000 steps and 5,000 samples, opened on the Timeline tab, then
 * each tab, back to Timeline. Returns the run's id.
 */
export async function liveRunPageFlow(
  page: Page,
  request: APIRequestContext,
  run: RunStep,
) {
  const random = seededRandom(LARGE_RUN_SEED + 2);
  const runId = seededUuid(random);
  await registerRun(request, largeLiveRun(runId, random, 1));

  await run({
    what: "first tab visible (Timeline, all records read)",
    budget: "firstTabVisible",
    action: () => page.goto(`/runs/${runId}?tab=timeline`),
    ready: shown(largeLiveTimelineReady(page)),
  });
  for (const [tab, ready] of [
    [
      "Resources",
      () => page.getByRole("heading", { name: /Resource/ }).first(),
    ],
    ["Live status", () => page.getByText("Run state", { exact: true })],
    ["Timeline", () => largeLiveTimelineReady(page)],
  ] as const)
    await run({
      what: `switch to ${tab}`,
      budget: "tabSwitch",
      action: () => page.getByRole("tab", { name: tab, exact: true }).click(),
      ready: shown(ready()),
    });
  return runId;
}

/**
 * After liveRunPageFlow: 10% of the samples are unavailable, and the Resources tab reports
 * them as such, never as zero. The page doesn't scroll sideways.
 */
export async function expectLiveRunPageResources(page: Page) {
  await page.getByRole("tab", { name: "Resources", exact: true }).click();
  await expect(page.getByText(/unavailable/i).first()).toBeVisible();
  await expectNoPageOverflow(page, "/runs/{live}");
}

/** A saved run with 1,000 steps and 500 agent messages: the first tab, then each tab. */
export async function savedRunPageFlow(
  page: Page,
  request: APIRequestContext,
  run: RunStep,
) {
  const random = seededRandom(LARGE_RUN_SEED + 3);
  const runId = seededUuid(random);
  await registerSavedRun(request, largeSavedRun(runId, random, 1));
  const attached = (locator: Locator) => () =>
    expect(locator).toBeAttached({ timeout: LARGE_RUN_TIMEOUT_MS });

  await run({
    what: "first tab visible (Diagnosis)",
    budget: "firstTabVisible",
    action: () => page.goto(`/runs/${runId}`),
    ready: shown(page.getByRole("tab", { name: "Diagnosis", selected: true })),
  });
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
  ] as const)
    await run({
      what: `switch to ${tab}`,
      budget: "tabSwitch",
      action: () => page.getByRole("tab", { name: tab, exact: true }).click(),
      ready: attached(ready()),
    });
  await expectNoPageOverflow(page, "/runs/{saved}");
}
