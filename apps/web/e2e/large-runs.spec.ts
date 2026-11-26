import { test } from "@playwright/test";
import { servePolicy } from "./audit";
import { resetFixtureApi } from "./fixture-control";
import {
  expectLiveRunPageResources,
  liveRunPageFlow,
  liveViewFlow,
  runStep,
  runsListFlow,
  savedRunPageFlow,
} from "./large-run-flows";
import {
  LARGE_AGENT_MESSAGES,
  LARGE_LIVE_SAMPLES,
  LARGE_LIVE_STEPS,
  LARGE_SAVED_STEPS,
} from "./large-runs";

// F04 phase 2: large runs render and work. This spec checks only what the pages show; how
// fast they show it is measured against the declared budgets by large-runs.budget.ts
// (`pnpm test:budgets`), outside the required suite, because the budgets were set on a
// development machine and CI hardware is slower (see docs/project-checklist.md, F04).

// A large page can be slow on CI; it is checked, not timed, here.
test.describe.configure({ timeout: 180_000 });

test.beforeEach(async ({ page, request }) => {
  await resetFixtureApi(request);
  await servePolicy(page);
});

test("the Runs list with 300 saved and 120 live runs pages and loads older history", async ({
  page,
  request,
}) => {
  await runsListFlow(page, request, runStep);
});

test(`/live reads all records of a ${LARGE_LIVE_STEPS}-step, 10,000-record run and shows every tab`, async ({
  page,
  request,
}) => {
  await liveViewFlow(page, request, runStep);
});

test(`a live run's page with ${LARGE_LIVE_STEPS} steps and ${LARGE_LIVE_SAMPLES} samples shows every tab and its unavailable samples`, async ({
  page,
  request,
}) => {
  await liveRunPageFlow(page, request, runStep);
  await expectLiveRunPageResources(page);
});

test(`a saved run with ${LARGE_SAVED_STEPS} steps and ${LARGE_AGENT_MESSAGES} agent messages shows every tab`, async ({
  page,
  request,
}) => {
  await savedRunPageFlow(page, request, runStep);
});
