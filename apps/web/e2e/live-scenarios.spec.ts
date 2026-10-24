import { expect, test } from "@playwright/test";
import { registerRun, resetFixtureApi } from "./fixture-control";
import { liveScenarios } from "./fixtures/live-scenarios";
import { newRunId } from "./live-runs";

// Each synthetic scenario (formerly a ?replay= development replay) through the real live path.

test.beforeEach(async ({ request }) => resetFixtureApi(request));

for (const scenario of liveScenarios)
  test(`scenario ${scenario.id}: ${scenario.description}`, async ({
    page,
    request,
  }) => {
    const runId = newRunId();
    await registerRun(request, scenario.build(runId));
    await page.goto(`/live?liveRun=${runId}`);
    for (const text of scenario.shows)
      await expect(page.getByText(text).first()).toBeVisible();
  });

test("the former replay parameter no longer serves synthetic data", async ({
  page,
}) => {
  await page.goto("/live?replay=active-steps");
  await expect(page.getByText("Development replay")).toHaveCount(0);
  await expect(page.getByRole("note")).toContainText("Live API");
});
