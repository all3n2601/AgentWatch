import { expect, test } from "@playwright/test";
import { registerRun, resetFixtureApi } from "./fixture-control";
import { fixtureRun, newRunId, snapshot, stepEvent } from "./live-runs";

test.beforeEach(async ({ request }) => resetFixtureApi(request));

test("the live view lists and reads a run from the live API", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  const queued = stepEvent("tool-1", 0, "queued", { runId });
  await registerRun(
    request,
    fixtureRun(runId, [
      snapshot(runId, { steps: [queued], records: [queued] }),
    ]),
  );
  await page.goto("/live");
  await expect(
    page
      .getByRole("navigation", { name: "Recorded live runs" })
      .getByRole("link"),
  ).toHaveCount(1);
  await expect(
    page.getByRole("status").filter({ hasText: "Live API connection" }),
  ).toContainText("Connected");
  await expect(page.getByText("1 record received for run")).toContainText(
    runId,
  );
});
