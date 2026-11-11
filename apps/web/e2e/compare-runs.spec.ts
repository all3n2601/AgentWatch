import { expect, test, type Page } from "@playwright/test";
import {
  registerSavedRun,
  resetFixtureApi,
  takeSavedHistoryDown,
} from "./fixture-control";
import { newSavedRunId, savedRun } from "./saved-runs";

// The Compare page shows exactly the saved runs named by UUID in the URL, never substitutes
// another run, and blocks differences between runs whose configuration is not known to match.

test.beforeEach(async ({ request }) => resetFixtureApi(request));

const side = (page: Page, label: "Baseline" | "Candidate") =>
  page.getByRole("group", { name: `${label} run` });

async function threeRuns(request: Parameters<typeof registerSavedRun>[0]) {
  const [older, selected, newest] = [
    newSavedRunId(),
    newSavedRunId(),
    newSavedRunId(),
  ];
  await registerSavedRun(request, savedRun(older, { sequence: 1 }));
  await registerSavedRun(
    request,
    savedRun(selected, { sequence: 2, cleanSeconds: 4.5, passed: false }),
  );
  await registerSavedRun(request, savedRun(newest, { sequence: 3 }));
  return { older, selected, newest };
}

test("the selected UUIDs are shown with their own evidence", async ({
  page,
  request,
}) => {
  const { older, selected } = await threeRuns(request);
  await page.goto(`/compare?run=${selected}&baseline=${older}`);
  await expect(
    side(page, "Baseline").getByText(older, { exact: true }),
  ).toBeVisible();
  await expect(
    side(page, "Candidate").getByText(selected, { exact: true }),
  ).toBeVisible();

  const clean = page.getByRole("row", { name: /Clean step completion time/ });
  await expect(clean.getByRole("cell").nth(0)).toHaveText("3.00 s");
  await expect(clean.getByRole("cell").nth(1)).toHaveText("4.50 s");
  const checks = page.getByRole("row", { name: /Acceptance checks/ });
  await expect(checks.getByRole("cell").nth(1)).toHaveText("Failed");
  await expect(page.getByText("Synthetic browser-test run 2")).toBeVisible();

  await expect(
    page.getByRole("link", { name: "Export baseline evidence" }),
  ).toHaveAttribute("href", new RegExp(`run_id=${older}$`));
  await expect(
    page.getByRole("link", { name: "Export candidate evidence" }),
  ).toHaveAttribute("href", new RegExp(`run_id=${selected}$`));
});

test("an unknown policy blocks the comparison and no difference is shown", async ({
  page,
  request,
}) => {
  const { older, selected } = await threeRuns(request);
  await page.goto(`/compare?run=${selected}&baseline=${older}`);
  await expect(
    page.getByRole("status").filter({ hasText: "Comparison blocked" }),
  ).toContainText(
    "Concurrency policy: not recorded (baseline) vs not recorded",
  );
  const policy = page.getByRole("row", { name: /Concurrency policy/ });
  await expect(policy.getByRole("cell").nth(2)).toHaveText("Unknown");
  await expect(
    page.getByText(/Saved runs do not record which concurrency policy/),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Recorded measurements (not compared)" }),
  ).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "Observed difference" }),
  ).toHaveCount(0);
  await expect(page.getByText("+1.50 s")).toHaveCount(0);
});

test("a cross-configuration pair lists the setting that differs", async ({
  page,
  request,
}) => {
  const baseline = newSavedRunId();
  const candidate = newSavedRunId();
  await registerSavedRun(request, savedRun(baseline, { sequence: 1 }));
  await registerSavedRun(
    request,
    savedRun(candidate, { sequence: 2, iterations: 8 }),
  );
  await page.goto(`/compare?run=${candidate}&baseline=${baseline}`);
  const row = page.getByRole("row", { name: /Iterations per step/ });
  await expect(row.getByRole("cell")).toHaveText(["4", "8", "Differs"]);
  await expect(
    page.getByRole("status").filter({ hasText: "Comparison blocked" }),
  ).toContainText("Iterations per step: 4 (baseline) vs 8 (candidate)");
  await expect(
    page.getByRole("columnheader", { name: "Relative change" }),
  ).toHaveCount(0);
});

test("a missing or invalid UUID is reported and never replaced by another run", async ({
  page,
  request,
}) => {
  const { older, newest } = await threeRuns(request);
  const missing = newSavedRunId();
  for (const requested of [missing, "not-a-uuid"]) {
    await page.goto(`/compare?run=${requested}&baseline=${older}`);
    await expect(side(page, "Candidate").getByRole("alert")).toContainText(
      `Run ${requested} could not be loaded`,
    );
    await expect(
      side(page, "Candidate").getByText(newest, { exact: true }),
    ).toBeHidden();
    await expect(
      page.getByRole("heading", { name: /measurements/ }),
    ).toHaveCount(0);
    await expect(
      page.getByRole("link", { name: "Export candidate evidence" }),
    ).toHaveCount(0);
  }
});

test("nothing is selected until the operator chooses runs", async ({
  page,
  request,
}) => {
  await threeRuns(request);
  await page.goto("/runs");
  await expect(
    page.getByRole("link", { name: "Compare runs" }).first(),
  ).toHaveAttribute("href", "/compare");
  await page.goto("/compare");
  await expect(page.getByText("No baseline selected.")).toBeVisible();
  await expect(page.getByText("No candidate selected.")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Recorded settings" }),
  ).toHaveCount(0);
});

test("runs are chosen from the keyboard and the other selection is kept", async ({
  page,
  request,
}) => {
  const { older, newest } = await threeRuns(request);
  await page.goto(`/compare?baseline=${older}`);
  const chooser = side(page, "Candidate").getByText("Choose candidate");
  await chooser.focus();
  await page.keyboard.press("Enter");
  const first = side(page, "Candidate").getByRole("link").first();
  await expect(first).toBeVisible();
  await page.keyboard.press("Tab");
  await expect(first).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(
    new RegExp(`/compare\\?run=${newest}&baseline=${older}$`),
  );
  await expect(
    side(page, "Candidate").getByText(newest, { exact: true }),
  ).toBeVisible();
  await expect(
    side(page, "Baseline").getByText(older, { exact: true }),
  ).toBeVisible();
});

test("unavailable history is reported, not shown as an empty comparison", async ({
  page,
  request,
}) => {
  const { older, selected } = await threeRuns(request);
  await takeSavedHistoryDown(request);
  await page.goto(`/compare?run=${selected}&baseline=${older}`);
  await expect(page.getByText(/Run history is unavailable/)).toBeVisible();
  await expect(side(page, "Baseline").getByRole("alert")).toContainText(
    `Run ${older} could not be loaded`,
  );
  await expect(side(page, "Candidate").getByRole("alert")).toContainText(
    `Run ${selected} could not be loaded`,
  );
});
