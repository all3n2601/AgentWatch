import { expect, test, type Page } from "@playwright/test";
import {
  registerSavedRun,
  resetFixtureApi,
  takeSavedHistoryDown,
} from "./fixture-control";
import { newSavedRunId, savedRun } from "./saved-runs";

// The Compare page shows exactly the saved runs named by UUID in the URL, never substitutes
// another run, and calculates no difference between runs whose recorded settings differ.

test.beforeEach(async ({ request }) => resetFixtureApi(request));

type Request = Parameters<typeof registerSavedRun>[0];

async function pair(
  request: Request,
  candidateOptions: Partial<Parameters<typeof savedRun>[1]> = {},
) {
  const [baseline, candidate, newest] = [
    newSavedRunId(),
    newSavedRunId(),
    newSavedRunId(),
  ];
  await registerSavedRun(request, savedRun(baseline, { sequence: 1 }));
  await registerSavedRun(
    request,
    savedRun(candidate, {
      sequence: 2,
      cleanSeconds: 4.5,
      ...candidateOptions,
    }),
  );
  await registerSavedRun(request, savedRun(newest, { sequence: 3 }));
  return { baseline, candidate, newest };
}

const runId = (page: Page, role: "Baseline" | "Candidate", id: string) =>
  page.getByRole("button", { name: `Copy ${role} run ID ${id}` });

/** Any status tone's text or surface colour. */
const TONE_CLASS = /(text|bg|border)-(success|warning|caution|danger|info)/;

const verdict = (page: Page) =>
  page.getByRole("tab", { name: /Configuration/ });

/** The Configuration panel; its scrolling list carries the same name. */
const configuration = (page: Page) =>
  page.getByRole("region", { name: "Configuration check" }).first();

test("a direct URL compares the selected UUIDs with neutral observed differences", async ({
  page,
  request,
}) => {
  const { baseline, candidate } = await pair(request);
  await page.goto(`/compare?run=${candidate}&baseline=${baseline}`);
  await expect(runId(page, "Baseline", baseline)).toBeVisible();
  await expect(runId(page, "Candidate", candidate)).toBeVisible();

  const headline = page
    .getByRole("status")
    .filter({ hasText: "Total step time +1.50 s" });
  await expect(headline).toContainText(
    "Policy not recorded; full compatibility cannot be verified",
  );
  await expect(headline).toContainText("Observed difference, not causal");
  await expect(page.getByText(/faster|slower/i)).toHaveCount(0);
  await expect(page.getByText("Configuration match")).toHaveCount(0);
  await expect(page.getByText("Policy not recorded").first()).toBeVisible();

  // The difference is shown without better-or-worse colouring.
  const clean = page
    .getByRole("listitem")
    .filter({ hasText: "Clean execution" });
  const delta = clean.getByText("+1.50 s");
  await expect(delta).toBeVisible();
  await expect(delta).not.toHaveClass(TONE_CLASS);

  const quality = page.getByRole("region", { name: "Quality and coverage" });
  await expect(quality).toContainText("ThroughputBaseline Not recorded");
  await page.getByText(/Recorded limitations/).click();
  await expect(page.getByText(/no batch throughput/)).toBeVisible();

  await expect(
    page.getByRole("link", { name: "Export baseline evidence" }),
  ).toHaveAttribute("href", new RegExp(`run_id=${baseline}$`));
  await expect(
    page.getByRole("link", { name: "Export candidate evidence" }),
  ).toHaveAttribute("href", new RegExp(`run_id=${candidate}$`));
});

test("unknown policy is listed as unknown, never as a match", async ({
  page,
  request,
}) => {
  const { baseline, candidate } = await pair(request);
  await page.goto(
    `/compare?run=${candidate}&baseline=${baseline}&view=configuration`,
  );
  await expect(verdict(page)).toContainText("1 unknown");
  const panel = configuration(page);
  const policy = panel.getByRole("definition").filter({ hasText: "Unknown" });
  await expect(policy).toHaveCount(1);
  await expect(panel).toContainText("Concurrency policy");
  await expect(panel).toContainText("Baseline Not recorded");
  await expect(
    page.getByText(/saved runs do not record which concurrency policy/),
  ).toBeVisible();
});

for (const { name, options, setting, baselineValue, candidateValue } of [
  {
    name: "a recorded mismatch",
    options: { iterations: 8, competingIterations: 32 },
    setting: "Iterations",
    baselineValue: "4",
    candidateValue: "8",
  },
  {
    name: "differing contention",
    options: { competingIterations: 64 },
    setting: "Competing iterations (contention)",
    baselineValue: "32",
    candidateValue: "64",
  },
]) {
  test(`${name} suppresses deltas, ratios and the headline`, async ({
    page,
    request,
  }) => {
    const { baseline, candidate } = await pair(request, options);
    await page.goto(`/compare?run=${candidate}&baseline=${baseline}`);
    const blocked = page
      .getByRole("alert")
      .filter({ hasText: "Not compared: recorded settings differ" });
    await expect(blocked).toContainText(`Differing settings: ${setting}`);
    await expect(page.getByText(/Total step time [+−±]/)).toHaveCount(0);
    await expect(page.getByText(/[+−±]\d[\d.]* (s|ms|μs)/)).toHaveCount(0);
    await expect(page.getByText(/\d×/)).toHaveCount(0);
    await expect(page.getByText("Not compared").first()).toBeVisible();
    await expect(
      page.getByText("Not calculated: recorded settings differ."),
    ).toBeVisible();

    await page.getByRole("tab", { name: /Steps/ }).click();
    await expect(
      page.getByRole("columnheader", { name: "Duration delta" }),
    ).toHaveCount(0);

    await page.getByRole("tab", { name: /Configuration/ }).click();
    const row = configuration(page)
      .getByRole("definition")
      .filter({
        hasText: `Baseline ${baselineValue}Candidate ${candidateValue}`,
      });
    await expect(row).toHaveCount(1);
  });
}

test("contention missing on one run is unknown, not a match", async ({
  page,
  request,
}) => {
  const { baseline, candidate } = await pair(request, {
    competingIterations: null,
  });
  await page.goto(
    `/compare?run=${candidate}&baseline=${baseline}&view=configuration`,
  );
  await expect(verdict(page)).toContainText("2 unknown");
  await expect(configuration(page)).toContainText("Candidate Not recorded");
});

test("a faster candidate's delta stays neutral, without improvement colouring", async ({
  page,
  request,
}) => {
  // Clean step 3 s against 1.5 s: a ratio below 1 would otherwise be graded "success".
  const { baseline, candidate } = await pair(request, { cleanSeconds: 1.5 });
  await page.goto(`/compare?run=${candidate}&baseline=${baseline}`);
  const clean = page
    .getByRole("listitem")
    .filter({ hasText: "Clean execution" });
  const delta = clean.getByText(/^−1\.50 s/);
  await expect(delta).toBeVisible();
  await expect(delta).toContainText("(0.3×)");
  await expect(delta).not.toHaveClass(TONE_CLASS);
  await expect(page.getByText(/faster|slower|improve/i)).toHaveCount(0);
});

test("a missing or invalid UUID is reported and never replaced by another run", async ({
  page,
  request,
}) => {
  const { baseline, newest } = await pair(request);
  for (const requested of [newSavedRunId(), "not-a-uuid"]) {
    await page.goto(`/compare?run=${requested}&baseline=${baseline}`);
    await expect(
      page.getByRole("status").filter({ hasText: "Candidate run not found" }),
    ).toContainText(requested);
    await expect(runId(page, "Candidate", newest)).toHaveCount(0);
    await expect(page.getByText(/Total step time/)).toHaveCount(0);
    await expect(
      page.getByRole("link", { name: "Export candidate evidence" }),
    ).toHaveCount(0);
  }
});

test("runs are chosen from the keyboard and the other selection is kept", async ({
  page,
  request,
}) => {
  const { baseline, newest } = await pair(request);
  await page.goto(`/compare?baseline=${baseline}`);
  const candidate = page.getByRole("combobox", { name: "Candidate" });
  await candidate.focus();
  await page.keyboard.press("Enter");
  const option = page.getByRole("option", {
    name: new RegExp(newest.slice(0, 8)),
  });
  await expect(option).toBeVisible();
  await option.focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(
    new RegExp(`/compare\\?run=${newest}&baseline=${baseline}$`),
  );
  await expect(runId(page, "Candidate", newest)).toBeVisible();
  await expect(runId(page, "Baseline", baseline)).toBeVisible();
});

test("the Runs selection opens Compare with the same verdict", async ({
  page,
  request,
}) => {
  const { baseline, candidate } = await pair(request, {
    competingIterations: 64,
  });
  await page.goto("/runs");
  for (const id of [baseline, candidate])
    await page
      .getByRole("checkbox", { name: new RegExp(id.slice(0, 8)) })
      .check();
  // Inactive Runs tabs stay mounted, so only the visible note is checked.
  await expect(
    page
      .getByText(/Recorded settings differ: Competing iterations/)
      .filter({ visible: true }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Compare runs" }).click();
  await expect(page).toHaveURL(
    new RegExp(`/compare\\?run=${candidate}&baseline=${baseline}`),
  );
  await expect(
    page
      .getByRole("alert")
      .filter({ hasText: "Not compared: recorded settings differ" }),
  ).toContainText("Competing iterations (contention)");
});

test("unavailable history is reported, not shown as an empty comparison", async ({
  page,
  request,
}) => {
  const { baseline, candidate } = await pair(request);
  await takeSavedHistoryDown(request);
  await page.goto(`/compare?run=${candidate}&baseline=${baseline}`);
  await expect(page.getByText(/could not be loaded/).first()).toBeVisible();
  await expect(page.getByText(/Total step time/)).toHaveCount(0);
});
