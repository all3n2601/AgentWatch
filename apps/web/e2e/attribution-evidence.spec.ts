import { expect, test, type Page } from "@playwright/test";
import {
  advanceAttributions,
  attributionReads,
  registerRun,
  resetFixtureApi,
} from "./fixture-control";
import {
  attributedRun,
  attributionItem,
  attributionRecord,
  attributionState,
  CONFIDENT_CPU,
  CONFIDENT_INFERENCE,
  CONFIDENT_NONE,
  EARLIER_KEY,
  featureRow,
  modelStatus,
  scored,
  UNCERTAIN,
  unscored,
} from "./fixtures/attributions";
import { newRunId } from "./live-runs";

// F03 acceptance: verdict evidence and rule comparison against the scripted live API
// (synthetic attributions only): missing versus zero, uncertainty, mixed evidence and
// compatible comparisons.

test.beforeEach(async ({ request }) => resetFixtureApi(request));

/** ATTRIBUTION_REREAD_INTERVAL_MS in lib/attribution-poll.ts. */
const REREAD_MS = 30_000;

async function openAttribution(page: Page, runId: string) {
  await page.goto(`/live?liveRun=${runId}`);
  await page.getByRole("tab", { name: "Attribution" }).click();
  await expect(
    page.getByRole("heading", { name: "Step attribution" }),
  ).toBeVisible();
}

const table = (page: Page) =>
  page.getByRole("region", { name: "Step attributions table" });
const tableRow = (page: Page, stepId: string) =>
  table(page).locator(`tr[data-step="${stepId}"]`);
const evidence = (page: Page) =>
  page.getByRole("region", { name: "Step evidence" });
/** The value cell of a key-value row in the evidence column. */
const evidenceValue = (page: Page, label: string) =>
  evidence(page)
    .locator("dt", { hasText: new RegExp(`^${label}$`) })
    .locator("xpath=following-sibling::dd[1]");
/** A whole key-value row, including a status pill after the value. */
const evidenceRow = (page: Page, label: string) =>
  evidence(page)
    .locator("dt", { hasText: new RegExp(`^${label}$`) })
    .locator("..");

async function selectStep(page: Page, stepId: string) {
  await table(page).getByRole("button", { name: stepId, exact: true }).click();
  await expect(evidence(page).getByText(stepId, { exact: true })).toBeVisible();
}

test("missing readings show a dash and their state; measured zeros stay zero", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  const zeroes = featureRow(runId, "zero-and-missing", {
    timings: {
      queue_seconds: 0,
      execution_seconds: 1.2,
      duration_seconds: 1.2,
      cpu_seconds: null,
    },
    coverage: null,
    throttledUsec: 0,
  });
  const noTelemetry = featureRow(runId, "no-telemetry", {
    status: "missing",
    sampleCount: 0,
    coverage: null,
    pressure: null,
    throttledUsec: null,
  });
  const broken = featureRow(runId, "broken-row");
  (broken.timings as Record<string, unknown>).queue_seconds = "soon";
  await registerRun(
    request,
    attributedRun(runId, [
      attributionState(runId, [
        attributionItem(
          1,
          attributionRecord(runId, "zero-and-missing", {
            feature: zeroes,
            verdict: "cpu_sandbox",
            model: scored(CONFIDENT_CPU, { excess: 0 }),
            missingInputs: [
              "workload_iterations",
              "tests_per_step",
              "worker_capacity",
              "telemetry_coverage",
            ],
          }),
        ),
        attributionItem(
          2,
          attributionRecord(runId, "no-telemetry", {
            feature: noTelemetry,
            model: unscored("unavailable", { reason: "prediction_failed" }),
          }),
        ),
        attributionItem(
          3,
          attributionRecord(runId, "broken-row", { feature: broken }),
        ),
      ]),
    ]),
  );
  await openAttribution(page, runId);

  // Estimated excess: a measured-zero estimate reads ~0.0%, a missing one a dash.
  await expect(tableRow(page, "zero-and-missing")).toContainText("~0.0%");
  await expect(tableRow(page, "no-telemetry").locator("td").nth(3)).toHaveText(
    /^—/,
  );

  await selectStep(page, "zero-and-missing");
  await expect(evidenceValue(page, "Queue")).toContainText("0 ms");
  const cpu = evidenceValue(page, "Process CPU time");
  await expect(cpu).toContainText("—");
  await expect(cpu).toContainText("Unavailable");
  await expect(cpu).not.toContainText("0");
  await expect(evidenceValue(page, "Coverage")).toContainText("Unavailable");
  await expect(evidenceValue(page, "cgroup throttled time")).toContainText(
    "0 ms",
  );
  await expect(
    evidence(page).getByRole("list", { name: "Missing model inputs" }),
  ).toContainText("telemetry_coverage");

  await selectStep(page, "no-telemetry");
  await expect(evidenceValue(page, "Telemetry")).toHaveText(
    "No telemetry for the step",
  );
  await expect(
    evidenceValue(page, "Max CPU pressure \\(some avg10\\)"),
  ).toContainText("Unavailable");
  await expect(evidence(page)).toContainText(
    "The served model rejected this step's values.",
  );

  // A malformed row is marked and explained; it never stops the other rows rendering.
  await expect(tableRow(page, "broken-row")).toContainText("Malformed");
  await expect(tableRow(page, "zero-and-missing")).toContainText("CPU sandbox");
  await selectStep(page, "broken-row");
  await expect(
    evidence(page).getByRole("list", { name: "Validation problems" }),
  ).toContainText("queue_seconds");
});

test("an abstention is shown as uncertain with its confidence and threshold, never as none", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  await registerRun(
    request,
    attributedRun(runId, [
      attributionState(runId, [
        attributionItem(
          1,
          attributionRecord(runId, "uncertain-step", {
            model: scored(UNCERTAIN),
          }),
        ),
        attributionItem(
          2,
          attributionRecord(runId, "confident-step", {
            model: scored(CONFIDENT_NONE),
          }),
        ),
      ]),
    ]),
  );
  await openAttribution(page, runId);

  await expect(page.getByText("Serving, not promoted")).toBeVisible();
  await expect(
    page.getByText("Not promoted", { exact: true }).first(),
  ).toBeVisible();
  const uncertainModel = tableRow(page, "uncertain-step").locator("td").nth(1);
  await expect(uncertainModel).toHaveText("Uncertain");
  await expect(
    tableRow(page, "uncertain-step").locator("td").nth(2),
  ).toHaveText("0.580");

  await selectStep(page, "uncertain-step");
  await expect(evidence(page)).toContainText(
    "Confidence 0.580 is below the abstention threshold 0.647, so no resource is attributed.",
  );
  const probabilities = evidence(page).getByRole("list", {
    name: "Calibrated probabilities",
  });
  await expect(probabilities.getByRole("listitem").first()).toHaveText(
    /CPU sandbox\s*0\.580/,
  );
  await expect(evidenceValue(page, "Model")).toHaveText(
    "Uncertain (abstained)",
  );
  await expect(evidence(page)).toContainText(
    "rules-v1 only detects CPU sandbox contention",
  );

  await selectStep(page, "confident-step");
  await expect(evidence(page)).toContainText(
    "No contended resource identified: calibrated confidence 0.900",
  );
});

const unavailableCases = [
  {
    name: "no model configured",
    model: modelStatus("not_configured"),
    row: unscored("unavailable", {
      state: "not_configured",
      reason: "not_configured",
    }),
    stepState: "completed",
    state: "No model configured",
    detail: "No model pair is configured",
  },
  {
    name: "a model that failed a load check",
    model: modelStatus("load_failed", { reason: "hash_mismatch" }),
    row: unscored("unavailable", {
      state: "load_failed",
      reason: "hash_mismatch",
    }),
    stepState: "completed",
    state: "Model failed to load",
    detail: "failed a load check (hash_mismatch)",
  },
  {
    name: "a refused unpromoted model",
    model: modelStatus("not_promoted_refused"),
    row: unscored("unavailable", {
      state: "not_promoted_refused",
      reason: "not_promoted_refused",
    }),
    stepState: "completed",
    state: "Model refused: not promoted",
    detail: "not promoted and serving it is not allowed",
  },
  {
    name: "a failed step",
    model: modelStatus(),
    row: unscored("not_scored_step_state"),
    stepState: "failed",
    state: "Serving, not promoted",
    detail: "the model scores only completed steps",
  },
];

for (const scenario of unavailableCases)
  test(`with ${scenario.name}, the rules verdict is shown alone with the reason`, async ({
    page,
    request,
  }) => {
    const runId = newRunId();
    await registerRun(
      request,
      attributedRun(runId, [
        attributionState(
          runId,
          [
            attributionItem(
              1,
              attributionRecord(runId, "rules-only", {
                verdict: "cpu_sandbox",
                model: scenario.row,
                stepState: scenario.stepState,
              }),
            ),
          ],
          { model: scenario.model },
        ),
      ]),
    );
    await openAttribution(page, runId);
    await expect(page.getByText(scenario.state, { exact: true })).toBeVisible();
    await expect(tableRow(page, "rules-only").locator("td").first()).toHaveText(
      "CPU sandbox",
    );
    await expect(tableRow(page, "rules-only").locator("td").nth(2)).toHaveText(
      "—",
    );
    await selectStep(page, "rules-only");
    await expect(evidence(page)).toContainText(scenario.detail);
    await expect(evidenceRow(page, "Comparison")).toContainText("Rules only");
  });

test("mixed evidence: differing verdicts and evidence changed after serving, re-read only while the tab is open", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  const rows = (changed: boolean) => [
    attributionItem(
      1,
      attributionRecord(runId, "differ-step", {
        verdict: "cpu_sandbox",
        model: scored(CONFIDENT_INFERENCE),
      }),
    ),
    attributionItem(
      2,
      attributionRecord(runId, "late-sample-step", {
        verdict: "cpu_sandbox",
        model: scored(CONFIDENT_CPU),
      }),
      { evidenceChanged: changed },
    ),
  ];
  await registerRun(
    request,
    attributedRun(runId, [
      attributionState(runId, rows(false)),
      attributionState(runId, rows(true)),
    ]),
  );
  await page.clock.install();
  await openAttribution(page, runId);

  await expect(tableRow(page, "differ-step")).toContainText(
    "Rules and model differ",
  );
  await expect(tableRow(page, "late-sample-step")).not.toContainText(
    "Evidence changed",
  );
  await selectStep(page, "differ-step");
  await expect(evidence(page)).toContainText(
    "The rules name CPU sandbox and the model names Inference for the same feature row.",
  );
  // Both probabilities are given as text, not as a threshold verdict.
  await expect(
    evidence(page).getByRole("list", { name: "Calibrated probabilities" }),
  ).toContainText("0.720");

  // The flag turns true after the row was served; only a re-read from after=0 shows it.
  await advanceAttributions(request, runId);
  await page.clock.runFor(REREAD_MS + 2_000);
  await expect(tableRow(page, "late-sample-step")).toContainText(
    "Evidence changed",
  );
  const reads = await attributionReads(request, runId);
  expect(reads[0]).toBe(0);
  expect(reads.slice(1)).toContain(2);
  expect(reads.lastIndexOf(0)).toBeGreaterThan(0);
  await selectStep(page, "late-sample-step");
  await expect(evidence(page)).toContainText(
    "a later audit computed a different feature row",
  );

  // Closing the tab stops attribution polling, the slow re-read included.
  await page.getByRole("tab", { name: "Status" }).click();
  const closedAt = (await attributionReads(request, runId)).length;
  await page.clock.runFor(2 * REREAD_MS);
  expect((await attributionReads(request, runId)).length).toBe(closedAt);
});

test("comparisons are counted per serving key, and earlier-key rows keep their own versions", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  const earlier = scored(CONFIDENT_CPU, {
    trainingVersion: "training-v1+synthetic",
  });
  await registerRun(
    request,
    attributedRun(runId, [
      attributionState(runId, [
        attributionItem(
          1,
          attributionRecord(runId, "old-a", {
            verdict: "cpu_sandbox",
            model: earlier,
          }),
          { servingKey: EARLIER_KEY },
        ),
        attributionItem(
          2,
          attributionRecord(runId, "old-b", {
            verdict: "none",
            model: earlier,
          }),
          { servingKey: EARLIER_KEY },
        ),
        attributionItem(
          3,
          attributionRecord(runId, "new-agree", {
            verdict: "cpu_sandbox",
            model: scored(CONFIDENT_CPU),
          }),
        ),
        attributionItem(
          4,
          attributionRecord(runId, "new-differ", {
            verdict: "cpu_sandbox",
            model: scored(CONFIDENT_INFERENCE),
          }),
        ),
        attributionItem(
          5,
          attributionRecord(runId, "new-uncertain", {
            model: scored(UNCERTAIN),
          }),
        ),
        attributionItem(
          6,
          attributionRecord(runId, "new-cancelled", {
            model: unscored("not_scored_step_state"),
            stepState: "cancelled",
          }),
        ),
        attributionItem(
          7,
          attributionRecord(runId, "new-odd", {
            verdict: "gpu_memory",
            model: scored(CONFIDENT_CPU),
          }),
        ),
      ]),
    ]),
  );
  await openAttribution(page, runId);

  const summary = page.getByRole("list", {
    name: "Comparison by serving configuration",
  });
  const groups = summary.getByRole("listitem");
  await expect(groups).toHaveCount(2);
  await expect(groups.first()).toContainText(
    "Current serving configuration · 5 steps",
  );
  await expect(groups.first()).toContainText(
    "2 compared: 1 name the same resource, 1 differ. 1 model uncertain, 1 rules only, 1 unrecognised verdict.",
  );
  await expect(groups.nth(1)).toContainText(
    `Earlier serving key ${EARLIER_KEY.slice(0, 12)} · 2 steps`,
  );
  await expect(groups.nth(1)).toContainText("training training-v1+synthetic");
  await expect(groups.nth(1)).toContainText(
    "2 compared: 1 name the same resource, 1 differ.",
  );
  await expect(
    page.getByText("counts are kept per key and never combined"),
  ).toBeVisible();

  await expect(tableRow(page, "new-odd").locator("td").first()).toHaveText(
    "Unrecognised: gpu_memory",
  );
  await expect(tableRow(page, "old-a")).toContainText("Earlier serving key");
  await expect(tableRow(page, "new-agree")).not.toContainText(
    "Earlier serving key",
  );

  await selectStep(page, "old-a");
  await expect(evidenceValue(page, "Model training")).toHaveText(
    "training-v1+synthetic",
  );
  await expect(evidenceRow(page, "Serving key")).toContainText("Earlier");
  await selectStep(page, "new-agree");
  await expect(evidenceValue(page, "Model training")).toHaveText(
    "training-v2+synthetic",
  );
  await expect(evidenceRow(page, "Serving key")).toContainText("Current");
});

test("every page is drained before the rows are shown", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  const rows = Array.from({ length: 300 }, (_, index) =>
    attributionItem(index + 1, attributionRecord(runId, `step-${index + 1}`)),
  );
  await registerRun(
    request,
    attributedRun(runId, [attributionState(runId, rows)], {
      state: "completed",
    }),
  );
  await openAttribution(page, runId);
  await expect(
    page.getByText("300 attributed steps from the live API."),
  ).toBeVisible();
  await expect(tableRow(page, "step-300")).toBeVisible();
  expect((await attributionReads(request, runId)).slice(0, 2)).toEqual([
    0, 256,
  ]);
});

test("an unknown run's attributions are not found, and no attributed steps is not an unavailable model", async ({
  page,
  request,
}) => {
  const missing = newRunId();
  await registerRun(
    request,
    attributedRun(missing, [
      { status: 404, body: { detail: "Live run not found" } },
    ]),
  );
  await openAttribution(page, missing);
  await expect(page.getByText("No attributions for this run")).toBeVisible();
  await expect(table(page)).toHaveCount(0);

  const empty = newRunId();
  await registerRun(
    request,
    attributedRun(empty, [
      attributionState(empty, [], { model: modelStatus("not_configured") }),
    ]),
  );
  await openAttribution(page, empty);
  await expect(page.getByText("No attributed steps yet")).toBeVisible();
  await expect(
    page.getByText("No model configured", { exact: true }),
  ).toBeVisible();
});

test("a live run's page has the same Attribution tab", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  await registerRun(
    request,
    attributedRun(runId, [
      attributionState(runId, [
        attributionItem(
          1,
          attributionRecord(runId, "run-page-step", {
            model: scored(UNCERTAIN),
          }),
        ),
      ]),
    ]),
  );
  await page.goto(`/runs/${runId}?tab=attribution`);
  await expect(
    page.getByRole("heading", { name: "Step attribution" }),
  ).toBeVisible();
  await expect(tableRow(page, "run-page-step").locator("td").nth(1)).toHaveText(
    "Uncertain",
  );
});

// 720 px is a 1440 px window at 200% browser zoom.
for (const width of [390, 720, 768, 1440])
  test(`the Attribution tab has no page-level horizontal overflow at ${width} px`, async ({
    page,
    request,
  }) => {
    await page.setViewportSize({ width, height: 900 });
    const runId = newRunId();
    await registerRun(
      request,
      attributedRun(runId, [
        attributionState(runId, [
          attributionItem(
            1,
            attributionRecord(runId, "a-rather-long-step-identifier-0001", {
              verdict: "cpu_sandbox",
              model: scored(CONFIDENT_INFERENCE),
            }),
          ),
        ]),
      ]),
    );
    await openAttribution(page, runId);
    await expect(
      tableRow(page, "a-rather-long-step-identifier-0001"),
    ).toBeVisible();
    const overflow = await page.evaluate(
      () =>
        document.documentElement.scrollWidth -
        document.documentElement.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(0);
  });
