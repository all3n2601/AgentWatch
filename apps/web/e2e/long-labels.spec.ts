import { expect, test } from "@playwright/test";
import { clippedLabels, servePolicy } from "./audit";
import {
  registerRun,
  registerSavedRun,
  resetFixtureApi,
} from "./fixture-control";
import { fixtureRun, newRunId, snapshot, stepEvent } from "./live-runs";
import { newSavedRunId, savedRun } from "./saved-runs";
import { TASK_ID } from "../tests/live-fixtures";

// F04 phase 2: the longest labels a run can carry wrap or offer a copy control at phone and
// desktop widths. Live task and step ids use the live contract's maximum of 128 characters
// (api/openapi/live.json); saved workloads and models have no limit, so they use 200.
// Synthetic fixture data only.

const LONG_TASK = `long-task-${"t".repeat(118)}`;
const LONG_STEP = `long-step-${"s".repeat(118)}`;
const LONG_MODEL = `long-model-${"m".repeat(189)}`;
const LONG_WORKLOAD = `long workload ${"w".repeat(186)}`;

test.beforeEach(async ({ page, request }) => {
  await resetFixtureApi(request);
  await servePolicy(page);
});

/** Replaces the fixture task id everywhere in a live fixture. */
const withTask = <T>(value: T): T =>
  JSON.parse(
    JSON.stringify(value).replaceAll(
      JSON.stringify(TASK_ID),
      JSON.stringify(LONG_TASK),
    ),
  );

async function seedLongLabels(request: Parameters<typeof registerRun>[0]) {
  const live = newRunId();
  const queued = stepEvent(LONG_STEP, 0, "queued", { runId: live });
  const running = stepEvent(LONG_STEP, 1, "running", {
    runId: live,
    started: 1,
  });
  await registerRun(
    request,
    withTask(
      fixtureRun(live, [
        snapshot(live, { steps: [running], records: [queued, running] }),
      ]),
    ),
  );
  const [baseline, candidate] = [newSavedRunId(), newSavedRunId()];
  for (const [id, sequence] of [
    [baseline, 1],
    [candidate, 2],
  ] as const) {
    const saved = savedRun(id, { sequence, model: LONG_MODEL });
    saved.result.experiment = {
      ...(saved.result.experiment as Record<string, unknown>),
      workload: LONG_WORKLOAD,
    };
    await registerSavedRun(request, saved);
  }
  return { live, baseline, candidate };
}

/**
 * Clipping in shared components, which F04 does not change; each is listed under "Findings
 * for the team" in the F04 phase 2 PR, with where the full value can still be read.
 */
const SHARED_CLIPPING = {
  // components/widgets/timeline-chart.tsx truncates row labels; the selected step's panel
  // shows the full step id (checked below).
  timeline: "[aria-label='Step timeline'] li",
  // components/ui/filter-select.tsx keeps the trigger to one line and components/ui/select.tsx
  // clips long one-word option text; the run id has its own copy control beside the picker.
  picker: "[role=combobox]",
};

for (const viewport of [
  { width: 390, height: 844 },
  { width: 1440, height: 900 },
])
  test(`long task, step, model and workload labels wrap or can be copied at ${viewport.width} px`, async ({
    page,
    request,
  }) => {
    await page.setViewportSize(viewport);
    const { live, baseline, candidate } = await seedLongLabels(request);
    const pages: [string, string[], string[]][] = [
      ["/runs", [LONG_TASK, LONG_MODEL], []],
      [`/runs?preview=saved:${candidate}`, [LONG_WORKLOAD], []],
      [`/live?liveRun=${live}`, [LONG_TASK], []],
      [
        `/live?liveRun=${live}&tab=timeline`,
        [LONG_STEP],
        [SHARED_CLIPPING.timeline],
      ],
      [`/runs/${live}`, [LONG_TASK], []],
      [`/runs/${live}?tab=timeline`, [LONG_STEP], [SHARED_CLIPPING.timeline]],
      [`/runs/${candidate}`, [LONG_MODEL], []],
      [`/runs/${candidate}?tab=evidence`, [LONG_WORKLOAD], []],
      [
        `/compare?run=${candidate}&baseline=${baseline}`,
        [LONG_MODEL],
        [SHARED_CLIPPING.picker],
      ],
    ];
    const problems: string[] = [];
    for (const [path, labels, exempt] of pages) {
      await page.goto(path);
      await expect(
        page.getByText(labels[0].slice(0, 24)).first(),
      ).toBeAttached();
      for (const label of labels)
        problems.push(
          ...(await clippedLabels(page, label, exempt)).map(
            (problem) => `${path}: ${problem}`,
          ),
        );
    }
    expect(problems).toEqual([]);

    // Where the timeline clips a step id, the selected step's panel shows all of it.
    await page.goto(`/runs/${live}?tab=timeline`);
    await expect(
      page
        .getByRole("region", { name: "Selected step" })
        .getByText(LONG_STEP, { exact: true }),
    ).toBeVisible();
  });
