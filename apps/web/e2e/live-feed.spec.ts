import { readFile } from "node:fs/promises";
import { expect, test, type Page } from "@playwright/test";
import {
  advanceRun,
  registerRun,
  resetFixtureApi,
  runReads,
} from "./fixture-control";
import {
  fixtureRun,
  newRunId,
  resourceSample,
  snapshot,
  stepEvent,
} from "./live-runs";

// F02 acceptance: the live view against the scripted live API (synthetic data only).

test.beforeEach(async ({ request }) => resetFixtureApi(request));

const POLL_MS = 2_000;
const connection = (page: Page) =>
  page.getByRole("status").filter({ hasText: "Live API connection" });
const feedTile = (page: Page) =>
  page.getByRole("status").filter({ hasText: "Feed:" });
const tile = (page: Page, label: string) =>
  page.getByText(label, { exact: true }).locator("..");

function runningTool(runId: string) {
  const queued = stepEvent("tool-1", 0, "queued", { runId });
  const running = stepEvent("tool-1", 1, "running", { runId, started: 1 });
  const completed = stepEvent("tool-1", 2, "completed", {
    runId,
    started: 1,
    finished: 3,
    cpu: 0.5,
  });
  return { queued, running, completed };
}

test("a run moves from active to terminal and polling stops once it closes", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  const { queued, running, completed } = runningTool(runId);
  await registerRun(
    request,
    fixtureRun(runId, [
      snapshot(runId, { steps: [running], records: [queued, running] }),
      snapshot(runId, {
        state: "completed",
        steps: [completed],
        records: [queued, running, completed],
      }),
    ]),
  );
  await page.clock.install();
  await page.goto(`/live?liveRun=${runId}`);
  await expect(connection(page)).toContainText("Connected");
  await expect(page.getByText("Partial step totals")).toBeVisible();
  await expect(
    page.getByText("1 active · 0 failed or cancelled"),
  ).toBeVisible();

  await advanceRun(request, runId);
  await page.clock.runFor(POLL_MS);
  await expect(page.getByText("Final step totals")).toBeVisible();
  await expect(connection(page)).toContainText("Polling stopped");
  await expect(page.getByText("3 records received for run")).toBeVisible();

  const readsWhenClosed = (await runReads(request, runId)).length;
  await page.clock.runFor(10 * POLL_MS);
  expect((await runReads(request, runId)).length).toBe(readsWhenClosed);

  // Late evidence can still be checked for explicitly.
  await page.getByRole("button", { name: "Check for late evidence" }).click();
  await expect
    .poll(async () => (await runReads(request, runId)).length)
    .toBe(readsWhenClosed + 1);
});

test("a pending completion applied later without redelivery ends the step on the timeline", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  const { queued, running, completed } = runningTool(runId);
  const timelineRow = page
    .getByRole("list", { name: "Step timeline" })
    .getByRole("listitem")
    .filter({ hasText: "tool-1" });
  await registerRun(
    request,
    fixtureRun(runId, [
      // The completion arrives before the running revision and waits as pending.
      snapshot(runId, {
        steps: [queued],
        records: [queued, [completed, "pending"]],
        gaps: [
          {
            step_id: "tool-1",
            missing_sequence: 1,
            first_pending_sequence: 2,
            expired: false,
          },
        ],
      }),
      // The running revision fills the gap and the service applies the completion, but reads
      // after record 2 never return it again: only the snapshot's step state shows the change.
      snapshot(runId, {
        state: "completed",
        steps: [completed],
        records: [queued, completed, running],
      }),
    ]),
  );
  await page.clock.install();
  await page.goto(`/live?liveRun=${runId}`);
  await expect(connection(page)).toContainText("Connected");
  await expect(timelineRow).toContainText("in progress");

  await advanceRun(request, runId);
  await page.clock.runFor(POLL_MS);
  await expect(page.getByText("Final step totals")).toBeVisible();
  await expect(timelineRow).not.toContainText("in progress");
  await expect(timelineRow).toContainText("completed");
});

test("a dropped connection retries with backoff, keeps the last data and recovers", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  const { queued, running } = runningTool(runId);
  const sample = resourceSample("s-1", { runId, second: 2 });
  await registerRun(
    request,
    fixtureRun(runId, [
      snapshot(runId, { steps: [running], records: [queued, running] }),
      { status: 503, body: { detail: "Database unavailable" } },
      snapshot(runId, {
        steps: [running],
        records: [queued, running, sample],
      }),
    ]),
  );
  await page.clock.install();
  await page.goto(`/live?liveRun=${runId}`);
  await expect(connection(page)).toContainText("Connected");

  await advanceRun(request, runId);
  await page.clock.runFor(POLL_MS);
  await expect(connection(page)).toContainText("Reconnecting (attempt 1)");
  await expect(connection(page)).toContainText("HTTP 503");
  // The last valid data and the selection stay on screen while retrying.
  await expect(page.getByText("2 records received for run")).toContainText(
    runId,
  );
  await expect(
    page.getByText("1 active · 0 failed or cancelled"),
  ).toBeVisible();

  await page.clock.runFor(1_300);
  await expect(connection(page)).toContainText("Reconnecting (attempt 2)");

  await advanceRun(request, runId);
  await page.getByRole("button", { name: "Retry now" }).click();
  await expect(connection(page)).toContainText("Connected");
  await expect(page.getByText("3 records received for run")).toBeVisible();
  // Every read after the first continued from the last record received.
  const reads = await runReads(request, runId);
  expect(reads[0]).toBe(0);
  expect(reads.slice(1).every((after) => after === 2)).toBe(true);
});

test("a failed run shows its reason and the steps it left unfinished", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  const { queued, running } = runningTool(runId);
  await registerRun(
    request,
    fixtureRun(
      runId,
      [
        snapshot(runId, {
          state: "failed",
          reason: "Run exceeded its 900 s time limit",
          steps: [running],
          records: [queued, running],
          unfinished: ["tool-1"],
        }),
      ],
      { state: "failed" },
    ),
  );
  await page.goto(`/live?liveRun=${runId}`);
  await expect(tile(page, "Run state")).toContainText("Failed");
  await expect(
    page.getByText("Run exceeded its 900 s time limit").first(),
  ).toBeVisible();
  await expect(
    page.getByText("1 step is still queued or running"),
  ).toBeVisible();
  await expect(page.getByText("Partial step totals")).toBeVisible();
});

test("an unknown run is reported as not found and no other run is shown", async ({
  page,
  request,
}) => {
  const other = newRunId();
  await registerRun(request, fixtureRun(other, [snapshot(other, {})]));
  const missing = newRunId();
  await page.goto(`/live?liveRun=${missing}`);
  const notFound = page
    .getByRole("alert")
    .filter({ hasText: "Live run not found" });
  await expect(notFound).toContainText(missing);
  await expect(connection(page)).toContainText("Stopped");
  await expect(page.getByText("records received for run")).toHaveCount(0);
});

test("a value that is not a UUID is rejected without reading any run", async ({
  page,
}) => {
  await page.goto("/live?liveRun=../../v1/runs");
  await expect(
    page.getByRole("alert").filter({ hasText: "Not a live run ID" }),
  ).toBeVisible();
  await expect(connection(page)).toHaveCount(0);
});

test("an off-contract response is retried and never rendered", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  const badSample = resourceSample("s-1", { runId, availability: "zero" });
  await registerRun(
    request,
    fixtureRun(runId, [snapshot(runId, { records: [badSample] })]),
  );
  await page.goto(`/live?liveRun=${runId}`);
  await expect(connection(page)).toContainText("Reconnecting");
  await expect(connection(page)).toContainText(
    "does not match live-contracts-v1",
  );
  await expect(page.getByText("records received for run")).toHaveCount(0);
});

test("unavailable samples are shown as gaps with their state, never as zero", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  const { queued, running, completed } = runningTool(runId);
  const PSI = "cpu.psi.some.avg10.v1";
  const unsupported = (id: string, resourceId: string, second: number) => ({
    ...resourceSample(id, {
      runId,
      resourceId,
      serviceId: `${resourceId}-service`,
      metric: PSI,
      value: null,
      availability: "unsupported",
      second,
    }),
    reason: "PSI is not available on this host",
  });
  await registerRun(
    request,
    fixtureRun(runId, [
      snapshot(runId, {
        state: "completed",
        steps: [completed],
        records: [
          queued,
          running,
          unsupported("s-1", "host-a", 1),
          unsupported("s-2", "host-a", 2),
          resourceSample("s-3", {
            runId,
            resourceId: "host-b",
            serviceId: "host-b-service",
            metric: PSI,
            value: null,
            availability: "missing",
            second: 2,
          }),
          completed,
        ],
      }),
    ]),
  );
  await page.goto(`/live?liveRun=${runId}`);
  const chartA = page.getByRole("group", { name: /CPU pressure.*host-a/ });
  await expect(chartA).toContainText("0 measured · 2 unsupported");
  await expect(chartA).toContainText(
    "Not measured (unsupported): PSI is not available on this host. Shown as gaps, never as zero.",
  );
  await expect(chartA).toContainText("No measured values.");
  const chartB = page.getByRole("group", { name: /CPU pressure.*host-b/ });
  await expect(chartB).toContainText("0 measured · 1 missing");
});

test("exports contain the selected UUID only, even after switching runs mid-request", async ({
  page,
  request,
}) => {
  const first = newRunId();
  const second = newRunId();
  const a = runningTool(first);
  await registerRun(
    request,
    fixtureRun(
      first,
      [
        snapshot(first, { steps: [a.running], records: [a.queued, a.running] }),
        {
          ...snapshot(first, {
            steps: [a.completed],
            records: [a.queued, a.running, a.completed],
          }),
          delayMs: 1_500,
        },
      ],
      { sequence: 1 },
    ),
  );
  // More than one export page (256 records), with a pending revision and its gap.
  const b = runningTool(second);
  const samples = Array.from({ length: 300 }, (_, i) =>
    resourceSample(`b-${i}`, { runId: second, second: i }),
  );
  const gap = {
    step_id: "tool-1",
    missing_sequence: 1,
    first_pending_sequence: 2,
    expired: false,
  };
  await registerRun(
    request,
    fixtureRun(
      second,
      [
        snapshot(second, {
          steps: [b.queued],
          records: [b.queued, [b.completed, "pending"], ...samples],
          gaps: [gap],
        }),
      ],
      { sequence: 2 },
    ),
  );

  await page.clock.install();
  await page.goto(`/live?liveRun=${first}`);
  await expect(page.getByText("2 records received for run")).toContainText(
    first,
  );

  // Start a slow read of the first run, then switch to the second while it is in flight.
  await advanceRun(request, first);
  await page.clock.runFor(POLL_MS);
  await page
    .getByRole("navigation", { name: "Recorded live runs" })
    .getByRole("link", { name: new RegExp(second.slice(0, 8)) })
    .click();
  await expect(page).toHaveURL(new RegExp(`liveRun=${second}`));
  await expect(page.getByText("302 records received for run")).toContainText(
    second,
  );
  await expect
    .poll(async () => (await runReads(request, first)).length)
    .toBe(2);
  // The delayed response for the first run was delivered but never applied.
  await expect(page.getByText("records received for run")).toContainText(
    second,
  );
  await expect(page.getByText(first)).toHaveCount(0);

  await expect(page.getByText("Saved evidence")).toBeVisible();
  await expect(page.getByText("Not available for live runs.")).toBeVisible();
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("link", { name: "Download live records (JSON)" }).click(),
  ]);
  expect(download.suggestedFilename()).toBe(
    `agentwatch-live-records-${second}.json`,
  );
  const exported = JSON.parse(await readFile(await download.path(), "utf8"));
  expect(exported.export_kind).toBe("agentwatch-live-records-v1");
  expect(exported.run_id).toBe(second);
  expect(exported.description).toContain("not saved evidence");
  expect(exported.pages_read).toBe(2);
  expect(exported.records).toHaveLength(302);
  expect(
    exported.records.every(
      (record: { payload: { run_id: string } }) =>
        record.payload.run_id === second,
    ),
  ).toBe(true);
  expect(exported.records[1].disposition).toBe("pending");
  expect(exported.run.gaps).toEqual([gap]);
  expect(Date.parse(exported.read_finished_at)).not.toBeNaN();
});

test("a run older than the newest 50 is read by UUID; the default is the newest", async ({
  page,
  request,
}) => {
  const runIds = Array.from({ length: 51 }, () => newRunId());
  for (const [index, runId] of runIds.entries()) {
    const queued = stepEvent("tool-1", 0, "queued", { runId });
    await registerRun(
      request,
      fixtureRun(
        runId,
        [snapshot(runId, { steps: [queued], records: [queued] })],
        {
          sequence: index + 1,
        },
      ),
    );
  }
  const [oldest] = runIds;
  const newest = runIds[runIds.length - 1];
  const picker = page.getByRole("navigation", { name: "Recorded live runs" });

  await page.goto(`/live?liveRun=${oldest}`);
  await expect(picker.getByRole("link")).toHaveCount(50);
  await expect(
    picker.getByRole("link", { name: new RegExp(oldest.slice(0, 8)) }),
  ).toHaveCount(0);
  await expect(page.getByText("1 record received for run")).toContainText(
    oldest,
  );

  await page.goto("/live");
  await expect(page.getByText("1 record received for run")).toContainText(
    newest,
  );
  await expect(picker.locator('[aria-current="page"]')).toContainText(
    newest.slice(0, 8),
  );
});

for (const skewMs of [-60_000, 60_000])
  test(`feed age uses the browser's clock when the server clock is ${skewMs / 1000} s off`, async ({
    page,
    request,
  }) => {
    const browserNow = Date.UTC(2026, 9, 24, 15, 0, 0);
    const runId = newRunId();
    const { queued, running } = runningTool(runId);
    const states = [
      snapshot(runId, { steps: [running], records: [queued, running] }),
      snapshot(runId, {
        steps: [running],
        records: [queued, running, resourceSample("s-1", { runId })],
      }),
    ];
    for (const state of states)
      if ("snapshot" in state)
        for (const record of state.snapshot.records)
          record.received_at = new Date(browserNow + skewMs).toISOString();
    await registerRun(request, fixtureRun(runId, states));

    await page.clock.install({ time: browserNow });
    await page.goto(`/live?liveRun=${runId}`);
    await expect(feedTile(page)).toContainText("Receiving");
    await expect(tile(page, "Last record received")).toContainText("Just now");

    // No new records for longer than the stale threshold: stale, whatever received_at says.
    await page.clock.runFor(31_000);
    await expect(feedTile(page)).toContainText("Stale");

    // A new record arrives: its age is anchored again on this browser's clock.
    await advanceRun(request, runId);
    await page.clock.runFor(POLL_MS);
    await expect(page.getByText("3 records received for run")).toBeVisible();
    await expect(feedTile(page)).toContainText("Receiving");
    await expect(tile(page, "Last record received")).toContainText("Just now");
  });

test.describe("checking a completed run for late evidence", () => {
  const lateCheck = (page: Page) =>
    page.getByRole("status", { name: "Late evidence check" });
  const checkButton = (page: Page) =>
    page.getByRole("button", { name: /Check for late evidence|Checking/ });

  function completedRun(runId: string, extra: unknown[] = []) {
    const { queued, running, completed } = runningTool(runId);
    return snapshot(runId, {
      state: "completed",
      steps: [completed],
      records: [queued, running, completed, ...(extra as (typeof queued)[])],
    });
  }

  test("shows that it is checking, then that no new evidence arrived", async ({
    page,
    request,
  }) => {
    const runId = newRunId();
    const slowRepeat = { ...completedRun(runId), delayMs: 1_500 };
    await registerRun(
      request,
      fixtureRun(runId, [completedRun(runId), slowRepeat], {
        state: "completed",
      }),
    );
    await page.goto(`/live?liveRun=${runId}`);
    await expect(connection(page)).toContainText("Polling stopped");
    await expect(lateCheck(page)).toBeEmpty();

    await advanceRun(request, runId);
    await checkButton(page).click();
    await expect(checkButton(page)).toHaveText("Checking…");
    await expect(checkButton(page)).toBeDisabled();
    await expect(lateCheck(page)).toHaveText(
      "Checking the live API for late evidence…",
    );

    await expect(lateCheck(page)).toHaveText(
      /^Checked at .+: no new evidence\.$/,
    );
    await expect(checkButton(page)).toHaveText("Check for late evidence");
    await expect(checkButton(page)).toBeEnabled();
    await expect(page.getByText("3 records received for run")).toBeVisible();
  });

  test("reports how many new records the check found", async ({
    page,
    request,
  }) => {
    const runId = newRunId();
    const lateSample = resourceSample("late-1", { runId, second: 9 });
    await registerRun(
      request,
      fixtureRun(
        runId,
        [completedRun(runId), completedRun(runId, [lateSample])],
        { state: "completed" },
      ),
    );
    await page.goto(`/live?liveRun=${runId}`);
    await expect(connection(page)).toContainText("Polling stopped");

    await advanceRun(request, runId);
    await checkButton(page).click();
    await expect(lateCheck(page)).toHaveText(
      /^Checked at .+: 1 new record received\.$/,
    );
    await expect(page.getByText("4 records received for run")).toBeVisible();
  });

  test("shows the error when the check fails", async ({ page, request }) => {
    const runId = newRunId();
    await registerRun(
      request,
      fixtureRun(
        runId,
        [
          completedRun(runId),
          { status: 503, body: { detail: "Database unavailable" } },
        ],
        { state: "completed" },
      ),
    );
    await page.goto(`/live?liveRun=${runId}`);
    await expect(connection(page)).toContainText("Polling stopped");

    await advanceRun(request, runId);
    await checkButton(page).click();
    await expect(lateCheck(page)).toHaveText(
      /^Check failed at .+: The live API responded with HTTP 503\.$/,
    );
    // The run's last valid data stays shown.
    await expect(page.getByText("Final step totals")).toBeVisible();
  });
});
