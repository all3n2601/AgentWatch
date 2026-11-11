import { expect, test, type Page } from "@playwright/test";
import { registerRun, resetFixtureApi } from "./fixture-control";
import { fixtureRun, newRunId, snapshot, stepEvent } from "./live-runs";

// The live run view keeps the policy active while a run was open, what its runner applied and
// the current policy apart, for the selected run's UUID only.

test.beforeEach(async ({ request }) => resetFixtureApi(request));

const controlledServices = {
  inference_requests: { service_id: "ollama", resource_class: "inference" },
  test_jobs: {
    service_id: "agentwatch-agent-runner",
    resource_class: "cpu_sandbox",
  },
};
const version = (n: number, limits: Record<string, number>) => ({
  target: "coding-agent",
  version: n,
  action: n === 0 ? "baseline" : "set",
  limits,
  restored_from: null,
  operator: n === 0 ? null : "reviewer",
  reason: n === 0 ? null : `Version ${n}`,
  request_id:
    n === 0 ? null : `6f1c1d1e-8a63-4f43-9c3c-${String(n).padStart(12, "0")}`,
  created_at: n === 0 ? null : "2026-11-03T06:30:29.001256Z",
  controlled_services: controlledServices,
  enforcement_scope: "runner_process",
});
const v1 = version(1, { inference_requests: 1 });
const v2 = version(2, { inference_requests: 1, test_jobs: 2 });

async function liveRun(
  request: Parameters<typeof registerRun>[0],
  sequence = 1,
) {
  const runId = newRunId();
  const queued = stepEvent("tool-1", 0, "queued", { runId });
  await registerRun(
    request,
    fixtureRun(
      runId,
      [snapshot(runId, { steps: [queued], records: [queued] })],
      { sequence },
    ),
  );
  return runId;
}

/** Serves each run's policy by UUID, and the current policy, from the web routes. */
async function servePolicies(
  page: Page,
  runs: Record<
    string,
    { applicable: boolean; versions: unknown[]; hold?: Promise<void> } | number
  >,
  current: unknown = {
    active: v2,
    history: { versions: [v2], next_before: null },
  },
) {
  const answered: string[] = [];
  await page.route("**/api/live/runs/*/policy", async (route) => {
    const runId = new URL(route.request().url()).pathname.split("/")[4];
    const run = runs[runId];
    if (typeof run === "number")
      return route.fulfill({
        status: run,
        json: { error: "Database unavailable", kind: "unavailable" },
      });
    const { hold, ...body } = run;
    await hold;
    // The page may have aborted a held request when another run was selected.
    await route
      .fulfill({ json: { run_id: runId, target: "coding-agent", ...body } })
      .catch(() => undefined);
    answered.push(runId);
  });
  await page.route("**/api/policies/concurrency", (route) =>
    typeof current === "number"
      ? route.fulfill({
          status: current,
          json: { error: "Database unavailable", kind: "unavailable" },
        })
      : route.fulfill({ json: current }),
  );
  return answered;
}

const policy = (page: Page) =>
  page.getByRole("region", { name: "Concurrency policy" });

test("a run shows its own linked versions, unknown applied limits and the current policy", async ({
  page,
  request,
}) => {
  const runId = await liveRun(request);
  await servePolicies(page, {
    [runId]: { applicable: true, versions: [v1, v2] },
  });
  await page.goto(`/live?liveRun=${runId}`);
  await expect(policy(page)).toContainText(
    "Policy while open: v1 (Inference: 1 · Tests: no limit) → v2 (Inference: 1 · Tests: 2)",
  );
  await expect(policy(page)).toContainText(
    "Not available from the API. Each agent runner records the limits it applied",
  );
  await expect(policy(page)).toContainText(
    "Version 2: Inference: 1 · Tests: 2",
  );
});

test("runs without an agent runner or recorded history are labelled, not given a baseline", async ({
  page,
  request,
}) => {
  const other = await liveRun(request, 1);
  const early = await liveRun(request, 2);
  await servePolicies(page, {
    [other]: { applicable: false, versions: [] },
    [early]: { applicable: true, versions: [] },
  });
  await page.goto(`/live?liveRun=${other}`);
  await expect(policy(page)).toContainText("Not an agent run");
  await expect(policy(page)).toContainText(
    "Not applicable: this run has no agent runner",
  );

  await page.goto(`/live?liveRun=${early}`);
  await expect(policy(page)).toContainText(
    "No recorded policy (registered before policy tracking)",
  );
  await expect(policy(page)).not.toContainText("baseline");
});

test("an unavailable policy service is reported, never shown as no limits", async ({
  page,
  request,
}) => {
  const runId = await liveRun(request);
  await servePolicies(page, { [runId]: 503 }, 503);
  await page.goto(`/live?liveRun=${runId}`);
  await expect(policy(page)).toContainText("Policy service unavailable");
  await expect(policy(page)).toContainText("Not available from the API");
  await expect(policy(page)).not.toContainText("No limits");
});

test("a late answer for the previous run never replaces the selected run's policy", async ({
  page,
  request,
}) => {
  const first = await liveRun(request, 1);
  const second = await liveRun(request, 2);
  let release = () => {};
  const hold = new Promise<void>((resolve) => (release = resolve));
  const answered = await servePolicies(page, {
    [first]: { applicable: true, versions: [v1], hold },
    [second]: { applicable: true, versions: [v2] },
  });
  await page.goto(`/live?liveRun=${first}`);
  await expect(policy(page)).toContainText("Loading policy…");
  await page
    .getByRole("navigation", { name: "Recorded live runs" })
    .getByRole("link")
    .filter({ hasText: second.slice(0, 8) })
    .click();
  await expect(page).toHaveURL(new RegExp(`liveRun=${second}`));
  await expect(policy(page)).toContainText("Policy while open: v2 ");

  release();
  await expect.poll(() => answered).toContain(first);
  await expect(policy(page)).toContainText("Policy while open: v2 ");
  await expect(policy(page)).not.toContainText("v1 (");
});
