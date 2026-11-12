import { expect, test, type Page } from "@playwright/test";

const policyPath = "/api/policies/concurrency";

test("policy mutations require same-origin JSON within the byte limit", async ({
  request,
  baseURL,
}) => {
  if (!baseURL) throw new Error("The policy test requires a base URL.");
  for (const [contentType, origin, data, status] of [
    ["text/plain", baseURL, "{}", 415],
    ["application/json", "https://other.example", "{}", 403],
    ["application/json", baseURL, "{}", 422],
    [
      "application/json",
      baseURL,
      JSON.stringify({ reason: "é".repeat(9_000) }),
      413,
    ],
  ] as const) {
    const response = await request.post(policyPath, {
      data,
      headers: { "content-type": contentType, origin },
    });
    expect(response.status()).toBe(status);
  }
});

type Limits = { inference_requests?: number; test_jobs?: number };

/** A stored policy version as the web route returns it. */
function policyVersion(version: number, limits: Limits, extra = {}) {
  return {
    target: "coding-agent",
    version,
    action: version === 0 ? "baseline" : "set",
    limits,
    restored_from: null,
    operator: version === 0 ? null : "reviewer",
    reason: version === 0 ? null : `Version ${version}`,
    request_id:
      version === 0
        ? null
        : `6f1c1d1e-8a63-4f43-9c3c-${String(version).padStart(12, "0")}`,
    created_at: version === 0 ? null : "2026-11-03T06:30:29.001256Z",
    controlled_services: {
      inference_requests: { service_id: "ollama", resource_class: "inference" },
      test_jobs: {
        service_id: "agentwatch-agent-runner",
        resource_class: "cpu_sandbox",
      },
    },
    enforcement_scope: "runner_process",
    ...extra,
  };
}
type Version = ReturnType<typeof policyVersion>;

const v0 = policyVersion(0, {});
const v1 = policyVersion(1, { test_jobs: 2 });
const v2 = policyVersion(2, { inference_requests: 1, test_jobs: 2 });

type Reply = { status: number; json: unknown };

/**
 * Scripts the web policy route. Each read answers with the next `reads` entry (the last
 * repeats); each change answers with the next `writes` entry and is recorded.
 */
async function servePolicy(
  page: Page,
  {
    reads,
    writes = [],
  }: { reads: { active: Version; versions: Version[] }[]; writes?: Reply[] },
) {
  const posted: Record<string, unknown>[] = [];
  let read = 0;
  await page.route(`**${policyPath}`, async (route) => {
    if (route.request().method() === "POST") {
      posted.push(route.request().postDataJSON());
      const reply = writes[posted.length - 1];
      if (!reply) return route.abort();
      return route.fulfill(reply);
    }
    const { active, versions } = reads[Math.min(read++, reads.length - 1)];
    await route.fulfill({
      json: { active, history: { versions, next_before: null } },
    });
  });
  return posted;
}

const conflict = (error: string) => ({
  status: 409,
  json: { error, kind: "conflict" },
});

/** The failed-change banner with this title; Next's route announcer is also an alert. */
const changeNotice = (page: Page, title: string) =>
  page.getByRole("alert").filter({ hasText: title });

async function attribute(page: Page) {
  await page.getByLabel("Operator").fill("review");
  await page.getByLabel("Reason").fill("Check limits");
}

test("malformed numeric input cannot remove an existing limit", async ({
  page,
}) => {
  const writes = await servePolicy(page, {
    reads: [{ active: v1, versions: [v1, v0] }],
  });
  await page.goto("/controls");
  await page.getByLabel("Inference requests").fill("8");
  await page.getByLabel("Test jobs").fill("1e");
  await attribute(page);
  await page.getByRole("button", { name: "Set limits" }).click();
  await expect(
    page.getByText(/Test jobs must be a whole number/),
  ).toBeVisible();
  await expect(page.getByLabel("Test jobs")).toHaveValue("1e");
  expect(writes).toHaveLength(0);
});

test("a limit set from the keyboard keeps the other control's active limit", async ({
  page,
}) => {
  const writes = await servePolicy(page, {
    reads: [
      { active: v1, versions: [v1, v0] },
      { active: v2, versions: [v2, v1, v0] },
    ],
    writes: [{ status: 201, json: { created: true, policy: v2 } }],
  });
  await page.goto("/controls");
  await expect(page.getByLabel("Test jobs")).toHaveValue("2");
  await expect(
    page.getByText(/Each agent runner records the limits it applied/),
  ).toBeVisible();

  await page.getByLabel("Inference requests").focus();
  await page.keyboard.type("1");
  await page.keyboard.press("Tab");
  await expect(page.getByLabel("Test jobs")).toBeFocused();
  await page.keyboard.press("Tab");
  await page.keyboard.type("review");
  await page.keyboard.press("Tab");
  await page.keyboard.type("Limit model requests");
  await expect(
    page.getByText("Inference requests: no limit → 1"),
  ).toBeVisible();
  await page.keyboard.press("Enter");

  await expect(page.getByText("Saved as version 2.")).toBeVisible();
  expect(writes).toEqual([
    expect.objectContaining({
      action: "set",
      expected_version: 1,
      limits: { inference_requests: 1, test_jobs: 2 },
      operator: "review",
      reason: "Limit model requests",
    }),
  ]);
});

test("limits equal to the active version are not saved", async ({ page }) => {
  const writes = await servePolicy(page, {
    reads: [{ active: v1, versions: [v1, v0] }],
  });
  await page.goto("/controls");
  await attribute(page);
  await expect(
    page.getByText("No change: these are the active limits."),
  ).toBeVisible();
  await page.getByRole("button", { name: "Set limits" }).click();
  await expect(
    page.getByText(/match active version 1; nothing to change/),
  ).toBeVisible();
  expect(writes).toHaveLength(0);
});

test("disable and restore are confirmed and dismissing sends nothing", async ({
  page,
}) => {
  const writes = await servePolicy(page, {
    reads: [{ active: v2, versions: [v2, v1, v0] }],
    writes: [
      {
        status: 201,
        json: {
          created: true,
          policy: policyVersion(
            3,
            { test_jobs: 2 },
            {
              action: "restore",
              restored_from: 1,
            },
          ),
        },
      },
    ],
  });
  await page.goto("/controls");
  await attribute(page);

  const dialogs: string[] = [];
  page.once("dialog", (dialog) => {
    dialogs.push(dialog.message());
    void dialog.dismiss();
  });
  await page.getByRole("button", { name: "Disable limits" }).click();
  await expect.poll(() => dialogs.length).toBe(1);
  expect(writes).toHaveLength(0);

  page.once("dialog", (dialog) => {
    dialogs.push(dialog.message());
    void dialog.accept();
  });
  await page.getByRole("button", { name: "Restore version 1" }).click();
  await expect(page.getByText("Saved as version 3.")).toBeVisible();
  expect(dialogs[1]).toContain("Inference: no limit · Tests: 2");
  expect(writes).toEqual([
    expect.objectContaining({
      action: "restore",
      restored_from: 1,
      expected_version: 2,
    }),
  ]);
});

test("a stale version is rejected and the form reloads the latest limits", async ({
  page,
}) => {
  await servePolicy(page, {
    reads: [
      { active: v1, versions: [v1, v0] },
      { active: v2, versions: [v2, v1, v0] },
    ],
    writes: [
      conflict("Policy changed: expected version 1, active version is 2"),
    ],
  });
  await page.goto("/controls");
  await page.getByLabel("Test jobs").fill("4");
  await attribute(page);
  await page.getByRole("button", { name: "Set limits" }).click();
  await expect(
    page.getByText(/another change was recorded after this page read/i),
  ).toBeVisible();
  await expect(
    page.getByText("Version 2: Inference: 1 · Tests: 2"),
  ).toBeVisible();
  await expect(page.getByLabel("Inference requests")).toHaveValue("1");
  await expect(page.getByLabel("Test jobs")).toHaveValue("2");
});

test("a reused request ID is reported as its own conflict", async ({
  page,
}) => {
  await servePolicy(page, {
    reads: [{ active: v1, versions: [v1, v0] }],
    writes: [conflict("Policy request reused with a different body")],
  });
  await page.goto("/controls");
  await page.getByLabel("Test jobs").fill("4");
  await attribute(page);
  await page.getByRole("button", { name: "Set limits" }).click();
  await expect(
    page.getByText(/this request ID was already used for a different change/i),
  ).toBeVisible();
});

test("a rejected change shows the service's reason", async ({ page }) => {
  await servePolicy(page, {
    reads: [{ active: v1, versions: [v1, v0] }],
    writes: [
      {
        status: 422,
        json: {
          error: "limits.test_jobs: Input should be less than or equal to 1024",
          kind: "invalid_request",
        },
      },
    ],
  });
  await page.goto("/controls");
  await page.getByLabel("Test jobs").fill("4");
  await attribute(page);
  await page.getByRole("button", { name: "Set limits" }).click();
  await expect(changeNotice(page, "Not saved")).toContainText(
    "limits.test_jobs: Input should be less than or equal to 1024",
  );
  await expect(page.getByRole("button", { name: "Retry" })).toHaveCount(0);
});

test("an unconfirmed change is retried with the same request ID", async ({
  page,
}) => {
  const writes = await servePolicy(page, {
    reads: [
      { active: v1, versions: [v1, v0] },
      { active: v2, versions: [v2, v1, v0] },
    ],
    writes: [
      {
        status: 503,
        json: { error: "Database unavailable", kind: "unavailable" },
      },
      { status: 200, json: { created: false, policy: v2 } },
    ],
  });
  await page.goto("/controls");
  await page.getByLabel("Inference requests").fill("1");
  await attribute(page);
  await page.getByRole("button", { name: "Set limits" }).click();
  await expect(changeNotice(page, "Not confirmed")).toContainText(
    "Database unavailable",
  );
  await page.getByRole("button", { name: "Retry" }).click();
  await expect(page.getByText("Already recorded as version 2.")).toBeVisible();
  expect(writes).toHaveLength(2);
  expect(writes[1].request_id).toBe(writes[0].request_id);
});

test("an inconsistent read pauses changes until the policy is read again", async ({
  page,
}) => {
  await servePolicy(page, {
    reads: [
      { active: v1, versions: [v2, v1, v0] },
      { active: v2, versions: [v2, v1, v0] },
    ],
  });
  await page.goto("/controls");
  await expect(page.getByText(/Stale read/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Set limits" })).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Disable limits" }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Read again" }).click();
  await expect(page.getByText(/Stale read/)).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Set limits" })).toBeEnabled();
});

test("an unavailable policy is never shown as no limits", async ({ page }) => {
  await page.route(`**${policyPath}`, (route) =>
    route.fulfill({
      status: 503,
      json: { error: "Database unavailable", kind: "unavailable" },
    }),
  );
  await page.goto("/controls");
  await expect(page.getByText("Policy service unavailable")).toBeVisible();
  await expect(page.getByText("No limits")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Set limits" })).toBeDisabled();
});
