import { expect, test } from "@playwright/test";

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

test("malformed numeric input cannot remove an existing limit", async ({
  page,
}) => {
  const active = {
    target: "coding-agent",
    version: 1,
    action: "set",
    limits: { test_jobs: 2 },
    restored_from: null,
    operator: "reviewer",
    reason: "Existing policy",
    request_id: "6f1c1d1e-8a63-4f43-9c3c-0f7a1f6f4b11",
    created_at: "2026-11-03T06:30:29.001256Z",
    controlled_services: {
      inference_requests: { service_id: "ollama", resource_class: "inference" },
      test_jobs: {
        service_id: "agentwatch-agent-runner",
        resource_class: "cpu_sandbox",
      },
    },
    enforcement_scope: "runner_process",
  };
  const writes: unknown[] = [];
  await page.route(`**${policyPath}`, async (route) => {
    if (route.request().method() === "POST") {
      writes.push(route.request().postDataJSON());
      await route.abort();
      return;
    }
    await route.fulfill({
      json: { active, history: { versions: [active], next_before: null } },
    });
  });
  await page.goto("/controls");
  await page.getByLabel("Inference requests").fill("8");
  await page.getByLabel("Test jobs").fill("1e");
  await page.getByLabel("Operator").fill("review");
  await page.getByLabel("Reason").fill("Check limits");
  await page.getByRole("button", { name: "Set limits" }).click();
  await expect(
    page.getByText(/Test jobs must be a whole number/),
  ).toBeVisible();
  await expect(page.getByLabel("Test jobs")).toHaveValue("1e");
  expect(writes).toHaveLength(0);
});
