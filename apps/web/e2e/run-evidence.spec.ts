import { mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { expect, test } from "@playwright/test";
import { seedWorkspace, servePolicy } from "./audit";
import { resetFixtureApi, takeSavedHistoryDown } from "./fixture-control";
import { E2E_DATA_DIR } from "./local-data";
import { newSavedRunId, savedRun } from "./saved-runs";

// F04: from a selected run to its evidence and export. Every link and download names the
// selected UUID; nothing falls back to another run. Synthetic fixture runs only.

const LOCAL_CODING = path.join(E2E_DATA_DIR, "coding-demo");

/** Writes a local coding artifact for `runId`, as the worker demo does. */
async function writeLocalArtifact(runId: string) {
  await mkdir(LOCAL_CODING, { recursive: true });
  const { result } = savedRun(runId, { sequence: 1 });
  await writeFile(
    path.join(LOCAL_CODING, "results.json"),
    JSON.stringify(result),
  );
  await writeFile(
    path.join(LOCAL_CODING, "spans.jsonl"),
    '{"name": "synthetic browser-test span"}\n',
  );
}

test.beforeEach(async ({ page, request }) => {
  await resetFixtureApi(request);
  await servePolicy(page);
});
// Other tests expect no local artifacts.
test.afterEach(() => rm(LOCAL_CODING, { recursive: true, force: true }));

test("a saved run's evidence links and exports all name the selected UUID", async ({
  page,
  request,
}) => {
  const { candidate } = await seedWorkspace(request);
  await page.goto(`/runs/${candidate}`);
  await page.getByRole("tab", { name: "Evidence" }).click();
  await expect(page).toHaveURL(/tab=evidence/);
  for (const file of ["results.json", "spans.jsonl", "samples.jsonl"])
    await expect(
      page.getByRole("link", { name: `Download ${file}` }),
    ).toHaveAttribute(
      "href",
      `/api/experiments/coding/artifact?file=${file}&run_id=${candidate}`,
    );
  await expect(
    page.getByRole("link", { name: "Export results.json" }),
  ).toHaveAttribute(
    "href",
    `/api/experiments/coding/artifact?file=results.json&run_id=${candidate}`,
  );
  await page.getByRole("link", { name: "Compare with another run" }).click();
  await expect(page).toHaveURL(new RegExp(`/compare\\?run=${candidate}$`));
});

test("a live run's evidence and export read only that run", async ({
  page,
  request,
}) => {
  const { live } = await seedWorkspace(request);
  await page.goto(`/runs/${live}?tab=evidence`);
  await expect(page.getByRole("tab", { name: "Evidence" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  for (const link of await page
    .getByRole("link", { name: /Export|Download/ })
    .all())
    await expect(link).toHaveAttribute("href", `/api/live/runs/${live}/export`);
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("link", { name: "Export JSON" }).click(),
  ]);
  expect(download.suggestedFilename()).toContain(live);
});

test("an unknown run is a not-found page, never another run", async ({
  page,
  request,
}) => {
  const { candidate } = await seedWorkspace(request);
  const unknown = newSavedRunId();
  const response = await page.goto(`/runs/${unknown}?tab=evidence`);
  expect(response?.status()).toBe(404);
  await expect(page.getByText(candidate.slice(0, 8))).toHaveCount(0);
});

test("artifact downloads reject run ids that are not UUIDs", async ({
  request,
}) => {
  // 36 characters of [0-9a-f-] passed the route's old pattern without being a UUID.
  for (const bad of ["-".repeat(36), "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaaa"])
    for (const param of ["run_id", "local_run_id"]) {
      const response = await request.get(
        `/api/experiments/coding/artifact?file=results.json&${param}=${bad}`,
      );
      expect(response.status(), `${param}=${bad}`).toBe(400);
    }
});

test("a local artifact downloads only while it still holds the run the page showed", async ({
  page,
  request,
}) => {
  const shown = newSavedRunId();
  await writeLocalArtifact(shown);
  await takeSavedHistoryDown(request);

  // With history down the run page shows the local artifact, labelled as such.
  await page.goto(`/runs/${shown}?tab=evidence`);
  await expect(page.getByText("Local file fallback")).toBeVisible();
  const href = `/api/experiments/coding/artifact?file=spans.jsonl&local_run_id=${shown}`;
  await expect(
    page.getByRole("link", { name: "Download spans.jsonl" }),
  ).toHaveAttribute("href", href);

  const served = await request.get(href);
  expect(served.status()).toBe(200);
  expect(served.headers()["content-disposition"]).toContain(shown);

  // The demo ran again after the page loaded: the files now hold another run.
  await writeLocalArtifact(newSavedRunId());
  const refused = await request.get(href);
  expect(refused.status()).toBe(409);
  expect((await refused.json()).error).toMatch(/different run/);

  await rm(LOCAL_CODING, { recursive: true, force: true });
  expect((await request.get(href)).status()).toBe(404);
});

test("back and forward keep the run tab chosen before leaving the page", async ({
  page,
  request,
}) => {
  // F04 check of bug 2: the run tabs write history.replaceState(null, …), which Next's
  // docs use too; this confirms back/forward still restore the tab and the page.
  const { candidate } = await seedWorkspace(request);
  await page.goto(`/runs/${candidate}`);
  await page.getByRole("tab", { name: "Timeline" }).click();
  await expect(page).toHaveURL(/tab=timeline/);
  await page
    .getByRole("main")
    .getByRole("link", { name: "Compare", exact: true })
    .click();
  await expect(page).toHaveURL(/\/compare\?run=/);

  await page.goBack();
  await expect(page).toHaveURL(new RegExp(`/runs/${candidate}\\?tab=timeline`));
  await expect(page.getByRole("tab", { name: "Timeline" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await page.goForward();
  await expect(page).toHaveURL(/\/compare\?run=/);
  await expect(
    page.getByRole("button", { name: `Copy Candidate run ID ${candidate}` }),
  ).toBeVisible();
});

test("the live view reopens the tab in its address after a reload and back", async ({
  page,
  request,
}) => {
  const { live } = await seedWorkspace(request);
  await page.goto(`/live?liveRun=${live}`);
  const timeline = page.getByRole("tab", { name: "Timeline" });
  await timeline.click();
  await expect(page).toHaveURL(/tab=timeline/);

  await page.reload();
  await expect(timeline).toHaveAttribute("aria-selected", "true");

  await page.getByRole("link", { name: "Open run page" }).click();
  await expect(page).toHaveURL(new RegExp(`/runs/${live}`));
  await page.goBack();
  await expect(page).toHaveURL(/tab=timeline/);
  await expect(timeline).toHaveAttribute("aria-selected", "true");
});
