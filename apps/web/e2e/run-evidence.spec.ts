import { mkdir, readFile, rm, utimes, writeFile } from "node:fs/promises";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { seedWorkspace, servePolicy } from "./audit";
import {
  advanceRun,
  registerRun,
  resetFixtureApi,
  restoreSavedHistory,
  takeSavedHistoryDown,
} from "./fixture-control";
import { fixtureRun, newRunId, snapshot } from "./live-runs";
import { E2E_DATA_DIR } from "./local-data";
import { newSavedRunId, savedRun } from "./saved-runs";

// F04: from a selected run to its evidence and export. Every link and download names the
// selected UUID; nothing falls back to another run. Synthetic fixture runs only.

const LOCAL_CODING = path.join(E2E_DATA_DIR, "coding-demo");

type RevokedObjectUrl = { url: string; inCreatingTask: boolean };
declare global {
  interface Window {
    /** Filled by watchObjectUrls in the page under test. */
    __revokedObjectUrls?: RevokedObjectUrl[];
  }
}

/**
 * Records each object URL the page revokes, and whether it was revoked in the same task that
 * created it. Firefox and Safari can cancel a download whose URL is revoked that early.
 */
async function watchObjectUrls(page: Page) {
  await page.addInitScript(() => {
    const creatingTask = new Set<string>();
    const revoked: RevokedObjectUrl[] = [];
    window.__revokedObjectUrls = revoked;
    const create = URL.createObjectURL.bind(URL);
    const revoke = URL.revokeObjectURL.bind(URL);
    URL.createObjectURL = (object) => {
      const url = create(object);
      creatingTask.add(url);
      setTimeout(() => creatingTask.delete(url), 0);
      return url;
    };
    URL.revokeObjectURL = (url) => {
      revoked.push({ url, inCreatingTask: creatingTask.has(url) });
      revoke(url);
    };
  });
}

/** Fixed file times, so "newer" never depends on how fast the test writes. */
const EARLIER = new Date("2026-11-01T12:00:00Z");
const LATER = new Date("2026-11-01T12:05:00Z");

/** One local evidence file, tagged with the run that wrote it, at a fixed modification time. */
async function writeLocalFile(file: string, runId: string, at: Date) {
  await mkdir(LOCAL_CODING, { recursive: true });
  const target = path.join(LOCAL_CODING, file);
  const content =
    file === "results.json"
      ? JSON.stringify(savedRun(runId, { sequence: 1 }).result)
      : `{"synthetic browser-test record": true, "agentwatch.run_id": "${runId}"}\n`;
  await writeFile(target, content);
  await utimes(target, at, at);
}

/**
 * A finished local coding artifact for `runId`. The worker demo writes spans.jsonl and
 * samples.jsonl during the run and results.json last, so results.json is the newest file.
 */
async function writeLocalArtifact(runId: string) {
  await writeLocalFile("spans.jsonl", runId, EARLIER);
  await writeLocalFile("samples.jsonl", runId, EARLIER);
  await writeLocalFile("results.json", runId, LATER);
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

for (const file of ["spans.jsonl", "samples.jsonl"])
  test(`a local ${file} newer than results.json is refused as unfinished evidence`, async ({
    request,
  }) => {
    // Run A finished; a rerun (run B) has rewritten this file but not results.json yet.
    const runA = newSavedRunId();
    await writeLocalArtifact(runA);
    const href = `/api/experiments/coding/artifact?file=${file}&local_run_id=${runA}`;
    expect((await request.get(href)).status()).toBe(200);

    const later = new Date(LATER.getTime() + 60_000);
    await writeLocalFile(file, newSavedRunId(), later);
    const refused = await request.get(href);
    expect(refused.status()).toBe(409);
    expect((await refused.json()).error).toMatch(/newer run/);
    // results.json itself still belongs to run A and is served.
    expect(
      (
        await request.get(
          `/api/experiments/coding/artifact?file=results.json&local_run_id=${runA}`,
        )
      ).status(),
    ).toBe(200);
  });

/** Clicks a link that downloads, returning the file's suggested name and text. */
async function download(page: Page, name: string) {
  const [file] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("link", { name, exact: true }).click(),
  ]);
  return {
    filename: file.suggestedFilename(),
    text: await readFile(await file.path(), "utf8"),
  };
}

test("a saved run's evidence files download from history for that run only", async ({
  page,
  request,
}) => {
  const { baseline, candidate } = await seedWorkspace(request);
  await page.goto(`/runs/${candidate}?tab=evidence`);
  for (const file of ["results.json", "spans.jsonl", "samples.jsonl"]) {
    const { filename, text } = await download(page, `Download ${file}`);
    expect(filename).toBe(`agentwatch-${candidate}-${file}`);
    expect(text).toContain(candidate);
    expect(text).not.toContain(baseline);
  }
  const exported = await download(page, "Export results.json");
  expect(JSON.parse(exported.text).run_id).toBe(candidate);

  // Compare offers each run's evidence; each download is that run's file.
  await page.goto(`/compare?run=${candidate}&baseline=${baseline}`);
  for (const [role, id] of [
    ["baseline", baseline],
    ["candidate", candidate],
  ]) {
    const { filename, text } = await download(page, `Export ${role} evidence`);
    expect(filename).toBe(`agentwatch-${id}-results.json`);
    expect(JSON.parse(text).run_id).toBe(id);
  }
});

test("a saved evidence download fails visibly when history is unavailable", async ({
  request,
}) => {
  const { candidate } = await seedWorkspace(request);
  const href = `/api/experiments/coding/artifact?file=spans.jsonl&run_id=${candidate}`;
  expect((await request.get(href)).status()).toBe(200);
  await takeSavedHistoryDown(request);
  const failed = await request.get(href);
  expect(failed.status()).toBe(503);
  expect(failed.headers()["content-disposition"]).toBeUndefined();
  expect(
    (
      await request.get(
        `/api/experiments/coding/artifact?file=spans.jsonl&run_id=${newSavedRunId()}`,
      )
    ).status(),
  ).toBe(503);
});

test("a failed saved download is reported in the page, which stays usable, and can be retried", async ({
  page,
  request,
}) => {
  const { baseline, candidate } = await seedWorkspace(request);
  const evidencePage = `/runs/${candidate}?tab=evidence`;
  await watchObjectUrls(page);
  await page.goto(evidencePage);
  await takeSavedHistoryDown(request);

  await page.getByRole("link", { name: "Download spans.jsonl" }).click();
  const failure = page
    .getByRole("alert")
    .filter({ hasText: "spans.jsonl not downloaded" });
  // The route's own reason: it forwards run history's 503.
  await expect(failure).toContainText("Saved artifact unavailable");
  await expect(failure).toContainText("HTTP 503");
  await expect(failure).toContainText("The run shown here is unchanged");
  // The run page is still there, not the route's raw JSON.
  await expect(page).toHaveURL(
    new RegExp(`${evidencePage.replace("?", "\\?")}$`),
  );
  await expect(page.getByText('"error"')).toHaveCount(0);
  await expect(
    page.getByRole("link", { name: "Download results.json" }),
  ).toBeVisible();

  // History is back: Try again saves the file and clears the error.
  await restoreSavedHistory(request);
  const [file] = await Promise.all([
    page.waitForEvent("download"),
    failure.getByRole("button", { name: "Try again" }).click(),
  ]);
  expect(file.suggestedFilename()).toBe(`agentwatch-${candidate}-spans.jsonl`);
  await expect(failure).toHaveCount(0);
  // The saved file's object URL outlives the click, so the download can start first.
  expect(
    await page.evaluate(() =>
      (window.__revokedObjectUrls ?? []).filter((url) => url.inCreatingTask),
    ),
  ).toEqual([]);

  // The header export and the other pages' exports report failures the same way.
  await takeSavedHistoryDown(request);
  await page.getByRole("link", { name: "Export results.json" }).click();
  await expect(
    page.getByRole("alert").filter({ hasText: "results.json not downloaded" }),
  ).toBeVisible();
  await page.getByRole("tab", { name: "Timeline" }).click();
  await expect(page.getByRole("tab", { name: "Timeline" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await page.getByRole("link", { name: "Download spans.jsonl" }).click();
  await expect(
    page.getByRole("alert").filter({ hasText: "spans.jsonl not downloaded" }),
  ).toBeVisible();
  await expect(page).toHaveURL(new RegExp(`/runs/${candidate}`));

  await restoreSavedHistory(request);
  await page.goto(`/compare?run=${candidate}&baseline=${baseline}`);
  await takeSavedHistoryDown(request);
  await page.getByRole("link", { name: "Export candidate evidence" }).click();
  await expect(
    page.getByRole("alert").filter({ hasText: "results.json not downloaded" }),
  ).toBeVisible();
  await expect(page).toHaveURL(/\/compare\?/);
});

test("a failed live records export names the live API, not run history", async ({
  page,
  request,
}) => {
  const runId = newRunId();
  await registerRun(
    request,
    fixtureRun(runId, [snapshot(runId, {}), { status: 503 }]),
  );
  await page.goto(`/live?liveRun=${runId}`);
  await page.getByRole("tab", { name: "Evidence", exact: true }).click();
  const download = page.getByRole("link", {
    name: "Download live records (JSON)",
  });
  await expect(download).toBeVisible();

  // The live API now answers 503; run history is untouched.
  await advanceRun(request, runId);
  await download.click();
  const failure = page
    .getByRole("alert")
    .filter({ hasText: "live-records.json not downloaded" });
  await expect(failure).toContainText("The live API");
  await expect(failure).toContainText("HTTP 503");
  await expect(failure).not.toContainText(/history/i);
  // Still the /live page with this run's Evidence tab, not the route's JSON.
  await expect(page).toHaveURL(
    new RegExp(`/live\\?liveRun=${runId}&tab=evidence$`),
  );
});
