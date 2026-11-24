import {
  expect,
  test,
  type APIRequestContext,
  type Page,
} from "@playwright/test";
import { runRowButton } from "./audit";
import {
  registerRun,
  registerSavedRun,
  resetFixtureApi,
  takeSavedHistoryDown,
} from "./fixture-control";
import { fixtureRun, newRunId, snapshot } from "./live-runs";
import { newSavedRunId, savedRun } from "./saved-runs";

// F04: the Runs page filters, paging, preview and comparison ticks, and that all of them
// survive a reload and the older/newest live-run links. Synthetic fixture runs only.

test.beforeEach(async ({ request }) => resetFixtureApi(request));

/** Saved runs with sequences 1..count; every second one failed its checks. */
async function savedRuns(request: APIRequestContext, count: number) {
  const ids: string[] = [];
  for (let sequence = 1; sequence <= count; sequence += 1) {
    const id = newSavedRunId();
    ids.push(id);
    await registerSavedRun(
      request,
      savedRun(id, { sequence, passed: sequence % 2 === 1 }),
    );
  }
  // Newest first, as the list shows them.
  return ids.reverse();
}

async function liveRuns(request: APIRequestContext, count: number) {
  const ids: string[] = [];
  for (let sequence = 1; sequence <= count; sequence += 1) {
    const id = newRunId();
    ids.push(id);
    await registerRun(
      request,
      fixtureRun(id, [snapshot(id, {})], { sequence }),
    );
  }
  return ids.reverse();
}

const filter = (page: Page, name: string) =>
  page.getByRole("combobox", { name, exact: true });

async function choose(page: Page, name: string, option: string) {
  await filter(page, name).click();
  await page.getByRole("option", { name: option, exact: true }).click();
  await expect(filter(page, name)).toContainText(option);
}

const pager = (page: Page) =>
  page.getByRole("navigation", { name: "Run list pages" });
const compareBox = (page: Page, id: string) =>
  page.getByRole("checkbox", { name: new RegExp(id.slice(0, 8)) });
/** The "N of M match" figure in the summary strip. */
const matchCount = (page: Page) =>
  page
    .getByRole("region", { name: "Runs matching the filters" })
    .getByRole("listitem")
    .first();
const matches = (count: number) => new RegExp(`^\\s*${count}\\s*of `);

test("filters, page, preview and ticked runs survive a reload", async ({
  page,
  request,
}) => {
  const saved = await savedRuns(request, 30);
  await liveRuns(request, 1);
  await page.goto("/runs");
  await expect(matchCount(page)).toHaveText(matches(31));

  await choose(page, "Source", "Saved");
  await expect(page).toHaveURL(/source=saved/);
  await expect(matchCount(page)).toHaveText(matches(30));
  await expect(pager(page)).toContainText("Page 1 of 2");

  await pager(page).getByRole("button", { name: "Next" }).click();
  await expect(page).toHaveURL(/page=2/);
  // Rows 26–30 of 30, newest first.
  const previewed = saved[27];
  await runRowButton(page, previewed).click();
  await expect(page).toHaveURL(new RegExp(`preview=saved%3A${previewed}`));
  for (const id of [saved[26], saved[28]]) await compareBox(page, id).check();
  await expect(page).toHaveURL(/compare=saved%3A/);

  await page.reload();
  await expect(filter(page, "Source")).toContainText("Saved");
  await expect(pager(page)).toContainText("Page 2 of 2");
  await expect(runRowButton(page, previewed)).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(compareBox(page, saved[26])).toBeChecked();
  await expect(compareBox(page, saved[28])).toBeChecked();
  await expect(
    page.getByRole("tab", { name: /Compare selection/ }),
  ).toContainText("2 of 2");

  // A new filter starts again from the first page.
  await choose(page, "State", "Needs attention");
  await expect(page).not.toHaveURL(/page=/);
  await expect(page).toHaveURL(/state=needs_attention/);
  await expect(matchCount(page)).toHaveText(matches(15));
});

test("search and Clear filters update the list, its count and the address", async ({
  page,
  request,
}) => {
  const saved = await savedRuns(request, 3);
  await page.goto("/runs");
  const search = page.getByRole("searchbox", { name: /Search loaded runs/ });
  await search.fill(saved[1]);
  await expect(page).toHaveURL(new RegExp(`q=${saved[1]}`));
  await expect(runRowButton(page, saved[1])).toBeVisible();
  await expect(runRowButton(page, saved[0])).toHaveCount(0);

  await search.fill("no-run-has-this-text");
  await expect(page.getByText("No runs match these filters")).toBeVisible();
  await choose(page, "Source", "Saved");
  await page.getByRole("button", { name: "Clear filters" }).click();
  await expect(search).toHaveValue("");
  await expect(filter(page, "Source")).toContainText("All sources");
  await expect(page).not.toHaveURL(/[?&](q|source)=/);
  for (const id of saved) await expect(runRowButton(page, id)).toBeVisible();
});

test("the preview only shows a run on the visible page", async ({
  page,
  request,
}) => {
  const saved = await savedRuns(request, 30);
  await page.goto("/runs");
  await runRowButton(page, saved[3]).click();
  await expect(page).toHaveURL(new RegExp(`preview=saved%3A${saved[3]}`));
  await pager(page).getByRole("button", { name: "Next" }).click();
  // The page-1 run is not on page 2, so the preview moves to page 2's first run.
  await expect(runRowButton(page, saved[25])).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  const preview = page.locator(
    "#" + (await runRowButton(page, saved[25]).getAttribute("aria-controls")),
  );
  await expect(preview).toContainText(saved[25].slice(0, 8));
  await expect(preview).not.toContainText(saved[3].slice(0, 8));
  await expect(page).not.toHaveURL(/preview=/);
});

test("a stale or malformed address falls back to the defaults", async ({
  page,
  request,
}) => {
  const saved = await savedRuns(request, 3);
  const unknown = newSavedRunId();
  await page.goto(
    `/runs?source=both&state=Done&kind=%3Cscript%3E&resource=x%20y&page=99&preview=saved:${unknown}&compare=saved:${unknown},bogus`,
  );
  await expect(filter(page, "Source")).toContainText("All sources");
  await expect(filter(page, "State")).toContainText("Any state");
  await expect(filter(page, "Kind")).toContainText("All kinds");
  await expect(filter(page, "Top resource")).toContainText("Any resource");
  await expect(pager(page)).toContainText("Page 1 of 1");
  await expect(runRowButton(page, saved[0])).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(
    page.getByRole("tab", { name: /Compare selection/ }),
  ).toContainText("0 of 2");
  // The address is rewritten to what is shown.
  await expect(page).toHaveURL(/\/runs$/);
});

test("the older and newest live-run links keep the filters and ticked runs", async ({
  page,
  request,
}) => {
  const live = await liveRuns(request, 55);
  const saved = await savedRuns(request, 2);
  await page.goto("/runs");
  for (const id of saved) await compareBox(page, id).check();
  await choose(page, "Source", "Live");
  // Fixture live runs share one registration time, so their order within a page is by id;
  // counts, not positions, show which live page is loaded.
  await expect(matchCount(page)).toHaveText(matches(50));

  await page.getByRole("link", { name: "Older live runs" }).click();
  await expect(page).toHaveURL(/liveBefore=/);
  await expect(page).toHaveURL(/source=live/);
  await expect(filter(page, "Source")).toContainText("Live");
  await expect(matchCount(page)).toHaveText(matches(5));
  await expect(runRowButton(page, live[54])).toBeVisible();
  await expect(
    page.getByRole("tab", { name: /Compare selection/ }),
  ).toContainText("2 of 2");

  await page.getByRole("link", { name: "Newest live runs" }).click();
  await expect(page).not.toHaveURL(/liveBefore=/);
  await expect(filter(page, "Source")).toContainText("Live");
  await expect(matchCount(page)).toHaveText(matches(50));
});

test("older saved runs load on request and a failed load keeps the shown runs", async ({
  page,
  request,
}) => {
  const saved = await savedRuns(request, 105);
  await page.goto("/runs");
  await expect(matchCount(page)).toHaveText(matches(100));
  await page.getByRole("button", { name: "Load older saved runs" }).click();
  await expect(matchCount(page)).toHaveText(matches(105));
  await expect(
    page.getByRole("button", { name: "Load older saved runs" }),
  ).toHaveCount(0);

  await page.reload();
  await takeSavedHistoryDown(request);
  await page.getByRole("button", { name: "Load older saved runs" }).click();
  await expect(page.getByText("Older saved runs not loaded")).toBeVisible();
  await expect(page.getByRole("button", { name: "Try again" })).toBeVisible();
  await expect(matchCount(page)).toHaveText(matches(100));
  await expect(runRowButton(page, saved[0])).toBeVisible();
});

test("a filter that only older history matches is kept until those runs load", async ({
  page,
  request,
}) => {
  // The one CPU run is the oldest, beyond the first history page of 100.
  const cpu = newSavedRunId();
  await registerSavedRun(request, {
    ...savedRun(cpu, { sequence: 1 }),
    kind: "cpu",
  });
  for (let sequence = 2; sequence <= 101; sequence += 1)
    await registerSavedRun(request, savedRun(newSavedRunId(), { sequence }));

  await page.goto("/runs?kind=cpu");
  await expect(filter(page, "Kind")).toContainText("(not in loaded runs)");
  await expect(page.getByText("No runs match these filters")).toBeVisible();
  await expect(page).toHaveURL(/kind=cpu/);

  await page.getByRole("button", { name: "Load older saved runs" }).click();
  await expect(matchCount(page)).toHaveText(matches(1));
  await expect(runRowButton(page, cpu)).toBeVisible();
  await expect(filter(page, "Kind")).not.toContainText("not in loaded runs");
  await expect(page).toHaveURL(/kind=cpu/);
});
