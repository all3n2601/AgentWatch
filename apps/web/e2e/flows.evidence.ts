import { execFileSync } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";
import {
  pageOverflow,
  runRowButton,
  seedWorkspace,
  seriousA11yViolations,
  servePolicy,
  type SeededWorkspace,
} from "./audit";
import { registerSavedRun, resetFixtureApi } from "./fixture-control";
import { AUDIT_VIEWPORTS } from "./viewports";
import { newSavedRunId, savedRun } from "./saved-runs";

// F04 handoff evidence (playwright.evidence.config.ts). Walks each audited flow, saving a
// screenshot per step and recording the page-overflow and axe results beside it in
// f04-evidence/INDEX.md. Every run shown is synthetic fixture data, never recorded evidence;
// the pass/fail verdicts come from the browser test suite, not from this walk-through.

const EVIDENCE_DIR = path.join(process.cwd(), "f04-evidence");
const SHOWN_VIEWPORTS = ["desktop-1440x900", "tablet-768", "mobile-390"];

type Shot = {
  viewport: string;
  flow: string;
  step: string;
  file: string;
  overflowPx: number;
  axe: string[];
};
const shots: Shot[] = [];

/** Saves a full-page screenshot and the checks for what it shows. */
async function capture(
  page: Page,
  viewport: string,
  flow: string,
  step: string,
) {
  const file = path.join(
    viewport,
    `${String(shots.filter((shot) => shot.viewport === viewport).length + 1).padStart(2, "0")}-${flow}-${step}.png`,
  );
  // Let charts finish their first layout so the picture matches what a viewer sees.
  await page.waitForLoadState("networkidle");
  await page.screenshot({
    path: path.join(EVIDENCE_DIR, file),
    fullPage: true,
    animations: "disabled",
  });
  const { scrollWidth, clientWidth } = await pageOverflow(page);
  shots.push({
    viewport,
    flow,
    step,
    file,
    overflowPx: Math.max(0, scrollWidth - clientWidth),
    axe: await seriousA11yViolations(page),
  });
}

async function walkFlows(
  page: Page,
  viewport: string,
  seeded: SeededWorkspace & { older: string[] },
) {
  const shot = (flow: string, step: string) =>
    capture(page, viewport, flow, step);

  // Selected run → evidence → export.
  // The newest saved run heads the first list page.
  const newest = seeded.older.at(-1)!;
  await page.goto("/runs");
  await expect(runRowButton(page, newest)).toBeVisible();
  await shot("runs", "list");
  await runRowButton(page, newest).click();
  await shot("runs", "preview-selected");
  await page.goto(`/runs/${seeded.candidate}`);
  await expect(page.getByRole("tab", { name: "Diagnosis" })).toBeVisible();
  await shot("run", "diagnosis");
  for (const tab of ["Timeline", "Agent", "Resources", "Evidence"]) {
    await page.getByRole("tab", { name: tab, exact: true }).click();
    await shot("run", tab.toLowerCase());
  }

  // Filters and paging.
  await page.goto("/runs?source=saved&state=needs_attention");
  await expect(
    page.getByRole("combobox", { name: "Source", exact: true }),
  ).toContainText("Saved");
  await shot("runs-filters", "saved-needs-attention");
  await page.goto("/runs?source=saved&page=2");
  await shot("runs-filters", "page-2");
  await page.goto("/runs?q=no-run-has-this-text");
  await expect(page.getByText("No runs match these filters")).toBeVisible();
  await shot("runs-filters", "no-match");

  // Comparisons.
  await page.goto(
    `/runs?compare=saved:${seeded.baseline},saved:${seeded.candidate}&view=compare`,
  );
  await shot("compare", "runs-selection");
  await page.goto(
    `/compare?run=${seeded.candidate}&baseline=${seeded.baseline}`,
  );
  await expect(
    page.getByRole("button", {
      name: `Copy Candidate run ID ${seeded.candidate}`,
    }),
  ).toBeVisible();
  await shot("compare", "page");

  // Live runs, refresh and reconnect.
  await page.goto(`/live?liveRun=${seeded.live}`);
  await expect(page.getByText(/records? received for run/)).toBeVisible();
  await shot("live", "status");
  await page.getByRole("tab", { name: "Timeline" }).click();
  await shot("live", "timeline");
  await page.reload();
  await expect(page.getByRole("tab", { name: "Timeline" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await shot("live", "timeline-after-reload");
  await page.goto(`/runs/${seeded.live}?tab=evidence`);
  await shot("live-run", "evidence-export");

  // Controls and the not-found state.
  await page.goto("/controls");
  await shot("controls", "policy");
  await page.goto(`/runs/${newSavedRunId()}`);
  await shot("run", "not-found");

  // Keyboard focus on a filter, as a sighted keyboard user sees it.
  await page.goto("/runs");
  // Tab from the search box, so focus arrives the way a keyboard user's does.
  await page.getByRole("searchbox", { name: /Search loaded runs/ }).focus();
  await page.keyboard.press("Tab");
  await expect(
    page.getByRole("combobox", { name: "Source", exact: true }),
  ).toBeFocused();
  await shot("keyboard", "filter-focus");
}

for (const name of SHOWN_VIEWPORTS) {
  const audit = AUDIT_VIEWPORTS.find((viewport) => viewport.name === name)!;
  test.describe(name, () => {
    test.use({
      viewport: audit.viewport,
      isMobile: audit.mobile,
      hasTouch: audit.mobile,
    });

    test(`flows at ${name}`, async ({ page, request }) => {
      await resetFixtureApi(request);
      await servePolicy(page);
      const seeded = await seedWorkspace(request);
      // Enough saved runs for a second list page; every second one failed its checks.
      const older: string[] = [];
      for (let sequence = 3; sequence <= 30; sequence += 1) {
        const id = newSavedRunId();
        older.push(id);
        await registerSavedRun(
          request,
          savedRun(id, { sequence, passed: sequence % 2 === 1 }),
        );
      }
      await mkdir(path.join(EVIDENCE_DIR, name), { recursive: true });
      await walkFlows(page, name, { ...seeded, older });
    });
  });
}

const git = (...args: string[]) =>
  execFileSync("git", args, { encoding: "utf8" }).trim();

test.afterAll(async () => {
  const dirty = git("status", "--porcelain").length > 0;
  const rows = shots.map(
    (shot) =>
      `| ${shot.viewport} | ${shot.flow} | ${shot.step} | [${shot.file}](${shot.file.split(path.sep).join("/")}) | ${shot.overflowPx === 0 ? "none" : `${shot.overflowPx} px`} | ${shot.axe.length === 0 ? "none" : shot.axe.length} |`,
  );
  const findings = shots.flatMap((shot) =>
    shot.axe.map(
      (finding) => `- ${shot.viewport} ${shot.flow}/${shot.step}: ${finding}`,
    ),
  );
  await writeFile(
    path.join(EVIDENCE_DIR, "INDEX.md"),
    [
      "# F04 browser evidence",
      "",
      `Generated ${new Date().toISOString()} from commit ${git("rev-parse", "--short", "HEAD")}${dirty ? " with uncommitted changes" : ""}.`,
      "",
      "Every run shown is **synthetic browser-test fixture data** served by `e2e/fixture-api.ts`;",
      "none is recorded evidence. The policy service is answered by a test route.",
      "The verdicts come from the browser suite (`pnpm test:e2e`); this index records what each",
      "screenshot shows and the two checks run on it.",
      "",
      "- **Page overflow**: how far the document is wider than the viewport (none expected).",
      "- **axe**: serious or critical WCAG 2.x A/AA findings (`@axe-core/playwright`), before the",
      "  documented exceptions in `e2e/a11y.spec.ts`.",
      "- Not yet covered: the Attribution tab (PR #106) and large-run fixtures, which follow once",
      "  #106 merges.",
      "",
      "| Viewport | Flow | Step | Screenshot | Page overflow | axe |",
      "| --- | --- | --- | --- | --- | --- |",
      ...rows,
      "",
      "## axe findings",
      "",
      ...(findings.length > 0 ? findings : ["None."]),
      "",
      "The HTML report for this run is in `report/index.html`.",
      "",
    ].join("\n"),
  );
});
