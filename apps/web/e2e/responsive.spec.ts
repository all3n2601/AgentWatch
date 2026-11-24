import { expect, test, type Locator, type Page } from "@playwright/test";
import {
  auditPages,
  openAuditPage,
  pageOverflow,
  seedWorkspace,
  unscrollableOverflow,
  type SeededWorkspace,
} from "./audit";
import { resetFixtureApi } from "./fixture-control";

// F04: every page at every audit viewport (playwright.config.ts projects). The page never
// scrolls sideways, wide content scrolls only inside a labelled region, and navigation and
// each page's main actions stay reachable. Synthetic fixture runs only.

/** Touch targets smaller than this are reported (apps/web/AGENTS.md aims for 44×44). */
const TOUCH_TARGET_PX = 44;

const NAVIGATION = [
  "Overview",
  "Runs",
  "Experiments",
  "Compare",
  "Data quality",
  "Live runs",
  "Controls",
  "Settings",
];

/** The actions a page exists for; each must be reachable at every width. */
const PAGE_ACTIONS: Record<
  string,
  (page: Page, seeded: SeededWorkspace) => Locator[]
> = {
  runs: (page) => [
    page.getByRole("searchbox", { name: /Search loaded runs/ }),
    page.getByRole("combobox", { name: "Source", exact: true }),
    page.getByRole("combobox", { name: "State", exact: true }),
    page.getByRole("combobox", { name: "Kind", exact: true }),
    page.getByRole("combobox", { name: "Top resource", exact: true }),
    page.getByRole("button", { name: "Next" }),
  ],
  "run-saved-diagnosis": (page) => [
    page.getByRole("link", { name: "Export results.json" }),
    page.getByRole("link", { name: "Compare" }).first(),
    page.getByRole("tab", { name: "Evidence" }),
  ],
  "run-saved-evidence": (page) => [
    page.getByRole("link", { name: "Download results.json" }),
    page.getByRole("link", { name: "Download spans.jsonl" }),
    page.getByRole("link", { name: "Download samples.jsonl" }),
  ],
  "run-live-status": (page) => [
    page.getByRole("link", { name: "Export JSON" }),
    page.getByRole("tab", { name: "Evidence" }),
  ],
  "run-live-attribution": (page) => [
    page
      .getByRole("region", { name: "Step attributions table" })
      .getByRole("button", { name: "tool-1", exact: true }),
    page.getByRole("tab", { name: "Attribution" }),
  ],
  "live-attribution": (page) => [
    page
      .getByRole("region", { name: "Step attributions table" })
      .getByRole("button", { name: "tool-2", exact: true }),
    page.getByRole("tab", { name: "Attribution" }),
  ],
  live: (page, seeded) => [
    page.getByRole("link", { name: new RegExp(seeded.live.slice(0, 8)) }),
    page.getByRole("link", { name: "Open run page" }),
  ],
  compare: (page) => [
    page.getByRole("button", { name: "Swap baseline and candidate" }),
    page.getByRole("combobox", { name: /Baseline/ }).first(),
    page.getByRole("combobox", { name: /Candidate/ }).first(),
  ],
};

/** Scrolls into view inside the viewport; a pager may rightly be disabled on one page. */
async function expectReachable(locator: Locator) {
  await locator.scrollIntoViewIfNeeded();
  await expect(locator).toBeInViewport();
}

test.beforeEach(async ({ request }) => resetFixtureApi(request));

test("every page fits the viewport and keeps navigation and actions reachable", async ({
  page,
  request,
}, testInfo) => {
  const seeded = await seedWorkspace(request);
  const smallTargets: string[] = [];
  for (const audited of auditPages(seeded)) {
    await test.step(audited.name, async () => {
      await openAuditPage(page, audited);
      const { scrollWidth, clientWidth } = await pageOverflow(page);
      expect(
        scrollWidth,
        `${audited.name}: the page scrolls sideways (${scrollWidth} > ${clientWidth} px)`,
      ).toBeLessThanOrEqual(clientWidth);
      expect(await unscrollableOverflow(page), audited.name).toEqual([]);

      const navigation = page.getByRole("navigation", {
        name: "Main navigation",
      });
      for (const name of NAVIGATION)
        await expectReachable(
          navigation.getByRole("link", { name, exact: true }),
        );
      for (const action of PAGE_ACTIONS[audited.name]?.(page, seeded) ?? [])
        await expectReachable(action);

      if (testInfo.project.use.hasTouch)
        smallTargets.push(
          ...(
            await page.evaluate((minimum) => {
              const found: string[] = [];
              for (const element of document.querySelectorAll<HTMLElement>(
                "a[href],button,input,select,[role=tab],[role=combobox]",
              )) {
                // A control inside a <label> is hit anywhere on the label.
                const box = (
                  element.closest("label") ?? element
                ).getBoundingClientRect();
                if (box.width === 0 || box.height === 0) continue;
                if (box.width < minimum || box.height < minimum)
                  found.push(
                    `${element.tagName.toLowerCase()} "${(element.getAttribute("aria-label") ?? element.textContent ?? "").trim().slice(0, 40)}" ${Math.round(box.width)}×${Math.round(box.height)}`,
                  );
              }
              return found;
            }, TOUCH_TARGET_PX)
          ).map((target) => `${audited.name}: ${target}`),
        );
    });
  }
  // Reported for the team, not failed: AGENTS.md sets 44 px as an aim, not a hard rule.
  if (smallTargets.length > 0)
    await testInfo.attach("touch-targets-below-44px.txt", {
      body: [...new Set(smallTargets)].join("\n"),
      contentType: "text/plain",
    });
});
