import { expect, test, type Page } from "@playwright/test";
import { runRowButton, seedWorkspace, servePolicy } from "./audit";
import { resetFixtureApi } from "./fixture-control";

// F04: every audited flow works from the keyboard alone, with a visible focus indicator.
// Synthetic fixture runs only.

test.beforeEach(async ({ page, request }) => {
  await resetFixtureApi(request);
  await servePolicy(page);
});

/** Which part of the frame the focused element is in, and whether focus is visible. */
async function focusedStop(page: Page) {
  return page.evaluate(() => {
    const element = document.activeElement;
    if (!(element instanceof HTMLElement) || element === document.body)
      return null;
    const region = element.closest("header")
      ? "banner"
      : element.closest("nav[aria-label='Main navigation']")
        ? "navigation"
        : element.closest("aside")
          ? "complementary"
          : element.closest("main")
            ? "main"
            : "other";
    const name =
      element.getAttribute("aria-label") ??
      element.textContent?.trim().slice(0, 40) ??
      "";
    // Compare the focused look with the unfocused look: a ring that is always drawn
    // (a glass shadow) is not a focus indicator.
    const look = () => {
      const style = getComputedStyle(element);
      return `${style.outlineStyle} ${style.outlineWidth} ${style.boxShadow} ${style.borderColor}`;
    };
    const focused = look();
    element.blur();
    const unfocused = look();
    element.focus({ focusVisible: true });
    return { region, name, visible: focused !== unfocused };
  });
}

/** Tabs through the page from the top, recording each stop. */
async function tabStops(page: Page, count: number) {
  const stops = [];
  for (let index = 0; index < count; index += 1) {
    await page.keyboard.press("Tab");
    const stop = await focusedStop(page);
    if (stop) stops.push(stop);
  }
  return stops;
}

test("every focus stop on the main pages shows a visible focus indicator", async ({
  page,
  request,
}) => {
  const seeded = await seedWorkspace(request);
  for (const path of [
    "/runs",
    `/runs/${seeded.candidate}?tab=evidence`,
    `/live?liveRun=${seeded.live}`,
    `/compare?run=${seeded.candidate}&baseline=${seeded.baseline}`,
    "/controls",
  ]) {
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1 }).first()).toBeVisible();
    const hidden = (await tabStops(page, 40))
      .filter((stop) => !stop.visible)
      .map((stop) => `${stop.region}: ${stop.name}`);
    expect(hidden, path).toEqual([]);
  }
});

test("the skip link is the first stop and moves keyboard focus into the page content", async ({
  page,
}) => {
  // <main id="main"> has no tabIndex={-1}; Chromium still moves the sequential focus
  // starting point to it, which this checks.
  await page.goto("/runs");
  await page.keyboard.press("Tab");
  const skip = page.getByRole("link", { name: "Skip to content" });
  await expect(skip).toBeFocused();
  await expect(skip).toBeInViewport();
  await page.keyboard.press("Enter");
  await page.keyboard.press("Tab");
  expect((await focusedStop(page))?.region).toBe("main");
});

test("at desktop width focus reaches the page content before the assistant beside it", async ({
  page,
  request,
}) => {
  // Known failure, F04 finding 3: WorkspacePage (components/shell/app-shell.tsx) renders the
  // assistant rail before <main> in the DOM and moves it right with CSS, so focus enters the
  // assistant first. The shell is Allen's and is not changed in F04. test.fail() reports
  // this test as passing while the order is wrong, and fails once it is fixed, so the
  // marker is removed with the fix.
  test.fail();
  const seeded = await seedWorkspace(request);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(`/runs/${seeded.candidate}`);
  await expect(page.getByRole("tab", { name: "Diagnosis" })).toBeVisible();
  const order = (await tabStops(page, 60))
    .map((stop) => stop.region)
    .filter((region, index, all) => region !== all[index - 1]);
  // The visual order at xl: header, sidebar, page content, then the assistant on the right.
  expect(order.indexOf("main")).toBeGreaterThan(-1);
  expect(order.indexOf("main")).toBeLessThan(order.indexOf("complementary"));
});

test("run tabs follow the arrow keys and keep the tab in the address", async ({
  page,
  request,
}) => {
  const seeded = await seedWorkspace(request);
  await page.goto(`/runs/${seeded.candidate}`);
  const tab = (name: string) => page.getByRole("tab", { name, exact: true });
  await tab("Diagnosis").focus();
  await page.keyboard.press("ArrowRight");
  await expect(tab("Summary and scope")).toBeFocused();
  await expect(tab("Summary and scope")).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(page).toHaveURL(/tab=summary/);
  await page.keyboard.press("End");
  await expect(tab("Evidence")).toHaveAttribute("aria-selected", "true");
  await expect(page).toHaveURL(/tab=evidence/);
  await page.keyboard.press("Home");
  await expect(tab("Diagnosis")).toHaveAttribute("aria-selected", "true");
  // The panel body is reachable to scroll from the keyboard.
  await page.keyboard.press("Tab");
  await expect(page.locator(":focus")).toBeVisible();
});

test("a Runs filter is chosen and dismissed from the keyboard, returning focus to it", async ({
  page,
  request,
}) => {
  const seeded = await seedWorkspace(request);
  await page.goto("/runs");
  const source = page.getByRole("combobox", { name: "Source", exact: true });
  await source.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("listbox")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("listbox")).toBeHidden();
  await expect(source).toBeFocused();
  await expect(source).toContainText("All sources");

  await page.keyboard.press("Space");
  await page.getByRole("option", { name: "Live" }).focus();
  await page.keyboard.press("Enter");
  await expect(source).toBeFocused();
  await expect(source).toContainText("Live");
  await expect(page).toHaveURL(/source=live/);
  await expect(runRowButton(page, seeded.live)).toBeVisible();
  await expect(runRowButton(page, seeded.candidate)).toHaveCount(0);
});

test("runs are previewed, ticked and compared from the keyboard", async ({
  page,
  request,
}) => {
  const seeded = await seedWorkspace(request);
  // Below xl the preview is under the list, so choosing a run moves focus to it.
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/runs");
  const row = runRowButton(page, seeded.baseline);
  await row.focus();
  await page.keyboard.press("Enter");
  await expect(row).toHaveAttribute("aria-pressed", "true");
  const preview = page.locator("#" + (await row.getAttribute("aria-controls")));
  await expect(preview.locator(":focus")).toHaveCount(1);
  await expect(preview).toContainText(seeded.baseline.slice(0, 8));

  for (const id of [seeded.baseline, seeded.candidate]) {
    const box = page.getByRole("checkbox", {
      name: new RegExp(id.slice(0, 8)),
    });
    await box.focus();
    await page.keyboard.press("Space");
    await expect(box).toBeChecked();
  }
  await page.getByRole("link", { name: "Compare runs" }).first().focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(
    new RegExp(
      `/compare\\?run=${seeded.candidate}&baseline=${seeded.baseline}`,
    ),
  );
});

test("a live run is chosen and exported from the keyboard", async ({
  page,
  request,
}) => {
  const seeded = await seedWorkspace(request);
  await page.goto("/live");
  const link = page.getByRole("link", {
    name: new RegExp(seeded.live.slice(0, 8)),
  });
  await link.focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(new RegExp(`liveRun=${seeded.live}`));
  await page.getByRole("link", { name: "Open run page" }).focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(new RegExp(`/runs/${seeded.live}`));
  const exportLink = page.getByRole("link", { name: "Export JSON" });
  await exportLink.focus();
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.keyboard.press("Enter"),
  ]);
  expect(download.suggestedFilename()).toContain(seeded.live);
});
