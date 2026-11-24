import AxeBuilder from "@axe-core/playwright";
import { expect, type APIRequestContext, type Page } from "@playwright/test";
import { registerRun, registerSavedRun } from "./fixture-control";
import {
  fixtureRun,
  newRunId,
  resourceSample,
  snapshot,
  stepEvent,
} from "./live-runs";
import { newSavedRunId, savedRun } from "./saved-runs";

// Shared by the F04 audit specs (responsive, a11y, keyboard) and the evidence run. Every run
// they register is synthetic browser-test data, never recorded evidence.

export type SeededWorkspace = {
  /** Two saved runs with matching settings (older is the baseline) and an agent model. */
  baseline: string;
  candidate: string;
  /** A live run with completed, running and queued steps and resource samples. */
  live: string;
};

/** A live run's snapshot: two completed tools, one running and one still queued. */
function busyLiveRun(runId: string) {
  const step = (id: string, finished: number | null, offset: number) => {
    const queued = stepEvent(id, 0, "queued", { runId, second: offset });
    const running = stepEvent(id, 1, "running", {
      runId,
      started: offset + 1,
      second: offset + 1,
    });
    const done =
      finished === null
        ? null
        : stepEvent(id, 2, "completed", {
            runId,
            started: offset + 1,
            finished,
            cpu: 0.4,
            second: finished,
          });
    return { queued, running, done };
  };
  const first = step("tool-1", 4, 0);
  const second = step("tool-2", 7, 3);
  const third = step("tool-3", null, 6);
  const waiting = stepEvent("tool-4", 0, "queued", { runId, second: 8 });
  const samples = [0, 2, 4, 6, 8].map((second) =>
    resourceSample(`cpu-${second}`, { runId, value: 20 + second * 3, second }),
  );
  return snapshot(runId, {
    steps: [first.done!, second.done!, third.running, waiting],
    records: [
      first.queued,
      first.running,
      first.done!,
      second.queued,
      second.running,
      second.done!,
      third.queued,
      third.running,
      waiting,
      ...samples,
    ],
  });
}

/** Registers the runs every audited page needs and returns their UUIDs. */
export async function seedWorkspace(
  request: APIRequestContext,
): Promise<SeededWorkspace> {
  const baseline = newSavedRunId();
  const candidate = newSavedRunId();
  const live = newRunId();
  await registerSavedRun(
    request,
    savedRun(baseline, { sequence: 1, model: "qwen3:1.7b" }),
  );
  await registerSavedRun(
    request,
    savedRun(candidate, {
      sequence: 2,
      cleanSeconds: 4.5,
      model: "qwen3:1.7b",
    }),
  );
  await registerRun(request, fixtureRun(live, [busyLiveRun(live)]));
  return { baseline, candidate, live };
}

const controlledServices = {
  inference_requests: { service_id: "ollama", resource_class: "inference" },
  test_jobs: {
    service_id: "agentwatch-agent-runner",
    resource_class: "cpu_sandbox",
  },
};

/** One active policy version, as the web policy routes return it. */
const activePolicy = {
  target: "coding-agent",
  version: 1,
  action: "set",
  limits: { test_jobs: 2 },
  restored_from: null,
  operator: "reviewer",
  reason: "Audit fixture",
  request_id: "6f1c1d1e-8a63-4f43-9c3c-000000000001",
  created_at: "2026-11-03T06:30:29.001256Z",
  controlled_services: controlledServices,
  enforcement_scope: "runner_process",
};

/**
 * Answers the policy routes with one active version, so Controls and the run policy panel
 * render their normal state instead of "unavailable". Policy changes are refused.
 */
export async function servePolicy(page: Page) {
  const controls = {
    active: activePolicy,
    history: { versions: [activePolicy], next_before: null },
  };
  await page.route("**/api/policies/concurrency", (route) =>
    route.request().method() === "GET"
      ? route.fulfill({ json: controls })
      : route.abort(),
  );
  await page.route("**/api/policies/concurrency/current", (route) =>
    route.fulfill({ json: controls }),
  );
}

/** A run's row in the Runs list, whose button names the full run id. */
export const runRowButton = (page: Page, runId: string) =>
  page.getByRole("button", { name: `(run ${runId})` });

export type AuditPage = {
  name: string;
  path: string;
  /** Visible once the page has rendered its data, not just its loading state. */
  ready: (page: Page) => ReturnType<Page["getByRole"]>;
};

/** Every user-facing page, with the seeded runs selected where a page shows one run. */
export function auditPages(seeded: SeededWorkspace): AuditPage[] {
  const heading = (name: string | RegExp) => (page: Page) =>
    page.getByRole("heading", { level: 1, name });
  return [
    { name: "overview", path: "/", ready: heading(/./) },
    {
      name: "runs",
      path: "/runs",
      ready: (page) => runRowButton(page, seeded.candidate),
    },
    {
      name: "run-saved-diagnosis",
      path: `/runs/${seeded.candidate}`,
      ready: (page) => page.getByRole("tab", { name: "Diagnosis" }),
    },
    {
      name: "run-saved-timeline",
      path: `/runs/${seeded.candidate}?tab=timeline`,
      ready: (page) =>
        page.getByRole("tab", { name: "Timeline", selected: true }),
    },
    {
      name: "run-saved-agent",
      path: `/runs/${seeded.candidate}?tab=agent`,
      ready: (page) => page.getByRole("tab", { name: "Agent", selected: true }),
    },
    {
      name: "run-saved-resources",
      path: `/runs/${seeded.candidate}?tab=resources`,
      ready: (page) =>
        page.getByRole("tab", { name: "Resources", selected: true }),
    },
    {
      name: "run-saved-evidence",
      path: `/runs/${seeded.candidate}?tab=evidence`,
      ready: (page) =>
        page.getByRole("link", { name: "Download results.json" }),
    },
    {
      name: "run-live-status",
      path: `/runs/${seeded.live}?tab=status`,
      ready: (page) =>
        page.getByRole("tab", { name: "Live status", selected: true }),
    },
    {
      name: "run-live-timeline",
      path: `/runs/${seeded.live}?tab=timeline`,
      ready: (page) => page.getByText("Steps over time").first(),
    },
    {
      name: "live",
      path: `/live?liveRun=${seeded.live}`,
      ready: (page) => page.getByText(/records? received for run/),
    },
    {
      name: "compare",
      path: `/compare?run=${seeded.candidate}&baseline=${seeded.baseline}`,
      ready: (page) =>
        page.getByRole("button", {
          name: `Copy Candidate run ID ${seeded.candidate}`,
        }),
    },
    {
      name: "controls",
      path: "/controls",
      ready: (page) => page.getByText(/v1/).first(),
    },
    { name: "experiments", path: "/experiments", ready: heading(/./) },
    { name: "quality", path: "/quality", ready: heading(/./) },
    { name: "settings", path: "/settings", ready: heading(/./) },
    {
      name: "not-found",
      path: "/runs/00000000-0000-4000-8000-000000000000",
      ready: heading(/./),
    },
  ];
}

export async function openAuditPage(page: Page, audited: AuditPage) {
  await servePolicy(page);
  await page.goto(audited.path);
  await expect(audited.ready(page)).toBeVisible();
}

/** Page-level horizontal scroll: the document is wider than the viewport. */
export function pageOverflow(page: Page) {
  return page.evaluate(() => {
    const root = document.documentElement;
    return { scrollWidth: root.scrollWidth, clientWidth: root.clientWidth };
  });
}

/**
 * Elements wider than their box that no labelled, keyboard-reachable horizontal scroll
 * region contains. Content may scroll inside such a region (AGENTS.md), never off-page.
 */
export function unscrollableOverflow(page: Page) {
  return page.evaluate(() => {
    const scrolls = (element: Element) => {
      const { overflowX } = getComputedStyle(element);
      return overflowX === "auto" || overflowX === "scroll";
    };
    const labelled = (element: Element) =>
      element.hasAttribute("aria-label") ||
      element.hasAttribute("aria-labelledby") ||
      element.closest("[aria-label],[aria-labelledby]") !== null;
    const reachable = (element: Element) =>
      element.matches("[tabindex]:not([tabindex='-1'])") ||
      element.querySelector(
        "a[href],button,input,select,textarea,[tabindex]:not([tabindex='-1'])",
      ) !== null;
    const describe = (element: Element) =>
      `${element.tagName.toLowerCase()}${element.id ? `#${element.id}` : ""}${
        element.getAttribute("aria-label")
          ? `[aria-label="${element.getAttribute("aria-label")}"]`
          : ""
      }`;
    const problems: string[] = [];
    for (const box of document.querySelectorAll("body *")) {
      if (!(box instanceof HTMLElement)) continue;
      if (
        box.offsetParent === null &&
        getComputedStyle(box).position !== "fixed"
      )
        continue;
      // Sub-pixel rounding is not overflow.
      if (box.scrollWidth <= box.clientWidth + 1 || !scrolls(box)) continue;
      if (!labelled(box) || !reachable(box))
        problems.push(
          `${describe(box)} scrolls horizontally without a label or keyboard access`,
        );
    }
    return problems;
  });
}

/**
 * An axe finding accepted on purpose: one rule on the nodes matching `target`, with the
 * reason and where the team tracks it. Never a whole rule, never a whole page.
 */
export type AxeException = { rule: string; target: RegExp; reason: string };

/** WCAG 2.x A and AA. "serious" and "critical" findings fail the test. */
export async function seriousA11yViolations(
  page: Page,
  exceptions: readonly AxeException[] = [],
) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"])
    .analyze();
  return results.violations
    .filter((violation) =>
      ["serious", "critical"].includes(violation.impact ?? ""),
    )
    .flatMap((violation) =>
      violation.nodes
        .filter(
          (node) =>
            !exceptions.some(
              (exception) =>
                exception.rule === violation.id &&
                exception.target.test(node.target.join(" ")),
            ),
        )
        .map(
          (node) =>
            `${violation.id} (${violation.impact}) at ${node.target.join(" ")}: ${node.failureSummary?.split("\n").join(" ")}`,
        ),
    );
}
