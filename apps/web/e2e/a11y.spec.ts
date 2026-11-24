import { expect, test } from "@playwright/test";
import {
  auditPages,
  openAuditPage,
  seedWorkspace,
  seriousA11yViolations,
  type AxeException,
} from "./audit";
import { resetFixtureApi } from "./fixture-control";

// F04: an axe-core WCAG 2.x A/AA scan of every page at every audit viewport. Serious and
// critical findings fail. An accepted finding is listed here by rule and node with its
// reason; none is disabled for a whole rule or page. Synthetic fixture runs only.

const EXCEPTIONS: readonly AxeException[] = [
  {
    // The step checksum's CopyChip (components/pages/run-detail/timeline-tab.tsx) shows the
    // first 8 characters of a one-word value while its name holds the whole value, so the
    // visible word is not in the name (WCAG 2.5.3). The fix belongs in the shared
    // components/ui/copy-chip.tsx; see "Findings for the team" in the F04 PR. UUID chips pass
    // because their first 8 characters are a whole hyphen-separated word.
    rule: "label-content-name-mismatch",
    target: /^button\[title="e2e-checksum"\]$/,
    reason: "Shared CopyChip truncation; tracked as an F04 team finding",
  },
];

test.beforeEach(async ({ request }) => resetFixtureApi(request));

test("no page has a serious or critical accessibility violation", async ({
  page,
  request,
}) => {
  const seeded = await seedWorkspace(request);
  const found: string[] = [];
  for (const audited of auditPages(seeded)) {
    await test.step(audited.name, async () => {
      await openAuditPage(page, audited);
      found.push(
        ...(await seriousA11yViolations(page, EXCEPTIONS)).map(
          (violation) => `${audited.name}: ${violation}`,
        ),
      );
    });
  }
  expect(found).toEqual([]);
});
