import { defineConfig, devices } from "@playwright/test";
import base from "./playwright.config";

// F04 handoff evidence: screenshots of each audited flow at desktop, tablet and phone widths,
// an INDEX.md describing them, and an HTML report, all under f04-evidence/ (gitignored).
// Run with `pnpm exec playwright test -c playwright.evidence.config.ts` after `pnpm build`;
// it uses the same fixture API and production build as the browser tests.

export default defineConfig({
  ...base,
  testMatch: "**/*.evidence.ts",
  outputDir: "test-results/f04-evidence",
  // No trace viewer in the report: f04-evidence/ is not ignored by `eslint .`, which would
  // lint the viewer's bundled scripts. The browser suite keeps traces for failures.
  use: { ...base.use, trace: "off" },
  reporter: [
    ["list"],
    ["html", { open: "never", outputFolder: "f04-evidence/report" }],
  ],
  projects: [{ name: "evidence", use: { ...devices["Desktop Chrome"] } }],
});
