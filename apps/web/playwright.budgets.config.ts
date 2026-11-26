import { defineConfig, devices } from "@playwright/test";
import base from "./playwright.config";

// F04 large-run budgets (e2e/large-runs.budget.ts): timings of the large-run flows against
// the budgets declared before the first measurement. Not part of the required browser suite,
// whose large-runs.spec.ts runs the same flows without timing them: the budgets were set on a
// development machine, and CI hardware misses some of them (docs/project-checklist.md, F04).
// Run with `pnpm test:budgets` after `pnpm build`; it uses the same fixture API and
// production build as the browser tests. Report the printed measurements with the machine
// they were taken on.

export default defineConfig({
  ...base,
  testMatch: "**/*.budget.ts",
  outputDir: "test-results/budgets",
  projects: [{ name: "budgets", use: { ...devices["Desktop Chrome"] } }],
});
