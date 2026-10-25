import os from "node:os";
import path from "node:path";
import { defineConfig, devices } from "@playwright/test";
import { FIXTURE_API_PORT, WEB_PORT } from "./e2e/fixture-control";

// Browser tests of the live view against a scripted live API (e2e/fixture-api.ts). They run
// the production build, so run `pnpm build` first; CI builds before this step.

const fixtureApiUrl = `http://127.0.0.1:${FIXTURE_API_PORT}`;
const webUrl = `http://127.0.0.1:${WEB_PORT}`;

export default defineConfig({
  testDir: "./e2e",
  testMatch: "**/*.spec.ts",
  // The tests share one fixture API, whose run list is global, so they run one at a time.
  fullyParallel: false,
  workers: 1,
  // A failure is reported, never retried into a pass.
  retries: 0,
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: { baseURL: webUrl, trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: "node --import tsx e2e/fixture-api.ts",
      url: `${fixtureApiUrl}/v1/live/runs`,
      reuseExistingServer: false,
    },
    {
      command: `pnpm exec next start --port ${WEB_PORT} --hostname 127.0.0.1`,
      url: `${webUrl}/api/live/runs`,
      reuseExistingServer: false,
      env: {
        AGENTWATCH_API_URL: fixtureApiUrl,
        // No local artifacts or GPU probes: the page must show only fixture API data.
        AGENTWATCH_DATA_DIR: path.join(
          os.tmpdir(),
          "agentwatch-e2e-no-local-data",
        ),
      },
    },
  ],
});
