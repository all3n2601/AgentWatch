import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { parseEnv } from "node:util";
import type { NextConfig } from "next";

/**
 * The repo-root `.env` (shared with Compose and the Python services) holds the
 * web app's settings too, but Next only reads env files in `apps/web`. Copy in
 * just the keys the web server reads; values already set (exported, or from
 * `apps/web/.env.local`) win. Server-only: none is `NEXT_PUBLIC_`.
 */
const ROOT_ENV_KEYS = [
  "AGENTWATCH_API_URL",
  "AGENTWATCH_DATA_DIR",
  "AGENTWATCH_DATASET_DIR",
  "AGENTWATCH_IMPORT_MAP",
  "AGENTWATCH_INSIGHTS_URL",
  "AGENTWATCH_INSIGHTS_MODEL",
  "AGENTWATCH_INSIGHTS_API_KEY",
] as const;

function loadRootEnv(file = path.resolve(process.cwd(), "../../.env")) {
  if (!existsSync(file)) return;
  const values = parseEnv(readFileSync(file, "utf8"));
  for (const key of ROOT_ENV_KEYS)
    if (!process.env[key] && values[key]) process.env[key] = values[key];
}
loadRootEnv();

const UUID =
  "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}";

/**
 * Pages that became tabs of the Run page or moved. With `?run=<uuid>` an old
 * link opens that run's tab; without one it lands on the page that replaced
 * it. Temporary (307), so the targets can still change.
 */
const MOVED_TO_RUN_TABS = [
  ["/traces", "timeline"],
  ["/agents", "agent"],
  ["/resources", "resources"],
] as const;

const nextConfig: NextConfig = {
  transpilePackages: ["@agentwatch/ui"],
  // Links are checked against the app's routes at typecheck time (lib/routes.ts).
  typedRoutes: true,
  async redirects() {
    return [
      { source: "/overview", destination: "/", permanent: false },
      { source: "/dataset", destination: "/quality", permanent: false },
      ...MOVED_TO_RUN_TABS.flatMap(([source, tab]) => [
        {
          source,
          has: [
            { type: "query" as const, key: "run", value: `(?<run>${UUID})` },
          ],
          destination: `/runs/:run?tab=${tab}`,
          permanent: false,
        },
        { source, destination: "/runs", permanent: false },
      ]),
    ];
  },
};

export default nextConfig;
