import type { Route } from "next";

/**
 * Every page URL the app links to, in one place. Literal paths are checked by
 * Next's typed routes (`typedRoutes` in next.config.ts); builders with
 * parameters return `Route`, the documented cast for computed hrefs, so a
 * renamed page breaks the typecheck instead of a link.
 */

/** Run page tabs (components/pages/run-detail/run-tabs.tsx). */
export const RUN_TABS = [
  "diagnosis",
  "summary",
  "timeline",
  "agent",
  "resources",
  "evidence",
  // A live run's connection, state and totals; saved runs have no such tab.
  "status",
] as const;
export type RunTab = (typeof RUN_TABS)[number];

/** Dashboard sections that have their own top-level route. */
export const SECTION_PATHS = {
  overview: "/",
  runs: "/runs",
  experiments: "/experiments",
  compare: "/compare",
  quality: "/quality",
  controls: "/controls",
  settings: "/settings",
  live: "/live",
} as const satisfies Record<string, Route>;

export type Section = keyof typeof SECTION_PATHS;

const withQuery = (path: string, query: Record<string, string | undefined>) => {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query))
    if (value) params.set(key, value);
  const search = params.toString();
  return (search ? `${path}?${search}` : path) as Route;
};

export const routes = {
  section: (section: Section): Route => SECTION_PATHS[section],
  overview: (model?: string) => withQuery("/", { model }),
  runs: (query: { q?: string; liveBefore?: string } = {}) =>
    withQuery("/runs", query),
  run: (runId: string, tab?: RunTab) =>
    withQuery(`/runs/${encodeURIComponent(runId)}`, { tab }),
  compare: (run?: string, baseline?: string) =>
    withQuery("/compare", { run, baseline }),
  live: (liveRun?: string) => withQuery("/live", { liveRun }),
  liveExport: (runId: string) =>
    `/api/live/runs/${encodeURIComponent(runId)}/export`,
};
