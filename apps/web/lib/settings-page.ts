import { z } from "zod";

/**
 * Settings page logic with no React or server dependencies, so it can be
 * tested directly.
 *
 * The display preferences are read by the workspace shell
 * (components/shell/use-preferences.ts) to set the sidebar, density and
 * refresh interval on every workspace page.
 */

export const PREFERENCES_STORAGE_KEY = "agentwatch.preferences";
export const PREFERENCES_CHANGE_EVENT = "agentwatch:preferences";

export const DENSITIES = ["comfortable", "compact"] as const;
export const SIDEBAR_STATES = ["expanded", "collapsed"] as const;
/** Page data refresh intervals in seconds; 0 means manual refresh only. */
export const REFRESH_INTERVALS_SECONDS = [0, 15, 30, 60] as const;

export type Density = (typeof DENSITIES)[number];
export type SidebarState = (typeof SIDEBAR_STATES)[number];
export type RefreshIntervalSeconds = (typeof REFRESH_INTERVALS_SECONDS)[number];

export type DisplayPreferences = {
  density: Density;
  refresh: RefreshIntervalSeconds;
  sidebar: SidebarState;
};

export const DEFAULT_PREFERENCES: DisplayPreferences = {
  density: "comfortable",
  refresh: 0,
  sidebar: "expanded",
};

const isOneOf = <T>(options: readonly T[], value: unknown): value is T =>
  options.some((option) => option === value);

/**
 * Stored preferences are untrusted browser state. A field saved before it
 * existed falls back to its default; malformed JSON or any invalid field
 * resets every field, as the dashboard does.
 */
export function parsePreferences(raw: string | null): DisplayPreferences {
  if (!raw) return { ...DEFAULT_PREFERENCES };
  let stored: unknown;
  try {
    stored = JSON.parse(raw);
  } catch {
    return { ...DEFAULT_PREFERENCES };
  }
  if (typeof stored !== "object" || stored === null || Array.isArray(stored))
    return { ...DEFAULT_PREFERENCES };
  const merged: Record<string, unknown> = { ...DEFAULT_PREFERENCES, ...stored };
  const { density, refresh, sidebar } = merged;
  if (
    !isOneOf(DENSITIES, density) ||
    !isOneOf(REFRESH_INTERVALS_SECONDS, refresh) ||
    !isOneOf(SIDEBAR_STATES, sidebar)
  )
    return { ...DEFAULT_PREFERENCES };
  return { density, refresh, sidebar };
}

/** The exact JSON the dashboard writes: three fields, in this order. */
export function serializePreferences(preferences: DisplayPreferences): string {
  return JSON.stringify({
    density: preferences.density,
    refresh: preferences.refresh,
    sidebar: preferences.sidebar,
  });
}

/** Segmented controls carry string values; this maps one back to seconds. */
export function parseRefreshOption(
  value: string,
): RefreshIntervalSeconds | null {
  const seconds = Number(value);
  return isOneOf(REFRESH_INTERVALS_SECONDS, seconds) ? seconds : null;
}

/**
 * A service URL safe to show in the browser: scheme, host, port and path only.
 * User info, query and fragment are dropped because they can carry
 * credentials. Returns null for a value that is not an http(s) URL.
 */
export function displayServiceUrl(raw: string | undefined): string | null {
  if (!raw?.trim()) return null;
  let url: URL;
  try {
    url = new URL(raw.trim());
  } catch {
    return null;
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") return null;
  const path = url.pathname.replace(/\/+$/, "");
  return `${url.protocol}//${url.host}${path}`;
}

export type ApiReadiness =
  | { state: "connected" }
  | { state: "storage-unavailable" }
  | { state: "unexpected"; httpStatus: number }
  | { state: "unreachable" };

/** Matches the worker's `GET /readyz` (services/worker/src/agentwatch_worker/app.py). */
const readySchema = z.object({ status: z.literal("ready") });

/** The ingestion API reports 503 when PostgreSQL is not reachable. */
const STORAGE_UNAVAILABLE_STATUS = 503;

export const READINESS_TIMEOUT_MS = 3000;

/**
 * Asks the ingestion API whether it and its database are ready. Bounded by
 * `timeoutMs`; a timeout or network error is "unreachable", never "connected".
 */
export async function checkApiReadiness(
  baseUrl: string,
  fetchImpl: typeof fetch = fetch,
  timeoutMs: number = READINESS_TIMEOUT_MS,
): Promise<ApiReadiness> {
  let response: Response;
  try {
    response = await fetchImpl(`${baseUrl.replace(/\/+$/, "")}/readyz`, {
      cache: "no-store",
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch {
    return { state: "unreachable" };
  }
  if (response.status === STORAGE_UNAVAILABLE_STATUS)
    return { state: "storage-unavailable" };
  if (!response.ok) return { state: "unexpected", httpStatus: response.status };
  try {
    return readySchema.safeParse(await response.json()).success
      ? { state: "connected" }
      : { state: "unexpected", httpStatus: response.status };
  } catch {
    return { state: "unexpected", httpStatus: response.status };
  }
}

/** Environment variables that turn on AI summaries and the assistant chat. */
export const INSIGHTS_ENV = {
  url: "AGENTWATCH_INSIGHTS_URL",
  model: "AGENTWATCH_INSIGHTS_MODEL",
  apiKey: "AGENTWATCH_INSIGHTS_API_KEY",
} as const;

/** What the browser may know about the insights configuration. */
export type InsightsStatus =
  { enabled: true; model: string } | { enabled: false };

/** Keeps only the on/off state and model name; the URL and key stay on the server. */
export function insightsStatus(
  config: { model: string } | null,
): InsightsStatus {
  return config ? { enabled: true, model: config.model } : { enabled: false };
}
