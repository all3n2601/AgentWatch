"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useTransition, type ReactNode } from "react";
import { Layers, RefreshCw, RotateCcw } from "lucide-react";
import { TabbedView } from "@/components/shell/tabbed-view";
import { Button } from "@/components/ui/button";
import { FieldRow, FieldSection } from "@/components/ui/field-row";
import { GlassPanel } from "@/components/ui/glass-panel";
import { SegmentedControl } from "@/components/ui/segmented-control";
import {
  KeyValueList,
  type KeyValueItem,
} from "@/components/widgets/evidence-lists";
import { SECTION_PATHS } from "@/lib/routes";
import {
  DEFAULT_PREFERENCES,
  INSIGHTS_ENV,
  parseRefreshOption,
  READINESS_TIMEOUT_MS,
  type ApiReadiness,
  type Density,
  type InsightsStatus,
  type SidebarState,
} from "@/lib/settings-page";
import type { ThemePreference } from "@/lib/theme";
import {
  useDisplayPreferences,
  useThemePreference,
} from "./use-settings-storage";
import { SETTINGS_TABS } from "./sections";

const THEME_OPTIONS = [
  { value: "system", label: "System" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
] as const satisfies readonly { value: ThemePreference; label: string }[];
const DENSITY_OPTIONS = [
  { value: "comfortable", label: "Comfortable" },
  { value: "compact", label: "Compact" },
] as const satisfies readonly { value: Density; label: string }[];
const SIDEBAR_OPTIONS = [
  { value: "expanded", label: "Expanded" },
  { value: "collapsed", label: "Collapsed" },
] as const satisfies readonly { value: SidebarState; label: string }[];
const REFRESH_OPTIONS = [
  { value: "0", label: "Manual" },
  { value: "15", label: "15 sec" },
  { value: "30", label: "30 sec" },
  { value: "60", label: "60 sec" },
] as const;

/** Server-side facts the form shows read-only; nothing secret crosses here. */
export type SettingsConnection = {
  /** The ingestion API base URL without credentials, or null if not a URL. */
  apiUrl: string | null;
  readiness: ApiReadiness | null;
  /** When the server ran the readiness check, as an ISO timestamp. */
  checkedAt: string;
  insights: InsightsStatus;
};

const TIMEOUT_SECONDS = READINESS_TIMEOUT_MS / 1000;

function readinessItems(connection: SettingsConnection): KeyValueItem[] {
  const { readiness } = connection;
  const status: KeyValueItem["status"] =
    readiness === null
      ? { label: "Not configured", tone: "warning" }
      : readiness.state === "connected"
        ? { label: "Connected", tone: "success" }
        : readiness.state === "storage-unavailable"
          ? { label: "Database unavailable", tone: "warning" }
          : readiness.state === "unexpected"
            ? { label: "Unexpected response", tone: "warning" }
            : { label: "Unavailable", tone: "danger" };
  const detail =
    readiness === null
      ? "AGENTWATCH_API_URL is not an http(s) URL."
      : readiness.state === "connected"
        ? "API and PostgreSQL storage ready"
        : readiness.state === "storage-unavailable"
          ? "The API answered, but its database is not ready"
          : readiness.state === "unexpected"
            ? `Readiness check returned HTTP ${readiness.httpStatus}`
            : `No response within ${TIMEOUT_SECONDS} s`;
  const checked = new Date(connection.checkedAt);
  return [
    {
      id: "api",
      label: "Ingestion API",
      value: (
        <span className="break-all">{connection.apiUrl ?? "Invalid URL"}</span>
      ),
      status,
    },
    { id: "readiness", label: "Readiness", value: detail },
    {
      id: "checked",
      label: "Last checked",
      value: (
        <time dateTime={connection.checkedAt}>
          {checked.toLocaleTimeString("en-GB", { timeZone: "UTC" })} UTC
        </time>
      ),
    },
  ];
}

function insightItems(insights: InsightsStatus): KeyValueItem[] {
  return [
    {
      id: "summaries",
      label: "AI summaries and chat",
      value: insights.enabled ? "Configured on the server" : "Not configured",
      status: insights.enabled
        ? { label: "Enabled", tone: "success" }
        : { label: "Disabled", tone: "info" },
    },
    {
      id: "model",
      label: "Model",
      value: insights.enabled ? (
        <span className="break-all">{insights.model}</span>
      ) : (
        "—"
      ),
    },
  ];
}

function CheckConnectionButton() {
  const router = useRouter();
  const [checking, startCheck] = useTransition();
  return (
    <Button
      variant="glass"
      size="pill"
      disabled={checking}
      aria-busy={checking}
      // The page is rendered per request, so a refresh reruns the check.
      onClick={() => startCheck(() => router.refresh())}
    >
      <RefreshCw
        className={checking ? "animate-spin motion-reduce:animate-none" : ""}
        aria-hidden
      />
      {checking ? "Checking…" : "Check connection"}
    </Button>
  );
}

type SettingsTab = (typeof SETTINGS_TABS)[number];

/** One tab's panel: a glass card holding that tab's labelled section. */
function TabSection({
  tab,
  children,
}: {
  tab: SettingsTab;
  children: ReactNode;
}) {
  return (
    <GlassPanel className="max-w-3xl">
      <FieldSection title={tab.label} id={tab.headingId}>
        {children}
      </FieldSection>
    </GlassPanel>
  );
}

/**
 * The settings tabs. This component is the single owner of the saved
 * preferences, so every tab edits the same state; saving notifies the
 * workspace shell, which applies the change at once.
 */
export function SettingsForm({
  connection,
  header,
  initialTab,
}: {
  connection: SettingsConnection;
  header: ReactNode;
  /** The `?view=` value from the URL; unknown values open the first tab. */
  initialTab?: string;
}) {
  const { preferences, save, saveError } = useDisplayPreferences();
  const theme = useThemePreference();
  const [workspace, display, data, insights] = SETTINGS_TABS;

  const tabs = [
    {
      id: workspace.id,
      label: workspace.label,
      content: (
        <TabSection tab={workspace}>
          <FieldRow
            label="Sidebar"
            labelId="settings-sidebar"
            hint="Expanded shows the labelled sidebar on every workspace page; Collapsed shows icons only."
          >
            <SegmentedControl
              labelledBy="settings-sidebar"
              value={preferences.sidebar}
              onValueChange={(sidebar) => save({ ...preferences, sidebar })}
              options={SIDEBAR_OPTIONS}
            />
          </FieldRow>
          <FieldRow
            label="Saved preferences"
            hint="Sidebar, density and refresh are stored in this browser only; other browsers and teammates keep their own."
          >
            <Button
              variant="glass"
              size="pill"
              onClick={() => save(DEFAULT_PREFERENCES)}
            >
              <RotateCcw aria-hidden />
              Reset display preferences
            </Button>
          </FieldRow>
        </TabSection>
      ),
    },
    {
      id: display.id,
      label: display.label,
      content: (
        <TabSection tab={display}>
          <FieldRow label="Theme" labelId="settings-theme">
            <SegmentedControl
              labelledBy="settings-theme"
              value={theme.preference}
              onValueChange={theme.select}
              options={THEME_OPTIONS}
            />
          </FieldRow>
          <FieldRow
            label="Density"
            labelId="settings-density"
            hint="Compact scales the whole interface down on every workspace page."
          >
            <SegmentedControl
              labelledBy="settings-density"
              value={preferences.density}
              onValueChange={(density) => save({ ...preferences, density })}
              options={DENSITY_OPTIONS}
            />
          </FieldRow>
        </TabSection>
      ),
    },
    {
      id: data.id,
      label: data.label,
      content: (
        <TabSection tab={data}>
          <KeyValueList items={readinessItems(connection)} className="mb-3" />
          <FieldRow
            label="Auto refresh"
            labelId="settings-refresh"
            hint="Reloads the open page's server data at this interval while the browser tab is visible. Runs still executing are not streamed."
          >
            <SegmentedControl
              labelledBy="settings-refresh"
              value={String(preferences.refresh)}
              onValueChange={(value) => {
                const refresh = parseRefreshOption(value);
                if (refresh !== null) save({ ...preferences, refresh });
              }}
              options={REFRESH_OPTIONS}
            />
          </FieldRow>
          <div className="mt-3 flex flex-wrap gap-3">
            <CheckConnectionButton />
            <Button asChild variant="glass" size="pill">
              <Link href={SECTION_PATHS.quality}>
                <Layers aria-hidden />
                View data quality
              </Link>
            </Button>
          </div>
        </TabSection>
      ),
    },
    {
      id: insights.id,
      label: insights.label,
      content: (
        <TabSection tab={insights}>
          <KeyValueList items={insightItems(connection.insights)} />
          <p className="mt-3 text-sm text-ink-muted">
            Set <code className="break-all">{INSIGHTS_ENV.url}</code> and{" "}
            <code className="break-all">{INSIGHTS_ENV.model}</code> on the
            server to turn on run summaries and the assistant, plus{" "}
            <code className="break-all">{INSIGHTS_ENV.apiKey}</code> if the
            endpoint needs a key. Summaries send computed aggregates only; the
            assistant also sends the run records it looks up, never prompts,
            code or tool arguments.
          </p>
        </TabSection>
      ),
    },
  ];

  return (
    <TabbedView
      label="Settings sections"
      initialTab={initialTab}
      tabs={tabs}
      header={
        <div className="space-y-3">
          {header}
          {saveError && (
            <p
              role="alert"
              className="max-w-3xl rounded-xl border border-danger-line bg-danger-soft px-4 py-3 text-sm text-danger"
            >
              {saveError}
            </p>
          )}
        </div>
      }
    />
  );
}
