import type { Metadata } from "next";
import type { SettingsConnection } from "@/components/pages/settings/settings-form";
import { SettingsPage } from "@/components/pages/settings/settings-page";
import { apiUrl } from "@/lib/experiments";
import { insightConfig } from "@/lib/insights";
import {
  checkApiReadiness,
  displayServiceUrl,
  insightsStatus,
} from "@/lib/settings-page";
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Settings",
  description: "Display preferences and the data connection",
};

/**
 * Reads the connection and insights configuration on the server and passes
 * the browser only what is safe to show: the API URL without credentials, its
 * readiness, and whether insights are on with which model.
 */
async function loadConnection(): Promise<SettingsConnection> {
  const configured = apiUrl();
  const shown = displayServiceUrl(configured);
  return {
    apiUrl: shown,
    readiness: shown ? await checkApiReadiness(configured) : null,
    checkedAt: new Date().toISOString(),
    insights: insightsStatus(insightConfig(process.env)),
  };
}

export default async function Page({
  searchParams,
}: {
  searchParams: Promise<{ view?: string | string[] }>;
}) {
  const { view } = await searchParams;
  return (
    <SettingsPage
      connection={await loadConnection()}
      initialTab={typeof view === "string" ? view : undefined}
    />
  );
}
