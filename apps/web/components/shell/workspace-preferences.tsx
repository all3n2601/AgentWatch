"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { usePreferences } from "@/components/shell/use-preferences";

/**
 * Applies the Settings page's display preferences to every workspace page:
 * density scales the interface (`data-density` on <html>, see
 * design-system.css), and a refresh interval re-renders the page's server data
 * while the tab is visible. Renders nothing.
 */
export function WorkspacePreferences() {
  const router = useRouter();
  const { preferences } = usePreferences();

  useEffect(() => {
    const root = document.documentElement;
    root.dataset.density = preferences.density;
    return () => {
      delete root.dataset.density;
    };
  }, [preferences.density]);

  useEffect(() => {
    if (preferences.refresh === 0) return;
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") router.refresh();
    }, preferences.refresh * 1000);
    return () => window.clearInterval(timer);
  }, [preferences.refresh, router]);

  return null;
}
