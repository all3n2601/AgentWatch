"use client";

import { useSyncExternalStore } from "react";
import {
  DEFAULT_PREFERENCES,
  parsePreferences,
  PREFERENCES_CHANGE_EVENT,
  PREFERENCES_STORAGE_KEY,
  type DisplayPreferences,
} from "@/lib/settings-page";

function subscribe(onChange: () => void) {
  window.addEventListener(PREFERENCES_CHANGE_EVENT, onChange);
  window.addEventListener("storage", onChange);
  return () => {
    window.removeEventListener(PREFERENCES_CHANGE_EVENT, onChange);
    window.removeEventListener("storage", onChange);
  };
}

/** The raw stored string, so the snapshot is stable between renders. */
function readRaw(): string | null {
  try {
    return localStorage.getItem(PREFERENCES_STORAGE_KEY);
  } catch {
    // Storage can be blocked; the defaults apply.
    return null;
  }
}

const parsed = new Map<string | null, DisplayPreferences>();
function cachedParse(raw: string | null): DisplayPreferences {
  let value = parsed.get(raw);
  if (!value) {
    value = parsePreferences(raw);
    parsed.set(raw, value);
  }
  return value;
}

/**
 * Display preferences saved by the Settings page (density, sidebar, refresh),
 * kept in sync across components and tabs. The server render uses defaults.
 */
export function usePreferences(): { preferences: DisplayPreferences } {
  const raw = useSyncExternalStore(subscribe, readRaw, () => null);
  return {
    preferences: raw === null ? DEFAULT_PREFERENCES : cachedParse(raw),
  };
}
