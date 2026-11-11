"use client";

import { useCallback, useState, useSyncExternalStore } from "react";
import { usePreferences } from "@/components/shell/use-preferences";
import {
  PREFERENCES_CHANGE_EVENT,
  PREFERENCES_STORAGE_KEY,
  serializePreferences,
  type DisplayPreferences,
} from "@/lib/settings-page";
import {
  DEFAULT_THEME_PREFERENCE,
  parseThemePreference,
  resolveTheme,
  THEME_CHANGE_EVENT,
  THEME_STORAGE_KEY,
  type ThemePreference,
} from "@/lib/theme";

function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function subscribeTheme(onChange: () => void) {
  window.addEventListener(THEME_CHANGE_EVENT, onChange);
  window.addEventListener("storage", onChange);
  return () => {
    window.removeEventListener(THEME_CHANGE_EVENT, onChange);
    window.removeEventListener("storage", onChange);
  };
}

const readTheme = () => readStorage(THEME_STORAGE_KEY);
const noStoredValue = () => null;

const SAVE_FAILED =
  "This browser blocked saving preferences, so the change was not kept. Allow site storage and try again.";

/**
 * The workspace display preferences: read through the shell's
 * `usePreferences`, written to the same storage key. Saving dispatches
 * `PREFERENCES_CHANGE_EVENT`, which the shell (sidebar, density, refresh)
 * listens for, so a change applies immediately; other tabs see the `storage`
 * event.
 */
export function useDisplayPreferences() {
  const { preferences } = usePreferences();
  const [saveError, setSaveError] = useState<string | null>(null);
  const save = useCallback((next: DisplayPreferences) => {
    try {
      localStorage.setItem(PREFERENCES_STORAGE_KEY, serializePreferences(next));
      setSaveError(null);
    } catch {
      setSaveError(SAVE_FAILED);
      return;
    }
    window.dispatchEvent(new Event(PREFERENCES_CHANGE_EVENT));
  }, []);
  return { preferences, save, saveError };
}

const DARK_QUERY = "(prefers-color-scheme: dark)";

/**
 * The colour theme, through the same storage key, change event and `html`
 * classes as components/theme-toggle.tsx, which does not export its setter.
 * The top bar's ThemeToggle listens for the event, so both stay in step, and
 * it keeps "system" in sync with later OS changes.
 */
export function useThemePreference() {
  const raw = useSyncExternalStore(subscribeTheme, readTheme, noStoredValue);
  const preference: ThemePreference = raw
    ? parseThemePreference(raw)
    : DEFAULT_THEME_PREFERENCE;
  const select = useCallback((next: ThemePreference) => {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // Storage can be blocked; the theme then lasts until the next reload,
      // as with the top bar's toggle.
    }
    const dark =
      resolveTheme(next, window.matchMedia(DARK_QUERY).matches) === "dark";
    const root = document.documentElement;
    root.classList.toggle("dark", dark);
    root.classList.toggle("light", !dark);
    root.style.colorScheme = dark ? "dark" : "light";
    window.dispatchEvent(new Event(THEME_CHANGE_EVENT));
  }, []);
  return { preference, select };
}
