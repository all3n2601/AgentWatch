export const THEME_STORAGE_KEY = "agentwatch.theme";
export const THEME_CHANGE_EVENT = "agentwatch:theme";
export const THEME_PREFERENCES = ["system", "light", "dark"] as const;

export type ThemePreference = (typeof THEME_PREFERENCES)[number];
export type ResolvedTheme = "light" | "dark";

export const DEFAULT_THEME_PREFERENCE: ThemePreference = "system";

/** Stored values are untrusted browser state; anything unrecognised means "system". */
export function parseThemePreference(raw: string | null): ThemePreference {
  return (
    THEME_PREFERENCES.find((preference) => preference === raw) ??
    DEFAULT_THEME_PREFERENCE
  );
}

export function resolveTheme(
  preference: ThemePreference,
  systemPrefersDark: boolean,
): ResolvedTheme {
  if (preference === "system") return systemPrefersDark ? "dark" : "light";
  return preference;
}

/**
 * Runs synchronously from the document head so the first paint already has the
 * right theme. It must stay in sync with `parseThemePreference` and
 * `resolveTheme`; `tests/theme.test.ts` executes it against both.
 */
export const THEME_INIT_SCRIPT = `(function(){try{var p=localStorage.getItem(${JSON.stringify(THEME_STORAGE_KEY)});var d=p==="dark"||(p!=="light"&&window.matchMedia("(prefers-color-scheme: dark)").matches);var r=document.documentElement;r.classList.toggle("dark",d);r.classList.toggle("light",!d);r.style.colorScheme=d?"dark":"light"}catch(e){}})()`;
