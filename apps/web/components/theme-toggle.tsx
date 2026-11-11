"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useCallback, useEffect, useSyncExternalStore } from "react";
import {
  DEFAULT_THEME_PREFERENCE,
  parseThemePreference,
  resolveTheme,
  THEME_CHANGE_EVENT,
  THEME_STORAGE_KEY,
  type ThemePreference,
} from "@/lib/theme";
import { cn } from "@/lib/utils";

const options = [
  { value: "system", label: "System theme", icon: Monitor },
  { value: "light", label: "Light theme", icon: Sun },
  { value: "dark", label: "Dark theme", icon: Moon },
] as const satisfies readonly {
  value: ThemePreference;
  label: string;
  icon: typeof Sun;
}[];

const DARK_QUERY = "(prefers-color-scheme: dark)";

function readPreference(): ThemePreference {
  try {
    return parseThemePreference(localStorage.getItem(THEME_STORAGE_KEY));
  } catch {
    return DEFAULT_THEME_PREFERENCE;
  }
}

function subscribe(onChange: () => void) {
  window.addEventListener(THEME_CHANGE_EVENT, onChange);
  window.addEventListener("storage", onChange);
  return () => {
    window.removeEventListener(THEME_CHANGE_EVENT, onChange);
    window.removeEventListener("storage", onChange);
  };
}

function applyTheme(preference: ThemePreference) {
  const dark =
    resolveTheme(preference, window.matchMedia(DARK_QUERY).matches) === "dark";
  const root = document.documentElement;
  root.classList.toggle("dark", dark);
  root.classList.toggle("light", !dark);
  root.style.colorScheme = dark ? "dark" : "light";
}

/**
 * `compact` is a single button that cycles themes, for the collapsed sidebar
 * rail and the glass top bar; `className` restyles that button.
 */
export function ThemeToggle({
  compact = false,
  className,
}: {
  compact?: boolean;
  className?: string;
}) {
  const preference = useSyncExternalStore(
    subscribe,
    readPreference,
    () => DEFAULT_THEME_PREFERENCE,
  );

  // The head script sets the initial theme. Re-apply the stored preference when
  // the OS scheme changes or another tab updates it; reading storage here, not
  // the hydration-time snapshot, avoids a flash of the wrong theme.
  useEffect(() => {
    const query = window.matchMedia(DARK_QUERY);
    const sync = () => applyTheme(readPreference());
    query.addEventListener("change", sync);
    window.addEventListener("storage", sync);
    return () => {
      query.removeEventListener("change", sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  const select = useCallback((next: ThemePreference) => {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // Storage can be blocked; the choice then lasts until the next reload.
    }
    applyTheme(next);
    window.dispatchEvent(new Event(THEME_CHANGE_EVENT));
  }, []);

  if (compact) {
    const current = options.find((option) => option.value === preference)!;
    const next = options[(options.indexOf(current) + 1) % options.length];
    return (
      <button
        type="button"
        aria-label={`${current.label}. Switch to ${next.label.toLowerCase()}`}
        title={`${current.label}. Switch to ${next.label.toLowerCase()}`}
        onClick={() => select(next.value)}
        className={cn(
          "flex h-9 w-full items-center justify-center rounded-lg border bg-background/30 text-muted-foreground transition-colors hover:text-foreground",
          className,
        )}
      >
        <current.icon className="size-4" aria-hidden />
      </button>
    );
  }

  return (
    <div
      role="group"
      aria-label="Colour theme"
      className="flex rounded-lg border bg-background/30 p-0.5"
    >
      {options.map(({ value, label, icon: Icon }) => (
        <button
          key={value}
          type="button"
          aria-label={label}
          aria-pressed={preference === value}
          title={label}
          onClick={() => select(value)}
          className={cn(
            "flex h-8 flex-1 items-center justify-center rounded-md text-muted-foreground transition-colors hover:text-foreground",
            preference === value && "bg-accent text-foreground",
          )}
        >
          <Icon className="size-4" aria-hidden />
        </button>
      ))}
    </div>
  );
}
