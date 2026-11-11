import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import {
  parseThemePreference,
  resolveTheme,
  THEME_INIT_SCRIPT,
  THEME_STORAGE_KEY,
} from "../lib/theme";

test("parseThemePreference accepts known values and falls back to system", () => {
  assert.equal(parseThemePreference("light"), "light");
  assert.equal(parseThemePreference("dark"), "dark");
  assert.equal(parseThemePreference("system"), "system");
  assert.equal(parseThemePreference(null), "system");
  assert.equal(parseThemePreference("solarized"), "system");
  assert.equal(parseThemePreference(""), "system");
});

test("resolveTheme follows the system only for the system preference", () => {
  assert.equal(resolveTheme("system", true), "dark");
  assert.equal(resolveTheme("system", false), "light");
  assert.equal(resolveTheme("light", true), "light");
  assert.equal(resolveTheme("dark", false), "dark");
});

function runInitScript(stored: string | null, systemPrefersDark: boolean) {
  const classes = new Set<string>();
  const root = {
    classList: {
      toggle: (name: string, force: boolean) =>
        force ? classes.add(name) : classes.delete(name),
    },
    style: { colorScheme: "" },
  };
  vm.runInNewContext(THEME_INIT_SCRIPT, {
    localStorage: {
      getItem: (key: string) => (key === THEME_STORAGE_KEY ? stored : null),
    },
    window: { matchMedia: () => ({ matches: systemPrefersDark }) },
    document: { documentElement: root },
  });
  return { classes: [...classes].sort(), colorScheme: root.style.colorScheme };
}

test("the head script agrees with resolveTheme for every stored value", () => {
  for (const stored of ["system", "light", "dark", null, "junk"] as const) {
    for (const systemPrefersDark of [true, false]) {
      const expected = resolveTheme(
        parseThemePreference(stored),
        systemPrefersDark,
      );
      const result = runInitScript(stored, systemPrefersDark);
      assert.deepEqual(result.classes, [expected]);
      assert.equal(result.colorScheme, expected);
    }
  }
});

test("the head script leaves the document alone when storage throws", () => {
  const classes: string[] = [];
  vm.runInNewContext(THEME_INIT_SCRIPT, {
    localStorage: {
      getItem: () => {
        throw new Error("blocked");
      },
    },
    window: { matchMedia: () => ({ matches: false }) },
    document: {
      documentElement: {
        classList: { toggle: (name: string) => classes.push(name) },
        style: {},
      },
    },
  });
  assert.deepEqual(classes, []);
});
