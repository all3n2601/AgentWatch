/**
 * The settings tabs, in order. `id` is the `?view=` value; `headingId` labels
 * the section inside the tab's panel.
 */
export const SETTINGS_TABS = [
  { id: "workspace", label: "Workspace", headingId: "settings-workspace" },
  { id: "display", label: "Display", headingId: "settings-display" },
  { id: "data", label: "Data connection", headingId: "settings-data" },
  { id: "insights", label: "Insights", headingId: "settings-insights" },
] as const;

export type SettingsTabId = (typeof SETTINGS_TABS)[number]["id"];
