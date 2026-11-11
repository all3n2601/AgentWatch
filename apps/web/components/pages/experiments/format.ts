import { RESOURCE_CLASSES } from "@/lib/tone";

/** Display names for the replay protocol's step roles. */
const ROLE_NAMES: Record<string, string> = {
  baseline: "Baseline",
  clean_control: "Clean control",
  contended: "Injected",
  clean: "Clean",
};
export const roleName = (role: string) =>
  ROLE_NAMES[role] ?? role.replaceAll("_", " ");

/** Series colour per replay condition; clean replays use the neutral series. */
const CONDITION_COLORS: Record<string, string> = Object.fromEntries(
  Object.entries(RESOURCE_CLASSES).map(([id, entry]) => [id, entry.color]),
);
export const conditionColor = (condition: string): string =>
  CONDITION_COLORS[condition] ?? RESOURCE_CLASSES.none.color;

const dateTime = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
  timeZone: "UTC",
});
const dateOnly = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
  timeZone: "UTC",
});

/** Recorded times are shown in UTC so every viewer reads the same clock. */
export const formatDateTime = (iso: string) =>
  `${dateTime.format(new Date(iso))} UTC`;
export const formatDate = (iso: string) => dateOnly.format(new Date(iso));

/** Coefficient of variation as "±6.3%". */
export const formatCv = (cv: number) => `±${(cv * 100).toFixed(1)}%`;
