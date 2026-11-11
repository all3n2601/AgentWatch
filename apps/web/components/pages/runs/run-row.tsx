import * as React from "react";
import { Database, FileArchive, Radio } from "lucide-react";
import { Chip } from "@/components/ui/chip";
import {
  StatusBadge,
  TONE_ICONS,
  TONE_TEXT,
} from "@/components/ui/status-badge";
import { Measurement } from "@/components/widgets/measurement";
import { displayName } from "@/lib/dataset-view";
import { MISSING_VALUE } from "@/lib/measurement-format";
import {
  STATE_PRESENTATION,
  TELEMETRY_PRESENTATION,
  type RunsPageRow,
} from "@/lib/runs-page";
import type { Tone } from "@/lib/tone";
import { cn } from "@/lib/utils";

const TONE_ACCENT: Record<Tone, string> = {
  success: "border-l-success-solid",
  warning: "border-l-warning-solid",
  caution: "border-l-caution-solid",
  danger: "border-l-danger-solid",
  info: "border-l-info-solid",
};

/** A reading the live run list does not report; a dash, never a zero. */
export function NotInList({ what }: { what: string }) {
  return (
    <span className="text-ink-subtle">
      <span aria-hidden>{MISSING_VALUE}</span>
      <span className="sr-only">
        {what} is not reported in the live run list
      </span>
    </span>
  );
}

/** Wall-clock time in the viewer's zone; the server renders its own zone first. */
export function RecordedTime({ at }: { at: string }) {
  return (
    <time dateTime={at} suppressHydrationWarning className="tabular-nums">
      {new Date(at).toLocaleString("en-US", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })}
    </time>
  );
}

export function SourceChip({ row }: { row: RunsPageRow }) {
  if (row.source === "live")
    return (
      <Chip size="sm" icon={Radio} className="py-0.5">
        Live
      </Chip>
    );
  return row.evidence?.persisted ? (
    <Chip size="sm" icon={Database} className="py-0.5">
      Saved
    </Chip>
  ) : (
    <Chip size="sm" icon={FileArchive} className="py-0.5">
      Local file
    </Chip>
  );
}

/*
 * On one-screen (`fit`) desktops a list narrower than the table layout packs
 * each row's readings as inline "label value" pairs instead of a stacked
 * grid, so the in-panel list shows several runs at once. Phones keep the grid.
 */
const FIT_COMPACT_ROW = "fit:@max-[44rem]:gap-y-1 fit:@max-[44rem]:py-1.5";
const FIT_COMPACT_FIELDS =
  "fit:@max-[44rem]:flex fit:@max-[44rem]:flex-wrap fit:@max-[44rem]:items-center fit:@max-[44rem]:gap-x-4 fit:@max-[44rem]:gap-y-1";
const FIT_COMPACT_TITLE =
  "fit:@max-[44rem]:flex fit:@max-[44rem]:items-center fit:@max-[44rem]:gap-x-3";
const FIT_COMPACT_FIELD =
  "fit:@max-[44rem]:flex fit:@max-[44rem]:items-center fit:@max-[44rem]:gap-1.5";

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className={cn("min-w-0", FIT_COMPACT_FIELD)}>
      {/* Wide lists show one header row instead; screen readers still hear each label. */}
      <dt className="text-xs text-ink-muted @min-[44rem]:sr-only">{label}</dt>
      <dd className="mt-0.5 min-w-0 truncate text-sm font-medium text-ink @min-[44rem]:mt-0 @min-[44rem]:text-xs fit:@max-[44rem]:mt-0 fit:@max-[44rem]:text-xs">
        {children}
      </dd>
    </div>
  );
}

/**
 * Column layout shared by the rows and the header: compare box, run, status,
 * recorded time, step time, queue share, top resource, telemetry.
 */
export const RUN_ROW_COLUMNS =
  "@min-[44rem]:grid-cols-[2.75rem_minmax(7.5rem,1.4fr)_7.25rem_8rem_4rem_3.5rem_minmax(4.5rem,1fr)_4.5rem]";

/** Column names for the wide list; hidden from assistive tech, which reads each cell's own label. */
export function RunRowHeader() {
  return (
    <div
      aria-hidden
      className={cn(
        "hidden gap-x-2 px-2 pb-1 text-xs font-medium text-ink-muted @min-[44rem]:grid",
        RUN_ROW_COLUMNS,
      )}
    >
      <span />
      <span>Run</span>
      <span>Status</span>
      <span>Recorded</span>
      <span>Step time</span>
      <span>Queue</span>
      <span>Top resource</span>
      <span>Telemetry</span>
    </div>
  );
}

/**
 * One run in the list. The title selects the run for the preview panel; the
 * checkbox (saved runs only) adds it to the comparison.
 */
export function RunRow({
  row,
  previewed,
  previewId,
  onPreview,
  compare,
}: {
  row: RunsPageRow;
  previewed: boolean;
  previewId: string;
  onPreview: (key: string) => void;
  compare: {
    checked: boolean;
    disabled: boolean;
    onToggle: (key: string) => void;
  };
}) {
  const state = STATE_PRESENTATION[row.state];
  const Icon = TONE_ICONS[state.tone];
  const evidence = row.evidence;
  const shortId = row.id.slice(0, 8);

  return (
    <li
      className={cn(
        "glass-inset grid grid-cols-[2.75rem_minmax(0,1fr)] items-center gap-x-1 gap-y-2 border-l-[3px] p-2 @min-[44rem]:gap-x-2 @min-[44rem]:py-1",
        RUN_ROW_COLUMNS,
        FIT_COMPACT_ROW,
        TONE_ACCENT[state.tone],
        previewed && "border-success-line bg-glass-strong shadow-glow",
      )}
    >
      {evidence ? (
        <label
          className="flex size-11 shrink-0 cursor-pointer items-center justify-center rounded-xl has-[:focus-visible]:ring-[3px] has-[:focus-visible]:ring-brand/40 has-[:disabled]:cursor-not-allowed"
          title={compare.disabled ? "Two runs are already selected" : undefined}
        >
          <input
            type="checkbox"
            className="size-5 accent-brand-strong outline-none disabled:opacity-40"
            checked={compare.checked}
            disabled={compare.disabled}
            onChange={() => compare.onToggle(row.key)}
            aria-label={`Select ${row.title} ${shortId} for comparison`}
          />
        </label>
      ) : (
        // Live runs have no saved timings to compare; keep the column aligned.
        <span className="size-11 shrink-0" aria-hidden />
      )}
      <button
        type="button"
        onClick={() => onPreview(row.key)}
        aria-pressed={previewed}
        aria-controls={previewId}
        className={cn(
          "min-h-11 min-w-0 rounded-xl px-2 py-1 text-left outline-none hover:bg-glass focus-visible:ring-[3px] focus-visible:ring-brand/40",
          FIT_COMPACT_TITLE,
        )}
      >
        <span className="flex min-w-0 items-center gap-2">
          <Icon
            className={cn("size-4 shrink-0", TONE_TEXT[state.tone])}
            aria-hidden
          />
          <span className="truncate text-sm font-semibold text-ink">
            {row.title}
          </span>
        </span>
        <span className="mt-0.5 flex min-w-0 items-center gap-2 text-xs text-ink-muted fit:@max-[44rem]:mt-0">
          <span className="font-mono" title={row.id}>
            {shortId}
          </span>
          <span className="sr-only">(run {row.id})</span>
          <SourceChip row={row} />
          {evidence?.model && evidence.model !== row.title && (
            <span className="truncate">{evidence.model}</span>
          )}
        </span>
        <span className="sr-only">Show in run preview</span>
      </button>
      <dl
        className={cn(
          "col-span-2 grid min-w-0 grid-cols-2 gap-x-4 gap-y-2 pl-1 @lg:grid-cols-3 @min-[44rem]:contents",
          FIT_COMPACT_FIELDS,
        )}
      >
        <Field label="Status">
          <StatusBadge
            label={state.label}
            tone={state.tone}
            className="py-0 text-xs"
          />
        </Field>
        <Field label={row.source === "live" ? "Registered" : "Finished"}>
          <RecordedTime at={row.at} />
        </Field>
        <Field label="Step time">
          {evidence ? (
            <Measurement
              value={evidence.queueSeconds + evidence.executionSeconds}
              unit="s"
            />
          ) : (
            <NotInList what="Step time" />
          )}
        </Field>
        <Field label="Queue share">
          {row.waitingShare === null ? (
            <NotInList what="Queue share" />
          ) : (
            <Measurement value={row.waitingShare * 100} unit="percent" />
          )}
        </Field>
        <Field label="Top resource">
          {!evidence ? (
            <NotInList what="Resource attribution" />
          ) : row.topResource ? (
            displayName(row.topResource)
          ) : (
            <span className="text-ink-muted">None attributed</span>
          )}
        </Field>
        <Field label="Telemetry">
          {evidence ? (
            <StatusBadge
              appearance="plain"
              label={TELEMETRY_PRESENTATION[evidence.telemetry].label}
              tone={TELEMETRY_PRESENTATION[evidence.telemetry].tone}
              className="text-sm [&_svg]:size-4"
            />
          ) : (
            <NotInList what="Telemetry" />
          )}
        </Field>
      </dl>
    </li>
  );
}
