import Link from "next/link";
import { ArrowDownToLine, ArrowRight } from "lucide-react";
import { PanelHeading } from "@/components/run-views";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { evidenceUrl, runLabel, runTime } from "@/lib/dashboard-data";
import type { Experiment } from "@/lib/experiment-schema";
import {
  comparability,
  measurementRows,
  POLICY_UNKNOWN_NOTE,
  runMeasurements,
  SINGLE_RUN_LIMITATION,
  type SettingStatus,
} from "@/lib/run-comparison";

// Compares exactly the saved runs named in the URL. A requested UUID that is not loaded is
// reported, never replaced by another run, and nothing is picked on the operator's behalf.

/** The Compare page URL for the given candidate (`run`) and baseline UUIDs. */
export function compareHref({
  candidate,
  baseline,
}: {
  candidate?: string;
  baseline?: string;
}) {
  const query = new URLSearchParams();
  if (candidate) query.set("run", candidate);
  if (baseline) query.set("baseline", baseline);
  const text = query.toString();
  return text ? `/compare?${text}` : "/compare";
}

const statusText: Record<SettingStatus, string> = {
  same: "Same",
  differs: "Differs",
  unknown: "Unknown",
};

type Side = "baseline" | "candidate";
type Selection = { id?: string; run?: Experiment };

function RecordedAt({ run }: { run: Experiment }) {
  const at = new Date(runTime(run));
  return (
    <time dateTime={at.toISOString()} suppressHydrationWarning>
      {at.toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })}
    </time>
  );
}

function RunChoice({
  side,
  selection,
  otherId,
  runs,
}: {
  side: Side;
  selection: Selection;
  otherId?: string;
  runs: Experiment[];
}) {
  const label = side === "baseline" ? "Baseline" : "Candidate";
  const name = label.toLowerCase();
  const hrefFor = (id: string) =>
    compareHref(
      side === "baseline"
        ? { baseline: id, candidate: otherId }
        : { candidate: id, baseline: otherId },
    );
  const choices = runs.filter(
    (run) => run.run_id !== otherId && run.run_id !== selection.id,
  );
  const { run } = selection;
  return (
    <Card
      className="min-w-0 gap-3 p-5"
      role="group"
      aria-label={`${label} run`}
    >
      <h2 className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </h2>
      {run ? (
        <div className="space-y-1">
          <p className="font-medium">{runLabel(run)}</p>
          <p className="break-all font-mono text-xs">{run.run_id}</p>
          <p className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <RecordedAt run={run} />
            <Badge variant="outline">
              {run.persisted ? "Saved run" : "Local artifact"}
            </Badge>
          </p>
        </div>
      ) : selection.id ? (
        <p role="alert" className="break-words text-xs text-amber-200">
          Run {selection.id} could not be loaded. It is not replaced with
          another run; choose a saved run below.
        </p>
      ) : (
        <p className="text-xs text-muted-foreground">No {name} selected.</p>
      )}
      <details className="rounded-lg border">
        <summary className="flex min-h-11 cursor-pointer items-center px-3 text-xs">
          Choose {name}
        </summary>
        {choices.length === 0 ? (
          <p className="border-t p-3 text-xs text-muted-foreground">
            No other loaded runs to choose from.
          </p>
        ) : (
          <ul
            aria-label={`Runs to use as ${name}`}
            className="max-h-64 space-y-1 overflow-y-auto border-t p-2"
          >
            {choices.map((item) => (
              <li key={item.run_id}>
                <Link
                  href={hrefFor(item.run_id)}
                  className="block min-h-11 rounded-lg p-3 hover:bg-accent focus-visible:outline-2 focus-visible:outline-primary"
                >
                  <span className="block text-xs font-medium">
                    {runLabel(item)}
                  </span>
                  <span className="mt-1 block break-all font-mono text-[10px] text-muted-foreground">
                    {item.run_id} · <RecordedAt run={item} />
                    {!item.persisted && " · local artifact"}
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </details>
    </Card>
  );
}

function PairEvidence({
  baseline,
  candidate,
}: {
  baseline: Experiment;
  candidate: Experiment;
}) {
  const result = comparability(baseline, candidate);
  const rows = measurementRows(baseline, candidate, result.comparable);
  const policyUnknown = result.settings.some(
    (row) => row.setting === "Concurrency policy" && row.status === "unknown",
  );
  return (
    <>
      <div
        role="status"
        className="space-y-2 rounded-lg border px-4 py-3 text-xs"
      >
        {result.comparable ? (
          <p>
            <strong>Settings match.</strong> The differences below were observed
            on these two runs only; they do not show that either configuration
            causes them.
          </p>
        ) : (
          <>
            <p>
              <strong>Comparison blocked.</strong> These runs do not share a
              known configuration, so no differences are shown. Each run&apos;s
              own measurements are listed side by side.
            </p>
            <ul className="list-disc space-y-1 pl-5">
              {result.reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
              {result.differences.map((row) => (
                <li key={row.setting}>
                  {row.setting}: {row.baseline} (baseline) vs {row.candidate}{" "}
                  (candidate)
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
      <Card className="gap-0 overflow-hidden">
        <div className="p-5 pb-0">
          <PanelHeading
            title="Recorded settings"
            detail="Every setting must be recorded and equal before runs are compared"
          />
        </div>
        <div
          className="overflow-x-auto"
          role="region"
          aria-label="Recorded settings table"
          tabIndex={0}
        >
          <table className="aw-table">
            <thead>
              <tr>
                <th scope="col">Setting</th>
                <th scope="col">Baseline</th>
                <th scope="col">Candidate</th>
                <th scope="col">Match</th>
              </tr>
            </thead>
            <tbody>
              {result.settings.map((row) => (
                <tr key={row.setting}>
                  <th scope="row" className="font-normal">
                    {row.setting}
                  </th>
                  <td className="break-words">{row.baseline}</td>
                  <td className="break-words">{row.candidate}</td>
                  <td>{statusText[row.status]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {policyUnknown && (
          <p className="p-5 pt-3 text-xs text-muted-foreground">
            {POLICY_UNKNOWN_NOTE}
          </p>
        )}
      </Card>
      <Card className="gap-0 overflow-hidden">
        <div className="p-5 pb-0">
          <PanelHeading
            title={
              result.comparable
                ? "Observed measurements"
                : "Recorded measurements (not compared)"
            }
            detail={
              result.comparable
                ? "Candidate minus baseline on these two runs"
                : "Shown per run; no difference is calculated across configurations"
            }
          />
        </div>
        <div
          className="overflow-x-auto"
          role="region"
          aria-label="Measurements table"
          tabIndex={0}
        >
          <table className="aw-table">
            <thead>
              <tr>
                <th scope="col">Measurement</th>
                <th scope="col">Baseline</th>
                <th scope="col">Candidate</th>
                {result.comparable && (
                  <>
                    <th scope="col">Observed difference</th>
                    <th scope="col">Relative change</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.measurement}>
                  <th scope="row" className="font-normal">
                    {row.measurement}
                  </th>
                  <td className="font-mono">{row.baseline}</td>
                  <td className="font-mono">{row.candidate}</td>
                  {result.comparable && (
                    <>
                      <td className="font-mono">
                        {row.difference?.seconds ?? "—"}
                      </td>
                      <td className="font-mono">
                        {row.difference?.relative ?? "—"}
                      </td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="p-5 pt-3 text-xs text-muted-foreground">
          {SINGLE_RUN_LIMITATION}
        </p>
      </Card>
      <div className="grid gap-4 lg:grid-cols-2">
        {(
          [
            ["Baseline", baseline],
            ["Candidate", candidate],
          ] as const
        ).map(([label, run]) => {
          const { limitations } = runMeasurements(run);
          return (
            <Card key={label} className="min-w-0 gap-2 p-5 text-xs">
              <h2 className="text-sm font-semibold">{label} limitations</h2>
              {limitations.length === 0 ? (
                <p className="text-muted-foreground">None recorded.</p>
              ) : (
                <ul className="list-disc space-y-1 pl-5">
                  {limitations.map((text, index) => (
                    <li key={index} className="break-words">
                      {text}
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          );
        })}
      </div>
      <div className="flex flex-wrap gap-2">
        <Button asChild variant="outline" className="min-h-11">
          <a href={evidenceUrl(baseline)}>
            <ArrowDownToLine className="size-3.5" /> Export baseline evidence
          </a>
        </Button>
        <Button asChild variant="outline" className="min-h-11">
          <a href={evidenceUrl(candidate)}>
            <ArrowDownToLine className="size-3.5" /> Export candidate evidence
          </a>
        </Button>
        <Button asChild variant="outline" className="min-h-11">
          <Link href={`/traces?run=${candidate.run_id}`}>
            Inspect candidate trace <ArrowRight className="size-3.5" />
          </Link>
        </Button>
      </div>
    </>
  );
}

/**
 * Baseline and candidate chosen by UUID (`?baseline=` and `?run=`). `loaded` is every run the
 * page holds, including requested runs outside the current time range; `runs` are the choices.
 */
export function ComparePage({
  runs,
  loaded,
  candidateId,
  baselineId,
}: {
  runs: Experiment[];
  loaded: Experiment[];
  candidateId?: string;
  baselineId?: string;
}) {
  const find = (id?: string) =>
    id ? loaded.find((run) => run.run_id === id) : undefined;
  const baseline = { id: baselineId, run: find(baselineId) };
  const candidate = { id: candidateId, run: find(candidateId) };
  return (
    <div className="space-y-5">
      <div className="grid gap-4 lg:grid-cols-2">
        <RunChoice
          side="baseline"
          selection={baseline}
          otherId={candidateId}
          runs={runs}
        />
        <RunChoice
          side="candidate"
          selection={candidate}
          otherId={baselineId}
          runs={runs}
        />
      </div>
      {baseline.run && candidate.run ? (
        <PairEvidence baseline={baseline.run} candidate={candidate.run} />
      ) : (
        <p className="text-xs text-muted-foreground">
          Choose a baseline and a candidate run to see their recorded settings
          and measurements.
        </p>
      )}
    </div>
  );
}
