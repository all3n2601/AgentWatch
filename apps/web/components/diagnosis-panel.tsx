import { Card } from "@/components/ui/card";
import { PanelHeading } from "@/components/run-views";
import { duration, percent } from "@/lib/dashboard-data";
import { displayName, LABEL_RESOURCES } from "@/lib/dataset-view";
import type { DiagnosisStep } from "@/lib/unified-runs";

const resourceColor = (resource: string) =>
  LABEL_RESOURCES.find((entry) => entry.id === resource)?.color ??
  "var(--series-slate)";

const BASIS: Record<DiagnosisStep["basis"], string> = {
  attributed: "Attributed by the rule recorded with this run",
  ran_on: "Resource the step ran on; no waiting attribution is recorded",
  none: "No resource recorded",
};

/**
 * The slowest-waiting steps first, with the resource involved and exactly what that claim
 * rests on. It never shows a confidence figure: none is calibrated for this data yet.
 */
export function DiagnosisPanel({
  steps,
  scope,
}: {
  steps: DiagnosisStep[];
  scope: "live" | "saved";
}) {
  const waited = steps.filter((step) => step.queueSeconds !== null);
  const totalWaiting = waited.reduce(
    (sum, s) => sum + (s.queueSeconds ?? 0),
    0,
  );
  const totalTime = waited.reduce(
    (sum, s) => sum + (s.durationSeconds ?? 0),
    0,
  );
  const top = steps.slice(0, 8);
  const longestWait = Math.max(...top.map((step) => step.queueSeconds ?? 0), 0);
  return (
    <div className="space-y-4">
      <Card className="grid grid-cols-2 gap-px overflow-hidden bg-border p-0 sm:grid-cols-4">
        {[
          [
            "Steps",
            String(steps.length),
            `${waited.length} with measured queue time`,
          ],
          ["Time waiting", duration(totalWaiting), "Sum of step queue delays"],
          [
            "Waiting share",
            totalTime > 0 ? percent(totalWaiting / totalTime) : "—",
            "Queue time over step time",
          ],
          [
            "Unmeasured steps",
            String(steps.length - waited.length),
            "Never started, or no ticks recorded",
          ],
        ].map(([label, value, detail]) => (
          <div key={label} className="bg-card p-4">
            <span className="text-xs text-muted-foreground">{label}</span>
            <div className="metric-number mt-2 text-2xl font-semibold leading-none">
              {value}
            </div>
            <p className="mt-2 text-[11px] text-muted-foreground">{detail}</p>
          </div>
        ))}
      </Card>
      <Card className="gap-0 p-5">
        <PanelHeading
          title="Steps that waited longest"
          detail="Ranked by queue delay. Queue delay is measured; the resource column says what it rests on."
        />
        <ul className="divide-y">
          {top.map((step) => (
            <li
              key={step.stepId}
              className="grid grid-cols-1 gap-2 py-3 text-xs sm:grid-cols-[minmax(0,1.2fr)_minmax(0,1.6fr)_minmax(0,1.4fr)] sm:items-center sm:gap-4"
            >
              <div className="min-w-0">
                <div className="truncate font-medium">{step.stepId}</div>
                <div className="text-[10px] text-muted-foreground">
                  {displayName(step.type)} · {step.state}
                </div>
              </div>
              <div>
                <div
                  role="img"
                  aria-label={`Queue ${step.queueSeconds === null ? "not measured" : duration(step.queueSeconds)}, execution ${step.executionSeconds === null ? "not measured" : duration(step.executionSeconds)}`}
                  className="flex h-2 gap-0.5 overflow-hidden rounded-full bg-muted"
                >
                  {step.queueSeconds !== null && (
                    <span
                      className="min-w-px rounded-full"
                      style={{
                        width: `${longestWait === 0 ? 0 : (step.queueSeconds / longestWait) * 100}%`,
                        background: "var(--series-amber)",
                      }}
                    />
                  )}
                </div>
                <div className="mt-1 font-mono text-[11px] text-muted-foreground">
                  {step.queueSeconds === null
                    ? "queue not measured"
                    : `${duration(step.queueSeconds)} queued`}
                  {step.waitingShare !== null &&
                    ` · ${percent(step.waitingShare)} of the step`}
                </div>
              </div>
              <div className="min-w-0">
                {step.resource ? (
                  <span className="flex items-center gap-2">
                    <i
                      className="size-2 shrink-0 rounded-sm"
                      style={{ background: resourceColor(step.resource) }}
                      aria-hidden
                    />
                    <span className="font-medium">
                      {displayName(step.resource)}
                    </span>
                  </span>
                ) : (
                  <span className="text-muted-foreground">
                    No resource named
                  </span>
                )}
                <div className="text-[10px] leading-4 text-muted-foreground">
                  {BASIS[step.basis]}
                </div>
              </div>
            </li>
          ))}
        </ul>
      </Card>
      <p className="text-xs leading-5 text-muted-foreground">
        {scope === "live"
          ? "Live runs record which resource each step ran on and its timings. No per-step attribution of waiting is stored for live runs, so none is shown."
          : "Attributions on saved runs come from the rule recorded with the run. This dashboard does not report how accurate that rule is."}
      </p>
    </div>
  );
}
