import { Card } from "@/components/ui/card";
import { PanelHeading } from "@/components/run-views";
import type { DatasetSnapshot } from "@/lib/dataset-schema";
import { conditionRatios, summarizeExperiments } from "@/lib/dataset-runs";
import { duration } from "@/lib/dashboard-data";
import { replayConditionName } from "@/lib/dataset-view";

const CONDITIONS = ["clean", "cpu", "retrieval", "inference", "mixed"] as const;
const COLORS: Record<string, string> = {
  clean: "var(--series-slate)",
  cpu: "var(--series-amber)",
  retrieval: "var(--series-sky)",
  inference: "var(--series-violet)",
  mixed: "var(--series-rose)",
};

/**
 * Replay experiments: each condition's median step time against the same experiment's clean
 * replays. Observed under injected contention, not a controlled comparison.
 */
export function ExperimentsView({ snapshot }: { snapshot: DatasetSnapshot }) {
  if (snapshot.status === "unavailable")
    return (
      <Card className="items-center gap-3 border-dashed p-12 text-center">
        <h2 className="text-base font-semibold">Experiments unavailable</h2>
        <p className="max-w-lg text-sm leading-6 text-muted-foreground">
          {snapshot.reason}
        </p>
      </Card>
    );
  const experiments = summarizeExperiments(snapshot.runs);
  const ratios = conditionRatios(experiments);
  return (
    <div className="space-y-4">
      <Card className="gap-0 p-5">
        <PanelHeading
          title="Slowdown by injected condition"
          detail={`Median across ${experiments.length} experiments of each condition's step time over that experiment's clean replays`}
        />
        <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {ratios.map((entry) => (
            <li key={entry.condition} className="flex items-center gap-3">
              <i
                className="h-10 w-1 shrink-0 rounded-full"
                style={{ background: COLORS[entry.condition] }}
                aria-hidden
              />
              <div>
                <div className="text-xs text-muted-foreground">
                  {replayConditionName(entry.condition)}
                </div>
                <div className="metric-number text-2xl font-semibold leading-none">
                  ×{entry.ratio.toFixed(1)}
                </div>
                <div className="mt-1 text-[11px] text-muted-foreground">
                  {entry.experiments} experiments
                </div>
              </div>
            </li>
          ))}
        </ul>
        <p className="mt-4 text-[11px] leading-5 text-muted-foreground">
          Ratios, not seconds, because the experiments replay tasks of different
          length. These are observed under injected contention and are not a
          controlled comparison.
        </p>
      </Card>
      <Card className="gap-0 overflow-hidden py-0">
        <div className="p-5 pb-0">
          <PanelHeading
            title="Experiments"
            detail="Median total step time per replay, by condition. A dash means that condition was not replayed."
          />
        </div>
        <div className="overflow-x-auto">
          <table className="aw-table">
            <thead>
              <tr>
                <th>Experiment</th>
                <th>Replays</th>
                <th>Recorded</th>
                {CONDITIONS.map((condition) => (
                  <th key={condition}>{replayConditionName(condition)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {experiments.map((experiment) => (
                <tr key={experiment.experiment}>
                  <td>
                    <div className="font-medium text-foreground">
                      {experiment.taskGroup}
                    </div>
                    <div className="font-mono text-[10px] text-muted-foreground">
                      {experiment.experiment.slice(-12)}
                    </div>
                  </td>
                  <td>{experiment.replays}</td>
                  <td>
                    {experiment.recordedAt === null
                      ? "—"
                      : new Date(experiment.recordedAt).toLocaleDateString(
                          "en-US",
                          { month: "short", day: "numeric" },
                        )}
                  </td>
                  {CONDITIONS.map((condition) => {
                    const value = experiment.medians[condition];
                    const clean = experiment.medians.clean;
                    return (
                      <td key={condition} className="font-mono">
                        {value === undefined ? (
                          <span className="text-muted-foreground">—</span>
                        ) : (
                          <>
                            {duration(value)}
                            {condition !== "clean" && clean ? (
                              <span className="ml-1.5 text-[10px] text-muted-foreground">
                                ×{(value / clean).toFixed(1)}
                              </span>
                            ) : null}
                          </>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
