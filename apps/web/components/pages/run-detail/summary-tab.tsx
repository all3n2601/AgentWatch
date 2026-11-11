import { Database, Layers, ListOrdered, ScrollText, Timer } from "lucide-react";
import { InsightPanel } from "@/components/insight-panel";
import { GlassPanel, PanelHeader } from "@/components/ui/glass-panel";
import { StatStrip } from "@/components/widgets/stat";
import type { RunDetailModel } from "@/lib/run-detail";
import { StepTiming } from "./step-timing";

function RunSummaryStrip({ model }: { model: RunDetailModel }) {
  const { totals } = model;
  return (
    <StatStrip
      label="Run timing summary"
      items={[
        {
          id: "total",
          icon: Timer,
          value: <StepTiming seconds={totals.stepSeconds} />,
          label: "profiled step time",
        },
        {
          id: "queue",
          icon: Layers,
          value: <StepTiming seconds={totals.queueSeconds} />,
          label: `queue over ${totals.measuredQueueSteps} measured steps`,
        },
        totals.attributedSteps === null
          ? {
              id: "attribution",
              icon: ListOrdered,
              value: `${totals.measuredQueueSteps}/${totals.totalSteps}`,
              label: "steps with a queue reading",
            }
          : {
              id: "attribution",
              icon: ListOrdered,
              value: `${totals.attributedSteps}/${totals.totalSteps}`,
              label: "excess-wait steps",
            },
        {
          id: "telemetry",
          icon: Database,
          tone: model.telemetry.tone,
          value: "",
          label: (
            <span title={model.telemetry.detail}>
              {model.telemetry.label}
              <span className="block text-xs text-ink-muted">
                {model.telemetry.detail}
              </span>
            </span>
          ),
        },
      ]}
    />
  );
}

/**
 * The run's totals, the optional AI summary and the scope notes that qualify every figure on
 * the page. Kept off the Diagnosis tab so the ranked steps and their evidence fit one screen.
 */
export function SummaryTab({
  model,
  insightsEnabled,
}: {
  model: RunDetailModel;
  /** Whether the server has an AI endpoint configured. */
  insightsEnabled: boolean;
}) {
  const scopeNotes = [...model.notes, ...model.limitations];
  return (
    <div className="space-y-4">
      <RunSummaryStrip model={model} />
      {insightsEnabled && (
        <InsightPanel request={{ kind: "run", runId: model.runId }} />
      )}
      <GlassPanel as="section" aria-labelledby="run-limitations">
        <PanelHeader
          titleId="run-limitations"
          icon={ScrollText}
          title="Scope and limitations"
          description="What the figures on this page cover and how they were measured."
        />
        {scopeNotes.length > 0 ? (
          <ul className="list-disc space-y-1 pl-5 text-sm text-ink-muted">
            {/* Notes and limitations may repeat text, so the position is part of the key. */}
            {scopeNotes.map((note, index) => (
              <li key={`${index}:${note}`}>{note}</li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-ink-muted">
            No scope notes or limitations were recorded for this run.
          </p>
        )}
      </GlassPanel>
    </div>
  );
}
