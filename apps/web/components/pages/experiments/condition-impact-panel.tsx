import {
  BrainCircuit,
  Cpu,
  DatabaseSearch,
  Layers,
  LayoutGrid,
  type LucideIcon,
} from "lucide-react";
import { GlassPanel, PanelHeader } from "@/components/ui/glass-panel";
import { RatioChip } from "@/components/widgets/delta-chip";
import type {
  ConditionImpact,
  InjectedCondition,
} from "@/lib/experiments-page";
import { formatQuantity } from "@/lib/measurement-format";
import { relativeLengths } from "@/lib/segments";
import { cn } from "@/lib/utils";
import { conditionColor } from "./format";

const ICONS: Record<InjectedCondition, LucideIcon> = {
  cpu: Cpu,
  retrieval: DatabaseSearch,
  inference: BrainCircuit,
  mixed: LayoutGrid,
};

const count = (value: number) => formatQuantity(value, "count");
const ratio = (value: number) => formatQuantity(value, "ratio");

/** Each injected condition against the same experiment's clean replays. */
export function ConditionImpactPanel({
  conditions,
  className,
}: {
  conditions: ConditionImpact[];
  className?: string;
}) {
  const lengths = relativeLengths(conditions.map((c) => c.medianRatio));
  return (
    <GlassPanel
      as="section"
      aria-labelledby="condition-impact"
      className={cn("min-w-0", className)}
    >
      <PanelHeader
        titleId="condition-impact"
        icon={Layers}
        title="Condition impact"
        description="Median across experiments of each condition's replay time over the same experiment's clean replays (same task and plan)."
      />
      <div
        role="region"
        aria-label="Condition impact table"
        tabIndex={0}
        className="relative -mx-1 overflow-x-auto rounded-xl px-1 outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40"
      >
        <table className="w-full min-w-[40rem] text-left text-sm text-ink">
          <thead className="text-[13px] text-ink-muted">
            <tr className="border-b border-line">
              <th scope="col" className="py-2 pr-3 font-medium">
                Condition
              </th>
              <th scope="col" className="py-2 pr-3 font-medium">
                Median vs clean
              </th>
              <th scope="col" className="py-2 pr-3 font-medium">
                Range across experiments
              </th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">
                Replays
              </th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">
                Excess-wait steps
              </th>
              <th scope="col" className="py-2 text-right font-medium">
                Host telemetry
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {conditions.map((condition, index) => {
              const Icon = ICONS[condition.id];
              const length = lengths[index];
              return (
                <tr key={condition.id}>
                  <th scope="row" className="py-2.5 pr-3 font-semibold">
                    <span className="flex items-center gap-2.5">
                      <Icon className="size-4 shrink-0" aria-hidden />
                      {condition.shortLabel}
                    </span>
                  </th>
                  <td className="py-2.5 pr-3">
                    <span className="flex items-center gap-3">
                      <RatioChip ratio={condition.medianRatio} />
                      <span
                        aria-hidden
                        className="h-2 w-24 overflow-hidden rounded-full bg-bar-track"
                      >
                        {length !== null && (
                          <span
                            className="block h-full rounded-full"
                            style={{
                              width: `${length * 100}%`,
                              background: conditionColor(condition.id),
                            }}
                          />
                        )}
                      </span>
                    </span>
                  </td>
                  <td className="py-2.5 pr-3 text-ink-muted tabular-nums">
                    {ratio(condition.minRatio)} – {ratio(condition.maxRatio)}
                    <span className="block text-xs">
                      {count(condition.experiments)} experiments
                    </span>
                  </td>
                  <td className="py-2.5 pr-3 text-right tabular-nums">
                    {count(condition.replays)}
                    {condition.failedReplays > 0 && (
                      <span className="block text-xs text-ink-muted">
                        {count(condition.failedReplays)} failed task
                      </span>
                    )}
                  </td>
                  <td className="py-2.5 pr-3 text-right tabular-nums">
                    {count(condition.excessSteps)}
                    <span className="block text-xs text-ink-muted">
                      of {count(condition.steps)} steps
                    </span>
                  </td>
                  <td className="py-2.5 text-right tabular-nums">
                    {count(condition.telemetryPresent)}
                    <span className="block text-xs text-ink-muted">
                      of {count(condition.steps)} steps sampled
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-ink-muted">
        Ratios, not seconds, because the experiments replay tasks of different
        length. Failed-task replays stay in the medians. Observed under injected
        contention, not a controlled comparison.
      </p>
    </GlassPanel>
  );
}
