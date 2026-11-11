import { ChartColumn, Info } from "lucide-react";
import {
  GlassPanel,
  InsetCard,
  PanelHeader,
} from "@/components/ui/glass-panel";
import { formatQuantity } from "@/lib/measurement-format";
import { cn } from "@/lib/utils";

/** The cautious one-sentence reading of the condition impact, with its limits. */
export function BusinessReadingPanel({
  reading,
  models,
  tasks,
  className,
}: {
  reading: string;
  models: string[];
  tasks: number;
  className?: string;
}) {
  return (
    <GlassPanel
      as="section"
      aria-labelledby="experiment-reading"
      className={cn("min-w-0", className)}
    >
      <PanelHeader
        titleId="experiment-reading"
        icon={ChartColumn}
        title="Business reading"
      />
      <InsetCard className="border-success-line bg-brand-soft">
        <p className="text-[15px] font-semibold text-ink">{reading}</p>
      </InsetCard>
      <p className="mt-3 flex gap-2 text-sm text-ink-muted">
        <Info className="mt-0.5 size-4 shrink-0" aria-hidden />
        <span>
          Observed, not causal. This shows relative performance of{" "}
          {models.join(", ") || "the replayed agent"} on{" "}
          {formatQuantity(tasks, "count")} tasks under injected contention, not
          a capacity recommendation.
        </span>
      </p>
    </GlassPanel>
  );
}
