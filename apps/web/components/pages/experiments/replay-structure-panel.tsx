import { Repeat } from "lucide-react";
import {
  GlassPanel,
  InsetCard,
  PanelHeader,
} from "@/components/ui/glass-panel";
import {
  KeyValueList,
  type KeyValueItem,
} from "@/components/widgets/evidence-lists";
import { displayName } from "@/lib/dataset-view";
import type { ExperimentsPage } from "@/lib/experiments-page";
import { formatQuantity } from "@/lib/measurement-format";
import { cn } from "@/lib/utils";
import { formatCv, formatDate, roleName } from "./format";

const count = (value: number) => formatQuantity(value, "count");

function structureItems(page: ExperimentsPage): KeyValueItem[] {
  const perExperiment = page.roleCounts.reduce((sum, r) => sum + r.replays, 0);
  return [
    {
      id: "tasks",
      label: "Tasks",
      value: page.groups
        .map((g) => `${g.taskGroup} (${count(g.experiments.length)})`)
        .join(", "),
    },
    ...(perExperiment > 0
      ? [
          {
            id: "replays",
            label: "Replays per experiment",
            value: `${count(perExperiment)}: ${page.roleCounts
              .map(
                (r) => `${count(r.replays)} ${roleName(r.role).toLowerCase()}`,
              )
              .join(", ")}`,
          },
        ]
      : []),
    {
      id: "clean",
      label: "Clean replays",
      value: `${count(page.clean.replays)} runs · ${count(page.clean.steps)} steps`,
    },
    {
      id: "cv",
      label: "Clean variability (median CV)",
      value: page.clean.medianCv === null ? "—" : formatCv(page.clean.medianCv),
    },
    {
      id: "model",
      label: "Agent model",
      value: page.models.length ? page.models.join(", ") : "Not recorded",
    },
    {
      id: "recorded",
      label: "Recorded (UTC)",
      value: page.recordedRange
        ? `${formatDate(page.recordedRange.first)} – ${formatDate(page.recordedRange.last)}`
        : "Not recorded",
    },
  ];
}

/** How the release's replays are organised: tasks, protocol, clean spread. */
export function ReplayStructurePanel({
  page,
  className,
}: {
  page: ExperimentsPage;
  className?: string;
}) {
  return (
    <GlassPanel
      as="section"
      aria-labelledby="replay-structure"
      className={cn("min-w-0", className)}
    >
      <PanelHeader
        titleId="replay-structure"
        icon={Repeat}
        title="Replay structure"
      />
      <div className="grid gap-3 md:grid-cols-2 md:items-start">
        <KeyValueList items={structureItems(page)} dense />
        {page.labelOutcomes && (
          <InsetCard>
            <h3 className="mb-2 text-sm font-semibold text-ink">
              Step label outcomes{" "}
              <span className="font-normal text-ink-muted">replay steps</span>
            </h3>
            <ul className="space-y-1 text-sm text-ink">
              {page.labelOutcomes.map((outcome) => (
                <li key={outcome.basis} className="flex justify-between gap-3">
                  <span>{displayName(outcome.basis)}</span>
                  <span className="tabular-nums">{count(outcome.steps)}</span>
                </li>
              ))}
            </ul>
          </InsetCard>
        )}
      </div>
    </GlassPanel>
  );
}
