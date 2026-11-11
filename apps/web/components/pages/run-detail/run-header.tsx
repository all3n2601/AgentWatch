import Link from "next/link";
import {
  ArrowDownToLine,
  Clock3,
  Database,
  GitCompareArrows,
} from "lucide-react";
import { PageHeader } from "@/components/shell/page-header";
import { Button } from "@/components/ui/button";
import { CopyChip } from "@/components/ui/copy-chip";
import { StatusBadge } from "@/components/ui/status-badge";
import { evidenceUrl } from "@/lib/dashboard-data";
import type { RunDetailModel } from "@/lib/run-detail";
import { routes } from "@/lib/routes";
import type { RunView } from "@/lib/unified-runs";

const headerTime = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hourCycle: "h23",
  timeZone: "UTC",
});

/**
 * Title, identity and provenance of the one run on the page. The source chip says which store
 * answered, and labels the local-file fallback explicitly so it is never read as history.
 */
export function RunHeader({
  model,
  view,
  sourceLabel,
}: {
  model: RunDetailModel;
  view: RunView;
  sourceLabel: string;
}) {
  const stateLabel = model.state.charAt(0).toUpperCase() + model.state.slice(1);
  return (
    <PageHeader
      title={
        <span className="break-words [overflow-wrap:anywhere]">
          {model.title}
        </span>
      }
      // Identity and state sit outside the h1 so the heading is just the run's name.
      description={
        <span className="mt-2 flex flex-wrap items-center gap-3">
          <CopyChip value={model.runId} label="run ID" />
          <StatusBadge
            label={stateLabel}
            className="px-3 py-1 text-[15px] font-medium"
          />
        </span>
      }
      scope={
        <span className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm sm:text-[15px]">
          <span className="inline-flex items-center gap-1.5">
            <Database className="size-4" aria-hidden />
            {sourceLabel}
          </span>
          <span aria-hidden>·</span>
          <span className="inline-flex items-center gap-1.5">
            <Clock3 className="size-4" aria-hidden />
            {model.time.label}{" "}
            <time dateTime={model.time.iso}>
              {headerTime.format(new Date(model.time.iso))} UTC
            </time>
          </span>
          <span aria-hidden>·</span>
          <span>{model.totals.totalSteps} steps</span>
        </span>
      }
      actions={
        <>
          {view.source === "saved" && (
            <Button asChild variant="glass" size="pill">
              <Link href={routes.compare(model.runId)}>
                <GitCompareArrows aria-hidden />
                Compare
              </Link>
            </Button>
          )}
          <Button asChild variant="glass" size="pill">
            <a
              href={
                view.source === "live"
                  ? routes.liveExport(model.runId)
                  : evidenceUrl(view.run)
              }
            >
              <ArrowDownToLine aria-hidden />
              {view.source === "live" ? "Export JSON" : "Export results.json"}
            </a>
          </Button>
        </>
      }
    />
  );
}
