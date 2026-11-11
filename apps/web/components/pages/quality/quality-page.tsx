import {
  Activity,
  CircleCheck,
  CircleX,
  Info,
  Network,
  Tag,
  TriangleAlert,
  type LucideIcon,
} from "lucide-react";
import { WorkspacePage } from "@/components/shell/app-shell";
import { PageHeader } from "@/components/shell/page-header";
import { TabbedView, type TabbedViewTab } from "@/components/shell/tabbed-view";
import type { CheckState } from "@/components/widgets/evidence-lists";
import { StatStrip, type StatStripItem } from "@/components/widgets/stat";
import { UnavailableState } from "@/components/widgets/state-views";
import { formatQuantity } from "@/lib/measurement-format";
import {
  checkTally,
  statusText,
  type QualityPageData,
} from "@/lib/quality-page";
import type { Tone } from "@/lib/tone";
import { shareText } from "./count-shares";
import {
  CompositionPanel,
  LabelsPanel,
  SplitPanel,
  TelemetryPanel,
  ValidationPanel,
  WorkloadPanel,
} from "./quality-panels";
import { FeatureSignalPanel, FieldHealthPanel } from "./quality-tables";

export type QualityPageState =
  | { status: "ready"; data: QualityPageData }
  | { status: "unavailable"; reason: string };

const count = (value: number) => formatQuantity(value, "count");

const STATE_DISPLAY: Record<CheckState, { tone: Tone; icon: LucideIcon }> = {
  pass: { tone: "success", icon: CircleCheck },
  warn: { tone: "warning", icon: TriangleAlert },
  fail: { tone: "danger", icon: CircleX },
  unknown: { tone: "info", icon: Info },
};

const capitalized = (text: string) =>
  text.charAt(0).toUpperCase() + text.slice(1);

function summaryItems(data: QualityPageData): StatStripItem[] {
  const gate = STATE_DISPLAY[data.gate.state];
  const emptySplits = data.split.parts.filter((part) => part.rows === 0).length;
  return [
    {
      id: "gate",
      icon: gate.icon,
      tone: gate.tone,
      caption: "Quality gate",
      value: capitalized(statusText(data.gate.status)),
      label: null,
    },
    {
      id: "anomalies",
      icon: data.gate.anomalies === 0 ? CircleCheck : CircleX,
      tone: data.gate.anomalies === 0 ? "success" : "danger",
      value: count(data.gate.anomalies),
      label:
        data.gate.anomalies === 1 ? "blocking anomaly" : "blocking anomalies",
    },
    {
      id: "telemetry",
      icon: Activity,
      caption: "Host telemetry present",
      value: `${shareText(data.telemetry.present, data.telemetry.total)} of steps`,
      label: null,
    },
    {
      id: "labelled",
      icon: Tag,
      caption: "Steps with a resource label",
      value:
        data.labelledRows === null
          ? "Not counted"
          : shareText(data.labelledRows, data.rows),
      label: null,
    },
    {
      id: "splits",
      icon: Network,
      caption: `Splits · ${data.split.method}`,
      value: `${count(data.split.parts.length)} splits${emptySplits ? ` · ${count(emptySplits)} empty` : ""}`,
      label: null,
    },
  ];
}

function qualityTabs(data: QualityPageData): TabbedViewTab[] {
  return [
    {
      id: "coverage",
      label: "Coverage",
      content: <TelemetryPanel data={data} />,
    },
    {
      id: "composition",
      label: "Composition",
      content: (
        <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
          <CompositionPanel data={data} />
          <WorkloadPanel data={data} />
        </div>
      ),
    },
    {
      id: "labels",
      label: "Labels",
      content: <LabelsPanel data={data} />,
    },
    {
      id: "splits",
      label: "Splits",
      badge: count(data.split.parts.length),
      content: <SplitPanel data={data} />,
    },
    {
      id: "validation",
      label: "Validation",
      badge: checkTally(data.checks),
      content: <ValidationPanel data={data} />,
    },
    {
      id: "signal",
      label: "Feature signal",
      content: <FeatureSignalPanel data={data} />,
    },
    {
      id: "fields",
      label: "Field health",
      badge: count(data.fieldHealth.rows.length),
      content: <FieldHealthPanel data={data} />,
    },
  ];
}

function ReadyQuality({
  data,
  initialTab,
}: {
  data: QualityPageData;
  initialTab?: string;
}) {
  return (
    <TabbedView
      label="Data quality views"
      initialTab={initialTab}
      tabs={qualityTabs(data)}
      header={
        <div className="space-y-4">
          <PageHeader
            title="Data quality"
            description="What the evidence in this release can and cannot support"
            scope={
              <>
                {count(data.rows)} steps · {count(data.runs)} runs ·{" "}
                {count(data.tasks)} tasks ·{" "}
                <span title={data.release} className="whitespace-nowrap">
                  {data.releaseName}
                </span>
              </>
            }
          />
          <StatStrip
            label="Release quality summary"
            items={summaryItems(data)}
            tone={data.gate.state === "pass" ? "success" : undefined}
          />
        </div>
      }
    />
  );
}

/** The Data quality page: an evidence audit of the current dataset release. */
export function QualityPage({
  state,
  initialTab,
}: {
  state: QualityPageState;
  /** The tab named by `?view=`; unknown ids open the first tab. */
  initialTab?: string;
}) {
  return (
    <WorkspacePage>
      {state.status === "ready" ? (
        <ReadyQuality data={state.data} initialTab={initialTab} />
      ) : (
        <div className="space-y-6">
          <PageHeader title="Data quality" />
          <UnavailableState
            title="Dataset unavailable"
            reason={
              <>
                {state.reason} Data quality reads the dataset release from{" "}
                <code>data/datasets</code>, or from{" "}
                <code>AGENTWATCH_DATASET_DIR</code> when it is set.
              </>
            }
          />
        </div>
      )}
    </WorkspacePage>
  );
}
