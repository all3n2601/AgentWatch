import Link from "next/link";
import { routes } from "@/lib/routes";
import { CircleCheck, Database, Layers, TriangleAlert } from "lucide-react";
import { AssistantChat } from "@/components/assistant-chat";
import { WorkspacePage } from "@/components/shell/app-shell";
import { AssistantRail } from "@/components/shell/assistant-rail";
import { PageHeader } from "@/components/shell/page-header";
import { Chip } from "@/components/ui/chip";
import {
  GlassPanel,
  InsetCard,
  PanelHeader,
} from "@/components/ui/glass-panel";
import { BarList } from "@/components/widgets/bar-list";
import { CampusLiveScene } from "@/components/widgets/campus-live-scene";
import { DonutChart, type DonutSlice } from "@/components/widgets/donut-chart";
import { StatCallout } from "@/components/widgets/stat";
import { EmptyState, UnavailableState } from "@/components/widgets/state-views";
import { campusStages, campusTraffic } from "@/lib/campus-status";
import { formatQuantity } from "@/lib/measurement-format";
import type { ModelOverview } from "@/lib/model-overview";
import type { Tone } from "@/lib/tone";
import { cn } from "@/lib/utils";

export type ModelOverviewState =
  | { status: "ready"; overview: ModelOverview }
  | { status: "unavailable"; reason: string }
  | { status: "no-models"; release: string };

/**
 * Host telemetry at or above this share of steps reads as healthy; below it
 * the callout asks for attention. Display emphasis only, not a quality gate.
 */
const TELEMETRY_HEALTHY_SHARE = 0.9;

const count = (value: number) => formatQuantity(value, "count");

function outcomeSlices(overview: ModelOverview): DonutSlice[] {
  const { passed, failed, unrecorded } = overview.outcomes;
  return [
    {
      id: "passed",
      label: "Passed",
      value: passed,
      color: "var(--aw-success-solid)",
    },
    {
      id: "failed",
      label: "Failed",
      value: failed,
      color: "var(--aw-danger-solid)",
    },
    ...(unrecorded > 0
      ? [
          {
            id: "unrecorded",
            label: "Unrecorded",
            value: unrecorded,
            color: "var(--aw-info-solid)",
          },
        ]
      : []),
  ];
}

function ModelSwitcher({ overview }: { overview: ModelOverview }) {
  if (overview.models.length < 2) return null;
  return (
    <nav aria-label="Agent model" className="flex flex-wrap gap-2">
      {overview.models.map((option) => {
        const current = option.id === overview.model.id;
        return (
          <Link
            key={option.id}
            href={routes.overview(option.id)}
            aria-current={current ? "page" : undefined}
            className={cn(
              "glass-raised inline-flex h-11 items-center rounded-full px-4 text-sm font-medium text-ink outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40",
              current && "border-success-line bg-brand-soft shadow-glow",
            )}
          >
            {option.name} · {count(option.runs)} runs
          </Link>
        );
      })}
    </nav>
  );
}

function ReadyOverview({ overview }: { overview: ModelOverview }) {
  const { outcomes, largestSlowdown } = overview;
  const telemetryShare =
    overview.steps === 0 ? 0 : overview.telemetryPresent / overview.steps;
  const telemetryTone: Tone =
    telemetryShare >= TELEMETRY_HEALTHY_SHARE ? "success" : "warning";
  const outcomeTone: Tone = outcomes.failed > 0 ? "warning" : "success";

  return (
    // In the one-screen layout the scene fills the space above the evidence
    // panel, which overlaps its foot as in the approved design.
    <div className="space-y-4 fit:flex fit:h-full fit:flex-col fit:space-y-0">
      <CampusLiveScene
        className="fit:flex-1 fit:pb-8"
        header={
          <PageHeader
            title={`${overview.model.name} performance across all runs`}
            scope={
              <>
                {count(overview.runs)} runs · {count(overview.steps)} measured
                steps ·{" "}
                <span title={overview.release} className="whitespace-nowrap">
                  {overview.releaseName}
                </span>
              </>
            }
            actions={<ModelSwitcher overview={overview} />}
          />
        }
        stages={campusStages(overview)}
        traffic={campusTraffic(overview)}
        releaseName={overview.releaseName}
        labels={{
          gateway: "Agent gateway",
          model: `${overview.model.name} · ${count(overview.runs)} runs`,
          retrieval: "Retrieval",
          cpuTool: "CPU tool",
          complete: "Complete",
        }}
        overlay={
          <>
            <StatCallout
              icon={outcomes.failed > 0 ? TriangleAlert : CircleCheck}
              tone={outcomeTone}
              label="Run outcomes"
              value={
                <>
                  {count(outcomes.passed)} passed · {count(outcomes.failed)}{" "}
                  failed
                  {outcomes.unrecorded > 0 &&
                    ` · ${count(outcomes.unrecorded)} unrecorded`}
                </>
              }
            />
            {largestSlowdown ? (
              <StatCallout
                icon={TriangleAlert}
                tone="warning"
                label="Largest observed slowdown"
                value={`${largestSlowdown.label} ${formatQuantity(largestSlowdown.ratio, "ratio")} vs clean`}
                detail={`Median of ${count(largestSlowdown.experiments)} experiments · not causal`}
              />
            ) : (
              <StatCallout
                icon={TriangleAlert}
                tone="info"
                label="Largest observed slowdown"
                value="No clean baseline"
              />
            )}
            <StatCallout
              icon={Database}
              tone={telemetryTone}
              label="Host telemetry"
              value={`${count(overview.telemetryPresent)} of ${count(overview.steps)} steps`}
            />
          </>
        }
      />

      <GlassPanel
        as="section"
        aria-labelledby="overview-evidence"
        className="fit:z-10 fit:-mt-6 fit:shrink-0"
      >
        <PanelHeader
          titleId="overview-evidence"
          icon={Layers}
          title="Model performance evidence"
          description={
            overview.runsWithoutModel > 0
              ? `${count(overview.runsWithoutModel)} coding and CPU runs without a model are excluded.`
              : undefined
          }
          actions={
            <Chip size="sm">
              {overview.model.id} · {count(overview.runs)} runs
            </Chip>
          }
        />
        <InsetCard className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] lg:gap-4 lg:divide-x lg:divide-line 2xl:gap-6">
          <div className="min-w-0 lg:pr-4 2xl:pr-6">
            <h3 className="mb-3 text-[15px] font-semibold text-ink">
              Call types{" "}
              <span className="text-xs font-normal text-ink-muted">steps</span>
            </h3>
            <BarList label="Call types" items={overview.callTypes} />
          </div>
          <div className="min-w-0 lg:pr-4 2xl:pr-6">
            <h3 className="mb-3 text-[15px] font-semibold text-ink">
              Excess-wait labels{" "}
              <span className="text-xs font-normal text-ink-muted">steps</span>
            </h3>
            <BarList label="Resource labels" items={overview.resourceLabels} />
          </div>
          <div>
            <h3 className="mb-3 text-[15px] font-semibold text-ink">
              Run outcomes
            </h3>
            <DonutChart
              label="Run outcomes"
              slices={outcomeSlices(overview)}
              size={112}
              legend="side"
            />
          </div>
        </InsetCard>
      </GlassPanel>
    </div>
  );
}

/** The Overview page: one agent model's runs in the selected dataset release. */
export function ModelOverviewPage({
  state,
  chatEnabled,
}: {
  state: ModelOverviewState;
  /** Whether the server has an assistant endpoint configured. */
  chatEnabled: boolean;
}) {
  return (
    <WorkspacePage
      rail={
        state.status === "ready" ? (
          <AssistantRail defaultOpen="wide">
            <AssistantChat
              enabled={chatEnabled}
              suggestions={[
                `Show me the failed runs for ${state.overview.model.name}`,
                "Which stage has the most excess wait?",
                "How much slower is mixed contention than clean?",
              ]}
            />
          </AssistantRail>
        ) : undefined
      }
    >
      {state.status === "ready" && <ReadyOverview overview={state.overview} />}
      {state.status === "unavailable" && (
        <div className="space-y-6">
          <PageHeader title="Overview" />
          <UnavailableState
            title="Dataset unavailable"
            reason={
              <>
                {state.reason} The Overview reads the dataset release: run{" "}
                <code>make pull-dataset</code>, or set{" "}
                <code>AGENTWATCH_DATASET_DIR</code>.
              </>
            }
          />
        </div>
      )}
      {state.status === "no-models" && (
        <div className="space-y-6">
          <PageHeader title="Overview" scope={state.release} />
          <EmptyState
            title="No agent-model runs"
            detail="This release has no runs that record an agent model, so there is no model population to summarise."
          />
        </div>
      )}
    </WorkspacePage>
  );
}
