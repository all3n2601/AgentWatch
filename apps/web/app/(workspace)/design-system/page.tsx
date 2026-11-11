import type { Metadata } from "next";
import { notFound } from "next/navigation";
import {
  Activity,
  CircleCheck,
  Clock,
  Cpu,
  Database,
  Download,
  FileText,
  FlaskConical,
  GitCompareArrows,
  Layers,
  Scale,
  Server,
  Settings,
  ShieldCheck,
  TriangleAlert,
} from "lucide-react";
import {
  ShowcaseFilters,
  ShowcaseSettingsForm,
} from "@/components/design-system/showcase-controls";
import { WorkspacePage } from "@/components/shell/app-shell";
import { AssistantRail } from "@/components/shell/assistant-rail";
import { PageHeader } from "@/components/shell/page-header";
import { Button } from "@/components/ui/button";
import { Chip } from "@/components/ui/chip";
import { CopyChip } from "@/components/ui/copy-chip";
import {
  GlassPanel,
  InsetCard,
  PanelHeader,
} from "@/components/ui/glass-panel";
import { StatusBadge } from "@/components/ui/status-badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AlertBanner } from "@/components/widgets/alert-banner";
import { BarList } from "@/components/widgets/bar-list";
import { CampusScene } from "@/components/widgets/campus-scene";
import { DeltaChip, RatioChip } from "@/components/widgets/delta-chip";
import { DonutChart } from "@/components/widgets/donut-chart";
import {
  Checklist,
  KeyValueList,
  RankedList,
} from "@/components/widgets/evidence-lists";
import { Measurement } from "@/components/widgets/measurement";
import { SegmentBar, WorkWaitLegend } from "@/components/widgets/segment-bar";
import { StatCallout, StatStrip } from "@/components/widgets/stat";
import {
  EmptyState,
  ErrorState,
  Skeleton,
  UnavailableState,
} from "@/components/widgets/state-views";
import { TONES } from "@/lib/tone";
import {
  SAMPLE_CALL_TYPES,
  SAMPLE_COVERAGE,
  SAMPLE_OUTCOMES,
  SAMPLE_RESOURCE_LABELS,
  SAMPLE_STEP_CONTEXT,
  SAMPLE_WORK_WAIT,
} from "./sample-data";

export const metadata: Metadata = { title: "Design system" };

function Section({
  id,
  title,
  children,
}: {
  id: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section aria-labelledby={id} className="space-y-4">
      <h2
        id={id}
        className="text-sm font-semibold tracking-[0.08em] text-ink-muted uppercase"
      >
        {title}
      </h2>
      {children}
    </section>
  );
}

/**
 * Component review page for the approved glass design system. Development
 * only: it shows illustrative values, which must not reach a deployed build.
 */
export default function DesignSystemPage() {
  if (process.env.NODE_ENV === "production") notFound();

  return (
    <WorkspacePage
      rail={
        <AssistantRail>
          <InsetCard className="text-[15px] text-ink">
            The assistant panel hosts record-aware answers. This review page
            shows its frame only.
          </InsetCard>
        </AssistantRail>
      }
    >
      <div className="space-y-10 fit:pb-6">
        <PageHeader
          title="Design system"
          description="Base components for the approved glass workspace"
          actions={<Chip icon={Layers}>docs/design-system.md</Chip>}
        />
        <AlertBanner
          tone="info"
          title="Sample values"
          detail="Every number on this page is illustrative, for comparing components with the approved mockups. None of it is recorded evidence."
        />

        <Section id="ds-overview" title="Overview scene">
          <CampusScene
            labels={[
              { anchor: "gateway", label: "Agent gateway", icon: FileText },
              { anchor: "model", label: "Model runtime", icon: Cpu },
              { anchor: "retrieval", label: "Retrieval", icon: Layers },
              { anchor: "cpuTool", label: "CPU tool", icon: Settings },
              { anchor: "complete", label: "Complete", icon: CircleCheck },
            ]}
            overlay={
              <>
                <StatCallout
                  icon={CircleCheck}
                  label="Run outcomes"
                  value="585 passed · 13 failed"
                />
                <StatCallout
                  icon={TriangleAlert}
                  tone="warning"
                  label="Largest observed slowdown"
                  value="Mixed 8.6×"
                />
                <StatCallout
                  icon={Database}
                  label="Host telemetry"
                  value="427 of 5,603 steps"
                />
              </>
            }
          />
          <GlassPanel as="section" aria-labelledby="ds-evidence" padding="lg">
            <PanelHeader
              titleId="ds-evidence"
              icon={Layers}
              title="Model performance evidence"
              actions={<Chip size="sm">qwen3:1.7b · 598 runs</Chip>}
            />
            <InsetCard className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] lg:divide-x lg:divide-line">
              <div className="min-w-0 lg:pr-8">
                <h3 className="mb-4 text-lg font-semibold text-ink">
                  Call types
                </h3>
                <BarList label="Call types" items={SAMPLE_CALL_TYPES} />
              </div>
              <div className="min-w-0 lg:pr-8">
                <h3 className="mb-4 text-lg font-semibold text-ink">
                  Resource labels
                </h3>
                <BarList
                  label="Resource labels"
                  items={SAMPLE_RESOURCE_LABELS}
                />
              </div>
              <div>
                <h3 className="mb-4 text-lg font-semibold text-ink">
                  Run outcomes
                </h3>
                <DonutChart label="Run outcomes" slices={SAMPLE_OUTCOMES} />
              </div>
            </InsetCard>
          </GlassPanel>
        </Section>

        <Section id="ds-runs" title="Runs: filters, summary strip, preview">
          <ShowcaseFilters />
          <StatStrip
            label="Runs in scope"
            items={[
              {
                id: "scope",
                icon: Layers,
                value: "598",
                label: "runs in scope",
              },
              {
                id: "passed",
                icon: CircleCheck,
                tone: "success",
                value: "585",
                label: "passed",
              },
              {
                id: "failed",
                icon: TriangleAlert,
                tone: "danger",
                value: "13",
                label: "failed",
              },
              {
                id: "experiments",
                icon: FlaskConical,
                value: "46",
                label: "experiments",
              },
            ]}
          />
          <GlassPanel
            as="section"
            aria-labelledby="ds-preview"
            className="max-w-xl"
          >
            <PanelHeader
              titleId="ds-preview"
              icon={Layers}
              title="Run preview"
              actions={
                <CopyChip
                  label="run ID"
                  value="4d3e2b31-9c7f-4d5e-8b9a-3a7e0c2f9d11"
                />
              }
            />
            <div className="space-y-3">
              <InsetCard className="flex items-center gap-3">
                <FileText className="size-6 shrink-0 text-ink" aria-hidden />
                <div className="min-w-0 flex-1">
                  <p className="text-lg font-semibold text-ink">
                    fix-window-count
                  </p>
                  <p className="text-sm text-ink-muted">Task</p>
                </div>
                <StatusBadge label="Degraded" />
              </InsetCard>
              <InsetCard>
                <h3 className="mb-3 font-semibold text-ink">
                  Work versus wait
                </h3>
                <SegmentBar
                  label="Work versus wait"
                  unit="s"
                  segments={SAMPLE_WORK_WAIT}
                />
              </InsetCard>
              <div className="flex flex-wrap gap-2">
                <Button variant="brand-soft" size="pill">
                  <Activity aria-hidden /> Open diagnosis
                </Button>
                <Button variant="glass" size="pill">
                  <GitCompareArrows aria-hidden /> Compare
                </Button>
                <Button variant="glass" size="pill">
                  <Download aria-hidden /> Export evidence
                </Button>
              </div>
            </div>
          </GlassPanel>
        </Section>

        <Section
          id="ds-detail"
          title="Run diagnosis: tabs, banner, step context, evidence"
        >
          <Tabs defaultValue="diagnosis">
            <TabsList variant="glass" aria-label="Run views">
              {["Diagnosis", "Timeline", "Agent", "Resources", "Evidence"].map(
                (tab) => (
                  <TabsTrigger key={tab} value={tab.toLowerCase()}>
                    {tab}
                  </TabsTrigger>
                ),
              )}
            </TabsList>
            <TabsContent value="diagnosis" className="mt-4 space-y-4">
              <AlertBanner
                tone="caution"
                title="Mixed contention produced the largest observed slowdown"
                detail="11.4× vs clean · Observed, not causal"
              />
              <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
                <GlassPanel as="section" aria-labelledby="ds-step">
                  <PanelHeader
                    titleId="ds-step"
                    title="Selected step context"
                    actions={<WorkWaitLegend />}
                  />
                  <SegmentBar
                    label="Selected step context"
                    unit="s"
                    segments={SAMPLE_STEP_CONTEXT}
                    showLabels
                    showLegend={false}
                  />
                </GlassPanel>
                <GlassPanel as="section" aria-labelledby="ds-evidence-step">
                  <PanelHeader
                    titleId="ds-evidence-step"
                    icon={Layers}
                    title="Evidence for selected step"
                    description="Retrieval · search-index"
                  />
                  <KeyValueList
                    items={[
                      {
                        id: "queue",
                        icon: Layers,
                        label: "Queue",
                        value: <Measurement value={69} unit="ms" />,
                      },
                      {
                        id: "exec",
                        icon: Settings,
                        label: "Execution",
                        value: <Measurement value={2.18} unit="s" />,
                      },
                      {
                        id: "cpu",
                        icon: Cpu,
                        label: "CPU",
                        value: (
                          <Measurement
                            value={null}
                            unit="percent"
                            state="not_instrumented"
                            showState
                          />
                        ),
                      },
                      {
                        id: "est",
                        icon: Clock,
                        label: "Estimated wait",
                        value: (
                          <Measurement
                            value={10.37}
                            unit="s"
                            state="estimated"
                            showState
                          />
                        ),
                      },
                      {
                        id: "zero",
                        icon: Server,
                        label: "Retries",
                        value: <Measurement value={0} unit="count" />,
                      },
                    ]}
                  />
                </GlassPanel>
              </div>
            </TabsContent>
            {["timeline", "agent", "resources", "evidence"].map((tab) => (
              <TabsContent key={tab} value={tab} className="mt-4">
                <EmptyState
                  title="Tab content"
                  detail="Each run tab renders its own view."
                />
              </TabsContent>
            ))}
          </Tabs>
        </Section>

        <Section
          id="ds-compare"
          title="Compare and experiments: deltas, ranking, checks"
        >
          <div className="grid gap-4 lg:grid-cols-3">
            <GlassPanel as="section" aria-labelledby="ds-ratios">
              <PanelHeader
                titleId="ds-ratios"
                icon={Scale}
                title="Ratio emphasis"
              />
              <div className="flex flex-wrap gap-2">
                {[0.9, 1.05, 1.4, 3.9, 8.6, 11.4].map((ratio) => (
                  <RatioChip key={ratio} ratio={ratio} />
                ))}
                <RatioChip ratio={null} />
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                <DeltaChip deltaSeconds={24.85} ratio={11.4} />
                <DeltaChip deltaSeconds={0.15} ratio={3.5} />
                <DeltaChip deltaSeconds={null} ratio={null} />
              </div>
            </GlassPanel>
            <GlassPanel as="section" aria-labelledby="ds-ranked">
              <PanelHeader
                titleId="ds-ranked"
                icon={Layers}
                title="Largest differences"
              />
              <RankedList
                items={[
                  {
                    id: "model",
                    label: "Model calls",
                    value: <DeltaChip deltaSeconds={12.88} ratio={15.6} />,
                  },
                  {
                    id: "retrieval",
                    label: "Retrieval",
                    value: <DeltaChip deltaSeconds={5.85} ratio={10.4} />,
                  },
                  {
                    id: "cpu",
                    label: "CPU tools",
                    value: <DeltaChip deltaSeconds={5.37} ratio={9.8} />,
                  },
                ]}
              />
            </GlassPanel>
            <GlassPanel as="section" aria-labelledby="ds-checks">
              <PanelHeader
                titleId="ds-checks"
                icon={ShieldCheck}
                title="Acceptance checks"
              />
              <Checklist
                items={[
                  {
                    id: "replay",
                    label: "Replay data complete",
                    state: "pass",
                  },
                  {
                    id: "controls",
                    label: "Clean controls present",
                    state: "pass",
                  },
                  {
                    id: "injection",
                    label: "Contention injection applied",
                    state: "warn",
                  },
                  {
                    id: "accuracy",
                    label: "Attribution accuracy",
                    state: "unknown",
                    detail: "Not reported in steps-v1.3",
                  },
                ]}
              />
            </GlassPanel>
          </div>
          <GlassPanel
            as="section"
            aria-labelledby="ds-config"
            className="max-w-xl"
          >
            <PanelHeader
              titleId="ds-config"
              icon={ShieldCheck}
              title="Configuration check"
            />
            <KeyValueList
              dense
              items={[
                {
                  id: "workload",
                  label: "Workload",
                  value: "fix-window-count",
                  status: { label: "Match" },
                },
                {
                  id: "model",
                  label: "Model",
                  value: "Qwen3 1.7B",
                  status: { label: "Match" },
                },
                {
                  id: "workers",
                  label: "Worker capacity",
                  value: "8 workers",
                  status: { label: "Mismatch" },
                },
              ]}
            />
          </GlassPanel>
        </Section>

        <Section id="ds-quality" title="Data quality: coverage and validation">
          <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
            <GlassPanel as="section" aria-labelledby="ds-coverage">
              <PanelHeader
                titleId="ds-coverage"
                icon={Layers}
                title="Telemetry coverage"
              />
              <SegmentBar
                label="Telemetry coverage"
                unit="count"
                segments={SAMPLE_COVERAGE}
              />
            </GlassPanel>
            <GlassPanel as="section" aria-labelledby="ds-validation">
              <PanelHeader
                titleId="ds-validation"
                icon={ShieldCheck}
                title="Validation"
              />
              <Checklist
                showStateBadge
                items={[
                  { id: "schema", label: "Schema", state: "pass" },
                  { id: "finite", label: "Finite values", state: "pass" },
                  { id: "ordering", label: "Ordering", state: "fail" },
                ]}
              />
            </GlassPanel>
          </div>
        </Section>

        <Section
          id="ds-states"
          title="Loading, empty, unavailable and error states"
        >
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <GlassPanel className="space-y-3">
              <Skeleton label="Loading runs" className="h-6 w-2/3" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-5/6" />
            </GlassPanel>
            <EmptyState
              title="No runs yet"
              detail="Recorded runs appear here after the first import."
            />
            <UnavailableState reason="The ingestion API did not respond. History is unavailable, not empty." />
            <ErrorState
              detail="The run could not be loaded."
              action={
                <Button variant="glass" size="pill">
                  Retry
                </Button>
              }
            />
          </div>
        </Section>

        <Section id="ds-settings" title="Settings form controls">
          <GlassPanel padding="lg">
            <ShowcaseSettingsForm />
          </GlassPanel>
        </Section>

        <Section id="ds-tones" title="Tones and buttons">
          <GlassPanel className="flex flex-wrap items-center gap-3">
            {TONES.map((tone) => (
              <StatusBadge
                key={tone}
                tone={tone}
                label={tone[0].toUpperCase() + tone.slice(1)}
              />
            ))}
            <StatusBadge appearance="dot" label="Degraded" />
            <StatusBadge appearance="plain" label="Failed" />
            <Button variant="brand" size="pill">
              Open candidate
            </Button>
            <Button variant="glass" size="pill">
              Open baseline
            </Button>
          </GlassPanel>
        </Section>
      </div>
    </WorkspacePage>
  );
}
