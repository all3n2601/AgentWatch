"use client";

import { useState, type ReactNode } from "react";
import { FileSearch, Scale } from "lucide-react";
import {
  FILL_PANEL,
  PANEL_SCROLL,
  TAB_GRID,
} from "@/components/shell/panel-layout";
import {
  GlassPanel,
  InsetCard,
  PanelHeader,
} from "@/components/ui/glass-panel";
import { StatusBadge } from "@/components/ui/status-badge";
import { KeyValueList } from "@/components/widgets/evidence-lists";
import { Measurement } from "@/components/widgets/measurement";
import {
  EmptyState,
  ErrorState,
  Skeleton,
  UnavailableState,
} from "@/components/widgets/state-views";
import { useAttributionFeed } from "@/components/use-attribution-feed";
import type { AttributionFeed } from "@/lib/attribution-poll";
import {
  classLabel,
  comparisonDisplay,
  comparisonGroups,
  formatProbability,
  hasMixedEvidence,
  modelStateDisplay,
  outcomeLabel,
  percentReading,
  probabilityEntries,
  rulesCoverageNote,
  rulesVerdictLabel,
  secondsReading,
  telemetryLabel,
  unavailableReason,
  type AttributionRow,
  type ComparisonGroup,
  type MalformedRow,
  type ModelStatus,
  type ValidRow,
} from "@/lib/attribution-view";
import { cn } from "@/lib/utils";

// The Attribution tab of a live run: each terminal step's rules verdict beside the served
// model's outcome (docs/attribution-serving.md, "Dashboard handoff"). Mounted only while the
// tab is open, so its polling stops when the tab closes.

const NO_CAUSE =
  "A verdict names the resource a step likely waited on; it does not establish a cause.";

const seconds = (value: number) => `${Math.round(value / 1000)} s`;
const shortKey = (key: string) => key.slice(0, 12);

function FeedStatus({ feed }: { feed: AttributionFeed }) {
  const { connection } = feed;
  return (
    <span role="status" aria-live="polite">
      {connection.status === "connecting" && "Reading attributions…"}
      {connection.status === "connected" &&
        `${feed.items.length} attributed step${feed.items.length === 1 ? "" : "s"} from the live API.`}
      {connection.status === "retrying" &&
        `Attributions could not be refreshed (${connection.message}); retrying in ${seconds(connection.retryInMs)}.`}
      {connection.status === "stopped" &&
        `Attributions could not be read: ${connection.message}`}
    </span>
  );
}

/** The serving model's state now, from the latest page. */
function ModelStatusCard({ model }: { model: ModelStatus }) {
  const state = modelStateDisplay(model.state);
  const identity = model.identity;
  return (
    <InsetCard className="space-y-2">
      <h3 className="text-sm font-semibold text-ink">Serving model now</h3>
      <div className="flex flex-wrap items-center gap-2">
        <StatusBadge label={state.label} tone={state.tone} />
        {model.not_promoted === true && (
          <StatusBadge label="Not promoted" tone="warning" />
        )}
      </div>
      {identity ? (
        <p className="text-sm text-ink-muted">
          Training{" "}
          <span className="font-mono break-all text-ink">
            {identity.training_version}
          </span>
          {" · "}calibration{" "}
          <span className="font-mono break-all text-ink">
            {identity.calibration_version}
          </span>
          {" · "}abstains below{" "}
          {formatProbability(identity.abstention_threshold)}
        </p>
      ) : (
        <p className="text-sm text-ink-muted">No model pair is loaded.</p>
      )}
      {model.state === "load_failed" && model.reason && (
        <p className="text-sm text-ink-muted">
          Load check failed: {model.reason}.
        </p>
      )}
    </InsetCard>
  );
}

function versionsText(values: (string | null)[]) {
  return values.map((value) => value ?? "none").join(", ");
}

function GroupSummary({ group }: { group: ComparisonGroup }) {
  const { counts } = group;
  const compared = counts.agree + counts.differ;
  return (
    <li className="space-y-1 text-sm">
      <p className="font-semibold text-ink">
        {group.current
          ? "Current serving configuration"
          : `Earlier serving key ${shortKey(group.servingKey)}`}
        <span className="font-normal text-ink-muted">
          {" "}
          · {group.rows} step{group.rows === 1 ? "" : "s"} · rules{" "}
          {versionsText(group.rulesVersions)} · training{" "}
          {versionsText(group.trainingVersions)} · calibration{" "}
          {versionsText(group.calibrationVersions)}
        </span>
      </p>
      <p className="text-ink-muted">
        {compared} compared: {counts.agree} name the same resource,{" "}
        {counts.differ} differ. {counts.model_abstained} model uncertain,{" "}
        {counts.no_model_output} rules only, {counts.verdict_unrecognised}{" "}
        unrecognised verdict
        {group.evidenceChanged > 0 &&
          `, ${group.evidenceChanged} with changed evidence`}
        .
      </p>
    </li>
  );
}

/** Counts per compatible group; rows of different serving keys are never added together. */
function ComparisonSummary({ rows }: { rows: AttributionRow[] }) {
  const groups = comparisonGroups(rows);
  const malformed = rows.filter((row) => row.kind === "malformed").length;
  return (
    <InsetCard className="space-y-2">
      <h3 className="text-sm font-semibold text-ink">Rules against model</h3>
      <ul
        className="space-y-2"
        aria-label="Comparison by serving configuration"
      >
        {groups.map((group) => (
          <GroupSummary
            key={`${group.servingKey}:${group.featureVersion}`}
            group={group}
          />
        ))}
      </ul>
      {groups.length > 1 && (
        <p className="text-sm text-ink-muted">
          Each serving key fixes the rules version, model pair and feature
          definition, so counts are kept per key and never combined.
        </p>
      )}
      {malformed > 0 && (
        <p className="text-sm text-ink-muted">
          {malformed} malformed row{malformed === 1 ? "" : "s"} not counted.
        </p>
      )}
      <p className="text-sm text-ink-muted">{NO_CAUSE}</p>
    </InsetCard>
  );
}

function ModelCell({ row }: { row: ValidRow }) {
  const { model } = row;
  if (model.outcome === "predicted" && model.predicted_class !== null)
    return <>{classLabel(model.predicted_class)}</>;
  if (model.outcome === "abstained")
    return <StatusBadge label="Uncertain" tone="caution" appearance="plain" />;
  return <span className="text-ink-muted">{outcomeLabel(model.outcome)}</span>;
}

function RowFlags({ row }: { row: AttributionRow }) {
  if (row.kind === "malformed")
    return <StatusBadge label="Malformed" tone="danger" appearance="plain" />;
  return (
    <span className="flex flex-col items-start gap-1">
      {row.comparison === "differ" && (
        <StatusBadge
          label="Rules and model differ"
          tone="warning"
          appearance="plain"
        />
      )}
      {row.evidenceChanged && (
        <StatusBadge
          label="Evidence changed"
          tone="warning"
          appearance="plain"
        />
      )}
      {!row.current && (
        <StatusBadge
          label="Earlier serving key"
          tone="info"
          appearance="plain"
        />
      )}
    </span>
  );
}

function AttributionTable({
  rows,
  selected,
  onSelect,
}: {
  rows: AttributionRow[];
  selected: number | null;
  onSelect: (sequence: number) => void;
}) {
  return (
    <div
      role="region"
      aria-label="Step attributions table"
      tabIndex={0}
      className="relative overflow-x-auto rounded-xl outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40"
    >
      <table className="w-full min-w-[44rem] text-left text-sm">
        <caption className="sr-only">
          Each attributed step: the rules verdict, the model outcome, its
          calibrated confidence, the estimated excess latency and telemetry
          status.
        </caption>
        <thead className="text-xs text-ink-muted">
          <tr className="border-b border-line">
            <th scope="col" className="py-2 pr-3 font-medium">
              Step
            </th>
            <th scope="col" className="py-2 pr-3 font-medium">
              Rules
            </th>
            <th scope="col" className="py-2 pr-3 font-medium">
              Model
            </th>
            <th scope="col" className="py-2 pr-3 font-medium">
              Confidence
            </th>
            <th scope="col" className="py-2 pr-3 font-medium">
              Excess (estimate)
            </th>
            <th scope="col" className="py-2 pr-3 font-medium">
              Telemetry
            </th>
            <th scope="col" className="py-2 font-medium">
              Flags
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {rows.map((row) => {
            const isSelected = row.sequence === selected;
            const name = row.stepId ?? `Row ${row.sequence}`;
            return (
              <tr
                key={row.sequence}
                data-step={row.stepId ?? undefined}
                className={cn(isSelected && "bg-brand-soft")}
              >
                <th scope="row" className="py-1 pr-3 font-normal">
                  <button
                    type="button"
                    aria-pressed={isSelected}
                    onClick={() => onSelect(row.sequence)}
                    className="min-h-11 max-w-[14rem] rounded-lg px-1 text-left font-mono break-all text-ink outline-none hover:underline focus-visible:ring-[3px] focus-visible:ring-brand/40"
                  >
                    {name}
                  </button>
                </th>
                {row.kind === "valid" ? (
                  <>
                    <td className="py-1 pr-3">
                      {rulesVerdictLabel(row.rules.verdict)}
                    </td>
                    <td className="py-1 pr-3">
                      <ModelCell row={row} />
                    </td>
                    <td className="py-1 pr-3 tabular-nums">
                      {row.model.confidence === null ? (
                        <span className="text-ink-subtle">—</span>
                      ) : (
                        formatProbability(row.model.confidence)
                      )}
                    </td>
                    <td className="py-1 pr-3">
                      <Measurement
                        value={percentReading(
                          row.model.excess_fraction_estimate,
                        )}
                        unit="percent"
                        state="estimated"
                      />
                    </td>
                    <td className="py-1 pr-3">
                      {telemetryLabel(row.telemetryStatus)}
                    </td>
                  </>
                ) : (
                  <td colSpan={5} className="py-1 pr-3 text-ink-muted">
                    Not shown: the row failed validation.
                  </td>
                )}
                <td className="py-1">
                  <RowFlags row={row} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <InsetCard className="space-y-2">
      <h3 className="text-sm font-semibold text-ink">{title}</h3>
      {children}
    </InsetCard>
  );
}

function SecondsValue({ value }: { value: number | null }) {
  const reading = secondsReading(value);
  return <Measurement value={reading.value} unit={reading.unit} showState />;
}

function ModelOutput({ row }: { row: ValidRow }) {
  const { model } = row;
  if (model.outcome === "unavailable")
    return (
      <p className="text-sm text-ink-muted">
        No model output. {unavailableReason(model.reason)}
      </p>
    );
  if (model.outcome === "not_scored_step_state")
    return (
      <p className="text-sm text-ink-muted">
        No model output: the model scores only completed steps, and this step{" "}
        {row.stepState === "failed" ? "failed" : "was cancelled"}.
      </p>
    );
  const { confidence, abstention_threshold: threshold } = model;
  // A validated scored row always has both; anything else was shown as malformed.
  if (confidence === null || threshold === null) return null;
  const probabilities = probabilityEntries(model.calibrated_probabilities);
  return (
    <div className="space-y-2">
      {model.outcome === "abstained" || model.predicted_class === null ? (
        <p className="text-sm text-ink">
          <StatusBadge label="Uncertain" tone="caution" /> Confidence{" "}
          {formatProbability(confidence)} is below the abstention threshold{" "}
          {formatProbability(threshold)}, so no resource is attributed. This is
          not “None identified”.
        </p>
      ) : (
        <p className="text-sm text-ink">
          {model.predicted_class === "none" ? (
            "No contended resource identified"
          ) : (
            <>
              Likely waited on{" "}
              <strong>{classLabel(model.predicted_class)}</strong>
            </>
          )}
          : calibrated confidence {formatProbability(confidence)}, at or above
          the abstention threshold {formatProbability(threshold)}.
        </p>
      )}
      <div>
        <h4 className="text-xs font-medium text-ink-muted">
          Calibrated probabilities
        </h4>
        <ul
          aria-label="Calibrated probabilities"
          className="mt-1 divide-y divide-line text-sm"
        >
          {probabilities.map((entry) => (
            <li key={entry.id} className="flex justify-between gap-3 py-1">
              <span>{entry.label}</span>
              <span className="tabular-nums">
                {formatProbability(entry.value)}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function ValidEvidence({ row }: { row: ValidRow }) {
  const { features, telemetry, timings } = row.feature;
  const comparison = comparisonDisplay(row.comparison);
  const note = rulesCoverageNote(row.rules.version);
  return (
    <div className="space-y-3">
      {hasMixedEvidence(row) && (
        <InsetCard
          role="status"
          className="space-y-1 border border-warning-line"
        >
          <p className="text-sm font-semibold text-ink">Mixed evidence</p>
          {row.comparison === "differ" &&
            row.model.predicted_class !== null && (
              <p className="text-sm text-ink-muted">
                The rules name {rulesVerdictLabel(row.rules.verdict)} and the
                model names {classLabel(row.model.predicted_class)} for the same
                feature row.
              </p>
            )}
          {row.evidenceChanged && (
            <p className="text-sm text-ink-muted">
              Evidence changed: a later audit computed a different feature row
              for this step. The verdicts below were made from the original row.
            </p>
          )}
        </InsetCard>
      )}
      <Section title="Verdicts">
        <KeyValueList
          dense
          items={[
            {
              id: "rules",
              label: `Rules (${row.rules.version})`,
              value: rulesVerdictLabel(row.rules.verdict),
            },
            {
              id: "model",
              label: "Model",
              value:
                row.model.outcome === "predicted" &&
                row.model.predicted_class !== null
                  ? classLabel(row.model.predicted_class)
                  : outcomeLabel(row.model.outcome),
            },
            {
              id: "comparison",
              label: "Comparison",
              value: "",
              status: comparison,
            },
          ]}
        />
        {note && <p className="text-sm text-ink-muted">{note}</p>}
      </Section>
      <Section title="Model output">
        <ModelOutput row={row} />
        <KeyValueList
          dense
          items={[
            {
              id: "excess",
              label: "Excess latency",
              value: (
                <Measurement
                  value={percentReading(row.model.excess_fraction_estimate)}
                  unit="percent"
                  state="estimated"
                  showState
                />
              ),
            },
          ]}
        />
        <p className="text-sm text-ink-muted">
          The model’s estimated share of the step’s duration spent as excess
          waiting: an estimate, not a measured queue delay.
        </p>
      </Section>
      <Section title="Measured timings">
        <KeyValueList
          dense
          items={[
            {
              id: "queue",
              label: "Queue",
              value: <SecondsValue value={timings.queue_seconds} />,
            },
            {
              id: "execution",
              label: "Execution wall time",
              value: <SecondsValue value={timings.execution_seconds} />,
            },
            {
              id: "duration",
              label: "Submission to finish",
              value: <SecondsValue value={timings.duration_seconds} />,
            },
            {
              id: "cpu",
              label: "Process CPU time",
              value: <SecondsValue value={timings.cpu_seconds} />,
            },
          ]}
        />
      </Section>
      <Section title="Telemetry coverage">
        <KeyValueList
          dense
          items={[
            {
              id: "status",
              label: "Telemetry",
              value: telemetryLabel(telemetry.status),
            },
            {
              id: "samples",
              label: "Samples in the step window",
              value: (
                <Measurement value={telemetry.sample_count} unit="count" />
              ),
            },
            {
              id: "coverage",
              label: "Coverage",
              value: (
                <Measurement
                  value={percentReading(telemetry.coverage)}
                  unit="percent"
                  showState
                />
              ),
            },
            {
              id: "pressure",
              label: "Max CPU pressure (some avg10)",
              value: (
                <Measurement
                  value={features.cpu_pressure_max_avg10}
                  unit="percent"
                  showState
                />
              ),
            },
            {
              id: "throttled",
              label: "cgroup throttled time",
              value: (
                <Measurement
                  value={
                    features.cgroup_throttled_usec_delta === null
                      ? null
                      : features.cgroup_throttled_usec_delta / 1000
                  }
                  unit="ms"
                  showState
                />
              ),
            },
          ]}
        />
      </Section>
      <Section title="Inputs the model saw as missing">
        {row.missingInputs.length > 0 ? (
          <ul
            aria-label="Missing model inputs"
            className="flex flex-wrap gap-1.5"
          >
            {row.missingInputs.map((name) => (
              <li
                key={name}
                className="glass-raised rounded-full px-2.5 py-0.5 font-mono text-xs"
              >
                {name}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-ink-muted">None.</p>
        )}
        <p className="text-sm text-ink-muted">
          Missing readings are passed to the model as missing, never as 0.
        </p>
      </Section>
      <Section title="Versions">
        <KeyValueList
          dense
          items={[
            {
              id: "rules",
              label: "Rules",
              value: (
                <span className="font-mono break-all">{row.rules.version}</span>
              ),
            },
            {
              id: "training",
              label: "Model training",
              value: (
                <span className="font-mono break-all">
                  {row.model.training_version ?? "None loaded"}
                </span>
              ),
            },
            {
              id: "calibration",
              label: "Calibration",
              value: (
                <span className="font-mono break-all">
                  {row.model.calibration_version ?? "None loaded"}
                </span>
              ),
            },
            {
              id: "features",
              label: "Features",
              value: (
                <span className="font-mono break-all">
                  {row.featureVersion}
                </span>
              ),
            },
            {
              id: "key",
              label: "Serving key",
              value: (
                <span className="font-mono break-all">
                  {shortKey(row.servingKey)}
                </span>
              ),
              status: row.current
                ? { label: "Current", tone: "info" }
                : { label: "Earlier", tone: "warning" },
            },
          ]}
        />
        {row.model.not_promoted === true && (
          <StatusBadge label="Not promoted" tone="warning" />
        )}
      </Section>
    </div>
  );
}

function MalformedEvidence({ row }: { row: MalformedRow }) {
  return (
    <InsetCard className="space-y-2">
      <StatusBadge label="Malformed" tone="danger" />
      <p className="text-sm text-ink-muted">
        This row failed validation, so none of its values are shown. Other rows
        are unaffected.
      </p>
      <ul
        aria-label="Validation problems"
        className="list-disc space-y-1 pl-5 text-sm text-ink"
      >
        {row.reasons.map((reason) => (
          <li key={reason}>{reason}</li>
        ))}
      </ul>
    </InsetCard>
  );
}

function Body({
  feed,
  rows,
  selected,
  onSelect,
}: {
  feed: AttributionFeed;
  rows: AttributionRow[];
  selected: number | null;
  onSelect: (sequence: number) => void;
}) {
  const { connection, model } = feed;
  if (connection.status === "stopped" && connection.kind === "not_found")
    return (
      <ErrorState
        title="No attributions for this run"
        detail="The live API has no run with this UUID; no other run is shown."
      />
    );
  if (!model) {
    if (connection.status === "retrying" || connection.status === "stopped")
      return (
        <UnavailableState
          title="Attributions unavailable"
          reason={connection.message}
        />
      );
    return <Skeleton className="h-40" label="Loading attributions" />;
  }
  return (
    <div className="space-y-3">
      <ModelStatusCard model={model} />
      {!feed.loaded ? (
        <Skeleton className="h-24" label="Reading every attribution page" />
      ) : rows.length === 0 ? (
        <EmptyState
          title="No attributed steps yet"
          detail="A row appears once a step completes, fails or is cancelled."
        />
      ) : (
        <>
          <ComparisonSummary rows={rows} />
          <AttributionTable
            rows={rows}
            selected={selected}
            onSelect={onSelect}
          />
        </>
      )}
    </div>
  );
}

/** The Attribution tab. `runOpen` is false once the live run has closed. */
export function LiveAttributionView({
  runId,
  runOpen,
}: {
  runId: string;
  runOpen: boolean;
}) {
  const { feed, rows } = useAttributionFeed(runId, runOpen);
  const [chosen, setChosen] = useState<number | null>(null);
  const selectedRow =
    rows.find((row) => row.sequence === chosen) ?? rows[0] ?? null;
  return (
    // TAB_GRID sets columns from `xl` only; below it the implicit `auto` column would grow to
    // the table's minimum width and scroll the page, so the single column is capped here.
    <div className={cn(TAB_GRID, "grid-cols-[minmax(0,1fr)] fit:h-full")}>
      <GlassPanel
        as="section"
        aria-labelledby="live-attribution"
        className={FILL_PANEL}
      >
        <PanelHeader
          titleId="live-attribution"
          icon={Scale}
          title="Step attribution"
          description={<FeedStatus feed={feed} />}
          className="fit:shrink-0"
        />
        <div
          role="group"
          aria-label="Step attribution content"
          tabIndex={0}
          className={PANEL_SCROLL}
        >
          <Body
            feed={feed}
            rows={rows}
            selected={selectedRow?.sequence ?? null}
            onSelect={setChosen}
          />
        </div>
      </GlassPanel>
      <GlassPanel
        as="section"
        aria-labelledby="attribution-evidence"
        className={FILL_PANEL}
      >
        <PanelHeader
          titleId="attribution-evidence"
          icon={FileSearch}
          title="Step evidence"
          description={
            selectedRow ? (
              <span className="font-mono break-all">
                {selectedRow.stepId ?? `Row ${selectedRow.sequence}`}
              </span>
            ) : (
              "Select a step to see its evidence."
            )
          }
          className="fit:shrink-0"
        />
        <div
          role="group"
          aria-label="Step evidence content"
          tabIndex={0}
          className={PANEL_SCROLL}
        >
          {selectedRow?.kind === "valid" && <ValidEvidence row={selectedRow} />}
          {selectedRow?.kind === "malformed" && (
            <MalformedEvidence row={selectedRow} />
          )}
        </div>
      </GlassPanel>
    </div>
  );
}
