import {
  conditionRatios,
  summarizeExperiments,
  type DatasetRun,
  type ExperimentSummary,
} from "./dataset-runs";
import { replayConditionName } from "./dataset-view";

/**
 * The Experiments page model: replay experiments of one dataset release, each
 * injected condition compared with the same experiment's clean replays. Every
 * figure here is observed under injected contention; none is a causal claim.
 */

/** Injected conditions in the order the protocol replays them. */
export const INJECTED_CONDITIONS = [
  "cpu",
  "retrieval",
  "inference",
  "mixed",
] as const;
export type InjectedCondition = (typeof INJECTED_CONDITIONS)[number];

const SHORT_NAMES: Record<string, string> = {
  clean: "Clean",
  cpu: "CPU",
  retrieval: "Retrieval",
  inference: "Inference",
  mixed: "Mixed",
};
export const conditionShortName = (condition: string) =>
  SHORT_NAMES[condition] ?? replayConditionName(condition);

/** One replay step's protocol role and label outcome (columns of `steps.parquet`). */
export type ReplayStep = {
  experiment: string;
  replayIndex: number;
  /** `baseline`, `clean_control` or `contended`. */
  replayRole: string;
  /** e.g. `clean_within_noise`, `contention_exceeded_threshold`. */
  labelBasis: string;
};

export type BuildRejection = { origin: string; reason: string; stage: string };

/** The slice of the release's build report the acceptance checks read. */
export type BuildReport = {
  buildId: string;
  /** Experiments the build received, before any were rejected; null when not recorded. */
  experiments: number | null;
  rejections: BuildRejection[];
};

export type ExperimentsPageInput = {
  release: string;
  runs: DatasetRun[];
  validation: { status: string; validatedAt: string; gateVersion: string };
  /** Null when the role and label columns could not be read. */
  replaySteps: ReplayStep[] | null;
  /** Null when no readable build report names this release. */
  build: BuildReport | null;
};

export type CheckState = "pass" | "warn" | "fail" | "unknown";
export type ExperimentCheck = {
  id: string;
  label: string;
  state: CheckState;
  detail: string;
};

export type ConditionImpact = {
  id: InjectedCondition;
  label: string;
  shortLabel: string;
  /** Median across experiments of (condition median / same experiment's clean median). */
  medianRatio: number;
  minRatio: number;
  maxRatio: number;
  experiments: number;
  replays: number;
  failedReplays: number;
  steps: number;
  /** Steps whose label names a resource (excess wait). */
  excessSteps: number;
  /** Steps whose host samples overlap the step. */
  telemetryPresent: number;
  injectionMethods: string[];
};

export type SequenceSlot = {
  index: number;
  condition: string;
  /** Most common step role at this replay position; null when roles were not read. */
  role: string | null;
  experiments: number;
  /** Replay total time over its experiment's clean median, across experiments. */
  medianRatio: number | null;
  minRatio: number | null;
  maxRatio: number | null;
};

export type ReplaySequence = {
  slots: SequenceSlot[];
  /** Experiments that follow this order exactly. */
  matching: number;
};

export type ExperimentRow = {
  id: string;
  shortId: string;
  taskGroup: string;
  recordedAt: string | null;
  replays: number;
  failedReplays: number;
  /** Replays with no recorded task outcome; never counted as passed. */
  unrecordedReplays: number;
  cleanReplays: number;
  cleanMedianSeconds: number | null;
  /** Coefficient of variation of the clean replays' total time; null below two replays. */
  cleanCv: number | null;
  ratios: Record<InjectedCondition, number | null>;
};

export type TaskGroupRows = { taskGroup: string; experiments: ExperimentRow[] };

export type ExperimentsPage = {
  release: string;
  releaseName: string;
  experiments: number;
  replayRuns: number;
  replaySteps: number;
  models: string[];
  recordedRange: { first: string; last: string } | null;
  conditions: ConditionImpact[];
  largest: ConditionImpact | null;
  clean: {
    replays: number;
    failedReplays: number;
    steps: number;
    telemetryPresent: number;
    medianCv: number | null;
  };
  sequence: ReplaySequence | null;
  /** Replays per experiment by protocol role, from the replay sequence. */
  roleCounts: { role: string; replays: number }[];
  /** Step label outcomes of the replay steps, largest first; null when not read. */
  labelOutcomes: { basis: string; steps: number }[] | null;
  checks: ExperimentCheck[];
  reading: string | null;
  groups: TaskGroupRows[];
};

/**
 * Passed checks out of all checks. A warning, failure or unrecorded check
 * ("Not checked") is never counted as passed.
 */
export function checkTally(checks: readonly ExperimentCheck[]): {
  passed: number;
  total: number;
} {
  return {
    passed: checks.filter((check) => check.state === "pass").length,
    total: checks.length,
  };
}

const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

const countFormat = new Intl.NumberFormat("en-US");
const n = (value: number) => countFormat.format(value);

/** Sample coefficient of variation; null for fewer than two values or a zero mean. */
export function coefficientOfVariation(values: number[]): number | null {
  if (values.length < 2) return null;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  if (mean === 0) return null;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    (values.length - 1);
  return Math.sqrt(variance) / mean;
}

/** "replay-fix-mean-empty-r1-20261105T180930Z-e9c062" → "r1-20261105T180930Z-e9c062". */
export function shortExperimentId(id: string, taskGroup: string) {
  const prefix = `replay-${taskGroup}-`;
  return id.startsWith(prefix) ? id.slice(prefix.length) : id;
}

const ratioOf = (summary: ExperimentSummary, condition: string) => {
  const clean = summary.medians.clean;
  const value = summary.medians[condition];
  return clean && value !== undefined ? value / clean : null;
};

function conditionImpacts(
  replays: DatasetRun[],
  summaries: ExperimentSummary[],
): ConditionImpact[] {
  return conditionRatios(summaries).flatMap((entry) => {
    const id = INJECTED_CONDITIONS.find((c) => c === entry.condition);
    if (!id) return [];
    const ratios = summaries.flatMap((summary) => {
      const ratio = ratioOf(summary, id);
      return ratio === null ? [] : [ratio];
    });
    const runs = replays.filter((run) => run.replayCondition === id);
    return [
      {
        id,
        label: replayConditionName(id),
        shortLabel: conditionShortName(id),
        medianRatio: entry.ratio,
        minRatio: Math.min(...ratios),
        maxRatio: Math.max(...ratios),
        experiments: entry.experiments,
        replays: runs.length,
        failedReplays: runs.filter((run) => run.passed === false).length,
        steps: runs.reduce((sum, run) => sum + run.steps, 0),
        excessSteps: runs.reduce((sum, run) => sum + run.excessSteps, 0),
        telemetryPresent: runs.reduce(
          (sum, run) => sum + run.telemetryPresent,
          0,
        ),
        injectionMethods: [
          ...new Set(runs.map((run) => run.injectionMethod)),
        ].sort(),
      },
    ];
  });
}

/**
 * The replay order most experiments follow, with each position's replay time
 * over its experiment's clean median. Experiments with another order are
 * counted out of `matching`, not merged into it.
 */
export function replaySequence(
  replays: DatasetRun[],
  summaries: ExperimentSummary[],
  steps: ReplayStep[] | null,
): ReplaySequence | null {
  const byExperiment = new Map<string, DatasetRun[]>();
  for (const run of replays)
    if (
      run.experiment !== null &&
      run.replayIndex !== null &&
      run.replayCondition !== null
    )
      byExperiment.set(run.experiment, [
        ...(byExperiment.get(run.experiment) ?? []),
        run,
      ]);
  if (byExperiment.size === 0) return null;

  const keyOf = (runs: DatasetRun[]) =>
    [...runs]
      .sort((a, b) => a.replayIndex! - b.replayIndex!)
      .map((run) => `${run.replayIndex}:${run.replayCondition}`)
      .join(",");
  const patterns = new Map<string, string[]>();
  for (const [experiment, runs] of byExperiment) {
    const key = keyOf(runs);
    patterns.set(key, [...(patterns.get(key) ?? []), experiment]);
  }
  const [modalKey, members] = [...patterns].sort(
    (a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]),
  )[0];
  const memberSet = new Set(members);
  const cleanMedian = new Map(
    summaries.map((summary) => [summary.experiment, summary.medians.clean]),
  );

  const roles = new Map<number, Map<string, number>>();
  for (const step of steps ?? []) {
    if (!memberSet.has(step.experiment)) continue;
    const counts = roles.get(step.replayIndex) ?? new Map<string, number>();
    counts.set(step.replayRole, (counts.get(step.replayRole) ?? 0) + 1);
    roles.set(step.replayIndex, counts);
  }

  const slots = modalKey.split(",").map((part): SequenceSlot => {
    const separator = part.indexOf(":");
    const index = Number(part.slice(0, separator));
    const condition = part.slice(separator + 1);
    const ratios = members.flatMap((experiment) => {
      const run = byExperiment
        .get(experiment)!
        .find((candidate) => candidate.replayIndex === index);
      const clean = cleanMedian.get(experiment);
      return run && clean ? [run.durationSeconds / clean] : [];
    });
    const roleCounts = roles.get(index);
    const role = roleCounts
      ? [...roleCounts].sort(
          (a, b) => b[1] - a[1] || a[0].localeCompare(b[0]),
        )[0][0]
      : null;
    return {
      index,
      condition,
      role,
      experiments: members.length,
      medianRatio: ratios.length ? median(ratios) : null,
      minRatio: ratios.length ? Math.min(...ratios) : null,
      maxRatio: ratios.length ? Math.max(...ratios) : null,
    };
  });
  return { slots, matching: members.length };
}

function experimentRows(
  replays: DatasetRun[],
  summaries: ExperimentSummary[],
): TaskGroupRows[] {
  const rows = summaries.map((summary): ExperimentRow => {
    const runs = replays.filter((run) => run.experiment === summary.experiment);
    const clean = runs
      .filter((run) => run.replayCondition === "clean")
      .map((run) => run.durationSeconds);
    return {
      id: summary.experiment,
      shortId: shortExperimentId(summary.experiment, summary.taskGroup),
      taskGroup: summary.taskGroup,
      recordedAt: summary.recordedAt,
      replays: summary.replays,
      failedReplays: runs.filter((run) => run.passed === false).length,
      unrecordedReplays: runs.filter((run) => run.passed === null).length,
      cleanReplays: clean.length,
      cleanMedianSeconds: summary.medians.clean ?? null,
      cleanCv: coefficientOfVariation(clean),
      ratios: Object.fromEntries(
        INJECTED_CONDITIONS.map((condition) => [
          condition,
          ratioOf(summary, condition),
        ]),
      ) as Record<InjectedCondition, number | null>,
    };
  });
  const groups = new Map<string, ExperimentRow[]>();
  for (const row of rows)
    groups.set(row.taskGroup, [...(groups.get(row.taskGroup) ?? []), row]);
  return [...groups]
    .map(([taskGroup, experiments]) => ({ taskGroup, experiments }))
    .sort(
      (a, b) =>
        b.experiments.length - a.experiments.length ||
        a.taskGroup.localeCompare(b.taskGroup),
    );
}

const plural = (count: number, one: string, many = `${one}s`) =>
  `${n(count)} ${count === 1 ? one : many}`;

/** "protocol_checks_failed: failed or missing checks stable_baseline" → "protocol_checks_failed". */
const reasonCode = (reason: string) => reason.split(":")[0].trim();

function buildChecks(
  input: ExperimentsPageInput,
  replays: DatasetRun[],
  summaries: ExperimentSummary[],
): ExperimentCheck[] {
  const checks: ExperimentCheck[] = [];
  const { build, replaySteps, validation } = input;

  // Protocol checks rejected whole experiments at build time.
  if (build === null) {
    checks.push({
      id: "protocol",
      label: "Recorded protocol checks",
      state: "unknown",
      detail:
        "No readable build report names this release, so protocol check results are not shown.",
    });
  } else {
    const rejected = build.rejections.filter(
      (rejection) => reasonCode(rejection.reason) === "protocol_checks_failed",
    );
    const failedExperiments = new Set(rejected.map((r) => r.origin)).size;
    const names = [
      ...new Set(
        rejected.flatMap((r) => {
          const match = /checks\s+(.+)$/.exec(r.reason);
          return match ? match[1].split(/[\s,]+/).filter(Boolean) : [];
        }),
      ),
    ];
    const received =
      build.experiments === null
        ? ""
        : `${n(build.experiments - failedExperiments)} of ${n(build.experiments)} recorded experiments passed. `;
    checks.push({
      id: "protocol",
      label: `Recorded protocol checks${names.length ? ` (${names.join(", ").replaceAll("_", " ")})` : ""}`,
      state: failedExperiments > 0 ? "warn" : "pass",
      detail:
        failedExperiments > 0
          ? `${received}${plural(failedExperiments, "experiment")} failed and ${failedExperiments === 1 ? "is" : "are"} excluded from this release, recorded as build rejections (${build.buildId}).`
          : `${received}No experiment was rejected (${build.buildId}).`,
    });

    const stepRejections = build.rejections.filter(
      (rejection) => reasonCode(rejection.reason) !== "protocol_checks_failed",
    );
    const byReason = new Map<string, number>();
    for (const rejection of stepRejections) {
      const code = reasonCode(rejection.reason);
      byReason.set(code, (byReason.get(code) ?? 0) + 1);
    }
    checks.push({
      id: "step-labels",
      label: "Steps labelled without rejection",
      state: stepRejections.length > 0 ? "warn" : "pass",
      detail:
        stepRejections.length > 0
          ? `${plural(stepRejections.length, "step record")} left unlabelled (${[
              ...byReason,
            ]
              .map(
                ([code, count]) => `${n(count)} ${code.replaceAll("_", " ")}`,
              )
              .join(", ")}).`
          : "The build report records no step-level rejections.",
    });
  }

  // Clean controls: label outcome recorded per clean-control step.
  if (replaySteps === null) {
    checks.push({
      id: "clean-noise",
      label: "Clean controls within baseline noise",
      state: "unknown",
      detail: "Replay roles and label outcomes could not be read.",
    });
  } else {
    const controls = replaySteps.filter(
      (step) => step.replayRole === "clean_control",
    );
    const exceeded = controls.filter(
      (step) => step.labelBasis === "clean_exceeded_noise",
    ).length;
    checks.push({
      id: "clean-noise",
      label: "Clean controls within baseline noise",
      state: controls.length === 0 ? "unknown" : exceeded > 0 ? "warn" : "pass",
      detail:
        controls.length === 0
          ? "No clean-control steps are recorded."
          : exceeded > 0
            ? `${n(exceeded)} of ${n(controls.length)} clean-control steps exceeded the baseline noise band.`
            : `All ${n(controls.length)} clean-control steps stayed within the baseline noise band.`,
    });
  }

  const complete = summaries.filter(
    (summary) =>
      summary.medians.clean !== undefined &&
      INJECTED_CONDITIONS.every((c) => summary.medians[c] !== undefined),
  ).length;
  checks.push({
    id: "structure",
    label: "Clean and all four injected conditions replayed",
    state: complete === summaries.length ? "pass" : "warn",
    detail: `${n(complete)} of ${n(summaries.length)} experiments, from the release rows.`,
  });

  const failed = replays.filter((run) => run.passed === false).length;
  const unrecorded = replays.filter((run) => run.passed === null).length;
  checks.push({
    id: "task-outcome",
    label: "Replays passed the task check",
    state: failed > 0 ? "warn" : unrecorded > 0 ? "unknown" : "pass",
    detail: [
      `${n(replays.length - failed - unrecorded)} of ${n(replays.length)} replay runs passed.`,
      failed > 0 ? `${n(failed)} failed and stay in the medians.` : "",
      unrecorded > 0 ? `${n(unrecorded)} have no recorded outcome.` : "",
    ]
      .filter(Boolean)
      .join(" "),
  });

  const gate = validation.status.trim().toLowerCase();
  checks.push({
    id: "quality-gate",
    label: "Release quality gate",
    state: gate === "passed" ? "pass" : gate === "failed" ? "fail" : "unknown",
    detail: `Status "${validation.status}", gate ${validation.gateVersion}, validated ${validation.validatedAt.slice(0, 10)}.`,
  });

  const injected = replays.filter(
    (run) => run.replayCondition !== null && run.replayCondition !== "clean",
  );
  const withMethod = injected.filter(
    (run) => run.injectionMethod !== "none",
  ).length;
  checks.push({
    id: "injection",
    label: "Contention injection applied",
    state: "unknown",
    detail: `${n(withMethod)} of ${n(injected.length)} injected replays record an injection method; injector logs are not part of the release, so application is not verified here.`,
  });

  const steps = replays.reduce((sum, run) => sum + run.steps, 0);
  const present = replays.reduce((sum, run) => sum + run.telemetryPresent, 0);
  checks.push({
    id: "telemetry",
    label: "Host telemetry coverage",
    state: "unknown",
    detail: `No coverage threshold is recorded. ${n(present)} of ${n(steps)} replay steps have host samples overlapping the step.`,
  });

  return checks;
}

/** "Inference" → "inference" inside a sentence; acronyms such as "CPU" stay. */
const midSentence = (label: string) =>
  label === label.toUpperCase() ? label : label.toLowerCase();

/** One cautious sentence naming the conditions with the largest observed slowdown. */
export function conditionReading(conditions: ConditionImpact[]): string | null {
  const ranked = [...conditions].sort((a, b) => b.medianRatio - a.medianRatio);
  const top = ranked.slice(0, 2).filter((c) => c.medianRatio > 1);
  if (top.length === 0) return null;
  const ratio = (c: ConditionImpact) => `${c.medianRatio.toFixed(1)}×`;
  if (top.length === 1)
    return `${top[0].shortLabel} contention shows the largest observed slowdown: ${ratio(top[0])} the same experiment's clean replays.`;
  return `${top[0].shortLabel} and ${midSentence(top[1].shortLabel)} contention show the largest observed slowdowns: ${ratio(top[0])} and ${ratio(top[1])} the same experiment's clean replays.`;
}

/** Protocol roles in replay order; unknown roles follow. */
const ROLE_ORDER = ["baseline", "clean", "clean_control", "contended"];
const roleOrder = (role: string) => {
  const index = ROLE_ORDER.indexOf(role);
  return index === -1 ? ROLE_ORDER.length : index;
};

/** Null when the release has no replay experiments. */
export function buildExperimentsPage(
  input: ExperimentsPageInput,
): ExperimentsPage | null {
  const replays = input.runs.filter((run) => run.experiment !== null);
  const summaries = summarizeExperiments(replays);
  if (summaries.length === 0) return null;

  const conditions = conditionImpacts(replays, summaries);
  const largest =
    [...conditions].sort((a, b) => b.medianRatio - a.medianRatio)[0] ?? null;
  const clean = replays.filter((run) => run.replayCondition === "clean");
  const groups = experimentRows(replays, summaries);
  const cvs = groups.flatMap((group) =>
    group.experiments.flatMap((row) =>
      row.cleanCv === null ? [] : [row.cleanCv],
    ),
  );
  const times = replays
    .flatMap((run) => (run.recordedAt === null ? [] : [run.recordedAt]))
    .sort();
  const sequence = replaySequence(replays, summaries, input.replaySteps);

  const roleCounts = new Map<string, number>();
  for (const slot of sequence?.slots ?? []) {
    const role =
      slot.role ?? (slot.condition === "clean" ? "clean" : "contended");
    roleCounts.set(role, (roleCounts.get(role) ?? 0) + 1);
  }

  let labelOutcomes: ExperimentsPage["labelOutcomes"] = null;
  if (input.replaySteps !== null) {
    const counts = new Map<string, number>();
    for (const step of input.replaySteps)
      counts.set(step.labelBasis, (counts.get(step.labelBasis) ?? 0) + 1);
    labelOutcomes = [...counts]
      .map(([basis, steps]) => ({ basis, steps }))
      .sort((a, b) => b.steps - a.steps || a.basis.localeCompare(b.basis));
  }

  return {
    release: input.release,
    releaseName: input.release.split("+")[0],
    experiments: summaries.length,
    replayRuns: replays.length,
    replaySteps: replays.reduce((sum, run) => sum + run.steps, 0),
    models: [
      ...new Set(
        replays.flatMap((run) =>
          run.agentModel === null ? [] : [run.agentModel],
        ),
      ),
    ].sort(),
    recordedRange: times.length
      ? { first: times[0], last: times[times.length - 1] }
      : null,
    conditions,
    largest,
    clean: {
      replays: clean.length,
      failedReplays: clean.filter((run) => run.passed === false).length,
      steps: clean.reduce((sum, run) => sum + run.steps, 0),
      telemetryPresent: clean.reduce(
        (sum, run) => sum + run.telemetryPresent,
        0,
      ),
      medianCv: cvs.length ? median(cvs) : null,
    },
    sequence,
    roleCounts: [...roleCounts]
      .map(([role, replaysPerExperiment]) => ({
        role,
        replays: replaysPerExperiment,
      }))
      .sort((a, b) => roleOrder(a.role) - roleOrder(b.role)),
    labelOutcomes,
    checks: buildChecks(input, replays, summaries),
    reading: conditionReading(conditions),
    groups,
  };
}

/** Axis maximum and ticks for ratios, starting at 0 with at most seven ticks. */
export function ratioAxis(maxRatio: number): { max: number; ticks: number[] } {
  const top = Number.isFinite(maxRatio) && maxRatio > 1 ? maxRatio : 1;
  const step = [0.5, 1, 2, 5, 10, 20, 50].find((s) => top / s <= 6) ?? 100;
  const max = Math.ceil(top / step) * step;
  const ticks: number[] = [];
  for (let tick = 0; tick <= max + 1e-9; tick += step)
    ticks.push(Number(tick.toFixed(2)));
  return { max, ticks };
}
