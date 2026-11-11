import { z } from "zod";
import { routes } from "./routes";
import {
  conditionRatios,
  summarizeExperiments,
  type DatasetRun,
} from "./dataset-runs";
import { displayName, replayConditionName } from "./dataset-view";
import { formatQuantity } from "./measurement-format";
import { buildModelOverview } from "./model-overview";
import type { Ready } from "./overview-model";
import {
  diagnoseView,
  type DiagnosisStep,
  type RunView,
  type UnifiedRun,
} from "./unified-runs";

/**
 * Read-only tools the investigation assistant may call. Arguments come from a
 * model, so every call is checked against an allowlist and a schema; results
 * carry only timing, outcome and label fields (never prompts, code or tool
 * arguments) and are bounded in size. Missing readings stay null, never zero.
 */

export type Loaded<T> = { ok: true; value: T } | { ok: false; message: string };

/** Where the tools read from. Injected so tests run without a database. */
export interface AssistantSources {
  dataset(): Promise<Loaded<Ready>>;
  importMap(): Promise<Record<string, string>>;
  /** Live runs from the ingestion database, newest first (one page). */
  liveRuns(): Promise<Loaded<UnifiedRun[]>>;
  /** Saved experiment runs from the ingestion database. */
  savedRuns(): Promise<Loaded<UnifiedRun[]>>;
  /** One stored run (live or saved) with its steps, or null when neither store has it. */
  runView(runId: string): Promise<RunView | null>;
}

/** A run a tool returned, so the chat can link the runs an answer cites. */
export interface RunRecord {
  runId: string;
  shortId: string;
  source: "dataset" | "live" | "saved";
  title: string;
  status: string;
  /** Where the run page lives: the run itself if stored, else its imported live run. */
  href: string | null;
}

export interface ToolOutput {
  /** JSON text handed back to the model. */
  content: string;
  records: RunRecord[];
}

const SHORT_ID = 8;
const MAX_RESULT_CHARS = 12_000;
const MAX_STEPS = 12;
const CONDITIONS = [
  "clean",
  "baseline",
  "cpu",
  "inference",
  "retrieval",
  "mixed",
] as const;

const short = (id: string) => id.slice(0, SHORT_ID);
const seconds = (value: number | null) =>
  value === null || !Number.isFinite(value) ? null : Number(value.toFixed(3));
const share = (part: number, whole: number) =>
  whole === 0 ? null : formatQuantity((part / whole) * 100, "percent");
const ratio = (value: number | null) =>
  value === null ? null : formatQuantity(value, "ratio");

const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

/** Each replay's total time over the median of its own experiment's clean replays. */
function slowdowns(runs: DatasetRun[]) {
  const clean = new Map<string, number[]>();
  for (const run of runs)
    if (run.experiment && run.replayCondition === "clean")
      clean.set(run.experiment, [
        ...(clean.get(run.experiment) ?? []),
        run.durationSeconds,
      ]);
  const medians = new Map(
    [...clean].map(([experiment, times]) => [experiment, median(times)]),
  );
  return (run: DatasetRun) => {
    const base = run.experiment ? medians.get(run.experiment) : undefined;
    return base ? run.durationSeconds / base : null;
  };
}

function datasetRecord(
  run: DatasetRun,
  importMap: Record<string, string>,
): RunRecord {
  const live = importMap[run.runId] ?? null;
  return {
    runId: run.runId,
    shortId: short(run.runId),
    source: "dataset",
    title: run.taskGroup,
    status:
      run.passed === null
        ? "outcome not recorded"
        : run.passed
          ? "passed"
          : "failed",
    href: live ? routes.run(live) : null,
  };
}

const storedRecord = (run: UnifiedRun): RunRecord => ({
  runId: run.id,
  shortId: short(run.id),
  source: run.source,
  title: run.title,
  status: run.state,
  href: routes.run(run.id),
});

function bounded(value: unknown): string {
  const text = JSON.stringify(value);
  return text.length <= MAX_RESULT_CHARS
    ? text
    : JSON.stringify({
        truncated: true,
        note: "Result too large; ask a narrower question.",
      });
}

// ------------------------------------------------------------------ schemas

const modelArg = z.string().trim().min(1).max(80).optional();
const taskArg = z.string().trim().min(1).max(80).optional();

const schemas = {
  get_overview: z.object({ model: modelArg }).strict(),
  search_runs: z
    .object({
      source: z.enum(["dataset", "live", "saved", "all"]).default("all"),
      outcome: z.enum(["passed", "failed", "any"]).default("any"),
      condition: z.enum(CONDITIONS).optional(),
      task: taskArg,
      model: modelArg,
      sort: z.enum(["newest", "slowest", "slowdown"]).default("newest"),
      limit: z.number().int().min(1).max(20).default(10),
    })
    .strict(),
  get_run: z
    .object({
      run_id: z
        .string()
        .trim()
        .toLowerCase()
        .regex(
          /^[0-9a-f-]{8,36}$/,
          "Use a run UUID or its first 8 characters.",
        ),
    })
    .strict(),
  compare_conditions: z.object({ task: taskArg, model: modelArg }).strict(),
} as const;

export type ToolName = keyof typeof schemas;
export const TOOL_NAMES = Object.keys(schemas) as ToolName[];

/** OpenAI-compatible tool definitions sent with each chat request. */
export const TOOL_DEFINITIONS = [
  {
    type: "function",
    function: {
      name: "get_overview",
      description:
        "Population summary for one agent model in the dataset release: run and step counts, outcomes, host-telemetry coverage, call types, excess-wait labels per resource, per-stage excess-wait shares and the largest observed slowdown.",
      parameters: {
        type: "object",
        properties: {
          model: {
            type: "string",
            description:
              "Agent model id, e.g. qwen3:1.7b. Omit for the default.",
          },
        },
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "search_runs",
      description:
        "Find runs. Dataset runs carry outcome, replay condition, duration, excess-wait steps and slowdown against the same experiment's clean replays; live and saved runs come from the ingestion database.",
      parameters: {
        type: "object",
        properties: {
          source: { type: "string", enum: ["dataset", "live", "saved", "all"] },
          outcome: { type: "string", enum: ["passed", "failed", "any"] },
          condition: { type: "string", enum: [...CONDITIONS] },
          task: {
            type: "string",
            description: "Task group name, exact or partial.",
          },
          model: { type: "string" },
          sort: { type: "string", enum: ["newest", "slowest", "slowdown"] },
          limit: { type: "integer", minimum: 1, maximum: 20 },
        },
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_run",
      description:
        "Details of one run by UUID or its first 8 characters: timing totals, labels and call types for dataset runs, and for stored runs the top steps by queue time and by duration.",
      parameters: {
        type: "object",
        properties: { run_id: { type: "string" } },
        required: ["run_id"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "compare_conditions",
      description:
        "Median slowdown of each injected contention condition against the same experiments' clean replays, optionally for one task group.",
      parameters: {
        type: "object",
        properties: { task: { type: "string" }, model: { type: "string" } },
        additionalProperties: false,
      },
    },
  },
] as const;

// ------------------------------------------------------------------ tools

const matches = (value: string, needle: string | undefined) =>
  needle === undefined || value.toLowerCase().includes(needle.toLowerCase());

async function getOverview(
  args: z.infer<(typeof schemas)["get_overview"]>,
  sources: AssistantSources,
): Promise<ToolOutput> {
  const dataset = await sources.dataset();
  if (!dataset.ok)
    return { content: bounded({ error: dataset.message }), records: [] };
  const overview = buildModelOverview(
    dataset.value,
    await sources.importMap(),
    args.model,
  );
  if (!overview)
    return {
      content: bounded({
        error: "No run in this release records an agent model.",
      }),
      records: [],
    };
  const judged = overview.outcomes.passed + overview.outcomes.failed;
  return {
    content: bounded({
      release: overview.releaseName,
      model: overview.model.id,
      other_models: overview.models
        .filter((m) => m.id !== overview.model.id)
        .map((m) => m.id),
      runs: overview.runs,
      measured_steps: overview.steps,
      outcomes: {
        ...overview.outcomes,
        failed_share: share(overview.outcomes.failed, judged),
      },
      host_telemetry_steps: overview.telemetryPresent,
      host_telemetry_share: share(overview.telemetryPresent, overview.steps),
      call_types: Object.fromEntries(
        overview.callTypes.map((c) => [c.id, c.value]),
      ),
      excess_wait_labels: Object.fromEntries(
        overview.resourceLabels.map((l) => [l.id, l.value]),
      ),
      stages: Object.fromEntries(
        Object.entries(overview.stages).map(([stage, s]) => [
          stage,
          {
            calls: s.calls,
            excess_wait_steps: s.excessSteps,
            excess_wait_share: share(s.excessSteps, s.calls),
          },
        ]),
      ),
      largest_observed_slowdown: overview.largestSlowdown && {
        condition: overview.largestSlowdown.condition,
        median_ratio_vs_clean: ratio(overview.largestSlowdown.ratio),
        experiments: overview.largestSlowdown.experiments,
      },
      runs_without_model_excluded: overview.runsWithoutModel,
      note: "Slowdowns are observed under injected contention, not causal. Missing host telemetry is not zero load.",
    }),
    records: [],
  };
}

async function searchRuns(
  args: z.infer<(typeof schemas)["search_runs"]>,
  sources: AssistantSources,
): Promise<ToolOutput> {
  const rows: Record<string, unknown>[] = [];
  const records: RunRecord[] = [];
  const notes: string[] = [];
  let total = 0;

  if (args.source === "all" || args.source === "dataset") {
    const dataset = await sources.dataset();
    if (!dataset.ok) notes.push(`Dataset: ${dataset.message}`);
    else {
      const importMap = await sources.importMap();
      const slowdownOf = slowdowns(dataset.value.runs);
      const found = dataset.value.runs.filter(
        (run) =>
          (args.outcome === "any" ||
            run.passed === (args.outcome === "passed")) &&
          (args.condition === undefined ||
            run.replayCondition === args.condition) &&
          matches(run.taskGroup, args.task) &&
          (args.model === undefined ||
            (run.agentModel !== null && matches(run.agentModel, args.model))),
      );
      total += found.length;
      const sorted =
        args.sort === "slowest"
          ? [...found].sort((a, b) => b.durationSeconds - a.durationSeconds)
          : args.sort === "slowdown"
            ? [...found].sort(
                (a, b) => (slowdownOf(b) ?? -1) - (slowdownOf(a) ?? -1),
              )
            : found; // the snapshot is already newest first
      for (const run of sorted.slice(0, args.limit)) {
        const record = datasetRecord(run, importMap);
        records.push(record);
        rows.push({
          short_id: record.shortId,
          source: "dataset",
          task: run.taskGroup,
          model: run.agentModel,
          condition: run.replayCondition,
          outcome: record.status,
          recorded_at: run.recordedAt,
          duration_s: seconds(run.durationSeconds),
          queue_s: seconds(run.queueSeconds),
          steps: run.steps,
          excess_wait_steps: run.excessSteps,
          slowdown_vs_clean: ratio(slowdownOf(run)),
          has_run_page: record.href !== null,
        });
      }
    }
  }

  const stored = async (
    label: string,
    load: () => Promise<Loaded<UnifiedRun[]>>,
  ) => {
    const result = await load();
    if (!result.ok) return notes.push(`${label}: ${result.message}`);
    const found = result.value.filter(
      (run) =>
        (args.outcome === "any" ||
          (args.outcome === "failed"
            ? run.state === "failed"
            : run.state === "passed" || run.state === "completed")) &&
        matches(run.title, args.task),
    );
    total += found.length;
    for (const run of found.slice(0, args.limit)) {
      const record = storedRecord(run);
      records.push(record);
      rows.push({
        short_id: record.shortId,
        source: run.source,
        title: run.title,
        state: run.state,
        at: run.at,
        steps: run.steps,
        queue_share:
          run.waitingShare === null
            ? null
            : formatQuantity(run.waitingShare * 100, "percent"),
      });
    }
  };
  // Condition and model filters only exist for dataset runs.
  const datasetOnly = args.condition !== undefined || args.model !== undefined;
  if (!datasetOnly && (args.source === "all" || args.source === "live"))
    await stored("Live runs", sources.liveRuns);
  if (!datasetOnly && (args.source === "all" || args.source === "saved"))
    await stored("Saved runs", sources.savedRuns);

  return {
    content: bounded({
      total_matches: total,
      returned: rows.length,
      runs: rows,
      notes,
    }),
    records,
  };
}

const stepRow = (step: DiagnosisStep) => ({
  step: step.stepId,
  type: step.type,
  state: step.state,
  duration_s: seconds(step.durationSeconds),
  queue_s: seconds(step.queueSeconds),
  execution_s: seconds(step.executionSeconds),
  waiting_share:
    step.waitingShare === null
      ? null
      : formatQuantity(step.waitingShare * 100, "percent"),
  resource: step.resource,
  resource_basis: step.basis,
});

async function getRun(
  args: z.infer<(typeof schemas)["get_run"]>,
  sources: AssistantSources,
): Promise<ToolOutput> {
  const id = args.run_id;
  const dataset = await sources.dataset();
  const importMap = await sources.importMap();
  const datasetMatches = dataset.ok
    ? dataset.value.runs.filter((run) => run.runId.startsWith(id))
    : [];
  if (datasetMatches.length > 1)
    return {
      content: bounded({
        error: "That prefix matches several runs.",
        matches: datasetMatches.slice(0, 10).map((r) => r.runId),
      }),
      records: [],
    };

  const records: RunRecord[] = [];
  const result: Record<string, unknown> = {};
  const run = datasetMatches[0];
  if (run && dataset.ok) {
    const record = datasetRecord(run, importMap);
    records.push(record);
    result.dataset = {
      short_id: record.shortId,
      run_id: run.runId,
      task: run.taskGroup,
      model: run.agentModel,
      experiment: run.experiment,
      condition:
        run.replayCondition && replayConditionName(run.replayCondition),
      injection_method: displayName(run.injectionMethod),
      outcome: record.status,
      recorded_at: run.recordedAt,
      steps: run.steps,
      duration_s: seconds(run.durationSeconds),
      queue_s: seconds(run.queueSeconds),
      execution_s: seconds(run.executionSeconds),
      slowdown_vs_clean: ratio(slowdowns(dataset.value.runs)(run)),
      excess_wait_steps: run.excessSteps,
      labels: run.labels,
      call_types: run.stepTypes,
      excess_wait_by_call_type: run.excessByStepType,
      host_telemetry_steps: run.telemetryPresent,
    };
  }

  // A full UUID may also be stored; an imported dataset run is read through its live twin.
  const storedId = run
    ? (importMap[run.runId] ?? (id.length === 36 ? id : null))
    : id.length === 36
      ? id
      : null;
  if (storedId) {
    const view = await sources.runView(storedId);
    if (view) {
      const steps = diagnoseView(view);
      const stored = view.source === "live" ? view.snapshot : view.run;
      const storedRunId =
        view.source === "live" ? view.snapshot.run_id : view.run.run_id;
      records.push({
        runId: storedRunId,
        shortId: short(storedRunId),
        source: view.source,
        title: run?.taskGroup ?? storedRunId,
        status: "state" in stored ? String(stored.state) : "recorded",
        href: routes.run(storedRunId),
      });
      const imported =
        run !== undefined && importMap[run.runId] === storedRunId;
      result.stored = {
        source: view.source,
        short_id: short(storedRunId),
        // State the link explicitly, or the model cannot tell these are one run.
        relation: imported
          ? `Live import of dataset run ${short(run.runId)}: the same recorded run, replayed into the ingestion database; its steps describe that run.`
          : "Stored run requested directly.",
        steps: steps.length,
        steps_with_measured_queue: steps.filter((s) => s.queueSeconds !== null)
          .length,
        // Two orderings, named, so the model never re-sorts or mislabels them.
        steps_by_queue_desc: steps.slice(0, MAX_STEPS / 2).map(stepRow),
        steps_by_duration_desc: [...steps]
          .filter((step) => step.durationSeconds !== null)
          .sort((a, b) => (b.durationSeconds ?? 0) - (a.durationSeconds ?? 0))
          .slice(0, MAX_STEPS / 2)
          .map(stepRow),
      };
    }
  }

  if (!result.dataset && !result.stored)
    return {
      content: bounded({ error: `No run matches ${id}.` }),
      records: [],
    };
  return { content: bounded(result), records };
}

async function compareConditions(
  args: z.infer<(typeof schemas)["compare_conditions"]>,
  sources: AssistantSources,
): Promise<ToolOutput> {
  const dataset = await sources.dataset();
  if (!dataset.ok)
    return { content: bounded({ error: dataset.message }), records: [] };
  const runs = dataset.value.runs.filter(
    (run) =>
      matches(run.taskGroup, args.task) &&
      (args.model === undefined ||
        (run.agentModel !== null && matches(run.agentModel, args.model))),
  );
  const experiments = summarizeExperiments(runs);
  const ratios = conditionRatios(experiments);
  return {
    content: bounded({
      task: args.task ?? "all tasks",
      experiments: experiments.length,
      conditions: ratios.map((entry) => ({
        condition: entry.condition,
        median_ratio_vs_clean: ratio(entry.ratio),
        experiments: entry.experiments,
      })),
      note: "Each ratio compares an experiment's injected replays with its own clean replays. Observed, not causal.",
    }),
    records: [],
  };
}

const RUNNERS: {
  [K in ToolName]: (
    args: z.infer<(typeof schemas)[K]>,
    sources: AssistantSources,
  ) => Promise<ToolOutput>;
} = {
  get_overview: getOverview,
  search_runs: searchRuns,
  get_run: getRun,
  compare_conditions: compareConditions,
};

/**
 * Run one model-requested tool call. Unknown tools and invalid arguments are
 * answered with an error result the model can read, never executed.
 */
export async function runTool(
  name: string,
  rawArguments: string,
  sources: AssistantSources,
): Promise<ToolOutput> {
  if (!(TOOL_NAMES as string[]).includes(name))
    return {
      content: bounded({ error: `Unknown tool ${name}.` }),
      records: [],
    };
  let parsedJson: unknown;
  try {
    parsedJson = rawArguments.trim() === "" ? {} : JSON.parse(rawArguments);
  } catch {
    return {
      content: bounded({ error: "Tool arguments were not valid JSON." }),
      records: [],
    };
  }
  const tool = name as ToolName;
  const parsed = schemas[tool].safeParse(parsedJson);
  if (!parsed.success)
    return {
      content: bounded({
        error: `Invalid arguments: ${parsed.error.issues.map((i) => i.message).join("; ")}`,
      }),
      records: [],
    };
  try {
    // The schema for `tool` produced `parsed.data`, so it matches that runner.
    return await (
      RUNNERS[tool] as (
        args: unknown,
        s: AssistantSources,
      ) => Promise<ToolOutput>
    )(parsed.data, sources);
  } catch {
    return {
      content: bounded({ error: "The data for this tool could not be read." }),
      records: [],
    };
  }
}
