// Synthetic live read-API responses for unit tests only; never recorded evidence.

export const RUN_ID = "5f0c7a52-3c1e-4d0b-9a57-2f8e6b1d4c90";
export const OTHER_RUN_ID = "a8e1d6f4-0b2c-4f7e-8d39-6c5b4a3e2f10";
export const TASK_ID = "synthetic-task-r1";
export const CLOCK_ID = "synthetic-clock";

export const at = (second: number) =>
  new Date(Date.UTC(2026, 9, 24, 12, 0, 0) + second * 1000).toISOString();

/** A run UUID derived from a number, for building long histories. */
export const runIdFor = (n: number) =>
  `00000000-0000-4000-8000-${n.toString(16).padStart(12, "0")}`;

const registration = (runId = RUN_ID) => ({
  contract_version: "live-contracts-v1",
  request_id: `request-${runId}`,
  task_id: TASK_ID,
  configuration_digest: "0".repeat(64),
  clock_ids: [CLOCK_ID],
  resources: [
    {
      resource_id: "tool-host-a",
      service_id: "tool-service-a",
      resource_class: "cpu_sandbox",
    },
  ],
});

export function runSummary(runId = RUN_ID, sequence = 1, state = "running") {
  return {
    run_id: runId,
    registration: registration(runId),
    state,
    revision: state === "queued" ? 0 : 1,
    cleanup_requested: false,
    created_at: at(0),
    updated_at: at(1),
    sequence,
  };
}

type StepOptions = {
  runId?: string;
  clock?: string;
  parent?: string | null;
  type?: "model_call" | "tool" | "retrieval" | "other";
  submitted?: number;
  started?: number | null;
  finished?: number | null;
  cpu?: number | null;
  reason?: string | null;
  second?: number;
};

const NS_PER_SECOND = 1e9;

export function stepEvent(
  stepId: string,
  sequence: number,
  state: string,
  options: StepOptions = {},
) {
  const seconds = (value: number | null | undefined) =>
    value === null || value === undefined
      ? null
      : Math.round(value * NS_PER_SECOND);
  return {
    contract_version: "live-contracts-v1",
    run_id: options.runId ?? RUN_ID,
    task_id: TASK_ID,
    observed_at: at(options.second ?? sequence),
    clock_id: options.clock ?? CLOCK_ID,
    record_type: "step_event",
    event_id: `${stepId}-${sequence}`,
    step_id: stepId,
    step_type: options.type ?? "tool",
    parent_step_id: options.parent ?? null,
    trace_id: "trace-1",
    span_id: `span-${stepId}`,
    resource_id: null,
    service_id: null,
    sequence,
    state,
    submitted_tick_ns: seconds(options.submitted ?? 0)!,
    started_tick_ns: seconds(options.started),
    finished_tick_ns: seconds(options.finished),
    cpu_seconds: options.cpu ?? null,
    cpu_scope:
      options.cpu === undefined || options.cpu === null ? null : "process",
    reason: options.reason ?? null,
  };
}

type Event = ReturnType<typeof stepEvent>;

/** A step view with the service's derived timings (agentwatch_sdk.live.step_timings). */
export function stepView(event: Event) {
  const { submitted_tick_ns: submit, started_tick_ns: start } = event;
  const finish = event.finished_tick_ns;
  const duration = finish === null ? null : (finish - submit) / NS_PER_SECOND;
  const queue = start === null ? null : (start - submit) / NS_PER_SECOND;
  return {
    event,
    timings: {
      queue_seconds: queue,
      execution_seconds:
        start === null || finish === null
          ? null
          : (finish - start) / NS_PER_SECOND,
      duration_seconds: duration,
      cpu_seconds: event.cpu_seconds,
      queue_fraction: queue !== null && duration ? queue / duration : null,
    },
  };
}

export function resourceSample(
  sampleId: string,
  options: {
    resourceId?: string;
    serviceId?: string;
    metric?: string;
    unit?: string;
    value?: number | null;
    availability?: string;
    second?: number;
  } = {},
) {
  const availability = options.availability ?? "measured";
  return {
    contract_version: "live-contracts-v1",
    run_id: RUN_ID,
    task_id: TASK_ID,
    observed_at: at(options.second ?? 0),
    clock_id: CLOCK_ID,
    record_type: "resource_sample",
    sample_id: sampleId,
    resource_id: options.resourceId ?? "tool-host-a",
    service_id: options.serviceId ?? "tool-service-a",
    resource_class: "cpu_sandbox",
    tick_ns: (options.second ?? 0) * NS_PER_SECOND,
    metric: options.metric ?? "cpu.psi.some.avg10.v1",
    unit: options.unit ?? "percent",
    value: options.value === undefined ? 1.5 : options.value,
    availability,
    reason: availability === "measured" ? null : `${availability} on this host`,
  };
}

export function savedRecord(
  payload: { record_type: string },
  sequence: number,
  disposition = "applied",
) {
  return {
    record_id: `record-${sequence}`,
    disposition,
    sequence,
    payload,
    received_at: at(sequence),
  };
}

export function runSnapshot(
  overrides: Record<string, unknown> = {},
  runId = RUN_ID,
) {
  return {
    ...runSummary(runId),
    request_id: `request-${runId}`,
    reason: null,
    cleanup_confirmed: null,
    unfinished_steps: [],
    steps: [],
    changes: [],
    gaps: [],
    records: [],
    next_cursor: null,
    ...overrides,
  };
}

/** A fetch replacement that serves fixed responses by URL path and records requests. */
export function fakeFetch(respond: (url: URL) => Response | Promise<Response>) {
  const requested: URL[] = [];
  const fetchImpl: typeof fetch = async (input) => {
    const url = new URL(String(input));
    requested.push(url);
    return respond(url);
  };
  return { fetchImpl, requested };
}
