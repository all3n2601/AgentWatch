import { z } from "zod";
import { agentSchema } from "./agent-schema";

const telemetry = z.object({
  sample_count: z.number().nonnegative(),
  valid_pressure_samples: z.number().nonnegative(),
  max_cpu_pressure_avg10_percent: z.number().nullable(),
  status: z.string(),
});
const step = z
  .object({
    step_id: z.string(),
    submitted_ns: z.number(),
    started_ns: z.number(),
    finished_ns: z.number(),
    duration_seconds: z.number().positive(),
    queue_seconds: z.number().nonnegative(),
    execution_seconds: z.number().positive(),
    cpu_seconds: z.number().nonnegative(),
    queue_fraction: z.number().min(0).max(1),
    resource: z.enum(["none", "cpu_sandbox"]),
    checksum: z.string(),
    evidence: z.string(),
    rule_version: z.string(),
    threshold_seconds: z.number(),
    telemetry: telemetry.optional(),
  })
  .refine(
    (s) => s.submitted_ns <= s.started_ns && s.started_ns < s.finished_ns,
    { message: "Invalid timing order" },
  )
  .refine(
    (s) =>
      Math.abs(s.duration_seconds - s.queue_seconds - s.execution_seconds) <
      0.00001,
    { message: "Timing breakdown does not match duration" },
  );
export const experimentSchema = z.object({
  run_id: z.string(),
  experiment: z.object({
    worker_capacity: z.number(),
    iterations: z.number(),
    baseline_repeats: z.number(),
    workload: z.string(),
    tests_per_step: z.number().nullable(),
  }),
  baseline: z.object({
    execution_cv: z.number(),
    queue_threshold_seconds: z.number(),
    steps: z.array(step).min(3),
  }),
  clean: step,
  contended: step,
  checks: z.object({ passed: z.boolean() }).catchall(z.boolean()),
  limitations: z.array(z.string()),
  agent: agentSchema.optional().catch(undefined),
});
export type Step = z.infer<typeof step>;
export type Experiment = z.infer<typeof experimentSchema> & {
  kind: "coding" | "cpu";
  persisted?: boolean;
  stored_at?: string;
};
export type DashboardData = {
  experiments: Experiment[];
  warnings: string[];
  source?: "database" | "local";
  nextCursor?: number | null;
};
