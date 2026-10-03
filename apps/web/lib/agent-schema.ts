import { z } from "zod";

const seconds = z.number().finite().nonnegative();
const spanSchema = z
  .object({
    name: z.string(),
    start_time: z.iso.datetime({ offset: true }),
    end_time: z.iso.datetime({ offset: true }),
    context: z.object({ trace_id: z.string(), span_id: z.string() }),
  })
  .refine((s) => Date.parse(s.end_time) >= Date.parse(s.start_time), {
    message: "Agent span ends before it starts",
  });
export const agentSchema = z.object({
  model: z.string(),
  status: z.enum(["running", "passed", "failed", "inconclusive"]),
  run_id: z.string().optional(),
  final_answer: z.string().optional(),
  checks: z.record(z.string(), z.boolean()).optional(),
  limitations: z.array(z.string()).default([]),
  messages: z.array(
    z.object({
      role: z.enum(["system", "user", "assistant", "tool"]),
      content: z.string(),
      tool_name: z.string().optional(),
      tool_calls: z
        .array(
          z.object({
            function: z.object({
              name: z.string(),
              arguments: z.record(z.string(), z.unknown()),
            }),
          }),
        )
        .optional(),
    }),
  ),
  inference: z.array(
    z.object({
      request_index: z.number().int().nonnegative(),
      wall_seconds: seconds,
      metrics: z.object({
        total_duration: seconds.optional(),
        load_duration: seconds.optional(),
        prompt_eval_count: z.number().int().nonnegative().optional(),
        prompt_eval_duration: seconds.optional(),
        eval_count: z.number().int().nonnegative().optional(),
        eval_duration: seconds.optional(),
        done_reason: z.string().optional(),
      }),
    }),
  ),
  spans: z.array(spanSchema).optional(),
});
export type AgentEvidence = z.infer<typeof agentSchema>;
