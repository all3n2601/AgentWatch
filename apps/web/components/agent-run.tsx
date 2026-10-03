"use client";

import {
  Bot,
  CheckCheck,
  Clock3,
  Cpu,
  MessageSquare,
  Terminal,
  ArrowRight,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { Experiment } from "@/lib/experiment-schema";
import type { AgentEvidence } from "@/lib/agent-schema";

const duration = (n: number) =>
  n < 1 ? `${(n * 1000).toFixed(0)} ms` : `${n.toFixed(2)} s`;
const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);

function AgentTimeline({
  agent,
  onInspectTool,
}: {
  agent: AgentEvidence;
  onInspectTool: () => void;
}) {
  const spans = agent.spans;
  if (!spans?.length)
    return (
      <Card className="p-6 text-sm text-muted-foreground">
        This earlier run has no saved agent span timestamps. Its measured
        request durations are shown below.
      </Card>
    );
  const sorted = [...spans].sort((a, b) =>
    a.name === "agent.tool_loop"
      ? -1
      : b.name === "agent.tool_loop"
        ? 1
        : Date.parse(a.start_time) - Date.parse(b.start_time),
  );
  const start = Math.min(...sorted.map((s) => Date.parse(s.start_time)));
  const end = Math.max(...sorted.map((s) => Date.parse(s.end_time)));
  const total = Math.max(end - start, 1);
  let modelIndex = 0;
  return (
    <Card className="overflow-hidden p-5 sm:p-6">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold">Agent timeline</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Recorded span timestamps · {duration(total / 1000)} end to end
          </p>
        </div>
        <Button size="sm" variant="outline" onClick={onInspectTool}>
          Inspect tool steps <ArrowRight className="size-3.5" />
        </Button>
      </div>
      <div className="space-y-3">
        {sorted.map((span) => {
          const model = span.name === "llm.chat";
          const tool = span.name === "tool.run_tests";
          const label = model
            ? `Model request ${++modelIndex}`
            : tool
              ? "run_tests"
              : "Complete agent loop";
          const left = ((Date.parse(span.start_time) - start) / total) * 100;
          const width =
            ((Date.parse(span.end_time) - Date.parse(span.start_time)) /
              total) *
            100;
          return (
            <div
              key={span.context.span_id}
              className="grid grid-cols-[110px_1fr_65px] items-center gap-3 text-xs sm:grid-cols-[160px_1fr_75px]"
            >
              <span className="truncate text-muted-foreground">{label}</span>
              <div
                className="trace-grid relative h-9 rounded bg-muted/20"
                aria-label={`${label}, starts ${duration((Date.parse(span.start_time) - start) / 1000)} into run`}
              >
                <div
                  className={`absolute top-2 h-5 min-w-px rounded ${model ? "bg-violet-400/80" : tool ? "bg-lime-300/80" : "bg-slate-400/35"}`}
                  style={{ left: `${left}%`, width: `${width}%` }}
                />
              </div>
              <span className="text-right font-mono">
                {duration(
                  (Date.parse(span.end_time) - Date.parse(span.start_time)) /
                    1000,
                )}
              </span>
            </div>
          );
        })}
      </div>
      <div className="mt-4 flex justify-between pl-[122px] text-[10px] text-muted-foreground sm:pl-[172px]">
        <span>0 s</span>
        <span>{duration(total / 1000)}</span>
      </div>
      <p className="mt-4 text-xs text-muted-foreground">
        The complete loop overlaps its child spans. Tool time includes warm-up,
        baselines, and the controlled busy-worker test.
      </p>
    </Card>
  );
}

export function AgentRun({
  experiment,
  onInspectTool,
}: {
  experiment: Experiment;
  onInspectTool: () => void;
}) {
  const agent = experiment.agent;
  if (!agent)
    return (
      <Card className="items-center gap-3 p-12 text-center">
        <Bot className="size-8 text-muted-foreground" />
        <h2 className="text-lg font-semibold">
          No agent evidence for this run
        </h2>
        <p className="max-w-md text-sm text-muted-foreground">
          Select a recorded Qwen3 run to explore its model requests and
          conversation. This run has no valid saved agent evidence.
        </p>
        <code className="rounded border bg-muted/30 px-3 py-2 text-xs">
          make record-llm
        </code>
      </Card>
    );
  const totalWall = sum(agent.inference.map((call) => call.wall_seconds));
  const counts = agent.inference.map((call) => call.metrics.eval_count);
  const tokens = counts.every((count) => count !== undefined)
    ? sum(counts as number[])
    : undefined;
  const maxWall = Math.max(
    ...agent.inference.map((call) => call.wall_seconds),
    0.001,
  );
  return (
    <div className="space-y-5">
      <Card className="flex flex-row flex-wrap items-center justify-between gap-4 border-violet-400/20 bg-violet-400/5 p-5 sm:p-6">
        <div className="flex items-center gap-3">
          <div className="rounded-xl border border-violet-400/20 bg-violet-400/10 p-3">
            <Bot className="size-6 text-violet-300" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">{agent.model}</h2>
            <p className="text-xs text-muted-foreground">
              Real model requests. Real tool results.
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          <Badge variant="outline" className="gap-1.5">
            <Cpu className="size-3" />
            CPU inference
          </Badge>
          <Badge
            variant="outline"
            className={
              agent.status === "passed"
                ? "border-lime-300/25 text-lime-200"
                : "text-amber-200"
            }
          >
            {agent.status}
          </Badge>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-4 min-[1600px]:grid-cols-4">
        {[
          {
            label: "Model requests",
            value: agent.inference.length,
            detail: "Including any correction attempts",
            icon: Bot,
          },
          {
            label: "Inference wall time",
            value: duration(totalWall),
            detail: "Sum of measured request durations",
            icon: Clock3,
          },
          {
            label: "Generated tokens",
            value: tokens ?? "Unavailable",
            detail: "Reported by the model server",
            icon: MessageSquare,
          },
          {
            label: "Tool feedback",
            value: agent.checks?.tool_result_returned
              ? "Verified"
              : "Unverified",
            detail: "Tool result returned to the model",
            icon: CheckCheck,
          },
        ].map((item) => (
          <Card key={item.label} className="gap-0 p-5">
            <div className="mb-3 flex items-center justify-between text-xs text-muted-foreground">
              {item.label}
              <item.icon className="size-4" />
            </div>
            <div className="text-2xl font-semibold tracking-tight">
              {item.value}
            </div>
            <p className="mt-2 text-[11px] text-muted-foreground">
              {item.detail}
            </p>
          </Card>
        ))}
      </div>
      <AgentTimeline agent={agent} onInspectTool={onInspectTool} />
      <div className="grid gap-5 xl:grid-cols-[1.1fr_1fr]">
        <Card className="gap-0 p-5 sm:p-6">
          <h2 className="text-sm font-semibold">Inference requests</h2>
          <p className="mb-6 mt-1 text-xs text-muted-foreground">
            Elapsed request time and server measurements
          </p>
          <div className="space-y-6">
            {agent.inference.map((call) => (
              <div key={call.request_index}>
                <div className="mb-2 flex justify-between text-xs">
                  <span>Request {call.request_index + 1}</span>
                  <span className="font-mono text-violet-200">
                    {duration(call.wall_seconds)}
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-violet-400"
                    style={{ width: `${(call.wall_seconds / maxWall) * 100}%` }}
                  />
                </div>
                <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-[11px] text-muted-foreground">
                  {[
                    ["Server total", call.metrics.total_duration],
                    ["Model loading", call.metrics.load_duration],
                    ["Prompt evaluation", call.metrics.prompt_eval_duration],
                    ["Token generation", call.metrics.eval_duration],
                  ].map(([label, value]) => (
                    <div
                      key={String(label)}
                      className="flex justify-between gap-2"
                    >
                      <span>{label}</span>
                      <span className="font-mono text-foreground">
                        {typeof value === "number"
                          ? duration(value / 1e9)
                          : "Unavailable"}
                      </span>
                    </div>
                  ))}
                </div>
                <p className="mt-3 text-[10px] text-muted-foreground">
                  {call.metrics.prompt_eval_count ?? "—"} input tokens ·{" "}
                  {call.metrics.eval_count ?? "—"} output tokens ·{" "}
                  {call.metrics.done_reason ?? "Stop reason unavailable"}
                </p>
              </div>
            ))}
          </div>
          <p className="mt-6 border-t pt-4 text-xs leading-relaxed text-muted-foreground">
            Request time includes transport and server work. Server durations
            are reported measurements; no inference queue split or GPU
            bottleneck is inferred.
          </p>
        </Card>
        <Card className="gap-0 p-5 sm:p-6">
          <h2 className="text-sm font-semibold">Model’s final answer</h2>
          <p className="mb-4 mt-1 text-xs text-muted-foreground">
            Preserved from the run · review against the tool evidence
          </p>
          <p className="whitespace-pre-wrap break-words text-sm leading-7">
            {(agent.final_answer || "No final answer was recorded.")
              .split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
              .map((part, index) =>
                part.startsWith("**") ? (
                  <strong key={index}>{part.slice(2, -2)}</strong>
                ) : part.startsWith("`") ? (
                  <code
                    key={index}
                    className="rounded bg-muted px-1 py-0.5 text-xs text-violet-200"
                  >
                    {part.slice(1, -1)}
                  </code>
                ) : (
                  part
                ),
              )}
          </p>
          <div className="mt-6 flex flex-wrap gap-2">
            {Object.entries(agent.checks ?? {}).map(([name, passed]) => (
              <Badge
                key={name}
                variant="outline"
                className={
                  passed ? "border-lime-300/20 text-lime-200" : "text-amber-200"
                }
              >
                {passed ? "✓" : "!"} {name.replaceAll("_", " ")}
              </Badge>
            ))}
          </div>
        </Card>
      </div>
      <Card className="gap-0 p-5 sm:p-6">
        <h2 className="text-sm font-semibold">Conversation & tool evidence</h2>
        <p className="mb-5 mt-1 text-xs text-muted-foreground">
          Expand a message to inspect the exact prompt, function arguments, or
          returned measurements.
        </p>
        <div className="divide-y rounded-lg border">
          {agent.messages.map((message, index) => (
            <details key={index} className="group px-4 py-3">
              <summary className="flex cursor-pointer list-none items-center gap-3 text-sm">
                <span className="w-5 text-xs text-muted-foreground">
                  {String(index + 1).padStart(2, "0")}
                </span>
                {message.role === "tool" ? (
                  <Terminal className="size-4 text-lime-300" />
                ) : (
                  <MessageSquare className="size-4 text-violet-300" />
                )}
                <span className="capitalize">{message.role}</span>
                <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground">
                  {message.tool_name ||
                    message.tool_calls
                      ?.map((call) => call.function.name)
                      .join(", ") ||
                    message.content.slice(0, 100)}
                </span>
                <span className="text-xs text-muted-foreground group-open:rotate-90">
                  ›
                </span>
              </summary>
              <div className="mt-4 space-y-3 border-l-2 border-violet-400/25 pl-4">
                {message.content && (
                  <pre className="whitespace-pre-wrap break-words text-xs leading-6 text-muted-foreground">
                    {message.role === "tool"
                      ? pretty(message.content)
                      : message.content}
                  </pre>
                )}
                {message.tool_calls?.map((call, i) => (
                  <pre
                    key={i}
                    className="whitespace-pre-wrap break-words rounded bg-muted/30 p-3 text-xs leading-6"
                  >
                    {call.function.name}(
                    {JSON.stringify(call.function.arguments, null, 2)})
                  </pre>
                ))}
              </div>
            </details>
          ))}
        </div>
      </Card>
    </div>
  );
}

function pretty(content: string) {
  try {
    return JSON.stringify(JSON.parse(content), null, 2);
  } catch {
    return content;
  }
}
