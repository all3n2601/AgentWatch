"use client";

import { routes } from "@/lib/routes";
import {
  ArrowUp,
  Info,
  Layers,
  LoaderCircle,
  RotateCcw,
  Sparkles,
  User,
} from "lucide-react";
import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import { z } from "zod";
import { RunRecordRow } from "@/components/widgets/run-record-row";
import { cn } from "@/lib/utils";

const MAX_QUESTION_CHARS = 2_000;
const HISTORY_SENT = 12;
const RECORDS_SHOWN = 5;

/** The answer as `/api/assistant` returns it; anything else is treated as a failure. */
const replySchema = z.object({
  text: z.string(),
  records: z
    .array(
      z.object({
        runId: z.string(),
        shortId: z.string(),
        source: z.enum(["dataset", "live", "saved"]),
        title: z.string(),
        status: z.string(),
        href: z
          .string()
          .regex(/^\/runs\/[0-9a-f-]{36}$/)
          .nullable(),
      }),
    )
    .max(50),
  tools: z.array(z.object({ name: z.string(), ok: z.boolean() })).max(20),
  removed: z.number().int().nonnegative(),
});
const errorSchema = z.object({ error: z.string() });
type RunRecord = z.infer<typeof replySchema>["records"][number];

type Turn =
  | { id: number; role: "user"; content: string }
  | {
      id: number;
      role: "assistant";
      content: string;
      records: RunRecord[];
      tools: { name: string; ok: boolean }[];
      removed: number;
    };

const TOOL_NAMES: Record<string, string> = {
  get_overview: "model overview",
  search_runs: "run search",
  get_run: "run details",
  compare_conditions: "condition comparison",
};

function AssistantTurn({
  turn,
}: {
  turn: Extract<Turn, { role: "assistant" }>;
}) {
  const shown = turn.records.slice(0, RECORDS_SHOWN);
  const lookedUp = [
    ...new Set(turn.tools.map((tool) => TOOL_NAMES[tool.name] ?? tool.name)),
  ];
  return (
    <div className="glass-inset space-y-2.5 p-3">
      <div className="flex items-start gap-3">
        <Layers className="mt-0.5 size-5 shrink-0 text-ink" aria-hidden />
        <p className="min-w-0 text-sm whitespace-pre-wrap text-ink">
          {turn.content}
        </p>
      </div>
      {shown.length > 0 && (
        <ul className="space-y-2" aria-label="Cited runs">
          {shown.map((record) => (
            <RunRecordRow
              key={`${record.source}:${record.runId}`}
              runId={record.runId}
              title={record.title}
              status={record.status}
              href={
                record.href === null
                  ? null
                  : routes.run(record.href.slice("/runs/".length))
              }
            />
          ))}
        </ul>
      )}
      {turn.removed > 0 && (
        <p className="text-xs text-warning">
          {turn.removed}{" "}
          {turn.removed === 1 ? "statement was" : "statements were"} left out:
          they quoted figures the retrieved records do not contain.
        </p>
      )}
      {lookedUp.length > 0 && (
        <p className="text-xs text-ink-subtle">
          Looked up: {lookedUp.join(", ")}
        </p>
      )}
    </div>
  );
}

/**
 * The Overview's investigation assistant. Questions go to `/api/assistant`,
 * which answers from AgentWatch's run data through the configured
 * OpenAI-compatible endpoint. Answers are plain text; nothing is rendered as
 * markup or executed. It opens empty: the conversation shows only what was asked.
 */
export function AssistantChat({
  enabled,
  suggestions = [],
}: {
  /** False when the server has no assistant endpoint configured. */
  enabled: boolean;
  /** Starter questions, centred in the empty conversation until the first one. */
  suggestions?: string[];
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nextId = useRef(0);
  const log = useRef<HTMLDivElement>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    log.current?.scrollTo({
      top: log.current.scrollHeight,
      behavior: "smooth",
    });
  }, [turns, pending, error]);
  useEffect(() => () => abort.current?.abort(), []);

  async function ask(question: string, history: Turn[]) {
    const content = question.trim().slice(0, MAX_QUESTION_CHARS);
    if (!content || pending) return;
    const userTurn: Turn = { id: nextId.current++, role: "user", content };
    const conversation = [...history, userTurn];
    setTurns(conversation);
    setDraft("");
    setError(null);
    setPending(true);
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    try {
      const response = await fetch("/api/assistant", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: conversation
            .slice(-HISTORY_SENT)
            .map((turn) => ({ role: turn.role, content: turn.content })),
        }),
        signal: controller.signal,
      });
      const body: unknown = await response.json().catch(() => null);
      if (!response.ok) {
        const failure = errorSchema.safeParse(body);
        throw new Error(
          failure.success
            ? failure.data.error
            : "The assistant did not answer.",
        );
      }
      const reply = replySchema.safeParse(body);
      if (!reply.success)
        throw new Error(
          "The assistant's reply was not in the expected format.",
        );
      const { text, records, tools, removed } = reply.data;
      setTurns((current) => [
        ...current,
        {
          id: nextId.current++,
          role: "assistant",
          content: text,
          records,
          tools,
          removed,
        },
      ]);
    } catch (caught) {
      if (controller.signal.aborted) return;
      setError(
        caught instanceof Error
          ? caught.message
          : "The assistant did not answer.",
      );
    } finally {
      if (abort.current === controller) {
        abort.current = null;
        setPending(false);
      }
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void ask(draft, turns);
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends; Shift+Enter keeps a new line.
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      void ask(draft, turns);
    }
  }

  function retry() {
    const lastQuestion = [...turns]
      .reverse()
      .find((turn) => turn.role === "user");
    if (!lastQuestion) return;
    void ask(
      lastQuestion.content,
      turns.slice(0, turns.lastIndexOf(lastQuestion)),
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div
        ref={log}
        role="log"
        aria-label="Conversation"
        aria-live="polite"
        className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1"
      >
        {enabled &&
          turns.length === 0 &&
          !pending &&
          suggestions.length > 0 && (
            <div className="flex h-full flex-col items-center justify-center gap-3 px-2 text-center">
              <Sparkles className="size-6 text-ink-subtle" aria-hidden />
              <p className="text-sm text-ink-muted">
                Ask about runs, stages or slowdowns
              </p>
              <ul
                className="flex w-full flex-col items-center gap-2"
                aria-label="Suggested questions"
              >
                {suggestions.map((suggestion) => (
                  <li key={suggestion} className="w-full">
                    <button
                      type="button"
                      onClick={() => void ask(suggestion, turns)}
                      className="glass-raised w-full rounded-2xl px-3 py-2.5 text-sm text-ink outline-none hover:bg-glass focus-visible:ring-[3px] focus-visible:ring-brand/40"
                    >
                      {suggestion}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        {turns.map((turn) =>
          turn.role === "user" ? (
            <div
              key={turn.id}
              className="flex items-start gap-3 rounded-2xl border border-success-line bg-brand-soft p-3"
            >
              <User className="mt-0.5 size-5 shrink-0 text-ink" aria-hidden />
              <p className="min-w-0 text-sm whitespace-pre-wrap text-ink">
                {turn.content}
              </p>
            </div>
          ) : (
            <AssistantTurn key={turn.id} turn={turn} />
          ),
        )}
        {pending && (
          <p
            className="flex items-center gap-2 px-1 text-sm text-ink-muted"
            role="status"
          >
            <LoaderCircle
              className="size-4 animate-spin motion-reduce:animate-none"
              aria-hidden
            />
            Reading run records…
          </p>
        )}
        {error && (
          <div
            role="alert"
            className="glass-inset flex items-start gap-3 border-danger-line p-3 text-sm text-danger"
          >
            <span className="min-w-0 flex-1">{error}</span>
            <button
              type="button"
              onClick={retry}
              className="glass-raised inline-flex h-9 shrink-0 items-center gap-1.5 rounded-full px-3 text-xs font-medium text-ink outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40"
            >
              <RotateCcw className="size-3.5" aria-hidden />
              Retry
            </button>
          </div>
        )}
      </div>

      <form onSubmit={submit} className="glass-inset flex items-end gap-2 p-2">
        <label htmlFor="assistant-question" className="sr-only">
          Ask AgentWatch
        </label>
        <textarea
          id="assistant-question"
          rows={2}
          value={draft}
          maxLength={MAX_QUESTION_CHARS}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={onKeyDown}
          disabled={!enabled}
          placeholder={
            enabled ? "Ask AgentWatch…" : "Assistant is off on this server"
          }
          className="max-h-32 min-h-11 flex-1 resize-none bg-transparent px-2 py-2 text-sm text-ink outline-none placeholder:text-ink-subtle disabled:cursor-not-allowed"
        />
        <button
          type="submit"
          disabled={!enabled || pending || draft.trim() === ""}
          aria-label="Send question"
          className={cn(
            "flex size-11 shrink-0 items-center justify-center rounded-full bg-brand-strong text-white shadow-glow outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40",
            "disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none",
          )}
        >
          <ArrowUp className="size-5" aria-hidden />
        </button>
      </form>
      <p className="flex items-center gap-1.5 text-xs text-ink-muted">
        <Info className="size-3.5 shrink-0" aria-hidden />
        {enabled
          ? "Answers cite retrieved records; figures not in them are left out."
          : "Set AGENTWATCH_INSIGHTS_URL and AGENTWATCH_INSIGHTS_MODEL to enable chat."}
      </p>
    </div>
  );
}
