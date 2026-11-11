import { z } from "zod";
import {
  askAssistant,
  chatMessageSchema,
  MAX_HISTORY_MESSAGES,
} from "@/lib/assistant-chat";
import { requestSources } from "@/lib/assistant-sources";
import { insightConfig } from "@/lib/insights";
export const dynamic = "force-dynamic";

const requestSchema = z.object({
  messages: z
    .array(chatMessageSchema)
    .min(1)
    .max(MAX_HISTORY_MESSAGES * 2)
    .refine(
      (messages) => messages.at(-1)?.role === "user",
      "The last message must be the question.",
    ),
});

const COOLDOWN_MS = 2_000;
let lastStart = 0;
const STATUS = { unavailable: 503, rejected: 502, invalid_response: 502 };

/** GET reports whether chat is configured, so the page can say so before anyone types. */
export function GET() {
  const config = insightConfig(process.env);
  return Response.json({
    enabled: config !== null,
    model: config?.model ?? null,
  });
}

/** One chat turn. History comes from the browser; data always comes from the server's own stores. */
export async function POST(request: Request) {
  const config = insightConfig(process.env);
  if (!config)
    return Response.json(
      {
        error:
          "The assistant is not configured on this server. Set AGENTWATCH_INSIGHTS_URL and AGENTWATCH_INSIGHTS_MODEL.",
      },
      { status: 404 },
    );
  const parsed = requestSchema.safeParse(
    await request.json().catch(() => null),
  );
  if (!parsed.success)
    return Response.json({ error: "Invalid question." }, { status: 400 });
  if (Date.now() - lastStart < COOLDOWN_MS)
    return Response.json(
      { error: "Wait a moment before asking again." },
      { status: 429 },
    );
  lastStart = Date.now();

  const result = await askAssistant(
    config,
    parsed.data.messages,
    requestSources(),
  );
  if (!result.ok)
    return Response.json(
      { error: result.message },
      { status: STATUS[result.kind] },
    );
  return Response.json({
    text: result.text,
    records: result.records,
    tools: result.tools,
    removed: result.removed,
    model: result.model,
    answeredAt: new Date().toISOString(),
  });
}
