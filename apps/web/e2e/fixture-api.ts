import http from "node:http";
import {
  FIXTURE_API_PORT,
  fixtureRunSchema,
  savedRunSchema,
  type FixtureRun,
  type SavedRun,
} from "./fixture-control";

// A scripted stand-in for the B01 live read API and the saved run history (/v1/runs), used only
// by the browser tests. Every run it serves is synthetic test data registered by a test through
// the /__control endpoints; it is never recorded evidence. Record paging follows docs/live-ingestion.md: `after` filters by
// record sequence, `limit` bounds a page, and `next_cursor` is the last sequence of a full page.

type JsonObject = Record<string, unknown>;
type Snapshot = JsonObject & { records: { sequence: number }[] };
type RunState = FixtureRun & { index: number; reads: number[] };

const runs = new Map<string, RunState>();
const savedRuns = new Map<string, SavedRun>();
/** While set, saved history answers 503, as the worker does without its database. */
let savedHistoryDown = false;

const send = (res: http.ServerResponse, status: number, body?: unknown) => {
  if (body === undefined) return void res.writeHead(status).end();
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
};

async function bodyOf(req: http.IncomingMessage): Promise<unknown> {
  let text = "";
  for await (const chunk of req) text += chunk;
  return text ? JSON.parse(text) : null;
}

function listRuns(url: URL) {
  const limit = Number(url.searchParams.get("limit") ?? 50);
  const before = url.searchParams.get("before");
  const ordered = [...runs.values()]
    .map((run) => run.summary)
    .filter((summary) => before === null || summary.sequence < Number(before))
    .sort((a, b) => b.sequence - a.sequence);
  const page = ordered.slice(0, limit);
  return {
    runs: page,
    next_cursor: ordered.length > limit ? page[page.length - 1].sequence : null,
  };
}

/** Newest first by sequence, as the worker's run store pages its history. */
function listSavedRuns(url: URL) {
  const limit = Number(url.searchParams.get("limit") ?? 50);
  const before = url.searchParams.get("before");
  const ordered = [...savedRuns.values()]
    .filter((run) => before === null || run.sequence < Number(before))
    .sort((a, b) => b.sequence - a.sequence);
  const page = ordered.slice(0, limit);
  return {
    runs: page,
    next_cursor: ordered.length > limit ? page[page.length - 1].sequence : null,
  };
}

function snapshotPage(snapshot: Snapshot, after: number, limit: number) {
  const remaining = snapshot.records
    .filter((record) => record.sequence > after)
    .sort((a, b) => a.sequence - b.sequence);
  const records = remaining.slice(0, limit);
  return {
    ...snapshot,
    records,
    next_cursor:
      remaining.length > limit ? records[records.length - 1].sequence : null,
  };
}

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function handle(req: http.IncomingMessage, res: http.ServerResponse) {
  const url = new URL(req.url ?? "/", "http://fixture");
  const parts = url.pathname.split("/").filter(Boolean);

  if (parts[0] === "__control") {
    if (req.method === "POST" && parts[1] === "reset") {
      runs.clear();
      savedRuns.clear();
      savedHistoryDown = false;
      return send(res, 204);
    }
    if (req.method === "PUT" && parts[1] === "saved-runs") {
      const saved = savedRunSchema.safeParse(await bodyOf(req));
      if (!saved.success)
        return send(res, 400, { detail: saved.error.message });
      savedRuns.set(parts[2], saved.data);
      return send(res, 204);
    }
    if (req.method === "POST" && parts[1] === "saved-history-down") {
      savedHistoryDown = true;
      return send(res, 204);
    }
    const run = runs.get(parts[2] ?? "");
    if (req.method === "PUT" && parts[1] === "runs") {
      const fixture = fixtureRunSchema.safeParse(await bodyOf(req));
      if (!fixture.success)
        return send(res, 400, { detail: fixture.error.message });
      runs.set(parts[2], { ...fixture.data, index: 0, reads: [] });
      return send(res, 204);
    }
    if (!run) return send(res, 404, { detail: "No such fixture run" });
    if (req.method === "POST" && parts[3] === "advance") {
      run.index = Math.min(run.index + 1, run.states.length - 1);
      return send(res, 200, { index: run.index });
    }
    if (req.method === "GET" && parts[3] === "reads")
      return send(res, 200, run.reads);
    return send(res, 404, { detail: "Unknown control request" });
  }

  if (parts[0] === "v1" && parts[1] === "runs" && parts.length <= 3) {
    if (savedHistoryDown)
      return send(res, 503, { detail: "Database unavailable" });
    if (parts.length === 2) return send(res, 200, listSavedRuns(url));
    const saved = savedRuns.get(parts[2]);
    return saved
      ? send(res, 200, saved)
      : send(res, 404, { detail: "Run not found" });
  }
  if (parts.join("/") === "v1/live/runs") return send(res, 200, listRuns(url));
  if (parts.length === 4 && parts[0] === "v1" && parts[2] === "runs") {
    const run = runs.get(parts[3]);
    if (!run) return send(res, 404, { detail: "Live run not found" });
    const after = Number(url.searchParams.get("after") ?? 0);
    const limit = Number(url.searchParams.get("limit") ?? 256);
    const state = run.states[run.index];
    if (state.delayMs) await wait(state.delayMs);
    // Logged when answered, so a test can wait for a delayed response to be delivered.
    run.reads.push(after);
    if ("status" in state)
      return send(
        res,
        state.status,
        state.body ?? { detail: "Scripted failure" },
      );
    return send(res, 200, snapshotPage(state.snapshot, after, limit));
  }
  send(res, 404, { detail: "Not found" });
}

http
  .createServer((req, res) => {
    handle(req, res).catch((error: unknown) =>
      send(res, 500, { detail: `Fixture error: ${String(error)}` }),
    );
  })
  .listen(FIXTURE_API_PORT, "127.0.0.1");
