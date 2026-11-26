import http from "node:http";
import {
  FIXTURE_API_PORT,
  fixtureRunSchema,
  savedRunSchema,
  type FixtureRun,
  type RequestLogEntry,
  type SavedRun,
} from "./fixture-control";

// A scripted stand-in for the B01 live read API and the saved run history (/v1/runs), used only
// by the browser tests. Every run it serves is synthetic test data registered by a test through
// the /__control endpoints; it is never recorded evidence. Record paging follows docs/live-ingestion.md: `after` filters by
// record sequence, `limit` bounds a page, and `next_cursor` is the last sequence of a full page.

type JsonObject = Record<string, unknown>;
type Snapshot = JsonObject & { records: { sequence: number }[] };
type RunState = FixtureRun & {
  index: number;
  reads: number[];
  attributionIndex: number;
  attributionReads: number[];
};
type AttributionRow = { sequence: number };

const runs = new Map<string, RunState>();
const savedRuns = new Map<string, SavedRun>();
/**
 * Every request the web server made to this API (control requests excluded), in arrival
 * order with this process's wall time. Tests read it to count polls and to show that no
 * read happened in a window.
 */
const requestLog: RequestLogEntry[] = [];
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

/** B03 paging: rows after `after`, oldest first; `next_cursor` only when more rows remain. */
function attributionsPage(
  page: { attributions: AttributionRow[] } & JsonObject,
  after: number,
  limit: number,
) {
  const remaining = page.attributions
    .filter((row) => row.sequence > after)
    .sort((a, b) => a.sequence - b.sequence);
  const attributions = remaining.slice(0, limit);
  return {
    ...page,
    attributions,
    next_cursor:
      remaining.length > limit
        ? attributions[attributions.length - 1].sequence
        : null,
  };
}

function serveAttributions(res: http.ServerResponse, run: RunState, url: URL) {
  const after = Number(url.searchParams.get("after") ?? 0);
  const limit = Number(url.searchParams.get("limit") ?? 50);
  run.attributionReads.push(after);
  const state = run.attributions?.[run.attributionIndex];
  if (!state)
    return send(res, 503, { detail: "No scripted attributions for this run" });
  if ("status" in state)
    return send(
      res,
      state.status,
      state.body ?? { detail: "Scripted failure" },
    );
  return send(res, 200, attributionsPage(state.page, after, limit));
}

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const ARTIFACT_FILES = ["results.json", "spans.jsonl", "samples.jsonl"];

/**
 * A saved run's evidence file, as the worker serves /v1/runs/{id}/artifacts/{file}:
 * results.json is the registered result; spans and samples are synthetic lines tagged with
 * the run id, so a test can tell which run a download came from.
 */
function serveArtifact(
  res: http.ServerResponse,
  saved: SavedRun,
  file: string,
) {
  if (file === "results.json") return send(res, 200, saved.result);
  const kind = file === "spans.jsonl" ? "span" : "sample";
  const lines = [0, 1, 2].map((index) =>
    JSON.stringify({
      synthetic: `browser-test ${kind} ${index}`,
      "agentwatch.run_id": saved.run_id,
    }),
  );
  res.writeHead(200, { "content-type": "application/x-ndjson" });
  res.end(`${lines.join("\n")}\n`);
}

async function handle(req: http.IncomingMessage, res: http.ServerResponse) {
  const url = new URL(req.url ?? "/", "http://fixture");
  const parts = url.pathname.split("/").filter(Boolean);

  if (parts[0] === "__control") {
    if (req.method === "POST" && parts[1] === "reset") {
      runs.clear();
      savedRuns.clear();
      requestLog.length = 0;
      savedHistoryDown = false;
      return send(res, 204);
    }
    if (req.method === "GET" && parts[1] === "requests")
      return send(res, 200, requestLog);
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
    if (req.method === "POST" && parts[1] === "saved-history-up") {
      savedHistoryDown = false;
      return send(res, 204);
    }
    const run = runs.get(parts[2] ?? "");
    if (req.method === "PUT" && parts[1] === "runs") {
      const fixture = fixtureRunSchema.safeParse(await bodyOf(req));
      if (!fixture.success)
        return send(res, 400, { detail: fixture.error.message });
      runs.set(parts[2], {
        ...fixture.data,
        index: 0,
        reads: [],
        attributionIndex: 0,
        attributionReads: [],
      });
      return send(res, 204);
    }
    if (!run) return send(res, 404, { detail: "No such fixture run" });
    if (req.method === "POST" && parts[3] === "advance") {
      run.index = Math.min(run.index + 1, run.states.length - 1);
      return send(res, 200, { index: run.index });
    }
    if (req.method === "POST" && parts[3] === "advance-attributions") {
      const last = (run.attributions?.length ?? 1) - 1;
      run.attributionIndex = Math.min(run.attributionIndex + 1, last);
      return send(res, 200, { index: run.attributionIndex });
    }
    if (req.method === "GET" && parts[3] === "reads")
      return send(res, 200, run.reads);
    if (req.method === "GET" && parts[3] === "attribution-reads")
      return send(res, 200, run.attributionReads);
    return send(res, 404, { detail: "Unknown control request" });
  }

  requestLog.push({
    method: req.method ?? "GET",
    path: url.pathname,
    query: url.search,
    atMs: Date.now(),
  });

  if (
    parts.length === 5 &&
    parts[0] === "v1" &&
    parts[1] === "runs" &&
    parts[3] === "artifacts"
  ) {
    if (savedHistoryDown)
      return send(res, 503, { detail: "Database unavailable" });
    const saved = savedRuns.get(parts[2]);
    if (!saved) return send(res, 404, { detail: "Run not found" });
    return ARTIFACT_FILES.includes(parts[4])
      ? serveArtifact(res, saved, parts[4])
      : send(res, 404, { detail: "Unknown artifact" });
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
  if (
    parts.length === 5 &&
    parts.slice(0, 3).join("/") === "v1/live/runs" &&
    parts[4] === "attributions"
  ) {
    const run = runs.get(parts[3]);
    return run
      ? serveAttributions(res, run, url)
      : send(res, 404, { detail: "Live run not found" });
  }
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
