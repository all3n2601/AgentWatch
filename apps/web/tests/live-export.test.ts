import assert from "node:assert/strict";
import test from "node:test";
import { createLiveApi } from "../lib/live-api";
import {
  LIVE_EXPORT_KIND,
  liveExportFilename,
  MAX_EXPORT_PAGES,
  readLiveRecordsExport,
} from "../lib/live-export";
import {
  fakeFetch,
  OTHER_RUN_ID,
  resourceSample,
  RUN_ID,
  runSnapshot,
  savedRecord,
  stepEvent,
} from "./live-fixtures";

const RUNS_URL = "http://live.test/v1/live/runs";
const json = (body: unknown, status = 200) => Response.json(body, { status });
const clock = () => {
  let second = 0;
  return () => new Date(Date.UTC(2026, 9, 24, 12, 0, second++));
};

const queued = stepEvent("tool-1", 0, "queued");
const held = stepEvent("tool-1", 2, "completed", { started: 1, finished: 2 });
const gap = {
  step_id: "tool-1",
  missing_sequence: 1,
  first_pending_sequence: 2,
  expired: true,
};

test("every page of the selected UUID is read, keeping gaps and dispositions as reported", async () => {
  const pages: Record<string, unknown> = {
    "0": runSnapshot({
      records: [savedRecord(queued, 1), savedRecord(held, 2, "expired")],
      gaps: [gap],
      next_cursor: 2,
    }),
    "2": runSnapshot({
      state: "failed",
      reason: "Evidence expired",
      records: [
        savedRecord(
          resourceSample("s-1", { value: null, availability: "missing" }),
          3,
        ),
      ],
      gaps: [gap],
      next_cursor: null,
    }),
  };
  const { fetchImpl, requested } = fakeFetch((url) =>
    json(pages[url.searchParams.get("after")!]),
  );
  const result = await readLiveRecordsExport(
    createLiveApi(RUNS_URL, { fetchImpl }),
    RUN_ID,
    clock(),
  );
  assert.ok(result.ok);
  const exported = result.data;
  assert.equal(exported.export_kind, LIVE_EXPORT_KIND);
  assert.equal(exported.run_id, RUN_ID);
  assert.match(exported.description, /not saved evidence/);
  assert.equal(exported.pages_read, 2);
  assert.equal(exported.read_started_at, "2026-10-24T12:00:00.000Z");
  assert.equal(exported.read_finished_at, "2026-10-24T12:00:01.000Z");
  assert.deepEqual(
    exported.records.map((r) => [r.sequence, r.disposition]),
    [
      [1, "applied"],
      [2, "expired"],
      [3, "applied"],
    ],
  );
  assert.equal(exported.records[2].payload.record_type, "resource_sample");
  // Run state and gaps are from the last page, unchanged and not filled in.
  assert.equal(exported.run.state, "failed");
  assert.deepEqual(exported.run.gaps, [gap]);
  assert.ok(!("next_cursor" in exported.run));
  assert.ok(!("records" in exported.run));
  assert.ok(
    requested.every((url) => url.pathname === `/v1/live/runs/${RUN_ID}`),
    "only the selected UUID is read",
  );
});

test("a run the API does not have fails the export instead of substituting another", async () => {
  const { fetchImpl, requested } = fakeFetch(() =>
    json({ detail: "Live run not found" }, 404),
  );
  const result = await readLiveRecordsExport(
    createLiveApi(RUNS_URL, { fetchImpl }),
    RUN_ID,
  );
  assert.ok(!result.ok);
  assert.equal(result.kind, "not_found");
  assert.equal(requested.length, 1);
});

test("a failed later page fails the whole export; no partial file is produced", async () => {
  const { fetchImpl } = fakeFetch((url) =>
    url.searchParams.get("after") === "0"
      ? json(runSnapshot({ records: [savedRecord(queued, 1)], next_cursor: 1 }))
      : json({ detail: "Database unavailable" }, 503),
  );
  const result = await readLiveRecordsExport(
    createLiveApi(RUNS_URL, { fetchImpl }),
    RUN_ID,
  );
  assert.ok(!result.ok);
  assert.equal(result.kind, "unavailable");
});

test("a page for another run is rejected", async () => {
  const { fetchImpl } = fakeFetch(() => json(runSnapshot({}, OTHER_RUN_ID)));
  const result = await readLiveRecordsExport(
    createLiveApi(RUNS_URL, { fetchImpl }),
    RUN_ID,
  );
  assert.ok(!result.ok);
  assert.equal(result.kind, "invalid_response");
});

test("reading stops at the service's retention bound", async () => {
  let page = 0;
  const { fetchImpl } = fakeFetch(() =>
    json(runSnapshot({ records: [], next_cursor: ++page })),
  );
  const result = await readLiveRecordsExport(
    createLiveApi(RUNS_URL, { fetchImpl }),
    RUN_ID,
  );
  assert.ok(!result.ok);
  assert.equal(page, MAX_EXPORT_PAGES);
});

test("the download is named after the selected UUID", () => {
  assert.equal(
    liveExportFilename(RUN_ID),
    `agentwatch-live-records-${RUN_ID}.json`,
  );
});
