import assert from "node:assert/strict";
import test from "node:test";
import {
  createLiveApi,
  failureStatus,
  isLiveRunId,
  LIVE_RUN_LIST_LIMIT,
  parseListQuery,
  parseSnapshotQuery,
} from "../lib/live-api";
import {
  fakeFetch,
  resourceSample,
  RUN_ID,
  runIdFor,
  runSnapshot,
  runSummary,
  savedRecord,
  stepEvent,
} from "./live-fixtures";

const BASE = "http://live.test";
const json = (body: unknown, status = 200) => Response.json(body, { status });

test("a selected run older than the newest history page is read by its UUID", async () => {
  const newest = Array.from({ length: LIVE_RUN_LIST_LIMIT }, (_, i) =>
    runSummary(runIdFor(i + 2), 100 - i),
  );
  const older = runIdFor(1);
  const { fetchImpl, requested } = fakeFetch((url) =>
    url.pathname === "/v1/live/runs"
      ? json({ runs: newest, next_cursor: newest.at(-1)!.sequence })
      : url.pathname === `/v1/live/runs/${older}`
        ? json(runSnapshot({}, older))
        : json({ detail: "Live run not found" }, 404),
  );
  const api = createLiveApi(BASE, fetchImpl);

  const page = await api.runPage();
  assert.ok(page.ok);
  assert.equal(page.data.runs.length, LIVE_RUN_LIST_LIMIT);
  assert.ok(!page.data.runs.some((run) => run.run_id === older));

  const selected = await api.runSnapshot(older);
  assert.ok(selected.ok);
  assert.equal(selected.data.run_id, older);
  assert.equal(requested.at(-1)!.pathname, `/v1/live/runs/${older}`);
  assert.equal(requested.at(-1)!.searchParams.get("after"), "0");
  assert.equal(requested.at(-1)!.searchParams.get("limit"), "256");
});

test("history pages continue from the previous cursor", async () => {
  const { fetchImpl, requested } = fakeFetch(() =>
    json({ runs: [], next_cursor: null }),
  );
  await createLiveApi(BASE, fetchImpl).runPage(42);
  assert.equal(requested[0].searchParams.get("before"), "42");
  assert.equal(requested[0].searchParams.get("limit"), "50");
});

test("snapshots keep records, gaps and fields added by a newer service", async () => {
  const body = runSnapshot({
    records: [
      savedRecord(stepEvent("tool-1", 0, "queued"), 1),
      savedRecord(stepEvent("tool-1", 2, "completed"), 2, "pending"),
      savedRecord(
        resourceSample("s-1", { value: null, availability: "unsupported" }),
        3,
      ),
    ],
    gaps: [
      {
        step_id: "tool-1",
        missing_sequence: 1,
        first_pending_sequence: 2,
        expired: false,
      },
    ],
    added_by_newer_service: "kept",
  });
  const { fetchImpl } = fakeFetch(() => json(body));
  const result = await createLiveApi(BASE, fetchImpl).runSnapshot(RUN_ID, 7);
  assert.ok(result.ok);
  assert.equal(result.data.records.length, 3);
  assert.equal(result.data.records[1].disposition, "pending");
  assert.equal(result.data.records[2].payload.record_type, "resource_sample");
  assert.equal(result.data.gaps[0].missing_sequence, 1);
  assert.equal(result.data.added_by_newer_service, "kept");
});

test("upstream statuses map to distinct failure kinds", async () => {
  const cases = [
    [404, "not_found"],
    [422, "invalid_request"],
    [503, "unavailable"],
    [500, "unavailable"],
  ] as const;
  for (const [status, kind] of cases) {
    const { fetchImpl } = fakeFetch(() => json({ detail: "x" }, status));
    const result = await createLiveApi(BASE, fetchImpl).runSnapshot(RUN_ID);
    assert.ok(!result.ok);
    assert.equal(result.kind, kind, `HTTP ${status}`);
  }
  assert.equal(failureStatus.not_found, 404);
  assert.equal(failureStatus.unavailable, 503);
});

test("transport failures and timeouts are reported as unavailable", async () => {
  const refused = fakeFetch(() => {
    throw new TypeError("fetch failed");
  });
  const unreachable = await createLiveApi(BASE, refused.fetchImpl).runPage();
  assert.ok(!unreachable.ok);
  assert.equal(unreachable.kind, "unavailable");

  const slow = fakeFetch(() => {
    throw new DOMException("timed out", "TimeoutError");
  });
  const timedOut = await createLiveApi(BASE, slow.fetchImpl).runPage();
  assert.ok(!timedOut.ok);
  assert.equal(timedOut.kind, "unavailable");
  assert.match(timedOut.message, /did not respond within 8 s/);
});

test("bodies that are not JSON or not the live contract are invalid responses", async () => {
  const notJson = fakeFetch(() => new Response("<html>proxy error</html>"));
  const garbled = await createLiveApi(BASE, notJson.fetchImpl).runPage();
  assert.ok(!garbled.ok);
  assert.equal(garbled.kind, "invalid_response");

  const zeroForMissing = runSnapshot({
    records: [savedRecord(resourceSample("s-1", { availability: "zero" }), 1)],
  });
  const offContract = fakeFetch(() => json(zeroForMissing));
  const result = await createLiveApi(BASE, offContract.fetchImpl).runSnapshot(
    RUN_ID,
  );
  assert.ok(!result.ok);
  assert.equal(result.kind, "invalid_response");
  assert.doesNotMatch(result.message, /zero/, "input values are not echoed");
});

test("record cursors and page sizes are bounded before reaching the service", () => {
  const parse = (query: string) =>
    parseSnapshotQuery(new URLSearchParams(query));
  assert.deepEqual(parse("").data, { after: 0, limit: 256 });
  assert.deepEqual(parse("after=12&limit=10").data, { after: 12, limit: 10 });
  for (const bad of [
    "after=-1",
    "after=1.5",
    "after=abc",
    `after=${Number.MAX_SAFE_INTEGER + 2}`,
    "limit=0",
    "limit=257",
  ])
    assert.equal(parse(bad).success, false, bad);

  const list = (query: string) => parseListQuery(new URLSearchParams(query));
  assert.deepEqual(list("").data, {});
  assert.deepEqual(list("before=9").data, { before: 9 });
  assert.equal(list("before=0").success, false);
});

test("only UUIDs are accepted as live run IDs", () => {
  assert.ok(isLiveRunId(RUN_ID));
  assert.ok(!isLiveRunId("../../v1/runs"));
  assert.ok(!isLiveRunId("not-a-uuid"));
});
