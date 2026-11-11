import assert from "node:assert/strict";
import test from "node:test";
import {
  createPolicyApi,
  createPolicyBrowserApi,
  policyFailureStatus,
} from "../lib/policy-api";
import {
  conflictCause,
  controlsConsistent,
  EFFECTIVE_LIMITS_UNAVAILABLE,
  formFromLimits,
  limitChanges,
  limitsFromForm,
  limitsText,
  policyStatusText,
  runAppliedLimitsText,
  runPolicyText,
  versionLimitsText,
} from "../lib/policy-view";
import { fakeFetch, RUN_ID } from "./live-fixtures";

const API_ROOT = "http://worker.test";
const URLS = {
  policy: `${API_ROOT}/v1/policies/concurrency/coding-agent`,
  liveRuns: `${API_ROOT}/v1/live/runs`,
};
const json = (body: unknown, status = 200) => Response.json(body, { status });

const controlledServices = {
  inference_requests: { service_id: "ollama", resource_class: "inference" },
  test_jobs: {
    service_id: "agentwatch-agent-runner",
    resource_class: "cpu_sandbox",
  },
};
const baseline = {
  target: "coding-agent",
  version: 0,
  action: "baseline",
  limits: {},
  restored_from: null,
  operator: null,
  reason: null,
  request_id: null,
  created_at: null,
  controlled_services: controlledServices,
  enforcement_scope: "runner_process",
};
const version2 = {
  ...baseline,
  version: 2,
  action: "set",
  limits: { test_jobs: 2 },
  operator: "alice",
  reason: "Limit sandbox contention",
  request_id: "6f1c1d1e-8a63-4f43-9c3c-0f7a1f6f4b11",
  created_at: "2026-11-03T06:30:29.001256Z",
};

test("parses baseline and versioned policies", async () => {
  const { fetchImpl, requested } = fakeFetch((url) =>
    url.pathname.endsWith("/versions")
      ? json({ versions: [version2, baseline], next_before: null })
      : json(baseline),
  );
  const api = createPolicyApi(URLS, { fetchImpl });

  const active = await api.active();
  assert.ok(active.ok);
  assert.equal(active.data.version, 0);
  assert.equal(active.data.action, "baseline");
  assert.deepEqual(active.data.limits, {});
  assert.equal(active.data.created_at, null);

  const page = await api.history(3);
  assert.ok(page.ok);
  assert.deepEqual(
    page.data.versions.map((v) => [v.version, v.limits]),
    [
      [2, { test_jobs: 2 }],
      [0, {}],
    ],
  );
  assert.equal(page.data.next_before, null);
  assert.equal(
    requested.at(-1)!.pathname,
    "/v1/policies/concurrency/coding-agent/versions",
  );
  assert.equal(requested.at(-1)!.searchParams.get("before"), "3");
});

test("limits outside the contract are an invalid response", async () => {
  // "No limit" is an absent key: null, zero, fractions and unknown controls are rejected.
  for (const limits of [
    { inference_requests: null, test_jobs: 2 },
    { test_jobs: 0 },
    { test_jobs: 1.5 },
    { test_jobs: 1025 },
    { test_jobs: 2, gpu_jobs: 1 },
  ]) {
    const { fetchImpl } = fakeFetch(() => json({ ...version2, limits }));
    const active = await createPolicyApi(URLS, { fetchImpl }).active();
    assert.ok(!active.ok, JSON.stringify(limits));
    assert.equal(active.kind, "invalid_response");
    assert.equal(policyStatusText(active), "Policy response is invalid");
  }
});

test("maps 409 to conflict and 422 to invalid_request", async () => {
  let respond = () =>
    json({ detail: "Policy is at version 3, not 2" }, 409) as Response;
  const { fetchImpl, requested } = fakeFetch(() => respond());
  const api = createPolicyApi(URLS, { fetchImpl });
  const change = {
    request_id: "1d6e2a59-55b4-4d0d-9c49-3e4fa1c07a10",
    expected_version: 2,
    action: "disable" as const,
    operator: "alice",
    reason: "Remove limits",
  };

  const conflict = await api.change(change);
  assert.ok(!conflict.ok);
  assert.equal(conflict.kind, "conflict");
  assert.equal(conflict.message, "Policy is at version 3, not 2");
  assert.equal(policyFailureStatus.conflict, 409);

  respond = () =>
    json(
      {
        detail: [
          {
            loc: ["body", "limits", "test_jobs"],
            msg: "Input should be greater than or equal to 1",
            type: "greater_than_equal",
          },
        ],
      },
      422,
    );
  const invalid = await api.change(change);
  assert.ok(!invalid.ok);
  assert.equal(invalid.kind, "invalid_request");
  assert.equal(
    invalid.message,
    "limits.test_jobs: Input should be greater than or equal to 1",
  );

  respond = () => json({ detail: "Unknown restore source version 9" }, 422);
  const restore = await api.change(change);
  assert.ok(!restore.ok);
  assert.equal(restore.kind, "invalid_request");
  assert.equal(restore.message, "Unknown restore source version 9");
  assert.equal(requested.at(-1)!.pathname.endsWith("/versions"), true);
});

test("a change is posted as JSON and the stored version is returned", async () => {
  const bodies: unknown[] = [];
  const fetchImpl: typeof fetch = async (_input, init) => {
    assert.equal(init?.method, "POST");
    bodies.push(JSON.parse(String(init?.body)));
    return json({ created: true, policy: version2 }, 201);
  };
  const change = {
    request_id: "1d6e2a59-55b4-4d0d-9c49-3e4fa1c07a10",
    expected_version: 1,
    action: "set" as const,
    limits: { test_jobs: 2 },
    operator: "alice",
    reason: "Limit sandbox contention",
  };
  const result = await createPolicyApi(URLS, { fetchImpl }).change(change);
  assert.ok(result.ok);
  assert.equal(result.data.created, true);
  assert.equal(result.data.policy.version, 2);
  assert.deepEqual(bodies, [change]);
});

test("maps 503 and network failure to unavailable", async () => {
  const down = fakeFetch(() => json({ detail: "Database unavailable" }, 503));
  const unavailable = await createPolicyApi(URLS, {
    fetchImpl: down.fetchImpl,
  }).active();
  assert.ok(!unavailable.ok);
  assert.equal(unavailable.kind, "unavailable");

  const unreachable = await createPolicyApi(URLS, {
    fetchImpl: async () => {
      throw new TypeError("fetch failed");
    },
  }).active();
  assert.ok(!unreachable.ok);
  assert.equal(unreachable.kind, "unavailable");

  for (const result of [unavailable, unreachable]) {
    assert.equal(policyStatusText(result), "Policy service unavailable");
    assert.doesNotMatch(policyStatusText(result), /No limits/);
  }
  assert.equal(policyStatusText(null), "Loading policy…");
});

test("policy status shows the active version and its limits", () => {
  assert.equal(
    policyStatusText({ ok: true, data: { version: 0, limits: {} } }),
    "Version 0: No limits (baseline)",
  );
  assert.equal(
    policyStatusText({
      ok: true,
      data: { version: 2, limits: { test_jobs: 2 } },
    }),
    "Version 2: Inference: no limit · Tests: 2",
  );
});

test("limitsText describes absent controls as no limit", () => {
  assert.equal(limitsText({}), "No limits");
  assert.equal(limitsText({ test_jobs: 2 }), "Inference: no limit · Tests: 2");
  assert.equal(
    limitsText({ inference_requests: 1, test_jobs: 4 }),
    "Inference: 1 · Tests: 4",
  );
});

test("only version 0 is labelled as the baseline", () => {
  assert.equal(
    versionLimitsText({ version: 0, limits: {} }),
    "No limits (baseline)",
  );
  // A disable or restore-to-baseline version has no limits but is not the baseline.
  assert.equal(versionLimitsText({ version: 3, limits: {} }), "No limits");
  assert.equal(
    versionLimitsText({ version: 4, limits: { inference_requests: 2 } }),
    "Inference: 2 · Tests: no limit",
  );
});

test("form limits omit empty inputs and reject values outside 1–1024", () => {
  assert.deepEqual(limitsFromForm({ inference: "", tests: " 2 " }), {
    ok: true,
    limits: { test_jobs: 2 },
  });
  assert.deepEqual(limitsFromForm({ inference: "3", tests: "" }), {
    ok: true,
    limits: { inference_requests: 3 },
  });
  for (const bad of ["0", "-1", "1.5", "abc", "1025", "1e2"]) {
    const result = limitsFromForm({ inference: bad, tests: "" });
    assert.equal(result.ok, false, bad);
  }
  const empty = limitsFromForm({ inference: "", tests: "" });
  assert.equal(empty.ok, false);
  assert.match(!empty.ok ? empty.message : "", /Disable/);
});

test("the form starts from the active limits so unchanged controls are kept", () => {
  assert.deepEqual(formFromLimits({}), { inference: "", tests: "" });
  assert.deepEqual(formFromLimits({ test_jobs: 2 }), {
    inference: "",
    tests: "2",
  });
  const form = { ...formFromLimits({ test_jobs: 2 }), inference: "8" };
  assert.deepEqual(limitsFromForm(form), {
    ok: true,
    limits: { inference_requests: 8, test_jobs: 2 },
  });
});

test("limit changes list only the controls that differ", () => {
  assert.deepEqual(limitChanges({ test_jobs: 2 }, { test_jobs: 2 }), []);
  assert.deepEqual(limitChanges({ test_jobs: 2 }, { inference_requests: 1 }), [
    { control: "Inference requests", active: "no limit", proposed: "1" },
    { control: "Test jobs", active: "2", proposed: "no limit" },
  ]);
});

test("conflicts are told apart by the service's detail text", () => {
  assert.equal(
    conflictCause("Policy changed: expected version 2, active version is 3"),
    "stale_version",
  );
  assert.equal(
    conflictCause("Policy request reused with a different body"),
    "request_reused",
  );
  assert.equal(conflictCause("Policy is at version 3, not 2"), "unknown");
});

test("a read whose history does not start at the active version is inconsistent", () => {
  const read = (active: number, newest: number[]) => ({
    active: { version: active },
    history: { versions: newest.map((version) => ({ version })) },
  });
  assert.equal(controlsConsistent(read(2, [2, 1, 0])), true);
  assert.equal(controlsConsistent(read(2, [3, 2, 1, 0])), false);
  assert.equal(controlsConsistent(read(3, [2, 1, 0])), false);
  assert.equal(controlsConsistent(read(0, [])), false);
});

test("run policy is read by UUID and summarised", async () => {
  const { fetchImpl, requested } = fakeFetch(() =>
    json({
      run_id: RUN_ID,
      target: "coding-agent",
      applicable: true,
      versions: [baseline, { ...version2, version: 1 }],
    }),
  );
  const result = await createPolicyApi(URLS, { fetchImpl }).runPolicy(RUN_ID);
  assert.ok(result.ok);
  assert.equal(requested[0].pathname, `/v1/live/runs/${RUN_ID}/policy`);
  assert.equal(
    runPolicyText(result),
    "Policy while open: v0 (No limits (baseline)) → v1 (Inference: no limit · Tests: 2)",
  );

  assert.equal(
    runPolicyText({
      ok: true,
      data: { ...result.data, applicable: false, versions: [] },
    }),
    "Not an agent run",
  );
  assert.equal(
    runPolicyText({ ok: true, data: { ...result.data, versions: [] } }),
    "No recorded policy (registered before policy tracking)",
  );
  assert.equal(
    runPolicyText({ ok: false, kind: "unavailable", message: "down" }),
    "Policy service unavailable",
  );
});

test("no run claims limits were applied; non-agent runs admit nothing", () => {
  const run = {
    run_id: RUN_ID,
    target: "coding-agent",
    applicable: true,
    versions: [],
  };
  assert.equal(
    runAppliedLimitsText({ ok: true, data: run }),
    EFFECTIVE_LIMITS_UNAVAILABLE,
  );
  assert.equal(
    runAppliedLimitsText({ ok: false, kind: "unavailable", message: "down" }),
    EFFECTIVE_LIMITS_UNAVAILABLE,
  );
  assert.equal(runAppliedLimitsText(null), EFFECTIVE_LIMITS_UNAVAILABLE);
  assert.match(
    runAppliedLimitsText({
      ok: true,
      data: { ...run, applicable: false, versions: [] },
    }),
    /^Not applicable/,
  );
});

test("the browser client keeps the failure kind the web route classified", async () => {
  const { fetchImpl, requested } = fakeFetch(() =>
    json({ error: "Policy is at version 3, not 2", kind: "conflict" }, 409),
  );
  const api = createPolicyBrowserApi({ baseUrl: "http://web.test", fetchImpl });
  const result = await api.change({
    request_id: "1d6e2a59-55b4-4d0d-9c49-3e4fa1c07a10",
    expected_version: 2,
    action: "disable",
    operator: "alice",
    reason: "Remove limits",
  });
  assert.ok(!result.ok);
  assert.equal(result.kind, "conflict");
  assert.equal(result.message, "Policy is at version 3, not 2");
  assert.equal(requested[0].pathname, "/api/policies/concurrency");
});
