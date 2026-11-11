import assert from "node:assert/strict";
import test from "node:test";
import type { DatasetSnapshot } from "../lib/dataset-schema";
import {
  buildQualityPage,
  checkState,
  checkTally,
  fieldHealth,
  qualityChecks,
  splitSummaries,
  telemetryParts,
} from "../lib/quality-page";

type Ready = Extract<DatasetSnapshot, { status: "ready" }>;

// Synthetic release for tests only; it is not recorded-run evidence.
const snapshot = (
  overrides: {
    validation?: Partial<Ready["validation"]>;
    split?: Partial<Ready["split"]>;
    missing?: Record<string, number>;
  } = {},
): Ready => ({
  status: "ready",
  manifest: {
    dataset_version: "steps-v9.9+abc123",
    schema_version: "steps-v9.9",
    release_format: 3,
    rows: 100,
    runs: 10,
    tasks: 2,
    kind: { agent_replay: 80, cpu: 20 },
    condition: { clean: 50, contended: 50 },
    label_resource: { none: 70, inference: 20, mixed: 10 },
    label_basis: { clean_within_noise: 60, mixed_confirmed: 40 },
    telemetry_status: { empty: 30, present: 50, no_overlap: 20 },
    missing: overrides.missing ?? {
      label_resource: 0,
      cpu_seconds: 40,
      agent_model: 20,
      queue_seconds: 0,
    },
    label_version: "labels-v1+aa,labels-v2+bb",
  },
  split: {
    split_id: "split-v2+abc",
    split_method: "split-v2",
    contract: { version: "eval-v2", status: "accepted", group_key: "task" },
    class_balance: {
      status: "met",
      per_class: { none: 40, inference: 12 },
      clean: 30,
      mixed: 5,
      minimums: { clean: 10, mixed: 2, per_class: 5 },
    },
    splits: {
      test: {
        rows: 60,
        runs: 6,
        groups: 1,
        label_resource: { none: 40, inference: 20 },
        condition: {},
        injection_method: {},
      },
      validation: {
        rows: 0,
        runs: 0,
        groups: 0,
        label_resource: {},
        condition: {},
        injection_method: {},
      },
      train: {
        rows: 40,
        runs: 4,
        groups: 1,
        label_resource: { none: 30, mixed: 10 },
        condition: {},
        injection_method: {},
      },
    },
    design_encoding: {
      status: "report_only",
      rows: 20,
      features: [{ feature: "queue_seconds", auc: 0.6 }],
    },
    leakage: {
      status: "passed",
      max_single_feature_auc: 0.9,
      features: [{ feature: "queue_seconds", class: "inference", auc: 0.85 }],
    },
    held_out_injection_methods: {
      status: "met",
      methods: ["a", "b"],
      only_in_test: ["b"],
    },
    ...overrides.split,
  },
  validation: {
    status: "passed",
    validated_at: "2026-11-01T00:00:00+00:00",
    examples: 100,
    anomalies: [],
    drift: [],
    gate_version: "0.1.0",
    tfdv_version: "1.0.0",
    ...overrides.validation,
  },
  runs: [],
});

test("recorded verdicts map to states; anything unrecognised is unknown, never a pass", () => {
  assert.equal(checkState("passed"), "pass");
  assert.equal(checkState("met"), "pass");
  assert.equal(checkState("Accepted"), "pass");
  assert.equal(checkState("not_met"), "fail");
  assert.equal(checkState("blocked"), "fail");
  assert.equal(checkState("failed"), "fail");
  assert.equal(checkState("inconclusive"), "warn");
  for (const status of ["report_only", "unavailable", "", "pass", "ok"])
    assert.equal(checkState(status), "unknown");
});

test("checks keep the recorded status and do not pass on unknown or empty drift", () => {
  const ready = snapshot({
    validation: { status: "pending", drift: [] },
    split: {
      leakage: {
        status: "mystery",
        max_single_feature_auc: 0.9,
        features: [],
      },
    },
  });
  const checks = Object.fromEntries(
    qualityChecks(ready.validation, ready.split).map((c) => [c.id, c]),
  );
  assert.equal(checks.gate.state, "unknown");
  assert.match(checks.gate.detail, /Recorded pending/);
  assert.equal(checks.leakage.state, "unknown");
  assert.match(checks.leakage.detail, /no per-class features recorded/);
  assert.equal(checks.drift.state, "unknown");
  assert.equal(checks.anomalies.state, "pass");
});

test("checks are grouped by who recorded them, gate checks first", () => {
  const ready = snapshot();
  assert.deepEqual(
    qualityChecks(ready.validation, ready.split).map((c) => [c.id, c.group]),
    [
      ["gate", "gate"],
      ["anomalies", "gate"],
      ["drift", "gate"],
      ["leakage", "split"],
      ["class-balance", "split"],
      ["held-out", "split"],
      ["contract", "split"],
    ],
  );
});

test("anomalies fail, drift findings need attention, failed held-out checks fail", () => {
  const ready = snapshot({
    validation: { status: "failed", anomalies: [{}], drift: ["kind"] },
    split: {
      held_out_injection_methods: {
        status: "not_met",
        methods: ["a"],
        only_in_test: [],
      },
    },
  });
  const checks = Object.fromEntries(
    qualityChecks(ready.validation, ready.split).map((c) => [c.id, c]),
  );
  assert.equal(checks.gate.state, "fail");
  assert.equal(checks.anomalies.state, "fail");
  assert.equal(checks.drift.state, "warn");
  assert.equal(checks["held-out"].state, "fail");
  assert.match(checks["held-out"].detail, /not met/);
});

test("telemetry keeps present, missed step and no samples distinct, in that order", () => {
  const parts = telemetryParts({
    empty: 3,
    novel: 1,
    present: 5,
    no_overlap: 2,
  });
  assert.deepEqual(
    parts.map((part) => [part.id, part.label, part.value]),
    [
      ["present", "Present", 5],
      ["no_overlap", "Missed step", 2],
      ["empty", "No samples", 3],
      ["novel", "novel", 1],
    ],
  );
});

test("field health lists only incomplete fields, highest share first, and flags impossible counts", () => {
  const health = fieldHealth({ a: 10, b: 0, c: 40, d: 10, e: 150 }, 100);
  assert.deepEqual(
    health.rows.map((row) => row.field),
    ["e", "c", "a", "d"],
  );
  assert.equal(health.completeFields, 1);
  assert.deepEqual(health.rows[0], {
    field: "e",
    missing: 150,
    recorded: null,
    share: null,
  });
  assert.deepEqual(health.rows[1], {
    field: "c",
    missing: 40,
    recorded: 60,
    share: 0.4,
  });
});

test("splits read train, validation, test and keep an empty split at zero rows", () => {
  const parts = splitSummaries(snapshot().split.splits);
  assert.deepEqual(
    parts.map((part) => [part.name, part.rows]),
    [
      ["train", 40],
      ["validation", 0],
      ["test", 60],
    ],
  );
  assert.equal(
    parts[1].labels.every((label) => label.value === 0),
    true,
  );
});

test("page figures come from the snapshot; labelled rows are unknown when not counted", () => {
  const page = buildQualityPage(snapshot());
  assert.equal(page.releaseName, "steps-v9.9");
  assert.deepEqual(page.labelVersions, ["labels-v1+aa", "labels-v2+bb"]);
  assert.equal(page.telemetry.total, 100);
  assert.equal(page.telemetry.present, 50);
  assert.equal(page.labelledRows, 100);
  assert.equal(page.leakage.rows[0].aboveBound, false);
  assert.equal(page.leakage.rows[0].className, "Inference");
  assert.equal(
    page.labels.find((label) => label.id === "cpu_sandbox")?.value,
    0,
  );
  assert.equal(
    buildQualityPage(snapshot({ missing: { cpu_seconds: 1 } })).labelledRows,
    null,
  );
});

test("check tally counts unknown and warning checks as not passed", () => {
  const check = (id: string, state: "pass" | "warn" | "fail" | "unknown") => ({
    id,
    label: id,
    group: "gate" as const,
    state,
    detail: "",
  });
  assert.equal(checkTally([]), "none recorded");
  assert.equal(
    checkTally([check("a", "pass"), check("b", "pass")]),
    "all passed",
  );
  assert.equal(
    checkTally([check("a", "pass"), check("drift", "unknown")]),
    "1 not passed",
  );
  assert.equal(
    checkTally([check("a", "warn"), check("b", "fail"), check("c", "pass")]),
    "2 not passed",
  );
});
