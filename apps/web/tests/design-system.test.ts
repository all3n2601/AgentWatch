import assert from "node:assert/strict";
import test from "node:test";
import {
  formatMeasurement,
  formatQuantity,
  MISSING_VALUE,
} from "../lib/measurement-format";
import { overviewScene, sceneManifestSchema } from "../lib/overview-scene";
import { donutArc, layoutSegments, relativeLengths } from "../lib/segments";
import { deltaTone, slowdownTone, statusTone } from "../lib/tone";

test("formatMeasurement keeps a measured zero distinct from a missing reading", () => {
  assert.equal(formatMeasurement(0, "ms").text, "0 ms");
  assert.equal(formatMeasurement(0, "ms").hasValue, true);

  for (const missing of [
    null,
    undefined,
    Number.NaN,
    Number.POSITIVE_INFINITY,
  ]) {
    const formatted = formatMeasurement(missing, "ms");
    assert.equal(formatted.text, MISSING_VALUE);
    assert.equal(formatted.state, "unavailable");
    assert.equal(formatted.hasValue, false);
  }
});

test("formatMeasurement hides values that are declared unavailable or not instrumented", () => {
  const notInstrumented = formatMeasurement(12, "s", "not_instrumented");
  assert.equal(notInstrumented.text, MISSING_VALUE);
  assert.equal(notInstrumented.accessibleText, "Not instrumented");
  assert.equal(formatMeasurement(12, "s", "unavailable").text, MISSING_VALUE);
});

test("formatMeasurement marks estimates visibly and for screen readers", () => {
  const estimated = formatMeasurement(2.184, "s", "estimated");
  assert.equal(estimated.text, "~2.18 s");
  assert.equal(estimated.accessibleText, "Estimated 2.18 s");
});

test("formatQuantity uses each unit's precision", () => {
  assert.equal(formatQuantity(68.94, "ms"), "69 ms");
  assert.equal(formatQuantity(27.239, "s"), "27.24 s");
  assert.equal(formatQuantity(68.44, "percent"), "68.4%");
  assert.equal(formatQuantity(11.42, "ratio"), "11.4×");
  assert.equal(formatQuantity(5603, "count"), "5,603");
});

test("statusTone maps known statuses and treats unknown ones as informational", () => {
  assert.equal(statusTone("Passed"), "success");
  assert.equal(statusTone(" failed "), "danger");
  assert.equal(statusTone("Degraded"), "caution");
  assert.equal(statusTone("Attention"), "warning");
  assert.equal(statusTone("something-new"), "info");
});

test("slowdownTone grades observed ratios at the declared boundaries", () => {
  assert.equal(slowdownTone(0.8), "success");
  assert.equal(slowdownTone(1), "success");
  assert.equal(slowdownTone(1.05), "info");
  assert.equal(slowdownTone(1.1), "warning");
  assert.equal(slowdownTone(1.5), "caution");
  assert.equal(slowdownTone(8), "danger");
  assert.equal(slowdownTone(Number.NaN), null);
  assert.equal(slowdownTone(-1), null);
});

test("neutral deltas take no tone, so a faster candidate never reads as better", () => {
  // Without `neutral`, a ratio below 1 would be graded "success".
  assert.equal(deltaTone(0.5, false), "success");
  for (const ratio of [0.5, 1, 1.2, 12])
    assert.equal(deltaTone(ratio, true), null);
  assert.equal(deltaTone(null, false), null);
  assert.equal(deltaTone(Number.NaN, false), null);
});

test("layoutSegments computes shares and offsets that sum to the whole", () => {
  const layout = layoutSegments([
    { id: "work", value: 8.48 },
    { id: "wait", value: 18.76 },
  ]);
  assert.equal(layout.status, "ok");
  if (layout.status !== "ok") return;
  assert.ok(Math.abs(layout.total - 27.24) < 1e-9);
  assert.ok(Math.abs(layout.segments[0].share - 0.3113) < 1e-4);
  assert.equal(layout.segments[1].offset, layout.segments[0].share);
  const sum = layout.segments.reduce((acc, segment) => acc + segment.share, 0);
  assert.ok(Math.abs(sum - 1) < 1e-12);
});

test("layoutSegments rejects invalid readings and reports an all-zero total as empty", () => {
  assert.equal(layoutSegments([{ id: "a", value: -1 }]).status, "invalid");
  assert.equal(
    layoutSegments([{ id: "a", value: Number.NaN }]).status,
    "invalid",
  );
  assert.equal(
    layoutSegments([
      { id: "a", value: 1 },
      { id: "a", value: 2 },
    ]).status,
    "invalid",
  );
  assert.equal(layoutSegments([{ id: "a", value: 0 }]).status, "empty");
  assert.equal(layoutSegments([]).status, "empty");
});

test("relativeLengths scales to the largest value and nulls invalid entries", () => {
  assert.deepEqual(
    relativeLengths([2847, 2158, 598]).map((n) => n?.toFixed(3)),
    ["1.000", "0.758", "0.210"],
  );
  assert.deepEqual(relativeLengths([0, 0]), [0, 0]);
  assert.deepEqual(relativeLengths([4, Number.NaN, -2]), [1, null, null]);
});

test("donutArc starts at twelve o'clock and never draws a negative arc", () => {
  const circumference = 100;
  assert.deepEqual(donutArc(0.25, 0, circumference), {
    dashArray: "25 75",
    dashOffset: 25,
  });
  assert.equal(donutArc(0.25, 0.5, circumference).dashOffset, -25);
  assert.equal(donutArc(0.001, 0, circumference, 2).dashArray, "0 100");
});

test("the committed scene manifest is valid and keeps anchors inside the image", () => {
  for (const point of Object.values(overviewScene.anchors)) {
    assert.ok(point.x >= 0 && point.x <= 1 && point.y >= 0 && point.y <= 1);
  }
  const outside = structuredClone(overviewScene);
  outside.anchors.model.y = -0.01;
  assert.equal(sceneManifestSchema.safeParse(outside).success, false);
});

test("each stage has a component model whose GLB keeps the spin and glow conventions", async () => {
  const { readFile } = await import("node:fs/promises");
  const { componentModels, SCENE_ANCHOR_IDS } =
    await import("../lib/overview-scene");
  const gltfJson = async (url: string) => {
    const bytes = await readFile(new URL(`../public${url}`, import.meta.url));
    // GLB: 12-byte header, then the JSON chunk's length and type, then its text.
    const length = bytes.readUInt32LE(12);
    return JSON.parse(bytes.subarray(20, 20 + length).toString("utf8")) as {
      nodes: { name: string; extras?: Record<string, unknown> }[];
      materials: { name: string }[];
    };
  };
  for (const stage of SCENE_ANCHOR_IDS) {
    const entry = componentModels[stage];
    assert.ok(entry.packetPaths.length > 0, `${stage} has packet paths`);
    const gltf = await gltfJson(entry.model);
    assert.ok(
      gltf.materials.some((m) => m.name === `aw_glow_${stage}`),
      `${stage} model uses its own glow material`,
    );
    for (const node of gltf.nodes.filter((n) => n.name.startsWith("spin_")))
      assert.ok(
        Number(node.extras?.aw_spin) > 0,
        `${node.name} has a spin speed`,
      );
  }
  // The GPU and CPU models must carry spinning fans.
  for (const stage of ["model", "cpuTool"] as const) {
    const gltf = await gltfJson(componentModels[stage].model);
    assert.ok(
      gltf.nodes.some(
        (n) => n.name.includes("fan") && n.name.startsWith("spin_"),
      ),
    );
  }
});

test("small durations keep their precision instead of reading as zero", async () => {
  const { formatQuantity: q, formatSeconds } =
    await import("../lib/measurement-format");
  assert.equal(q(0.3, "ms"), "0.30 ms");
  assert.equal(q(0.004, "ms"), "< 0.01 ms");
  assert.equal(q(0, "ms"), "0 ms");
  assert.equal(q(4.26, "ms"), "4.3 ms");
  assert.equal(formatSeconds(2.5), "2.50 s");
  assert.equal(formatSeconds(0.0042), "4.2 ms");
  assert.equal(formatSeconds(0.00023), "230 µs");
  assert.equal(formatSeconds(0), "0 ms");
});
