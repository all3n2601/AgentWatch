import assert from "node:assert/strict";
import test from "node:test";
import { attributionPageSchema } from "../lib/attribution-schema";
import {
  attributionRow,
  comparisonDisplay,
  comparisonGroups,
  hasMixedEvidence,
  probabilityEntries,
  rulesCoverageNote,
  rulesVerdictLabel,
  secondsReading,
  type AttributionRow,
  type MalformedRow,
  type ValidRow,
} from "../lib/attribution-view";
import {
  attributionItem,
  attributionPage,
  attributionRecord,
  CONFIDENT_CPU,
  CONFIDENT_INFERENCE,
  CURRENT_KEY,
  EARLIER_KEY,
  featureRow,
  scored,
  UNCERTAIN,
  unscored,
} from "./attribution-fixtures";
import { OTHER_RUN_ID, RUN_ID } from "./live-fixtures";

type Item = Parameters<typeof attributionRow>[0];

const row = (item: Item, runId = RUN_ID): AttributionRow =>
  attributionRow(item, runId, CURRENT_KEY);

function valid(item: Item): ValidRow {
  const result = row(item);
  assert.equal(result.kind, "valid", JSON.stringify(result));
  return result as ValidRow;
}

function malformed(item: Item): MalformedRow {
  const result = row(item);
  assert.equal(result.kind, "malformed");
  return result as MalformedRow;
}

const withModel = (model: Record<string, unknown>, extra = {}) =>
  attributionItem(1, attributionRecord(RUN_ID, "step-1", { model, ...extra }));

test("a page with one malformed row still parses, and only that row is malformed", () => {
  const page = attributionPageSchema.parse(
    attributionPage(RUN_ID, [
      attributionItem(1, attributionRecord(RUN_ID, "good")),
      { sequence: 2, record: { step_id: "bad" } },
    ]),
  );
  const rows = page.attributions.map((item) => row(item));
  assert.equal(rows[0].kind, "valid");
  assert.equal(rows[1].kind, "malformed");
  assert.equal(rows[1].stepId, "bad");
});

test("a page without a sequence on a row is rejected, since the cursor could not pass it", () => {
  const page = attributionPage(RUN_ID, [{ record: {} } as never]);
  assert.equal(attributionPageSchema.safeParse(page).success, false);
});

test("measured zero stays zero and missing readings stay null", () => {
  const feature = featureRow(RUN_ID, "step-1", {
    timings: {
      queue_seconds: 0,
      execution_seconds: 1.2,
      duration_seconds: 1.2,
      cpu_seconds: null,
    },
    coverage: null,
    throttledUsec: 0,
  });
  const result = valid(
    attributionItem(
      1,
      attributionRecord(RUN_ID, "step-1", {
        feature,
        model: scored(CONFIDENT_CPU, { excess: 0 }),
        verdict: "cpu_sandbox",
      }),
    ),
  );
  assert.equal(result.feature.timings.queue_seconds, 0);
  assert.equal(result.feature.timings.cpu_seconds, null);
  assert.equal(result.feature.telemetry.coverage, null);
  assert.equal(result.feature.features.cgroup_throttled_usec_delta, 0);
  assert.equal(result.model.excess_fraction_estimate, 0);
  assert.deepEqual(secondsReading(0), { value: 0, unit: "ms" });
  assert.deepEqual(secondsReading(null), { value: null, unit: "s" });
  assert.deepEqual(secondsReading(2.5), { value: 2.5, unit: "s" });
});

test("an abstention is its own state, never none, and is not counted as a comparison", () => {
  const result = valid(withModel(scored(UNCERTAIN), { verdict: "none" }));
  assert.equal(result.model.outcome, "abstained");
  assert.equal(result.model.predicted_class, null);
  assert.equal(result.comparison, "model_abstained");
  assert.equal(hasMixedEvidence(result), false);
});

test("confidence equal to the threshold is a prediction, as the worker serves it", () => {
  const probabilities = {
    none: 0.153,
    cpu_sandbox: 0.647,
    inference: 0.1,
    retrieval: 0.1,
  };
  const result = valid(
    withModel(scored(probabilities), { verdict: "cpu_sandbox" }),
  );
  assert.equal(result.model.outcome, "predicted");
  assert.equal(result.comparison, "agree");
});

test("rules and model naming different resources is mixed evidence; so is changed evidence", () => {
  const differ = valid(
    withModel(scored(CONFIDENT_INFERENCE), { verdict: "cpu_sandbox" }),
  );
  assert.equal(differ.comparison, "differ");
  assert.equal(hasMixedEvidence(differ), true);

  const changed = valid(
    attributionItem(1, attributionRecord(RUN_ID, "step-1"), {
      evidenceChanged: true,
    }),
  );
  assert.equal(changed.comparison, "agree");
  assert.equal(hasMixedEvidence(changed), true);
});

test("an unrecognised rules verdict keeps its raw string and is not compared", () => {
  const result = valid(
    withModel(scored(CONFIDENT_CPU), { verdict: "gpu_memory" }),
  );
  assert.equal(result.comparison, "verdict_unrecognised");
  assert.equal(rulesVerdictLabel("gpu_memory"), "Unrecognised: gpu_memory");
  assert.equal(rulesVerdictLabel("cpu_sandbox"), "CPU sandbox");
});

test("the rules-v1 coverage note applies to rules-v1 builds only", () => {
  assert.ok(rulesCoverageNote("rules-v1"));
  assert.ok(rulesCoverageNote("rules-v1+1bf54c19"));
  assert.equal(rulesCoverageNote("rules-v2+abcd1234"), null);
  assert.equal(rulesCoverageNote("rules-v10"), null);
});

test("rows without a model output are rules only, with their reasons kept", () => {
  for (const [model, stepState] of [
    [
      unscored("unavailable", {
        state: "not_configured",
        reason: "not_configured",
      }),
      "completed",
    ],
    [unscored("unavailable", { reason: "prediction_failed" }), "completed"],
    [unscored("not_scored_step_state"), "failed"],
  ] as const) {
    const result = valid(withModel(model, { stepState }));
    assert.equal(result.comparison, "no_model_output");
  }
});

test("contradictory rows are malformed, with a reason for each contradiction", () => {
  const cases: [string, Item][] = [
    [
      "abstained flag",
      withModel({ ...scored(CONFIDENT_CPU), abstained: true }),
    ],
    [
      "class not the top",
      withModel({ ...scored(CONFIDENT_CPU), predicted_class: "retrieval" }),
    ],
    [
      "abstention names a class",
      withModel({ ...scored(UNCERTAIN), predicted_class: "cpu_sandbox" }),
    ],
    [
      "predicted below threshold",
      withModel({
        ...scored(UNCERTAIN),
        outcome: "predicted",
        abstained: false,
        predicted_class: "cpu_sandbox",
      }),
    ],
    [
      "confidence missing",
      withModel({ ...scored(CONFIDENT_CPU), confidence: null }),
    ],
    [
      "unavailable with values",
      withModel({
        ...unscored("unavailable", { reason: "prediction_failed" }),
        confidence: 0.9,
      }),
    ],
    ["unavailable without reason", withModel(unscored("unavailable"))],
    ["completed step not scored", withModel(unscored("not_scored_step_state"))],
    [
      "failed step scored",
      withModel(scored(CONFIDENT_CPU), { stepState: "failed" }),
    ],
    [
      "not promoted label",
      withModel({ ...scored(CONFIDENT_CPU), not_promoted: false }),
    ],
    [
      "probabilities do not sum to 1",
      withModel(
        scored({ none: 0.2, cpu_sandbox: 0.9, inference: 0, retrieval: 0 }),
      ),
    ],
  ];
  for (const [name, item] of cases) {
    const result = malformed(item);
    assert.ok(result.reasons.length > 0, name);
    assert.equal(result.stepId, "step-1", name);
  }
});

test("a feature row that disagrees with its record is malformed", () => {
  const otherStep = attributionItem(
    1,
    attributionRecord(RUN_ID, "step-1", {
      feature: featureRow(RUN_ID, "step-2"),
    }),
  );
  assert.match(malformed(otherStep).reasons.join(" "), /another step/);

  const feature = featureRow(RUN_ID, "step-1");
  (feature.features as Record<string, unknown>).queue_seconds = 9;
  const timings = attributionItem(
    1,
    attributionRecord(RUN_ID, "step-1", { feature }),
  );
  assert.match(malformed(timings).reasons.join(" "), /Timing features/);

  const negative = featureRow(RUN_ID, "step-1", {
    timings: {
      queue_seconds: -1,
      execution_seconds: 1,
      duration_seconds: 1,
      cpu_seconds: 1,
    },
  });
  malformed(
    attributionItem(
      1,
      attributionRecord(RUN_ID, "step-1", { feature: negative }),
    ),
  );
});

test("a record of another run is malformed, never shown under the selected run", () => {
  const item = attributionItem(
    1,
    attributionRecord(OTHER_RUN_ID, "step-1", {
      feature: featureRow(OTHER_RUN_ID, "step-1"),
    }),
  );
  const result = row(item, RUN_ID);
  assert.equal(result.kind, "malformed");
});

test("comparison groups never combine serving keys, and keep each key's versions", () => {
  const rows = [
    row(
      attributionItem(
        1,
        attributionRecord(RUN_ID, "old-1", {
          verdict: "cpu_sandbox",
          model: scored(CONFIDENT_CPU, {
            trainingVersion: "training-v1+synthetic",
          }),
        }),
        { servingKey: EARLIER_KEY },
      ),
    ),
    row(
      attributionItem(
        2,
        attributionRecord(RUN_ID, "new-1", {
          verdict: "cpu_sandbox",
          model: scored(CONFIDENT_CPU),
        }),
      ),
    ),
    row(
      attributionItem(
        3,
        attributionRecord(RUN_ID, "new-2", {
          verdict: "cpu_sandbox",
          model: scored(CONFIDENT_INFERENCE),
        }),
      ),
    ),
    row(
      attributionItem(
        4,
        attributionRecord(RUN_ID, "new-3", { model: scored(UNCERTAIN) }),
      ),
    ),
    row(
      attributionItem(
        5,
        attributionRecord(RUN_ID, "new-4", {
          model: unscored("not_scored_step_state"),
          stepState: "cancelled",
        }),
      ),
    ),
    row({ sequence: 6 }),
  ];
  const groups = comparisonGroups(rows);
  assert.equal(groups.length, 2);
  const [current, earlier] = groups;
  assert.equal(current.current, true);
  assert.equal(current.rows, 4);
  assert.deepEqual(current.counts, {
    agree: 1,
    differ: 1,
    model_abstained: 1,
    no_model_output: 1,
    verdict_unrecognised: 0,
  });
  assert.equal(earlier.current, false);
  assert.equal(earlier.servingKey, EARLIER_KEY);
  assert.deepEqual(earlier.trainingVersions, ["training-v1+synthetic"]);
  assert.equal(earlier.counts.agree, 1);
});

test("probabilities are listed most probable first", () => {
  assert.deepEqual(
    probabilityEntries(CONFIDENT_INFERENCE).map((entry) => entry.id),
    ["inference", "none", "retrieval", "cpu_sandbox"],
  );
  assert.deepEqual(probabilityEntries(null), []);
});

/** A feature row edited after it was built, to break one invariant at a time. */
function editedFeature(
  edit: (row: {
    telemetry: Record<string, unknown>;
    features: Record<string, unknown>;
  }) => void,
) {
  const feature = featureRow(RUN_ID, "step-1") as unknown as {
    telemetry: Record<string, unknown>;
    features: Record<string, unknown>;
  };
  edit(feature);
  return attributionItem(
    1,
    attributionRecord(RUN_ID, "step-1", {
      feature: feature as unknown as Record<string, unknown>,
    }),
  );
}

test("present telemetry must have samples, and only present telemetry has them", () => {
  const presentWithout = editedFeature(({ telemetry, features }) => {
    telemetry.sample_count = 0;
    features.telemetry_sample_count = 0;
    telemetry.observed_interval_seconds = null;
  });
  assert.match(malformed(presentWithout).reasons.join(" "), /sample count/);

  const emptyWith = featureRow(RUN_ID, "step-1", {
    status: "empty",
    coverage: null,
  });
  assert.match(
    malformed(
      attributionItem(
        1,
        attributionRecord(RUN_ID, "step-1", { feature: emptyWith }),
      ),
    ).reasons.join(" "),
    /sample count/,
  );
});

test("coverage must be null for missing, empty and unaligned telemetry", () => {
  for (const status of ["missing", "empty", "unaligned"]) {
    const feature = featureRow(RUN_ID, "step-1", {
      status,
      sampleCount: 0,
      coverage: 0,
      pressure: null,
      throttledUsec: null,
    });
    const result = malformed(
      attributionItem(1, attributionRecord(RUN_ID, "step-1", { feature })),
    );
    assert.match(result.reasons.join(" "), /coverage value/, status);
  }
  // no_overlap has a sampler stream, so its coverage may be a measured 0.
  const noOverlap = featureRow(RUN_ID, "step-1", {
    status: "no_overlap",
    sampleCount: 0,
    coverage: 0,
    pressure: null,
    throttledUsec: null,
  });
  valid(
    attributionItem(
      1,
      attributionRecord(RUN_ID, "step-1", { feature: noOverlap }),
    ),
  );
});

test("an observed sampling interval needs at least two samples", () => {
  const one = featureRow(RUN_ID, "step-1", { sampleCount: 1 });
  (one.telemetry as Record<string, unknown>).observed_interval_seconds = 0.25;
  assert.match(
    malformed(
      attributionItem(1, attributionRecord(RUN_ID, "step-1", { feature: one })),
    ).reasons.join(" "),
    /at least two samples/,
  );
  valid(
    attributionItem(
      1,
      attributionRecord(RUN_ID, "step-1", {
        feature: featureRow(RUN_ID, "step-1", { sampleCount: 2 }),
      }),
    ),
  );
});

test("queue_fraction must equal queue_seconds / duration_seconds within 1e-6", () => {
  const off = editedFeature(({ features }) => {
    features.queue_fraction = (features.queue_fraction as number) + 1e-5;
  });
  assert.match(malformed(off).reasons.join(" "), /queue_fraction/);

  const close = editedFeature(({ features }) => {
    features.queue_fraction = (features.queue_fraction as number) + 1e-7;
  });
  valid(close);

  const missingFraction = editedFeature(({ features }) => {
    features.queue_fraction = null;
  });
  assert.match(malformed(missingFraction).reasons.join(" "), /queue_fraction/);
});

test("queue_fraction is null when a timing is missing or the duration is zero", () => {
  const cases = [
    {
      queue_seconds: null,
      execution_seconds: 1,
      duration_seconds: 1,
      cpu_seconds: 1,
    },
    {
      queue_seconds: 0.5,
      execution_seconds: 1,
      duration_seconds: null,
      cpu_seconds: 1,
    },
    {
      queue_seconds: 0,
      execution_seconds: 0,
      duration_seconds: 0,
      cpu_seconds: 0,
    },
  ];
  for (const timings of cases) {
    const feature = featureRow(RUN_ID, "step-1", { timings });
    // The fixture derives a null fraction for each case, as the worker does.
    valid(attributionItem(1, attributionRecord(RUN_ID, "step-1", { feature })));
    (feature.features as Record<string, unknown>).queue_fraction = 0;
    assert.match(
      malformed(
        attributionItem(1, attributionRecord(RUN_ID, "step-1", { feature })),
      ).reasons.join(" "),
      /queue_fraction/,
      JSON.stringify(timings),
    );
  }
});

test("rules and model both saying none is agreement, labelled Agree", () => {
  const result = valid(
    withModel(
      scored({
        none: 0.9,
        cpu_sandbox: 0.04,
        inference: 0.03,
        retrieval: 0.03,
      }),
    ),
  );
  assert.equal(result.rules.verdict, "none");
  assert.equal(result.comparison, "agree");
  assert.equal(comparisonDisplay("agree").label, "Agree");
});
