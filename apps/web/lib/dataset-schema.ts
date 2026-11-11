import { z } from "zod";
import type { DatasetRun } from "./dataset-runs";

const counts = z.record(z.string(), z.number().int().nonnegative());

/** The slice of a release `manifest.json` the dashboard charts. */
export const manifestSchema = z.object({
  dataset_version: z.string(),
  schema_version: z.string(),
  release_format: z.number().int(),
  rows: z.number().int().positive(),
  runs: z.number().int().nonnegative(),
  tasks: z.number().int().nonnegative(),
  kind: counts,
  condition: counts,
  label_resource: counts,
  label_basis: counts,
  telemetry_status: counts,
  missing: counts,
  label_version: z.string(),
});

const splitPart = z.object({
  rows: z.number().int().nonnegative(),
  runs: z.number().int().nonnegative(),
  groups: z.number().int().nonnegative(),
  label_resource: counts,
  condition: counts,
  injection_method: counts,
});

const aucFeature = z.object({
  feature: z.string(),
  auc: z.number().min(0).max(1),
});

/** The slice of a `split-v2` assignment file the dashboard charts. */
export const splitSchema = z.object({
  split_id: z.string(),
  split_method: z.string(),
  contract: z.object({
    version: z.string(),
    status: z.string(),
    group_key: z.string(),
  }),
  class_balance: z.object({
    status: z.string(),
    per_class: counts,
    clean: z.number().int(),
    mixed: z.number().int(),
    minimums: z.object({
      clean: z.number().int(),
      mixed: z.number().int(),
      per_class: z.number().int(),
    }),
  }),
  splits: z.record(z.string(), splitPart),
  design_encoding: z.object({
    status: z.string(),
    rows: z.number().int(),
    features: z.array(aucFeature),
  }),
  leakage: z.object({
    status: z.string(),
    max_single_feature_auc: z.number(),
    features: z.array(aucFeature.extend({ class: z.string() })),
  }),
  held_out_injection_methods: z.object({
    status: z.string(),
    methods: z.array(z.string()),
    only_in_test: z.array(z.string()),
  }),
});

/** The slice of a quality-gate `result.json` the dashboard shows. */
export const validationSchema = z.object({
  status: z.string(),
  validated_at: z.string(),
  examples: z.number().int(),
  anomalies: z.array(z.unknown()),
  drift: z.array(z.unknown()),
  gate_version: z.string(),
  tfdv_version: z.string(),
});

export type Manifest = z.infer<typeof manifestSchema>;
export type SplitFile = z.infer<typeof splitSchema>;
export type Validation = z.infer<typeof validationSchema>;

export type DatasetSnapshot =
  | {
      status: "ready";
      manifest: Manifest;
      split: SplitFile;
      validation: Validation;
      /** One summary per run in the release, newest first. */
      runs: DatasetRun[];
    }
  | { status: "unavailable"; reason: string };
