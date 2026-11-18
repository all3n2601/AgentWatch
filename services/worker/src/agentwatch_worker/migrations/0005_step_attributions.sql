-- Additive B03 attribution history: one append-only row per terminal step, feature version and
-- serving configuration. A row's record is never rewritten; only evidence_changed may flip to
-- true when later evidence would produce a different feature row.
CREATE TABLE live_step_attributions (
    sequence BIGSERIAL UNIQUE NOT NULL,
    run_id UUID NOT NULL REFERENCES live_runs(run_id),
    step_id TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    serving_key TEXT NOT NULL,
    record JSONB NOT NULL,
    step_terminal_received_at TIMESTAMPTZ NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    evidence_changed BOOLEAN NOT NULL DEFAULT false,
    UNIQUE (run_id, step_id, feature_version, serving_key)
);
CREATE INDEX live_step_attributions_history ON live_step_attributions(run_id, sequence);
