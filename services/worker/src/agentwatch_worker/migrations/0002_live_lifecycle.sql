-- Additive live history: existing completed-run evidence is never rewritten.
CREATE TABLE live_runs (
    sequence BIGSERIAL UNIQUE NOT NULL,
    run_id UUID PRIMARY KEY,
    request_id UUID UNIQUE NOT NULL,
    registration JSONB NOT NULL,
    state TEXT NOT NULL DEFAULT 'queued',
    revision INTEGER NOT NULL DEFAULT 0,
    reason TEXT,
    cleanup_requested BOOLEAN NOT NULL DEFAULT false,
    unfinished_steps JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE live_run_changes (
    change_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES live_runs(run_id),
    payload JSONB NOT NULL,
    revision INTEGER NOT NULL,
    UNIQUE (run_id, revision)
);
CREATE TABLE live_records (
    sequence BIGSERIAL UNIQUE NOT NULL,
    record_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES live_runs(run_id),
    record_type TEXT NOT NULL,
    step_id TEXT,
    step_revision INTEGER,
    sample_key JSONB,
    payload JSONB NOT NULL,
    disposition TEXT NOT NULL CHECK (disposition IN ('pending', 'applied', 'expired')),
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, step_id, step_revision),
    UNIQUE (run_id, sample_key)
);
CREATE INDEX live_records_history ON live_records(run_id, sequence);
CREATE INDEX live_records_steps ON live_records(run_id, step_id, step_revision);
CREATE INDEX live_pending ON live_records(run_id, received_at) WHERE disposition = 'pending';
