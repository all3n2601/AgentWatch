CREATE TABLE runs (
    sequence BIGSERIAL UNIQUE NOT NULL,
    run_id UUID PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('coding', 'cpu')),
    result JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE spans (
    run_id UUID NOT NULL REFERENCES runs(run_id),
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (run_id, trace_id, span_id)
);
CREATE TABLE samples (
    run_id UUID NOT NULL REFERENCES runs(run_id),
    timestamp_ns BIGINT NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (run_id, timestamp_ns)
);
CREATE INDEX runs_history ON runs(sequence DESC);
