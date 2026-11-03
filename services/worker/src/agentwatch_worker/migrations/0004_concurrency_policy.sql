-- Additive O02 policy history. Version 0 is the synthesized baseline and has no row, so
-- existing databases need no backfill; links therefore carry no FK on (target, version).
CREATE TABLE concurrency_policy_versions (
    target TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    action TEXT NOT NULL CHECK (action IN ('set', 'disable', 'restore')),
    limits JSONB NOT NULL,
    restored_from INTEGER CHECK (restored_from >= 0 AND restored_from < version),
    operator TEXT NOT NULL,
    reason TEXT NOT NULL,
    request_id UUID NOT NULL,
    request JSONB NOT NULL,
    controlled_services JSONB NOT NULL,
    enforcement_scope TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (target, version),
    UNIQUE (target, request_id)
);
CREATE TABLE live_run_policy_links (
    run_id UUID NOT NULL REFERENCES live_runs(run_id),
    target TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version >= 0),
    linked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, target, version)
);
