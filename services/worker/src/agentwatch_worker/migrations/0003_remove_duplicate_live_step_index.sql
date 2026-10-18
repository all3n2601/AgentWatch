-- UNIQUE (run_id, step_id, step_revision) already supplies this exact index ordering.
-- Preserve that constraint and all history; do not rewrite the applied 0002 migration.
DROP INDEX IF EXISTS live_records_steps;
