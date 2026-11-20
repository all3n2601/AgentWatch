-- Additive: whether the evidence of a stored attribution row was re-checked after its run
-- closed. Rows written before the run closed start unaudited and are audited by the first pass
-- on the closed run, so a failed close-time pass is retried by later reads instead of lost.
-- Existing rows are unaudited and are audited once after the upgrade.
ALTER TABLE live_step_attributions ADD COLUMN evidence_audited BOOLEAN NOT NULL DEFAULT false;
