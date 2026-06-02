-- RANGE_FADE: make the confidence-reject floor a real, DB-tunable config field (Jun 2026)
--
-- Background: commit fa97512 (May 31 2026) added a confidence REJECT gate to
-- range_fade_strategy.py read via getattr(cfg, "min_reject_confidence", 0.60). Because
-- min_reject_confidence was NOT a RangeFadeConfig field and NOT a DB column, two things
-- were true: (1) the floor was a hardcoded 0.60 that could not be tuned without a code
-- change, and (2) bt.py --override min_reject_confidence=X silently no-opped, because
-- apply_config_overrides() skips any key that is not already an attribute of the config
-- (the `if not hasattr(cfg, key): continue` guard) — so a 90d EURUSD sweep returned
-- identical cells for 0.0/0.55/0.60/0.65/0.99 and could not validate the value.
--
-- This migration adds the column so the existing generic DB-load loop in
-- range_fade_config_service._load_from_database() (for key in row.keys(): if
-- hasattr(config, key): setattr(...)) populates the dataclass field. Companion code:
--   * range_fade_config_service.py — new `min_reject_confidence` dataclass field (0.60)
--     + get_pair_min_reject_confidence() per-pair getter
--   * range_fade_strategy.py:354   — now reads cfg.get_pair_min_reject_confidence(epic)
-- Per-pair tuning is supported via range_fade_pair_overrides.parameter_overrides JSONB.
--
-- BEHAVIOR-PRESERVING: default and seeded value are 0.60 — the same value the hardcoded
-- getattr default already enforced. No live behavior change; this only makes the floor
-- tunable + backtest-overridable. 0.0 = gate off.

ALTER TABLE range_fade_global_config
  ADD COLUMN IF NOT EXISTS min_reject_confidence DOUBLE PRECISION DEFAULT 0.60;

-- Pin the active rows to the validated 0.60 floor (both config_sets; live EURUSD is
-- monitor-only but keep the floor consistent so a future live-enable inherits it).
UPDATE range_fade_global_config
  SET min_reject_confidence = 0.60
  WHERE is_active = TRUE AND min_reject_confidence IS NULL;
