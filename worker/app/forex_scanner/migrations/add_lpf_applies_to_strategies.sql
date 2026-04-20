-- Per-strategy scoping for Loss Prevention Filter rules.
--
-- Motivation: Apr 20 2026 MEAN_REVERSION launch showed 0% validation rate on
-- EURJPY and USDCHF because LPF rules calibrated against SMC_SIMPLE losses
-- (rules targeting ranging regime, low ADX, low volatility, low-RSI buys,
-- SMC confidence sweet-spots, and move-exhaustion patterns) penalised every
-- MR signal — those conditions are exactly what MR is designed to fire in.
--
-- Solution: rules get an optional applies_to_strategies JSON array. NULL means
-- the rule applies to all strategies (legacy behaviour). A populated array
-- (e.g. ["SMC_SIMPLE"]) restricts the rule to signals whose strategy matches.
-- Enforced in loss_prevention_filter.py:evaluate().

BEGIN;

ALTER TABLE loss_prevention_rules
    ADD COLUMN IF NOT EXISTS applies_to_strategies JSONB;

COMMENT ON COLUMN loss_prevention_rules.applies_to_strategies IS
    'NULL=applies to all strategies. JSON array (e.g. ["SMC_SIMPLE"]) '
    'restricts rule to listed strategies only. Enables per-strategy '
    'scoping — rules calibrated against SMC losses should not penalise '
    'MEAN_REVERSION signals that fire in different regimes.';

-- Scope SMC-calibrated rules to SMC_SIMPLE only.
-- Kept universal (NULL scope) intentionally:
--   friday_evening_block — weekend gap risk applies to every strategy
--   sell_near_support    — structural S/R risk applies to every strategy
--   asian_session        — session-level confidence boost, not SMC-specific
--   trending_aligned     — no-op for MR (won't fire in trending regime)
UPDATE loss_prevention_rules
SET applies_to_strategies = '["SMC_SIMPLE"]'::jsonb
WHERE rule_name IN (
    -- Category A: pair-specific losses calibrated from SMC trade outcomes
    'audjpy_bad_hours', 'audjpy_low_vol',
    'eurjpy_high_conf', 'eurjpy_ranging',
    'gbpusd_bias_misread', 'gbpusd_ny_sell',
    'usdchf_ranging', 'usdjpy_high_conf',
    -- Category B: confidence-range rules tuned on SMC confidence distribution
    'conf_quality_combo', 'extreme_confidence',
    'gbpusd_low_conf_sell', 'high_confidence',
    -- Category C: time-based losses; hour-profile differs per strategy
    'bad_hours', 'hour_11_ranging', 'sydney_ranging',
    -- Category D: regime/bias rules; MR is designed to fire in low_volatility/ranging
    'buy_bearish_bias', 'sell_bullish_bias',
    'low_volatility', 'trending_misaligned', 'usdjpy_breakout_block',
    -- Category E: technical indicators that conflict with MR's entry thesis
    -- (MR requires low ADX by design and buys into low-RSI extremes)
    'low_adx', 'low_mtf_confluence', 'low_rsi_buy', 'trending_low_efficiency',
    -- Category F: SMC-specific confidence-boost range
    'sweet_spot_conf',
    -- Category G: move-exhaustion rules; fading exhaustion is MR's core thesis
    'extreme_exhaustion', 'moderate_exhaustion', 'strong_exhaustion'
);

COMMIT;
