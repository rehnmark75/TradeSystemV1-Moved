-- ============================================================================
-- SMC_SIMPLE_V2 pair overrides
--
-- Completes demo-only wiring for SMC_SIMPLE_V2 by adding DB-backed pair
-- enablement and monitor/trading posture. Seed rows are monitor-only for every
-- demo FX scanner pair so the strategy can build alert/rejection history
-- without placing orders.
-- ============================================================================

CREATE TABLE IF NOT EXISTS smc_simple_v2_pair_overrides (
    id SERIAL PRIMARY KEY,
    config_set VARCHAR(20) NOT NULL DEFAULT 'demo',
    epic VARCHAR(50) NOT NULL,
    pair_name VARCHAR(20),

    is_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    is_traded BOOLEAN NOT NULL DEFAULT FALSE,
    monitor_only BOOLEAN NOT NULL DEFAULT TRUE,

    structure_lookback_bars INTEGER,
    entry_lookback_bars INTEGER,
    retest_tolerance_pips DOUBLE PRECISION,
    sweep_tolerance_pips DOUBLE PRECISION,
    min_break_pips DOUBLE PRECISION,
    min_rejection_wick_ratio DOUBLE PRECISION,
    max_rejection_body_ratio DOUBLE PRECISION,
    min_confirm_body_ratio DOUBLE PRECISION,

    fixed_stop_loss_pips DOUBLE PRECISION,
    fixed_take_profit_pips DOUBLE PRECISION,
    directions VARCHAR(20),
    entry_models TEXT,
    min_signal_gap_minutes INTEGER,

    adx_min DOUBLE PRECISION,
    adx_max DOUBLE PRECISION,
    atr_percentile_min DOUBLE PRECISION,
    atr_percentile_max DOUBLE PRECISION,
    bb_width_percentile_min DOUBLE PRECISION,
    bb_width_percentile_max DOUBLE PRECISION,
    efficiency_ratio_min DOUBLE PRECISION,
    ema200_mode VARCHAR(20),
    macd_mode VARCHAR(20),
    allowed_hours_utc VARCHAR(80),
    base_confidence DOUBLE PRECISION,

    parameter_overrides JSONB NOT NULL DEFAULT '{}'::jsonb,
    notes TEXT,
    updated_by VARCHAR(100),
    change_reason TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(),

    UNIQUE(config_set, epic)
);

CREATE INDEX IF NOT EXISTS idx_smc_simple_v2_pair_overrides_epic
    ON smc_simple_v2_pair_overrides(epic);

CREATE INDEX IF NOT EXISTS idx_smc_simple_v2_pair_overrides_enabled
    ON smc_simple_v2_pair_overrides(config_set, is_enabled, monitor_only);

DROP TRIGGER IF EXISTS update_smc_simple_v2_pair_overrides_updated_at
    ON smc_simple_v2_pair_overrides;

CREATE TRIGGER update_smc_simple_v2_pair_overrides_updated_at
    BEFORE UPDATE ON smc_simple_v2_pair_overrides
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

WITH seed(epic, pair_name) AS (
    VALUES
        ('CS.D.EURUSD.CEEM.IP', 'EURUSD'),
        ('CS.D.GBPUSD.MINI.IP', 'GBPUSD'),
        ('CS.D.USDJPY.MINI.IP', 'USDJPY'),
        ('CS.D.AUDUSD.MINI.IP', 'AUDUSD'),
        ('CS.D.USDCHF.MINI.IP', 'USDCHF'),
        ('CS.D.USDCAD.MINI.IP', 'USDCAD'),
        ('CS.D.NZDUSD.MINI.IP', 'NZDUSD'),
        ('CS.D.EURJPY.MINI.IP', 'EURJPY'),
        ('CS.D.AUDJPY.MINI.IP', 'AUDJPY')
)
INSERT INTO smc_simple_v2_pair_overrides (
    config_set,
    epic,
    pair_name,
    is_enabled,
    is_traded,
    monitor_only,
    structure_lookback_bars,
    entry_lookback_bars,
    retest_tolerance_pips,
    sweep_tolerance_pips,
    min_break_pips,
    min_rejection_wick_ratio,
    max_rejection_body_ratio,
    min_confirm_body_ratio,
    fixed_stop_loss_pips,
    fixed_take_profit_pips,
    directions,
    entry_models,
    min_signal_gap_minutes,
    allowed_hours_utc,
    base_confidence,
    parameter_overrides,
    notes,
    updated_by,
    change_reason
)
SELECT
    'demo',
    seed.epic,
    seed.pair_name,
    TRUE,
    FALSE,
    TRUE,
    12,
    25,
    1.5,
    0.3,
    0.2,
    0.45,
    0.65,
    0.35,
    5.0,
    6.0,
    'BULL,BEAR',
    'REJECTION_BREAK,LIQUIDITY_SWEEP_RECLAIM,ENGULFING_CONTINUATION,BROKEN_LEVEL_RETEST',
    60,
    '7,8,9,10,11,12,13,14,15,16,17,18,19,20',
    0.60,
    jsonb_build_object(
        'monitoring_experiment', 'jun2026_smc_simple_v2_all_fx_monitor',
        'seed_basis', 'Complete SMC_SIMPLE_V2 wiring: all FX scanner pairs monitor-only; both directions; all existing entry models'
    ),
    'SMC_SIMPLE_V2 monitor-only forward-test row',
    'codex',
    'Enable SMC_SIMPLE_V2 pair monitoring without trade execution'
FROM seed
ON CONFLICT (config_set, epic) DO UPDATE
SET
    pair_name = EXCLUDED.pair_name,
    is_enabled = TRUE,
    is_traded = FALSE,
    monitor_only = TRUE,
    structure_lookback_bars = EXCLUDED.structure_lookback_bars,
    entry_lookback_bars = EXCLUDED.entry_lookback_bars,
    retest_tolerance_pips = EXCLUDED.retest_tolerance_pips,
    sweep_tolerance_pips = EXCLUDED.sweep_tolerance_pips,
    min_break_pips = EXCLUDED.min_break_pips,
    min_rejection_wick_ratio = EXCLUDED.min_rejection_wick_ratio,
    max_rejection_body_ratio = EXCLUDED.max_rejection_body_ratio,
    min_confirm_body_ratio = EXCLUDED.min_confirm_body_ratio,
    fixed_stop_loss_pips = EXCLUDED.fixed_stop_loss_pips,
    fixed_take_profit_pips = EXCLUDED.fixed_take_profit_pips,
    directions = EXCLUDED.directions,
    entry_models = EXCLUDED.entry_models,
    min_signal_gap_minutes = EXCLUDED.min_signal_gap_minutes,
    allowed_hours_utc = EXCLUDED.allowed_hours_utc,
    base_confidence = EXCLUDED.base_confidence,
    parameter_overrides =
        COALESCE(smc_simple_v2_pair_overrides.parameter_overrides, '{}'::jsonb)
        || EXCLUDED.parameter_overrides,
    notes = EXCLUDED.notes,
    updated_by = 'codex',
    change_reason = 'Enable SMC_SIMPLE_V2 pair monitoring without trade execution';

DELETE FROM smc_simple_v2_pair_overrides
WHERE config_set <> 'demo';

UPDATE scanner_global_config
SET enabled_strategies = COALESCE((
    SELECT jsonb_agg(strategy_name ORDER BY strategy_name)
    FROM jsonb_array_elements_text(enabled_strategies) AS strategy(strategy_name)
    WHERE strategy_name <> 'SMC_SIMPLE_V2'
), '[]'::jsonb)
WHERE config_set <> 'demo'
  AND enabled_strategies @> '["SMC_SIMPLE_V2"]'::jsonb;

SELECT
    config_set,
    epic,
    pair_name,
    is_enabled,
    is_traded,
    monitor_only,
    directions,
    entry_models,
    allowed_hours_utc
FROM smc_simple_v2_pair_overrides
ORDER BY config_set, epic;
