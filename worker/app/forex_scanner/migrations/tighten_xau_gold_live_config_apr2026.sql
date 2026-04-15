-- Tighten XAU_GOLD live configuration for gold-specific execution quality.
-- Keeps HTF slow EMA at 100 per operator preference.

BEGIN;

UPDATE xau_gold_global_config
SET parameter_value = CASE parameter_name
    WHEN 'ema_slow_period' THEN '100'
    WHEN 'swing_lookback' THEN '20'
    WHEN 'min_confidence' THEN '0.58'
    WHEN 'adx_ranging_threshold' THEN '20'
    WHEN 'atr_expansion_pct' THEN '85'
    WHEN 'asian_allowed' THEN 'false'
    WHEN 'bos_displacement_atr_mult' THEN '1.2'
    WHEN 'fib_pullback_min' THEN '0.382'
    WHEN 'fib_pullback_max' THEN '0.618'
    WHEN 'bos_expiry_hours' THEN '12'
    WHEN 'bos_search_bars' THEN '24'
    WHEN 'entry_check_bars' THEN '12'
    WHEN 'macd_filter_enabled' THEN 'true'
    WHEN 'require_ob_or_fvg' THEN 'true'
    ELSE parameter_value
END,
updated_at = NOW()
WHERE parameter_name IN (
    'ema_slow_period',
    'swing_lookback',
    'min_confidence',
    'adx_ranging_threshold',
    'atr_expansion_pct',
    'asian_allowed',
    'bos_displacement_atr_mult',
    'fib_pullback_min',
    'fib_pullback_max',
    'bos_expiry_hours',
    'bos_search_bars',
    'entry_check_bars',
    'macd_filter_enabled',
    'require_ob_or_fvg'
);

UPDATE trailing_pair_config
SET early_breakeven_trigger_points = 25,
    early_breakeven_buffer_points = 3,
    break_even_trigger_points = 30,
    stage1_trigger_points = 50,
    stage1_lock_points = 25,
    stage2_trigger_points = 80,
    stage2_lock_points = 50,
    stage3_trigger_points = 110,
    stage3_atr_multiplier = 1.50,
    stage3_min_distance = 30,
    min_trail_distance = 30,
    updated_by = 'codex',
    change_reason = 'Tighten XAU_GOLD entries and restore gold-specific trailing calibration',
    updated_at = NOW()
WHERE epic = 'CS.D.CFEGOLD.CEE.IP';

COMMIT;
