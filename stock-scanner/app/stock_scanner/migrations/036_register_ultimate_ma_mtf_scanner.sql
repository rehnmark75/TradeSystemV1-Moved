-- Migration: 036_register_ultimate_ma_mtf_scanner.sql
-- Description: Register the configurable Ultimate MA MTF reclaim/cross scanner.

INSERT INTO stock_signal_scanners (
    scanner_name,
    description,
    min_score_threshold,
    max_signals_per_run
) VALUES (
    'ultimate_ma_mtf',
    'Buy close above green/rising MA, sell close below red/falling MA with ADX, ATR, and volume filters',
    55,
    50
)
ON CONFLICT (scanner_name) DO UPDATE SET
    description = EXCLUDED.description,
    min_score_threshold = EXCLUDED.min_score_threshold,
    max_signals_per_run = EXCLUDED.max_signals_per_run,
    is_active = TRUE,
    updated_at = NOW();
