-- Migration: 035_register_squeeze_momentum_scanner.sql
-- Description: Register the BB/KC squeeze release momentum scanner.

INSERT INTO stock_signal_scanners (
    scanner_name,
    description,
    min_score_threshold,
    max_signals_per_run
) VALUES (
    'squeeze_momentum',
    'BB/KC squeeze release with EMA50, ADX, ATR, and volume filters',
    55,
    50
)
ON CONFLICT (scanner_name) DO UPDATE SET
    description = EXCLUDED.description,
    min_score_threshold = EXCLUDED.min_score_threshold,
    max_signals_per_run = EXCLUDED.max_signals_per_run,
    is_active = TRUE,
    updated_at = NOW();
