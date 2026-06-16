-- Migration: 037_register_regime_adaptive_composite_scanner.sql
-- Description: Register the long-only regime-adaptive composite scanner.

INSERT INTO stock_signal_scanners (
    scanner_name,
    description,
    min_score_threshold,
    max_signals_per_run
) VALUES (
    'regime_adaptive_composite',
    'Long-only adaptive scanner for trend, compression, and ranging regimes',
    62,
    30
)
ON CONFLICT (scanner_name) DO UPDATE SET
    description = EXCLUDED.description,
    min_score_threshold = EXCLUDED.min_score_threshold,
    max_signals_per_run = EXCLUDED.max_signals_per_run,
    is_active = TRUE,
    updated_at = NOW();
