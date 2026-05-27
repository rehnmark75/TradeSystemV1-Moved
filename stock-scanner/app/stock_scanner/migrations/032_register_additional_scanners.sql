-- Migration: 032_register_additional_scanners.sql
-- Description: Register additional stock scanners enabled in ScannerManager.

INSERT INTO stock_signal_scanners (scanner_name, description, min_score_threshold, max_signals_per_run) VALUES
    ('pocket_pivot', 'Pocket Pivot institutional accumulation within an uptrend', 65, 30),
    ('earnings_drift', 'Post-earnings drift continuation after a held catalyst move', 65, 30),
    ('short_squeeze_breakout', 'High short-interest names breaking resistance on unusual volume', 65, 30),
    ('sector_rotation_leader', 'Relative strength leaders in improving or leading sectors', 65, 30),
    ('volatility_contraction_breakout', 'Tight range contraction followed by a volume breakout', 65, 30),
    ('high_retest', '52-week high breakout retest and reclaim setup', 65, 30),
    ('relative_strength_leader', 'Fresh relative strength leaders with improving rank and constructive trend', 65, 30),
    ('premarket_catalyst', 'Pre-market gap and news catalyst setup from stored premarket signals', 65, 25)
ON CONFLICT (scanner_name) DO UPDATE SET
    description = EXCLUDED.description,
    min_score_threshold = EXCLUDED.min_score_threshold,
    max_signals_per_run = EXCLUDED.max_signals_per_run,
    is_active = TRUE,
    updated_at = NOW();
