-- 040_register_ema_cross_scanner.sql
-- Register ema_cross_9_21_50 in the scanner registry so its signals satisfy the
-- stock_scanner_signals -> stock_signal_scanners FK and can be saved/traded.
-- Activated Jul 1 2026. Idempotent.

INSERT INTO stock_signal_scanners
    (scanner_name, description, is_active, min_score_threshold, max_signals_per_run)
VALUES
    ('ema_cross_9_21_50',
     'Long-only: close > EMA50 & EMA21, crosses up through EMA9 (trend-quality stack gate). composite_score = EMA50-slope-led trend conviction so the top-pool trades the best. Validated config = scanner + ATR-trail exit + breadth gate (~1.7 PF, OOS 1.76).',
     true, 0, 50)
ON CONFLICT (scanner_name) DO UPDATE
    SET description = EXCLUDED.description,
        is_active = TRUE,
        updated_at = NOW();
