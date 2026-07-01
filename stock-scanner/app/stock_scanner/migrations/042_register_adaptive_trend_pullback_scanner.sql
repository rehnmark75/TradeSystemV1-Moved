-- 042_register_adaptive_trend_pullback_scanner.sql
-- Register adaptive_trend_pullback in the scanner registry so its signals satisfy
-- the stock_scanner_signals -> stock_signal_scanners FK and can be saved/traded.
-- LIVE demo forward-test (Jul 1 2026): RoboMarkets demo account 92116829, real
-- execution / zero capital risk. Runs + logs signals AND is live-tradable — it is
-- deliberately NOT listed in MONITOR_ONLY_SCANNERS in
-- trading-ui/app/api/signals/top/route.ts, per feedback_forward_test_means_demo_
-- execution (a forward test = real demo fills, not monitor-only logging).
--
-- min_score_threshold is set to 0 (like ema_cross_9_21_50): the strategy's real
-- go/no-go gate is fully internal (min_perf_score=4.0 in
-- strategies/adaptive_trend_pullback.py), and composite_score is a display/
-- ranking score only (typically ~30-60, capped by perf_score's practical ceiling
-- of ~5 for equities). This DB column is not read anywhere as a runtime filter
-- (registry/FK metadata only) — see scanner_manager.py / route.ts for the actual
-- execution gates (candidate_score, AUTO_TRADE_MIN_SCORE, etc.), which are fully
-- independent of this scanner's composite_score. Idempotent.

INSERT INTO stock_signal_scanners
    (scanner_name, description, is_active, min_score_threshold, max_signals_per_run)
VALUES
    ('adaptive_trend_pullback',
     'LuxAlgo blend: adaptive SuperTrend (K-means factor) + Nadaraya-Watson kernel envelope pullback + predictive-ranges midline, breadth-gated. LIVE demo forward-test Jul 1 2026.',
     true, 0, 50)
ON CONFLICT (scanner_name) DO UPDATE
    SET description = EXCLUDED.description,
        is_active = TRUE,
        min_score_threshold = EXCLUDED.min_score_threshold,
        max_signals_per_run = EXCLUDED.max_signals_per_run,
        updated_at = NOW();
