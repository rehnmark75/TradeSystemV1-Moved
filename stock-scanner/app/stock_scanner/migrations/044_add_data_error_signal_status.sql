-- 044_add_data_error_signal_status.sql
-- Outcome-tracking data-contamination fix (Jul 2026, financial-data-engineer autopsy).
--
-- Root cause: zlma_trend / reversal_scanner / rsi_divergence's
-- `_get_daily_candles()` used "ORDER BY timestamp ASC LIMIT N" instead of
-- "ORDER BY timestamp DESC LIMIT N" (+ reverse). Once a ticker accumulated
-- more than N days of history, this silently returned the OLDEST N candles
-- instead of the most recent N, freezing candles[-1] ("current price" in the
-- no-screening-metrics-yet fallback path) at a stale, multi-month-old close.
-- That stale price flowed into entry_price whenever _get_candidate_data()
-- found no watchlist/screening_metrics row for calculation_date (a routine
-- daily-lag condition on intraday/post-market runs), producing impossible
-- >100% "gains" once performance_tracker.py compared it against the real,
-- fresh close (e.g. SLS entry frozen at $1.825 (2025-12-08 close) vs real
-- exit $14.768 -> fake +709% "win"). Code fix: see zlma_trend.py,
-- reversal_scanner.py, rsi_divergence.py (_get_daily_candles + a
-- MAX_FALLBACK_CANDLE_AGE_DAYS staleness guard) and performance_tracker.py
-- (MAX_ABS_REALIZED_PNL_PCT guard in _update_signal_status).
--
-- This migration widens the status CHECK constraint so contaminated rows can
-- be flagged 'data_error' (audit-preserving) instead of silently polluting
-- win/loss/PF aggregates under 'closed'/'partial_exit'.
--
-- Idempotent: safe to re-run.

ALTER TABLE stock_scanner_signals
    DROP CONSTRAINT IF EXISTS stock_scanner_signals_status_check;

ALTER TABLE stock_scanner_signals
    ADD CONSTRAINT stock_scanner_signals_status_check
    CHECK (status IN (
        'active', 'triggered', 'partial_exit', 'closed', 'expired',
        'cancelled', 'data_error'
    ));
