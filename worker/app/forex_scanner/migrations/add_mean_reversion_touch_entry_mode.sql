-- ============================================================================
-- MEAN_REVERSION: Add touch entry mode, session window, and low-vol filter
--
-- Extends the existing rejection-entry BB+RSI strategy with a configurable
-- "touch" entry mode (close beyond band + RSI extreme on same candle) and an
-- optional late-session window + low-volatility regime filter, replacing the
-- ADX gate on a per-pair basis.
--
-- Research basis: docs/research/usdchf_late_us_mean_reversion_strategy.md
--   USDCHF 5m 18-22 UTC: 104 trades, 67.3% WR, PF 2.41 (May 2026 analysis)
--   Production bt.py validation required before enabling trading.
--
-- Run: docker exec postgres psql -U postgres -d strategy_config -f /tmp/add_mean_reversion_touch_entry_mode.sql
-- ============================================================================

-- ── Global config table ──────────────────────────────────────────────────────

ALTER TABLE mean_reversion_global_config
    ADD COLUMN IF NOT EXISTS entry_mode VARCHAR(20) NOT NULL DEFAULT 'rejection',
    ADD COLUMN IF NOT EXISTS session_start_hour INTEGER DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS session_end_hour INTEGER DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS low_vol_regime_filter_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS regime_atr_max_pips FLOAT NOT NULL DEFAULT 7.0,
    ADD COLUMN IF NOT EXISTS regime_ema_period INTEGER NOT NULL DEFAULT 50,
    ADD COLUMN IF NOT EXISTS regime_ema_lookback_candles INTEGER NOT NULL DEFAULT 24,
    ADD COLUMN IF NOT EXISTS regime_ema_max_change_pips FLOAT NOT NULL DEFAULT 5.0;

-- ── Per-pair overrides table ─────────────────────────────────────────────────

ALTER TABLE mean_reversion_pair_overrides
    ADD COLUMN IF NOT EXISTS entry_mode VARCHAR(20) DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS session_start_hour INTEGER DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS session_end_hour INTEGER DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS low_vol_regime_filter_enabled BOOLEAN DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS regime_atr_max_pips FLOAT DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS regime_ema_period INTEGER DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS regime_ema_lookback_candles INTEGER DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS regime_ema_max_change_pips FLOAT DEFAULT NULL;

-- ── USDCHF: touch mode, late-US session, low-vol filter ─────────────────────
-- SL=10/TP=7 from research doc; monitor_only stays TRUE pending bt.py validation.
-- rsi_oversold/overbought tightened to 31/69 to match research thresholds.

UPDATE mean_reversion_pair_overrides
SET
    entry_mode                  = 'touch',
    session_start_hour          = 18,
    session_end_hour            = 22,
    low_vol_regime_filter_enabled = TRUE,
    regime_atr_max_pips         = 7.0,
    regime_ema_max_change_pips  = 5.0,
    fixed_stop_loss_pips        = 10.0,
    fixed_take_profit_pips      = 7.0
WHERE epic = 'CS.D.USDCHF.MINI.IP';

-- ── Verification ─────────────────────────────────────────────────────────────

SELECT 'Global config columns added' AS status;
SELECT entry_mode, session_start_hour, session_end_hour,
       low_vol_regime_filter_enabled, regime_atr_max_pips, regime_ema_max_change_pips
FROM mean_reversion_global_config WHERE is_active = TRUE;

SELECT 'USDCHF pair override' AS status;
SELECT epic, entry_mode, session_start_hour, session_end_hour,
       low_vol_regime_filter_enabled, fixed_stop_loss_pips, fixed_take_profit_pips,
       is_enabled, monitor_only
FROM mean_reversion_pair_overrides
WHERE epic = 'CS.D.USDCHF.MINI.IP';
