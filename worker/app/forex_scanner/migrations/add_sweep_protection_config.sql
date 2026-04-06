-- Migration: Add sweep protection filter config to smc_simple_global_config
-- v2.42.0 - Apr 2026
-- Adds multi-condition overextension filter to block entries at likely liquidity sweep zones.
-- Triggered after EURUSD loss on Apr 6, 2026: RSI 85 + EMA 35 pips + price above BB upper.
--
-- Per-pair overrides are stored in smc_simple_pair_overrides.parameter_overrides JSONB:
--   {"sweep_protection_enabled": false}              -- disable for specific pair
--   {"sweep_protection_mode": "monitor"}             -- monitor-only for specific pair
--   {"sweep_rsi_threshold_buy": 80.0}                -- higher RSI threshold for that pair
--   {"sweep_rsi_threshold_sell": 20.0}
--   {"sweep_max_ema_distance_pips": 40.0}            -- wider EMA distance for that pair
--   {"sweep_min_conditions": 3}                      -- require all 3 conditions to block

ALTER TABLE smc_simple_global_config
    ADD COLUMN IF NOT EXISTS sweep_protection_enabled BOOLEAN DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS sweep_protection_mode TEXT DEFAULT 'block',
    ADD COLUMN IF NOT EXISTS sweep_rsi_threshold_buy DECIMAL(5,1) DEFAULT 78.0,
    ADD COLUMN IF NOT EXISTS sweep_rsi_threshold_sell DECIMAL(5,1) DEFAULT 22.0,
    ADD COLUMN IF NOT EXISTS sweep_max_ema_distance_pips DECIMAL(5,1) DEFAULT 35.0,
    ADD COLUMN IF NOT EXISTS sweep_max_ema_distance_pips_jpy DECIMAL(5,1) DEFAULT 45.0,
    ADD COLUMN IF NOT EXISTS sweep_min_conditions INTEGER DEFAULT 2;

-- Set values on the active config row
UPDATE smc_simple_global_config
SET
    sweep_protection_enabled = TRUE,
    sweep_protection_mode = 'block',
    sweep_rsi_threshold_buy = 78.0,
    sweep_rsi_threshold_sell = 22.0,
    sweep_max_ema_distance_pips = 35.0,
    sweep_max_ema_distance_pips_jpy = 45.0,
    sweep_min_conditions = 2
WHERE is_active = TRUE;

-- Verify
SELECT sweep_protection_enabled, sweep_protection_mode,
       sweep_rsi_threshold_buy, sweep_rsi_threshold_sell,
       sweep_max_ema_distance_pips, sweep_max_ema_distance_pips_jpy,
       sweep_min_conditions
FROM smc_simple_global_config
WHERE is_active = TRUE;
