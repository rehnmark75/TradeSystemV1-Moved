-- 039_add_atr_trailing_columns.sql
-- ATR-trailing stop support for the stock breakeven monitor.
--
-- Validated Jun/Jul 2026 (analysis/ema_cross_lab.py): on the ema_cross_9_21_50
-- ranked pool, dropping the premature 7% TP and trailing the stop by ~3x daily
-- ATR once armed lifts PF ~1.18 -> ~1.54 with NO change to the fixed 2.5%
-- initial stop / position sizing. Trailing is DISABLED by default and the
-- monitor stays dry-run by default; nothing changes in live until explicitly
-- enabled via env on the stock-breakeven-monitor container.
--
-- Idempotent: safe to re-run.

ALTER TABLE stock_breakeven_monitors
    ADD COLUMN IF NOT EXISTS peak_price      DECIMAL(12,4),  -- high-water mark since entry
    ADD COLUMN IF NOT EXISTS tp_widened      BOOLEAN DEFAULT FALSE,  -- TP pushed to backstop
    ADD COLUMN IF NOT EXISTS trail_moves     INTEGER DEFAULT 0,      -- count of trail SL moves
    ADD COLUMN IF NOT EXISTS trail_last_at   TIMESTAMP;              -- last trail move time

-- Register the trailing settings as documentation (the monitor reads these from
-- env; these rows are informational/for a future UI). All OFF by default.
INSERT INTO stock_auto_trade_settings (key, value, value_type, label, description)
VALUES
  ('TRAILING_STOP_ENABLED', 'false', 'bool', 'ATR Trailing Enabled',
   'Enable ATR trailing stop in the stock breakeven monitor (env-driven on stock-breakeven-monitor). OFF by default.'),
  ('TRAILING_STOP_ARM_PCT', '2.0', 'float', 'Trail Arm %',
   'Unrealized profit %% at which the ATR trail arms (validated 2.0).'),
  ('TRAILING_STOP_ATR_MULT', '3.0', 'float', 'Trail ATR Mult',
   'Trailing stop = peak * (1 - mult*dailyATR%%/100). Validated plateau 2.5-3.5, default 3.0.'),
  ('TRAILING_STOP_TP_BACKSTOP_PCT', '30.0', 'float', 'Trail TP Backstop %',
   'When trailing, the hard TP is widened to this %% so winners run and the trail governs the exit.')
ON CONFLICT (key) DO NOTHING;
