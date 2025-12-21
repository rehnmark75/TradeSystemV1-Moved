-- ============================================================================
-- TRADE_LOG TABLE COLUMN COMMENTS
-- ============================================================================
-- Purpose: Add clarifying comments to trade_log columns to prevent confusion
-- Created: 2025-12-19
--
-- IMPORTANT NAMING CLARIFICATION:
-- The column "limit_price" stores the TAKE PROFIT level, NOT the limit order
-- entry price! This is standard broker terminology where:
--   - "limit" = the profit target level
--   - "stop" = the loss protection level
--
-- For stop-entry orders (momentum confirmation style):
--   - entry_price is the actual order entry level (stop-entry price)
--   - BUY stop: entry_price is ABOVE market (enter when price breaks up)
--   - SELL stop: entry_price is BELOW market (enter when price breaks down)
-- ============================================================================

-- Add column comments for clarity
COMMENT ON COLUMN trade_log.entry_price IS
'Order entry level. For stop-entry orders, this is the momentum confirmation price (2-3 pips from market). BUY stops are placed ABOVE market, SELL stops BELOW market.';

COMMENT ON COLUMN trade_log.limit_price IS
'TAKE PROFIT level (absolute price). WARNING: Despite the name, this is NOT the limit order entry price - it stores the TP target. Broker terminology: "limit" = profit target.';

COMMENT ON COLUMN trade_log.sl_price IS
'Stop loss level (absolute price). The protective stop to limit losses.';

COMMENT ON COLUMN trade_log.tp_price IS
'Alternative take profit storage. Same concept as limit_price - stores TP target level.';

COMMENT ON COLUMN trade_log.status IS
'Order/trade status: pending_limit (unfilled stop-entry), limit_not_filled (expired without fill), tracking (filled, being monitored), closed, expired, etc.';

COMMENT ON COLUMN trade_log.monitor_until IS
'For stop-entry orders: auto-expiry time. Order cancelled if not filled by this time.';

COMMENT ON COLUMN trade_log.endpoint IS
'Order source: dev (market orders), dev-limit (stop-entry orders)';

-- Add table comment
COMMENT ON TABLE trade_log IS
'Trade execution and monitoring log. IMPORTANT: limit_price stores TAKE PROFIT (not limit order entry). entry_price stores the actual order entry level for stop-entry orders.';
