-- =============================================================================
-- 033: Drop dead position-tracking artifacts
-- =============================================================================
-- `stock_positions` was part of the original schema design but was never wired
-- up — no code path reads or writes it (verified Jun 2026). The authoritative
-- fill/position ledger is `broker_trades`, populated by BrokerTradeSync directly
-- from the RoboMarkets API.
--
-- `stock_orders.filled_price` / `filled_quantity` were likewise never populated
-- by any code path (fill status is set coarsely by the stale-order guardian;
-- real fill price/qty live in broker_trades.open_price / quantity).
--
-- Safe to drop: no foreign keys reference stock_positions, and no UI/analytics
-- read either the table or the two columns.
-- Idempotent (IF EXISTS) so it can be re-applied safely.
-- =============================================================================

DROP TABLE IF EXISTS stock_positions CASCADE;

ALTER TABLE stock_orders DROP COLUMN IF EXISTS filled_price;
ALTER TABLE stock_orders DROP COLUMN IF EXISTS filled_quantity;
