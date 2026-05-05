-- Unified read-only view over strategy_rejections for MEAN_REVERSION, IMPULSE_FADE, XAU_GOLD.
-- Note: SMC_SIMPLE rejections live in smc_simple_rejections (forex DB); they are not merged here.
-- Stage codes are uppercased so XAU_GOLD (lowercase) matches the rest.
--
-- Apply:
--   docker exec task-worker cat /app/forex_scanner/migrations/create_v_strategy_rejections_unified.sql \
--     | docker exec -i postgres psql -U postgres -d strategy_config

CREATE OR REPLACE VIEW v_strategy_rejections_unified AS
SELECT
    strategy,
    epic,
    pair,
    scan_timestamp,
    UPPER(stage)  AS stage,
    reason,
    direction,
    hour_utc,
    session,
    details
FROM strategy_rejections;
