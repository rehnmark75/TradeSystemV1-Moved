-- ============================================================================
-- ADD SCANNER CONTROLLER COLUMNS TO scanner_global_config
-- ============================================================================
-- Purpose: Add columns required for scanner_controller.py database migration
-- Database: strategy_config
--
-- Usage:
--   docker exec postgres psql -U postgres -d strategy_config -f /app/forex_scanner/migrations/add_scanner_controller_columns.sql
-- ============================================================================

\c strategy_config;

-- ============================================================================
-- ADD NEW COLUMNS
-- ============================================================================

-- spread_pips: Default spread in pips for signal detection
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'scanner_global_config' AND column_name = 'spread_pips'
    ) THEN
        ALTER TABLE scanner_global_config ADD COLUMN spread_pips DECIMAL(4,2) NOT NULL DEFAULT 1.5;
        RAISE NOTICE 'Added column: spread_pips';
    ELSE
        RAISE NOTICE 'Column spread_pips already exists';
    END IF;
END $$;

-- use_bid_adjustment: Whether to apply bid adjustment for signals
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'scanner_global_config' AND column_name = 'use_bid_adjustment'
    ) THEN
        ALTER TABLE scanner_global_config ADD COLUMN use_bid_adjustment BOOLEAN NOT NULL DEFAULT FALSE;
        RAISE NOTICE 'Added column: use_bid_adjustment';
    ELSE
        RAISE NOTICE 'Column use_bid_adjustment already exists';
    END IF;
END $$;

-- epic_list: JSONB array of epic codes to scan
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'scanner_global_config' AND column_name = 'epic_list'
    ) THEN
        ALTER TABLE scanner_global_config ADD COLUMN epic_list JSONB NOT NULL DEFAULT '[
            "CS.D.EURUSD.CEEM.IP",
            "CS.D.GBPUSD.MINI.IP",
            "CS.D.USDJPY.MINI.IP",
            "CS.D.AUDUSD.MINI.IP",
            "CS.D.USDCHF.MINI.IP",
            "CS.D.USDCAD.MINI.IP",
            "CS.D.NZDUSD.MINI.IP",
            "CS.D.EURJPY.MINI.IP",
            "CS.D.AUDJPY.MINI.IP"
        ]'::jsonb;
        RAISE NOTICE 'Added column: epic_list';
    ELSE
        RAISE NOTICE 'Column epic_list already exists';
    END IF;
END $$;

-- pair_info: JSONB dict mapping epic codes to pair info (name, pip_multiplier)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'scanner_global_config' AND column_name = 'pair_info'
    ) THEN
        ALTER TABLE scanner_global_config ADD COLUMN pair_info JSONB NOT NULL DEFAULT '{
            "CS.D.EURUSD.CEEM.IP": {"pair": "EURUSD", "pip_multiplier": 10000},
            "CS.D.GBPUSD.MINI.IP": {"pair": "GBPUSD", "pip_multiplier": 10000},
            "CS.D.AUDUSD.MINI.IP": {"pair": "AUDUSD", "pip_multiplier": 10000},
            "CS.D.NZDUSD.MINI.IP": {"pair": "NZDUSD", "pip_multiplier": 10000},
            "CS.D.USDCHF.MINI.IP": {"pair": "USDCHF", "pip_multiplier": 10000},
            "CS.D.USDCAD.MINI.IP": {"pair": "USDCAD", "pip_multiplier": 10000},
            "CS.D.USDJPY.MINI.IP": {"pair": "USDJPY", "pip_multiplier": 100},
            "CS.D.EURJPY.MINI.IP": {"pair": "EURJPY", "pip_multiplier": 100},
            "CS.D.GBPJPY.MINI.IP": {"pair": "GBPJPY", "pip_multiplier": 100},
            "CS.D.AUDJPY.MINI.IP": {"pair": "AUDJPY", "pip_multiplier": 100},
            "CS.D.CADJPY.MINI.IP": {"pair": "CADJPY", "pip_multiplier": 100},
            "CS.D.CHFJPY.MINI.IP": {"pair": "CHFJPY", "pip_multiplier": 100},
            "CS.D.NZDJPY.MINI.IP": {"pair": "NZDJPY", "pip_multiplier": 100},
            "CS.D.EURGBP.MINI.IP": {"pair": "EURGBP", "pip_multiplier": 10000},
            "CS.D.EURAUD.MINI.IP": {"pair": "EURAUD", "pip_multiplier": 10000},
            "CS.D.GBPAUD.MINI.IP": {"pair": "GBPAUD", "pip_multiplier": 10000}
        }'::jsonb;
        RAISE NOTICE 'Added column: pair_info';
    ELSE
        RAISE NOTICE 'Column pair_info already exists';
    END IF;
END $$;

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON COLUMN scanner_global_config.spread_pips IS 'Default spread in pips for signal detection calculations';
COMMENT ON COLUMN scanner_global_config.use_bid_adjustment IS 'Apply bid price adjustment for accurate signal entry pricing';
COMMENT ON COLUMN scanner_global_config.epic_list IS 'JSONB array of epic codes to scan (e.g., ["CS.D.EURUSD.CEEM.IP", ...])';
COMMENT ON COLUMN scanner_global_config.pair_info IS 'JSONB dict mapping epic codes to pair metadata: {"epic": {"pair": "EURUSD", "pip_multiplier": 10000}}';

-- ============================================================================
-- AUDIT ENTRY
-- ============================================================================
INSERT INTO scanner_config_audit (
    config_id,
    change_type,
    changed_by,
    change_reason,
    new_values,
    category
)
SELECT
    id,
    'UPDATE',
    'migration',
    'Added scanner_controller columns: spread_pips, use_bid_adjustment, epic_list, pair_info',
    jsonb_build_object(
        'spread_pips', spread_pips,
        'use_bid_adjustment', use_bid_adjustment,
        'epic_list', epic_list,
        'pair_info', pair_info
    ),
    'core'
FROM scanner_global_config
WHERE is_active = TRUE;

-- ============================================================================
-- VERIFICATION
-- ============================================================================
SELECT
    id,
    version,
    spread_pips,
    use_bid_adjustment,
    jsonb_array_length(epic_list) AS epic_count,
    jsonb_object_keys(pair_info) IS NOT NULL AS pair_info_exists,
    updated_at
FROM scanner_global_config
WHERE is_active = TRUE;

DO $$
BEGIN
    RAISE NOTICE 'Migration complete: scanner_controller columns added';
END $$;
