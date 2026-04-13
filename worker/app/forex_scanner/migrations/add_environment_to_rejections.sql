-- Migration: add_environment_to_rejections.sql
-- Purpose: Add environment column to validator_rejections and smc_simple_rejections
--          to distinguish demo vs live signals in rejection analytics.
-- Database: forex
-- Note: Backfill existing rows with 'demo' since all data predates live trading.

-- ============================================================
-- validator_rejections
-- ============================================================

ALTER TABLE validator_rejections
    ADD COLUMN IF NOT EXISTS environment VARCHAR(10) NOT NULL DEFAULT 'demo';

-- Backfill existing rows (DEFAULT handles new rows; this covers pre-existing data)
UPDATE validator_rejections
    SET environment = 'demo'
    WHERE environment IS NULL OR environment = '';

CREATE INDEX IF NOT EXISTS idx_validator_rejections_environment
    ON validator_rejections(environment);

CREATE INDEX IF NOT EXISTS idx_validator_rejections_epic_env
    ON validator_rejections(epic, environment);

-- ============================================================
-- smc_simple_rejections
-- ============================================================

ALTER TABLE smc_simple_rejections
    ADD COLUMN IF NOT EXISTS environment VARCHAR(10) NOT NULL DEFAULT 'demo';

-- Backfill existing rows
UPDATE smc_simple_rejections
    SET environment = 'demo'
    WHERE environment IS NULL OR environment = '';

CREATE INDEX IF NOT EXISTS idx_smc_simple_rejections_environment
    ON smc_simple_rejections(environment);

CREATE INDEX IF NOT EXISTS idx_smc_simple_rejections_epic_env
    ON smc_simple_rejections(epic, environment);
