-- ============================================================================
-- Auto-Pause Layer — Fully-Auto Resume opt-in (Phase C, per-cell)
-- Database: strategy_config
-- ============================================================================
-- auto_resume gates whether a cell FULLY auto-resumes (flips monitor_only back
-- on its own) when resume rule R1 is met, vs the default propose-only behaviour
-- (logs a proposal; a human re-enables). It is per-cell so a cell graduates
-- from propose-only -> fully-auto individually, once its proposals are trusted.
--
-- Resume still requires R1: >=15 fresh reconstructed shadow outcomes,
-- shadow PF > 1.1 (hysteresis above the 0.8 trip), >=10-day cooldown. The
-- shadow metric is a conservative approximation (fixed SL/TP vs live trailing;
-- counts signals not trade-equivalents) — enable auto_resume on demo first.
-- ============================================================================

ALTER TABLE auto_pause_eligibility
    ADD COLUMN IF NOT EXISTS auto_resume BOOLEAN NOT NULL DEFAULT FALSE;

COMMENT ON COLUMN auto_pause_eligibility.auto_resume IS
  'If TRUE, the layer fully auto-resumes this cell (flips monitor_only back) when '
  'resume rule R1 is met. If FALSE (default), resume is propose-only.';

-- Audit: when a cell was auto-resumed.
ALTER TABLE auto_pause_state
    ADD COLUMN IF NOT EXISTS resumed_at TIMESTAMPTZ;
