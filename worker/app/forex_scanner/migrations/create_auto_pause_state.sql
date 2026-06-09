-- ============================================================================
-- Auto-Pause Layer — Runtime State / Audit (Phase 3)
-- Database: strategy_config
-- ============================================================================
-- One row per cell the auto-pause layer has paused. Records WHEN it was paused
-- (needed to measure "fresh shadow signals since pause" and the resume cooldown)
-- and tracks shadow-resume PROPOSALS.
--
-- Distinct from auto_pause_eligibility:
--   * auto_pause_eligibility = the frozen allowlist (which cells are protected).
--   * auto_pause_state       = runtime lifecycle (when paused, resume tracking).
--
-- PHASE 3 IS PROPOSE-ONLY: resume_proposed_at / resume_proposal_count record
-- that a resume *would* fire; the layer does NOT auto-resume yet (staged
-- rollout: propose -> confirm -> fully-auto).
-- ============================================================================

CREATE TABLE IF NOT EXISTS auto_pause_state (
    id                    SERIAL PRIMARY KEY,
    strategy              VARCHAR(64)  NOT NULL,
    epic                  VARCHAR(64)  NOT NULL,
    config_set            VARCHAR(16)  NOT NULL DEFAULT 'demo',
    state                 VARCHAR(16)  NOT NULL DEFAULT 'paused',  -- 'paused'
    paused_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
    pause_reason          TEXT,

    -- Shadow-resume tracking (Phase 3, propose-only)
    last_eval_at          TIMESTAMPTZ,
    shadow_n              INTEGER,
    shadow_pf             NUMERIC(8,3),
    resume_proposed_at    TIMESTAMPTZ,
    resume_proposal_count INTEGER      NOT NULL DEFAULT 0,

    created_at            TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT uq_auto_pause_state UNIQUE (strategy, epic, config_set)
);

COMMENT ON TABLE auto_pause_state IS
  'Auto-pause runtime lifecycle/audit: when a cell was auto-paused + shadow '
  'resume-proposal tracking (Phase 3 is propose-only — no auto-resume yet). '
  'Distinct from auto_pause_eligibility (the frozen allowlist).';

CREATE INDEX IF NOT EXISTS idx_auto_pause_state_lookup
  ON auto_pause_state (config_set, state);
