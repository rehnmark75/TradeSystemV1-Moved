-- RANGE_FADE: NY/overlap-session-scoped 1h-ADX ceiling (Jun 2026)
--
-- The loss engine for RANGE_FADE is fading AGAINST a strong 1h trend during the
-- NY session. On demo (n=86 closed, Apr-Jun 2026) trades at hour 15-18 UTC with
-- 1h ADX >= 35 ran 21% WR / PF 0.08 / -716 SEK, while the SAME high-ADX condition
-- OUTSIDE NY was net-profitable (PF 1.29). So the gate is session-scoped, NOT a
-- global ADX ceiling (a global ceiling would kill the profitable outside-NY trades).
--
-- The gate lives in range_fade_strategy.py (it rejects when is_in_ny_session(hour)
-- AND adx_htf_val >= ny_session_htf_adx_ceiling). It is NOT an LPF rule because
-- build_lpf_context() never maps ADX into the LPF context.
--
-- Effective window is 15-18 UTC because is_session_allowed already caps at
-- new_york_end_hour_utc=18 (19-20 are session_blocked upstream).

ALTER TABLE range_fade_global_config
  ADD COLUMN IF NOT EXISTS ny_session_htf_adx_ceiling DOUBLE PRECISION DEFAULT 999.0, -- 999 = off
  ADD COLUMN IF NOT EXISTS ny_session_start_hour_utc INTEGER DEFAULT 15,
  ADD COLUMN IF NOT EXISTS ny_session_end_hour_utc INTEGER DEFAULT 20;

-- Enable on demo (the traded environment). Live stays off (999): RANGE_FADE on
-- live is EURUSD monitor-only.
UPDATE range_fade_global_config
  SET ny_session_htf_adx_ceiling = 35
  WHERE is_active = TRUE AND config_set = 'demo';

-- The two pre-existing RANGE_FADE LPF rules are dead (LPF context lacks ADX; the
-- NY block is handled by the strategy session gate). Disabled as part of this change.
UPDATE loss_prevention_rules SET is_enabled = FALSE
  WHERE rule_name IN ('range_fade_ny_session_block','range_fade_high_adx_block');
