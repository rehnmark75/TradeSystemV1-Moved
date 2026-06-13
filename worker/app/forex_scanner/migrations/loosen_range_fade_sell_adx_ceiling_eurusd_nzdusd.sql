-- RANGE_FADE: loosen SELL-side 5m-ADX ceiling on EURUSD + NZDUSD demo (Jun 13 2026)
--
-- Run against the strategy_config database:
--   docker exec postgres psql -U postgres -d strategy_config \
--     -f /app/forex_scanner/migrations/loosen_range_fade_sell_adx_ceiling_eurusd_nzdusd.sql
--
-- Why
-- ---
-- The ADX_CEILING gate (range_fade_strategy.py, via get_pair_adx_ceiling) checks
-- the 5m ADX against a global default of 25.0 (range_fade_config_service.py:77;
-- there is no DB column, so it is the dataclass default). The per-gate edge layer
-- (GateEdgeAnalyzer / v_rejection_gate_edge, commit 614d27e) showed this gate is
-- NOT a uniform win — it is pair/direction-specific:
--
--   ADX_CEILING blocked population, 14d backfill (fixed 10/15 reference grid):
--     EURUSD SELL  n=119  WR 81%  ref-PF 6.6   -> blocking WINNERS (REVIEW)
--     NZDUSD SELL  n=29   WR 93%  ref-PF 20    -> blocking WINNERS (REVIEW)
--     AUDJPY BUY   n=97   WR 33%  ref-PF 0.74  -> blocking losers  (KEEP)
--     AUDUSD SELL  n=24   WR 29%  ref-PF 0.62  -> blocking losers  (KEEP)
--
-- get_pair_adx_ceiling() prefers a direction-scoped `{dir}_adx_ceiling` override
-- before falling back to the global 25. So we raise ONLY the SELL ceiling on the
-- two pairs where the gate blocks winners, leaving BUY and every other pair on the
-- global 25. 40 (not 999/off) keeps an extreme-trend guard while releasing the
-- bulk of the blocked 25<ADX<=40 SELL fades.
--
-- Scope: DEMO only. All RANGE_FADE pairs are monitor-only, so this is pure
-- data-collection (lets the blocked SELLs reach alert_history for forward
-- evaluation), zero execution risk. Live is left untouched.
--
-- Reference grid is a fixed 10/15 anchor, NOT RANGE_FADE's native bracket — treat
-- as a relative gate-vs-gate signal, not a validated live edge. Prior RANGE_FADE
-- ADX levers were OOS-skeptical, but those were ADX FLOORS / regime-quality gates
-- in hostile June EURUSD; this is a direction-scoped CEILING on a clean n=119.
--
-- Forward gate
-- -----------
-- After ~30+ new EURUSD/NZDUSD SELL signals land in the 25<ADX<=40 band,
-- re-check v_rejection_gate_edge + monitor_only_outcomes on the NATIVE bracket.
-- Keep if PF >= 1.2; otherwise revert:
--   UPDATE range_fade_pair_overrides
--     SET parameter_overrides = parameter_overrides - 'sell_adx_ceiling'
--     WHERE config_set='demo'
--       AND epic IN ('CS.D.EURUSD.CEEM.IP','CS.D.NZDUSD.MINI.IP');

UPDATE range_fade_pair_overrides
   SET parameter_overrides =
       COALESCE(parameter_overrides, '{}'::jsonb) || '{"sell_adx_ceiling": 40}'::jsonb
 WHERE config_set = 'demo'
   AND epic IN ('CS.D.EURUSD.CEEM.IP', 'CS.D.NZDUSD.MINI.IP');

-- Verify:
--   SELECT config_set, epic, parameter_overrides->>'sell_adx_ceiling'
--   FROM range_fade_pair_overrides
--   WHERE epic IN ('CS.D.EURUSD.CEEM.IP','CS.D.NZDUSD.MINI.IP')
--   ORDER BY config_set, epic;
