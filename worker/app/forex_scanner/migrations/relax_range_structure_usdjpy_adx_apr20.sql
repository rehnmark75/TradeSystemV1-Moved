-- =========================================================================
-- relax_range_structure_usdjpy_adx_apr20.sql
--
-- Apr 20, 2026 — After swapping the RANGE_STRUCTURE v1.0 inline fractal +
-- OB/FVG proxies for the real SMC helpers (SMCMarketStructure /
-- SMCOrderBlocks / SMCFairValueGaps), the USDJPY per-pair ADX ceilings of
-- 18/20 were over-restrictive — only 1 trade fired in 90 days. Since the
-- real helpers are themselves more selective about swing/OB/FVG quality,
-- the tight per-pair ADX cap is redundant. Relax to match the global
-- default (20 primary / 22 HTF).
--
-- SL floor (sl_pips_min=8) stays — that is a plan-hard rule and was a
-- correct v1.0 decision; ATR-based SL tuning will happen once signals
-- materialise at meaningful n.
-- =========================================================================

UPDATE range_structure_pair_overrides
SET
    adx_hard_ceiling_primary = 20.0,
    adx_hard_ceiling_htf     = 22.0,
    updated_at               = NOW(),
    notes = COALESCE(notes, '')
            || E'\n[2026-04-20] Relaxed ADX ceilings 18/20 -> 20/22 after real SMC helper swap (inline fractal+OB/FVG proxies replaced).'
WHERE epic = 'CS.D.USDJPY.MINI.IP';

-- Verify
SELECT epic, adx_hard_ceiling_primary, adx_hard_ceiling_htf, sl_pips_min, updated_at
FROM range_structure_pair_overrides
WHERE epic = 'CS.D.USDJPY.MINI.IP';
