-- Recalibrate IG CFEGOLD trailing thresholds after confirming broker point size.
--
-- Finding:
--   IG portal shows $12 gold movement as 24 IG points.
--   Therefore 1 IG point = $0.50 = 5 internal XAU pips.
--
-- The live trailing engine uses internal XAU pips where 1 point = $0.10, so
-- these values intentionally map to wider gold-dollar movement than the old
-- static 50/80/110 schedule.

UPDATE trailing_pair_config
SET early_breakeven_trigger_points = 80,  -- $8.00
    early_breakeven_buffer_points = 5,    -- $0.50
    stage1_trigger_points = 100,          -- $10.00
    stage1_lock_points = 50,              -- $5.00
    stage2_trigger_points = 140,          -- $14.00
    stage2_lock_points = 90,              -- $9.00
    stage3_trigger_points = 170,          -- $17.00
    stage3_min_distance = 45,             -- $4.50
    min_trail_distance = 45,              -- $4.50
    break_even_trigger_points = 80,       -- $8.00
    partial_close_trigger_points = 100,   -- $10.00; partial close remains disabled unless enabled separately
    updated_by = 'migration',
    change_reason = 'Recalibrate XAU gold trailing after confirming IG CFEGOLD point size: $12 = 24 IG points'
WHERE epic = 'CS.D.CFEGOLD.CEE.IP'
  AND strategy = 'DEFAULT'
  AND config_set IN ('demo', 'live');
