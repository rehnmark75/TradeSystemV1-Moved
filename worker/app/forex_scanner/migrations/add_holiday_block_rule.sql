-- LPF: block trades on EU/US/UK bank holidays (low-liquidity, erratic moves)
--
-- Seeded in response to week of Apr 27–May 2 2026 analysis: May 1 (EU Labour Day)
-- produced 3 of 6 EURUSD losers (-294 SEK) on a single day with reduced liquidity.
--
-- Rule type: date_block
--   - dates       : explicit YYYY-MM-DD list (movable holidays: Easter etc.)
--   - month_days  : recurring MM-DD list (Christmas, NYE, Labour Day, etc.)
--   - epic_contains : optional substring filter; absent = applies to all pairs
--
-- Penalty 1.00 = hard block in 'block' mode regardless of other category penalties.
-- Category 'C' (time/calendar) — aggregated as max-per-category.

INSERT INTO loss_prevention_rules (
    rule_name, category, penalty, condition_config, is_enabled,
    apply_in_backtest, applies_to_strategies, config_set, description
)
SELECT
    'holiday_major_fx',
    'C',
    1.00,
    jsonb_build_object(
        'type', 'date_block',
        'label', 'Major FX bank holiday',
        'month_days', jsonb_build_array(
            '01-01',  -- New Year's Day (global)
            '05-01',  -- Labour Day (EU)
            '07-04',  -- US Independence Day
            '12-24',  -- Christmas Eve (early close)
            '12-25',  -- Christmas Day
            '12-26',  -- Boxing Day (UK)
            '12-31'   -- New Year's Eve (early close, thin liquidity)
        ),
        'dates', jsonb_build_array(
            -- 2026 movable / one-off holidays
            '2026-04-03',  -- Good Friday
            '2026-04-06',  -- Easter Monday (EU/UK)
            '2026-01-19',  -- US MLK Day
            '2026-02-16',  -- US Presidents Day
            '2026-05-25',  -- US Memorial Day + UK Spring Bank
            '2026-09-07',  -- US Labor Day
            '2026-11-26',  -- US Thanksgiving
            '2026-11-27'   -- US Black Friday (half-day, thin)
        )
    ),
    TRUE,
    TRUE,
    NULL,            -- universal: applies to all strategies
    cs.config_set,
    'Block all signals on major EU/US/UK bank holidays. Triggered by May 1 2026 EURUSD loss cluster (3 losers / -294 SEK on a single low-liquidity day).'
FROM (VALUES ('demo'), ('live')) AS cs(config_set)
ON CONFLICT (rule_name, config_set) DO UPDATE
SET condition_config = EXCLUDED.condition_config,
    penalty          = EXCLUDED.penalty,
    is_enabled       = EXCLUDED.is_enabled,
    description      = EXCLUDED.description,
    updated_at       = NOW();
