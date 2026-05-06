-- LPF rule: block EURUSD SMC_SIMPLE MOMENTUM entries that chase a rejected/blocked
-- signal after price has already moved >= 10 pips in that direction.
--
-- Rationale: when Claude or LPF rejects a EURUSD signal and price then runs 10+ pips,
-- a new MOMENTUM entry at the extended level enters at degraded R:R relative to the
-- original setup. PULLBACK entries are exempt — a pullback to the prior area is the
-- reset we want. apply_in_backtest=FALSE because the rule queries live alert_history.
--
-- Category G: history-based staleness (separate from A-F so it never cap-competes
-- with other pair rules in the per-category aggregation).

-- demo config
INSERT INTO loss_prevention_rules (
    rule_name,
    category,
    penalty,
    condition_config,
    is_enabled,
    apply_in_backtest,
    applies_to_strategies,
    config_set,
    description
) VALUES (
    'eurusd_missed_signal_chase',
    'G',
    1.0,
    '{
        "type": "missed_signal_chase",
        "epic_contains": "EURUSD",
        "lookback_hours": 2,
        "move_threshold_pips": 10,
        "pip_scale": 10000,
        "strategy": "SMC_SIMPLE"
    }'::jsonb,
    TRUE,
    FALSE,
    '["SMC_SIMPLE"]'::jsonb,
    'demo',
    'Block EURUSD MOMENTUM entries chasing a rejected signal after >= 10 pips have moved. PULLBACK entries exempt (self-filtering via entry_type check in code).'
)
ON CONFLICT (rule_name, config_set) DO UPDATE SET
    category = EXCLUDED.category,
    penalty = EXCLUDED.penalty,
    condition_config = EXCLUDED.condition_config,
    is_enabled = EXCLUDED.is_enabled,
    apply_in_backtest = EXCLUDED.apply_in_backtest,
    applies_to_strategies = EXCLUDED.applies_to_strategies,
    description = EXCLUDED.description;

-- live config
INSERT INTO loss_prevention_rules (
    rule_name,
    category,
    penalty,
    condition_config,
    is_enabled,
    apply_in_backtest,
    applies_to_strategies,
    config_set,
    description
) VALUES (
    'eurusd_missed_signal_chase',
    'G',
    1.0,
    '{
        "type": "missed_signal_chase",
        "epic_contains": "EURUSD",
        "lookback_hours": 2,
        "move_threshold_pips": 10,
        "pip_scale": 10000,
        "strategy": "SMC_SIMPLE"
    }'::jsonb,
    TRUE,
    FALSE,
    '["SMC_SIMPLE"]'::jsonb,
    'live',
    'Block EURUSD MOMENTUM entries chasing a rejected signal after >= 10 pips have moved. PULLBACK entries exempt (self-filtering via entry_type check in code).'
)
ON CONFLICT (rule_name, config_set) DO UPDATE SET
    category = EXCLUDED.category,
    penalty = EXCLUDED.penalty,
    condition_config = EXCLUDED.condition_config,
    is_enabled = EXCLUDED.is_enabled,
    apply_in_backtest = EXCLUDED.apply_in_backtest,
    applies_to_strategies = EXCLUDED.applies_to_strategies,
    description = EXCLUDED.description;
