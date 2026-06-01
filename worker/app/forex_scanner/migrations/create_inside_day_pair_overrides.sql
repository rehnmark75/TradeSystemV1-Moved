-- INSIDE_DAY per-pair UI configuration table.
-- The strategy's core tunables are code-defined constants in v1; this table lets
-- trading-ui expose pair enablement, monitor posture, and documented overrides
-- using the same settings workflow as other strategies.

CREATE TABLE IF NOT EXISTS inside_day_pair_overrides (
    id                      SERIAL PRIMARY KEY,
    config_set              VARCHAR(20)  NOT NULL DEFAULT 'demo',
    epic                    VARCHAR(60)  NOT NULL,
    pair_name               VARCHAR(10),
    is_enabled              BOOLEAN      DEFAULT false,
    is_traded               BOOLEAN      DEFAULT false,
    monitor_only            BOOLEAN      DEFAULT true,

    weekly_bias_q           DOUBLE PRECISION,
    inside_day_min_pips     DOUBLE PRECISION,
    inside_day_max_pips     DOUBLE PRECISION,
    atr_period              INTEGER,
    atr_buffer_fraction     DOUBLE PRECISION,
    reward_risk             DOUBLE PRECISION,
    base_confidence         DOUBLE PRECISION,

    parameter_overrides     JSONB        DEFAULT '{}',
    notes                   TEXT,
    created_at              TIMESTAMPTZ  DEFAULT now(),
    updated_at              TIMESTAMPTZ  DEFAULT now(),

    UNIQUE (config_set, epic)
);

INSERT INTO inside_day_pair_overrides
    (config_set, epic, pair_name, is_enabled, monitor_only, notes)
VALUES
    ('demo', 'CS.D.EURUSD.CEEM.IP', 'EURUSD', true, true, 'INSIDE_DAY v1 launch pair. OOS and spread-stress passed; monitor-only.'),
    ('demo', 'CS.D.USDJPY.MINI.IP', 'USDJPY', true, true, 'INSIDE_DAY v1 launch pair. OOS and spread-stress passed; monitor-only.')
ON CONFLICT (config_set, epic) DO NOTHING;
