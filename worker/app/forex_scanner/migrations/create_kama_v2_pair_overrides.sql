-- KAMA V2 per-pair configuration table
-- Mirrors smc_momentum_pair_overrides pattern.
-- NULL in any per-pair column → strategy falls back to KamaV2Config global default.

CREATE TABLE IF NOT EXISTS kama_v2_pair_overrides (
    id                      SERIAL PRIMARY KEY,
    config_set              VARCHAR(20)  NOT NULL DEFAULT 'demo',
    epic                    VARCHAR(60)  NOT NULL,
    pair_name               VARCHAR(10),
    is_enabled              BOOLEAN      DEFAULT false,
    is_traded               BOOLEAN      DEFAULT false,
    monitor_only            BOOLEAN      DEFAULT true,

    -- Per-pair overrides (NULL = use global default from KamaV2Config)
    fixed_stop_loss_pips    DOUBLE PRECISION,
    fixed_take_profit_pips  DOUBLE PRECISION,
    cross_er_min            DOUBLE PRECISION,
    adx_min                 DOUBLE PRECISION,
    session_filter          BOOLEAN,
    blocked_hours_utc       VARCHAR(100),
    signal_cooldown_minutes DOUBLE PRECISION,

    -- Arbitrary future overrides via JSONB (avoids schema churn)
    parameter_overrides     JSONB        DEFAULT '{}',
    notes                   TEXT,
    created_at              TIMESTAMPTZ  DEFAULT now(),
    updated_at              TIMESTAMPTZ  DEFAULT now(),

    UNIQUE (config_set, epic)
);

-- Seed rows for all FX pairs (AUDUSD active monitor-only, rest disabled pending BT screening)
INSERT INTO kama_v2_pair_overrides
    (config_set, epic, pair_name, is_enabled, monitor_only, notes)
VALUES
    ('demo', 'CS.D.AUDUSD.MINI.IP', 'AUDUSD', true,  true, 'Launched monitor-only 2026-05-29. Research: n=93 WR=55.9% PF=1.95'),
    ('demo', 'CS.D.EURUSD.CEEM.IP', 'EURUSD', false, true, 'Pending 90d BT screening'),
    ('demo', 'CS.D.GBPUSD.MINI.IP', 'GBPUSD', false, true, 'Pending 90d BT screening'),
    ('demo', 'CS.D.USDJPY.MINI.IP', 'USDJPY', false, true, 'Pending 90d BT screening'),
    ('demo', 'CS.D.EURJPY.MINI.IP', 'EURJPY', false, true, 'Pending 90d BT screening'),
    ('demo', 'CS.D.AUDJPY.MINI.IP', 'AUDJPY', false, true, 'Pending 90d BT screening'),
    ('demo', 'CS.D.USDCAD.MINI.IP', 'USDCAD', false, true, 'Pending 90d BT screening'),
    ('demo', 'CS.D.NZDUSD.MINI.IP', 'NZDUSD', false, true, 'Pending 90d BT screening'),
    ('demo', 'CS.D.USDCHF.MINI.IP', 'USDCHF', false, true, 'Pending 90d BT screening')
ON CONFLICT (config_set, epic) DO NOTHING;
