-- ============================================================================
-- RANGE_FADE Strategy Database Configuration
-- ============================================================================
-- Source of truth for both ERF (15m) and ERF5 (5m) profiles.
--
-- Apply:
--   docker exec postgres psql -U postgres -d strategy_config \
--     -f /app/forex_scanner/migrations/create_eurusd_range_fade_config.sql
-- ============================================================================

CREATE TABLE IF NOT EXISTS eurusd_range_fade_global_config (
    id                          SERIAL PRIMARY KEY,
    profile_name                VARCHAR(8)    NOT NULL,
    is_active                   BOOLEAN       NOT NULL DEFAULT TRUE,
    strategy_name               VARCHAR(50)   NOT NULL,
    version                     VARCHAR(16)   NOT NULL DEFAULT '0.3.0',
    monitor_only                BOOLEAN       NOT NULL DEFAULT TRUE,
    primary_timeframe           VARCHAR(8)    NOT NULL,
    confirmation_timeframe      VARCHAR(8)    NOT NULL DEFAULT '1h',
    bb_period                   INTEGER       NOT NULL DEFAULT 20,
    bb_mult                     NUMERIC(4,2)  NOT NULL DEFAULT 2.0,
    rsi_period                  INTEGER       NOT NULL DEFAULT 14,
    rsi_oversold                INTEGER       NOT NULL,
    rsi_overbought              INTEGER       NOT NULL,
    range_lookback_bars         INTEGER       NOT NULL,
    range_proximity_pips        NUMERIC(5,2)  NOT NULL,
    min_band_width_pips         NUMERIC(5,2)  NOT NULL,
    max_band_width_pips         NUMERIC(5,2)  NOT NULL,
    htf_ema_period              INTEGER       NOT NULL DEFAULT 50,
    htf_slope_bars              INTEGER       NOT NULL DEFAULT 3,
    allow_neutral_htf           BOOLEAN       NOT NULL,
    max_current_range_pips      NUMERIC(5,2)  NOT NULL,
    min_confidence              NUMERIC(4,3)  NOT NULL,
    max_confidence              NUMERIC(4,3)  NOT NULL,
    fixed_stop_loss_pips        NUMERIC(5,2)  NOT NULL,
    fixed_take_profit_pips      NUMERIC(5,2)  NOT NULL,
    signal_cooldown_minutes     INTEGER       NOT NULL,
    london_start_hour_utc       INTEGER       NOT NULL,
    new_york_end_hour_utc       INTEGER       NOT NULL,
    notes                       TEXT,
    created_at                  TIMESTAMPTZ   DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ   DEFAULT NOW(),
    UNIQUE (profile_name, is_active)
);

CREATE TABLE IF NOT EXISTS eurusd_range_fade_pair_overrides (
    id                          SERIAL PRIMARY KEY,
    epic                        VARCHAR(50)   NOT NULL,
    pair_name                   VARCHAR(16),
    profile_name                VARCHAR(8)    NOT NULL,
    is_enabled                  BOOLEAN       NOT NULL DEFAULT TRUE,
    is_traded                   BOOLEAN       NOT NULL DEFAULT FALSE,
    monitor_only                BOOLEAN       NOT NULL DEFAULT TRUE,
    signal_cooldown_minutes     INTEGER,
    rsi_oversold                INTEGER,
    rsi_overbought              INTEGER,
    range_lookback_bars         INTEGER,
    range_proximity_pips        NUMERIC(5,2),
    max_current_range_pips      NUMERIC(5,2),
    fixed_stop_loss_pips        NUMERIC(5,2),
    fixed_take_profit_pips      NUMERIC(5,2),
    london_start_hour_utc       INTEGER,
    new_york_end_hour_utc       INTEGER,
    allow_neutral_htf           BOOLEAN,
    parameter_overrides         JSONB         DEFAULT '{}'::jsonb,
    notes                       TEXT,
    disabled_reason             TEXT,
    created_at                  TIMESTAMPTZ   DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ   DEFAULT NOW(),
    UNIQUE (epic, profile_name)
);

INSERT INTO eurusd_range_fade_global_config (
    profile_name, is_active, strategy_name, primary_timeframe, confirmation_timeframe,
    rsi_oversold, rsi_overbought, range_lookback_bars, range_proximity_pips,
    min_band_width_pips, max_band_width_pips, allow_neutral_htf,
    max_current_range_pips, min_confidence, max_confidence,
    fixed_stop_loss_pips, fixed_take_profit_pips, signal_cooldown_minutes,
    london_start_hour_utc, new_york_end_hour_utc, notes
)
SELECT
    '15m', TRUE, 'RANGE_FADE', '15m', '1h',
    30, 70, 48, 4.0,
    8.0, 45.0, TRUE,
    16.0, 0.52, 0.84,
    8.0, 12.0, 45,
    8, 18, 'Initial 15m ERF seed'
WHERE NOT EXISTS (
    SELECT 1 FROM eurusd_range_fade_global_config WHERE profile_name = '15m' AND is_active = TRUE
);

INSERT INTO eurusd_range_fade_global_config (
    profile_name, is_active, strategy_name, primary_timeframe, confirmation_timeframe,
    rsi_oversold, rsi_overbought, range_lookback_bars, range_proximity_pips,
    min_band_width_pips, max_band_width_pips, allow_neutral_htf,
    max_current_range_pips, min_confidence, max_confidence,
    fixed_stop_loss_pips, fixed_take_profit_pips, signal_cooldown_minutes,
    london_start_hour_utc, new_york_end_hour_utc, notes
)
SELECT
    '5m', TRUE, 'RANGE_FADE_5M', '5m', '1h',
    32, 68, 144, 3.0,
    6.0, 28.0, FALSE,
    12.0, 0.52, 0.84,
    8.0, 12.0, 30,
    6, 18, 'Initial 5m ERF5 seed'
WHERE NOT EXISTS (
    SELECT 1 FROM eurusd_range_fade_global_config WHERE profile_name = '5m' AND is_active = TRUE
);

INSERT INTO eurusd_range_fade_pair_overrides (
    epic, pair_name, profile_name, is_enabled, is_traded, monitor_only, notes
)
VALUES
    ('CS.D.EURUSD.CEEM.IP', 'EURUSD', '15m', TRUE, FALSE, TRUE, 'Initial ERF 15m monitor-only seed'),
    ('CS.D.EURUSD.CEEM.IP', 'EURUSD', '5m', TRUE, FALSE, TRUE, 'Initial ERF5 monitor-only seed')
ON CONFLICT (epic, profile_name) DO NOTHING;

UPDATE eurusd_range_fade_global_config
SET strategy_name = CASE
    WHEN strategy_name = 'EURUSD_RANGE_FADE' THEN 'RANGE_FADE'
    WHEN strategy_name = 'EURUSD_RANGE_FADE_5M' THEN 'RANGE_FADE_5M'
    ELSE strategy_name
END
WHERE strategy_name IN ('EURUSD_RANGE_FADE', 'EURUSD_RANGE_FADE_5M');

SELECT 'RANGE_FADE configuration tables created' AS status;
