-- ============================================================================
-- FREEDOMSCALP Strategy Database Configuration (Jul 9 2026)
-- Gaussian trend-flip scalp on gold — TradingView "FreedomScalp V3" port.
-- Demo-only forward test; backtest-refuted (see freedomscalp_lab.py), wired
-- to arbitrate the TV-fill-model dispute with real demo fills.
-- Gold pip convention: 1 pip = $0.10 (TP $4.20 = 42 pips, SL $6.00 = 60 pips)
-- ============================================================================

CREATE TABLE IF NOT EXISTS freedomscalp_global_config (
    id SERIAL PRIMARY KEY,
    parameter_name VARCHAR(100) NOT NULL UNIQUE,
    parameter_value TEXT NOT NULL,
    value_type VARCHAR(20) NOT NULL DEFAULT 'string',
    category VARCHAR(50) NOT NULL DEFAULT 'general',
    display_order INTEGER DEFAULT 0,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    is_editable BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

INSERT INTO freedomscalp_global_config
    (parameter_name, parameter_value, value_type, category, description, display_order)
VALUES
    ('strategy_name', 'FREEDOMSCALP', 'string', 'general', 'Strategy name', 1),
    ('version', '1.0.0', 'string', 'general', 'Strategy version', 2),

    ('entry_timeframe', '5m', 'string', 'timeframes', 'Signal timeframe', 10),

    -- Gaussian core (TradingView defaults)
    ('gaussian_length', '8', 'int', 'core', 'Gaussian filter length', 20),
    ('gaussian_poles', '2', 'int', 'core', 'Gaussian filter poles (1-4)', 21),
    ('smoothing_length', '10', 'int', 'core', 'linreg smoothing length', 22),
    ('flatten_offset', '3', 'int', 'core', 'linreg flatten offset', 23),
    ('supertrend_factor', '0.15', 'float', 'core', 'SuperTrend factor on smoothed line', 24),
    ('supertrend_atr_period', '21', 'int', 'core', 'SuperTrend ATR period', 25),

    -- Filters (TV defaults: EMA + spike + session ON, regime gates OFF)
    ('ema_filter_enabled', 'true', 'bool', 'filters', 'EMA regime gate', 30),
    ('ema_period', '50', 'int', 'filters', 'EMA gate period', 31),
    ('spike_filter_enabled', 'true', 'bool', 'filters', 'Block spike bars', 32),
    ('spike_atr_mult', '3.0', 'float', 'filters', 'Spike bar ATR multiple', 33),
    ('adx_gate_enabled', 'false', 'bool', 'filters', 'ADX regime gate', 34),
    ('adx_threshold', '20.0', 'float', 'filters', 'ADX gate threshold', 35),
    ('atr_expansion_enabled', 'false', 'bool', 'filters', 'ATR expansion gate', 36),
    ('atr_expansion_mult', '1.0', 'float', 'filters', 'ATR expansion multiple', 37),

    -- Session (UTC; Pine default 0600-2030 UTC+2)
    ('session_enabled', 'true', 'bool', 'session', 'Session window filter', 40),
    ('session_start_utc', '400', 'int', 'session', 'Session start HHMM UTC', 41),
    ('session_end_utc', '1830', 'int', 'session', 'Session end HHMM UTC', 42),

    -- Risk (gold pips = $0.10)
    ('fixed_stop_loss_pips', '60.0', 'float', 'risk', 'SL in gold pips ($6.00)', 50),
    ('fixed_take_profit_pips', '42.0', 'float', 'risk', 'TP in gold pips ($4.20)', 51),

    -- Confidence
    ('min_confidence', '0.60', 'float', 'confidence', 'Minimum confidence', 60),
    ('base_confidence', '0.62', 'float', 'confidence', 'Base signal confidence', 61),

    -- Cooldown
    ('signal_cooldown_minutes', '30', 'int', 'cooldown', 'Per-pair signal cooldown', 70)
ON CONFLICT (parameter_name) DO NOTHING;


CREATE TABLE IF NOT EXISTS freedomscalp_pair_overrides (
    id SERIAL PRIMARY KEY,
    epic VARCHAR(50) NOT NULL UNIQUE,
    pair_name VARCHAR(10),
    fixed_stop_loss_pips FLOAT,
    fixed_take_profit_pips FLOAT,
    min_confidence FLOAT,
    signal_cooldown_minutes INTEGER,
    is_enabled BOOLEAN DEFAULT TRUE,
    is_traded BOOLEAN DEFAULT TRUE,
    parameter_overrides JSONB DEFAULT '{}'::jsonb,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Gold only. Demo forward test, MONITOR-ONLY (signals logged, no execution).
INSERT INTO freedomscalp_pair_overrides
    (epic, pair_name, is_enabled, is_traded, parameter_overrides, notes)
VALUES
    ('CS.D.CFEGOLD.CEE.IP', 'XAUUSD', TRUE, TRUE,
     '{"monitor_only": true}'::jsonb,
     'Monitor-only forward test Jul 9 2026 — fill-model arbitration. Backtest-refuted at pessimistic fills; TV fills PF 1.59 on 2026. Flip monitor_only off only after signal-quality review; kill if PF < 1.0 after 30 outcomes.')
ON CONFLICT (epic) DO NOTHING;

CREATE INDEX IF NOT EXISTS idx_freedomscalp_global_config_active
    ON freedomscalp_global_config(is_active);
CREATE INDEX IF NOT EXISTS idx_freedomscalp_pair_overrides_enabled
    ON freedomscalp_pair_overrides(is_enabled);

SELECT 'FREEDOMSCALP strategy configuration tables created successfully' AS status;
