-- Multi-Strategy Routing System Migration
-- Creates all tables for adaptive multi-strategy trading system
-- All configuration is database-driven (no local config files)
-- Run against: strategy_config database

-- ============================================================================
-- TABLE 1: enabled_strategies
-- Master table controlling which strategies are available system-wide
-- ============================================================================
CREATE TABLE IF NOT EXISTS enabled_strategies (
    strategy_name VARCHAR(50) PRIMARY KEY,
    is_enabled BOOLEAN DEFAULT FALSE,
    is_backtest_only BOOLEAN DEFAULT TRUE,  -- If TRUE, only use in backtest, not live
    display_name VARCHAR(100),
    description TEXT,
    strategy_type VARCHAR(30) DEFAULT 'signal',  -- signal, filter, hybrid
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Seed initial strategies
INSERT INTO enabled_strategies (strategy_name, is_enabled, is_backtest_only, display_name, description, strategy_type)
VALUES
    ('SMC_SIMPLE', TRUE, FALSE, 'SMC Simple', 'Primary Smart Money Concepts strategy for trending markets', 'signal'),
    ('RANGING_MARKET', FALSE, TRUE, 'Ranging Market', 'Multi-oscillator strategy for ADX < 20 conditions', 'signal'),
    ('VOLUME_PROFILE', FALSE, TRUE, 'Volume Profile', 'HVN/POC-based strategy, strong in Asian session', 'signal'),
    ('MEAN_REVERSION', FALSE, TRUE, 'Mean Reversion', 'LuxAlgo-based multi-oscillator mean reversion', 'signal'),
    ('BB_SUPERTREND', FALSE, TRUE, 'Bollinger + Supertrend', 'Squeeze detection with Supertrend confirmation', 'signal')
ON CONFLICT (strategy_name) DO NOTHING;

-- ============================================================================
-- TABLE 2: strategy_routing_rules
-- Maps market regimes to appropriate strategies with priority ordering
-- ============================================================================
CREATE TABLE IF NOT EXISTS strategy_routing_rules (
    id SERIAL PRIMARY KEY,
    regime VARCHAR(50) NOT NULL,           -- trending, ranging, breakout, high_volatility, low_volatility
    session VARCHAR(50),                   -- asian, london, new_york, NULL = any session
    volatility_state VARCHAR(30),          -- high, normal, low, NULL = any
    adx_min DECIMAL(5,2),                  -- Minimum ADX for this rule (NULL = no minimum)
    adx_max DECIMAL(5,2),                  -- Maximum ADX for this rule (NULL = no maximum)
    strategy_name VARCHAR(50) NOT NULL,
    priority INTEGER NOT NULL DEFAULT 100, -- Lower = higher priority
    min_win_rate DECIMAL(4,3) DEFAULT 0.40,
    min_sample_size INTEGER DEFAULT 10,
    is_active BOOLEAN DEFAULT TRUE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (strategy_name) REFERENCES enabled_strategies(strategy_name) ON DELETE CASCADE,
    UNIQUE(regime, session, volatility_state, strategy_name)
);

-- Create index for fast lookups
CREATE INDEX IF NOT EXISTS idx_routing_rules_lookup
ON strategy_routing_rules(regime, session, is_active, priority);

-- Seed initial routing rules based on regime-strategy mapping from plan
INSERT INTO strategy_routing_rules (regime, session, volatility_state, adx_min, adx_max, strategy_name, priority, min_win_rate, notes)
VALUES
    -- Trending regime: SMC Simple is primary
    ('trending', NULL, NULL, 20, NULL, 'SMC_SIMPLE', 10, 0.50, 'Primary strategy for trending markets'),

    -- Ranging regime: Use Ranging Market, fallback to Mean Reversion
    ('ranging', NULL, NULL, NULL, 20, 'RANGING_MARKET', 10, 0.45, 'Primary for ranging markets (ADX < 20)'),
    ('ranging', NULL, NULL, NULL, 20, 'MEAN_REVERSION', 20, 0.45, 'Fallback for ranging markets'),

    -- High volatility: Volume Profile primary, BB+Supertrend fallback
    ('high_volatility', NULL, 'high', NULL, NULL, 'VOLUME_PROFILE', 10, 0.45, 'HVN bounce strategy for high vol'),
    ('high_volatility', NULL, 'high', NULL, NULL, 'BB_SUPERTREND', 20, 0.45, 'Squeeze expansion for high vol'),

    -- Breakout regime: BB+Supertrend primary
    ('breakout', NULL, NULL, NULL, NULL, 'BB_SUPERTREND', 10, 0.45, 'Squeeze breakout detection'),
    ('breakout', NULL, NULL, NULL, NULL, 'VOLUME_PROFILE', 20, 0.45, 'Value area breakouts'),

    -- Asian session: Volume Profile has 66.7% edge discovered
    ('trending', 'asian', NULL, NULL, NULL, 'VOLUME_PROFILE', 5, 0.55, 'Asian session edge (66.7% WR discovered)'),
    ('ranging', 'asian', NULL, NULL, NULL, 'VOLUME_PROFILE', 5, 0.55, 'Asian session edge'),

    -- Low volatility: Conservative SMC Simple or monitor-only
    ('low_volatility', NULL, 'low', NULL, NULL, 'SMC_SIMPLE', 10, 0.50, 'Conservative mode in low vol')
ON CONFLICT DO NOTHING;

-- ============================================================================
-- TABLE 3: strategy_regime_performance
-- Tracks rolling performance of each strategy per regime/epic combination
-- ============================================================================
CREATE TABLE IF NOT EXISTS strategy_regime_performance (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(50) NOT NULL,
    epic VARCHAR(50) NOT NULL,
    regime VARCHAR(50) NOT NULL,
    window_days INTEGER NOT NULL,          -- 7, 14, 30 day windows

    -- Performance metrics
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4),
    profit_factor DECIMAL(6,3),
    total_pips DECIMAL(10,2) DEFAULT 0,
    avg_win_pips DECIMAL(8,2),
    avg_loss_pips DECIMAL(8,2),
    max_drawdown_pips DECIMAL(8,2),
    sharpe_ratio DECIMAL(6,3),
    r_multiple DECIMAL(6,3),

    -- Bayesian fitness (self-tuning)
    fitness_score DECIMAL(5,4),            -- 0.0 to 1.0
    confidence_modifier DECIMAL(4,3),      -- Applied to signal confidence

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    trading_mode VARCHAR(20) DEFAULT 'MONITOR_ONLY',  -- ACTIVE, REDUCED, MONITOR_ONLY, DISABLED
    cooldown_until TIMESTAMP,
    consecutive_losses INTEGER DEFAULT 0,

    -- Timestamps
    period_start TIMESTAMP,
    period_end TIMESTAMP,
    last_trade_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    FOREIGN KEY (strategy_name) REFERENCES enabled_strategies(strategy_name) ON DELETE CASCADE,
    UNIQUE(strategy_name, epic, regime, window_days)
);

-- Indexes for performance queries
CREATE INDEX IF NOT EXISTS idx_regime_perf_lookup
ON strategy_regime_performance(strategy_name, epic, regime, is_active);

CREATE INDEX IF NOT EXISTS idx_regime_perf_fitness
ON strategy_regime_performance(fitness_score DESC, trading_mode);

-- ============================================================================
-- TABLE 4: regime_fitness_scores
-- Aggregated fitness scores by strategy/regime for quick routing decisions
-- ============================================================================
CREATE TABLE IF NOT EXISTS regime_fitness_scores (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(50) NOT NULL,
    regime VARCHAR(50) NOT NULL,
    volatility_state VARCHAR(20),          -- high, normal, low
    session VARCHAR(30),                   -- asian, london, new_york

    -- Weighted fitness across all epics
    fitness_score DECIMAL(5,4),            -- 0.0 to 1.0
    sample_size INTEGER DEFAULT 0,

    -- Trading decision
    trading_mode VARCHAR(20) DEFAULT 'MONITOR_ONLY',
    confidence_modifier DECIMAL(4,3) DEFAULT 1.0,

    -- Circuit breaker
    consecutive_losses INTEGER DEFAULT 0,
    cooldown_until TIMESTAMP,
    switches_last_48h INTEGER DEFAULT 0,
    last_switch_at TIMESTAMP,

    -- Timestamps
    last_updated TIMESTAMP DEFAULT NOW(),

    FOREIGN KEY (strategy_name) REFERENCES enabled_strategies(strategy_name) ON DELETE CASCADE,
    UNIQUE(strategy_name, regime, volatility_state, session)
);

CREATE INDEX IF NOT EXISTS idx_fitness_routing
ON regime_fitness_scores(regime, volatility_state, session, fitness_score DESC);

-- ============================================================================
-- TABLE 5: strategy_switch_log
-- Audit trail of strategy switches for analysis
-- ============================================================================
CREATE TABLE IF NOT EXISTS strategy_switch_log (
    id SERIAL PRIMARY KEY,
    epic VARCHAR(50) NOT NULL,
    from_strategy VARCHAR(50),
    to_strategy VARCHAR(50) NOT NULL,
    regime VARCHAR(50),
    session VARCHAR(30),
    adx_value DECIMAL(6,2),
    volatility_state VARCHAR(20),

    -- Decision context
    from_fitness DECIMAL(5,4),
    to_fitness DECIMAL(5,4),
    switch_reason TEXT,                    -- 'regime_change', 'performance_degradation', 'cooldown_expired'

    created_at TIMESTAMP DEFAULT NOW(),

    FOREIGN KEY (from_strategy) REFERENCES enabled_strategies(strategy_name) ON DELETE SET NULL,
    FOREIGN KEY (to_strategy) REFERENCES enabled_strategies(strategy_name) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_switch_log_time ON strategy_switch_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_switch_log_epic ON strategy_switch_log(epic, created_at DESC);

-- ============================================================================
-- TABLE 6: ranging_market_global_config
-- Configuration for Ranging Market Strategy (database-driven)
-- ============================================================================
CREATE TABLE IF NOT EXISTS ranging_market_global_config (
    id SERIAL PRIMARY KEY,
    config_version INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN DEFAULT FALSE,

    -- ADX Filter (core condition for ranging)
    adx_max_threshold DECIMAL(5,2) DEFAULT 20.0,
    adx_period INTEGER DEFAULT 14,

    -- Squeeze Momentum Settings
    squeeze_bb_length INTEGER DEFAULT 20,
    squeeze_bb_mult DECIMAL(4,2) DEFAULT 2.0,
    squeeze_kc_length INTEGER DEFAULT 20,
    squeeze_kc_mult DECIMAL(4,2) DEFAULT 1.5,
    squeeze_momentum_length INTEGER DEFAULT 12,
    squeeze_signal_weight DECIMAL(4,3) DEFAULT 0.30,

    -- Wave Trend Settings
    wavetrend_channel_length INTEGER DEFAULT 9,
    wavetrend_avg_length INTEGER DEFAULT 12,
    wavetrend_ob_level1 INTEGER DEFAULT 53,
    wavetrend_ob_level2 INTEGER DEFAULT 60,
    wavetrend_os_level1 INTEGER DEFAULT -53,
    wavetrend_os_level2 INTEGER DEFAULT -60,
    wavetrend_signal_weight DECIMAL(4,3) DEFAULT 0.25,

    -- RSI Settings
    rsi_period INTEGER DEFAULT 14,
    rsi_overbought INTEGER DEFAULT 70,
    rsi_oversold INTEGER DEFAULT 30,
    rsi_signal_weight DECIMAL(4,3) DEFAULT 0.25,

    -- RVI Settings (Relative Vigor Index)
    rvi_period INTEGER DEFAULT 10,
    rvi_signal_period INTEGER DEFAULT 4,
    rvi_signal_weight DECIMAL(4,3) DEFAULT 0.20,

    -- Signal Generation
    min_oscillator_agreement INTEGER DEFAULT 2,  -- Minimum oscillators agreeing
    min_combined_score DECIMAL(4,3) DEFAULT 0.55,

    -- Support/Resistance Bounce
    sr_bounce_required BOOLEAN DEFAULT TRUE,
    sr_proximity_pips DECIMAL(5,2) DEFAULT 5.0,

    -- Risk Management
    fixed_stop_loss_pips DECIMAL(5,2) DEFAULT 12.0,
    fixed_take_profit_pips DECIMAL(5,2) DEFAULT 18.0,
    min_confidence DECIMAL(4,3) DEFAULT 0.50,
    max_confidence DECIMAL(4,3) DEFAULT 0.85,

    -- Timeframes
    primary_timeframe VARCHAR(10) DEFAULT '15m',
    confirmation_timeframe VARCHAR(10) DEFAULT '1h',

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Seed default configuration
INSERT INTO ranging_market_global_config (config_version, is_active)
VALUES (1, FALSE)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- TABLE 7: ranging_market_pair_overrides
-- Per-pair overrides for Ranging Market Strategy
-- ============================================================================
CREATE TABLE IF NOT EXISTS ranging_market_pair_overrides (
    id SERIAL PRIMARY KEY,
    epic VARCHAR(50) NOT NULL UNIQUE,
    is_enabled BOOLEAN DEFAULT TRUE,

    -- Override parameters (NULL = use global)
    adx_max_threshold DECIMAL(5,2),
    fixed_stop_loss_pips DECIMAL(5,2),
    fixed_take_profit_pips DECIMAL(5,2),
    min_confidence DECIMAL(4,3),
    max_confidence DECIMAL(4,3),
    min_combined_score DECIMAL(4,3),

    -- Additional overrides as JSON
    parameter_overrides JSONB DEFAULT '{}',

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Seed pairs (matching SMC Simple enabled pairs)
INSERT INTO ranging_market_pair_overrides (epic, is_enabled)
SELECT epic, TRUE FROM smc_simple_pair_overrides WHERE is_enabled = TRUE
ON CONFLICT (epic) DO NOTHING;

-- ============================================================================
-- TABLE 8: volume_profile_global_config
-- Configuration for Volume Profile Strategy (database-driven)
-- ============================================================================
CREATE TABLE IF NOT EXISTS volume_profile_global_config (
    id SERIAL PRIMARY KEY,
    config_version INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN DEFAULT FALSE,

    -- Volume Profile Settings
    vp_lookback_bars INTEGER DEFAULT 100,
    vp_value_area_percent DECIMAL(4,2) DEFAULT 70.0,
    hvn_threshold_percentile INTEGER DEFAULT 80,
    lvn_threshold_percentile INTEGER DEFAULT 20,

    -- POC Settings
    poc_proximity_pips DECIMAL(5,2) DEFAULT 5.0,
    poc_reversion_enabled BOOLEAN DEFAULT TRUE,

    -- Value Area Settings
    va_high_proximity_pips DECIMAL(5,2) DEFAULT 8.0,
    va_low_proximity_pips DECIMAL(5,2) DEFAULT 8.0,
    va_breakout_enabled BOOLEAN DEFAULT TRUE,

    -- HVN Bounce Settings
    hvn_bounce_enabled BOOLEAN DEFAULT TRUE,
    hvn_bounce_proximity_pips DECIMAL(5,2) DEFAULT 5.0,

    -- Session Filters (Asian session edge)
    asian_session_boost DECIMAL(4,3) DEFAULT 1.15,  -- 15% confidence boost
    london_session_boost DECIMAL(4,3) DEFAULT 1.0,
    ny_session_boost DECIMAL(4,3) DEFAULT 1.0,

    -- Risk Management
    fixed_stop_loss_pips DECIMAL(5,2) DEFAULT 15.0,
    fixed_take_profit_pips DECIMAL(5,2) DEFAULT 22.0,
    min_confidence DECIMAL(4,3) DEFAULT 0.50,
    max_confidence DECIMAL(4,3) DEFAULT 0.85,

    -- Timeframes
    primary_timeframe VARCHAR(10) DEFAULT '15m',
    profile_timeframe VARCHAR(10) DEFAULT '1h',

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Seed default configuration
INSERT INTO volume_profile_global_config (config_version, is_active)
VALUES (1, FALSE)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- TABLE 9: volume_profile_pair_overrides
-- Per-pair overrides for Volume Profile Strategy
-- ============================================================================
CREATE TABLE IF NOT EXISTS volume_profile_pair_overrides (
    id SERIAL PRIMARY KEY,
    epic VARCHAR(50) NOT NULL UNIQUE,
    is_enabled BOOLEAN DEFAULT TRUE,

    -- Override parameters (NULL = use global)
    fixed_stop_loss_pips DECIMAL(5,2),
    fixed_take_profit_pips DECIMAL(5,2),
    min_confidence DECIMAL(4,3),
    max_confidence DECIMAL(4,3),
    asian_session_boost DECIMAL(4,3),

    -- Additional overrides as JSON
    parameter_overrides JSONB DEFAULT '{}',

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Seed pairs
INSERT INTO volume_profile_pair_overrides (epic, is_enabled)
SELECT epic, TRUE FROM smc_simple_pair_overrides WHERE is_enabled = TRUE
ON CONFLICT (epic) DO NOTHING;

-- ============================================================================
-- TABLE 10: mean_reversion_global_config
-- Configuration for Mean Reversion Strategy (database-driven)
-- ============================================================================
CREATE TABLE IF NOT EXISTS mean_reversion_global_config (
    id SERIAL PRIMARY KEY,
    config_version INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN DEFAULT FALSE,

    -- LuxAlgo Premium Settings
    luxalgo_length INTEGER DEFAULT 14,
    luxalgo_sensitivity DECIMAL(4,2) DEFAULT 2.0,

    -- Multi-TF RSI Settings
    rsi_fast_period INTEGER DEFAULT 7,
    rsi_slow_period INTEGER DEFAULT 21,
    rsi_overbought INTEGER DEFAULT 70,
    rsi_oversold INTEGER DEFAULT 30,

    -- RSI-EMA Divergence
    rsi_ema_period INTEGER DEFAULT 9,
    divergence_lookback INTEGER DEFAULT 5,
    divergence_enabled BOOLEAN DEFAULT TRUE,

    -- Bollinger Bands for extremes
    bb_period INTEGER DEFAULT 20,
    bb_std_dev DECIMAL(4,2) DEFAULT 2.0,
    bb_touch_required BOOLEAN DEFAULT TRUE,

    -- Signal Requirements
    min_divergence_score DECIMAL(4,3) DEFAULT 0.60,
    min_oscillator_agreement INTEGER DEFAULT 2,

    -- Risk Management
    fixed_stop_loss_pips DECIMAL(5,2) DEFAULT 12.0,
    fixed_take_profit_pips DECIMAL(5,2) DEFAULT 15.0,
    min_confidence DECIMAL(4,3) DEFAULT 0.50,
    max_confidence DECIMAL(4,3) DEFAULT 0.80,

    -- Timeframes
    primary_timeframe VARCHAR(10) DEFAULT '15m',
    divergence_timeframe VARCHAR(10) DEFAULT '1h',

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Seed default configuration
INSERT INTO mean_reversion_global_config (config_version, is_active)
VALUES (1, FALSE)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- TABLE 11: mean_reversion_pair_overrides
-- ============================================================================
CREATE TABLE IF NOT EXISTS mean_reversion_pair_overrides (
    id SERIAL PRIMARY KEY,
    epic VARCHAR(50) NOT NULL UNIQUE,
    is_enabled BOOLEAN DEFAULT TRUE,

    fixed_stop_loss_pips DECIMAL(5,2),
    fixed_take_profit_pips DECIMAL(5,2),
    min_confidence DECIMAL(4,3),
    max_confidence DECIMAL(4,3),

    parameter_overrides JSONB DEFAULT '{}',

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO mean_reversion_pair_overrides (epic, is_enabled)
SELECT epic, TRUE FROM smc_simple_pair_overrides WHERE is_enabled = TRUE
ON CONFLICT (epic) DO NOTHING;

-- ============================================================================
-- TABLE 12: bb_supertrend_global_config
-- Configuration for Bollinger + Supertrend Strategy (database-driven)
-- ============================================================================
CREATE TABLE IF NOT EXISTS bb_supertrend_global_config (
    id SERIAL PRIMARY KEY,
    config_version INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN DEFAULT FALSE,

    -- Bollinger Bands Settings
    bb_period INTEGER DEFAULT 20,
    bb_std_dev DECIMAL(4,2) DEFAULT 2.0,

    -- Keltner Channels (for squeeze detection)
    kc_period INTEGER DEFAULT 20,
    kc_atr_mult DECIMAL(4,2) DEFAULT 1.5,

    -- Squeeze Detection
    squeeze_lookback INTEGER DEFAULT 5,
    squeeze_min_bars INTEGER DEFAULT 3,       -- Min bars in squeeze before signal

    -- Supertrend Settings
    supertrend_period INTEGER DEFAULT 10,
    supertrend_multiplier DECIMAL(4,2) DEFAULT 3.0,

    -- Expansion Settings
    expansion_threshold DECIMAL(4,3) DEFAULT 1.5,  -- BB width expansion ratio
    expansion_confirmation_bars INTEGER DEFAULT 2,

    -- Signal Requirements
    require_squeeze_before BOOLEAN DEFAULT TRUE,
    require_supertrend_confirm BOOLEAN DEFAULT TRUE,
    min_squeeze_duration INTEGER DEFAULT 3,

    -- Risk Management
    fixed_stop_loss_pips DECIMAL(5,2) DEFAULT 15.0,
    fixed_take_profit_pips DECIMAL(5,2) DEFAULT 25.0,
    min_confidence DECIMAL(4,3) DEFAULT 0.50,
    max_confidence DECIMAL(4,3) DEFAULT 0.85,

    -- Timeframes
    primary_timeframe VARCHAR(10) DEFAULT '15m',
    confirmation_timeframe VARCHAR(10) DEFAULT '1h',

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Seed default configuration
INSERT INTO bb_supertrend_global_config (config_version, is_active)
VALUES (1, FALSE)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- TABLE 13: bb_supertrend_pair_overrides
-- ============================================================================
CREATE TABLE IF NOT EXISTS bb_supertrend_pair_overrides (
    id SERIAL PRIMARY KEY,
    epic VARCHAR(50) NOT NULL UNIQUE,
    is_enabled BOOLEAN DEFAULT TRUE,

    fixed_stop_loss_pips DECIMAL(5,2),
    fixed_take_profit_pips DECIMAL(5,2),
    min_confidence DECIMAL(4,3),
    max_confidence DECIMAL(4,3),
    supertrend_multiplier DECIMAL(4,2),

    parameter_overrides JSONB DEFAULT '{}',

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO bb_supertrend_pair_overrides (epic, is_enabled)
SELECT epic, TRUE FROM smc_simple_pair_overrides WHERE is_enabled = TRUE
ON CONFLICT (epic) DO NOTHING;

-- ============================================================================
-- TABLE 14: router_global_config
-- Global configuration for the Strategy Router itself
-- ============================================================================
CREATE TABLE IF NOT EXISTS router_global_config (
    id SERIAL PRIMARY KEY,
    config_version INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,

    -- Routing Behavior
    multi_strategy_enabled BOOLEAN DEFAULT FALSE,  -- Master switch for multi-strategy
    fallback_to_smc_simple BOOLEAN DEFAULT TRUE,   -- If no strategy matches, use SMC Simple

    -- Fitness Calculation Weights (Bayesian formula)
    fitness_weight_win_rate DECIMAL(4,3) DEFAULT 0.30,
    fitness_weight_profit_factor DECIMAL(4,3) DEFAULT 0.30,
    fitness_weight_sharpe DECIMAL(4,3) DEFAULT 0.20,
    fitness_weight_r_multiple DECIMAL(4,3) DEFAULT 0.20,

    -- Rolling Window Weights
    window_7d_weight DECIMAL(4,3) DEFAULT 0.50,
    window_14d_weight DECIMAL(4,3) DEFAULT 0.30,
    window_30d_weight DECIMAL(4,3) DEFAULT 0.20,

    -- Trading Mode Thresholds
    active_min_fitness DECIMAL(4,3) DEFAULT 0.65,
    reduced_min_fitness DECIMAL(4,3) DEFAULT 0.35,
    monitor_min_fitness DECIMAL(4,3) DEFAULT 0.0,

    -- Confidence Modifiers by Mode
    active_confidence_modifier DECIMAL(4,3) DEFAULT 1.0,
    active_max_confidence_boost DECIMAL(4,3) DEFAULT 1.3,
    reduced_confidence_modifier DECIMAL(4,3) DEFAULT 0.7,

    -- Circuit Breaker Settings
    consecutive_loss_limit INTEGER DEFAULT 3,
    cooldown_hours INTEGER DEFAULT 2,
    max_drawdown_percent DECIMAL(5,2) DEFAULT 15.0,
    max_switches_per_48h INTEGER DEFAULT 3,
    min_hours_between_switches INTEGER DEFAULT 12,

    -- Performance Requirements
    min_sample_size INTEGER DEFAULT 10,
    min_win_rate DECIMAL(4,3) DEFAULT 0.40,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Seed default configuration
INSERT INTO router_global_config (config_version, is_active, multi_strategy_enabled)
VALUES (1, TRUE, FALSE)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to check if a strategy is enabled for live trading
CREATE OR REPLACE FUNCTION is_strategy_enabled_for_live(p_strategy_name VARCHAR)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM enabled_strategies
        WHERE strategy_name = p_strategy_name
        AND is_enabled = TRUE
        AND is_backtest_only = FALSE
    );
END;
$$ LANGUAGE plpgsql;

-- Function to check if a strategy is enabled for backtest
CREATE OR REPLACE FUNCTION is_strategy_enabled_for_backtest(p_strategy_name VARCHAR)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM enabled_strategies
        WHERE strategy_name = p_strategy_name
        AND is_enabled = TRUE
    );
END;
$$ LANGUAGE plpgsql;

-- Function to get the best strategy for a regime
CREATE OR REPLACE FUNCTION get_best_strategy_for_regime(
    p_regime VARCHAR,
    p_session VARCHAR DEFAULT NULL,
    p_volatility_state VARCHAR DEFAULT NULL
)
RETURNS TABLE(strategy_name VARCHAR, priority INTEGER, min_win_rate DECIMAL) AS $$
BEGIN
    RETURN QUERY
    SELECT r.strategy_name, r.priority, r.min_win_rate
    FROM strategy_routing_rules r
    JOIN enabled_strategies e ON r.strategy_name = e.strategy_name
    WHERE r.regime = p_regime
    AND r.is_active = TRUE
    AND e.is_enabled = TRUE
    AND (r.session IS NULL OR r.session = p_session)
    AND (r.volatility_state IS NULL OR r.volatility_state = p_volatility_state)
    ORDER BY r.priority ASC
    LIMIT 5;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- AUDIT TRIGGER
-- ============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to all config tables
DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOR tbl IN
        SELECT unnest(ARRAY[
            'enabled_strategies',
            'strategy_routing_rules',
            'strategy_regime_performance',
            'regime_fitness_scores',
            'ranging_market_global_config',
            'ranging_market_pair_overrides',
            'volume_profile_global_config',
            'volume_profile_pair_overrides',
            'mean_reversion_global_config',
            'mean_reversion_pair_overrides',
            'bb_supertrend_global_config',
            'bb_supertrend_pair_overrides',
            'router_global_config'
        ])
    LOOP
        EXECUTE format('
            DROP TRIGGER IF EXISTS update_%s_updated_at ON %s;
            CREATE TRIGGER update_%s_updated_at
            BEFORE UPDATE ON %s
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        ', tbl, tbl, tbl, tbl);
    END LOOP;
END $$;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Verify tables created
SELECT 'Tables created:' as status;
SELECT tablename FROM pg_tables
WHERE schemaname = 'public'
AND tablename IN (
    'enabled_strategies',
    'strategy_routing_rules',
    'strategy_regime_performance',
    'regime_fitness_scores',
    'strategy_switch_log',
    'ranging_market_global_config',
    'ranging_market_pair_overrides',
    'volume_profile_global_config',
    'volume_profile_pair_overrides',
    'mean_reversion_global_config',
    'mean_reversion_pair_overrides',
    'bb_supertrend_global_config',
    'bb_supertrend_pair_overrides',
    'router_global_config'
);

-- Show initial strategy status
SELECT 'Initial strategy status:' as status;
SELECT strategy_name, is_enabled, is_backtest_only, display_name FROM enabled_strategies;

-- Show routing rules
SELECT 'Routing rules:' as status;
SELECT regime, session, strategy_name, priority FROM strategy_routing_rules WHERE is_active ORDER BY regime, priority;
