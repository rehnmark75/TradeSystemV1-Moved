-- ============================================================================
-- RANGING MARKET STRATEGY v4.0 UPGRADE
-- ============================================================================
-- Adds signal qualification columns and per-pair tuning support
-- Date: 2026-02-02
-- ============================================================================

-- Add new columns to global config
ALTER TABLE ranging_market_global_config
ADD COLUMN IF NOT EXISTS trust_regime_routing BOOLEAN DEFAULT TRUE,
ADD COLUMN IF NOT EXISTS use_adx_filter BOOLEAN DEFAULT TRUE,
ADD COLUMN IF NOT EXISTS min_quality_score INTEGER DEFAULT 50,
ADD COLUMN IF NOT EXISTS high_quality_threshold INTEGER DEFAULT 75,
ADD COLUMN IF NOT EXISTS weight_oscillator_agreement INTEGER DEFAULT 30,
ADD COLUMN IF NOT EXISTS weight_oscillator_strength INTEGER DEFAULT 20,
ADD COLUMN IF NOT EXISTS weight_sr_proximity INTEGER DEFAULT 20,
ADD COLUMN IF NOT EXISTS weight_adx_condition INTEGER DEFAULT 15,
ADD COLUMN IF NOT EXISTS weight_session_bonus INTEGER DEFAULT 15,
ADD COLUMN IF NOT EXISTS use_squeeze_momentum BOOLEAN DEFAULT TRUE,
ADD COLUMN IF NOT EXISTS use_rsi BOOLEAN DEFAULT TRUE,
ADD COLUMN IF NOT EXISTS use_stochastic BOOLEAN DEFAULT TRUE,
ADD COLUMN IF NOT EXISTS stoch_period INTEGER DEFAULT 14,
ADD COLUMN IF NOT EXISTS stoch_smooth_k INTEGER DEFAULT 3,
ADD COLUMN IF NOT EXISTS stoch_overbought INTEGER DEFAULT 80,
ADD COLUMN IF NOT EXISTS stoch_oversold INTEGER DEFAULT 20,
ADD COLUMN IF NOT EXISTS sr_lookback_bars INTEGER DEFAULT 20,
ADD COLUMN IF NOT EXISTS asian_session_bonus INTEGER DEFAULT 15,
ADD COLUMN IF NOT EXISTS london_session_bonus INTEGER DEFAULT 5,
ADD COLUMN IF NOT EXISTS ny_session_bonus INTEGER DEFAULT 5,
ADD COLUMN IF NOT EXISTS overlap_session_bonus INTEGER DEFAULT 0;

-- Update version
UPDATE ranging_market_global_config
SET version = '4.0.0'
WHERE is_active = TRUE;

-- Add new columns to pair overrides
ALTER TABLE ranging_market_pair_overrides
ADD COLUMN IF NOT EXISTS min_quality_score INTEGER,
ADD COLUMN IF NOT EXISTS rsi_overbought INTEGER,
ADD COLUMN IF NOT EXISTS rsi_oversold INTEGER,
ADD COLUMN IF NOT EXISTS stoch_overbought INTEGER,
ADD COLUMN IF NOT EXISTS stoch_oversold INTEGER,
ADD COLUMN IF NOT EXISTS sr_proximity_pips DECIMAL(8,2);

-- Set EURUSD-specific defaults for testing
-- These can be tuned based on backtest results
UPDATE ranging_market_pair_overrides
SET
    min_quality_score = 45,           -- Slightly lower for more signals during testing
    adx_max_threshold = 25,           -- More permissive ADX (trust regime routing)
    rsi_overbought = 70,
    rsi_oversold = 30,
    stoch_overbought = 80,
    stoch_oversold = 20,
    sr_proximity_pips = 10.0
WHERE epic = 'CS.D.EURUSD.CEEM.IP';

-- Update global defaults for better ranging detection
UPDATE ranging_market_global_config
SET
    adx_max_threshold = 25,           -- More permissive (was 20)
    trust_regime_routing = TRUE,       -- Trust the multi-strategy router
    min_quality_score = 50,           -- Require 50% quality score
    signal_cooldown_minutes = 30,     -- Reduced from 45
    min_confidence = 0.40             -- Lower base confidence (quality score handles filtering)
WHERE is_active = TRUE;

-- Verify changes
SELECT
    'Global Config' as type,
    version,
    adx_max_threshold,
    trust_regime_routing,
    min_quality_score,
    signal_cooldown_minutes
FROM ranging_market_global_config
WHERE is_active = TRUE;

SELECT
    'EURUSD Override' as type,
    epic,
    min_quality_score,
    adx_max_threshold,
    rsi_overbought,
    rsi_oversold
FROM ranging_market_pair_overrides
WHERE epic = 'CS.D.EURUSD.CEEM.IP';
