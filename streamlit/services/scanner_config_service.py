"""
Scanner Global Configuration Service for Streamlit

Provides database-driven configuration management with:
- Read/write operations for global scanner config
- Caching via Streamlit's @st.cache_resource
- Audit trail for changes

This service manages the scanner_global_config table in strategy_config database.
"""

import streamlit as st
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager
from decimal import Decimal

import psycopg2
import psycopg2.extras

from .db_utils import get_psycopg2_pool

logger = logging.getLogger(__name__)


# =============================================================================
# CONNECTION MANAGEMENT
# =============================================================================

@contextmanager
def get_connection():
    """Get a connection from the centralized pool"""
    pool = get_psycopg2_pool("strategy_config")
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


# =============================================================================
# READ OPERATIONS
# =============================================================================

@st.cache_data(ttl=60)  # Cache for 1 minute - config changes rarely
def get_global_config() -> Optional[Dict[str, Any]]:
    """Load active global configuration from database (cached 1 min)"""
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM scanner_global_config
                    WHERE is_active = TRUE
                    ORDER BY updated_at DESC
                    LIMIT 1
                """)
                row = cur.fetchone()
                if row:
                    result = dict(row)
                    # Convert Decimals to floats for JSON serialization
                    for key, value in result.items():
                        if isinstance(value, Decimal):
                            result[key] = float(value)
                    return result
    except Exception as e:
        logger.error(f"Failed to load scanner global config: {e}")
        st.error(f"Failed to load scanner configuration: {e}")
    return None


def get_config_by_category(category: str) -> Dict[str, Any]:
    """
    Get configuration values for a specific category.

    Categories:
    - core: Scanner core settings
    - dedup: Duplicate detection settings
    - risk: Risk management settings
    - trading_hours: Trading hours settings
    - safety: Safety filter settings
    - adx: ADX filter settings
    """
    config = get_global_config()
    if not config:
        return {}

    category_fields = {
        'core': [
            'scan_interval', 'min_confidence', 'default_timeframe',
            'use_1m_base_synthesis', 'scan_align_to_boundaries', 'scan_boundary_offset_seconds',
            'enable_multi_timeframe_analysis', 'min_confluence_score'
        ],
        'indicators': [
            'macd_fast_period', 'macd_slow_period', 'macd_signal_period',
            'kama_period', 'kama_fast', 'kama_slow',
            'bb_period', 'bb_std_dev', 'supertrend_period', 'supertrend_multiplier',
            'zero_lag_length', 'zero_lag_band_mult',
            'two_pole_filter_length', 'two_pole_sma_length', 'two_pole_signal_delay'
        ],
        'data_quality': [
            'lookback_reduction_factor', 'enable_data_quality_filtering',
            'block_trading_on_data_issues', 'min_quality_score_for_trading'
        ],
        'trading_control': [
            'auto_trading_enabled', 'enable_order_execution'
        ],
        'dedup': [
            'enable_duplicate_check', 'duplicate_sensitivity', 'signal_cooldown_minutes',
            'alert_cooldown_minutes', 'strategy_cooldown_minutes', 'global_cooldown_seconds',
            'max_alerts_per_hour', 'max_alerts_per_epic_hour', 'price_similarity_threshold',
            'confidence_similarity_threshold', 'deduplication_preset', 'use_database_dedup_check',
            'database_dedup_window_minutes', 'enable_signal_hash_check', 'deduplication_debug_mode',
            'enable_price_similarity_check', 'enable_strategy_cooldowns', 'deduplication_lookback_hours',
            'deduplication_presets', 'enable_alert_deduplication', 'signal_hash_cache_expiry_minutes',
            'max_signal_hash_cache_size', 'enable_time_based_hash_components'
        ],
        'risk': [
            'position_size_percent', 'stop_loss_pips', 'take_profit_pips', 'max_open_positions',
            'max_daily_trades', 'risk_per_trade_percent', 'min_position_size', 'max_position_size',
            'max_risk_per_trade', 'default_risk_reward', 'default_stop_distance'
        ],
        'trading_hours': [
            'trading_start_hour', 'trading_end_hour', 'respect_market_hours', 'weekend_scanning',
            'enable_trading_time_controls', 'trading_cutoff_time_utc', 'trade_cooldown_enabled',
            'trade_cooldown_minutes', 'user_timezone', 'respect_trading_hours',
            # Session Hours Configuration (Jan 2026)
            'session_asian_start_hour', 'session_asian_end_hour',
            'session_london_start_hour', 'session_london_end_hour',
            'session_newyork_start_hour', 'session_newyork_end_hour',
            'session_overlap_start_hour', 'session_overlap_end_hour',
            'block_asian_session'
        ],
        'order_executor': [
            # Order Executor Thresholds (Jan 2026)
            'executor_high_confidence_threshold', 'executor_medium_confidence_threshold',
            'executor_max_stop_loss_pips', 'executor_max_take_profit_pips',
            'executor_high_conf_stop_multiplier', 'executor_low_conf_stop_multiplier'
        ],
        # NOTE: 'safety' section removed (Jan 2026) - EMA200/consensus filters were redundant
        # with SMC Simple strategy's built-in 4H 50 EMA bias check
        # NOTE: 'adx' section removed (Jan 2026) - was never used by active strategies
        'smc_conflict': [
            'smart_money_readonly_enabled', 'smart_money_analysis_timeout',
            'smc_conflict_filter_enabled', 'smc_min_directional_consensus',
            'smc_reject_order_flow_conflict', 'smc_reject_ranging_structure',
            'smc_min_structure_score'
        ],
        'claude_validation': [
            'require_claude_approval', 'claude_fail_secure', 'claude_model',
            'min_claude_quality_score', 'claude_include_chart', 'claude_chart_timeframes',
            'claude_vision_enabled', 'claude_vision_strategies', 'claude_validate_in_backtest',
            'save_claude_rejections', 'claude_save_vision_artifacts'
        ],
        'sr_validation': [
            'enable_sr_validation', 'enable_enhanced_sr_validation', 'sr_analysis_timeframe',
            'sr_lookback_hours', 'sr_left_bars', 'sr_right_bars', 'sr_volume_threshold',
            'sr_level_tolerance_pips', 'sr_min_level_distance_pips', 'sr_recent_flip_bars',
            'sr_min_flip_strength', 'sr_cache_duration_minutes', 'min_bars_for_sr_analysis'
        ],
        'signal_freshness': [
            'enable_signal_freshness_check', 'max_signal_age_minutes'
        ],
        'news_filtering': [
            'enable_news_filtering', 'reduce_confidence_near_news', 'news_filter_fail_secure'
        ],
        'market_intelligence': [
            'enable_market_intelligence_capture', 'enable_market_intelligence_filtering',
            'market_intelligence_min_confidence', 'market_intelligence_block_unsuitable_regimes',
            'market_bias_filter_enabled', 'market_bias_min_consensus'
        ],
        'validation': [
            'strategy_testing_mode', 'validate_spread', 'max_spread_pips',
            'min_signal_confirmations', 'scalping_min_confidence',
            'allowed_trading_epics', 'blocked_trading_epics'
        ]
    }

    fields = category_fields.get(category, [])
    return {k: config.get(k) for k in fields if k in config}


# =============================================================================
# WRITE OPERATIONS
# =============================================================================

def save_global_config(
    config_id: int,
    updates: Dict[str, Any],
    updated_by: str,
    change_reason: str,
    category: str = None
) -> bool:
    """
    Save updates to global configuration.

    Args:
        config_id: ID of the config row to update
        updates: Dictionary of field:value pairs to update
        updated_by: User making the change
        change_reason: Reason for the change
        category: Optional category for audit grouping

    Returns:
        True if successful, False otherwise
    """
    try:
        with get_connection() as conn:
            # Get current values for audit
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM scanner_global_config WHERE id = %s", (config_id,))
                row = cur.fetchone()
                old_values = dict(row) if row else {}

            # Build update query
            update_fields = []
            values = []

            # JSONB fields that need special handling
            # NOTE: adx_thresholds, adx_pair_multipliers removed (Jan 2026)
            jsonb_fields = [
                'deduplication_presets'
            ]

            for key, value in updates.items():
                update_fields.append(f"{key} = %s")
                if key in jsonb_fields:
                    values.append(json.dumps(value) if isinstance(value, dict) else value)
                else:
                    values.append(value)

            update_fields.extend(['updated_by = %s', 'change_reason = %s'])
            values.extend([updated_by, change_reason])
            values.append(config_id)

            with conn.cursor() as cur:
                # Update config
                cur.execute(f"""
                    UPDATE scanner_global_config
                    SET {', '.join(update_fields)}
                    WHERE id = %s
                """, values)

                # Add audit record
                cur.execute("""
                    INSERT INTO scanner_config_audit
                    (config_id, change_type, changed_by, change_reason, previous_values, new_values, category)
                    VALUES (%s, 'UPDATE', %s, %s, %s, %s, %s)
                """, (
                    config_id,
                    updated_by,
                    change_reason,
                    json.dumps({k: _serialize_value(old_values.get(k)) for k in updates.keys()}),
                    json.dumps({k: _serialize_value(v) for k, v in updates.items()}),
                    category
                ))

            conn.commit()
            logger.info(f"[CONFIG:DB] Updated scanner global config id={config_id}, {len(updates)} fields")
            return True

    except Exception as e:
        logger.error(f"Failed to save scanner global config: {e}")
        st.error(f"Failed to save configuration: {e}")
    return False


def _serialize_value(value: Any) -> Any:
    """Serialize a value for JSON storage."""
    if isinstance(value, Decimal):
        return float(value)
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, (dict, list)):
        return value
    else:
        return str(value) if value is not None else None


def apply_preset(
    config_id: int,
    preset_type: str,
    preset_name: str,
    updated_by: str
) -> bool:
    """
    Apply a preset configuration.

    Args:
        config_id: Config ID to update
        preset_type: Type of preset ('deduplication', 'safety', 'large_candle')
        preset_name: Name of the preset to apply
        updated_by: User making the change

    Returns:
        True if successful
    """
    config = get_global_config()
    if not config:
        return False

    # Get the preset data
    preset_field = f"{preset_type}_presets"
    if preset_type == 'large_candle':
        preset_field = 'large_candle_filter_presets'

    presets = config.get(preset_field, {})
    if preset_name not in presets:
        st.error(f"Preset '{preset_name}' not found in {preset_field}")
        return False

    preset_values = presets[preset_name]

    # Apply the preset values
    return save_global_config(
        config_id,
        preset_values,
        updated_by,
        f"Applied {preset_type} preset: {preset_name}",
        category=preset_type
    )


# =============================================================================
# AUDIT OPERATIONS
# =============================================================================

def get_audit_history(
    limit: int = 50,
    category: str = None
) -> List[Dict[str, Any]]:
    """
    Get recent configuration changes.

    Args:
        limit: Maximum number of records to return
        category: Optional category filter

    Returns:
        List of audit records
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if category:
                    cur.execute("""
                        SELECT
                            id,
                            config_id,
                            change_type,
                            changed_by,
                            changed_at,
                            change_reason,
                            previous_values,
                            new_values,
                            category
                        FROM scanner_config_audit
                        WHERE category = %s
                        ORDER BY changed_at DESC
                        LIMIT %s
                    """, (category, limit))
                else:
                    cur.execute("""
                        SELECT
                            id,
                            config_id,
                            change_type,
                            changed_by,
                            changed_at,
                            change_reason,
                            previous_values,
                            new_values,
                            category
                        FROM scanner_config_audit
                        ORDER BY changed_at DESC
                        LIMIT %s
                    """, (limit,))

                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get scanner audit history: {e}")
    return []


def get_config_version_info() -> Dict[str, Any]:
    """Get version and metadata about current configuration."""
    config = get_global_config()
    if not config:
        return {'error': 'No configuration found'}

    return {
        'id': config.get('id'),
        'version': config.get('version'),
        'is_active': config.get('is_active'),
        'updated_at': config.get('updated_at'),
        'updated_by': config.get('updated_by'),
        'change_reason': config.get('change_reason'),
    }


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_config_value(field: str, value: Any) -> tuple[bool, str]:
    """
    Validate a configuration value.

    Returns:
        Tuple of (is_valid, error_message)
    """
    validations = {
        'scan_interval': lambda v: (30 <= v <= 600, "Must be between 30 and 600 seconds"),
        'min_confidence': lambda v: (0.0 <= v <= 1.0, "Must be between 0.0 and 1.0"),
        'max_alerts_per_hour': lambda v: (1 <= v <= 200, "Must be between 1 and 200"),
        'max_alerts_per_epic_hour': lambda v: (1 <= v <= 50, "Must be between 1 and 50"),
        'max_open_positions': lambda v: (1 <= v <= 20, "Must be between 1 and 20"),
        'max_daily_trades': lambda v: (1 <= v <= 100, "Must be between 1 and 100"),
        'trading_start_hour': lambda v: (0 <= v <= 23, "Must be between 0 and 23"),
        'trading_end_hour': lambda v: (0 <= v <= 23, "Must be between 0 and 23"),
        'trading_cutoff_time_utc': lambda v: (0 <= v <= 23, "Must be between 0 and 23"),
        # NOTE: adx_period validation removed (Jan 2026)
    }

    if field in validations:
        is_valid, error_msg = validations[field](value)
        return is_valid, error_msg if not is_valid else ""

    return True, ""


def get_field_metadata(field: str) -> Dict[str, Any]:
    """
    Get metadata for a configuration field.

    Returns:
        Dictionary with min, max, step, help text, etc.
    """
    metadata = {
        # Core settings
        'scan_interval': {
            'min': 30, 'max': 600, 'step': 10,
            'help': 'Seconds between scanner runs',
            'unit': 'seconds'
        },
        'min_confidence': {
            'min': 0.0, 'max': 1.0, 'step': 0.01,
            'help': 'Minimum signal confidence threshold (0.0-1.0)',
            'format': '%.2f'
        },
        'default_timeframe': {
            'options': ['5m', '15m', '1h', '4h'],
            'help': 'Default signal timeframe'
        },

        # Dedup settings
        'signal_cooldown_minutes': {
            'min': 1, 'max': 120, 'step': 1,
            'help': 'Cooldown between signals for same epic',
            'unit': 'minutes'
        },
        'max_alerts_per_hour': {
            'min': 1, 'max': 200, 'step': 5,
            'help': 'Maximum alerts across all pairs per hour'
        },
        'max_alerts_per_epic_hour': {
            'min': 1, 'max': 50, 'step': 1,
            'help': 'Maximum alerts per pair per hour'
        },
        'duplicate_sensitivity': {
            'options': ['strict', 'smart', 'loose'],
            'help': 'How strictly to detect duplicate signals'
        },
        'deduplication_preset': {
            'options': ['strict', 'standard', 'relaxed'],
            'help': 'Preset deduplication configuration'
        },

        # Risk settings
        'stop_loss_pips': {
            'min': 1, 'max': 100, 'step': 1,
            'help': 'Default stop loss in pips',
            'unit': 'pips'
        },
        'take_profit_pips': {
            'min': 1, 'max': 200, 'step': 1,
            'help': 'Default take profit in pips',
            'unit': 'pips'
        },
        'max_open_positions': {
            'min': 1, 'max': 20, 'step': 1,
            'help': 'Maximum concurrent open positions'
        },
        'max_daily_trades': {
            'min': 1, 'max': 100, 'step': 1,
            'help': 'Maximum trades per day'
        },
        'default_risk_reward': {
            'min': 0.5, 'max': 5.0, 'step': 0.1,
            'help': 'Default risk:reward ratio',
            'format': '%.1f'
        },

        # NOTE: Safety settings removed (Jan 2026) - redundant with SMC Simple strategy
        # NOTE: ADX filter settings removed (Jan 2026) - was never used by active strategies

        # Indicator settings - MACD
        'macd_fast_period': {
            'min': 2, 'max': 50, 'step': 1,
            'help': 'Fast EMA period for MACD calculation (default: 12)'
        },
        'macd_slow_period': {
            'min': 5, 'max': 100, 'step': 1,
            'help': 'Slow EMA period for MACD calculation (default: 26)'
        },
        'macd_signal_period': {
            'min': 2, 'max': 50, 'step': 1,
            'help': 'Signal line EMA period (default: 9)'
        },

        # Indicator settings - KAMA
        'kama_period': {
            'min': 2, 'max': 50, 'step': 1,
            'help': 'Efficiency ratio period (default: 10)'
        },
        'kama_fast': {
            'min': 1, 'max': 20, 'step': 1,
            'help': 'Fast smoothing constant (default: 2)'
        },
        'kama_slow': {
            'min': 10, 'max': 100, 'step': 1,
            'help': 'Slow smoothing constant (default: 30)'
        },

        # Indicator settings - Bollinger Bands & Supertrend
        'bb_period': {
            'min': 5, 'max': 100, 'step': 1,
            'help': 'Bollinger Bands period (default: 20)'
        },
        'bb_std_dev': {
            'min': 0.5, 'max': 5.0, 'step': 0.1,
            'help': 'Standard deviation multiplier (default: 2.0)',
            'format': '%.1f'
        },
        'supertrend_period': {
            'min': 5, 'max': 50, 'step': 1,
            'help': 'ATR period for Supertrend (default: 10)'
        },
        'supertrend_multiplier': {
            'min': 1.0, 'max': 10.0, 'step': 0.1,
            'help': 'ATR multiplier (default: 3.0)',
            'format': '%.1f'
        },

        # Indicator settings - Zero Lag
        'zero_lag_length': {
            'min': 5, 'max': 100, 'step': 1,
            'help': 'Zero lag EMA period (default: 21)'
        },
        'zero_lag_band_mult': {
            'min': 0.5, 'max': 5.0, 'step': 0.1,
            'help': 'Band multiplier for volatility bands (default: 1.5)',
            'format': '%.1f'
        },

        # Indicator settings - Two-Pole
        'two_pole_filter_length': {
            'min': 5, 'max': 100, 'step': 1,
            'help': 'Two-pole filter length (default: 20)'
        },
        'two_pole_sma_length': {
            'min': 5, 'max': 100, 'step': 1,
            'help': 'SMA length for smoothing (default: 25)'
        },
        'two_pole_signal_delay': {
            'min': 1, 'max': 20, 'step': 1,
            'help': 'Signal delay periods (default: 4)'
        },
        # Data Quality Settings
        'lookback_reduction_factor': {
            'min': 0.1, 'max': 1.0, 'step': 0.05,
            'help': 'Reduces lookback period for data fetching. Lower = less data, faster scans (default: 0.7)'
        },
        'enable_data_quality_filtering': {
            'help': 'Filter out data points flagged as unsafe or low quality before analysis'
        },
        'block_trading_on_data_issues': {
            'help': 'Prevent trading when data quality issues are detected'
        },
        'min_quality_score_for_trading': {
            'min': 0.0, 'max': 1.0, 'step': 0.05,
            'help': 'Minimum data quality score required for trading (0.0-1.0). Default: 0.5'
        },

        # S/R Validation Settings
        'enable_sr_validation': {
            'help': 'Enable Support/Resistance validation for signals'
        },
        'enable_enhanced_sr_validation': {
            'help': 'Use enhanced S/R validator with level flip detection'
        },
        'sr_analysis_timeframe': {
            'options': ['5m', '15m', '30m', '1h', '4h'],
            'help': 'Timeframe for S/R analysis (default: 15m)'
        },
        'sr_lookback_hours': {
            'min': 24, 'max': 168, 'step': 12,
            'help': 'Hours of data for S/R analysis (default: 72 = 3 days)'
        },
        'sr_left_bars': {
            'min': 5, 'max': 50, 'step': 1,
            'help': 'Left bars for pivot detection (default: 15)'
        },
        'sr_right_bars': {
            'min': 5, 'max': 50, 'step': 1,
            'help': 'Right bars for pivot detection (default: 15)'
        },
        'sr_volume_threshold': {
            'min': 5.0, 'max': 50.0, 'step': 5.0,
            'help': 'Volume threshold percentile (default: 20.0)'
        },
        'sr_level_tolerance_pips': {
            'min': 0.5, 'max': 10.0, 'step': 0.5,
            'help': 'Tolerance for S/R level matching in pips (default: 2.0)'
        },
        'sr_min_level_distance_pips': {
            'min': 5.0, 'max': 50.0, 'step': 5.0,
            'help': 'Minimum distance between S/R levels in pips (default: 20.0)'
        },
        'sr_recent_flip_bars': {
            'min': 20, 'max': 100, 'step': 10,
            'help': 'Bars to check for recent level flips (default: 50)'
        },
        'sr_min_flip_strength': {
            'min': 0.3, 'max': 1.0, 'step': 0.1,
            'help': 'Minimum strength for level flip detection (default: 0.6)'
        },
        'sr_cache_duration_minutes': {
            'min': 5, 'max': 60, 'step': 5,
            'help': 'Duration to cache S/R data in minutes (default: 10)'
        },
        'min_bars_for_sr_analysis': {
            'min': 50, 'max': 500, 'step': 50,
            'help': 'Minimum bars required for S/R analysis (default: 100)'
        },

        # Signal Freshness Settings
        'enable_signal_freshness_check': {
            'help': 'Check if signals are recent enough for trading'
        },
        'max_signal_age_minutes': {
            'min': 5, 'max': 120, 'step': 5,
            'help': 'Maximum age of signal in minutes (default: 30)'
        },

        # News Filtering Settings
        'enable_news_filtering': {
            'help': 'Filter signals based on economic news events'
        },
        'reduce_confidence_near_news': {
            'help': 'Reduce signal confidence when near high-impact news'
        },
        'news_filter_fail_secure': {
            'help': 'Block signals if news filter fails (fail-secure mode)'
        },

        # Market Intelligence Settings
        'enable_market_intelligence_capture': {
            'help': 'Capture market intelligence data with each signal'
        },
        'enable_market_intelligence_filtering': {
            'help': 'Use market intelligence for signal filtering'
        },
        'market_intelligence_min_confidence': {
            'min': 0.3, 'max': 1.0, 'step': 0.05,
            'help': 'Minimum market intelligence confidence (default: 0.7)'
        },
        'market_intelligence_block_unsuitable_regimes': {
            'help': 'Block signals in unsuitable market regimes'
        },
        'market_bias_filter_enabled': {
            'help': 'Block counter-trend trades when market consensus is high'
        },
        'market_bias_min_consensus': {
            'min': 0.5, 'max': 1.0, 'step': 0.05,
            'help': 'Minimum market consensus to trigger bias filter (default: 0.70)'
        },

        # Validation Settings
        'strategy_testing_mode': {
            'help': 'Skip certain validations for strategy testing'
        },
        'validate_spread': {
            'help': 'Validate spread before trading'
        },
        'max_spread_pips': {
            'min': 1.0, 'max': 10.0, 'step': 0.5,
            'help': 'Maximum allowed spread in pips (default: 3.0)'
        },
        'min_signal_confirmations': {
            'min': 0, 'max': 5, 'step': 1,
            'help': 'Minimum number of signal confirmations required (default: 0)'
        },
        'scalping_min_confidence': {
            'min': 0.3, 'max': 0.8, 'step': 0.05,
            'help': 'Minimum confidence for scalping strategies (default: 0.45)'
        },
    }

    return metadata.get(field, {})


# =============================================================================
# COMPARISON HELPERS
# =============================================================================

def values_equal(new_val: Any, old_val: Any, tolerance: float = 1e-9) -> bool:
    """
    Compare values with tolerance for floating point precision.

    Used to detect actual changes vs floating point noise.
    """
    if new_val is None and old_val is None:
        return True
    if new_val is None or old_val is None:
        return False

    # Handle numeric comparison with tolerance
    if isinstance(new_val, (int, float, Decimal)) and isinstance(old_val, (int, float, Decimal)):
        return abs(float(new_val) - float(old_val)) < tolerance

    # Handle dict/list comparison
    if isinstance(new_val, (dict, list)) and isinstance(old_val, (dict, list)):
        return json.dumps(new_val, sort_keys=True) == json.dumps(old_val, sort_keys=True)

    return new_val == old_val
