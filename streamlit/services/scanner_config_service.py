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

from .db_utils import get_connection_string

logger = logging.getLogger(__name__)


# =============================================================================
# CONNECTION MANAGEMENT
# =============================================================================

@st.cache_resource
def get_scanner_config_pool():
    """Get cached connection pool for strategy_config database"""
    from psycopg2 import pool as psycopg2_pool
    conn_str = get_connection_string("strategy_config")
    return psycopg2_pool.ThreadedConnectionPool(
        minconn=2,
        maxconn=10,
        dsn=conn_str
    )


@contextmanager
def get_connection():
    """Get a connection from the pool"""
    pool = get_scanner_config_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


# =============================================================================
# READ OPERATIONS
# =============================================================================

def get_global_config() -> Optional[Dict[str, Any]]:
    """Load active global configuration from database"""
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
            'use_1m_base_synthesis', 'scan_align_to_boundaries', 'scan_boundary_offset_seconds'
        ],
        'dedup': [
            'enable_duplicate_check', 'duplicate_sensitivity', 'signal_cooldown_minutes',
            'alert_cooldown_minutes', 'strategy_cooldown_minutes', 'global_cooldown_seconds',
            'max_alerts_per_hour', 'max_alerts_per_epic_hour', 'price_similarity_threshold',
            'confidence_similarity_threshold', 'deduplication_preset', 'use_database_dedup_check',
            'database_dedup_window_minutes', 'enable_signal_hash_check', 'deduplication_debug_mode',
            'enable_price_similarity_check', 'enable_strategy_cooldowns', 'deduplication_lookback_hours',
            'deduplication_presets'
        ],
        'risk': [
            'position_size_percent', 'stop_loss_pips', 'take_profit_pips', 'max_open_positions',
            'max_daily_trades', 'risk_per_trade_percent', 'min_position_size', 'max_position_size',
            'max_risk_per_trade', 'default_risk_reward', 'default_stop_distance'
        ],
        'trading_hours': [
            'trading_start_hour', 'trading_end_hour', 'respect_market_hours', 'weekend_scanning',
            'enable_trading_time_controls', 'trading_cutoff_time_utc', 'trade_cooldown_enabled',
            'trade_cooldown_minutes', 'user_timezone', 'respect_trading_hours'
        ],
        'safety': [
            'enable_critical_safety_filters', 'enable_ema200_contradiction_filter',
            'enable_ema_stack_contradiction_filter', 'require_indicator_consensus',
            'min_confirming_indicators', 'enable_emergency_circuit_breaker',
            'max_contradictions_allowed', 'active_safety_preset', 'enable_large_candle_filter',
            'large_candle_atr_multiplier', 'consecutive_large_candles_threshold',
            'movement_lookback_periods', 'large_candle_filter_cooldown', 'ema200_minimum_margin',
            'safety_filter_log_level', 'excessive_movement_threshold_pips',
            'safety_filter_presets', 'large_candle_filter_presets'
        ],
        'adx': [
            'adx_filter_enabled', 'adx_filter_mode', 'adx_period', 'adx_grace_period_bars',
            'adx_thresholds', 'adx_pair_multipliers'
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
            jsonb_fields = [
                'adx_thresholds', 'adx_pair_multipliers',
                'deduplication_presets', 'safety_filter_presets', 'large_candle_filter_presets'
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
        'adx_period': lambda v: (5 <= v <= 50, "Must be between 5 and 50"),
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

        # Safety settings
        'max_contradictions_allowed': {
            'min': 0, 'max': 10, 'step': 1,
            'help': 'Maximum indicator contradictions before rejecting signal'
        },
        'active_safety_preset': {
            'options': ['strict', 'balanced', 'permissive', 'emergency'],
            'help': 'Active safety filter preset'
        },
        'large_candle_atr_multiplier': {
            'min': 1.0, 'max': 5.0, 'step': 0.1,
            'help': 'ATR multiplier for large candle detection',
            'format': '%.1f'
        },

        # ADX settings
        'adx_period': {
            'min': 5, 'max': 50, 'step': 1,
            'help': 'ADX calculation period'
        },
        'adx_filter_mode': {
            'options': ['strict', 'moderate', 'permissive', 'disabled'],
            'help': 'ADX filter strictness level'
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
