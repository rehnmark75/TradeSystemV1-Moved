"""
Intelligence System Configuration Service for Streamlit

Provides database-driven configuration management with:
- Read/write operations for global intelligence config
- Preset management
- Regime-strategy modifier management
- Caching via Streamlit's @st.cache_resource
- Audit trail for changes
"""

import streamlit as st
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

from .db_utils import get_psycopg2_pool

logger = logging.getLogger(__name__)


@contextmanager
def get_connection():
    """Get a connection from the centralized pool"""
    pool = get_psycopg2_pool("strategy_config")
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


@st.cache_data(ttl=60)  # Cache for 1 minute - config changes rarely
def get_intelligence_config() -> Dict[str, Any]:
    """
    Load all intelligence configuration from database (cached 1 min).

    Optimized to use a single query with UNION ALL instead of 4 separate queries.

    Returns dict with:
    - parameters: Dict of parameter_name -> value
    - presets: Dict of preset_name -> preset details
    - regime_modifiers: Dict of regime -> {strategy: modifier}
    """
    result = {
        'parameters': {},
        'presets': {},
        'preset_components': {},
        'regime_modifiers': {},
    }

    try:
        with get_connection() as conn:
            # Single optimized query to fetch all config data at once
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    -- Fetch all configuration in a single query using UNION ALL
                    -- Type 1: Global parameters
                    SELECT 1 as query_type,
                           parameter_name, parameter_value, value_type, category,
                           subcategory, description, display_order, min_value, max_value,
                           valid_options, is_editable,
                           NULL::INTEGER as preset_id, NULL::VARCHAR as preset_name,
                           NULL::NUMERIC as threshold, NULL::BOOLEAN as use_intelligence_engine,
                           NULL::VARCHAR as component_name, NULL::BOOLEAN as is_enabled,
                           NULL::VARCHAR as regime, NULL::VARCHAR as strategy,
                           NULL::NUMERIC as confidence_modifier
                    FROM intelligence_global_config
                    WHERE is_active = TRUE

                    UNION ALL

                    -- Type 2: Presets
                    SELECT 2 as query_type,
                           NULL, NULL, NULL, NULL, NULL,
                           description, display_order, NULL, NULL, NULL, NULL,
                           id, preset_name, threshold, use_intelligence_engine,
                           NULL, NULL, NULL, NULL, NULL
                    FROM intelligence_presets
                    WHERE is_active = TRUE

                    UNION ALL

                    -- Type 3: Preset components
                    SELECT 3 as query_type,
                           NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                           NULL, p.preset_name, NULL, NULL,
                           pc.component_name, pc.is_enabled, NULL, NULL, NULL
                    FROM intelligence_preset_components pc
                    JOIN intelligence_presets p ON p.id = pc.preset_id
                    WHERE p.is_active = TRUE

                    UNION ALL

                    -- Type 4: Regime modifiers
                    SELECT 4 as query_type,
                           NULL, NULL, NULL, NULL, NULL,
                           description, NULL, NULL, NULL, NULL, NULL,
                           NULL, NULL, NULL, NULL, NULL, NULL,
                           regime, strategy, confidence_modifier
                    FROM intelligence_regime_modifiers
                    WHERE is_active = TRUE
                """)

                for row in cur.fetchall():
                    query_type = row['query_type']

                    if query_type == 1:  # Global parameters
                        name = row['parameter_name']
                        value = row['parameter_value']
                        value_type = row['value_type']

                        # Convert value based on type
                        if value_type == 'bool':
                            converted_value = value.lower() in ('true', '1', 'yes')
                        elif value_type == 'int':
                            converted_value = int(value)
                        elif value_type == 'float':
                            converted_value = float(value)
                        else:
                            converted_value = value

                        result['parameters'][name] = {
                            'value': converted_value,
                            'raw_value': value,
                            'type': value_type,
                            'category': row['category'],
                            'subcategory': row['subcategory'],
                            'description': row['description'],
                            'display_order': row['display_order'],
                            'min_value': float(row['min_value']) if row['min_value'] is not None else None,
                            'max_value': float(row['max_value']) if row['max_value'] is not None else None,
                            'valid_options': row['valid_options'],
                            'is_editable': row['is_editable'],
                        }

                    elif query_type == 2:  # Presets
                        preset_name = row['preset_name']
                        result['presets'][preset_name] = {
                            'id': row['preset_id'],
                            'name': preset_name,
                            'threshold': float(row['threshold']),
                            'use_intelligence_engine': row['use_intelligence_engine'],
                            'description': row['description'],
                            'display_order': row['display_order'],
                        }

                    elif query_type == 3:  # Preset components
                        preset_name = row['preset_name']
                        if preset_name not in result['preset_components']:
                            result['preset_components'][preset_name] = {}
                        result['preset_components'][preset_name][row['component_name']] = row['is_enabled']

                    elif query_type == 4:  # Regime modifiers
                        regime = row['regime']
                        if regime not in result['regime_modifiers']:
                            result['regime_modifiers'][regime] = {}
                        result['regime_modifiers'][regime][row['strategy']] = {
                            'modifier': float(row['confidence_modifier']),
                            'description': row['description'],
                        }

    except Exception as e:
        logger.error(f"Failed to load intelligence config: {e}")
        st.error(f"Failed to load intelligence configuration: {e}")

    return result


def get_parameter_value(parameter_name: str, default: Any = None) -> Any:
    """Get a single parameter value from intelligence config"""
    config = get_intelligence_config()
    if parameter_name in config['parameters']:
        return config['parameters'][parameter_name]['value']
    return default


def get_active_preset() -> str:
    """Get the name of the currently active preset"""
    return get_parameter_value('intelligence_preset', 'collect_only')


def get_preset_details(preset_name: str) -> Optional[Dict[str, Any]]:
    """Get details for a specific preset"""
    config = get_intelligence_config()
    return config['presets'].get(preset_name)


def get_regime_modifier(regime: str, strategy: str) -> float:
    """Get confidence modifier for regime-strategy combination"""
    config = get_intelligence_config()
    if regime in config['regime_modifiers']:
        if strategy in config['regime_modifiers'][regime]:
            return config['regime_modifiers'][regime][strategy]['modifier']
    return 1.0


def save_parameter(
    parameter_name: str,
    new_value: Any,
    updated_by: str = 'streamlit_user',
    change_reason: str = ''
) -> bool:
    """Update a single parameter value"""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Get current value for audit
                cur.execute("""
                    SELECT parameter_value FROM intelligence_global_config
                    WHERE parameter_name = %s
                """, (parameter_name,))
                row = cur.fetchone()
                old_value = row[0] if row else None

                # Determine value type for storage
                if isinstance(new_value, bool):
                    value_str = 'true' if new_value else 'false'
                else:
                    value_str = str(new_value)

                # Update parameter
                cur.execute("""
                    UPDATE intelligence_global_config
                    SET parameter_value = %s, updated_at = NOW()
                    WHERE parameter_name = %s
                """, (value_str, parameter_name))

                # Add audit record
                cur.execute("""
                    INSERT INTO intelligence_config_audit
                    (table_name, change_type, changed_by, change_reason, previous_values, new_values)
                    VALUES ('intelligence_global_config', 'UPDATE', %s, %s, %s, %s)
                """, (
                    updated_by,
                    change_reason or f"Updated {parameter_name}",
                    json.dumps({parameter_name: old_value}),
                    json.dumps({parameter_name: value_str})
                ))

                conn.commit()
                logger.info(f"Updated intelligence parameter {parameter_name}={value_str}")
                return True

    except Exception as e:
        logger.error(f"Failed to save parameter {parameter_name}: {e}")
        st.error(f"Failed to save parameter: {e}")
        return False


def save_multiple_parameters(
    updates: Dict[str, Any],
    updated_by: str = 'streamlit_user',
    change_reason: str = ''
) -> bool:
    """Update multiple parameters in a single transaction"""
    if not updates:
        return True

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                old_values = {}
                new_values = {}

                for param_name, new_value in updates.items():
                    # Get current value
                    cur.execute("""
                        SELECT parameter_value FROM intelligence_global_config
                        WHERE parameter_name = %s
                    """, (param_name,))
                    row = cur.fetchone()
                    old_values[param_name] = row[0] if row else None

                    # Determine value type for storage
                    if isinstance(new_value, bool):
                        value_str = 'true' if new_value else 'false'
                    else:
                        value_str = str(new_value)
                    new_values[param_name] = value_str

                    # Update parameter
                    cur.execute("""
                        UPDATE intelligence_global_config
                        SET parameter_value = %s, updated_at = NOW()
                        WHERE parameter_name = %s
                    """, (value_str, param_name))

                # Add single audit record for batch update
                cur.execute("""
                    INSERT INTO intelligence_config_audit
                    (table_name, change_type, changed_by, change_reason, previous_values, new_values)
                    VALUES ('intelligence_global_config', 'BATCH_UPDATE', %s, %s, %s, %s)
                """, (
                    updated_by,
                    change_reason or f"Updated {len(updates)} parameters",
                    json.dumps(old_values),
                    json.dumps(new_values)
                ))

                conn.commit()
                logger.info(f"Updated {len(updates)} intelligence parameters")
                return True

    except Exception as e:
        logger.error(f"Failed to save parameters: {e}")
        st.error(f"Failed to save parameters: {e}")
        return False


def save_preset(
    preset_name: str,
    threshold: float,
    use_intelligence_engine: bool,
    description: str,
    updated_by: str = 'streamlit_user'
) -> bool:
    """Update or create a preset"""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE intelligence_presets
                    SET threshold = %s, use_intelligence_engine = %s,
                        description = %s, updated_at = NOW()
                    WHERE preset_name = %s
                """, (threshold, use_intelligence_engine, description, preset_name))

                if cur.rowcount == 0:
                    # Insert new preset
                    cur.execute("""
                        INSERT INTO intelligence_presets
                        (preset_name, threshold, use_intelligence_engine, description)
                        VALUES (%s, %s, %s, %s)
                    """, (preset_name, threshold, use_intelligence_engine, description))

                # Add audit record
                cur.execute("""
                    INSERT INTO intelligence_config_audit
                    (table_name, change_type, changed_by, new_values)
                    VALUES ('intelligence_presets', 'UPDATE', %s, %s)
                """, (
                    updated_by,
                    json.dumps({
                        'preset_name': preset_name,
                        'threshold': threshold,
                        'use_intelligence_engine': use_intelligence_engine,
                    })
                ))

                conn.commit()
                return True

    except Exception as e:
        logger.error(f"Failed to save preset {preset_name}: {e}")
        st.error(f"Failed to save preset: {e}")
        return False


def save_regime_modifier(
    regime: str,
    strategy: str,
    confidence_modifier: float,
    description: str = None,
    updated_by: str = 'streamlit_user'
) -> bool:
    """Update or create a regime-strategy modifier"""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE intelligence_regime_modifiers
                    SET confidence_modifier = %s, description = %s, updated_at = NOW()
                    WHERE regime = %s AND strategy = %s
                """, (confidence_modifier, description, regime, strategy))

                if cur.rowcount == 0:
                    # Insert new modifier
                    cur.execute("""
                        INSERT INTO intelligence_regime_modifiers
                        (regime, strategy, confidence_modifier, description)
                        VALUES (%s, %s, %s, %s)
                    """, (regime, strategy, confidence_modifier, description))

                # Add audit record
                cur.execute("""
                    INSERT INTO intelligence_config_audit
                    (table_name, change_type, changed_by, new_values)
                    VALUES ('intelligence_regime_modifiers', 'UPDATE', %s, %s)
                """, (
                    updated_by,
                    json.dumps({
                        'regime': regime,
                        'strategy': strategy,
                        'confidence_modifier': confidence_modifier,
                    })
                ))

                conn.commit()
                return True

    except Exception as e:
        logger.error(f"Failed to save regime modifier: {e}")
        st.error(f"Failed to save regime modifier: {e}")
        return False


def get_audit_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent audit history for intelligence config changes"""
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, table_name, change_type, changed_by, changed_at,
                           change_reason, previous_values, new_values
                    FROM intelligence_config_audit
                    ORDER BY changed_at DESC
                    LIMIT %s
                """, (limit,))
                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Failed to load audit history: {e}")
        return []


def get_config_summary() -> Dict[str, Any]:
    """Get a summary of the current intelligence configuration"""
    config = get_intelligence_config()

    active_preset = config['parameters'].get('intelligence_preset', {}).get('value', 'unknown')
    preset_details = config['presets'].get(active_preset, {})

    enabled_components = []
    component_params = [
        'enable_market_regime_filter',
        'enable_volatility_filter',
        'enable_volume_filter',
        'enable_time_filter',
        'enable_confidence_filter',
        'enable_spread_filter',
        'enable_recent_signals_filter',
    ]
    for comp in component_params:
        if config['parameters'].get(comp, {}).get('value', False):
            enabled_components.append(comp.replace('enable_', '').replace('_filter', ''))

    return {
        'preset': active_preset,
        'preset_description': preset_details.get('description', ''),
        'threshold': preset_details.get('threshold', 0.0),
        'use_engine': preset_details.get('use_intelligence_engine', False),
        'enabled_components': enabled_components,
        'smart_money_collection': config['parameters'].get('enable_smart_money_collection', {}).get('value', False),
        'order_flow_collection': config['parameters'].get('enable_order_flow_collection', {}).get('value', False),
        'enhanced_regime_detection': config['parameters'].get('enhanced_regime_detection_enabled', {}).get('value', False),
        'cleanup_days': config['parameters'].get('intelligence_cleanup_days', {}).get('value', 60),
        'total_parameters': len(config['parameters']),
        'total_presets': len(config['presets']),
        'total_regime_modifiers': sum(len(v) for v in config['regime_modifiers'].values()),
    }


def get_parameters_by_category() -> Dict[str, List[Dict[str, Any]]]:
    """Get parameters grouped by category"""
    config = get_intelligence_config()

    by_category = {}
    for name, details in config['parameters'].items():
        category = details['category']
        if category not in by_category:
            by_category[category] = []
        by_category[category].append({
            'name': name,
            **details
        })

    # Sort parameters within each category by display_order
    for category in by_category:
        by_category[category].sort(key=lambda x: x.get('display_order', 999))

    return by_category
