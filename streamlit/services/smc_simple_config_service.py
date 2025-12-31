"""
SMC Simple Strategy Configuration Service for Streamlit

Provides database-driven configuration management with:
- Read/write operations for global and per-pair config
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

from .db_utils import get_connection_string

logger = logging.getLogger(__name__)


@st.cache_resource
def get_strategy_config_pool():
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
    pool = get_strategy_config_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


def get_global_config() -> Optional[Dict[str, Any]]:
    """Load active global configuration from database"""
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM smc_simple_global_config
                    WHERE is_active = TRUE
                    ORDER BY updated_at DESC
                    LIMIT 1
                """)
                row = cur.fetchone()
                if row:
                    return dict(row)
    except Exception as e:
        logger.error(f"Failed to load global config: {e}")
        st.error(f"Failed to load configuration: {e}")
    return None


def get_pair_overrides(config_id: int) -> List[Dict[str, Any]]:
    """Load all pair overrides for a config"""
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM smc_simple_pair_overrides
                    WHERE config_id = %s
                    ORDER BY epic
                """, (config_id,))
                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Failed to load pair overrides: {e}")
    return []


def get_pair_override(config_id: int, epic: str) -> Optional[Dict[str, Any]]:
    """Load specific pair override"""
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM smc_simple_pair_overrides
                    WHERE config_id = %s AND epic = %s
                """, (config_id, epic))
                row = cur.fetchone()
                if row:
                    return dict(row)
    except Exception as e:
        logger.error(f"Failed to load pair override: {e}")
    return None


def save_global_config(
    config_id: int,
    updates: Dict[str, Any],
    updated_by: str,
    change_reason: str
) -> bool:
    """Save updates to global configuration"""
    try:
        with get_connection() as conn:
            # Get current values for audit
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM smc_simple_global_config WHERE id = %s", (config_id,))
                old_values = dict(cur.fetchone()) if cur.rowcount else {}

            # Build update query
            update_fields = []
            values = []
            for key, value in updates.items():
                update_fields.append(f"{key} = %s")
                # Handle JSONB fields
                if key in ['adaptive_cooldown_config', 'confidence_weights', 'pair_pip_values']:
                    values.append(json.dumps(value) if isinstance(value, dict) else value)
                elif key == 'enabled_pairs':
                    values.append(list(value) if value else [])
                elif key == 'allowed_sessions':
                    values.append(list(value) if value else [])
                else:
                    values.append(value)

            update_fields.extend(['updated_by = %s', 'change_reason = %s'])
            values.extend([updated_by, change_reason])
            values.append(config_id)

            with conn.cursor() as cur:
                # Update config
                cur.execute(f"""
                    UPDATE smc_simple_global_config
                    SET {', '.join(update_fields)}
                    WHERE id = %s
                """, values)

                # Add audit record
                cur.execute("""
                    INSERT INTO smc_simple_config_audit
                    (config_id, change_type, changed_by, change_reason, previous_values, new_values)
                    VALUES (%s, 'UPDATE', %s, %s, %s, %s)
                """, (config_id, updated_by, change_reason,
                      json.dumps({k: str(v) for k, v in old_values.items() if k in updates}),
                      json.dumps({k: str(v) for k, v in updates.items()})))

            conn.commit()
            logger.info(f"Updated global config id={config_id}")
            return True

    except Exception as e:
        logger.error(f"Failed to save global config: {e}")
        st.error(f"Failed to save configuration: {e}")
    return False


def save_pair_override(
    config_id: int,
    epic: str,
    overrides: Dict[str, Any],
    updated_by: str,
    change_reason: str
) -> bool:
    """Save pair-specific overrides (insert or update)"""
    try:
        with get_connection() as conn:
            # Check if override exists
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id FROM smc_simple_pair_overrides
                    WHERE config_id = %s AND epic = %s
                """, (config_id, epic))
                existing = cur.fetchone()

            with conn.cursor() as cur:
                if existing:
                    # Update existing
                    override_id = existing[0]

                    # Build update fields
                    update_fields = []
                    values = []
                    for key, value in overrides.items():
                        if key in ['id', 'config_id', 'epic', 'created_at']:
                            continue
                        update_fields.append(f"{key} = %s")
                        if key in ['parameter_overrides', 'blocking_conditions']:
                            values.append(json.dumps(value) if isinstance(value, dict) else value)
                        else:
                            values.append(value)

                    update_fields.extend(['updated_by = %s', 'change_reason = %s'])
                    values.extend([updated_by, change_reason])
                    values.append(override_id)

                    cur.execute(f"""
                        UPDATE smc_simple_pair_overrides
                        SET {', '.join(update_fields)}
                        WHERE id = %s
                    """, values)

                    # Audit
                    cur.execute("""
                        INSERT INTO smc_simple_config_audit
                        (pair_override_id, change_type, changed_by, change_reason, new_values)
                        VALUES (%s, 'UPDATE', %s, %s, %s)
                    """, (override_id, updated_by, change_reason, json.dumps({k: str(v) for k, v in overrides.items()})))
                else:
                    # Insert new
                    columns = ['config_id', 'epic', 'updated_by', 'change_reason']
                    values = [config_id, epic, updated_by, change_reason]

                    for key, value in overrides.items():
                        if key in ['id', 'config_id', 'epic', 'created_at', 'updated_at']:
                            continue
                        columns.append(key)
                        if key in ['parameter_overrides', 'blocking_conditions']:
                            values.append(json.dumps(value) if isinstance(value, dict) else value)
                        else:
                            values.append(value)

                    placeholders = ', '.join(['%s'] * len(columns))
                    cur.execute(f"""
                        INSERT INTO smc_simple_pair_overrides ({', '.join(columns)})
                        VALUES ({placeholders})
                        RETURNING id
                    """, values)
                    override_id = cur.fetchone()[0]

                    # Audit
                    cur.execute("""
                        INSERT INTO smc_simple_config_audit
                        (pair_override_id, change_type, changed_by, change_reason, new_values)
                        VALUES (%s, 'CREATE', %s, %s, %s)
                    """, (override_id, updated_by, change_reason, json.dumps({k: str(v) for k, v in overrides.items()})))

            conn.commit()
            logger.info(f"Saved pair override for {epic}")
            return True

    except Exception as e:
        logger.error(f"Failed to save pair override: {e}")
        st.error(f"Failed to save pair override: {e}")
    return False


def delete_pair_override(
    override_id: int,
    updated_by: str,
    change_reason: str
) -> bool:
    """Delete a pair override"""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Audit first
                cur.execute("""
                    INSERT INTO smc_simple_config_audit
                    (pair_override_id, change_type, changed_by, change_reason)
                    VALUES (%s, 'DELETE', %s, %s)
                """, (override_id, updated_by, change_reason))

                # Delete
                cur.execute("DELETE FROM smc_simple_pair_overrides WHERE id = %s", (override_id,))

            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Failed to delete pair override: {e}")
    return False


def get_audit_history(limit: int = 50, epic: str = None) -> List[Dict[str, Any]]:
    """Get recent configuration changes"""
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                query = """
                    SELECT
                        a.id,
                        a.config_id,
                        a.pair_override_id,
                        a.change_type,
                        a.changed_by,
                        a.changed_at,
                        a.change_reason,
                        a.previous_values,
                        a.new_values,
                        p.epic
                    FROM smc_simple_config_audit a
                    LEFT JOIN smc_simple_pair_overrides p ON a.pair_override_id = p.id
                """
                params = []
                if epic:
                    query += " WHERE p.epic = %s"
                    params.append(epic)
                query += " ORDER BY a.changed_at DESC LIMIT %s"
                params.append(limit)

                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get audit history: {e}")
    return []


def get_enabled_pairs() -> List[str]:
    """Get list of enabled pairs from config"""
    config = get_global_config()
    if config and config.get('enabled_pairs'):
        return list(config['enabled_pairs'])
    return []


def get_effective_config_for_pair(epic: str) -> Dict[str, Any]:
    """Get merged global + pair-specific config"""
    config = get_global_config()
    if not config:
        return {}

    # Start with global values
    effective = {}
    for key, value in config.items():
        if key not in ['id', 'created_at', 'updated_at', '_pair_overrides']:
            # Handle special types
            if isinstance(value, datetime):
                effective[key] = value.isoformat()
            else:
                effective[key] = value

    # Get pair override
    override = get_pair_override(config['id'], epic)
    if override:
        # Apply simple overrides
        for field in ['sl_buffer_pips', 'min_confidence', 'allow_asian_session',
                     'min_volume_ratio', 'macd_filter_enabled',
                     'high_volume_confidence', 'low_atr_confidence',
                     'high_atr_confidence', 'near_ema_confidence', 'far_ema_confidence']:
            if override.get(field) is not None:
                effective[field] = override[field]

        # Apply parameter_overrides JSONB
        param_overrides = override.get('parameter_overrides')
        if param_overrides:
            if isinstance(param_overrides, str):
                param_overrides = json.loads(param_overrides)
            effective.update(param_overrides)

    return effective


# ============================================================================
# PARAMETER OPTIMIZER FUNCTIONS
# ============================================================================

def fetch_optimizer_recommendations(days: int = 30) -> Dict[str, Any]:
    """
    Fetch parameter optimization recommendations from the FastAPI rejection outcome API.

    Returns:
        Dictionary with recommendations and metadata
    """
    import os
    import requests

    fastapi_url = os.getenv('FASTAPI_URL', 'http://fastapi-dev:8000')
    api_key = os.getenv('FASTAPI_API_KEY', '436abe054a074894a0517e5172f0e5b6')
    headers = {
        'X-APIM-Gateway': 'verified',
        'X-API-KEY': api_key
    }

    result = {
        'recommendations': [],
        'stage_metrics': [],
        'pair_metrics': [],
        'error': None
    }

    try:
        # Fetch stage metrics
        stage_resp = requests.get(
            f"{fastapi_url}/api/rejection-outcomes/win-rate-by-stage",
            params={'days': days},
            headers=headers,
            timeout=30
        )
        if stage_resp.ok:
            result['stage_metrics'] = stage_resp.json()

        # Fetch pair metrics
        pair_resp = requests.get(
            f"{fastapi_url}/api/rejection-outcomes/by-pair",
            params={'days': days},
            headers=headers,
            timeout=30
        )
        if pair_resp.ok:
            result['pair_metrics'] = pair_resp.json()

        # Fetch parameter suggestions
        suggest_resp = requests.get(
            f"{fastapi_url}/api/rejection-outcomes/parameter-suggestions",
            params={'days': days},
            headers=headers,
            timeout=30
        )
        if suggest_resp.ok:
            result['suggestions'] = suggest_resp.json()

        # Generate recommendations from the data
        result['recommendations'] = _generate_recommendations(
            result['stage_metrics'],
            result['pair_metrics'],
            days
        )

    except requests.RequestException as e:
        logger.error(f"Failed to fetch recommendations: {e}")
        result['error'] = str(e)
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        result['error'] = str(e)

    return result


def _generate_recommendations(
    stage_metrics: List[Dict],
    pair_metrics: List[Dict],
    days: int
) -> List[Dict[str, Any]]:
    """Generate actionable recommendations from metrics data."""
    from decimal import Decimal

    recommendations = []
    config = get_global_config()
    if not config:
        return recommendations

    config_id = config.get('id')
    pair_overrides = get_pair_overrides(config_id) if config_id else []
    pair_override_map = {o['epic']: o for o in pair_overrides}

    # Stage-based recommendations
    # Maps rejection stages to the parameters that control them
    # Only stages with attempted_direction can be analyzed for outcomes:
    # - SESSION, COOLDOWN, TIER1_EMA reject before direction is known
    # - TIER2_SWING, TIER3_PULLBACK, RISK_LIMIT, CONFIDENCE, VOLUME_LOW, MACD_MISALIGNED have direction
    # Parameter bounds to prevent over-optimization
    PARAM_BOUNDS = {
        'min_confidence_threshold': {'min': 0.40, 'max': 0.60},
        'min_body_percentage': {'min': 0.10, 'max': 0.50},
        'fib_pullback_min': {'min': 0.15, 'max': 0.40},
        'min_volume_ratio': {'min': 0.30, 'max': 0.80},
        'min_rr_ratio': {'min': 1.0, 'max': 2.5},
    }

    STAGE_MAPPINGS = {
        'CONFIDENCE': {
            'param': 'min_confidence_threshold',
            'relax_delta': -0.02,
            'tighten_delta': +0.02,
            'description': 'Minimum confidence score threshold',
        },
        'CONFIDENCE_CAP': {
            'param': 'min_confidence_threshold',  # Same param, different stage
            'relax_delta': -0.02,
            'tighten_delta': +0.02,
            'description': 'Confidence cap threshold',
        },
        'TIER2_SWING': {
            'param': 'min_body_percentage',
            'relax_delta': -0.05,
            'tighten_delta': +0.05,
            'description': 'Minimum candle body percentage for swing break',
        },
        'TIER3_PULLBACK': {
            'param': 'fib_pullback_min',
            'relax_delta': -0.02,
            'tighten_delta': +0.02,
            'description': 'Minimum Fibonacci pullback level',
        },
        'VOLUME_LOW': {
            'param': 'min_volume_ratio',
            'relax_delta': -0.05,
            'tighten_delta': +0.05,
            'description': 'Minimum volume ratio filter',
        },
        'MACD_MISALIGNED': {
            'param': 'macd_alignment_filter_enabled',
            'relax_delta': False,  # Disable filter
            'tighten_delta': True,  # Enable filter
            'is_boolean': True,
            'description': 'MACD alignment filter',
        },
        'RISK_LIMIT': {
            'param': 'min_rr_ratio',
            'relax_delta': -0.1,
            'tighten_delta': +0.1,
            'description': 'Minimum risk-reward ratio',
        },
    }

    for stage in stage_metrics:
        stage_name = stage.get('rejection_stage', '')
        win_rate = stage.get('would_be_win_rate', 50)
        total = stage.get('total_analyzed', 0)
        missed_pips = stage.get('missed_profit_pips', 0)
        avoided_pips = stage.get('avoided_loss_pips', 0)

        if total < 20 or stage_name not in STAGE_MAPPINGS:
            continue

        mapping = STAGE_MAPPINGS[stage_name]
        param = mapping['param']
        current_val = config.get(param)

        if current_val is None:
            continue

        # Handle boolean vs numeric parameters
        is_boolean = mapping.get('is_boolean', False)

        # Convert Decimal if needed
        if not is_boolean and isinstance(current_val, Decimal):
            current_val = float(current_val)

        # Get bounds for this parameter
        bounds = PARAM_BOUNDS.get(param, {})
        param_min = bounds.get('min', float('-inf'))
        param_max = bounds.get('max', float('inf'))

        if win_rate > 60:
            # Too aggressive - relax
            if is_boolean:
                new_val = mapping['relax_delta']  # Boolean value directly
            else:
                new_val = round(current_val + mapping['relax_delta'], 3)
                # Apply bounds
                new_val = max(param_min, new_val)

            # Skip if no change needed or already at bound
            if new_val == current_val or (not is_boolean and current_val <= param_min):
                continue

            recommendations.append({
                'scope': 'global',
                'target': param,
                'current_value': current_val,
                'recommended_value': new_val,
                'action': 'relax',
                'reason': f"{stage_name} rejects {win_rate:.0f}% would-be winners ({total} samples, {missed_pips:.0f} missed pips)",
                'confidence': min(total / 100, 1.0),
                'impact_pips': missed_pips,
                'stage': stage_name,
                'description': mapping.get('description', ''),
            })
        elif win_rate < 40:
            # Working well - could tighten
            if is_boolean:
                new_val = mapping['tighten_delta']  # Boolean value directly
            else:
                new_val = round(current_val + mapping['tighten_delta'], 3)
                # Apply bounds
                new_val = min(param_max, new_val)

            # Skip if no change needed or already at bound
            if new_val == current_val or (not is_boolean and current_val >= param_max):
                continue

            recommendations.append({
                'scope': 'global',
                'target': param,
                'current_value': current_val,
                'recommended_value': new_val,
                'action': 'tighten',
                'reason': f"{stage_name} correctly filters {100-win_rate:.0f}% losers ({total} samples, {avoided_pips:.0f} avoided pips)",
                'confidence': min(total / 100, 1.0),
                'impact_pips': avoided_pips,
                'stage': stage_name,
                'description': mapping.get('description', ''),
            })

    # Pair-based recommendations
    global_min_conf = float(config.get('min_confidence_threshold', 0.48))
    global_min_vol = float(config.get('min_volume_ratio', 0.5))

    for pair_info in pair_metrics:
        epic = pair_info.get('epic', '')
        pair = pair_info.get('pair', '')
        win_rate = pair_info.get('would_be_win_rate', 50)
        total = pair_info.get('total_analyzed', 0)
        status = pair_info.get('status', 'NEUTRAL')
        missed_pips = pair_info.get('missed_profit_pips', 0)
        avoided_pips = pair_info.get('avoided_loss_pips', 0)

        if total < 15:
            continue

        override = pair_override_map.get(epic, {})

        if status == 'TOO_AGGRESSIVE' and win_rate > 60:
            # Relax confidence for this pair
            current_conf = override.get('min_confidence') or global_min_conf
            if isinstance(current_conf, Decimal):
                current_conf = float(current_conf)

            # Bounds: don't go below 0.40 (absolute minimum safe threshold)
            MIN_CONFIDENCE_FLOOR = 0.40
            new_conf = max(MIN_CONFIDENCE_FLOOR, current_conf - 0.03)

            # Skip if already at floor or change is negligible
            if current_conf <= MIN_CONFIDENCE_FLOOR or abs(new_conf - current_conf) < 0.01:
                continue

            recommendations.append({
                'scope': 'pair',
                'target': (epic, 'min_confidence'),
                'pair_name': pair,
                'current_value': current_conf,
                'recommended_value': round(new_conf, 3),
                'action': 'relax',
                'reason': f"{pair} filters reject {win_rate:.0f}% would-be winners ({total} samples)",
                'confidence': min(total / 50, 1.0),
                'impact_pips': missed_pips,
                'stage': 'PAIR_SPECIFIC',
            })

        elif status == 'WORKING_WELL' and win_rate < 40:
            # Tighten volume filter for this pair
            current_vol = override.get('min_volume_ratio') or global_min_vol
            if isinstance(current_vol, Decimal):
                current_vol = float(current_vol)

            # Bounds: don't go above 0.80 (would filter too many signals)
            MAX_VOLUME_CEILING = 0.80
            new_vol = min(MAX_VOLUME_CEILING, current_vol + 0.05)

            # Skip if already at ceiling or change is negligible
            if current_vol >= MAX_VOLUME_CEILING or abs(new_vol - current_vol) < 0.01:
                continue

            recommendations.append({
                'scope': 'pair',
                'target': (epic, 'min_volume_ratio'),
                'pair_name': pair,
                'current_value': current_vol,
                'recommended_value': round(new_vol, 2),
                'action': 'tighten',
                'reason': f"{pair} filters are protective ({100-win_rate:.0f}% losers filtered)",
                'confidence': min(total / 50, 1.0),
                'impact_pips': avoided_pips,
                'stage': 'PAIR_SPECIFIC',
            })

    return recommendations


def apply_optimizer_recommendations(
    recommendations: List[Dict[str, Any]],
    updated_by: str = 'parameter_optimizer'
) -> Dict[str, Any]:
    """
    Apply parameter optimization recommendations to the database.

    Returns:
        Dictionary with success status and details
    """
    result = {
        'success': True,
        'applied_count': 0,
        'failed_count': 0,
        'details': []
    }

    if not recommendations:
        return result

    config = get_global_config()
    if not config:
        result['success'] = False
        result['error'] = 'No active configuration found'
        return result

    config_id = config['id']

    for rec in recommendations:
        try:
            if rec['scope'] == 'global':
                param = rec['target']
                new_value = rec['recommended_value']
                reason = f"Auto-optimized: {rec['reason'][:100]}"

                success = save_global_config(
                    config_id,
                    {param: new_value},
                    updated_by,
                    reason
                )

                if success:
                    result['applied_count'] += 1
                    result['details'].append({
                        'type': 'global',
                        'param': param,
                        'old': rec['current_value'],
                        'new': new_value,
                        'success': True
                    })
                else:
                    result['failed_count'] += 1
                    result['details'].append({
                        'type': 'global',
                        'param': param,
                        'success': False,
                        'error': 'Save failed'
                    })

            elif rec['scope'] == 'pair':
                epic, param = rec['target']
                new_value = rec['recommended_value']
                reason = f"Auto-optimized: {rec['reason'][:100]}"

                success = save_pair_override(
                    config_id,
                    epic,
                    {param: new_value},
                    updated_by,
                    reason
                )

                if success:
                    result['applied_count'] += 1
                    result['details'].append({
                        'type': 'pair',
                        'epic': epic,
                        'param': param,
                        'old': rec['current_value'],
                        'new': new_value,
                        'success': True
                    })
                else:
                    result['failed_count'] += 1
                    result['details'].append({
                        'type': 'pair',
                        'epic': epic,
                        'param': param,
                        'success': False,
                        'error': 'Save failed'
                    })

        except Exception as e:
            result['failed_count'] += 1
            result['details'].append({
                'error': str(e),
                'recommendation': rec
            })

    result['success'] = result['failed_count'] == 0
    return result
