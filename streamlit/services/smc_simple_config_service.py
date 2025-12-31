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
