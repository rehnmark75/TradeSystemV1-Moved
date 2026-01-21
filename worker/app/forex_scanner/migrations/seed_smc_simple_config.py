#!/usr/bin/env python3
"""
Migration script to seed database from config_smc_simple.py

Reads all configuration parameters from the file-based config and inserts
them into the strategy_config database.

Run from task-worker container:
    docker exec -it task-worker python /app/forex_scanner/migrations/seed_smc_simple_config.py
"""

import sys
import os
import json
from datetime import time as dt_time

# Add app to path
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/forex_scanner')

import psycopg2
import psycopg2.extras


def get_database_url():
    """Get database URL"""
    return os.getenv(
        'STRATEGY_CONFIG_DATABASE_URL',
        'postgresql://postgres:postgres@postgres:5432/strategy_config'
    )


def load_file_config():
    """Load configuration from file"""
    try:
        from configdata.strategies import config_smc_simple as smc_config
    except ImportError:
        from forex_scanner.configdata.strategies import config_smc_simple as smc_config
    return smc_config


def time_to_str(t):
    """Convert time to string"""
    if isinstance(t, dt_time):
        return t.strftime('%H:%M')
    return str(t)


def migrate_global_config(conn, smc_config):
    """Migrate global configuration from file to database"""
    print("\nMigrating global configuration...")

    # Build insert data from file config
    global_data = {
        'version': getattr(smc_config, 'STRATEGY_VERSION', '2.11.0'),
        'is_active': True,
        'strategy_name': getattr(smc_config, 'STRATEGY_NAME', 'SMC_SIMPLE'),
        'strategy_status': getattr(smc_config, 'STRATEGY_STATUS', ''),

        # TIER 1
        'htf_timeframe': getattr(smc_config, 'HTF_TIMEFRAME', '4h'),
        'ema_period': getattr(smc_config, 'EMA_PERIOD', 50),
        'ema_buffer_pips': float(getattr(smc_config, 'EMA_BUFFER_PIPS', 2.5)),
        'require_close_beyond_ema': getattr(smc_config, 'REQUIRE_CLOSE_BEYOND_EMA', True),
        'min_distance_from_ema_pips': float(getattr(smc_config, 'MIN_DISTANCE_FROM_EMA_PIPS', 3)),

        # TIER 2
        'trigger_timeframe': getattr(smc_config, 'TRIGGER_TIMEFRAME', '15m'),
        'swing_lookback_bars': getattr(smc_config, 'SWING_LOOKBACK_BARS', 20),
        'swing_strength_bars': getattr(smc_config, 'SWING_STRENGTH_BARS', 2),
        'require_body_close_break': getattr(smc_config, 'REQUIRE_BODY_CLOSE_BREAK', False),
        'wick_tolerance_pips': float(getattr(smc_config, 'WICK_TOLERANCE_PIPS', 3)),
        'volume_confirmation_enabled': getattr(smc_config, 'VOLUME_CONFIRMATION_ENABLED', True),
        'volume_sma_period': getattr(smc_config, 'VOLUME_SMA_PERIOD', 20),
        'volume_spike_multiplier': float(getattr(smc_config, 'VOLUME_SPIKE_MULTIPLIER', 1.2)),

        # Dynamic swing lookback
        'use_dynamic_swing_lookback': getattr(smc_config, 'USE_DYNAMIC_SWING_LOOKBACK', True),
        'swing_lookback_atr_low': getattr(smc_config, 'SWING_LOOKBACK_ATR_LOW', 8),
        'swing_lookback_atr_high': getattr(smc_config, 'SWING_LOOKBACK_ATR_HIGH', 15),
        'swing_lookback_min': getattr(smc_config, 'SWING_LOOKBACK_MIN', 15),
        'swing_lookback_max': getattr(smc_config, 'SWING_LOOKBACK_MAX', 30),

        # TIER 3
        'entry_timeframe': getattr(smc_config, 'ENTRY_TIMEFRAME', '5m'),
        'pullback_enabled': getattr(smc_config, 'PULLBACK_ENABLED', True),
        'fib_pullback_min': float(getattr(smc_config, 'FIB_PULLBACK_MIN', 0.236)),
        'fib_pullback_max': float(getattr(smc_config, 'FIB_PULLBACK_MAX', 0.70)),
        'max_pullback_wait_bars': getattr(smc_config, 'MAX_PULLBACK_WAIT_BARS', 12),
        'pullback_confirmation_bars': getattr(smc_config, 'PULLBACK_CONFIRMATION_BARS', 2),

        # Fib optimal zone
        'fib_optimal_zone_min': float(getattr(smc_config, 'FIB_OPTIMAL_ZONE', (0.382, 0.618))[0]),
        'fib_optimal_zone_max': float(getattr(smc_config, 'FIB_OPTIMAL_ZONE', (0.382, 0.618))[1]),

        # Momentum mode
        'momentum_mode_enabled': getattr(smc_config, 'MOMENTUM_MODE_ENABLED', True),
        'momentum_min_depth': float(getattr(smc_config, 'MOMENTUM_MIN_DEPTH', -0.50)),
        'momentum_max_depth': float(getattr(smc_config, 'MOMENTUM_MAX_DEPTH', 0.0)),
        'momentum_confidence_penalty': float(getattr(smc_config, 'MOMENTUM_CONFIDENCE_PENALTY', 0.05)),

        # ATR swing validation
        'use_atr_swing_validation': getattr(smc_config, 'USE_ATR_SWING_VALIDATION', True),
        'atr_period': getattr(smc_config, 'ATR_PERIOD', 14),
        'min_swing_atr_multiplier': float(getattr(smc_config, 'MIN_SWING_ATR_MULTIPLIER', 0.25)),
        'fallback_min_swing_pips': float(getattr(smc_config, 'FALLBACK_MIN_SWING_PIPS', 5)),

        # Momentum quality
        'momentum_quality_enabled': getattr(smc_config, 'MOMENTUM_QUALITY_ENABLED', True),
        'min_breakout_atr_ratio': float(getattr(smc_config, 'MIN_BREAKOUT_ATR_RATIO', 0.5)),
        'min_body_percentage': float(getattr(smc_config, 'MIN_BODY_PERCENTAGE', 0.20)),

        # Limit orders
        'limit_order_enabled': getattr(smc_config, 'LIMIT_ORDER_ENABLED', True),
        'limit_expiry_minutes': getattr(smc_config, 'LIMIT_EXPIRY_MINUTES', 45),
        'pullback_offset_atr_factor': float(getattr(smc_config, 'PULLBACK_OFFSET_ATR_FACTOR', 0.2)),
        'pullback_offset_min_pips': float(getattr(smc_config, 'PULLBACK_OFFSET_MIN_PIPS', 2.0)),
        'pullback_offset_max_pips': float(getattr(smc_config, 'PULLBACK_OFFSET_MAX_PIPS', 3.0)),
        'momentum_offset_pips': float(getattr(smc_config, 'MOMENTUM_OFFSET_PIPS', 3.0)),
        'min_risk_after_offset_pips': float(getattr(smc_config, 'MIN_RISK_AFTER_OFFSET_PIPS', 5.0)),
        'max_sl_atr_multiplier': float(getattr(smc_config, 'MAX_SL_ATR_MULTIPLIER', 3.0)),
        'max_sl_absolute_pips': float(getattr(smc_config, 'MAX_SL_ABSOLUTE_PIPS', 30.0)),
        'max_risk_after_offset_pips': float(getattr(smc_config, 'MAX_RISK_AFTER_OFFSET_PIPS', 55.0)),

        # Risk management
        'min_rr_ratio': float(getattr(smc_config, 'MIN_RR_RATIO', 1.5)),
        'optimal_rr_ratio': float(getattr(smc_config, 'OPTIMAL_RR_RATIO', 2.5)),
        'max_rr_ratio': float(getattr(smc_config, 'MAX_RR_RATIO', 5.0)),
        'sl_buffer_pips': getattr(smc_config, 'SL_BUFFER_PIPS', 6),
        'sl_atr_multiplier': float(getattr(smc_config, 'SL_ATR_MULTIPLIER', 1.0)),
        'use_atr_stop': getattr(smc_config, 'USE_ATR_STOP', True),
        'min_tp_pips': getattr(smc_config, 'MIN_TP_PIPS', 8),
        'use_swing_target': getattr(smc_config, 'USE_SWING_TARGET', True),
        'tp_structure_lookback': getattr(smc_config, 'TP_STRUCTURE_LOOKBACK', 50),
        'risk_per_trade_pct': float(getattr(smc_config, 'RISK_PER_TRADE_PCT', 1.0)),

        # Session filter
        'session_filter_enabled': getattr(smc_config, 'SESSION_FILTER_ENABLED', True),
        'london_session_start': time_to_str(getattr(smc_config, 'LONDON_SESSION_START', dt_time(7, 0))),
        'london_session_end': time_to_str(getattr(smc_config, 'LONDON_SESSION_END', dt_time(16, 0))),
        'ny_session_start': time_to_str(getattr(smc_config, 'NY_SESSION_START', dt_time(12, 0))),
        'ny_session_end': time_to_str(getattr(smc_config, 'NY_SESSION_END', dt_time(21, 0))),
        'allowed_sessions': getattr(smc_config, 'ALLOWED_SESSIONS', ['london', 'new_york', 'overlap']),
        'block_asian_session': getattr(smc_config, 'BLOCK_ASIAN_SESSION', True),

        # Signal limits
        'max_concurrent_signals': getattr(smc_config, 'MAX_CONCURRENT_SIGNALS', 3),
        'signal_cooldown_hours': getattr(smc_config, 'SIGNAL_COOLDOWN_HOURS', 3),

        # Adaptive cooldown (JSONB)
        'adaptive_cooldown_config': json.dumps({
            'enabled': getattr(smc_config, 'ADAPTIVE_COOLDOWN_ENABLED', True),
            'base_cooldown_hours': float(getattr(smc_config, 'BASE_COOLDOWN_HOURS', 2.0)),
            'cooldown_after_win_multiplier': float(getattr(smc_config, 'COOLDOWN_AFTER_WIN_MULTIPLIER', 0.5)),
            'cooldown_after_loss_multiplier': float(getattr(smc_config, 'COOLDOWN_AFTER_LOSS_MULTIPLIER', 1.5)),
            'consecutive_loss_penalty_hours': float(getattr(smc_config, 'CONSECUTIVE_LOSS_PENALTY_HOURS', 1.0)),
            'max_consecutive_losses_before_block': getattr(smc_config, 'MAX_CONSECUTIVE_LOSSES_BEFORE_BLOCK', 3),
            'consecutive_loss_block_hours': float(getattr(smc_config, 'CONSECUTIVE_LOSS_BLOCK_HOURS', 8.0)),
            'win_rate_lookback_trades': getattr(smc_config, 'WIN_RATE_LOOKBACK_TRADES', 20),
            'high_win_rate_threshold': float(getattr(smc_config, 'HIGH_WIN_RATE_THRESHOLD', 0.60)),
            'low_win_rate_threshold': float(getattr(smc_config, 'LOW_WIN_RATE_THRESHOLD', 0.40)),
            'critical_win_rate_threshold': float(getattr(smc_config, 'CRITICAL_WIN_RATE_THRESHOLD', 0.30)),
            'high_win_rate_cooldown_reduction': float(getattr(smc_config, 'HIGH_WIN_RATE_COOLDOWN_REDUCTION', 0.25)),
            'low_win_rate_cooldown_increase': float(getattr(smc_config, 'LOW_WIN_RATE_COOLDOWN_INCREASE', 0.50)),
            'high_volatility_atr_multiplier': float(getattr(smc_config, 'HIGH_VOLATILITY_ATR_MULTIPLIER', 1.5)),
            'volatility_cooldown_adjustment': float(getattr(smc_config, 'VOLATILITY_COOLDOWN_ADJUSTMENT', 0.30)),
            'strong_trend_cooldown_reduction': float(getattr(smc_config, 'STRONG_TREND_COOLDOWN_REDUCTION', 0.30)),
            'session_change_reset_cooldown': getattr(smc_config, 'SESSION_CHANGE_RESET_COOLDOWN', True),
            'min_cooldown_hours': float(getattr(smc_config, 'MIN_COOLDOWN_HOURS', 1.0)),
            'max_cooldown_hours': float(getattr(smc_config, 'MAX_COOLDOWN_HOURS', 12.0)),
        }),

        # Confidence scoring
        'min_confidence_threshold': float(getattr(smc_config, 'MIN_CONFIDENCE_THRESHOLD', 0.48)),
        'max_confidence_threshold': float(getattr(smc_config, 'MAX_CONFIDENCE_THRESHOLD', 0.75)),
        'high_confidence_threshold': float(getattr(smc_config, 'HIGH_CONFIDENCE_THRESHOLD', 0.75)),
        'confidence_weights': json.dumps(getattr(smc_config, 'CONFIDENCE_WEIGHTS', {})),

        # Volume filter
        'min_volume_ratio': float(getattr(smc_config, 'MIN_VOLUME_RATIO', 0.50)),
        'volume_filter_enabled': getattr(smc_config, 'VOLUME_FILTER_ENABLED', True),
        'allow_no_volume_data': getattr(smc_config, 'ALLOW_NO_VOLUME_DATA', True),

        # Dynamic confidence thresholds
        'volume_adjusted_confidence_enabled': getattr(smc_config, 'VOLUME_ADJUSTED_CONFIDENCE_ENABLED', True),
        'high_volume_threshold': float(getattr(smc_config, 'HIGH_VOLUME_THRESHOLD', 0.70)),
        'atr_adjusted_confidence_enabled': getattr(smc_config, 'ATR_ADJUSTED_CONFIDENCE_ENABLED', True),
        'low_atr_threshold': float(getattr(smc_config, 'LOW_ATR_THRESHOLD', 0.0004)),
        'high_atr_threshold': float(getattr(smc_config, 'HIGH_ATR_THRESHOLD', 0.0008)),
        'ema_distance_adjusted_confidence_enabled': getattr(smc_config, 'EMA_DISTANCE_ADJUSTED_CONFIDENCE_ENABLED', True),
        'near_ema_threshold_pips': float(getattr(smc_config, 'NEAR_EMA_THRESHOLD_PIPS', 20.0)),
        'far_ema_threshold_pips': float(getattr(smc_config, 'FAR_EMA_THRESHOLD_PIPS', 30.0)),

        # MACD filter
        'macd_alignment_filter_enabled': getattr(smc_config, 'MACD_ALIGNMENT_FILTER_ENABLED', True),
        'macd_alignment_mode': getattr(smc_config, 'MACD_ALIGNMENT_MODE', 'momentum'),
        'macd_min_strength': float(getattr(smc_config, 'MACD_MIN_STRENGTH', 0.0)),

        # Logging
        'enable_debug_logging': getattr(smc_config, 'ENABLE_DEBUG_LOGGING', True),
        'log_rejected_signals': getattr(smc_config, 'LOG_REJECTED_SIGNALS', True),
        'log_swing_detection': getattr(smc_config, 'LOG_SWING_DETECTION', False),
        'log_ema_checks': getattr(smc_config, 'LOG_EMA_CHECKS', False),

        # Rejection tracking
        'rejection_tracking_enabled': getattr(smc_config, 'REJECTION_TRACKING_ENABLED', True),
        'rejection_batch_size': getattr(smc_config, 'REJECTION_BATCH_SIZE', 50),
        'rejection_log_to_console': getattr(smc_config, 'REJECTION_LOG_TO_CONSOLE', False),
        'rejection_retention_days': getattr(smc_config, 'REJECTION_RETENTION_DAYS', 90),

        # Backtest
        'backtest_spread_pips': float(getattr(smc_config, 'BACKTEST_SPREAD_PIPS', 1.5)),
        'backtest_slippage_pips': float(getattr(smc_config, 'BACKTEST_SLIPPAGE_PIPS', 0.5)),

        # Scalp reversal override (counter-trend)
        'scalp_reversal_enabled': getattr(smc_config, 'SCALP_REVERSAL_ENABLED', True),
        'scalp_reversal_min_runway_pips': float(getattr(smc_config, 'SCALP_REVERSAL_MIN_RUNWAY_PIPS', 15.0)),
        'scalp_reversal_min_entry_momentum': float(getattr(smc_config, 'SCALP_REVERSAL_MIN_ENTRY_MOMENTUM', 0.60)),
        'scalp_reversal_block_regimes': getattr(smc_config, 'SCALP_REVERSAL_BLOCK_REGIMES', ['breakout']),
        'scalp_reversal_block_volatility_states': getattr(smc_config, 'SCALP_REVERSAL_BLOCK_VOLATILITY_STATES', ['high']),
        'scalp_reversal_allow_rsi_extremes': getattr(smc_config, 'SCALP_REVERSAL_ALLOW_RSI_EXTREMES', True),

        # Enabled pairs
        'enabled_pairs': getattr(smc_config, 'ENABLED_PAIRS', []),

        # Pair pip values
        'pair_pip_values': json.dumps(getattr(smc_config, 'PAIR_PIP_VALUES', {})),

        # Audit
        'updated_by': 'migration_script',
        'change_reason': 'Initial migration from config_smc_simple.py',
    }

    # Build insert query
    columns = list(global_data.keys())
    placeholders = ', '.join(['%s'] * len(columns))
    column_names = ', '.join(columns)

    with conn.cursor() as cur:
        # Deactivate any existing config
        cur.execute("UPDATE smc_simple_global_config SET is_active = FALSE")

        # Insert new config
        cur.execute(f"""
            INSERT INTO smc_simple_global_config ({column_names})
            VALUES ({placeholders})
            RETURNING id
        """, list(global_data.values()))

        config_id = cur.fetchone()[0]
        print(f"  Inserted global config with id={config_id}")

        return config_id


def migrate_pair_overrides(conn, config_id, smc_config):
    """Migrate per-pair overrides from file to database"""
    print("\nMigrating pair overrides...")

    # Get all override dictionaries
    pair_overrides = getattr(smc_config, 'PAIR_PARAMETER_OVERRIDES', {})
    session_overrides = getattr(smc_config, 'PAIR_SESSION_OVERRIDES', {})
    sl_buffers = getattr(smc_config, 'PAIR_SL_BUFFERS', {})
    min_confidence = getattr(smc_config, 'PAIR_MIN_CONFIDENCE', {})
    high_vol_conf = getattr(smc_config, 'PAIR_HIGH_VOLUME_CONFIDENCE', {})
    low_atr_conf = getattr(smc_config, 'PAIR_LOW_ATR_CONFIDENCE', {})
    high_atr_conf = getattr(smc_config, 'PAIR_HIGH_ATR_CONFIDENCE', {})
    near_ema_conf = getattr(smc_config, 'PAIR_NEAR_EMA_CONFIDENCE', {})
    far_ema_conf = getattr(smc_config, 'PAIR_FAR_EMA_CONFIDENCE', {})
    blocking = getattr(smc_config, 'PAIR_BLOCKING_CONDITIONS', {})
    macd_overrides = getattr(smc_config, 'PAIR_MACD_FILTER_OVERRIDES', {})

    # Collect all unique epics (only IG format)
    all_epics = set()
    for d in [pair_overrides, session_overrides, sl_buffers, min_confidence,
              high_vol_conf, low_atr_conf, high_atr_conf, near_ema_conf,
              far_ema_conf, blocking, macd_overrides]:
        all_epics.update([e for e in d.keys() if e.startswith('CS.D.')])

    print(f"  Found {len(all_epics)} unique epics with overrides")

    count = 0
    with conn.cursor() as cur:
        for epic in sorted(all_epics):
            # Build override record
            override_data = {
                'config_id': config_id,
                'epic': epic,
                'is_enabled': True,
                'description': None,
                'parameter_overrides': None,
                'high_volume_confidence': None,
                'low_atr_confidence': None,
                'high_atr_confidence': None,
                'near_ema_confidence': None,
                'far_ema_confidence': None,
                'allow_asian_session': None,
                'sl_buffer_pips': None,
                'min_confidence': None,
                'min_volume_ratio': None,
                'macd_filter_enabled': None,
                'blocking_conditions': None,
                'updated_by': 'migration_script',
                'change_reason': 'Migration from config_smc_simple.py',
            }

            # Get parameter overrides
            if epic in pair_overrides:
                po = pair_overrides[epic]
                override_data['is_enabled'] = po.get('enabled', True)
                override_data['description'] = po.get('description', '')
                overrides_dict = po.get('overrides', {})
                # Extract min_volume_ratio if in overrides
                if 'MIN_VOLUME_RATIO' in overrides_dict:
                    override_data['min_volume_ratio'] = float(overrides_dict.pop('MIN_VOLUME_RATIO'))
                if overrides_dict:
                    override_data['parameter_overrides'] = json.dumps(overrides_dict)

            # Session override
            if epic in session_overrides:
                override_data['allow_asian_session'] = session_overrides[epic].get('allow_asian')

            # SL buffer
            if epic in sl_buffers:
                override_data['sl_buffer_pips'] = sl_buffers[epic]

            # Confidence thresholds
            if epic in min_confidence:
                override_data['min_confidence'] = float(min_confidence[epic])
            if epic in high_vol_conf:
                override_data['high_volume_confidence'] = float(high_vol_conf[epic])
            if epic in low_atr_conf:
                override_data['low_atr_confidence'] = float(low_atr_conf[epic])
            if epic in high_atr_conf:
                override_data['high_atr_confidence'] = float(high_atr_conf[epic])
            if epic in near_ema_conf:
                override_data['near_ema_confidence'] = float(near_ema_conf[epic])
            if epic in far_ema_conf:
                override_data['far_ema_confidence'] = float(far_ema_conf[epic])

            # Blocking conditions
            if epic in blocking:
                override_data['blocking_conditions'] = json.dumps(blocking[epic])

            # MACD filter
            if epic in macd_overrides:
                override_data['macd_filter_enabled'] = macd_overrides[epic]

            # Insert
            columns = list(override_data.keys())
            placeholders = ', '.join(['%s'] * len(columns))
            column_names = ', '.join(columns)

            cur.execute(f"""
                INSERT INTO smc_simple_pair_overrides ({column_names})
                VALUES ({placeholders})
                ON CONFLICT (config_id, epic) DO UPDATE SET
                    is_enabled = EXCLUDED.is_enabled,
                    description = EXCLUDED.description,
                    parameter_overrides = COALESCE(EXCLUDED.parameter_overrides, smc_simple_pair_overrides.parameter_overrides),
                    allow_asian_session = COALESCE(EXCLUDED.allow_asian_session, smc_simple_pair_overrides.allow_asian_session),
                    sl_buffer_pips = COALESCE(EXCLUDED.sl_buffer_pips, smc_simple_pair_overrides.sl_buffer_pips),
                    min_confidence = COALESCE(EXCLUDED.min_confidence, smc_simple_pair_overrides.min_confidence),
                    min_volume_ratio = COALESCE(EXCLUDED.min_volume_ratio, smc_simple_pair_overrides.min_volume_ratio),
                    high_volume_confidence = COALESCE(EXCLUDED.high_volume_confidence, smc_simple_pair_overrides.high_volume_confidence),
                    low_atr_confidence = COALESCE(EXCLUDED.low_atr_confidence, smc_simple_pair_overrides.low_atr_confidence),
                    high_atr_confidence = COALESCE(EXCLUDED.high_atr_confidence, smc_simple_pair_overrides.high_atr_confidence),
                    near_ema_confidence = COALESCE(EXCLUDED.near_ema_confidence, smc_simple_pair_overrides.near_ema_confidence),
                    far_ema_confidence = COALESCE(EXCLUDED.far_ema_confidence, smc_simple_pair_overrides.far_ema_confidence),
                    blocking_conditions = COALESCE(EXCLUDED.blocking_conditions, smc_simple_pair_overrides.blocking_conditions),
                    macd_filter_enabled = COALESCE(EXCLUDED.macd_filter_enabled, smc_simple_pair_overrides.macd_filter_enabled),
                    updated_at = CURRENT_TIMESTAMP
            """, list(override_data.values()))
            count += 1

    print(f"  Migrated {count} pair overrides")
    return count


def verify_migration(conn):
    """Verify migration was successful"""
    print("\nVerifying migration...")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Check global config
        cur.execute("SELECT * FROM smc_simple_global_config WHERE is_active = TRUE")
        global_row = cur.fetchone()

        if global_row:
            print(f"\n  Global config verified:")
            print(f"    - ID: {global_row['id']}")
            print(f"    - Version: {global_row['version']}")
            print(f"    - EMA Period: {global_row['ema_period']}")
            print(f"    - Min Confidence: {global_row['min_confidence_threshold']}")
            print(f"    - Enabled Pairs: {len(global_row['enabled_pairs'])} pairs")
        else:
            print("  ERROR: No active global config found!")
            return False

        # Check pair overrides
        cur.execute("""
            SELECT COUNT(*) as count FROM smc_simple_pair_overrides
            WHERE config_id = %s
        """, (global_row['id'],))
        override_count = cur.fetchone()['count']
        print(f"    - Pair overrides: {override_count} records")

        # List pair overrides
        cur.execute("""
            SELECT epic, description, min_confidence, sl_buffer_pips, allow_asian_session
            FROM smc_simple_pair_overrides
            WHERE config_id = %s
            ORDER BY epic
        """, (global_row['id'],))
        overrides = cur.fetchall()
        print(f"\n  Pair override details:")
        for o in overrides:
            desc = o['description'][:50] + '...' if o['description'] and len(o['description']) > 50 else (o['description'] or 'N/A')
            print(f"    - {o['epic']}: conf={o['min_confidence']}, sl={o['sl_buffer_pips']}, asian={o['allow_asian_session']}")

        return True


def main():
    print("=" * 60)
    print("SMC Simple Config Migration: File -> Database")
    print("=" * 60)

    # Load file config
    print("\nLoading file configuration...")
    smc_config = load_file_config()
    print(f"  Loaded version: {getattr(smc_config, 'STRATEGY_VERSION', 'unknown')}")

    # Connect to database
    db_url = get_database_url()
    print(f"\nConnecting to database...")
    print(f"  URL: {db_url}")

    conn = psycopg2.connect(db_url)

    try:
        # Migrate global config
        config_id = migrate_global_config(conn, smc_config)
        conn.commit()

        # Migrate pair overrides
        migrate_pair_overrides(conn, config_id, smc_config)
        conn.commit()

        # Verify
        if verify_migration(conn):
            print("\n" + "=" * 60)
            print("Migration completed successfully!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("Migration verification FAILED!")
            print("=" * 60)
            return 1

    except Exception as e:
        conn.rollback()
        print(f"\nMigration FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
