# alerts/alert_history.py
"""
REFACTORED: Alert History Database System
Single Responsibility: Alert operations ONLY
Database connection management delegated to injected DatabaseManager

CHANGES MADE:
✅ Removed direct database connection creation (uses injected db_manager)
✅ All database operations now go through DatabaseManager
✅ Maintained all expected methods and functionality
✅ No breaking changes to external interface
✅ Cleaner error handling and resource management
✅ FIXED: Added missing _update_strategy_summary and _update_claude_analysis_summary methods
✅ FIXED: Enhanced save_alert method with better error handling
✅ FIXED: Corrected save_alert method signature to match calling code
✅ FIXED: Removed duplicate and broken method definitions
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
import numpy as np  # Add this import
import hashlib
from psycopg2.extras import RealDictCursor
try:
    from utils.scanner_utils import make_json_serializable
except ImportError:
    from forex_scanner.utils.scanner_utils import make_json_serializable

try:
    import config
except ImportError:
    from forex_scanner import config


class AlertHistoryManager:
    """
    REFACTORED: Manages alert history operations using injected DatabaseManager
    
    Single Responsibility: Alert-specific database operations ONLY
    No longer manages database connections - delegates to DatabaseManager
    """
    
    def __init__(self, db_manager):
        """
        Initialize with injected DatabaseManager
        
        Args:
            db_manager: DatabaseManager instance for all database operations
        """
        if db_manager is None:
            raise ValueError("DatabaseManager is required - cannot be None")
            
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self._initialize_alert_tables()
    
    def _get_connection(self):
        """
        Get database connection through injected DatabaseManager
        This is the ONLY place we get connections now
        """
        return self.db_manager.get_connection()
    
    def safe_json_dumps(self,obj):
        """Safely convert object to JSON string using existing utility"""
        if obj is None:
            return None
        
        try:
            # Use existing function that handles numpy types
            cleaned_obj = make_json_serializable(obj)
            return json.dumps(cleaned_obj)
        except (TypeError, ValueError) as e:
            # Fallback to string representation
            return str(obj)
    
    def _execute_with_connection(self, operation_func, operation_name="database operation"):
        """
        Execute database operation with proper connection management
        
        Args:
            operation_func: Function that takes (conn, cursor) and performs the operation
            operation_name: Name for logging purposes
            
        Returns:
            Result from operation_func or None on error
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            result = operation_func(conn, cursor)
            conn.commit()
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {operation_name} failed: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            raise
            
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def _convert_dict_to_json(self, value):
        """Convert dictionary to JSON string, handle None values"""
        if value is None:
            return None
        if isinstance(value, dict):
            return json.dumps(value)
        if isinstance(value, str):
            return value  # Already a string
        return str(value)  # Convert other types to string

    def _safe_json_convert(self, value):
        """Safely convert dict/list to JSON string"""
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value)
            except (TypeError, ValueError) as e:
                self.logger.warning(f"Failed to serialize to JSON: {e}")
                return None
        return str(value) if value else None
    
    def _safe_float_convert(self, value):
        """Safely convert value to float"""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    
    def _initialize_alert_tables(self):
        """Create alert history tables if they don't exist"""
        def create_tables_operation(conn, cursor):
            # Main alert history table - ENHANCED with Claude analysis fields
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_history (
                    id SERIAL PRIMARY KEY,
                    alert_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    epic VARCHAR(50) NOT NULL,
                    pair VARCHAR(10) NOT NULL,
                    signal_type VARCHAR(10) NOT NULL,
                    strategy VARCHAR(50) NOT NULL,
                    confidence_score DECIMAL(5,4) NOT NULL,
                    price DECIMAL(10,5) NOT NULL,
                    bid_price DECIMAL(10,5),
                    ask_price DECIMAL(10,5),
                    spread_pips DECIMAL(5,2),
                    timeframe VARCHAR(10) NOT NULL,
                    
                    -- Generic strategy parameters (JSON)
                    strategy_config JSON,
                    strategy_indicators JSON,
                    strategy_metadata JSON,
                    
                    -- Common technical indicators
                    ema_short DECIMAL(10,5),
                    ema_long DECIMAL(10,5), 
                    ema_trend DECIMAL(10,5),
                    
                    macd_line DECIMAL(10,6),
                    macd_signal DECIMAL(10,6),
                    macd_histogram DECIMAL(10,6),

                    -- Technical indicators (ADX, RSI, ATR)
                    adx DECIMAL(6,2),
                    adx_plus DECIMAL(6,2),
                    adx_minus DECIMAL(6,2),
                    rsi DECIMAL(6,2),
                    atr DECIMAL(10,6),

                    volume DECIMAL(15,2),
                    volume_ratio DECIMAL(8,4),
                    volume_confirmation BOOLEAN DEFAULT FALSE,
                    
                    nearest_support DECIMAL(10,5),
                    nearest_resistance DECIMAL(10,5),
                    distance_to_support_pips DECIMAL(8,2),
                    distance_to_resistance_pips DECIMAL(8,2),
                    risk_reward_ratio DECIMAL(8,4),
                    
                    market_session VARCHAR(20),
                    is_market_hours BOOLEAN DEFAULT TRUE,
                    market_regime VARCHAR(30),
                    
                    signal_trigger VARCHAR(50),
                    signal_conditions JSON,
                    crossover_type VARCHAR(30),

                    -- Signal validation metadata
                    trigger_type VARCHAR(20),
                    validation_details JSON,
                    swing_proximity_distance DECIMAL(8,2),
                    swing_proximity_valid BOOLEAN,

                    -- Market bias tracking
                    market_bias VARCHAR(20),
                    market_bias_conflict BOOLEAN,
                    directional_consensus DECIMAL(4,3),

                    -- ENHANCED Claude Analysis Fields
                    claude_analysis JSON,           -- Full Claude analysis response
                    claude_score INTEGER,           -- Claude score (1-10)
                    claude_decision VARCHAR(10),    -- APPROVE/REJECT
                    claude_approved BOOLEAN,        -- True/False decision
                    claude_reason TEXT,             -- Reason for decision
                    claude_mode VARCHAR(20),        -- 'minimal' or 'full'
                    claude_raw_response TEXT,       -- Raw Claude API response
                    
                    alert_message TEXT,
                    alert_level VARCHAR(10) DEFAULT 'INFO',
                    status VARCHAR(20) DEFAULT 'NEW',
                    
                    -- Deduplication fields
                    signal_hash VARCHAR(64),
                    data_source VARCHAR(20) DEFAULT 'live_scanner',
                    market_timestamp TIMESTAMP,
                    cooldown_key VARCHAR(100),
                    
                    strategy_config_hash VARCHAR(64)
                )
            ''')
            
            # Strategy performance summary table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_summary (
                    id SERIAL PRIMARY KEY,
                    date_tracked DATE NOT NULL,
                    strategy VARCHAR(50) NOT NULL,
                    total_signals INTEGER DEFAULT 0,
                    bull_signals INTEGER DEFAULT 0,
                    bear_signals INTEGER DEFAULT 0,
                    avg_confidence DECIMAL(5,4),
                    high_confidence_signals INTEGER DEFAULT 0,
                    unique_pairs INTEGER DEFAULT 0,
                    
                    -- ENHANCED Claude Analysis Statistics
                    claude_analyses_count INTEGER DEFAULT 0,
                    claude_approved_count INTEGER DEFAULT 0,
                    claude_rejected_count INTEGER DEFAULT 0,
                    claude_avg_score DECIMAL(4,2),
                    claude_success_rate DECIMAL(5,4),
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date_tracked, strategy)
                )
            ''')
            
            # Strategy configuration tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_configs (
                    id SERIAL PRIMARY KEY,
                    strategy VARCHAR(50) NOT NULL,
                    config_hash VARCHAR(64) NOT NULL,
                    config_name VARCHAR(50),
                    config_data JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(strategy, config_hash)
                )
            ''')
            
            # ENHANCED Claude Analysis Summary Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS claude_analysis_summary (
                    id SERIAL PRIMARY KEY,
                    date_tracked DATE NOT NULL,
                    total_analyses INTEGER DEFAULT 0,
                    approved_count INTEGER DEFAULT 0,
                    rejected_count INTEGER DEFAULT 0,
                    avg_score DECIMAL(4,2),
                    high_score_count INTEGER DEFAULT 0,  -- Score >= 8
                    low_score_count INTEGER DEFAULT 0,   -- Score <= 3
                    unique_pairs INTEGER DEFAULT 0,
                    avg_processing_time DECIMAL(6,3),    -- Future: API response time
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date_tracked)
                )
            ''')
            
            # Create indexes for better performance
            indexes = [
                'CREATE INDEX IF NOT EXISTS idx_alert_history_timestamp ON alert_history(alert_timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_alert_history_epic ON alert_history(epic)',
                'CREATE INDEX IF NOT EXISTS idx_alert_history_strategy ON alert_history(strategy)',
                'CREATE INDEX IF NOT EXISTS idx_alert_history_signal_type ON alert_history(signal_type)',
                'CREATE INDEX IF NOT EXISTS idx_alert_history_confidence ON alert_history(confidence_score)',
                'CREATE INDEX IF NOT EXISTS idx_alert_history_status ON alert_history(status)',
                'CREATE INDEX IF NOT EXISTS idx_alert_history_trigger ON alert_history(signal_trigger)',

                # Technical indicator indexes
                'CREATE INDEX IF NOT EXISTS idx_alert_history_adx ON alert_history(adx)',
                'CREATE INDEX IF NOT EXISTS idx_alert_history_rsi ON alert_history(rsi)',
                'CREATE INDEX IF NOT EXISTS idx_alert_history_trigger_type ON alert_history(trigger_type)',
                'CREATE INDEX IF NOT EXISTS idx_alert_history_market_bias ON alert_history(market_bias)',
                'CREATE INDEX IF NOT EXISTS idx_alert_history_bias_conflict ON alert_history(market_bias_conflict)',
                'CREATE INDEX IF NOT EXISTS idx_alert_history_swing_valid ON alert_history(swing_proximity_valid)',
                
                # ENHANCED Claude Analysis Indexes
                'CREATE INDEX IF NOT EXISTS idx_alert_history_claude_approved ON alert_history(claude_approved)',
                'CREATE INDEX IF NOT EXISTS idx_alert_history_claude_score ON alert_history(claude_score)',
                'CREATE INDEX IF NOT EXISTS idx_alert_history_claude_decision ON alert_history(claude_decision)',
                
                'CREATE INDEX IF NOT EXISTS idx_alert_history_signal_hash ON alert_history(signal_hash)',
                'CREATE INDEX IF NOT EXISTS idx_alert_history_data_source ON alert_history(data_source)',
                'CREATE INDEX IF NOT EXISTS idx_alert_history_market_timestamp ON alert_history(market_timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_strategy_summary_date ON strategy_summary(date_tracked)',
                'CREATE INDEX IF NOT EXISTS idx_strategy_summary_strategy ON strategy_summary(strategy)',
                'CREATE INDEX IF NOT EXISTS idx_strategy_configs_hash ON strategy_configs(config_hash)',
                'CREATE INDEX IF NOT EXISTS idx_claude_summary_date ON claude_analysis_summary(date_tracked)'
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            self.logger.info("✅ Alert history tables initialized successfully (with Claude analysis support)")
        
        try:
            self._execute_with_connection(create_tables_operation, "table initialization")
        except Exception as e:
            self.logger.error(f"Failed to initialize alert history tables: {e}")
            raise

    def _is_market_closed_timestamp(self, market_timestamp) -> bool:
        """
        FIXED METHOD: Check if market is actually closed based on current time
        
        The previous implementation was checking the market_timestamp field value,
        but this field can contain 1970 epoch values when data is stale.
        
        NEW LOGIC: Check actual market hours using current UTC time
        """
        try:
            from datetime import datetime, timezone
            
            # Get current UTC time for market hours check
            current_utc = datetime.now(timezone.utc)
            weekday = current_utc.weekday()  # 0=Monday, 6=Sunday
            hour = current_utc.hour
            
            # Market is closed from Friday 22:00 UTC to Sunday 22:00 UTC
            if weekday == 5:  # Saturday
                self.logger.info(f"🚫 Market closed (Saturday) - saving alerts for later")
                return True
            elif weekday == 6:  # Sunday
                if hour < 22:  # Closed until 22:00 UTC
                    self.logger.info(f"🚫 Market closed (Sunday before 22:00 UTC) - saving alerts for later")
                    return True
            elif weekday == 4:  # Friday
                if hour >= 22:  # Closed after 22:00 UTC
                    self.logger.info(f"🚫 Market closed (Friday after 22:00 UTC) - saving alerts for later")
                    return True
            
            # Market is open
            self.logger.debug(f"✅ Market open ({current_utc.strftime('%A %H:%M UTC')}) - processing alerts normally")
            return False
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking market hours: {e}")
            return False  # Default to market open if we can't determine

    def invalidate_market_closed_alerts(self) -> int:
        """
        Invalidate all existing alerts that have market_timestamp starting with 1970
        Returns the number of alerts invalidated
        """
        def invalidate_operation(conn, cursor):
            # First, find how many alerts will be affected
            cursor.execute("""
                SELECT COUNT(*) FROM alert_history 
                WHERE market_timestamp::text LIKE '1970%'
                AND status != 'INVALIDATED'
            """)
            
            count_to_invalidate = cursor.fetchone()[0]
            
            if count_to_invalidate == 0:
                self.logger.info("✅ No market closed alerts found to invalidate")
                return 0
            
            self.logger.info(f"🔍 Found {count_to_invalidate} alerts with market closed timestamps")
            
            # Update the alerts to mark them as invalidated
            cursor.execute("""
                UPDATE alert_history 
                SET status = 'INVALIDATED',
                    notes = COALESCE(notes, '') || ' [AUTO-INVALIDATED: Market closed timestamp detected]',
                    updated_at = CURRENT_TIMESTAMP
                WHERE market_timestamp::text LIKE '1970%'
                AND status != 'INVALIDATED'
            """)
            
            invalidated_count = cursor.rowcount
            self.logger.info(f"✅ Successfully invalidated {invalidated_count} market closed alerts")
            
            return invalidated_count
        
        try:
            return self._execute_with_connection(invalidate_operation, "invalidate market closed alerts")
        except Exception as e:
            self.logger.error(f"❌ Error invalidating market closed alerts: {e}")
            return 0
    
  
    def save_alert(self, signal: Dict, alert_message: str = None, alert_level: str = 'INFO', claude_result: Dict = None) -> Optional[int]:
        """
        CORRECTED METHOD: Save alert to database with proper parameter handling
        
        Args:
            signal: Signal dictionary containing all signal data
            alert_message: Optional alert message (will be generated if not provided)
            alert_level: Alert level (default: 'INFO')
            claude_result: Optional Claude analysis result dictionary
            
        Returns:
            Alert ID if successful, None if failed
        """
        try:
            # Validate input
            if not isinstance(signal, dict):
                self.logger.error(f"❌ Signal must be a dictionary, got {type(signal)}")
                return None
            
            epic = signal.get('epic', 'Unknown')
            signal_type = signal.get('signal_type', 'Unknown')
            confidence = signal.get('confidence_score', 0)
            
            self.logger.info(f"💾 Saving alert: {epic} {signal_type} ({confidence:.1%})")
            
            # Check market status (but don't reject - just log)
            market_closed = self._is_market_closed_timestamp(None)  # Use current time
            if market_closed:
                self.logger.info(f"💾 Market closed - saving signal for {epic} (queued for market open)")
                signal['execution_status'] = 'queued_for_market_open'
                signal['market_status'] = 'closed'
            else:
                signal['execution_status'] = 'ready_for_execution'
                signal['market_status'] = 'open'
            
            # Check for stale timestamp (log but don't reject)
            market_timestamp = signal.get('market_timestamp')
            if market_timestamp is not None:
                if hasattr(market_timestamp, 'strftime'):
                    timestamp_str = market_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    timestamp_str = str(market_timestamp)
                
                if timestamp_str.startswith('1970'):
                    self.logger.warning(f"⚠️ Stale data timestamp for {epic}: {timestamp_str}")
                    signal['data_quality'] = 'stale_timestamp'
            
            def save_alert_operation(conn, cursor):
                # Extract data from signal
                alert_data = self._extract_alert_data(signal, alert_message, alert_level)
                
                # Process Claude analysis result
                claude_data = self._extract_claude_analysis(signal, claude_result)
                alert_data.update(claude_data)

                # Process Market Intelligence data
                intelligence_data = self._extract_market_intelligence_data(signal)
                alert_data.update(intelligence_data)

                # Add deduplication metadata fields
                alert_data['signal_hash'] = signal.get('signal_hash')
                alert_data['data_source'] = signal.get('data_source', 'scanner')
                alert_data['market_timestamp'] = signal.get('market_timestamp')
                alert_data['cooldown_key'] = signal.get('cooldown_key')
                
                # Calculate strategy config hash for tracking
                strategy_config = signal.get('strategy_config', {})
                config_hash = hashlib.md5(json.dumps(strategy_config, sort_keys=True).encode()).hexdigest()
                alert_data['strategy_config_hash'] = config_hash
                
                # Add current timestamp for alert_timestamp
                from datetime import datetime
                alert_data['alert_timestamp'] = datetime.utcnow()
                
                # Insert alert into database
                insert_query = '''
                    INSERT INTO alert_history (
                        alert_timestamp,
                        epic, pair, signal_type, strategy, confidence_score, price, bid_price, ask_price,
                        spread_pips, timeframe, strategy_config, strategy_indicators, strategy_metadata,
                        ema_short, ema_long, ema_trend, macd_line, macd_signal, macd_histogram,
                        adx, adx_plus, adx_minus, rsi, atr,
                        volume, volume_ratio, volume_confirmation,
                        nearest_support, nearest_resistance, distance_to_support_pips, distance_to_resistance_pips, risk_reward_ratio,
                        market_session, is_market_hours, market_regime,
                        signal_trigger, signal_conditions, crossover_type,
                        trigger_type, validation_details, swing_proximity_distance, swing_proximity_valid,
                        market_bias, market_bias_conflict, directional_consensus,
                        market_structure_analysis, order_flow_analysis, confluence_details,
                        ob_proximity_score, nearest_ob_distance_pips,
                        liquidity_sweep_detected, liquidity_sweep_type, liquidity_sweep_quality,
                        claude_analysis, claude_score, claude_decision, claude_approved,
                        claude_reason, claude_mode, claude_raw_response, vision_chart_url,
                        alert_message, alert_level,
                        signal_hash, data_source, market_timestamp, cooldown_key,
                        strategy_config_hash,
                        efficiency_ratio, market_regime_detected, regime_confidence,
                        bb_width_percentile, atr_percentile, volatility_state,
                        entry_quality_score, distance_from_optimal_fib, entry_candle_momentum,
                        mtf_confluence_score, htf_candle_position, all_timeframes_aligned,
                        volume_at_swing_break, volume_trend, volume_quality_score,
                        adx_value, adx_plus_di, adx_minus_di, adx_trend_strength,
                        performance_metrics,
                        kama_value, kama_er, kama_trend, kama_signal,
                        bb_upper, bb_middle, bb_lower, bb_width, bb_percent_b, price_vs_bb,
                        stoch_k, stoch_d, stoch_zone,
                        supertrend_value, supertrend_direction,
                        rsi_zone, rsi_divergence,
                        ema_9, ema_21, ema_50, ema_200, price_vs_ema_200, ema_stack_order,
                        candle_body_pips, candle_upper_wick_pips, candle_lower_wick_pips, candle_type
                    ) VALUES (
                        %(alert_timestamp)s,
                        %(epic)s, %(pair)s, %(signal_type)s, %(strategy)s, %(confidence_score)s, %(price)s, %(bid_price)s, %(ask_price)s,
                        %(spread_pips)s, %(timeframe)s, %(strategy_config)s, %(strategy_indicators)s, %(strategy_metadata)s,
                        %(ema_short)s, %(ema_long)s, %(ema_trend)s, %(macd_line)s, %(macd_signal)s, %(macd_histogram)s,
                        %(adx)s, %(adx_plus)s, %(adx_minus)s, %(rsi)s, %(atr)s,
                        %(volume)s, %(volume_ratio)s, %(volume_confirmation)s,
                        %(nearest_support)s, %(nearest_resistance)s,
                        %(distance_to_support_pips)s, %(distance_to_resistance_pips)s, %(risk_reward_ratio)s,
                        %(market_session)s, %(is_market_hours)s, %(market_regime)s,
                        %(signal_trigger)s, %(signal_conditions)s, %(crossover_type)s,
                        %(trigger_type)s, %(validation_details)s, %(swing_proximity_distance)s, %(swing_proximity_valid)s,
                        %(market_bias)s, %(market_bias_conflict)s, %(directional_consensus)s,
                        %(market_structure_analysis)s, %(order_flow_analysis)s, %(confluence_details)s,
                        %(ob_proximity_score)s, %(nearest_ob_distance_pips)s,
                        %(liquidity_sweep_detected)s, %(liquidity_sweep_type)s, %(liquidity_sweep_quality)s,
                        %(claude_analysis)s, %(claude_score)s, %(claude_decision)s, %(claude_approved)s,
                        %(claude_reason)s, %(claude_mode)s, %(claude_raw_response)s, %(vision_chart_url)s,
                        %(alert_message)s, %(alert_level)s,
                        %(signal_hash)s, %(data_source)s, %(market_timestamp)s, %(cooldown_key)s,
                        %(strategy_config_hash)s,
                        %(efficiency_ratio)s, %(market_regime_detected)s, %(regime_confidence)s,
                        %(bb_width_percentile)s, %(atr_percentile)s, %(volatility_state)s,
                        %(entry_quality_score)s, %(distance_from_optimal_fib)s, %(entry_candle_momentum)s,
                        %(mtf_confluence_score)s, %(htf_candle_position)s, %(all_timeframes_aligned)s,
                        %(volume_at_swing_break)s, %(volume_trend)s, %(volume_quality_score)s,
                        %(adx_value)s, %(adx_plus_di)s, %(adx_minus_di)s, %(adx_trend_strength)s,
                        %(performance_metrics)s,
                        %(kama_value)s, %(kama_er)s, %(kama_trend)s, %(kama_signal)s,
                        %(bb_upper)s, %(bb_middle)s, %(bb_lower)s, %(bb_width)s, %(bb_percent_b)s, %(price_vs_bb)s,
                        %(stoch_k)s, %(stoch_d)s, %(stoch_zone)s,
                        %(supertrend_value)s, %(supertrend_direction)s,
                        %(rsi_zone)s, %(rsi_divergence)s,
                        %(ema_9)s, %(ema_21)s, %(ema_50)s, %(ema_200)s, %(price_vs_ema_200)s, %(ema_stack_order)s,
                        %(candle_body_pips)s, %(candle_upper_wick_pips)s, %(candle_lower_wick_pips)s, %(candle_type)s
                    ) RETURNING id
                '''
                
                cursor.execute(insert_query, alert_data)
                alert_id = cursor.fetchone()[0]
                
                strategy = signal.get('strategy', 'unknown')
                trigger = alert_data.get('signal_trigger', 'unknown')
                claude_status = f"Claude: {alert_data.get('claude_decision', 'N/A')}" if alert_data.get('claude_decision') else "No Claude"
                
                self.logger.info(f"✅ Saved alert #{alert_id}: {signal.get('epic')} {signal.get('signal_type')} @ {signal.get('price')} [{strategy}:{trigger}] - {claude_status}")
                
                return alert_id, config_hash, claude_data
            
            # Execute the main save operation
            result = self._execute_with_connection(save_alert_operation, "save alert")
            if not result:
                return None
                
            alert_id, config_hash, claude_data = result
            
            # Update daily summary (protected - won't fail main save)
            try:
                self._update_strategy_summary(signal, config_hash, claude_data)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to update strategy summary: {e}")
            
            # Update Claude analysis summary (protected - won't fail main save)
            try:
                if claude_data.get('claude_score') is not None:
                    self._update_claude_analysis_summary(claude_data)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to update Claude analysis summary: {e}")
            
            return alert_id

        except Exception as e:
            epic = signal.get('epic', 'Unknown') if isinstance(signal, dict) else 'Unknown'
            self.logger.error(f"❌ Error saving alert for {epic}: {e}")
            import traceback
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            return None

    def save_claude_rejection(
        self,
        signal: Dict,
        claude_result: Dict,
        rejection_reason: str
    ) -> Optional[int]:
        """
        Save a Claude-rejected signal to alert_history for analysis and auditing.

        This method is called when Claude AI rejects a trade signal, allowing
        for later analysis of rejected signals to improve strategy performance.

        Args:
            signal: Signal dictionary containing all signal data
            claude_result: Claude analysis result with score, decision, reason, etc.
            rejection_reason: Human-readable rejection reason string

        Returns:
            Alert ID if successful, None if failed
        """
        try:
            epic = signal.get('epic', 'Unknown')
            signal_type = signal.get('signal_type', signal.get('signal', 'Unknown'))

            self.logger.info(f"🚫 Saving Claude rejection: {epic} {signal_type}")

            # Mark signal as Claude rejected
            signal['status'] = 'CLAUDE_REJECTED'
            signal['claude_decision'] = claude_result.get('decision', 'REJECT')
            signal['claude_approved'] = False
            signal['claude_score'] = claude_result.get('score', 0)
            signal['claude_reason'] = rejection_reason
            signal['claude_raw_response'] = claude_result.get('raw_response', '')
            signal['claude_mode'] = claude_result.get('mode', 'vision')

            # Add metadata about the rejection
            signal['rejection_timestamp'] = datetime.now().isoformat()
            signal['rejection_type'] = 'claude_analysis'

            # Determine if it was a score-based or decision-based rejection
            score = claude_result.get('score', 0)
            decision = claude_result.get('decision', 'UNKNOWN')

            if decision == 'REJECT':
                rejection_category = 'explicit_rejection'
            elif score < 6:  # Assumes min_claude_score of 6
                rejection_category = 'low_score'
            else:
                rejection_category = 'other'

            signal['rejection_category'] = rejection_category

            # Build alert message for the rejection
            alert_message = (
                f"🚫 CLAUDE REJECTED: {epic} {signal_type}\n"
                f"Score: {score}/10 | Decision: {decision}\n"
                f"Reason: {rejection_reason}"
            )

            # Save to database with REJECTED alert level for easy filtering
            alert_id = self.save_alert(
                signal=signal,
                alert_message=alert_message,
                alert_level='REJECTED',
                claude_result=claude_result
            )

            if alert_id:
                self.logger.info(f"✅ Claude rejection saved: alert_id={alert_id}, score={score}/10")
            else:
                self.logger.warning(f"⚠️ Failed to save Claude rejection for {epic}")

            return alert_id

        except Exception as e:
            epic = signal.get('epic', 'Unknown') if isinstance(signal, dict) else 'Unknown'
            self.logger.error(f"❌ Error saving Claude rejection for {epic}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def get_claude_rejections(
        self,
        days: int = 7,
        strategy: str = None,
        min_score: int = None
    ) -> List[Dict]:
        """
        Retrieve Claude-rejected signals for analysis.

        Args:
            days: Number of days to look back (default: 7)
            strategy: Filter by strategy name (optional)
            min_score: Only return rejections with score >= min_score (optional)

        Returns:
            List of rejection records
        """
        try:
            def fetch_rejections_operation(conn, cursor):
                query = '''
                    SELECT
                        id, epic, pair, signal_type, strategy, confidence_score,
                        entry_price, stop_loss, take_profit,
                        claude_score, claude_decision, claude_reason, claude_raw_response,
                        alert_timestamp, alert_message
                    FROM alert_history
                    WHERE alert_level = 'REJECTED'
                    AND alert_timestamp >= NOW() - INTERVAL '%s days'
                '''
                params = [days]

                if strategy:
                    query += ' AND strategy = %s'
                    params.append(strategy)

                if min_score is not None:
                    query += ' AND claude_score >= %s'
                    params.append(min_score)

                query += ' ORDER BY alert_timestamp DESC'

                cursor.execute(query, params)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

                return [dict(zip(columns, row)) for row in rows]

            return self._execute_with_connection(fetch_rejections_operation, "fetch Claude rejections") or []

        except Exception as e:
            self.logger.error(f"❌ Error fetching Claude rejections: {e}")
            return []

    def get_claude_rejection_stats(self, days: int = 30) -> Dict:
        """
        Get statistics about Claude rejections for analysis.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with rejection statistics
        """
        try:
            def fetch_stats_operation(conn, cursor):
                cursor.execute('''
                    SELECT
                        COUNT(*) as total_rejections,
                        AVG(claude_score) as avg_rejection_score,
                        MIN(claude_score) as min_score,
                        MAX(claude_score) as max_score,
                        COUNT(DISTINCT epic) as unique_pairs_rejected,
                        COUNT(DISTINCT strategy) as strategies_affected
                    FROM alert_history
                    WHERE alert_level = 'REJECTED'
                    AND alert_timestamp >= NOW() - INTERVAL '%s days'
                ''', [days])

                row = cursor.fetchone()
                if row:
                    return {
                        'total_rejections': row[0] or 0,
                        'avg_rejection_score': float(row[1]) if row[1] else 0,
                        'min_score': row[2] or 0,
                        'max_score': row[3] or 0,
                        'unique_pairs_rejected': row[4] or 0,
                        'strategies_affected': row[5] or 0,
                        'period_days': days
                    }
                return {'total_rejections': 0, 'period_days': days}

            return self._execute_with_connection(fetch_stats_operation, "fetch rejection stats") or {}

        except Exception as e:
            self.logger.error(f"❌ Error fetching Claude rejection stats: {e}")
            return {'error': str(e)}

    def _update_strategy_summary(self, signal: Dict, config_hash: str, claude_data: Dict):
        """Update daily strategy summary statistics with Claude analysis data"""
        try:
            strategy = signal.get('strategy', '')
            signal_type = signal.get('signal_type', '')
            confidence = signal.get('confidence_score', 0)
            epic = signal.get('epic', '')
            
            today = datetime.now().date()
            
            # Calculate Claude statistics
            claude_analyzed = 1 if claude_data.get('claude_score') is not None else 0
            claude_approved = 1 if claude_data.get('claude_approved') else 0
            claude_rejected = 1 if claude_analyzed and not claude_data.get('claude_approved') else 0
            claude_score = claude_data.get('claude_score') or 0
            
            def update_summary_operation(conn, cursor):
                # Upsert daily summary with Claude stats
                cursor.execute('''
                    INSERT INTO strategy_summary 
                    (date_tracked, strategy, total_signals, bull_signals, bear_signals, 
                     avg_confidence, high_confidence_signals, unique_pairs,
                     claude_analyses_count, claude_approved_count, claude_rejected_count, claude_avg_score)
                    VALUES (%s, %s, 1, %s, %s, %s, %s, 1, %s, %s, %s, %s)
                    ON CONFLICT (date_tracked, strategy)
                    DO UPDATE SET
                        total_signals = strategy_summary.total_signals + 1,
                        bull_signals = strategy_summary.bull_signals + %s,
                        bear_signals = strategy_summary.bear_signals + %s,
                        avg_confidence = (strategy_summary.avg_confidence * strategy_summary.total_signals + %s) / (strategy_summary.total_signals + 1),
                        high_confidence_signals = strategy_summary.high_confidence_signals + %s,
                        claude_analyses_count = strategy_summary.claude_analyses_count + %s,
                        claude_approved_count = strategy_summary.claude_approved_count + %s,
                        claude_rejected_count = strategy_summary.claude_rejected_count + %s,
                        claude_avg_score = CASE 
                            WHEN strategy_summary.claude_analyses_count + %s > 0 
                            THEN (strategy_summary.claude_avg_score * strategy_summary.claude_analyses_count + %s) / (strategy_summary.claude_analyses_count + %s)
                            ELSE strategy_summary.claude_avg_score
                        END
                ''', (
                    today, strategy,
                    1 if signal_type == 'BULL' else 0,  # bull_signals
                    1 if signal_type == 'BEAR' else 0,  # bear_signals
                    confidence,  # avg_confidence
                    1 if confidence >= 0.8 else 0,  # high_confidence_signals
                    claude_analyzed,  # claude_analyses_count
                    claude_approved,  # claude_approved_count
                    claude_rejected,  # claude_rejected_count
                    claude_score,  # claude_avg_score
                    # UPDATE values
                    1 if signal_type == 'BULL' else 0,  # bull_signals update
                    1 if signal_type == 'BEAR' else 0,  # bear_signals update
                    confidence,  # avg_confidence update
                    1 if confidence >= 0.8 else 0,  # high_confidence_signals update
                    claude_analyzed,  # claude_analyses_count update
                    claude_approved,  # claude_approved_count update
                    claude_rejected,  # claude_rejected_count update
                    claude_analyzed,  # claude_analyses_count for CASE
                    claude_score,  # claude_avg_score numerator
                    claude_analyzed   # claude_analyses_count denominator
                ))
                
                self.logger.debug(f"✅ Updated strategy summary for {strategy} on {today}")
                return True
            
            self._execute_with_connection(update_summary_operation, "update strategy summary")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error updating strategy summary: {e}")
            # Don't raise - this is secondary functionality, main alert save should continue

    def _update_claude_analysis_summary(self, claude_data: Dict):
        """Update Claude analysis summary statistics"""
        try:
            if not claude_data.get('claude_score'):
                return
            
            today = datetime.now().date()
            claude_score = claude_data.get('claude_score', 0)
            claude_approved = claude_data.get('claude_approved', False)
            
            def update_claude_summary_operation(conn, cursor):
                # Upsert Claude daily summary
                cursor.execute('''
                    INSERT INTO claude_analysis_summary 
                    (date_tracked, total_analyses, approved_count, rejected_count, avg_score)
                    VALUES (%s, 1, %s, %s, %s)
                    ON CONFLICT (date_tracked)
                    DO UPDATE SET
                        total_analyses = claude_analysis_summary.total_analyses + 1,
                        approved_count = claude_analysis_summary.approved_count + %s,
                        rejected_count = claude_analysis_summary.rejected_count + %s,
                        avg_score = (claude_analysis_summary.avg_score * claude_analysis_summary.total_analyses + %s) / (claude_analysis_summary.total_analyses + 1)
                ''', (
                    today,
                    1 if claude_approved else 0,  # approved_count
                    1 if not claude_approved else 0,  # rejected_count
                    claude_score,  # avg_score
                    1 if claude_approved else 0,  # approved_count update
                    1 if not claude_approved else 0,  # rejected_count update
                    claude_score   # avg_score update
                ))
                
                self.logger.debug(f"✅ Updated Claude analysis summary for {today}")
                return True
            
            self._execute_with_connection(update_claude_summary_operation, "update Claude analysis summary")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error updating Claude analysis summary: {e}")
            # Don't raise - this is secondary functionality, main alert save should continue

    def _extract_market_intelligence_data(self, signal: Dict) -> Dict:
        """
        Extract market intelligence data from signal and incorporate into strategy_metadata

        Args:
            signal: Signal dictionary containing market intelligence data

        Returns:
            Dictionary with enhanced strategy_metadata including market intelligence
        """
        try:
            # Get existing strategy metadata
            existing_metadata = signal.get('strategy_metadata', {})

            # Extract market intelligence data from signal
            market_intelligence = signal.get('market_intelligence', {})

            # If no market intelligence data, serialize and return existing metadata
            if not market_intelligence:
                # Must serialize to JSON string for PostgreSQL JSON column
                if existing_metadata and isinstance(existing_metadata, dict):
                    try:
                        from forex_scanner.utils.scanner_utils import make_json_serializable
                        cleaned = make_json_serializable(existing_metadata)
                        return {'strategy_metadata': json.dumps(cleaned) if cleaned else None}
                    except Exception:
                        return {'strategy_metadata': json.dumps(str(existing_metadata)) if existing_metadata else None}
                return {'strategy_metadata': None}

            # Create enhanced metadata with market intelligence
            enhanced_metadata = existing_metadata.copy() if existing_metadata else {}

            # Add market intelligence section
            enhanced_metadata['market_intelligence'] = {
                'regime_analysis': market_intelligence.get('regime_analysis', {}),
                'session_analysis': market_intelligence.get('session_analysis', {}),
                'market_context': market_intelligence.get('market_context', {}),
                'strategy_adaptation': market_intelligence.get('strategy_adaptation', {}),
                'intelligence_applied': True,
                'intelligence_source': market_intelligence.get('intelligence_source', 'MarketIntelligenceEngine'),
                'analysis_timestamp': market_intelligence.get('analysis_timestamp')
            }

            # Extract key values for potential indexing (future Phase 2)
            regime_analysis = market_intelligence.get('regime_analysis', {})
            market_context = market_intelligence.get('market_context', {})

            # Determine volatility level from regime scores or explicit field
            volatility_level = 'medium'  # default
            if 'volatility_level' in market_intelligence:
                volatility_level = market_intelligence['volatility_level']
            elif 'regime_scores' in regime_analysis:
                regime_scores = regime_analysis['regime_scores']
                high_vol = regime_scores.get('high_volatility', 0)
                low_vol = regime_scores.get('low_volatility', 0)
                if high_vol > 0.6:
                    volatility_level = 'high'
                elif low_vol > 0.6:
                    volatility_level = 'low'

            # Get market bias
            market_bias = 'neutral'  # default
            market_strength = market_context.get('market_strength', {})
            if 'market_bias' in market_strength:
                market_bias = market_strength['market_bias']

            # Store additional fields for future indexing
            enhanced_metadata['market_intelligence']['_indexable_fields'] = {
                'regime_confidence': regime_analysis.get('confidence', 0.5),
                'volatility_level': volatility_level,
                'market_bias': market_bias,
                'dominant_regime': regime_analysis.get('dominant_regime', 'unknown'),
                'intelligence_applied': True
            }

            self.logger.debug(f"📊 Market intelligence data extracted: regime={regime_analysis.get('dominant_regime', 'unknown')}, "
                            f"confidence={regime_analysis.get('confidence', 0.5):.1%}, volatility={volatility_level}")

            # JSON serialize the enhanced metadata to match database expectations
            try:
                from forex_scanner.utils.scanner_utils import make_json_serializable
                cleaned_metadata = make_json_serializable(enhanced_metadata)
                serialized_metadata = json.dumps(cleaned_metadata) if cleaned_metadata else None
                return {'strategy_metadata': serialized_metadata}
            except Exception as e:
                self.logger.warning(f"⚠️ Error serializing market intelligence metadata: {e}")
                # Fallback to string conversion
                return {'strategy_metadata': json.dumps(str(enhanced_metadata)) if enhanced_metadata else None}

        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting market intelligence data: {e}")
            # Return existing metadata if extraction fails - properly serialized
            try:
                existing_metadata = signal.get('strategy_metadata', {})
                if existing_metadata:
                    from forex_scanner.utils.scanner_utils import make_json_serializable
                    cleaned_metadata = make_json_serializable(existing_metadata)
                    serialized_metadata = json.dumps(cleaned_metadata) if cleaned_metadata else None
                    return {'strategy_metadata': serialized_metadata}
                else:
                    return {'strategy_metadata': None}
            except Exception as fallback_error:
                self.logger.warning(f"⚠️ Error serializing fallback metadata: {fallback_error}")
                return {'strategy_metadata': None}

    def _extract_claude_analysis(self, signal: Dict, claude_result: Dict = None) -> Dict:
        """
        ENHANCED: Extract Claude analysis data from signal and/or claude_result
        
        Args:
            signal: Signal dictionary (may already contain Claude data)
            claude_result: Separate Claude analysis result dict
            
        Returns:
            Dict with Claude analysis fields for database insertion
        """
        claude_data = {
            'claude_analysis': None,
            'claude_score': None,
            'claude_decision': None,
            'claude_approved': None,
            'claude_reason': None,
            'claude_mode': None,
            'claude_raw_response': None,
            'vision_chart_url': None  # MinIO URL for vision analysis chart
        }
        
        try:
            # FIX: Ensure signal is a dictionary before trying to use .get()
            if not isinstance(signal, dict):
                self.logger.warning(f"⚠️ Signal is not a dictionary, got {type(signal)}: {signal}")
                # If signal is a string, try to parse it or create a minimal dict
                if isinstance(signal, str):
                    self.logger.warning("🔄 Converting string signal to dictionary")
                    # Create minimal signal dict with the string as epic
                    signal = {'epic': signal}
                else:
                    self.logger.error(f"❌ Cannot process non-dict signal of type {type(signal)}")
                    return claude_data
            
            # FIX: Ensure claude_result is a dictionary if provided
            if claude_result is not None and not isinstance(claude_result, dict):
                self.logger.warning(f"⚠️ Claude result is not a dictionary, got {type(claude_result)}: {claude_result}")
                # If claude_result is a string, try to parse it or ignore
                if isinstance(claude_result, str):
                    self.logger.warning("🔄 Ignoring string claude_result, expected dictionary")
                claude_result = None
            
            # Priority 1: Use dedicated claude_result parameter if provided
            if claude_result:
                self.logger.debug("🤖 Processing dedicated Claude result parameter")
                claude_data.update({
                    'claude_analysis': claude_result.get('analysis', claude_result.get('reason')),
                    'claude_score': claude_result.get('score'),
                    'claude_decision': claude_result.get('decision'),
                    'claude_approved': claude_result.get('decision') == 'APPROVE' if claude_result.get('decision') else None,
                    'claude_reason': claude_result.get('reason'),
                    'claude_mode': claude_result.get('mode', 'minimal'),
                    'claude_raw_response': claude_result.get('raw_response', str(claude_result)),
                    'vision_chart_url': claude_result.get('vision_chart_url')  # MinIO chart URL
                })
                
            # Priority 2: Check if signal already has Claude data embedded
            elif signal.get('claude_analysis') or signal.get('claude_result'):
                self.logger.debug("🤖 Processing Claude data from signal")
                
                # Handle embedded claude_result
                embedded_claude = signal.get('claude_result')
                if embedded_claude and isinstance(embedded_claude, dict):
                    claude_data.update({
                        'claude_analysis': embedded_claude.get('analysis', embedded_claude.get('reason')),
                        'claude_score': embedded_claude.get('score'),
                        'claude_decision': embedded_claude.get('decision'),
                        'claude_approved': embedded_claude.get('decision') == 'APPROVE' if embedded_claude.get('decision') else None,
                        'claude_reason': embedded_claude.get('reason'),
                        'claude_mode': embedded_claude.get('mode', 'minimal'),
                        'claude_raw_response': embedded_claude.get('raw_response'),
                        'vision_chart_url': embedded_claude.get('vision_chart_url')  # MinIO chart URL
                    })
                
                # Handle direct Claude fields in signal
                else:
                    claude_data.update({
                        'claude_analysis': signal.get('claude_analysis'),
                        'claude_score': signal.get('claude_score'),
                        'claude_decision': signal.get('claude_decision'),
                        'claude_approved': signal.get('claude_approved'),
                        'claude_reason': signal.get('claude_reason'),
                        'claude_mode': signal.get('claude_mode'),
                        'claude_raw_response': signal.get('claude_raw_response')
                    })
            
            # Clean up None values and ensure proper types
            for key, value in claude_data.items():
                if value is not None:
                    # Convert score to integer if provided
                    if key == 'claude_score' and value is not None:
                        try:
                            claude_data[key] = int(float(value))
                        except (ValueError, TypeError):
                            claude_data[key] = None
                            
                    # Convert approved to boolean if provided
                    elif key == 'claude_approved' and value is not None:
                        if isinstance(value, str):
                            claude_data[key] = value.lower() in ['true', 'approve', 'approved', '1', 'yes']
                        else:
                            claude_data[key] = bool(value)
                            
                    # Ensure text fields are strings
                    elif key in ['claude_analysis', 'claude_reason', 'claude_decision', 'claude_mode', 'claude_raw_response']:
                        if value is not None:
                            claude_data[key] = str(value)
            
            # Log Claude extraction result
            if any(v is not None for v in claude_data.values()):
                decision = claude_data.get('claude_decision', 'N/A')
                score = claude_data.get('claude_score', 'N/A')
                self.logger.debug(f"✅ Claude data extracted: Decision={decision}, Score={score}")
            else:
                self.logger.debug("ℹ️ No Claude analysis data found")
            
            return claude_data
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting Claude analysis: {e}")
            import traceback
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            return claude_data
    
    def _extract_alert_data(self, signal: Dict, alert_message: str = None, alert_level: str = 'INFO') -> Dict:
        """
        FIXED: Extract comprehensive alert data from signal dictionary with proper JSON serialization
        Uses .get() method to safely handle missing fields and properly handles numpy types
        """
        try:
            # ================================
            # BASIC SIGNAL INFO
            # ================================
            epic = str(signal.get('epic', 'Unknown'))
            
            # Extract pair from epic if not provided
            pair = signal.get('pair')
            if not pair:
                pair = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CFD.IP', '')
            pair = str(pair)
            
            signal_type = str(signal.get('signal_type', 'Unknown'))
            strategy = str(signal.get('strategy', 'unknown_strategy'))
            confidence_score = float(signal.get('confidence_score', signal.get('confidence', 0.0)))
            
            # Price data - try multiple field names
            price = float(signal.get('price', signal.get('price_mid', signal.get('execution_price', 0.0))))
            bid_price = signal.get('bid_price', signal.get('price_bid'))
            ask_price = signal.get('ask_price', signal.get('price_ask'))
            
            # ================================
            # MARKET DATA
            # ================================
            timeframe = str(signal.get('timeframe', '15m'))
            spread_pips = signal.get('spread_pips', signal.get('spread'))
            
            # ================================
            # TECHNICAL INDICATORS
            # ================================
            # EMA indicators
            ema_short = signal.get('ema_short', signal.get('ema_9'))
            ema_long = signal.get('ema_long', signal.get('ema_21'))
            ema_trend = signal.get('ema_trend', signal.get('ema_200'))
            
            # MACD indicators
            macd_line = signal.get('macd_line', signal.get('macd'))
            macd_signal = signal.get('macd_signal')
            macd_histogram = signal.get('macd_histogram', signal.get('macd_hist'))

            # ADX/RSI/ATR indicators (NEW)
            adx = signal.get('adx')
            adx_plus = signal.get('adx_plus', signal.get('di_plus', signal.get('plus_di')))
            adx_minus = signal.get('adx_minus', signal.get('di_minus', signal.get('minus_di')))
            rsi = signal.get('rsi')
            atr = signal.get('atr')

            # ================================
            # VOLUME DATA
            # ================================
            volume = signal.get('volume')
            volume_ratio = signal.get('volume_ratio')
            volume_confirmation = signal.get('volume_confirmation', False)
            
            # ================================
            # SUPPORT/RESISTANCE
            # ================================
            nearest_support = signal.get('nearest_support', signal.get('support_level'))
            nearest_resistance = signal.get('nearest_resistance', signal.get('resistance_level'))
            distance_to_support_pips = signal.get('distance_to_support_pips')
            distance_to_resistance_pips = signal.get('distance_to_resistance_pips')
            risk_reward_ratio = signal.get('risk_reward_ratio')
            
            # ================================
            # MARKET CONDITIONS
            # ================================
            market_session = signal.get('market_session', signal.get('session'))
            is_market_hours = signal.get('is_market_hours', True)
            market_regime = signal.get('market_regime')
            
            # ================================
            # SIGNAL METADATA - SAFELY HANDLE MISSING FIELDS
            # ================================
            signal_trigger = signal.get('signal_trigger', signal.get('trigger'))
            crossover_type = signal.get('crossover_type', 'unknown')  # DEFAULT VALUE!
            signal_conditions = signal.get('signal_conditions', signal.get('conditions'))
            signal_strength = signal.get('signal_strength')
            signal_quality = signal.get('signal_quality')
            technical_score = signal.get('technical_score')

            # ================================
            # VALIDATION METADATA (NEW)
            # ================================
            trigger_type = signal.get('trigger_type')  # e.g., 'macd', 'adx', 'ema_crossover'
            validation_details = signal.get('validation_details')  # JSON with full validation context
            swing_proximity_distance = signal.get('swing_proximity_distance')
            swing_proximity_valid = signal.get('swing_proximity_valid')

            # ================================
            # MARKET BIAS TRACKING (NEW)
            # ================================
            # Extract from market_intelligence if available
            market_intelligence = signal.get('market_intelligence', {})
            market_context = market_intelligence.get('market_context', {})
            market_strength = market_context.get('market_strength', {})

            market_bias = signal.get('market_bias', market_strength.get('market_bias'))
            market_bias_conflict = signal.get('market_bias_conflict')
            directional_consensus = signal.get('directional_consensus', market_strength.get('directional_consensus'))

            # ================================
            # SMART MONEY ANALYSIS FIELDS
            # ================================
            # These come from SmartMoneyReadOnlyAnalyzer via signal_processor
            market_structure_analysis = signal.get('market_structure_analysis')
            order_flow_analysis = signal.get('order_flow_analysis')
            confluence_details = signal.get('confluence_details')
            smart_money_metadata = signal.get('smart_money_metadata')

            # OB Proximity and Liquidity Sweep fields (NEW)
            ob_proximity_score = signal.get('ob_proximity_score')
            nearest_ob_distance_pips = signal.get('nearest_ob_distance_pips')
            liquidity_sweep_detected = signal.get('liquidity_sweep_detected', False)
            liquidity_sweep_type = signal.get('liquidity_sweep_type')
            liquidity_sweep_quality = signal.get('liquidity_sweep_quality')

            # ================================
            # DEDUPLICATION FIELDS
            # ================================
            signal_hash = signal.get('signal_hash')
            data_source = signal.get('data_source', 'live_scanner')
            market_timestamp = signal.get('market_timestamp')
            cooldown_key = signal.get('cooldown_key')
            
            # ================================
            # STRATEGY DATA (JSON FIELDS) - FIXED EXTRACTION
            # ================================
            strategy_config = signal.get('strategy_config', {})
            strategy_indicators = signal.get('strategy_indicators', {})
            strategy_metadata = signal.get('strategy_metadata', {})

            # FIXED: Ensure bid_price and ask_price are always provided
            if bid_price is None or ask_price is None:
                current_price = signal.get('price', 0)
                spread_pips_val = signal.get('spread_pips', 1.5)
                
                # Get pip_size for correct calculation
                pip_size = 0.0001  # Default for non-JPY pairs
                if 'JPY' in epic:
                    pip_size = 0.01
                
                if bid_price is None:
                    bid_price = current_price - (spread_pips_val * pip_size / 2)
                if ask_price is None:
                    ask_price = current_price + (spread_pips_val * pip_size / 2)

            # ================================
            # SAFE JSON SERIALIZATION FUNCTION
            # ================================
            def safe_json_serialize(obj):
                """Convert any object to JSON-safe format, handling numpy types"""
                if obj is None:
                    return None
                
                # Handle numpy types
                if hasattr(obj, 'dtype'):
                    # Numpy scalar types
                    if hasattr(obj, 'item'):
                        obj = obj.item()  # Convert numpy scalar to Python type
                    else:
                        obj = obj.tolist()  # Convert numpy array to list
                
                # Handle different object types
                if isinstance(obj, dict):
                    # Recursively clean dictionary
                    cleaned = {}
                    for key, value in obj.items():
                        cleaned[str(key)] = safe_json_serialize(value)
                    return cleaned
                elif isinstance(obj, (list, tuple)):
                    # Recursively clean list/tuple
                    return [safe_json_serialize(item) for item in obj]
                elif isinstance(obj, (bool, int, float, str)):
                    # Already JSON-safe
                    return obj
                elif hasattr(obj, '__bool__'):
                    # Convert boolean-like objects
                    return bool(obj)
                elif hasattr(obj, '__int__'):
                    # Convert int-like objects
                    return int(obj)
                elif hasattr(obj, '__float__'):
                    # Convert float-like objects
                    return float(obj)
                else:
                    # Convert everything else to string
                    return str(obj)

            # ================================
            # BUILD COMPLETE ALERT DATA DICT
            # ================================
            alert_data = {
                # Basic fields
                'epic': epic,
                'pair': pair,
                'signal_type': signal_type,
                'strategy': strategy,
                'confidence_score': confidence_score,
                'price': float(price) if price is not None else 0.0,
                'bid_price': float(bid_price) if bid_price is not None else 0.0,
                'ask_price': float(ask_price) if ask_price is not None else 0.0,
                'spread_pips': float(spread_pips) if spread_pips is not None else 0.0,
                'timeframe': timeframe,
                
                # Technical indicators
                'ema_short': ema_short,
                'ema_long': ema_long,
                'ema_trend': ema_trend,
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,

                # ADX/RSI/ATR indicators (NEW)
                'adx': adx,
                'adx_plus': adx_plus,
                'adx_minus': adx_minus,
                'rsi': rsi,
                'atr': atr,

                # Volume
                'volume': volume,
                'volume_ratio': volume_ratio,
                'volume_confirmation': volume_confirmation,
                
                # Support/Resistance
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'distance_to_support_pips': distance_to_support_pips,
                'distance_to_resistance_pips': distance_to_resistance_pips,
                'risk_reward_ratio': risk_reward_ratio,
                
                # Market conditions
                'market_session': market_session,
                'is_market_hours': is_market_hours,
                'market_regime': market_regime,
                
                # Signal metadata
                'signal_trigger': signal_trigger,
                'crossover_type': crossover_type,
                'signal_strength': signal_strength,
                'signal_quality': signal_quality,
                'technical_score': technical_score,

                # Validation metadata (NEW)
                'trigger_type': trigger_type,
                'validation_details': validation_details,
                'swing_proximity_distance': swing_proximity_distance,
                'swing_proximity_valid': swing_proximity_valid,

                # Market bias tracking (NEW)
                'market_bias': market_bias,
                'market_bias_conflict': market_bias_conflict,
                'directional_consensus': directional_consensus,

                # Smart money analysis (already JSON strings from signal_processor)
                'market_structure_analysis': market_structure_analysis,
                'order_flow_analysis': order_flow_analysis,
                'confluence_details': confluence_details,

                # OB Proximity and Liquidity Sweep (NEW)
                'ob_proximity_score': ob_proximity_score,
                'nearest_ob_distance_pips': nearest_ob_distance_pips,
                'liquidity_sweep_detected': liquidity_sweep_detected,
                'liquidity_sweep_type': liquidity_sweep_type,
                'liquidity_sweep_quality': liquidity_sweep_quality,

                # Performance Metrics (v2.11.0)
                'efficiency_ratio': signal.get('efficiency_ratio'),
                'market_regime_detected': signal.get('market_regime_detected'),
                'regime_confidence': signal.get('regime_confidence'),
                'bb_width_percentile': signal.get('bb_width_percentile'),
                'atr_percentile': signal.get('atr_percentile'),
                'volatility_state': signal.get('volatility_state'),
                'entry_quality_score': signal.get('entry_quality_score'),
                'distance_from_optimal_fib': signal.get('distance_from_optimal_fib'),
                'entry_candle_momentum': signal.get('entry_candle_momentum'),
                'mtf_confluence_score': signal.get('mtf_confluence_score'),
                'htf_candle_position': signal.get('htf_candle_position'),
                'all_timeframes_aligned': signal.get('all_timeframes_aligned'),
                'volume_at_swing_break': signal.get('volume_at_swing_break'),
                'volume_trend': signal.get('volume_trend'),
                'volume_quality_score': signal.get('volume_quality_score'),
                'adx_value': signal.get('adx_value'),
                'adx_plus_di': signal.get('adx_plus_di'),
                'adx_minus_di': signal.get('adx_minus_di'),
                'adx_trend_strength': signal.get('adx_trend_strength'),
                'performance_metrics': None,  # Will be set below if available

                # Extended Indicators (v2.12.0) - KAMA
                'kama_value': signal.get('kama_value'),
                'kama_er': signal.get('kama_er'),
                'kama_trend': signal.get('kama_trend'),
                'kama_signal': signal.get('kama_signal'),

                # Bollinger Bands
                'bb_upper': signal.get('bb_upper'),
                'bb_middle': signal.get('bb_middle'),
                'bb_lower': signal.get('bb_lower'),
                'bb_width': signal.get('bb_width'),
                'bb_percent_b': signal.get('bb_percent_b'),
                'price_vs_bb': signal.get('price_vs_bb'),

                # Stochastic
                'stoch_k': signal.get('stoch_k'),
                'stoch_d': signal.get('stoch_d'),
                'stoch_zone': signal.get('stoch_zone'),

                # Supertrend
                'supertrend_value': signal.get('supertrend_value'),
                'supertrend_direction': signal.get('supertrend_direction'),

                # RSI Extended
                'rsi_zone': signal.get('rsi_zone'),
                'rsi_divergence': signal.get('rsi_divergence'),

                # EMA Stack
                'ema_9': signal.get('ema_9'),
                'ema_21': signal.get('ema_21'),
                'ema_50': signal.get('ema_50'),
                'ema_200': signal.get('ema_200'),
                'price_vs_ema_200': signal.get('price_vs_ema_200'),
                'ema_stack_order': signal.get('ema_stack_order'),

                # Candle Analysis
                'candle_body_pips': signal.get('candle_body_pips'),
                'candle_upper_wick_pips': signal.get('candle_upper_wick_pips'),
                'candle_lower_wick_pips': signal.get('candle_lower_wick_pips'),
                'candle_type': signal.get('candle_type'),

                # Deduplication
                'signal_hash': signal_hash,
                'data_source': data_source,
                'market_timestamp': market_timestamp,
                'cooldown_key': cooldown_key,
                
                # Alert metadata
                'alert_message': alert_message or f"{signal_type} signal for {epic}",
                'alert_level': alert_level,
                'status': 'NEW'
            }
            
            # ================================
            # SAFE JSON FIELD HANDLING - FIXED!
            # ================================
            
            # Clean and serialize strategy_config
            try:
                if strategy_config:
                    cleaned_config = safe_json_serialize(strategy_config)
                    alert_data['strategy_config'] = json.dumps(cleaned_config) if cleaned_config else None
                else:
                    alert_data['strategy_config'] = None
            except Exception as e:
                self.logger.warning(f"⚠️ Error serializing strategy_config: {e}")
                alert_data['strategy_config'] = json.dumps(str(strategy_config)) if strategy_config else None
                    
            # Clean and serialize strategy_indicators
            try:
                if strategy_indicators:
                    # DEBUG: Log what we received
                    strategy_name = signal.get('strategy', 'unknown')
                    indicator_keys = list(strategy_indicators.keys()) if isinstance(strategy_indicators, dict) else []
                    self.logger.info(f"🔍 [ALERT_HISTORY] Received strategy_indicators for {strategy_name} with keys: {indicator_keys}")

                    cleaned_indicators = safe_json_serialize(strategy_indicators)
                    alert_data['strategy_indicators'] = json.dumps(cleaned_indicators) if cleaned_indicators else None

                    # DEBUG: Verify what we're about to save
                    if cleaned_indicators:
                        saved_keys = list(cleaned_indicators.keys()) if isinstance(cleaned_indicators, dict) else []
                        self.logger.info(f"   Saving to DB with keys: {saved_keys}")
                else:
                    self.logger.info(f"⚠️  [ALERT_HISTORY] No strategy_indicators in signal for {signal.get('strategy', 'unknown')}")
                    alert_data['strategy_indicators'] = None
            except Exception as e:
                self.logger.warning(f"⚠️ Error serializing strategy_indicators: {e}")
                alert_data['strategy_indicators'] = json.dumps(str(strategy_indicators)) if strategy_indicators else None
                    
            # Clean and serialize strategy_metadata
            try:
                if strategy_metadata:
                    cleaned_metadata = safe_json_serialize(strategy_metadata)
                    alert_data['strategy_metadata'] = json.dumps(cleaned_metadata) if cleaned_metadata else None
                else:
                    alert_data['strategy_metadata'] = None
            except Exception as e:
                self.logger.warning(f"⚠️ Error serializing strategy_metadata: {e}")
                alert_data['strategy_metadata'] = json.dumps(str(strategy_metadata)) if strategy_metadata else None
            
            # Clean and serialize signal_conditions - FIXED!
            try:
                if signal_conditions is not None:
                    cleaned_conditions = safe_json_serialize(signal_conditions)
                    if isinstance(cleaned_conditions, (dict, list)):
                        alert_data['signal_conditions'] = json.dumps(cleaned_conditions)
                    else:
                        alert_data['signal_conditions'] = str(cleaned_conditions) if cleaned_conditions else None
                else:
                    alert_data['signal_conditions'] = None
            except Exception as e:
                self.logger.warning(f"⚠️ Error serializing signal_conditions: {e}")
                alert_data['signal_conditions'] = str(signal_conditions) if signal_conditions else None

            # Clean and serialize validation_details (NEW)
            try:
                if validation_details is not None:
                    cleaned_validation = safe_json_serialize(validation_details)
                    if isinstance(cleaned_validation, (dict, list)):
                        alert_data['validation_details'] = json.dumps(cleaned_validation)
                    else:
                        alert_data['validation_details'] = str(cleaned_validation) if cleaned_validation else None
                else:
                    alert_data['validation_details'] = None
            except Exception as e:
                self.logger.warning(f"⚠️ Error serializing validation_details: {e}")
                alert_data['validation_details'] = str(validation_details) if validation_details else None

            # Clean and serialize performance_metrics (v2.11.0)
            try:
                perf_metrics = signal.get('performance_metrics')
                if perf_metrics is not None:
                    cleaned_metrics = safe_json_serialize(perf_metrics)
                    if isinstance(cleaned_metrics, (dict, list)):
                        alert_data['performance_metrics'] = json.dumps(cleaned_metrics)
                    else:
                        alert_data['performance_metrics'] = None
                else:
                    alert_data['performance_metrics'] = None
            except Exception as e:
                self.logger.warning(f"⚠️ Error serializing performance_metrics: {e}")
                alert_data['performance_metrics'] = None

            # ================================
            # HANDLE DATETIME CONVERSIONS
            # ================================
            if isinstance(market_timestamp, str):
                try:
                    from datetime import datetime
                    alert_data['market_timestamp'] = datetime.fromisoformat(market_timestamp.replace('Z', '+00:00'))
                except Exception as e:
                    self.logger.warning(f"⚠️ Error parsing market_timestamp: {e}")
                    alert_data['market_timestamp'] = None
            
            # ================================
            # TYPE CONVERSION AND VALIDATION - ENHANCED
            # ================================
            
            # Convert numeric fields safely with numpy handling
            numeric_fields = [
                'confidence_score', 'price', 'bid_price', 'ask_price', 'spread_pips',
                'ema_short', 'ema_long', 'ema_trend', 'macd_line', 'macd_signal', 'macd_histogram',
                'adx', 'adx_plus', 'adx_minus', 'rsi', 'atr',  # Technical indicators
                'volume', 'volume_ratio', 'nearest_support', 'nearest_resistance',
                'distance_to_support_pips', 'distance_to_resistance_pips', 'risk_reward_ratio',
                'signal_strength', 'technical_score',
                'swing_proximity_distance', 'directional_consensus',  # Validation/market bias fields
                # Performance metrics (v2.11.0)
                'efficiency_ratio', 'regime_confidence', 'bb_width_percentile', 'atr_percentile',
                'entry_quality_score', 'distance_from_optimal_fib', 'entry_candle_momentum',
                'mtf_confluence_score', 'volume_at_swing_break', 'volume_quality_score',
                'adx_value', 'adx_plus_di', 'adx_minus_di',
                # Extended indicators (v2.12.0)
                'kama_value', 'kama_er',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent_b',
                'stoch_k', 'stoch_d',
                'supertrend_value',
                'ema_9', 'ema_21', 'ema_50', 'ema_200',
                'candle_body_pips', 'candle_upper_wick_pips', 'candle_lower_wick_pips'
            ]
            
            for field in numeric_fields:
                if alert_data[field] is not None:
                    try:
                        # Handle numpy types first
                        value = alert_data[field]
                        if hasattr(value, 'item'):
                            value = value.item()  # Convert numpy scalar to Python type
                        
                        alert_data[field] = float(value)
                        
                        # Convert NaN to None
                        if str(alert_data[field]).lower() == 'nan':
                            alert_data[field] = None
                    except (ValueError, TypeError):
                        self.logger.warning(f"⚠️ Could not convert {field} to float: {alert_data[field]}")
                        alert_data[field] = None
            
            # Convert boolean fields safely with numpy handling
            boolean_fields = ['volume_confirmation', 'is_market_hours']
            for field in boolean_fields:
                if alert_data[field] is not None:
                    try:
                        value = alert_data[field]
                        # Handle numpy boolean types
                        if hasattr(value, 'item'):
                            value = value.item()  # Convert numpy scalar to Python type
                        elif hasattr(value, '__bool__'):
                            value = bool(value)
                        
                        alert_data[field] = bool(value)
                    except (ValueError, TypeError):
                        alert_data[field] = False
            
            # Clean empty string values
            for key, value in alert_data.items():
                if isinstance(value, str) and value.strip() in ['', 'None', 'null', 'undefined']:
                    alert_data[key] = None
            
            self.logger.debug(f"📊 Extracted {len([v for v in alert_data.values() if v is not None])} non-null fields from signal")
            
            return alert_data
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting alert data: {e}")
            import traceback
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            
            # Return minimal safe data as fallback
            return {
                'epic': str(signal.get('epic', 'Unknown')),
                'pair': str(signal.get('pair', 'Unknown')),
                'signal_type': str(signal.get('signal_type', 'Unknown')),
                'strategy': str(signal.get('strategy', 'unknown')),
                'confidence_score': float(signal.get('confidence_score', 0.0)),
                'price': float(signal.get('price', 0.0)),
                'bid_price': float(signal.get('price', 0.0)),  # Use price as fallback
                'ask_price': float(signal.get('price', 0.0)),  # Use price as fallback
                'timeframe': str(signal.get('timeframe', '15m')),
                'crossover_type': 'unknown',
                'signal_trigger': 'unknown',
                'data_source': 'live_scanner',
                'strategy_config': None,
                'strategy_indicators': None,
                'strategy_metadata': None,
                'signal_conditions': None,
                'alert_message': alert_message or 'Signal detected',
                'alert_level': alert_level,
                'status': 'NEW'
            }

    def get_recent_alerts(self, limit: int = 50, strategy: str = None, days: int = 7, include_claude: bool = True) -> pd.DataFrame:
        """ENHANCED: Get recent alerts with optional Claude analysis filtering"""
        def get_alerts_operation(conn, cursor):
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Build WHERE clause
            where_conditions = ["alert_timestamp > %s"]
            params = [cutoff_date]
            
            if strategy:
                where_conditions.append("strategy = %s")
                params.append(strategy)
            
            if include_claude:
                # Include Claude analysis fields in selection
                select_fields = "*, claude_score, claude_decision, claude_approved, claude_reason"
            else:
                select_fields = "*"
            
            where_clause = " AND ".join(where_conditions)
            
            query = f'''
                SELECT {select_fields} FROM alert_history 
                WHERE {where_clause}
                ORDER BY alert_timestamp DESC 
                LIMIT %s
            '''
            params.append(limit)
            
            # Use pandas to read directly from connection
            import pandas as pd
            return pd.read_sql_query(query, conn, params=params)
        
        try:
            return self._execute_with_connection(get_alerts_operation, "get recent alerts")
        except Exception as e:
            self.logger.error(f"Error getting recent alerts: {e}")
            return pd.DataFrame()

    def check_duplicate_by_hash(self, signal_hash: str) -> Optional[int]:
        """
        Check if a signal with the given hash already exists
        
        Args:
            signal_hash: MD5 hash of signal key components
            
        Returns:
            Alert ID if duplicate found, None otherwise
        """
        def check_operation(conn, cursor):
            cursor.execute("""
                SELECT id FROM alert_history 
                WHERE signal_hash = %s 
                LIMIT 1
            """, (signal_hash,))
            
            result = cursor.fetchone()
            if result:
                return result[0]
            return None
        
        try:
            return self._execute_with_connection(check_operation, "check duplicate by hash")
        except Exception as e:
            self.logger.error(f"❌ Error checking duplicate by hash: {e}")
            return None

    def save_alert_with_deduplication(self, signal: Dict, alert_message: str = None, alert_level: str = 'INFO', claude_result: Dict = None) -> Optional[int]:
        """
        Save alert with built-in deduplication
        """
        try:
            # Generate signal hash for deduplication
            epic = signal.get('epic', '')
            signal_type = signal.get('signal_type', '')
            strategy = signal.get('strategy', '')
            timestamp = signal.get('timestamp', '')
            
            # Convert timestamp to minute precision
            if timestamp:
                try:
                    import pandas as pd
                    timestamp = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M')
                except:
                    timestamp = str(timestamp)[:16]
            
            hash_input = f"{epic}_{signal_type}_{strategy}_{timestamp}"
            signal_hash = hashlib.md5(hash_input.encode()).hexdigest()
            
            # Check for duplicate
            existing_id = self.check_duplicate_by_hash(signal_hash)
            if existing_id:
                self.logger.info(f"🔄 Duplicate signal detected (hash: {signal_hash[:8]}), skipping save")
                return existing_id
            
            # Add hash to signal and save
            signal['signal_hash'] = signal_hash
            return self.save_alert(signal, alert_message, alert_level, claude_result)
            
        except Exception as e:
            self.logger.error(f"❌ Deduplication save failed: {e}")
            return None


# Backward compatibility exports
__all__ = ['AlertHistoryManager']