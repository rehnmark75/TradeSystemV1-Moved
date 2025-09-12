# core/alerts/alert_history.py
"""
Enhanced Alert History Database System
Stores all trading alerts to PostgreSQL for historical analysis
Generic system supporting EMA, MACD, and any future strategies
ENHANCED: Now properly handles Claude API results (approve/deny decisions)
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib

import config


class AlertHistoryManager:
    """Manages storing and retrieving alert history from PostgreSQL with Claude analysis support"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self._initialize_alert_tables()
    
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
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
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
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_history_timestamp ON alert_history(alert_timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_history_epic ON alert_history(epic)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_history_strategy ON alert_history(strategy)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_history_signal_type ON alert_history(signal_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_history_confidence ON alert_history(confidence_score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_history_status ON alert_history(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_history_trigger ON alert_history(signal_trigger)')
            
            # ENHANCED Claude Analysis Indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_history_claude_approved ON alert_history(claude_approved)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_history_claude_score ON alert_history(claude_score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_history_claude_decision ON alert_history(claude_decision)')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_history_signal_hash ON alert_history(signal_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_history_data_source ON alert_history(data_source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_history_market_timestamp ON alert_history(market_timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_summary_date ON strategy_summary(date_tracked)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_summary_strategy ON strategy_summary(strategy)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_configs_hash ON strategy_configs(config_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_claude_summary_date ON claude_analysis_summary(date_tracked)')
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info("✅ Alert history tables initialized successfully (with Claude analysis support)")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize alert history tables: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
    
    def save_alert(self, signal: Dict = None, alert_message: str = None, alert_level: str = 'INFO', claude_result: Dict = None, **kwargs) -> Optional[int]:
        """
        Save a trading alert to the database (ENHANCED with Claude analysis support)
        
        Args:
            signal: Signal dictionary from any strategy detection
            alert_message: Custom alert message
            alert_level: Alert importance level
            claude_result: Claude analysis result dict (from analyze_signal_minimal or analyze_signal)
            **kwargs: Backward compatibility for signal_data parameter
            
        Returns:
            Alert ID if successful, None if failed
        """
        # Handle backward compatibility - if signal_data is passed instead of signal
        if signal is None and 'signal_data' in kwargs:
            signal = kwargs['signal_data']
            self.logger.info("🔄 Using signal_data parameter for backward compatibility")
        
        if signal is None:
            self.logger.error("❌ No signal data provided to save_alert")
            return None
            
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Extract data from signal
            alert_data = self._extract_alert_data(signal, alert_message, alert_level)
            
            # ENHANCED: Process Claude analysis result
            claude_data = self._extract_claude_analysis(signal, claude_result)
            alert_data.update(claude_data)
            
            # Add deduplication metadata fields
            alert_data['signal_hash'] = signal.get('signal_hash')
            alert_data['data_source'] = signal.get('data_source', 'live_scanner')
            alert_data['market_timestamp'] = signal.get('market_timestamp')
            alert_data['cooldown_key'] = signal.get('cooldown_key')
            
            # Handle market_timestamp conversion if needed
            if isinstance(alert_data['market_timestamp'], str):
                try:
                    from datetime import datetime
                    alert_data['market_timestamp'] = datetime.fromisoformat(alert_data['market_timestamp'].replace('Z', '+00:00'))
                except:
                    alert_data['market_timestamp'] = None
            
            # Save strategy configuration if new
            config_hash = self._save_strategy_config(signal, cursor)
            alert_data['strategy_config_hash'] = config_hash
            
            # ENHANCED Insert query with Claude fields
            insert_query = '''
                INSERT INTO alert_history (
                    epic, pair, signal_type, strategy, confidence_score, price,
                    bid_price, ask_price, spread_pips, timeframe,
                    strategy_config, strategy_indicators, strategy_metadata,
                    ema_short, ema_long, ema_trend,
                    macd_line, macd_signal, macd_histogram,
                    volume, volume_ratio, volume_confirmation,
                    nearest_support, nearest_resistance, 
                    distance_to_support_pips, distance_to_resistance_pips, risk_reward_ratio,
                    market_session, is_market_hours, market_regime,
                    signal_trigger, signal_conditions, crossover_type,
                    claude_analysis, claude_score, claude_decision, claude_approved, 
                    claude_reason, claude_mode, claude_raw_response,
                    alert_message, alert_level,
                    signal_hash, data_source, market_timestamp, cooldown_key,
                    strategy_config_hash
                ) VALUES (
                    %(epic)s, %(pair)s, %(signal_type)s, %(strategy)s, %(confidence_score)s, %(price)s,
                    %(bid_price)s, %(ask_price)s, %(spread_pips)s, %(timeframe)s,
                    %(strategy_config)s, %(strategy_indicators)s, %(strategy_metadata)s,
                    %(ema_short)s, %(ema_long)s, %(ema_trend)s,
                    %(macd_line)s, %(macd_signal)s, %(macd_histogram)s,
                    %(volume)s, %(volume_ratio)s, %(volume_confirmation)s,
                    %(nearest_support)s, %(nearest_resistance)s,
                    %(distance_to_support_pips)s, %(distance_to_resistance_pips)s, %(risk_reward_ratio)s,
                    %(market_session)s, %(is_market_hours)s, %(market_regime)s,
                    %(signal_trigger)s, %(signal_conditions)s, %(crossover_type)s,
                    %(claude_analysis)s, %(claude_score)s, %(claude_decision)s, %(claude_approved)s,
                    %(claude_reason)s, %(claude_mode)s, %(claude_raw_response)s,
                    %(alert_message)s, %(alert_level)s,
                    %(signal_hash)s, %(data_source)s, %(market_timestamp)s, %(cooldown_key)s,
                    %(strategy_config_hash)s
                ) RETURNING id
            '''
            
            cursor.execute(insert_query, alert_data)
            alert_id = cursor.fetchone()[0]
            
            conn.commit()
            cursor.close()
            conn.close()
            
            strategy = signal.get('strategy', 'unknown')
            trigger = alert_data.get('signal_trigger', 'unknown')
            claude_status = f"Claude: {alert_data.get('claude_decision', 'N/A')}" if alert_data.get('claude_decision') else "No Claude"
            
            self.logger.info(f"✅ Saved alert #{alert_id}: {signal.get('epic')} {signal.get('signal_type')} @ {signal.get('price')} [{strategy}:{trigger}] - {claude_status}")
            
            # Update daily summary (now includes Claude stats)
            self._update_strategy_summary(signal, config_hash, claude_data)
            
            # Update Claude analysis summary
            if claude_data.get('claude_score') is not None:
                self._update_claude_analysis_summary(claude_data)
            
            return alert_id
            
        except Exception as e:
            self.logger.error(f"Error saving alert: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
            return None
    
    def _extract_claude_analysis(self, signal: Dict, claude_result: Dict = None) -> Dict:
        """
        ENHANCED: Extract Claude analysis data from signal and/or claude_result
        
        Args:
            signal: Signal dictionary (may already contain Claude data)
            claude_result: Separate Claude analysis result dict
            
        Returns:
            Dict with Claude analysis fields for database storage
        """
        claude_data = {
            'claude_analysis': None,
            'claude_score': None,
            'claude_decision': None,
            'claude_approved': None,
            'claude_reason': None,
            'claude_mode': None,
            'claude_raw_response': None
        }
        
        # Check if Claude result is passed separately
        if claude_result:
            self.logger.debug("Processing separate Claude result")
            claude_data.update(self._process_claude_result(claude_result))
        
        # Check if signal already contains Claude analysis (from signal processing)
        elif signal.get('claude_analysis') or any(key in signal for key in ['claude_score', 'claude_approved', 'claude_decision']):
            self.logger.debug("Processing Claude data from signal")
            
            # If signal has structured Claude data (from minimal analysis)
            if signal.get('claude_score') is not None:
                claude_data['claude_score'] = signal.get('claude_score')
                claude_data['claude_decision'] = signal.get('claude_decision')
                claude_data['claude_approved'] = signal.get('claude_approved')
                claude_data['claude_reason'] = signal.get('claude_reason')
                claude_data['claude_mode'] = signal.get('claude_mode', 'minimal')
                claude_data['claude_raw_response'] = signal.get('claude_raw_response')
                
                # Create structured analysis JSON
                claude_data['claude_analysis'] = self._safe_json_convert({
                    'score': signal.get('claude_score'),
                    'decision': signal.get('claude_decision'),
                    'approved': signal.get('claude_approved'),
                    'reason': signal.get('claude_reason'),
                    'mode': signal.get('claude_mode', 'minimal'),
                    'timestamp': datetime.now().isoformat()
                })
            
            # If signal has full Claude analysis text
            elif signal.get('claude_analysis'):
                claude_analysis_text = signal['claude_analysis']
                claude_data['claude_raw_response'] = claude_analysis_text
                claude_data['claude_mode'] = 'full'
                
                # Try to extract structured data from full analysis text
                extracted = self._parse_full_claude_analysis(claude_analysis_text)
                claude_data.update(extracted)
                
                # Store full analysis in JSON format
                claude_data['claude_analysis'] = self._safe_json_convert({
                    'full_analysis': claude_analysis_text,
                    'mode': 'full',
                    'timestamp': datetime.now().isoformat(),
                    **extracted
                })
        
        return claude_data
    
    def _process_claude_result(self, claude_result: Dict) -> Dict:
        """Process Claude result dict into database format"""
        processed = {}
        
        if claude_result.get('score') is not None:
            processed['claude_score'] = int(claude_result['score'])
        
        if claude_result.get('decision'):
            processed['claude_decision'] = str(claude_result['decision']).upper()
        
        if claude_result.get('approved') is not None:
            processed['claude_approved'] = bool(claude_result['approved'])
        
        if claude_result.get('reason'):
            processed['claude_reason'] = str(claude_result['reason'])
        
        if claude_result.get('mode'):
            processed['claude_mode'] = str(claude_result['mode'])
        
        if claude_result.get('raw_response'):
            processed['claude_raw_response'] = str(claude_result['raw_response'])
        
        # Create comprehensive analysis JSON
        processed['claude_analysis'] = self._safe_json_convert({
            **claude_result,
            'timestamp': datetime.now().isoformat()
        })
        
        return processed
    
    def _parse_full_claude_analysis(self, analysis_text: str) -> Dict:
        """
        Try to extract structured data from full Claude analysis text
        This is a fallback for when full analysis is used instead of minimal
        """
        extracted = {}
        
        if not analysis_text:
            return extracted
        
        text_upper = analysis_text.upper()
        
        # Try to extract approval/rejection
        if 'APPROVE' in text_upper and 'REJECT' not in text_upper:
            extracted['claude_decision'] = 'APPROVE'
            extracted['claude_approved'] = True
        elif 'REJECT' in text_upper:
            extracted['claude_decision'] = 'REJECT'
            extracted['claude_approved'] = False
        
        # Try to extract score (look for patterns like "8/10", "Score: 7", etc.)
        import re
        score_patterns = [
            r'(?:score|rating):\s*(\d+)(?:/10)?',
            r'(\d+)/10',
            r'score\s+(\d+)',
            r'rating\s+(\d+)'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, analysis_text, re.IGNORECASE)
            if match:
                try:
                    score = int(match.group(1))
                    if 1 <= score <= 10:
                        extracted['claude_score'] = score
                        break
                except ValueError:
                    continue
        
        return extracted
    
    def _extract_alert_data(self, signal: Dict, alert_message: str = None, alert_level: str = 'INFO') -> Dict:
        """
        FIXED: Extract comprehensive alert data from signal dictionary
        Uses .get() method to safely handle missing fields instead of direct access
        """
        try:
            # ================================
            # BASIC SIGNAL INFO
            # ================================
            epic = str(signal.get('epic', 'Unknown'))
            
            # Extract pair from epic if not provided
            pair = signal.get('pair')
            if not pair:
                pair = epic.replace('CS.D.', '').replace('.MINI.IP', '')
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
            # DEDUPLICATION FIELDS
            # ================================
            signal_hash = signal.get('signal_hash')
            data_source = signal.get('data_source', 'live_scanner')
            market_timestamp = signal.get('market_timestamp')
            cooldown_key = signal.get('cooldown_key')
            
            # ================================
            # STRATEGY DATA (JSON FIELDS)
            # ================================
            strategy_config = signal.get('strategy_config', {})
            strategy_indicators = signal.get('strategy_indicators', {})
            strategy_metadata = signal.get('strategy_metadata', {})
            
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
                'price': price,
                'bid_price': bid_price,
                'ask_price': ask_price,
                'spread_pips': spread_pips,
                'timeframe': timeframe,
                
                # Technical indicators
                'ema_short': ema_short,
                'ema_long': ema_long,
                'ema_trend': ema_trend,
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                
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
                
                # Signal metadata - ALL SAFELY HANDLED
                'signal_trigger': signal_trigger,
                'crossover_type': crossover_type,  # Now safe!
                'signal_strength': signal_strength,
                'signal_quality': signal_quality,
                'technical_score': technical_score,
                
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
            # HANDLE JSON FIELDS SAFELY
            # ================================
            import json
            
            # Convert strategy data to JSON strings with error handling
            try:
                alert_data['strategy_config'] = json.dumps(strategy_config) if strategy_config else None
            except (TypeError, ValueError) as e:
                self.logger.warning(f"⚠️ Error serializing strategy_config: {e}")
                alert_data['strategy_config'] = json.dumps(str(strategy_config)) if strategy_config else None
                
            try:
                alert_data['strategy_indicators'] = json.dumps(strategy_indicators) if strategy_indicators else None
            except (TypeError, ValueError) as e:
                self.logger.warning(f"⚠️ Error serializing strategy_indicators: {e}")
                alert_data['strategy_indicators'] = json.dumps(str(strategy_indicators)) if strategy_indicators else None
                
            try:
                alert_data['strategy_metadata'] = json.dumps(strategy_metadata) if strategy_metadata else None
            except (TypeError, ValueError) as e:
                self.logger.warning(f"⚠️ Error serializing strategy_metadata: {e}")
                alert_data['strategy_metadata'] = json.dumps(str(strategy_metadata)) if strategy_metadata else None
            
            # Convert signal_conditions to JSON if it's a dict/list
            try:
                if isinstance(signal_conditions, (dict, list)):
                    alert_data['signal_conditions'] = json.dumps(signal_conditions)
                else:
                    alert_data['signal_conditions'] = str(signal_conditions) if signal_conditions else None
            except Exception as e:
                self.logger.warning(f"⚠️ Error serializing signal_conditions: {e}")
                alert_data['signal_conditions'] = str(signal_conditions) if signal_conditions else None
            
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
            # TYPE CONVERSION AND VALIDATION
            # ================================
            # Convert numeric fields safely
            numeric_fields = [
                'confidence_score', 'price', 'bid_price', 'ask_price', 'spread_pips',
                'ema_short', 'ema_long', 'ema_trend', 'macd_line', 'macd_signal', 'macd_histogram',
                'volume', 'volume_ratio', 'nearest_support', 'nearest_resistance',
                'distance_to_support_pips', 'distance_to_resistance_pips', 'risk_reward_ratio',
                'signal_strength', 'technical_score'
            ]
            
            for field in numeric_fields:
                if alert_data[field] is not None:
                    try:
                        alert_data[field] = float(alert_data[field])
                        # Convert NaN to None
                        if str(alert_data[field]).lower() == 'nan':
                            alert_data[field] = None
                    except (ValueError, TypeError):
                        self.logger.warning(f"⚠️ Could not convert {field} to float: {alert_data[field]}")
                        alert_data[field] = None
            
            # Convert boolean fields safely
            boolean_fields = ['volume_confirmation', 'is_market_hours']
            for field in boolean_fields:
                if alert_data[field] is not None:
                    try:
                        alert_data[field] = bool(alert_data[field])
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
                'timeframe': str(signal.get('timeframe', '15m')),
                'crossover_type': 'unknown',  # Safe default
                'signal_trigger': 'unknown',   # Safe default
                'data_source': 'live_scanner', # Safe default
                'alert_message': alert_message or 'Signal detected',
                'alert_level': alert_level,
                'status': 'NEW'
            }
    
    def _determine_signal_trigger(self, signal: Dict) -> str:
        """Determine what triggered the signal based on strategy - UNCHANGED"""
        strategy = signal.get('strategy', '')
        
        if 'ema' in strategy.lower():
            if signal.get('crossover_type'):
                return f"ema_crossover_{signal.get('crossover_type')}"
            return "ema_alignment"
        elif 'macd' in strategy.lower():
            if signal.get('macd_crossover'):
                return "macd_crossover"
            elif signal.get('macd_histogram_change'):
                return "macd_histogram"
            return "macd_signal"
        elif 'rsi' in strategy.lower():
            return "rsi_level"
        else:
            return signal.get('signal_trigger', 'generic_signal')
    
    def _extract_strategy_config(self, signal: Dict) -> Dict:
        """Extract strategy configuration parameters - UNCHANGED"""
        config_data = {}
        
        # Look for common configuration fields
        config_fields = [
            'ema_fast', 'ema_slow', 'ema_trend', 'ema_periods',
            'macd_fast', 'macd_slow', 'macd_signal_period',
            'rsi_period', 'rsi_overbought', 'rsi_oversold',
            'timeframe', 'min_confidence'
        ]
        
        for field in config_fields:
            if field in signal:
                config_data[field] = signal[field]
        
        # Also check for any existing strategy_config
        if signal.get('strategy_config'):
            config_data.update(signal['strategy_config'])
        
        return config_data
    
    def _extract_strategy_indicators(self, signal: Dict) -> Dict:
        """Extract indicator values used for this signal - UNCHANGED"""
        indicators = {}
        
        # EMA indicators
        ema_fields = ['ema_9', 'ema_21', 'ema_50', 'ema_200', 'ema_short', 'ema_long', 'ema_trend']
        for field in ema_fields:
            if field in signal:
                indicators[field] = signal[field]
        
        # MACD indicators
        macd_fields = ['macd_line', 'macd_signal', 'macd_histogram', 'macd_signal_line']
        for field in macd_fields:
            if field in signal:
                indicators[field] = signal[field]
        
        # Other indicators
        other_fields = ['rsi', 'volume', 'atr', 'bollinger_upper', 'bollinger_lower']
        for field in other_fields:
            if field in signal:
                indicators[field] = signal[field]
        
        # Also check for existing strategy_indicators
        if signal.get('strategy_indicators'):
            indicators.update(signal['strategy_indicators'])
        
        return indicators
    
    def _extract_strategy_metadata(self, signal: Dict) -> Dict:
        """Extract additional strategy metadata - UNCHANGED"""
        metadata = {}
        
        # Timing information
        if signal.get('timestamp'):
            metadata['signal_timestamp'] = signal['timestamp']
        if signal.get('scan_time'):
            metadata['scan_timestamp'] = signal['scan_time']
        
        # Confidence breakdown
        if signal.get('confidence_breakdown'):
            metadata['confidence_breakdown'] = signal['confidence_breakdown']
        
        # Market conditions
        market_fields = ['market_session', 'trading_hours', 'volatility_regime']
        for field in market_fields:
            if field in signal:
                metadata[field] = signal[field]
        
        # Strategy-specific metadata
        if signal.get('strategy_metadata'):
            metadata.update(signal['strategy_metadata'])
        
        return metadata
    
    def _extract_signal_conditions(self, signal: Dict) -> Dict:
        """Extract the conditions that were met for this signal - UNCHANGED"""
        conditions = {}
        
        # Common signal conditions
        condition_fields = [
            'price_above_ema', 'ema_crossover', 'macd_crossover', 
            'volume_confirmation', 'trend_alignment', 'support_resistance_check'
        ]
        
        for field in condition_fields:
            if field in signal:
                conditions[field] = signal[field]
        
        # Also check for existing signal_conditions
        if signal.get('signal_conditions'):
            conditions.update(signal['signal_conditions'])
        
        return conditions
    
    def _save_strategy_config(self, signal: Dict, cursor) -> Optional[str]:
        """Save strategy configuration and return config hash - UNCHANGED"""
        try:
            strategy = signal.get('strategy', '')
            config_data = self._extract_strategy_config(signal)
            
            if not config_data:
                return None
            
            # Create config hash
            config_str = json.dumps(config_data, sort_keys=True)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()
            
            # Try to insert (will ignore if duplicate due to UNIQUE constraint)
            cursor.execute('''
                INSERT INTO strategy_configs (strategy, config_hash, config_data)
                VALUES (%s, %s, %s)
                ON CONFLICT (strategy, config_hash) DO NOTHING
            ''', (strategy, config_hash, json.dumps(config_data)))
            
            return config_hash
            
        except Exception as e:
            self.logger.warning(f"Error saving strategy config: {e}")
            return None
    
    def _update_strategy_summary(self, signal: Dict, config_hash: str, claude_data: Dict):
        """ENHANCED: Update daily strategy summary statistics with Claude analysis data"""
        try:
            strategy = signal.get('strategy', '')
            signal_type = signal.get('signal_type', '')
            confidence = signal.get('confidence_score', 0)
            epic = signal.get('epic', '')
            
            today = datetime.now().date()
            
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Calculate Claude statistics
            claude_analyzed = 1 if claude_data.get('claude_score') is not None else 0
            claude_approved = 1 if claude_data.get('claude_approved') else 0
            claude_rejected = 1 if claude_analyzed and not claude_data.get('claude_approved') else 0
            claude_score = claude_data.get('claude_score') or 0
            
            # ENHANCED Upsert daily summary with Claude stats
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
                        WHEN strategy_summary.claude_analyses_count = 0 THEN %s
                        ELSE (strategy_summary.claude_avg_score * strategy_summary.claude_analyses_count + %s) / (strategy_summary.claude_analyses_count + %s)
                    END,
                    claude_success_rate = CASE 
                        WHEN strategy_summary.claude_analyses_count + %s = 0 THEN 0
                        ELSE strategy_summary.claude_approved_count::DECIMAL / (strategy_summary.claude_analyses_count + %s)
                    END
            ''', (
                today, strategy,
                1 if signal_type == 'BULL' else 0,
                1 if signal_type == 'BEAR' else 0,
                confidence,
                1 if confidence >= 0.8 else 0,
                claude_analyzed, claude_approved, claude_rejected, claude_score,
                1 if signal_type == 'BULL' else 0,
                1 if signal_type == 'BEAR' else 0,
                confidence,
                1 if confidence >= 0.8 else 0,
                claude_analyzed, claude_approved, claude_rejected,
                claude_score, claude_score, claude_analyzed,
                claude_analyzed, claude_analyzed
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Error updating strategy summary: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
    
    def _update_claude_analysis_summary(self, claude_data: Dict):
        """ENHANCED: Update daily Claude analysis summary statistics"""
        try:
            today = datetime.now().date()
            
            claude_score = claude_data.get('claude_score', 0)
            claude_approved = 1 if claude_data.get('claude_approved') else 0
            claude_rejected = 1 if not claude_data.get('claude_approved') else 0
            high_score = 1 if claude_score >= 8 else 0
            low_score = 1 if claude_score <= 3 else 0
            
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO claude_analysis_summary 
                (date_tracked, total_analyses, approved_count, rejected_count, 
                 avg_score, high_score_count, low_score_count, unique_pairs)
                VALUES (%s, 1, %s, %s, %s, %s, %s, 1)
                ON CONFLICT (date_tracked) 
                DO UPDATE SET
                    total_analyses = claude_analysis_summary.total_analyses + 1,
                    approved_count = claude_analysis_summary.approved_count + %s,
                    rejected_count = claude_analysis_summary.rejected_count + %s,
                    avg_score = (claude_analysis_summary.avg_score * claude_analysis_summary.total_analyses + %s) / (claude_analysis_summary.total_analyses + 1),
                    high_score_count = claude_analysis_summary.high_score_count + %s,
                    low_score_count = claude_analysis_summary.low_score_count + %s
            ''', (
                today, claude_approved, claude_rejected, claude_score, high_score, low_score,
                claude_approved, claude_rejected, claude_score, high_score, low_score
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Error updating Claude analysis summary: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
    
    def get_recent_alerts(self, limit: int = 50, strategy: str = None, days: int = 7, include_claude: bool = True) -> pd.DataFrame:
        """ENHANCED: Get recent alerts with optional Claude analysis filtering"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            conn = self.db_manager.get_connection()
            
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
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting recent alerts: {e}")
            if 'conn' in locals():
                conn.close()
            return pd.DataFrame()
    
    def get_claude_analysis_stats(self, days: int = 30) -> Dict:
        """NEW: Get Claude analysis performance statistics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            conn = self.db_manager.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Overall Claude statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_analyses,
                    COUNT(CASE WHEN claude_approved = true THEN 1 END) as approved_count,
                    COUNT(CASE WHEN claude_approved = false THEN 1 END) as rejected_count,
                    AVG(claude_score) as avg_score,
                    COUNT(CASE WHEN claude_score >= 8 THEN 1 END) as high_score_count,
                    COUNT(CASE WHEN claude_score <= 3 THEN 1 END) as low_score_count,
                    COUNT(DISTINCT epic) as unique_pairs_analyzed,
                    MIN(claude_score) as min_score,
                    MAX(claude_score) as max_score
                FROM alert_history 
                WHERE alert_timestamp > %s AND claude_score IS NOT NULL
            ''', [cutoff_date])
            
            overall_stats = dict(cursor.fetchone() or {})
            
            # Claude analysis by strategy
            cursor.execute('''
                SELECT 
                    strategy,
                    COUNT(*) as analysis_count,
                    COUNT(CASE WHEN claude_approved = true THEN 1 END) as approved_count,
                    COUNT(CASE WHEN claude_approved = false THEN 1 END) as rejected_count,
                    AVG(claude_score) as avg_score,
                    ROUND(COUNT(CASE WHEN claude_approved = true THEN 1 END)::DECIMAL / COUNT(*) * 100, 2) as approval_rate
                FROM alert_history 
                WHERE alert_timestamp > %s AND claude_score IS NOT NULL
                GROUP BY strategy
                ORDER BY analysis_count DESC
            ''', [cutoff_date])
            
            strategy_breakdown = [dict(row) for row in cursor.fetchall()]
            
            # Claude analysis by signal type
            cursor.execute('''
                SELECT 
                    signal_type,
                    COUNT(*) as analysis_count,
                    COUNT(CASE WHEN claude_approved = true THEN 1 END) as approved_count,
                    AVG(claude_score) as avg_score,
                    ROUND(COUNT(CASE WHEN claude_approved = true THEN 1 END)::DECIMAL / COUNT(*) * 100, 2) as approval_rate
                FROM alert_history 
                WHERE alert_timestamp > %s AND claude_score IS NOT NULL
                GROUP BY signal_type
                ORDER BY analysis_count DESC
            ''', [cutoff_date])
            
            signal_type_breakdown = [dict(row) for row in cursor.fetchall()]
            
            # Daily Claude analysis trends
            cursor.execute('''
                SELECT 
                    DATE(alert_timestamp) as analysis_date,
                    COUNT(*) as daily_analyses,
                    COUNT(CASE WHEN claude_approved = true THEN 1 END) as daily_approved,
                    AVG(claude_score) as daily_avg_score
                FROM alert_history 
                WHERE alert_timestamp > %s AND claude_score IS NOT NULL
                GROUP BY DATE(alert_timestamp)
                ORDER BY analysis_date DESC
                LIMIT 30
            ''', [cutoff_date])
            
            daily_trends = [dict(row) for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            # Calculate derived metrics
            total_analyses = overall_stats.get('total_analyses', 0)
            if total_analyses > 0:
                overall_stats['approval_rate'] = round((overall_stats.get('approved_count', 0) / total_analyses) * 100, 2)
                overall_stats['rejection_rate'] = round((overall_stats.get('rejected_count', 0) / total_analyses) * 100, 2)
            else:
                overall_stats['approval_rate'] = 0
                overall_stats['rejection_rate'] = 0
            
            return {
                'period_days': days,
                'overall_stats': overall_stats,
                'strategy_breakdown': strategy_breakdown,
                'signal_type_breakdown': signal_type_breakdown,
                'daily_trends': daily_trends
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Claude analysis stats: {e}")
            if 'conn' in locals():
                cursor.close()
                conn.close()
            return {}
    
    def get_strategy_performance(self, days: int = 30, strategy: str = None, include_claude: bool = True) -> Dict:
        """ENHANCED: Get strategy performance statistics with Claude analysis correlation"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            conn = self.db_manager.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            where_clause = "WHERE alert_timestamp > %s"
            params = [cutoff_date]
            
            if strategy:
                where_clause += " AND strategy = %s"
                params.append(strategy)
            
            # Overall statistics with Claude correlation
            if include_claude:
                cursor.execute(f'''
                    SELECT 
                        COUNT(*) as total_signals,
                        COUNT(DISTINCT epic) as unique_pairs,
                        COUNT(DISTINCT strategy) as strategies_used,
                        AVG(confidence_score) as avg_confidence,
                        COUNT(CASE WHEN signal_type = 'BULL' THEN 1 END) as bull_signals,
                        COUNT(CASE WHEN signal_type = 'BEAR' THEN 1 END) as bear_signals,
                        COUNT(CASE WHEN confidence_score >= 0.8 THEN 1 END) as high_confidence_signals,
                        
                        -- Claude analysis stats
                        COUNT(CASE WHEN claude_score IS NOT NULL THEN 1 END) as claude_analyzed_count,
                        COUNT(CASE WHEN claude_approved = true THEN 1 END) as claude_approved_count,
                        AVG(claude_score) as claude_avg_score,
                        
                        -- Correlation stats
                        CORR(confidence_score, claude_score) as confidence_claude_correlation,
                        COUNT(CASE WHEN confidence_score >= 0.8 AND claude_approved = true THEN 1 END) as high_conf_claude_approved,
                        COUNT(CASE WHEN confidence_score < 0.6 AND claude_approved = false THEN 1 END) as low_conf_claude_rejected
                    FROM alert_history 
                    {where_clause}
                ''', params)
            else:
                cursor.execute(f'''
                    SELECT 
                        COUNT(*) as total_signals,
                        COUNT(DISTINCT epic) as unique_pairs,
                        COUNT(DISTINCT strategy) as strategies_used,
                        AVG(confidence_score) as avg_confidence,
                        COUNT(CASE WHEN signal_type = 'BULL' THEN 1 END) as bull_signals,
                        COUNT(CASE WHEN signal_type = 'BEAR' THEN 1 END) as bear_signals,
                        COUNT(CASE WHEN confidence_score >= 0.8 THEN 1 END) as high_confidence_signals
                    FROM alert_history 
                    {where_clause}
                ''', params)
            
            stats = dict(cursor.fetchone())
            
            # Strategy breakdown with Claude analysis
            if include_claude:
                cursor.execute(f'''
                    SELECT 
                        strategy,
                        COUNT(*) as signal_count,
                        AVG(confidence_score) as avg_confidence,
                        COUNT(CASE WHEN signal_type = 'BULL' THEN 1 END) as bull_count,
                        COUNT(CASE WHEN signal_type = 'BEAR' THEN 1 END) as bear_count,
                        COUNT(CASE WHEN claude_approved = true THEN 1 END) as claude_approved_count,
                        COUNT(CASE WHEN claude_score IS NOT NULL THEN 1 END) as claude_analyzed_count,
                        AVG(claude_score) as claude_avg_score
                    FROM alert_history 
                    {where_clause}
                    GROUP BY strategy
                    ORDER BY signal_count DESC
                ''', params)
            else:
                cursor.execute(f'''
                    SELECT 
                        strategy,
                        COUNT(*) as signal_count,
                        AVG(confidence_score) as avg_confidence,
                        COUNT(CASE WHEN signal_type = 'BULL' THEN 1 END) as bull_count,
                        COUNT(CASE WHEN signal_type = 'BEAR' THEN 1 END) as bear_count
                    FROM alert_history 
                    {where_clause}
                    GROUP BY strategy
                    ORDER BY signal_count DESC
                ''', params)
            
            strategy_breakdown = [dict(row) for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            result = {
                'period_days': days,
                'overall_stats': stats,
                'strategy_breakdown': strategy_breakdown
            }
            
            if include_claude:
                result['claude_integration'] = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting strategy performance: {e}")
            if 'conn' in locals():
                cursor.close()
                conn.close()
            return {}
    
    def get_alerts_by_claude_decision(self, decision: str = 'APPROVE', days: int = 7, limit: int = 20) -> pd.DataFrame:
        """NEW: Get alerts filtered by Claude decision (APPROVE/REJECT)"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            conn = self.db_manager.get_connection()
            
            query = '''
                SELECT 
                    alert_timestamp, epic, signal_type, strategy, confidence_score, price,
                    claude_score, claude_decision, claude_approved, claude_reason,
                    signal_trigger, alert_message
                FROM alert_history 
                WHERE alert_timestamp > %s 
                AND claude_decision = %s
                ORDER BY alert_timestamp DESC 
                LIMIT %s
            '''
            
            df = pd.read_sql_query(query, conn, params=[cutoff_date, decision.upper(), limit])
            conn.close()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting alerts by Claude decision: {e}")
            if 'conn' in locals():
                conn.close()
            return pd.DataFrame()
    
    def analyze_claude_decision_patterns(self, days: int = 30) -> Dict:
        """NEW: Analyze patterns in Claude's decision making"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            conn = self.db_manager.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Confidence vs Claude approval correlation
            cursor.execute('''
                SELECT 
                    CASE 
                        WHEN confidence_score >= 0.9 THEN '90-100%'
                        WHEN confidence_score >= 0.8 THEN '80-89%'
                        WHEN confidence_score >= 0.7 THEN '70-79%'
                        WHEN confidence_score >= 0.6 THEN '60-69%'
                        ELSE '50-59%'
                    END as confidence_range,
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN claude_approved = true THEN 1 END) as approved_count,
                    ROUND(AVG(claude_score), 2) as avg_claude_score
                FROM alert_history 
                WHERE alert_timestamp > %s AND claude_score IS NOT NULL
                GROUP BY 
                    CASE 
                        WHEN confidence_score >= 0.9 THEN '90-100%'
                        WHEN confidence_score >= 0.8 THEN '80-89%'
                        WHEN confidence_score >= 0.7 THEN '70-79%'
                        WHEN confidence_score >= 0.6 THEN '60-69%'
                        ELSE '50-59%'
                    END
                ORDER BY MIN(confidence_score) DESC
            ''', [cutoff_date])
            
            confidence_patterns = [dict(row) for row in cursor.fetchall()]
            
            # Strategy vs Claude approval patterns
            cursor.execute('''
                SELECT 
                    strategy,
                    signal_type,
                    COUNT(*) as signal_count,
                    COUNT(CASE WHEN claude_approved = true THEN 1 END) as approved_count,
                    ROUND(AVG(claude_score), 2) as avg_claude_score,
                    ROUND(COUNT(CASE WHEN claude_approved = true THEN 1 END)::DECIMAL / COUNT(*) * 100, 2) as approval_rate
                FROM alert_history 
                WHERE alert_timestamp > %s AND claude_score IS NOT NULL
                GROUP BY strategy, signal_type
                HAVING COUNT(*) >= 3
                ORDER BY signal_count DESC
            ''', [cutoff_date])
            
            strategy_signal_patterns = [dict(row) for row in cursor.fetchall()]
            
            # Common rejection reasons analysis
            cursor.execute('''
                SELECT 
                    claude_reason,
                    COUNT(*) as frequency,
                    ROUND(AVG(confidence_score), 3) as avg_confidence_of_rejected
                FROM alert_history 
                WHERE alert_timestamp > %s 
                AND claude_approved = false 
                AND claude_reason IS NOT NULL
                GROUP BY claude_reason
                ORDER BY frequency DESC
                LIMIT 10
            ''', [cutoff_date])
            
            rejection_reasons = [dict(row) for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            return {
                'period_days': days,
                'confidence_patterns': confidence_patterns,
                'strategy_signal_patterns': strategy_signal_patterns,
                'common_rejection_reasons': rejection_reasons
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing Claude decision patterns: {e}")
            if 'conn' in locals():
                cursor.close()
                conn.close()
            return {}