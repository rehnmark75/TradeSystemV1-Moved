#!/usr/bin/env python3
"""
Scanner Utilities
Helper functions for the intelligent forex scanner
FIXED: Corrected method definitions, indentation, and logic errors
"""

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
import pytz


def make_json_serializable(obj):
    """Convert objects to JSON-serializable format"""
    if obj is None:
        return None
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict() if hasattr(obj, 'to_dict') else str(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        try:
            return {key: make_json_serializable(value) for key, value in obj.__dict__.items() if not key.startswith('_')}
        except:
            return str(obj)
    else:
        return str(obj)


def clean_signal_for_json(signal):
    """Clean a signal dictionary to ensure all values are JSON-serializable"""
    if not isinstance(signal, dict):
        return make_json_serializable(signal)
    
    cleaned_signal = {}
    
    for key, value in signal.items():
        try:
            if key in ['timestamp', 'signal_time', 'entry_time', 'exit_time', 'created_at', 'updated_at']:
                # Handle timestamp fields specifically
                if isinstance(value, (pd.Timestamp, datetime, date)):
                    cleaned_signal[key] = value.isoformat()
                elif isinstance(value, str):
                    try:
                        parsed_time = pd.to_datetime(value)
                        cleaned_signal[key] = parsed_time.isoformat()
                    except:
                        cleaned_signal[key] = value
                else:
                    cleaned_signal[key] = str(value) if value is not None else None
            
            elif key in ['price', 'price_mid', 'price_bid', 'price_ask', 'execution_price', 'stop_loss', 'take_profit']:
                # Handle price fields
                if isinstance(value, (int, float, np.number)):
                    cleaned_signal[key] = float(value)
                else:
                    cleaned_signal[key] = str(value) if value is not None else None
            
            elif key in ['confidence_score', 'confluence_score', 'intelligence_score', 'enhanced_score']:
                # Handle score fields
                if isinstance(value, (int, float, np.number)):
                    cleaned_signal[key] = float(value)
                else:
                    cleaned_signal[key] = 0.0
            
            else:
                # For any other field, use the general serialization function
                cleaned_signal[key] = make_json_serializable(value)
                
        except Exception as e:
            # If individual field cleaning fails, convert to string as fallback
            cleaned_signal[key] = str(value) if value is not None else None
    
    return cleaned_signal


def safe_datetime_conversion(value, timezone_manager=None):
    """
    Safely convert any timestamp format to datetime object
    FIXED: Made this a standalone function (not a method)
    
    Args:
        value: Timestamp in various formats (int, float, str, datetime, pd.Timestamp)
        timezone_manager: Optional timezone manager for conversions
    
    Returns:
        datetime object or None if conversion fails
    """
    try:
        # Handle None
        if value is None:
            return None
        
        # Handle integers and floats (Unix timestamps)
        if isinstance(value, (int, float)):
            # Check if it's a reasonable timestamp (between 1970 and 2050)
            if 0 < value < 2524608000:  # Jan 1, 2050
                return datetime.fromtimestamp(value, tz=pytz.UTC)
            else:
                # Might be milliseconds, convert to seconds
                if value > 1000000000000:  # Likely milliseconds
                    return datetime.fromtimestamp(value / 1000, tz=pytz.UTC)
                else:
                    return None
        
        # Handle string timestamps
        if isinstance(value, str):
            # Try pandas parsing first (handles most formats)
            try:
                parsed_dt = pd.to_datetime(value)
                if isinstance(parsed_dt, pd.Timestamp):
                    dt = parsed_dt.to_pydatetime()
                    # Add UTC timezone if naive
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=pytz.UTC)
                    return dt
            except:
                pass
            
            # Try manual parsing for common formats
            common_formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%fZ'
            ]
            
            for fmt in common_formats:
                try:
                    dt = datetime.strptime(value.replace('Z', ''), fmt.replace('Z', ''))
                    return dt.replace(tzinfo=pytz.UTC)
                except ValueError:
                    continue
            
            return None
        
        # Handle pandas Timestamp
        if isinstance(value, pd.Timestamp):
            dt = value.to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=pytz.UTC)
            return dt
        
        # Handle datetime objects
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=pytz.UTC)
            return value
        
        # Unknown type
        return None
        
    except Exception as e:
        logging.getLogger(__name__).debug(f"Failed to convert timestamp: {value} ({type(value)}): {e}")
        return None


class EnhancedTimezoneManager:
    """Enhanced timezone manager for consistent timestamp handling"""
    
    def __init__(self, user_timezone: str = 'Europe/Stockholm'):
        self.user_timezone = user_timezone
        self.utc_tz = pytz.UTC
        self.local_tz = pytz.timezone(user_timezone)
        self.logger = logging.getLogger(__name__)
        
        # Log the timezone setup
        current_local = datetime.now(self.local_tz)
        tz_name = current_local.strftime('%Z')  # Gets CEST, CET, etc.
        
        self.logger.debug(f"🌍 Enhanced timezone manager initialized:")
        self.logger.debug(f"   User timezone: {user_timezone} ({tz_name})")

    def utc_to_local(self, utc_datetime):
        """Convert UTC datetime to local timezone - FIXED"""
        try:
            # Handle None input
            if utc_datetime is None:
                return datetime.now(self.local_tz)
            
            # Handle integer/float timestamps (Unix epoch)
            if isinstance(utc_datetime, (int, float)):
                utc_datetime = datetime.fromtimestamp(utc_datetime, tz=self.utc_tz)
            
            # Handle string input
            elif isinstance(utc_datetime, str):
                utc_datetime = safe_datetime_conversion(utc_datetime)
                if utc_datetime is None:
                    return datetime.now(self.local_tz)
            
            # Handle pandas Timestamp
            elif isinstance(utc_datetime, pd.Timestamp):
                utc_datetime = utc_datetime.to_pydatetime()
            
            # Handle datetime objects
            if isinstance(utc_datetime, datetime):
                # Handle timezone-naive datetime (assume UTC)
                if utc_datetime.tzinfo is None:
                    utc_datetime = self.utc_tz.localize(utc_datetime)
                elif utc_datetime.tzinfo != self.utc_tz:
                    # Convert to UTC first
                    utc_datetime = utc_datetime.astimezone(self.utc_tz)
                
                # Convert to local timezone
                local_datetime = utc_datetime.astimezone(self.local_tz)
                return local_datetime
            
            # If we get here, unknown type
            self.logger.warning(f"⚠️ Unknown datetime type for UTC conversion: {type(utc_datetime)}")
            return datetime.now(self.local_tz)
            
        except Exception as e:
            self.logger.error(f"❌ Error converting UTC to local: {e}")
            self.logger.error(f"   Input type: {type(utc_datetime)}, value: {utc_datetime}")
            # Return current local time as fallback
            return datetime.now(self.local_tz)
    
    def local_to_utc(self, local_datetime):
        """Convert local datetime to UTC - FIXED"""
        try:
            # Handle None input
            if local_datetime is None:
                return datetime.now(self.utc_tz)
            
            # Handle integer/float timestamps (assume they're already UTC)
            if isinstance(local_datetime, (int, float)):
                return datetime.fromtimestamp(local_datetime, tz=self.utc_tz)
            
            # Handle string input
            elif isinstance(local_datetime, str):
                local_datetime = safe_datetime_conversion(local_datetime)
                if local_datetime is None:
                    return datetime.now(self.utc_tz)
                # Assume it was local time if no timezone
                if local_datetime.tzinfo is None or local_datetime.tzinfo == pytz.UTC:
                    local_datetime = local_datetime.replace(tzinfo=None)
                    local_datetime = self.local_tz.localize(local_datetime)
            
            # Handle pandas Timestamp
            elif isinstance(local_datetime, pd.Timestamp):
                local_datetime = local_datetime.to_pydatetime()
            
            # Handle datetime objects
            if isinstance(local_datetime, datetime):
                # Handle timezone-naive datetime (assume local time)
                if local_datetime.tzinfo is None:
                    local_datetime = self.local_tz.localize(local_datetime)
                elif local_datetime.tzinfo != self.local_tz:
                    # Convert to local first
                    local_datetime = local_datetime.astimezone(self.local_tz)
                
                # Convert to UTC
                utc_datetime = local_datetime.astimezone(self.utc_tz)
                return utc_datetime
            
            # If we get here, unknown type
            self.logger.warning(f"⚠️ Unknown datetime type for local conversion: {type(local_datetime)}")
            return datetime.now(self.utc_tz)
            
        except Exception as e:
            self.logger.error(f"❌ Error converting local to UTC: {e}")
            self.logger.error(f"   Input type: {type(local_datetime)}, value: {local_datetime}")
            # Return current UTC time as fallback
            return datetime.now(self.utc_tz)
    
    def get_current_local_time(self):
        """Get current time in local timezone"""
        return datetime.now(self.local_tz)
    
    def get_current_utc_time(self):
        """Get current time in UTC"""
        return datetime.now(self.utc_tz)
    
    def format_for_display(self, dt, show_timezone: bool = True):
        """Format datetime for user display (always in local timezone)"""
        try:
            # Convert to local time first if it looks like UTC
            if self._is_utc_time(dt):
                local_dt = self.utc_to_local(dt)
            else:
                local_dt = dt
            
            if isinstance(local_dt, str):
                local_dt = pd.to_datetime(local_dt)
            
            if isinstance(local_dt, pd.Timestamp):
                local_dt = local_dt.to_pydatetime()
            
            # Ensure it's in local timezone
            if local_dt.tzinfo is None:
                local_dt = self.local_tz.localize(local_dt)
            elif local_dt.tzinfo != self.local_tz:
                local_dt = local_dt.astimezone(self.local_tz)
            
            if show_timezone:
                return local_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            else:
                return local_dt.strftime('%Y-%m-%d %H:%M:%S')
                
        except Exception as e:
            self.logger.error(f"❌ Error formatting datetime: {e}")
            return str(dt)
    
    def _is_utc_time(self, dt):
        """Check if datetime is in UTC timezone"""
        try:
            if isinstance(dt, str):
                # Check if string contains UTC indicators
                return '+00:00' in dt or 'UTC' in dt or dt.endswith('Z')
            
            if isinstance(dt, pd.Timestamp):
                dt = dt.to_pydatetime()
            
            if isinstance(dt, datetime):
                return dt.tzinfo == self.utc_tz or (dt.tzinfo and dt.tzinfo.utcoffset(dt) == timedelta(0))
            
            return False
        except:
            return False
    
    def normalize_signal_timestamp(self, signal: dict):
        """Normalize signal timestamps to have both UTC and local versions"""
        enhanced_signal = signal.copy()
        
        # Handle main timestamp
        if 'timestamp' in signal:
            try:
                utc_timestamp = self.to_utc(signal['timestamp'])
                local_timestamp = self.utc_to_local(utc_timestamp)
                
                enhanced_signal['timestamp_utc'] = utc_timestamp
                enhanced_signal['timestamp_local'] = local_timestamp
                enhanced_signal['timestamp_display'] = self.format_for_display(local_timestamp)
                
                # Keep original for backward compatibility
                enhanced_signal['timestamp'] = utc_timestamp
                
            except Exception as e:
                self.logger.warning(f"⚠️ Could not normalize timestamp: {e}")
                enhanced_signal['timestamp_display'] = str(signal.get('timestamp', 'Unknown time'))
        
        return enhanced_signal
    
    def to_utc(self, dt):
        """Convert any datetime format to UTC (smart conversion) - FIXED"""
        try:
            # Handle None input
            if dt is None:
                return datetime.now(self.utc_tz)
            
            # Handle integer timestamps (Unix epoch)
            if isinstance(dt, (int, float)):
                # Convert Unix timestamp to datetime
                dt = datetime.fromtimestamp(dt, tz=self.utc_tz)
                return dt
            
            # Handle string input
            if isinstance(dt, str):
                # Try to parse ISO format strings first
                try:
                    dt = pd.to_datetime(dt)
                except Exception as e:
                    self.logger.debug(f"Failed to parse datetime string '{dt}': {e}")
                    return datetime.now(self.utc_tz)
            
            # Handle pandas Timestamp
            if isinstance(dt, pd.Timestamp):
                dt = dt.to_pydatetime()
            
            # Handle datetime objects
            if isinstance(dt, datetime):
                if dt.tzinfo is None:
                    # Assume local time if no timezone info
                    dt = self.local_tz.localize(dt)
                
                # Convert to UTC
                return dt.astimezone(self.utc_tz)
            
            # If we get here, we don't know how to handle this type
            self.logger.warning(f"⚠️ Unknown datetime type: {type(dt)}, value: {dt}")
            return datetime.now(self.utc_tz)
            
        except Exception as e:
            self.logger.error(f"❌ Error converting to UTC: {e}")
            self.logger.error(f"   Input type: {type(dt)}, value: {dt}")
            return datetime.now(self.utc_tz)


class IntelligenceFilters:
    """Intelligence filtering functions for signal quality assessment"""
    
    def __init__(self, db_manager, signal_detector, logger):
        self.db_manager = db_manager
        self.signal_detector = signal_detector
        self.logger = logger
    
    def is_optimal_trading_time(self) -> bool:
        """Check if current time is optimal for trading"""
        try:
            # London/New York overlap is typically best
            london_tz = pytz.timezone('Europe/London')
            current_london = datetime.now(london_tz)
            hour = current_london.hour
            
            # Optimal hours: 8-17 London time
            return 8 <= hour <= 17
            
        except:
            return True  # Default to allowing if check fails
    
    def has_sufficient_volume(self, epic: str) -> bool:
        """Check if recent volume is sufficient"""
        try:
            # Get recent volume data
            pair_name = epic.replace('CS.D.', '').replace('.MINI.IP', '')
            df = self.signal_detector.data_fetcher.get_enhanced_data(epic, pair_name, bars=20)
            if df is None or 'ltv' not in df.columns:
                return True  # Default to allowing
            
            recent_volume = df['ltv'].tail(5).mean()
            avg_volume = df['ltv'].mean()
            
            # Require recent volume to be at least 80% of average
            return recent_volume >= (avg_volume * 0.8)
            
        except:
            return True
    
    def has_acceptable_spread(self, epic: str) -> bool:
        """Check if spread is acceptable"""
        try:
            # Get latest bid/ask data
            pair_name = epic.replace('CS.D.', '').replace('.MINI.IP', '')
            df = self.signal_detector.data_fetcher.get_enhanced_data(epic, pair_name, bars=1)
            if df is None:
                return True
            
            # Simplified spread check - in real implementation, you'd calculate actual spread
            return True  # Placeholder
            
        except:
            return True
    
    def too_many_recent_signals(self, epic: str) -> bool:
        """Check if too many signals were generated recently"""
        try:
            # Check alert_history for recent signals on this epic
            recent_cutoff = datetime.now() - timedelta(hours=1)
            
            query = """
                SELECT COUNT(*) as count 
                FROM alert_history 
                WHERE epic = %s AND alert_timestamp > %s
            """
            
            result = self.db_manager.execute_query_raw(query, (epic, recent_cutoff))
            if result and len(result) > 0:
                recent_count = result[0][0]
                return recent_count >= 3  # Max 3 signals per hour per epic
            
            return False
            
        except:
            return False
    
    def calculate_intelligence_score(self, signal: Dict) -> float:
        """Calculate comprehensive intelligence score for signal quality"""
        try:
            scores = []
            
            # Market regime score (0.0 - 1.0)
            regime_score = self.get_market_regime_score(signal['epic'])
            scores.append(regime_score * 0.3)
            
            # Volatility score (0.0 - 1.0)
            volatility_score = self.get_volatility_score(signal['epic'])
            scores.append(volatility_score * 0.2)
            
            # Volume score (0.0 - 1.0)
            volume_score = self.get_volume_score(signal['epic'])
            scores.append(volume_score * 0.2)
            
            # Time score (0.0 - 1.0)
            time_score = self.get_time_score()
            scores.append(time_score * 0.1)
            
            # Confidence alignment score (0.0 - 1.0)
            confidence_score = min(1.0, signal['confidence_score'] / 0.8)
            scores.append(confidence_score * 0.2)
            
            return sum(scores)
            
        except Exception as e:
            self.logger.error(f"Error calculating intelligence score: {e}")
            return 0.5  # Neutral score
    
    def calculate_enhanced_score(self, signal: Dict) -> float:
        """Enhanced scoring with additional factors"""
        base_score = signal.get('intelligence_score', 0.5)
        
        # Add enhanced factors
        enhancement_factors = []
        
        # Cross-timeframe confirmation
        mtf_score = self.get_multi_timeframe_score(signal['epic'])
        enhancement_factors.append(mtf_score * 0.3)
        
        # News sentiment score
        news_score = self.get_news_sentiment_score(signal['epic'])
        enhancement_factors.append(news_score * 0.2)
        
        # Correlation with other pairs
        correlation_score = self.get_correlation_score(signal['epic'])
        enhancement_factors.append(correlation_score * 0.2)
        
        # Historical performance at similar conditions
        performance_score = self.get_historical_performance_score(signal)
        enhancement_factors.append(performance_score * 0.3)
        
        enhancement = sum(enhancement_factors)
        return min(1.0, base_score + enhancement * 0.3)  # Boost by up to 30%
    
    def get_market_regime_score(self, epic: str) -> float:
        """Get market regime suitability score"""
        return 0.7
    
    def get_volatility_score(self, epic: str) -> float:
        """Get volatility suitability score"""
        return 0.8
    
    def get_volume_score(self, epic: str) -> float:
        """Get volume quality score"""
        return 0.7
    
    def get_time_score(self) -> float:
        """Get time-based score"""
        return 1.0 if self.is_optimal_trading_time() else 0.3
    
    def get_multi_timeframe_score(self, epic: str) -> float:
        """Get multi-timeframe confirmation score"""
        return 0.6
    
    def get_news_sentiment_score(self, epic: str) -> float:
        """Get news sentiment score"""
        return 0.5
    
    def get_correlation_score(self, epic: str) -> float:
        """Get correlation-based score"""
        return 0.5
    
    def get_historical_performance_score(self, signal: Dict) -> float:
        """Get historical performance score for similar conditions"""
        return 0.6


class ClaudeIntegration:
    """Claude analysis integration helpers"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def extract_claude_quality_score(self, analysis: str) -> int:
        """Extract Claude's quality score from analysis text"""
        try:
            import re
            
            # Pattern 1: "Quality Score: X/10"
            pattern1 = r'Quality Score[:\s]+(\d+)/10'
            match = re.search(pattern1, analysis, re.IGNORECASE)
            if match:
                return int(match.group(1))
            
            # Pattern 2: "Score: X/10"
            pattern2 = r'Score[:\s]+(\d+)/10'
            match = re.search(pattern2, analysis, re.IGNORECASE)
            if match:
                return int(match.group(1))
            
            # Pattern 3: Just a number after "Quality"
            pattern3 = r'Quality[:\s]+(\d+)'
            match = re.search(pattern3, analysis, re.IGNORECASE)
            if match:
                return int(match.group(1))
            
            # Default if no score found
            return 5
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not extract Claude quality score: {e}")
            return 5
    
    def extract_claude_decision(self, analysis: str) -> str:
        """Extract Claude's trade decision from analysis text"""
        try:
            analysis_upper = analysis.upper()
            
            if 'TRADE' in analysis_upper and 'NO TRADE' not in analysis_upper:
                return 'TRADE'
            elif 'NO TRADE' in analysis_upper or 'AVOID' in analysis_upper:
                return 'NO TRADE'
            elif 'WAIT' in analysis_upper or 'HOLD' in analysis_upper:
                return 'WAIT'
            else:
                return 'UNKNOWN'
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not extract Claude decision: {e}")
            return 'UNKNOWN'


class OrderExecutionHelper:
    """Helper for order execution functionality"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def initialize_order_executor(self, db_manager, config):
        """Initialize order executor with fallback methods"""
        if not getattr(config, 'ENABLE_ORDER_EXECUTION', False):
            self.logger.info("💡 Order execution disabled in configuration")
            return None
        
        try:
            from alerts.order_executor import OrderExecutor
            
            # Try different initialization patterns
            try:
                # Method 1: Try with db_manager parameter
                order_executor = OrderExecutor(db_manager)
                self.logger.info("💰 Order executor initialized with db_manager")
                return order_executor
            except TypeError:
                try:
                    # Method 2: Try without parameters
                    order_executor = OrderExecutor()
                    self.logger.info("💰 Order executor initialized without parameters")
                    
                    # Try to set db_manager as attribute if the class supports it
                    if hasattr(order_executor, 'db_manager'):
                        order_executor.db_manager = db_manager
                        self.logger.info("   → db_manager set as attribute")
                    elif hasattr(order_executor, 'set_db_manager'):
                        order_executor.set_db_manager(db_manager)
                        self.logger.info("   → db_manager set via method")
                    
                    return order_executor
                        
                except Exception as e2:
                    self.logger.warning(f"⚠️ Could not initialize OrderExecutor: {e2}")
                    return None
                    
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import OrderExecutor: {e}")
            self.logger.info("💡 Order execution will be disabled")
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ OrderExecutor initialization failed: {e}")
            return None
    
    def execute_order(self, order_executor, signal: Dict, config):
        """Execute trading order using the order executor"""
        try:
            if not order_executor:
                self.logger.debug("💡 Order executor not available - auto-trading disabled")
                return
            
            # Check if order executor has enabled flag
            if hasattr(order_executor, 'enabled') and not order_executor.enabled:
                self.logger.info("💡 Auto-trading disabled in order executor configuration")
                return
            
            # Check confidence threshold for orders
            min_confidence_for_orders = getattr(config, 'MIN_CONFIDENCE_FOR_ORDERS', 0.75)
            signal_confidence = signal.get('confidence_score', 0)
            
            if signal_confidence < min_confidence_for_orders:
                self.logger.info(f"🚫 Signal confidence {signal_confidence:.1%} below order threshold {min_confidence_for_orders:.1%}")
                return
            
            self.logger.info(f"💰 Executing order for {signal['epic']} {signal['signal_type']}")
            
            # Execute the order - try different method names based on OrderExecutor implementation
            result = None
            
            # Try common method names
            if hasattr(order_executor, 'execute_signal_order'):
                result = order_executor.execute_signal_order(signal)
            elif hasattr(order_executor, 'execute_order'):
                result = order_executor.execute_order(signal)
            elif hasattr(order_executor, 'place_order'):
                result = order_executor.place_order(signal)
            elif hasattr(order_executor, 'process_signal'):
                result = order_executor.process_signal(signal)
            else:
                self.logger.warning("⚠️ OrderExecutor doesn't have a recognized order execution method")
                # Log available methods for debugging
                methods = [method for method in dir(order_executor) if not method.startswith('_')]
                self.logger.debug(f"Available OrderExecutor methods: {methods}")
                return
            
            if result:
                if result.get('status') == 'success':
                    self.logger.info(f"✅ Order executed successfully: {result.get('order_id', 'N/A')}")
                    # Update signal with execution info
                    signal['order_executed'] = True
                    signal['order_id'] = result.get('order_id')
                elif result.get('status') == 'paper_trade':
                    self.logger.info(f"📋 Paper trade logged: {result.get('message', 'N/A')}")
                    signal['paper_trade'] = True
                else:
                    self.logger.error(f"❌ Order execution failed: {result.get('message', 'Unknown error')}")
            else:
                self.logger.error("❌ No response from order executor")
                
        except Exception as e:
            self.logger.error(f"❌ Order execution error: {e}")
            import traceback
            self.logger.debug(f"Order execution traceback: {traceback.format_exc()}")


def is_new_signal(last_signals: Dict, epic: str, signal_time, logger) -> bool:
    """Enhanced signal deduplication check"""
    if epic not in last_signals:
        return True
    
    last_time = last_signals[epic]
    
    # Handle different timestamp formats
    try:
        if isinstance(signal_time, str):
            signal_time = pd.to_datetime(signal_time)
        if isinstance(last_time, str):
            last_time = pd.to_datetime(last_time)
        
        # Check if signals are from the same time period (within 1 minute)
        if hasattr(signal_time, 'timestamp') and hasattr(last_time, 'timestamp'):
            time_diff = abs(signal_time.timestamp() - last_time.timestamp())
            return time_diff > 60  # More than 1 minute apart
        else:
            return signal_time != last_time
            
    except Exception as e:
        logger.warning(f"Error comparing signal times for {epic}: {e}")
        return True  # Default to accepting the signal if comparison fails


def save_signal_with_logging(alert_history, signal: Dict, logger) -> Optional[int]:
    """Save signal to alert_history table with comprehensive logging"""
    try:
        # Clean the signal FIRST to ensure JSON compatibility
        cleaned_signal = clean_signal_for_json(signal)
        
        epic = cleaned_signal.get('epic', 'Unknown')
        signal_type = cleaned_signal.get('signal_type', 'Unknown')
        confidence = cleaned_signal.get('confidence_score', 0)
        strategy = cleaned_signal.get('strategy', 'Unknown')
        
        alert_message = f"Scanner signal: {signal_type} {epic} @ {confidence:.1%} confidence"
        
        # FIXED: Save to database with comprehensive logging
        logger.info(f"💾 Saving signal to alert_history database...")
        logger.info(f"   Epic: {epic}")
        logger.info(f"   Signal: {signal_type}")
        logger.info(f"   Strategy: {strategy}")
        logger.info(f"   Confidence: {confidence:.1%}")
        
        # Try primary method first with cleaned signal
        alert_id = alert_history.save_alert(
            cleaned_signal,  # Use cleaned signal instead of original
            alert_message,
            alert_level='INFO'
        )
        
        if alert_id:
            logger.info(f"✅ Signal successfully saved to database!")
            logger.info(f"   Alert ID: {alert_id}")
            logger.info(f"   Database table: alert_history")
            logger.info(f"   Message: {alert_message}")
            return alert_id
        else:
            logger.warning(f"⚠️ Failed to save signal to alert_history database")
            logger.warning(f"   Signal: {signal_type} {epic}")
            logger.warning(f"   Check database connection and permissions")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error saving signal to database: {e}")
        logger.error(f"   Signal: {signal.get('signal_type', 'Unknown')} {signal.get('epic', 'Unknown')}")
        # Log what types are causing issues for debugging
        problematic_fields = []
        for k, v in signal.items():
            try:
                json.dumps(make_json_serializable(v))
            except (TypeError, ValueError):
                problematic_fields.append((k, type(v).__name__))
        if problematic_fields:
            logger.debug(f"   Non-serializable fields: {problematic_fields}")
        return None


def update_alert_with_claude(db_manager, alert_id: int, claude_analysis: str, logger):
    """Update alert record with Claude analysis"""
    try:
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE alert_history 
            SET claude_analysis = %s, 
                status = 'ANALYZED',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (claude_analysis, alert_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.debug(f"✅ Alert {alert_id} updated with Claude analysis in database")
        
    except Exception as e:
        logger.error(f"❌ Error updating alert with Claude: {e}")
        if 'conn' in locals():
            try:
                conn.rollback()
                cursor.close()
                conn.close()
            except:
                pass


def evaluate_trade_approval(signal: Dict, claude_quality_score: Optional[int], 
                          claude_decision: Optional[str], min_confidence: float, config) -> bool:
    """Enhanced trade approval logic"""
    confidence = signal.get('confidence_score', 0)
    
    # Base approval on confidence
    if confidence < min_confidence:
        return False
    
    # Claude quality check
    min_claude_quality = getattr(config, 'MIN_CLAUDE_QUALITY_SCORE', 5)
    if claude_quality_score is not None and claude_quality_score < min_claude_quality:
        return False
    
    # Claude decision check
    if claude_decision and claude_decision in ['NO TRADE', 'AVOID']:
        return False
    
    return True


def send_enhanced_notification(notification_manager, signal: Dict, claude_quality_score: Optional[int], 
                             claude_decision: Optional[str], trade_approved: bool, logger):
    """Send enhanced notification with proper timestamp formatting"""
    try:
        epic = signal['epic']
        signal_type = signal['signal_type']
        confidence = signal.get('confidence_score', 0)
        timestamp_display = signal.get('timestamp_display', 'Unknown time')
        
        # Create enhanced notification message
        claude_info = f"🤖 CLAUDE: {claude_quality_score or 'N/A'}/10"
        
        notification_title = f"🚨 TRADING SIGNAL - {claude_info} 🚨"
        
        notification_body = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epic: {epic}
Signal: {signal_type}
Confidence: {confidence:.1f}%
Time: {timestamp_display}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 {signal_type} Signal: {epic}
Strategy: {signal.get('strategy', 'Unknown')}
Confidence: {confidence:.1f}%"""
        
        if 'price_mid' in signal:
            notification_body += f"\nMID: {signal['price_mid']:.5f}, EXEC: {signal.get('execution_price', 'N/A')}"
        
        if claude_quality_score and claude_decision:
            notification_body += f"\n🤖 Claude: {claude_quality_score}/10, {claude_decision}"
        
        notification_body += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Send notification using existing notification manager
        notification_manager.send_signal_alert(signal)
        
        logger.info(f"📢 Notification sent for {epic}")
        
    except Exception as e:
        logger.error(f"❌ Failed to send notification: {e}")


def create_intelligent_scanner(intelligence_mode: str = 'backtest_consistent', **kwargs):
    """
    Factory function to create intelligent scanner with specified mode
    
    Args:
        intelligence_mode: 
            - 'disabled': No intelligence (matches current backtester)
            - 'backtest_consistent': Apply same filters in backtest and live
            - 'live_only': Full intelligence for live only
            - 'enhanced': Advanced intelligence with ML
    """
    try:
        from core.database import DatabaseManager
        import config
        
        db_manager = DatabaseManager(config.DATABASE_URL)
        
        # Import the main scanner class
        from .intelligent_scanner import IntelligentForexScanner
        
        return IntelligentForexScanner(
            db_manager=db_manager,
            epic_list=getattr(config, 'EPIC_LIST', ['CS.D.EURUSD.MINI.IP']),
            intelligence_mode=intelligence_mode,
            user_timezone=getattr(config, 'USER_TIMEZONE', 'Europe/Stockholm'),
            **kwargs
        )
        
    except Exception as e:
        import logging
        logging.error(f"❌ Failed to create intelligent scanner: {e}")
        raise