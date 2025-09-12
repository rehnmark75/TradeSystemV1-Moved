# utils/helpers.py
"""
Utility functions for the Forex Scanner
"""

import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import pytz
try:
    import config
except ImportError:
    from forex_scanner import config


import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import pytz
try:
    import config
except ImportError:
    from forex_scanner import config


class TimezoneAwareFormatter(logging.Formatter):
    """Custom formatter that shows timestamps in user's timezone"""
    
    def __init__(self, fmt=None, datefmt=None, timezone='Europe/Stockholm'):
        super().__init__(fmt, datefmt)
        self.timezone = pytz.timezone(timezone)
    
    def formatTime(self, record, datefmt=None):
        """Format time in user's timezone"""
        # Convert UTC timestamp to user timezone
        utc_time = datetime.fromtimestamp(record.created, tz=pytz.UTC)
        local_time = utc_time.astimezone(self.timezone)
        
        if datefmt:
            return local_time.strftime(datefmt)
        else:
            return local_time.strftime('%Y-%m-%d %H:%M:%S')


def setup_logging(
    level: int = logging.INFO, 
    log_file: Optional[str] = None,
    timezone: str = 'Europe/Stockholm'
) -> logging.Logger:
    """
    Set up logging configuration with timezone awareness
    
    Args:
        level: Logging level
        log_file: Optional log file path
        timezone: Timezone for log timestamps
        
    Returns:
        Configured logger
    """
    # Create timezone-aware formatter
    formatter = TimezoneAwareFormatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        timezone=timezone
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log the timezone being used
    root_logger.info(f"🕒 Logging timezone set to: {timezone}")
    
    return root_logger


def extract_pair_from_epic(epic: str) -> str:
    """
    Extract currency pair from IG epic format
    
    Args:
        epic: IG epic code (e.g., 'CS.D.EURUSD.MINI.IP')
        
    Returns:
        Currency pair (e.g., 'EURUSD')
    """
    pair_info = config.PAIR_INFO.get(epic)
    if pair_info:
        return pair_info['pair']
    
    # Fallback: try to extract from epic format
    parts = epic.split('.')
    if len(parts) >= 3:
        return parts[2]
    
    return 'EURUSD'  # Default fallback


def get_pip_multiplier(pair: str) -> int:
    """
    Get pip multiplier for currency pair
    
    Args:
        pair: Currency pair (e.g., 'EURUSD', 'USDJPY')
        
    Returns:
        Pip multiplier (10000 for most pairs, 100 for JPY pairs)
    """
    if 'JPY' in pair.upper():
        return 100  # JPY pairs: 1 pip = 0.01
    else:
        return 10000  # Most pairs: 1 pip = 0.0001


def format_price(price: float, pair: str) -> str:
    """
    Format price according to pair convention
    
    Args:
        price: Price value
        pair: Currency pair
        
    Returns:
        Formatted price string
    """
    if 'JPY' in pair.upper():
        return f"{price:.3f}"  # 3 decimal places for JPY pairs
    else:
        return f"{price:.5f}"  # 5 decimal places for other pairs


def calculate_pips_difference(price1: float, price2: float, pair: str) -> float:
    """
    Calculate difference between two prices in pips
    
    Args:
        price1: First price
        price2: Second price
        pair: Currency pair
        
    Returns:
        Difference in pips
    """
    pip_multiplier = get_pip_multiplier(pair)
    return abs(price1 - price2) * pip_multiplier


def format_signal_summary(signal: Dict[str, Any]) -> str:
    """
    Format signal for display/logging
    
    Args:
        signal: Signal dictionary
        
    Returns:
        Formatted signal string
    """
    epic = signal.get('epic', 'Unknown')
    signal_type = signal.get('signal_type', 'Unknown')
    confidence = signal.get('confidence_score', 0)
    timestamp = signal.get('timestamp', 'Unknown')
    
    # Handle different price formats
    if 'price_mid' in signal:
        price_info = f"MID: {signal['price_mid']:.5f}, EXEC: {signal['execution_price']:.5f}"
    else:
        price_info = f"Price: {signal.get('price', 0):.5f}"
    
    return f"{timestamp} | {epic} | {signal_type} | {price_info} | Conf: {confidence:.1%}"


def validate_configuration() -> Dict[str, bool]:
    """
    Validate configuration settings
    
    Returns:
        Dictionary of validation results
    """
    validation = {
        'database_url': bool(config.DATABASE_URL),
        'epic_list': bool(config.EPIC_LIST and len(config.EPIC_LIST) > 0),
        'claude_api_key': bool(config.CLAUDE_API_KEY),
        'scan_interval': config.SCAN_INTERVAL > 0,
        'spread_pips': config.SPREAD_PIPS > 0,
        'min_confidence': 0 <= config.MIN_CONFIDENCE <= 1,
        'ema_periods': len(config.EMA_PERIODS) >= 3
    }
    
    return validation


def get_market_session() -> str:
    """
    Determine current market session based on time
    
    Returns:
        Market session name
    """
    now = datetime.now()
    hour = now.hour
    
    # Simplified session detection (UTC time)
    if 0 <= hour < 8:
        return 'Asian'
    elif 8 <= hour < 16:
        return 'European'
    elif 16 <= hour < 24:
        return 'American'
    else:
        return 'Unknown'


def is_trading_hours() -> bool:
    """
    Check if current time is within trading hours
    
    Returns:
        True if within trading hours
    """
    now = datetime.now()
    hour = now.hour
    
    # Basic trading hours (can be enhanced with timezone handling)
    return config.MARKET_OPEN_HOUR <= hour <= config.MARKET_CLOSE_HOUR


def calculate_position_size(
    account_balance: float, 
    risk_percent: float, 
    stop_loss_pips: float, 
    pair: str
) -> float:
    """
    Calculate position size based on risk management
    
    Args:
        account_balance: Account balance
        risk_percent: Risk percentage (e.g., 0.02 for 2%)
        stop_loss_pips: Stop loss in pips
        pair: Currency pair
        
    Returns:
        Position size in lots
    """
    risk_amount = account_balance * risk_percent
    pip_value = 1.0  # Simplified - would need actual pip value calculation
    
    if stop_loss_pips > 0:
        position_size = risk_amount / (stop_loss_pips * pip_value)
        return round(position_size, 2)
    
    return 0.01  # Minimum position size


def create_signal_id(signal: Dict[str, Any]) -> str:
    """
    Create unique identifier for signal
    
    Args:
        signal: Signal dictionary
        
    Returns:
        Unique signal ID
    """
    epic = signal.get('epic', 'unknown')
    timestamp = signal.get('timestamp', datetime.now())
    signal_type = signal.get('signal_type', 'unknown')
    
    # Convert timestamp to string if it's a datetime object
    if isinstance(timestamp, datetime):
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
    else:
        timestamp_str = str(timestamp).replace(' ', '_').replace(':', '')
    
    return f"{epic}_{signal_type}_{timestamp_str}"


def filter_signals_by_quality(signals: list, min_confidence: float = 0.6) -> list:
    """
    Filter signals by quality criteria
    
    Args:
        signals: List of signal dictionaries
        min_confidence: Minimum confidence threshold
        
    Returns:
        Filtered list of signals
    """
    quality_signals = []
    
    for signal in signals:
        confidence = signal.get('confidence_score', 0)
        
        # Basic quality filters
        if confidence >= min_confidence:
            # Additional quality checks can be added here
            volume_ratio = signal.get('volume_ratio_20', 1.0)
            
            # Prefer signals with higher volume
            if volume_ratio >= 1.2:  # 20% above average volume
                quality_signals.append(signal)
            elif confidence >= 0.8:  # High confidence signals regardless of volume
                quality_signals.append(signal)
    
    return quality_signals


def get_signal_statistics(signals: list) -> Dict[str, Any]:
    """
    Calculate statistics for a list of signals
    
    Args:
        signals: List of signal dictionaries
        
    Returns:
        Statistics dictionary
    """
    if not signals:
        return {'total': 0}
    
    total_signals = len(signals)
    bull_signals = [s for s in signals if s.get('signal_type') == 'BULL']
    bear_signals = [s for s in signals if s.get('signal_type') == 'BEAR']
    
    confidences = [s.get('confidence_score', 0) for s in signals]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Epic distribution
    epic_counts = {}
    for signal in signals:
        epic = signal.get('epic', 'unknown')
        epic_counts[epic] = epic_counts.get(epic, 0) + 1
    
    return {
        'total': total_signals,
        'bull_signals': len(bull_signals),
        'bear_signals': len(bear_signals),
        'bull_percentage': len(bull_signals) / total_signals * 100 if total_signals > 0 else 0,
        'bear_percentage': len(bear_signals) / total_signals * 100 if total_signals > 0 else 0,
        'average_confidence': avg_confidence,
        'max_confidence': max(confidences) if confidences else 0,
        'min_confidence': min(confidences) if confidences else 0,
        'epic_distribution': epic_counts
    }


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that handles division by zero
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

# ADD these enhanced classes to your existing utils/helpers.py file

class EnhancedTimezoneFormatter(logging.Formatter):
    """Enhanced logging formatter with better timezone handling and Claude score parsing"""
    
    def __init__(self, fmt=None, datefmt=None, timezone='Europe/Stockholm'):
        super().__init__(fmt, datefmt)
        self.timezone = pytz.timezone(timezone)
    
    def formatTime(self, record, datefmt=None):
        """Format time in user's timezone with timezone abbreviation"""
        try:
            # Convert UTC timestamp to user timezone
            utc_time = datetime.fromtimestamp(record.created, tz=pytz.UTC)
            local_time = utc_time.astimezone(self.timezone)
            
            if datefmt:
                return local_time.strftime(datefmt)
            else:
                # Include timezone abbreviation (CEST, CET, etc.)
                return local_time.strftime('%Y-%m-%d %H:%M:%S %Z')
        except Exception as e:
            # Fallback to default formatting
            return super().formatTime(record, datefmt)


def setup_enhanced_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    timezone: str = 'Europe/Stockholm'
) -> logging.Logger:
    """
    Enhanced logging setup that fixes timestamp display issues
    REPLACE your existing setup_logging function with this one
    
    Args:
        level: Logging level
        log_file: Optional log file path
        timezone: Timezone for log timestamps
        
    Returns:
        Configured logger
    """
    # Create enhanced timezone-aware formatter  
    formatter = EnhancedTimezoneFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %Z',
        timezone=timezone
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers to avoid conflicts
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Test the logging setup
    current_time = datetime.now(pytz.timezone(timezone))
    tz_name = current_time.strftime('%Z')
    
    root_logger.info(f"🕒 Enhanced logging initialized with timezone: {timezone}")
    root_logger.info(f"   Current timezone: {tz_name}")
    root_logger.info(f"   Current local time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    return root_logger


# ADD this utility function to help normalize timestamps anywhere in your code
def normalize_timestamp_for_display(timestamp, timezone='Europe/Stockholm'):
    """
    Utility function to normalize any timestamp for display in local timezone
    """
    try:
        import pandas as pd
        
        # Convert to pandas timestamp for easier handling
        if isinstance(timestamp, str):
            dt = pd.to_datetime(timestamp)
        else:
            dt = timestamp
        
        # Handle timezone-naive timestamps (assume UTC)
        if dt.tz is None:
            dt = dt.tz_localize('UTC')
        
        # Convert to local timezone
        local_tz = pytz.timezone(timezone)
        local_dt = dt.tz_convert(local_tz)
        
        # Return formatted string
        return local_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        
    except Exception as e:
        return str(timestamp)


# ADD this function to extract Claude scores from any log message
def extract_claude_info_from_log(log_message: str) -> dict:
    """
    Extract Claude quality score and decision from log messages
    Useful for parsing your existing logs
    """
    import re
    
    result = {
        'quality_score': None,
        'decision': None,
        'has_claude_info': False
    }
    
    try:
        # Extract quality score
        score_patterns = [
            r'Claude Quality Score[:\s]+(\d+)/10',
            r'CLAUDE[:\s]+(\d+)/10',
            r'Quality[:\s]+(\d+)/10'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, log_message, re.IGNORECASE)
            if match:
                result['quality_score'] = int(match.group(1))
                result['has_claude_info'] = True
                break
        
        # Extract decision
        if 'TRADE' in log_message.upper() and 'NO TRADE' not in log_message.upper():
            result['decision'] = 'TRADE'
            result['has_claude_info'] = True
        elif 'NO TRADE' in log_message.upper():
            result['decision'] = 'NO TRADE'
            result['has_claude_info'] = True
        
        return result
        
    except Exception as e:
        return result