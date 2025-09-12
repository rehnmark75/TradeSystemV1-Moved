# core/strategies/helpers/ema_mtf_analyzer.py
"""
EMA Multi-Timeframe Analyzer Module
Handles multi-timeframe analysis for EMA strategy signals
"""

import pandas as pd
import logging
from typing import Optional
try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class EMAMultiTimeframeAnalyzer:
    """Analyzes signals across multiple timeframes for confirmation"""
    
    def __init__(self, logger: logging.Logger = None, data_fetcher=None):
        self.logger = logger or logging.getLogger(__name__)
        self.data_fetcher = data_fetcher
    
    def get_1h_two_pole_color(self, epic: str, current_time: pd.Timestamp) -> Optional[str]:
        """
        Get the Two-Pole Oscillator color from 1H timeframe
        
        Args:
            epic: Trading instrument identifier
            current_time: Current timestamp for data lookup
        
        Returns:
            'green' if oscillator is rising (bullish)
            'purple' if oscillator is falling (bearish)
            None if data not available
        """
        try:
            if not self.data_fetcher:
                self.logger.debug("No data fetcher available for 1H data")
                return None
            
            # Extract pair from epic (e.g., "CS.D.EURUSD.MINI.IP" -> "EURUSD")
            parts = epic.split('.')
            if len(parts) >= 3:
                pair = parts[2].replace('MINI', '').replace('CFD', '')
            else:
                self.logger.debug(f"Cannot extract pair from epic: {epic}")
                return None
            
            # Fetch 1H data with lookback_hours parameter
            df_1h = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe='1h',
                lookback_hours=150  # Get 150 hours of 1H data
            )
            
            if df_1h is None or df_1h.empty:
                self.logger.debug(f"No 1H data available for {epic}")
                return None
            
            # Get the most recent 1H candle's Two-Pole color
            # Find the candle closest to current_time
            if 'start_time' in df_1h.columns:
                # Find the most recent candle before or at current_time
                valid_candles = df_1h[df_1h['start_time'] <= current_time]
                if valid_candles.empty:
                    latest_1h = df_1h.iloc[-1]  # Use last available if all are future
                else:
                    latest_1h = valid_candles.iloc[-1]
            else:
                latest_1h = df_1h.iloc[-1]
            
            if latest_1h.get('two_pole_is_green', False):
                return 'green'
            elif latest_1h.get('two_pole_is_purple', False):
                return 'purple'
            else:
                self.logger.debug("1H Two-Pole color not clearly defined")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting 1H Two-Pole color: {e}")
            return None
    
    def validate_1h_two_pole(self, epic: str, current_time: pd.Timestamp, signal_type: str) -> bool:
        """
        Validate signal against 1H Two-Pole Oscillator color
        
        Args:
            epic: Trading instrument identifier
            current_time: Current timestamp for data lookup
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            True if 1H timeframe confirms signal, False otherwise
        """
        if not getattr(config, 'TWO_POLE_MTF_VALIDATION', True):
            return True
        
        one_hour_color = self.get_1h_two_pole_color(epic, current_time)
        
        if signal_type == 'BULL':
            if one_hour_color == 'purple':
                self.logger.warning(f"❌ EMA BULL signal REJECTED: 1H Two-Pole Oscillator is PURPLE (bearish)")
                return False
            elif one_hour_color == 'green':
                self.logger.info(f"✅ Both 15m and 1H Two-Pole are GREEN - BULL signal confirmed")
                return True
            else:
                self.logger.debug(f"⚠️ 1H Two-Pole color unavailable, using 15m only")
                return True  # Allow if unavailable
        
        elif signal_type == 'BEAR':
            if one_hour_color == 'green':
                self.logger.warning(f"❌ EMA BEAR signal REJECTED: 1H Two-Pole Oscillator is GREEN (bullish)")
                return False
            elif one_hour_color == 'purple':
                self.logger.info(f"✅ Both 15m and 1H Two-Pole are PURPLE - BEAR signal confirmed")
                return True
            else:
                self.logger.debug(f"⚠️ 1H Two-Pole color unavailable, using 15m only")
                return True  # Allow if unavailable
        
        return True
    
    def get_higher_timeframe_trend(self, epic: str, current_timeframe: str) -> Optional[str]:
        """
        Get trend direction from higher timeframe
        
        Args:
            epic: Trading instrument identifier
            current_timeframe: Current timeframe (e.g., '5m', '15m')
            
        Returns:
            'bullish', 'bearish', or None if unavailable
        """
        try:
            if not self.data_fetcher:
                return None
            
            # Map current timeframe to higher timeframe
            timeframe_map = {
                '1m': '15m',
                '5m': '1h',
                '15m': '4h',
                '1h': '1d'
            }
            
            higher_tf = timeframe_map.get(current_timeframe)
            if not higher_tf:
                return None
            
            # Extract pair from epic
            parts = epic.split('.')
            if len(parts) >= 3:
                pair = parts[2].replace('MINI', '').replace('CFD', '')
            else:
                return None
            
            # Fetch higher timeframe data
            df_higher = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=higher_tf,
                lookback_hours=200
            )
            
            if df_higher is None or df_higher.empty:
                return None
            
            # Get latest candle
            latest = df_higher.iloc[-1]
            
            # Determine trend based on EMA alignment
            ema_short = latest.get('ema_12', 0) or latest.get('ema_21', 0)
            ema_long = latest.get('ema_50', 0)
            ema_trend = latest.get('ema_200', 0)
            
            if ema_short > ema_long > ema_trend:
                return 'bullish'
            elif ema_short < ema_long < ema_trend:
                return 'bearish'
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting higher timeframe trend: {e}")
            return None