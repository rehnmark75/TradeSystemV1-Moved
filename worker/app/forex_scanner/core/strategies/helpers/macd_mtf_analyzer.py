# core/strategies/helpers/macd_mtf_analyzer.py
"""
MACD Multi-TimeFrame Analyzer Module  
Handles multi-timeframe MACD analysis and validation
"""

import pandas as pd
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class MACDMultiTimeframeAnalyzer:
    """Handles multi-timeframe MACD analysis for signal validation"""
    
    def __init__(self, logger: logging.Logger = None, data_fetcher=None):
        self.logger = logger or logging.getLogger(__name__)
        self.data_fetcher = data_fetcher
        
        # MTF configuration
        self.mtf_enabled = getattr(config, 'MACD_MTF_ENABLED', True)
        self.check_timeframes = getattr(config, 'MACD_MTF_TIMEFRAMES', ['15m', '1h'])
        self.cache_duration_minutes = 5  # Cache MTF data for 5 minutes
        
        # Simple cache for MTF data
        self.mtf_cache = {}
        self.cache_timestamps = {}
    
    def is_mtf_enabled(self) -> bool:
        """Check if MTF analysis is enabled and available"""
        return self.mtf_enabled and self.data_fetcher is not None
    
    def validate_higher_timeframe_macd(self, epic: str, current_time: pd.Timestamp, signal_type: str) -> Dict:
        """
        Validate MACD signal against higher timeframes
        
        Args:
            epic: Trading pair epic
            current_time: Current timestamp
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            Dictionary with MTF validation results
        """
        if not self.is_mtf_enabled():
            return {
                'mtf_enabled': False,
                'validation_passed': True,  # Allow signal if MTF disabled
                'message': 'MTF analysis disabled'
            }
        
        try:
            results = {
                'mtf_enabled': True,
                'timeframes_checked': [],
                'timeframes_aligned': [],
                'validation_passed': False,
                'confidence_boost': 0.0,
                'details': {}
            }
            
            aligned_count = 0
            total_checked = 0
            
            # Check each higher timeframe
            for timeframe in self.check_timeframes:
                tf_result = self._check_timeframe_macd(epic, current_time, timeframe, signal_type)
                results['timeframes_checked'].append(timeframe)
                results['details'][timeframe] = tf_result
                
                total_checked += 1
                
                if tf_result.get('aligned', False):
                    results['timeframes_aligned'].append(timeframe)
                    aligned_count += 1
            
            # Calculate validation result
            if total_checked > 0:
                alignment_ratio = aligned_count / total_checked
                results['alignment_ratio'] = alignment_ratio
                
                # Require at least 50% of timeframes to be aligned
                min_alignment = getattr(config, 'MACD_MTF_MIN_ALIGNMENT', 0.5)
                results['validation_passed'] = alignment_ratio >= min_alignment
                
                # Calculate confidence boost based on alignment
                if alignment_ratio >= 0.75:
                    results['confidence_boost'] = 0.15  # Strong MTF alignment
                elif alignment_ratio >= 0.5:
                    results['confidence_boost'] = 0.08  # Moderate MTF alignment
                else:
                    results['confidence_boost'] = 0.0   # Weak MTF alignment
            
            self.logger.debug(f"MTF MACD validation for {epic} {signal_type}: {aligned_count}/{total_checked} aligned")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in MTF MACD validation: {e}")
            return {
                'mtf_enabled': True,
                'validation_passed': True,  # Allow signal on error
                'error': str(e)
            }
    
    def _check_timeframe_macd(self, epic: str, current_time: pd.Timestamp, timeframe: str, signal_type: str) -> Dict:
        """
        Check MACD alignment for a specific timeframe
        
        Args:
            epic: Trading pair epic
            current_time: Current timestamp
            timeframe: Timeframe to check ('15m', '1h', etc.)
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            Dictionary with timeframe-specific results
        """
        try:
            # Check cache first
            cache_key = f"{epic}_{timeframe}_{signal_type}"
            if self._is_cache_valid(cache_key, current_time):
                cached_result = self.mtf_cache.get(cache_key)
                if cached_result:
                    self.logger.debug(f"Using cached MTF data for {cache_key}")
                    return cached_result
            
            # Fetch data for the timeframe
            tf_data = self._fetch_timeframe_data(epic, timeframe, current_time)
            if tf_data is None or len(tf_data) < 30:
                return {
                    'aligned': False,
                    'reason': 'insufficient_data',
                    'data_length': len(tf_data) if tf_data is not None else 0
                }
            
            # Calculate MACD indicators if not present
            tf_data = self._ensure_macd_indicators(tf_data)
            
            # Get latest MACD values
            latest = tf_data.iloc[-1]
            histogram = latest.get('macd_histogram', 0)
            macd_line = latest.get('macd_line', 0)
            macd_signal = latest.get('macd_signal', 0)
            
            # Check MACD alignment with signal direction
            if signal_type == 'BULL':
                aligned = (
                    histogram > 0 and  # Positive momentum
                    macd_line > macd_signal  # MACD above signal line
                )
            else:  # BEAR
                aligned = (
                    histogram < 0 and  # Negative momentum
                    macd_line < macd_signal  # MACD below signal line
                )
            
            result = {
                'aligned': aligned,
                'histogram': histogram,
                'macd_line': macd_line, 
                'macd_signal': macd_signal,
                'data_length': len(tf_data),
                'timeframe': timeframe
            }
            
            # Cache the result
            self.mtf_cache[cache_key] = result
            self.cache_timestamps[cache_key] = current_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error checking timeframe {timeframe} MACD: {e}")
            return {
                'aligned': False,
                'error': str(e),
                'timeframe': timeframe
            }
    
    def _fetch_timeframe_data(self, epic: str, timeframe: str, current_time: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        Fetch data for a specific timeframe
        
        Args:
            epic: Trading pair epic
            timeframe: Timeframe string
            current_time: Current timestamp
            
        Returns:
            DataFrame with price data or None if error
        """
        try:
            if not self.data_fetcher:
                return None
            
            # Calculate how much data we need (50 bars minimum for stable MACD)
            lookback_hours = self._get_lookback_hours(timeframe)
            
            # Fetch data
            data = self.data_fetcher.get_candles(
                epic=epic,
                timeframe=timeframe,
                count=60,  # Get extra bars for stability
                end_time=current_time
            )
            
            if data is not None and len(data) > 0:
                self.logger.debug(f"Fetched {len(data)} bars for {epic} {timeframe}")
                return data
            else:
                self.logger.warning(f"No data returned for {epic} {timeframe}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching {timeframe} data for {epic}: {e}")
            return None
    
    def _ensure_macd_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD indicators if not present in DataFrame
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with MACD indicators added
        """
        try:
            df_copy = df.copy()
            
            # Calculate EMAs for MACD (12, 26, 9)
            if 'ema_12' not in df_copy.columns:
                df_copy['ema_12'] = df_copy['close'].ewm(span=12).mean()
            if 'ema_26' not in df_copy.columns:
                df_copy['ema_26'] = df_copy['close'].ewm(span=26).mean()
            
            # Calculate MACD line
            if 'macd_line' not in df_copy.columns:
                df_copy['macd_line'] = df_copy['ema_12'] - df_copy['ema_26']
            
            # Calculate MACD signal line
            if 'macd_signal' not in df_copy.columns:
                df_copy['macd_signal'] = df_copy['macd_line'].ewm(span=9).mean()
            
            # Calculate MACD histogram
            if 'macd_histogram' not in df_copy.columns:
                df_copy['macd_histogram'] = df_copy['macd_line'] - df_copy['macd_signal']
            
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD indicators: {e}")
            return df
    
    def _get_lookback_hours(self, timeframe: str) -> int:
        """Get appropriate lookback hours for timeframe"""
        timeframe_hours = {
            '5m': 8,    # 8 hours = 96 bars
            '15m': 24,  # 24 hours = 96 bars  
            '1h': 96,   # 96 hours = 96 bars
            '4h': 384,  # 384 hours = 96 bars
            '1d': 2160  # 90 days = ~96 bars
        }
        return timeframe_hours.get(timeframe, 24)
    
    def _is_cache_valid(self, cache_key: str, current_time: pd.Timestamp) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache_timestamps:
            return False
        
        cached_time = self.cache_timestamps[cache_key]
        age_minutes = (current_time - cached_time).total_seconds() / 60
        
        return age_minutes < self.cache_duration_minutes
    
    def clear_cache(self):
        """Clear MTF cache"""
        self.mtf_cache.clear()
        self.cache_timestamps.clear()
        self.logger.debug("MTF cache cleared")
    
    def get_mtf_summary(self, epic: str, current_time: pd.Timestamp, signal_type: str) -> str:
        """
        Get a human-readable summary of MTF analysis
        
        Args:
            epic: Trading pair epic
            current_time: Current timestamp
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            String summary of MTF results
        """
        try:
            if not self.is_mtf_enabled():
                return "MTF disabled"
            
            results = self.validate_higher_timeframe_macd(epic, current_time, signal_type)
            
            if results.get('validation_passed'):
                aligned = results.get('timeframes_aligned', [])
                return f"MTF✅ {'/'.join(aligned)} aligned"
            else:
                checked = results.get('timeframes_checked', [])
                aligned = results.get('timeframes_aligned', [])
                return f"MTF❌ {len(aligned)}/{len(checked)} aligned"
                
        except Exception as e:
            return f"MTF⚠️ {str(e)[:20]}"