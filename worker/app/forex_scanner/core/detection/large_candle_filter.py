# core/detection/large_candle_filter.py
"""
Large Candle Movement Filter
Prevents entries after excessive price movements that could indicate exhaustion

This filter analyzes recent candle size and movement to prevent entries
when price has moved too aggressively, which often leads to reversals.

CRITICAL: Database-driven configuration - NO FALLBACK to config.py
All settings must come from scanner_global_config table.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from datetime import datetime

# Import scanner config service for database-driven settings - REQUIRED, NO FALLBACK
try:
    from forex_scanner.services.scanner_config_service import get_scanner_config, ScannerConfig
    SCANNER_CONFIG_AVAILABLE = True
except ImportError:
    SCANNER_CONFIG_AVAILABLE = False


class LargeCandleFilter:
    """
    Filters out signals that occur after unusually large price movements

    Key Features:
    - Detects abnormally large candles based on ATR (Average True Range)
    - Prevents entries after parabolic moves
    - Configurable sensitivity for different market conditions
    - Works with multiple timeframes

    CRITICAL: Database-driven configuration - NO FALLBACK to config.py
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # ✅ CRITICAL: Database-driven configuration - NO FALLBACK to config.py
        if not SCANNER_CONFIG_AVAILABLE:
            raise RuntimeError(
                "❌ CRITICAL: Scanner config service not available - database is REQUIRED, no fallback allowed"
            )

        try:
            self._scanner_cfg = get_scanner_config()
        except Exception as e:
            raise RuntimeError(
                f"❌ CRITICAL: Failed to load scanner config from database: {e} - no fallback allowed"
            )

        if not self._scanner_cfg:
            raise RuntimeError(
                "❌ CRITICAL: Scanner config returned None - database is REQUIRED, no fallback allowed"
            )

        # Configuration from database - NO FALLBACK
        self.large_candle_multiplier = self._scanner_cfg.large_candle_atr_multiplier
        self.consecutive_large_threshold = self._scanner_cfg.consecutive_large_candles_threshold
        self.movement_lookback_periods = self._scanner_cfg.movement_lookback_periods
        self.excessive_movement_threshold = self._scanner_cfg.excessive_movement_threshold_pips
        self.filter_cooldown_periods = self._scanner_cfg.large_candle_filter_cooldown

        self.logger.info("[CONFIG:DB] ✅ Large candle filter config loaded from database (NO FALLBACK)")
        
        # Track filtered signals for debugging
        self.filter_stats = {
            'total_signals_checked': 0,
            'filtered_large_candle': 0,
            'filtered_excessive_movement': 0,
            'filtered_parabolic_move': 0
        }
    
    def should_block_entry(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        signal_type: str,
        timeframe: str = '5m'
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if entry should be blocked due to large candle movement
        
        Args:
            df: DataFrame with OHLC data and indicators
            epic: Trading instrument
            signal_type: 'BULL' or 'BEAR'
            timeframe: Chart timeframe
            
        Returns:
            Tuple of (should_block: bool, reason: str or None)
        """
        try:
            self.filter_stats['total_signals_checked'] += 1
            
            if len(df) < self.movement_lookback_periods + 1:
                return False, None
            
            # Get latest candles for analysis
            latest_candles = df.tail(self.movement_lookback_periods + 1)
            current_candle = latest_candles.iloc[-1]
            
            # Check 1: Single large candle filter
            large_candle_block, large_candle_reason = self._check_large_candle(
                latest_candles, current_candle
            )
            if large_candle_block:
                self.filter_stats['filtered_large_candle'] += 1
                return True, large_candle_reason
            
            # Check 2: Excessive cumulative movement
            excessive_movement_block, movement_reason = self._check_excessive_movement(
                latest_candles, epic, signal_type
            )
            if excessive_movement_block:
                self.filter_stats['filtered_excessive_movement'] += 1
                return True, movement_reason
            
            # Check 3: Parabolic movement detection
            parabolic_block, parabolic_reason = self._check_parabolic_movement(
                latest_candles, signal_type
            )
            if parabolic_block:
                self.filter_stats['filtered_parabolic_move'] += 1
                return True, parabolic_reason
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"❌ Large candle filter error for {epic}: {e}")
            return False, None
    
    def _check_large_candle(
        self, 
        latest_candles: pd.DataFrame, 
        current_candle: pd.Series
    ) -> Tuple[bool, Optional[str]]:
        """Check for abnormally large single candles"""
        try:
            # Calculate ATR for context
            atr = self._calculate_atr(latest_candles)
            if atr <= 0:
                return False, None
            
            # Check recent candles for size
            large_candles_found = []
            
            for i in range(len(latest_candles)):
                candle = latest_candles.iloc[i]
                candle_range = candle['high'] - candle['low']
                candle_body = abs(candle['close'] - candle['open'])
                
                # Check if candle is abnormally large
                if candle_range > (atr * self.large_candle_multiplier):
                    periods_ago = len(latest_candles) - i - 1
                    large_candles_found.append({
                        'periods_ago': periods_ago,
                        'range_atr_ratio': candle_range / atr,
                        'body_size': candle_body,
                        'is_current': periods_ago == 0
                    })
            
            if not large_candles_found:
                return False, None
            
            # Block if we have recent large candles
            recent_large = [lc for lc in large_candles_found if lc['periods_ago'] <= self.filter_cooldown_periods]
            
            if recent_large:
                if len(recent_large) >= self.consecutive_large_threshold:
                    return True, f"Multiple large candles detected ({len(recent_large)} in last {self.filter_cooldown_periods} periods)"
                
                # Check if the large candle is very recent
                most_recent = min(recent_large, key=lambda x: x['periods_ago'])
                if most_recent['periods_ago'] <= 1 and most_recent['range_atr_ratio'] > 3.0:
                    return True, f"Very large candle {most_recent['periods_ago']} periods ago (range {most_recent['range_atr_ratio']:.1f}x ATR)"
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"❌ Large candle check failed: {e}")
            return False, None
    
    def _check_excessive_movement(
        self, 
        latest_candles: pd.DataFrame, 
        epic: str, 
        signal_type: str
    ) -> Tuple[bool, Optional[str]]:
        """Check for excessive cumulative price movement"""
        try:
            # Calculate cumulative movement in recent periods
            if len(latest_candles) < 2:
                return False, None
            
            start_price = latest_candles.iloc[0]['close']
            end_price = latest_candles.iloc[-1]['close']
            high_price = latest_candles['high'].max()
            low_price = latest_candles['low'].min()
            
            # Calculate pip movement (assume 4-decimal place pairs)
            pip_multiplier = 10000 if 'JPY' not in epic.upper() else 100
            
            total_range_pips = (high_price - low_price) * pip_multiplier
            net_movement_pips = abs(end_price - start_price) * pip_multiplier
            
            # Check against thresholds
            if total_range_pips > self.excessive_movement_threshold:
                # Additional check: is the movement in the same direction as signal?
                movement_direction = 'BULL' if end_price > start_price else 'BEAR'
                
                # Block if signal is in same direction as recent large movement
                if movement_direction == signal_type:
                    return True, f"Excessive movement in signal direction: {total_range_pips:.1f} pips in {len(latest_candles)} periods"
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"❌ Excessive movement check failed: {e}")
            return False, None
    
    def _check_parabolic_movement(
        self, 
        latest_candles: pd.DataFrame, 
        signal_type: str
    ) -> Tuple[bool, Optional[str]]:
        """Detect parabolic price movements that often precede reversals"""
        try:
            if len(latest_candles) < 3:
                return False, None
            
            closes = latest_candles['close'].values
            
            # Calculate acceleration in price movement
            price_changes = np.diff(closes)
            
            if len(price_changes) < 2:
                return False, None
            
            # Check for accelerating moves
            recent_changes = price_changes[-3:]  # Last 3 price changes
            
            # For bullish signals, check for accelerating upward movement
            if signal_type == 'BULL':
                positive_changes = recent_changes[recent_changes > 0]
                if len(positive_changes) >= 2:
                    # Check if movement is accelerating
                    if len(positive_changes) == len(recent_changes):  # All moves up
                        avg_change = np.mean(positive_changes)
                        last_change = recent_changes[-1]
                        if last_change > avg_change * 1.5:  # Last move 50% larger than average
                            return True, "Parabolic bullish movement detected - potential exhaustion"
            
            # For bearish signals, check for accelerating downward movement  
            elif signal_type == 'BEAR':
                negative_changes = recent_changes[recent_changes < 0]
                if len(negative_changes) >= 2:
                    if len(negative_changes) == len(recent_changes):  # All moves down
                        avg_change = np.mean(np.abs(negative_changes))
                        last_change = abs(recent_changes[-1])
                        if last_change > avg_change * 1.5:
                            return True, "Parabolic bearish movement detected - potential exhaustion"
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"❌ Parabolic movement check failed: {e}")
            return False, None
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range for context"""
        try:
            if len(df) < period:
                period = len(df) - 1
            
            if period <= 0:
                return 0.0
            
            # Use ATR from dataframe if available
            if 'atr' in df.columns:
                return float(df['atr'].iloc[-1])
            
            # Calculate ATR manually
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return float(atr) if not pd.isna(atr) else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ ATR calculation failed: {e}")
            return 0.0
    
    def get_filter_statistics(self) -> Dict:
        """Get filtering statistics for monitoring"""
        total_checked = self.filter_stats['total_signals_checked']
        if total_checked == 0:
            return self.filter_stats
        
        stats = self.filter_stats.copy()
        stats['filter_rate'] = {
            'large_candle_rate': (stats['filtered_large_candle'] / total_checked) * 100,
            'excessive_movement_rate': (stats['filtered_excessive_movement'] / total_checked) * 100,
            'parabolic_movement_rate': (stats['filtered_parabolic_move'] / total_checked) * 100,
            'total_filter_rate': ((stats['filtered_large_candle'] + 
                                 stats['filtered_excessive_movement'] + 
                                 stats['filtered_parabolic_move']) / total_checked) * 100
        }
        
        return stats
    
    def reset_statistics(self):
        """Reset filter statistics"""
        self.filter_stats = {
            'total_signals_checked': 0,
            'filtered_large_candle': 0,
            'filtered_excessive_movement': 0,
            'filtered_parabolic_move': 0
        }


# LEGACY: Configuration moved to database (scanner_global_config table)
# All settings are now managed via the Streamlit UI or direct database updates.
# This dict is kept only for documentation purposes.
#
# Settings now in database:
# - large_candle_atr_multiplier (default 2.5)
# - consecutive_large_candles_threshold (default 2)
# - movement_lookback_periods (default 5)
# - excessive_movement_threshold_pips (default 15)
# - large_candle_filter_cooldown (default 3)
# - enable_large_candle_filter (default True)