# core/strategies/helpers/ema_mtf_analyzer.py
"""
Multi-Timeframe Analyzer Module (EMA/Supertrend)
Handles multi-timeframe analysis for both EMA and Supertrend strategies
"""

import pandas as pd
import logging
from typing import Optional
try:
    from configdata import config
    from configdata.strategies import config_ema_strategy
except ImportError:
    from forex_scanner.configdata import config
    from forex_scanner.configdata.strategies import config_ema_strategy

from .supertrend_calculator import SupertrendCalculator


class EMAMultiTimeframeAnalyzer:
    """Analyzes signals across multiple timeframes for confirmation"""

    def __init__(self, logger: logging.Logger = None, data_fetcher=None, use_supertrend: Optional[bool] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.data_fetcher = data_fetcher

        # Determine which mode to use
        if use_supertrend is None:
            self.use_supertrend = getattr(config_ema_strategy, 'USE_SUPERTREND_MODE', False)
        else:
            self.use_supertrend = use_supertrend

        # Initialize Supertrend calculator if needed
        if self.use_supertrend:
            self.supertrend_calc = SupertrendCalculator(logger=self.logger)
            self.logger.info("📊 MTF Analyzer initialized in SUPERTREND mode")
        else:
            self.supertrend_calc = None
            self.logger.info("📊 MTF Analyzer initialized in EMA mode (legacy)")
    
    def get_4h_ema_alignment(self, epic: str, current_time: pd.Timestamp) -> Optional[str]:
        """
        Get 4H EMA alignment for trend direction filter

        Replaces 1H Two-Pole color check with 4H EMA trend validation

        Args:
            epic: Trading instrument identifier
            current_time: Current timestamp for data lookup

        Returns:
            'bullish': 4H EMAs aligned bullish (21 > 50 > 200)
            'bearish': 4H EMAs aligned bearish (21 < 50 < 200)
            'neutral': EMAs not clearly aligned
            None: Data not available
        """
        try:
            if not self.data_fetcher:
                self.logger.debug("No data fetcher available for 4H data")
                return None

            # Extract pair from epic
            parts = epic.split('.')
            if len(parts) >= 3:
                pair = parts[2].replace('MINI', '').replace('CFD', '')
            else:
                self.logger.debug(f"Cannot extract pair from epic: {epic}")
                return None

            # Fetch 4H data
            df_4h = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe='4h',
                lookback_hours=500  # ~125 4H candles (3+ weeks)
            )

            if df_4h is None or df_4h.empty:
                self.logger.debug(f"No 4H data available for {epic}")
                return None

            # Get most recent 4H candle
            if 'start_time' in df_4h.columns:
                valid_candles = df_4h[df_4h['start_time'] <= current_time]
                if valid_candles.empty:
                    latest_4h = df_4h.iloc[-1]
                else:
                    latest_4h = valid_candles.iloc[-1]
            else:
                latest_4h = df_4h.iloc[-1]

            # Check EMA alignment
            ema21 = latest_4h.get('ema_21', 0)
            ema50 = latest_4h.get('ema_50', 0)
            ema200 = latest_4h.get('ema_200', 0)

            eps = 1e-8

            # Bullish alignment
            if ema21 > ema50 + eps and ema50 > ema200 + eps:
                self.logger.info(f"✅ 4H trend: BULLISH (21:{ema21:.5f} > 50:{ema50:.5f} > 200:{ema200:.5f})")
                return 'bullish'

            # Bearish alignment
            elif ema21 < ema50 - eps and ema50 < ema200 - eps:
                self.logger.info(f"✅ 4H trend: BEARISH (21:{ema21:.5f} < 50:{ema50:.5f} < 200:{ema200:.5f})")
                return 'bearish'

            # No clear alignment
            else:
                self.logger.info(f"⚠️ 4H trend: NEUTRAL (EMAs not aligned)")
                return 'neutral'

        except Exception as e:
            self.logger.error(f"Error getting 4H EMA alignment: {e}")
            return None

    def get_4h_rsi(self, epic: str, current_time: pd.Timestamp) -> Optional[float]:
        """
        Get 4H RSI value for momentum confirmation

        Replaces 1H Two-Pole momentum check with 4H RSI

        Args:
            epic: Trading instrument identifier
            current_time: Current timestamp for data lookup

        Returns:
            RSI value (0-100) or None if unavailable
        """
        try:
            if not self.data_fetcher:
                return None

            # Extract pair
            parts = epic.split('.')
            if len(parts) >= 3:
                pair = parts[2].replace('MINI', '').replace('CFD', '')
            else:
                return None

            # Fetch 4H data
            df_4h = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe='4h',
                lookback_hours=500
            )

            if df_4h is None or df_4h.empty:
                return None

            # Get most recent candle
            if 'start_time' in df_4h.columns:
                valid_candles = df_4h[df_4h['start_time'] <= current_time]
                if valid_candles.empty:
                    latest_4h = df_4h.iloc[-1]
                else:
                    latest_4h = valid_candles.iloc[-1]
            else:
                latest_4h = df_4h.iloc[-1]

            rsi_4h = latest_4h.get('rsi', None)

            if rsi_4h is not None:
                self.logger.debug(f"📊 4H RSI: {rsi_4h:.1f}")

            return rsi_4h

        except Exception as e:
            self.logger.error(f"Error getting 4H RSI: {e}")
            return None

    def validate_4h_trend_and_momentum(self, epic: str, current_time: pd.Timestamp, signal_type: str) -> tuple:
        """
        Validate signal against 4H trend and RSI momentum

        Replaces validate_1h_two_pole() with comprehensive 4H validation

        Args:
            epic: Trading instrument
            current_time: Current timestamp
            signal_type: 'BULL' or 'BEAR'

        Returns:
            (is_valid, confidence_boost): Validation result and confidence bonus
        """
        from configdata.strategies.config_ema_strategy import (
            EMA_4H_TREND_FILTER_ENABLED,
            EMA_4H_RSI_FILTER_ENABLED,
            EMA_4H_RSI_BULL_MIN,
            EMA_4H_RSI_BEAR_MAX,
            EMA_4H_RSI_BONUS
        )

        confidence_boost = 0.0

        # Check 4H EMA trend alignment
        if EMA_4H_TREND_FILTER_ENABLED:
            trend_4h = self.get_4h_ema_alignment(epic, current_time)

            if signal_type == 'BULL':
                if trend_4h == 'bearish':
                    self.logger.warning(f"❌ BULL signal REJECTED: 4H trend is BEARISH")
                    return False, 0.0
                elif trend_4h == 'bullish':
                    confidence_boost += EMA_4H_RSI_BONUS
                    self.logger.info(f"✅ 4H trend aligned with BULL signal (+{EMA_4H_RSI_BONUS:.1%})")
                # If neutral or unavailable, allow signal (graceful degradation)

            elif signal_type == 'BEAR':
                if trend_4h == 'bullish':
                    self.logger.warning(f"❌ BEAR signal REJECTED: 4H trend is BULLISH")
                    return False, 0.0
                elif trend_4h == 'bearish':
                    confidence_boost += EMA_4H_RSI_BONUS
                    self.logger.info(f"✅ 4H trend aligned with BEAR signal (+{EMA_4H_RSI_BONUS:.1%})")

        # Check 4H RSI momentum
        if EMA_4H_RSI_FILTER_ENABLED:
            rsi_4h = self.get_4h_rsi(epic, current_time)

            if rsi_4h is not None:
                if signal_type == 'BULL':
                    if rsi_4h < EMA_4H_RSI_BULL_MIN:
                        self.logger.warning(f"❌ BULL signal REJECTED: 4H RSI too low ({rsi_4h:.1f} < {EMA_4H_RSI_BULL_MIN})")
                        return False, 0.0
                    else:
                        self.logger.info(f"✅ 4H RSI confirms BULL momentum: {rsi_4h:.1f}")

                elif signal_type == 'BEAR':
                    if rsi_4h > EMA_4H_RSI_BEAR_MAX:
                        self.logger.warning(f"❌ BEAR signal REJECTED: 4H RSI too high ({rsi_4h:.1f} > {EMA_4H_RSI_BEAR_MAX})")
                        return False, 0.0
                    else:
                        self.logger.info(f"✅ 4H RSI confirms BEAR momentum: {rsi_4h:.1f}")

        return True, confidence_boost
    
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
    def get_4h_supertrend_alignment(self, epic: str, current_time: pd.Timestamp) -> Optional[str]:
        """
        Get 4H Supertrend alignment for trend direction filter

        Replaces 4H EMA check with 4H Supertrend trend validation.
        Uses Medium Supertrend on 4H for more reliable filtering.

        Args:
            epic: Trading instrument identifier
            current_time: Current timestamp for data lookup

        Returns:
            'bullish': 4H Supertrend is bullish (trend = 1)
            'bearish': 4H Supertrend is bearish (trend = -1)
            None: Data not available or Supertrend not initialized
        """
        try:
            if not self.data_fetcher:
                self.logger.debug("No data fetcher available for 4H Supertrend")
                return None

            if not self.supertrend_calc:
                self.logger.debug("Supertrend calculator not initialized")
                return None

            # Extract pair from epic
            parts = epic.split('.')
            if len(parts) >= 3:
                pair = parts[2].replace('MINI', '').replace('CFD', '')
            else:
                self.logger.debug(f"Cannot extract pair from epic: {epic}")
                return None

            # Fetch 4H data
            df_4h = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe='4h',
                lookback_hours=500  # ~125 4H candles
            )

            if df_4h is None or df_4h.empty:
                self.logger.debug(f"No 4H data available for {epic}")
                return None

            # Ensure required columns exist
            if not all(col in df_4h.columns for col in ['high', 'low', 'close']):
                self.logger.debug("Missing required OHLC columns in 4H data")
                return None

            # Get Supertrend config
            st_config = getattr(config_ema_strategy, 'SUPERTREND_STRATEGY_CONFIG', {}).get('default', {})
            use_medium = getattr(config_ema_strategy, 'SUPERTREND_4H_USE_MEDIUM', True)

            if use_medium:
                # Use Medium Supertrend on 4H (more stable)
                period = st_config.get('medium_period', 14)
                multiplier = st_config.get('medium_multiplier', 2.0)
            else:
                # Use Fast Supertrend on 4H (more responsive)
                period = st_config.get('fast_period', 7)
                multiplier = st_config.get('fast_multiplier', 1.5)

            atr_period = st_config.get('atr_period', 14)

            # Calculate 4H Supertrend
            supertrend_4h = self.supertrend_calc.calculate_supertrend(
                df_4h,
                period=period,
                multiplier=multiplier,
                atr_period=atr_period
            )

            # Get most recent 4H candle
            if 'start_time' in df_4h.columns:
                valid_candles = df_4h[df_4h['start_time'] <= current_time]
                if valid_candles.empty:
                    latest_idx = -1
                else:
                    latest_idx = valid_candles.index[-1]
            else:
                latest_idx = -1

            trend_4h = supertrend_4h['trend'].iloc[latest_idx]

            if trend_4h == 1:
                self.logger.debug(f"4H Supertrend: BULLISH (period={period}, mult={multiplier})")
                return 'bullish'
            elif trend_4h == -1:
                self.logger.debug(f"4H Supertrend: BEARISH (period={period}, mult={multiplier})")
                return 'bearish'
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error getting 4H Supertrend alignment: {e}", exc_info=True)
            return None
