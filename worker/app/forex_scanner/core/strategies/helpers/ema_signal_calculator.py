# core/strategies/helpers/ema_signal_calculator.py
"""
Multi-Strategy Signal Calculator Module (EMA/Supertrend)
Handles confidence calculation and signal strength assessment for both EMA and Supertrend strategies
"""

import pandas as pd
import logging
from typing import Optional, Dict
try:
    from configdata import config
    from configdata.strategies import config_ema_strategy
except ImportError:
    from forex_scanner.configdata import config
    from forex_scanner.configdata.strategies import config_ema_strategy


class EMASignalCalculator:
    """Calculates confidence scores and signal strength for EMA and Supertrend signals"""

    def __init__(self, logger: logging.Logger = None, trend_validator=None, swing_validator=None, use_supertrend: Optional[bool] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.trend_validator = trend_validator
        self.swing_validator = swing_validator  # Swing proximity validator
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.45)

        # Determine which mode to use
        if use_supertrend is None:
            self.use_supertrend = getattr(config_ema_strategy, 'USE_SUPERTREND_MODE', False)
        else:
            self.use_supertrend = use_supertrend

        if self.use_supertrend:
            self.logger.info("📊 Signal Calculator initialized in SUPERTREND mode")
        else:
            self.logger.info("📊 Signal Calculator initialized in EMA mode (legacy)")
    
    def calculate_simple_confidence(self, latest_row: pd.Series, signal_type: str) -> float:
        """
        SIMPLE CONFIDENCE CALCULATION: Based on EMA alignment and crossover strength
        
        Factors considered:
        - EMA trend alignment (most important)
        - Price position relative to EMAs
        - MACD histogram alignment (if available)
        - Crossover strength (how far from EMA)
        
        Args:
            latest_row: DataFrame row with indicator data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            Confidence score between 0.0 and 0.95
        """
        try:
            base_confidence = 0.5  # Start with 50%
            
            # Get EMA values
            ema_short = latest_row.get('ema_short', 0)
            ema_long = latest_row.get('ema_long', 0)
            ema_trend = latest_row.get('ema_trend', 0)
            close = latest_row.get('close', 0)
            
            if ema_short == 0 or ema_long == 0 or ema_trend == 0:
                return 0.3  # Low confidence if EMAs missing
            
            # 1. EMA TREND ALIGNMENT (40% weight)
            if signal_type == 'BULL':
                # Bull signal: check if ema_short > ema_long > ema_trend
                if ema_short > ema_long and ema_long > ema_trend:
                    base_confidence += 0.3  # Strong alignment
                elif ema_short > ema_long:
                    base_confidence += 0.1  # Partial alignment
            else:  # BEAR
                # Bear signal: check if ema_short < ema_long < ema_trend  
                if ema_short < ema_long and ema_long < ema_trend:
                    base_confidence += 0.3  # Strong alignment
                elif ema_short < ema_long:
                    base_confidence += 0.1  # Partial alignment
            
            # 2. PRICE POSITION (20% weight)
            if signal_type == 'BULL':
                if close > ema_short and close > ema_long:
                    base_confidence += 0.15  # Price above EMAs
                elif close > ema_short:
                    base_confidence += 0.05  # Price above short EMA
            else:  # BEAR
                if close < ema_short and close < ema_long:
                    base_confidence += 0.15  # Price below EMAs
                elif close < ema_short:
                    base_confidence += 0.05  # Price below short EMA
            
            # 3. MACD CONFIRMATION (15% weight)
            macd_histogram = latest_row.get('macd_histogram', 0)
            if macd_histogram != 0:
                if signal_type == 'BULL' and macd_histogram > 0:
                    base_confidence += 0.1  # MACD supports bull signal
                elif signal_type == 'BEAR' and macd_histogram < 0:
                    base_confidence += 0.1  # MACD supports bear signal
            
            # 4. CROSSOVER STRENGTH (10% weight)
            # Check how strong the crossover was
            bull_cross = latest_row.get('bull_cross', False)
            bear_cross = latest_row.get('bear_cross', False)
            
            if (signal_type == 'BULL' and bull_cross) or (signal_type == 'BEAR' and bear_cross):
                base_confidence += 0.05  # Confirmed crossover
            
            # 5. EMA SEPARATION (15% weight) - EMAs not too close
            ema_separation = abs(ema_short - ema_long)
            if ema_separation > 0.0001:  # Reasonable separation for forex
                base_confidence += 0.1
            
            # 6. RSI ZONE VALIDATION (replaces Two-Pole - 15% weight)
            from configdata.strategies.config_ema_strategy import EMA_USE_RSI_FILTER
            if EMA_USE_RSI_FILTER and self.trend_validator:
                rsi_valid, rsi_boost = self.trend_validator.validate_rsi_zone(latest_row, signal_type)
                if rsi_valid:
                    base_confidence += rsi_boost
                    if rsi_boost > 0:
                        self.logger.debug(f"RSI zone boost: +{rsi_boost:.1%}")
                else:
                    # Signal blocked by RSI validation
                    return 0.0

            # 7. REJECTION CANDLE QUALITY (for bounce signals - up to 10% weight)
            if self.trend_validator:
                rejection_boost = self.trend_validator.calculate_rejection_candle_quality(latest_row, signal_type)
                base_confidence += rejection_boost
                if rejection_boost > 0:
                    self.logger.debug(f"Rejection candle boost: +{rejection_boost:.1%}")
            
            # Cap confidence at 95%
            final_confidence = min(0.95, base_confidence)
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating simple confidence: {e}")
            return 0.4  # Safe fallback
    
    def calculate_crossover_strength(self, latest_row: pd.Series, signal_type: str) -> float:
        """
        Calculate the strength of the EMA crossover
        
        Args:
            latest_row: DataFrame row with price and EMA data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            Strength value between 0.0 and 1.0
        """
        try:
            close = latest_row.get('close', 0)
            ema_short = latest_row.get('ema_short', 0)
            
            if close == 0 or ema_short == 0:
                return 0.0
            
            # Calculate distance from EMA as percentage
            distance = abs(close - ema_short) / close
            
            # Convert to strength (closer = weaker, farther = stronger)
            # But cap at reasonable values for forex (typically < 1% moves)
            max_distance = 0.01  # 1% maximum expected distance
            strength = min(1.0, distance / max_distance)
            
            # Validate direction
            if signal_type == 'BULL':
                # For bull signal, price should be above EMA
                if close <= ema_short:
                    return 0.0
            else:  # BEAR
                # For bear signal, price should be below EMA
                if close >= ema_short:
                    return 0.0
            
            return strength
            
        except Exception as e:
            self.logger.error(f"Error calculating crossover strength: {e}")
            return 0.0
    
    def calculate_trend_strength(self, latest_row: pd.Series) -> float:
        """
        Calculate overall trend strength based on EMA alignment
        
        Returns:
            Trend strength value between -1.0 (strong bear) and 1.0 (strong bull)
        """
        try:
            ema_short = latest_row.get('ema_short', 0)
            ema_long = latest_row.get('ema_long', 0)
            ema_trend = latest_row.get('ema_trend', 0)
            
            if ema_short == 0 or ema_long == 0 or ema_trend == 0:
                return 0.0
            
            # Calculate separations
            short_long_sep = (ema_short - ema_long) / ema_long
            long_trend_sep = (ema_long - ema_trend) / ema_trend
            
            # Average separation indicates trend strength
            avg_separation = (short_long_sep + long_trend_sep) / 2
            
            # Cap at reasonable values
            return max(-1.0, min(1.0, avg_separation * 100))
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.0
    
    def validate_confidence_threshold(self, confidence: float) -> bool:
        """
        Check if confidence meets minimum threshold

        Args:
            confidence: Calculated confidence score

        Returns:
            True if confidence meets threshold, False otherwise
        """
        if confidence < self.min_confidence:
            self.logger.debug(f"Signal confidence {confidence:.1%} below threshold {self.min_confidence:.1%}")
            return False
        return True

    def validate_swing_proximity(
        self,
        df: pd.DataFrame,
        current_price: float,
        signal_type: str,
        epic: str = None
    ) -> Dict[str, any]:
        """
        Validate swing proximity for signal entry timing (NEW)

        Args:
            df: DataFrame with price data and swing analysis
            current_price: Current market price
            signal_type: 'BULL' or 'BEAR'
            epic: Epic/symbol for pip calculation

        Returns:
            Dictionary with validation result and confidence adjustment
        """
        try:
            if not self.swing_validator:
                return {
                    'valid': True,
                    'confidence_penalty': 0.0,
                    'reason': 'Swing validator not configured'
                }

            # Call swing proximity validator
            result = self.swing_validator.validate_entry_proximity(
                df=df,
                current_price=current_price,
                direction=signal_type,
                epic=epic
            )

            if not result['valid']:
                self.logger.warning(
                    f"⚠️ Swing proximity violation: {result.get('rejection_reason', 'Unknown')}"
                )

            return {
                'valid': result['valid'],
                'confidence_penalty': result.get('confidence_penalty', 0.0),
                'reason': result.get('rejection_reason'),
                'swing_distance': result.get('distance_to_swing'),
                'swing_price': result.get('nearest_swing_price'),
                'swing_type': result.get('swing_type')
            }

        except Exception as e:
            self.logger.error(f"Swing proximity validation error: {e}")
            return {
                'valid': True,
                'confidence_penalty': 0.0,
                'reason': f'Validation error: {str(e)}'
            }

    # =========================================================================
    # SUPERTREND SIGNAL DETECTION AND CONFIDENCE CALCULATION
    # =========================================================================

    def detect_supertrend_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Supertrend confluence signals with ENHANCED LUXALGO-INSPIRED FILTERING

        Enhancements:
        1. Increased slow SuperTrend stability (12 bars vs 5 bars) - Option 2
        2. Performance-based filtering - Option 3
        3. Trend strength filtering - Option 5
        4. Adaptive clustering ready - Option 4

        Requires all 3 Supertrends to agree for a signal:
        - BULL: All 3 Supertrends are bullish (trend = 1)
        - BEAR: All 3 Supertrends are bearish (trend = -1)

        Args:
            df: DataFrame with Supertrend columns

        Returns:
            DataFrame with signal columns added
        """
        try:
            # Get Supertrend trends
            fast_trend = df['st_fast_trend']
            medium_trend = df['st_medium_trend']
            slow_trend = df['st_slow_trend']

            # Calculate confluence
            df['st_bullish_count'] = (
                (fast_trend == 1).astype(int) +
                (medium_trend == 1).astype(int) +
                (slow_trend == 1).astype(int)
            )

            df['st_bearish_count'] = (
                (fast_trend == -1).astype(int) +
                (medium_trend == -1).astype(int) +
                (slow_trend == -1).astype(int)
            )

            # Confluence percentage
            df['st_confluence_pct'] = df[['st_bullish_count', 'st_bearish_count']].max(axis=1) / 3 * 100

            # Get required confluence from config
            min_confluence = getattr(config_ema_strategy, 'SUPERTREND_MIN_CONFLUENCE', 3)

            # ✅ YOUTUBE STRATEGY: Add EMA 200 filter
            # Check if EMA 200 exists in dataframe
            has_ema_200 = 'ema_200' in df.columns or 'ema_trend' in df.columns
            ema_200_col = 'ema_200' if 'ema_200' in df.columns else 'ema_trend'

            # ✅ YOUTUBE STRATEGY: Detect ENTRY into full confluence
            # Signal only when we TRANSITION from not-all-agreeing to all-agreeing
            # This prevents continuous signals while already in confluence

            # Check if we HAD full confluence on previous candle
            prev_bullish_confluence = (df['st_bullish_count'].shift(1) >= min_confluence)
            prev_bearish_confluence = (df['st_bearish_count'].shift(1) >= min_confluence)

            # Check if we HAVE full confluence on current candle
            curr_bullish_confluence = (df['st_bullish_count'] >= min_confluence)
            curr_bearish_confluence = (df['st_bearish_count'] >= min_confluence)

            # NEW confluence entry = have it now but didn't have it before
            entering_bull_confluence = curr_bullish_confluence & ~prev_bullish_confluence
            entering_bear_confluence = curr_bearish_confluence & ~prev_bearish_confluence

            # ✅ OPTION 2: ENHANCED ANTI-CHOP FILTER - 12 bars stability (INCREASED from 5)
            # Require slow SuperTrend to be stable for 12 bars instead of 5
            # This dramatically reduces signals in choppy markets
            STABILITY_BARS = getattr(config_ema_strategy, 'SUPERTREND_STABILITY_BARS', 12)

            slow_stable_bull = (slow_trend == 1)
            slow_stable_bear = (slow_trend == -1)

            for i in range(1, STABILITY_BARS):
                slow_stable_bull = slow_stable_bull & (slow_trend.shift(i) == 1)
                slow_stable_bear = slow_stable_bear & (slow_trend.shift(i) == -1)

            # Apply stability filter to confluence entry
            entering_bull_confluence = entering_bull_confluence & slow_stable_bull
            entering_bear_confluence = entering_bear_confluence & slow_stable_bear

            # ✅ OPTION 3: PERFORMANCE-BASED FILTER (LuxAlgo inspired)
            # Only signal when recent SuperTrend performance is positive
            ENABLE_PERFORMANCE_FILTER = getattr(config_ema_strategy, 'SUPERTREND_PERFORMANCE_FILTER', True)
            PERFORMANCE_THRESHOLD = getattr(config_ema_strategy, 'SUPERTREND_PERFORMANCE_THRESHOLD', 0.0)

            if ENABLE_PERFORMANCE_FILTER:
                # Calculate rolling performance of fast SuperTrend
                st_trend = fast_trend.shift(1)  # Previous candle's trend
                price_change = df['close'].diff()  # Price movement
                raw_performance = st_trend * price_change  # Positive if trend correct

                # Apply exponential smoothing with 20-bar lookback
                PERFORMANCE_ALPHA = getattr(config_ema_strategy, 'SUPERTREND_PERFORMANCE_ALPHA', 0.1)
                df['st_performance'] = raw_performance.ewm(alpha=PERFORMANCE_ALPHA, min_periods=1).mean()

                # Filter signals with poor performance
                entering_bull_confluence = entering_bull_confluence & (df['st_performance'] > PERFORMANCE_THRESHOLD)
                entering_bear_confluence = entering_bear_confluence & (df['st_performance'] > PERFORMANCE_THRESHOLD)

                self.logger.debug(f"📊 Performance filter applied (threshold: {PERFORMANCE_THRESHOLD})")

            # ✅ OPTION 5: TREND STRENGTH FILTER
            # Filter out signals in choppy markets (when SuperTrends are too close)
            ENABLE_TREND_STRENGTH_FILTER = getattr(config_ema_strategy, 'SUPERTREND_TREND_STRENGTH_FILTER', True)
            MIN_TREND_STRENGTH_PCT = getattr(config_ema_strategy, 'SUPERTREND_MIN_TREND_STRENGTH', 0.3)

            if ENABLE_TREND_STRENGTH_FILTER:
                # Calculate separation between fast and slow SuperTrends as % of price
                st_fast = df['st_fast']
                st_slow = df['st_slow']
                close = df['close']

                df['trend_strength'] = abs(st_fast - st_slow) / close * 100

                # Filter signals in weak trends
                entering_bull_confluence = entering_bull_confluence & (df['trend_strength'] > MIN_TREND_STRENGTH_PCT)
                entering_bear_confluence = entering_bear_confluence & (df['trend_strength'] > MIN_TREND_STRENGTH_PCT)

                self.logger.debug(f"📊 Trend strength filter applied (min: {MIN_TREND_STRENGTH_PCT}%)")

            # Bull signal: Entering bullish confluence + Price above EMA 200
            if has_ema_200:
                df['bull_alert'] = (
                    entering_bull_confluence &
                    (df['close'] > df[ema_200_col])  # YouTube strategy: Price must be above EMA 200
                )
            else:
                df['bull_alert'] = entering_bull_confluence

            # Bear signal: Entering bearish confluence + Price below EMA 200
            if has_ema_200:
                df['bear_alert'] = (
                    entering_bear_confluence &
                    (df['close'] < df[ema_200_col])  # YouTube strategy: Price must be below EMA 200
                )
            else:
                df['bear_alert'] = entering_bear_confluence

            # Track flips for confidence scoring (legacy)
            df['st_fast_flip'] = fast_trend != fast_trend.shift(1)
            df['st_medium_flip'] = medium_trend != medium_trend.shift(1)
            df['st_slow_flip'] = slow_trend != slow_trend.shift(1)
            df['st_fresh_signal'] = entering_bull_confluence | entering_bear_confluence

            # Debug logging
            bull_signals = df['bull_alert'].sum()
            bear_signals = df['bear_alert'].sum()
            self.logger.info(
                f"🎯 Supertrend Signals Detected: BULL={bull_signals}, BEAR={bear_signals}"
            )

            return df

        except Exception as e:
            self.logger.error(f"Error detecting Supertrend signals: {e}", exc_info=True)
            # Add empty columns to avoid errors
            df['bull_alert'] = False
            df['bear_alert'] = False
            df['st_confluence_pct'] = 0
            return df

    def calculate_supertrend_confidence(
        self,
        latest_row: pd.Series,
        signal_type: str,
        mtf_4h_aligned: bool = False
    ) -> float:
        """
        Calculate confidence for Supertrend signals

        Args:
            latest_row: DataFrame row with Supertrend data
            signal_type: 'BULL' or 'BEAR'
            mtf_4h_aligned: Whether 4H Supertrend aligns with signal

        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            # Base confidence when all 3 Supertrends agree
            base = getattr(config_ema_strategy, 'SUPERTREND_CONFIDENCE_BASE', 0.60)
            confidence = base

            # Get confluence data
            bullish_count = latest_row.get('st_bullish_count', 0)
            bearish_count = latest_row.get('st_bearish_count', 0)
            confluence_pct = latest_row.get('st_confluence_pct', 0)

            # 1. FULL CONFLUENCE BONUS (35%)
            if confluence_pct == 100:  # All 3 agree
                full_bonus = getattr(config_ema_strategy, 'SUPERTREND_CONFIDENCE_FULL_CONFLUENCE', 0.35)
                confidence += full_bonus
                self.logger.debug(f"✅ Full confluence (3/3): +{full_bonus:.2f}")
            elif confluence_pct >= 66:  # 2/3 agree
                partial_bonus = getattr(config_ema_strategy, 'SUPERTREND_CONFIDENCE_PARTIAL_CONFLUENCE', 0.15)
                confidence += partial_bonus
                self.logger.debug(f"⚠️ Partial confluence (2/3): +{partial_bonus:.2f}")

            # 2. 4H MULTI-TIMEFRAME ALIGNMENT BONUS (10%)
            if mtf_4h_aligned:
                mtf_bonus = getattr(config_ema_strategy, 'SUPERTREND_CONFIDENCE_4H_ALIGNMENT', 0.10)
                confidence += mtf_bonus
                self.logger.debug(f"📊 4H alignment bonus: +{mtf_bonus:.2f}")

            # 3. FRESH SIGNAL BONUS (check if recent flip)
            fresh_signal = latest_row.get('st_fresh_signal', False)
            if fresh_signal:
                confidence += 0.05
                self.logger.debug("🆕 Fresh signal bonus: +0.05")

            # 4. ATR VOLATILITY CHECK (reduce confidence in extreme volatility)
            atr = latest_row.get('atr', 0)
            close = latest_row.get('close', 1)
            if atr > 0 and close > 0:
                atr_pct = (atr / close) * 100
                if atr_pct > 1.5:  # High volatility
                    volatility_penalty = 0.05
                    confidence -= volatility_penalty
                    self.logger.debug(f"⚠️ High volatility penalty: -{volatility_penalty:.2f}")

            # Cap confidence at 0.95
            confidence = min(confidence, 0.95)
            confidence = max(confidence, 0.0)

            self.logger.info(
                f"📊 Supertrend Confidence: {confidence:.2f} "
                f"(Confluence: {confluence_pct:.0f}%, 4H: {mtf_4h_aligned}, Fresh: {fresh_signal})"
            )

            return confidence

        except Exception as e:
            self.logger.error(f"Error calculating Supertrend confidence: {e}", exc_info=True)
            return 0.5  # Default medium confidence