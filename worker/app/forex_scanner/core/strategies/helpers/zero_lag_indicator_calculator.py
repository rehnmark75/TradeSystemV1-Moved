# core/strategies/helpers/zero_lag_indicator_calculator.py
"""
Zero Lag Indicator Calculator Module  
Handles Zero Lag EMA calculation and signal detection logic
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List
try:
    import config
except ImportError:
    from forex_scanner import config


class ZeroLagIndicatorCalculator:
    """Calculates Zero Lag indicators and detects crossover signals"""
    
    def __init__(self, logger: logging.Logger = None, eps: float = 1e-8):
        self.logger = logger or logging.getLogger(__name__)
        self.eps = eps  # Epsilon for stability
    
    def ensure_zero_lag_indicators(self, df: pd.DataFrame, length: int = 21, 
                                   band_multiplier: float = 2.0) -> pd.DataFrame:
        """
        Calculate Zero Lag EMA and related indicators
        
        Zero Lag EMA Formula: ZLEMA = EMA(src + (src - src[lag]), length)
        Where lag = (length - 1) / 2
        
        Args:
            df: DataFrame with price data
            length: Zero Lag EMA period
            band_multiplier: Volatility band multiplier
            
        Returns:
            DataFrame with Zero Lag indicators added
        """
        try:
            if df is None or df.empty:
                self.logger.debug("Empty DataFrame provided")
                return df
            
            if len(df) < length:
                self.logger.debug(f"Insufficient data: {len(df)} < {length}")
                return df
            
            df = df.copy()
            
            # Check if ZLEMA already exists
            if 'zlema' in df.columns:
                self.logger.debug("ZLEMA already present in DataFrame")
            else:
                # Calculate Zero Lag EMA
                df = self._calculate_zero_lag_ema(df, length)
            
            # Calculate volatility bands (pass length parameter)
            df = self._calculate_volatility_bands(df, band_multiplier, length)
            
            # Add trend state
            df = self._calculate_trend_state(df)
            
            # Add EMA 200 for macro trend filter
            if 'ema_200' not in df.columns:
                df['ema_200'] = df['close'].ewm(span=200).mean()

            # Add RSI for signal validation
            if 'rsi' not in df.columns:
                df = self._calculate_rsi(df, 14)  # Standard 14-period RSI

            self.logger.debug(f"Zero Lag indicators calculated for {len(df)} bars")
            return df
            
        except Exception as e:
            self.logger.error(f"Error ensuring Zero Lag indicators: {e}")
            return df
    
    def _calculate_zero_lag_ema(self, df: pd.DataFrame, length: int) -> pd.DataFrame:
        """
        Calculate Zero Lag EMA using EXACT Pine Script ta.ema() formula
        
        Pine Script Logic:
        lag = math.floor((length - 1) / 2)
        zlema = ta.ema(src + (src - src[lag]), length)
        
        Pine Script ta.ema() uses: alpha = 2 / (length + 1)
        """
        try:
            # Source price (close)
            src = df['close']
            
            # Calculate lag exactly as Pine Script: floor((length - 1) / 2)
            lag = int((length - 1) // 2)
            
            # Zero Lag EMA: EMA of (src + (src - src[lag]))
            lagged_src = src.shift(lag)
            momentum_adjustment = src - lagged_src
            zlema_input = src + momentum_adjustment
            
            # CRITICAL FIX: Use pandas ewm() which matches Pine Script ta.ema() exactly
            # Pine Script ta.ema() uses alpha = 2 / (length + 1)
            # pandas ewm(span=length, adjust=False) uses the same formula
            # The manual loop was causing initialization errors
            
            # Use pandas ewm with adjust=False for exact Pine Script match
            df['zlema'] = zlema_input.ewm(span=length, adjust=False).mean()
            
            # Forward fill NaN values from start
            df['zlema'] = df['zlema'].bfill().ffill()
            
            # Calculate alpha for debug logging (Pine Script formula)
            alpha = 2 / (length + 1)
            self.logger.debug(f"Zero Lag EMA calculated (Pine Script exact): length={length}, lag={lag}, alpha={alpha:.6f}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating Zero Lag EMA: {e}")
            return df
    
    def _calculate_volatility_bands(self, df: pd.DataFrame, multiplier: float, length: int = 21) -> pd.DataFrame:
        """
        Calculate volatility bands using Pine Script formula:
        volatility = ta.highest(ta.atr(length), length*3) * mult
        """
        try:
            if 'zlema' not in df.columns:
                return df
            
            # Use the provided length parameter (from Zero Lag EMA calculation)
            
            # Calculate ATR using Pine Script exact method
            # Pine Script: ta.atr(length) = ta.rma(ta.tr, length)
            # where ta.tr = max(high - low, abs(high - close[1]), abs(low - close[1]))
            
            prev_close = df['close'].shift(1)
            high_low = df['high'] - df['low']
            high_prev_close = (df['high'] - prev_close).abs()
            low_prev_close = (df['low'] - prev_close).abs()
            
            # True Range - Pine Script method
            true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
            
            # Pine Script ta.rma (RMA = Running Moving Average) is equivalent to EMA with alpha = 1/length
            # Use exact Pine Script formula: atr = ta.rma(tr, length) = ta.ema(tr, length) with different alpha
            atr_alpha = 1.0 / length
            atr = true_range.ewm(alpha=atr_alpha, adjust=False).mean()
            
            # Pine Script: ta.highest(ta.atr(length), length*3) * mult
            # Pine Script exact: ta.highest(ta.atr(length), length*3) * mult
            volatility_lookback = length * 3  # Restore Pine Script exact calculation
            volatility = atr.rolling(window=volatility_lookback).max() * multiplier
            
            # CRITICAL FIX: Handle insufficient data for volatility calculation
            total_bars_needed = length + volatility_lookback  # 70 + 210 = 280
            if len(df) < total_bars_needed:
                # Use fallback: simple ATR-based volatility for insufficient data
                self.logger.warning(f"Insufficient data for full volatility calculation ({len(df)} < {total_bars_needed}). Using fallback.")
                fallback_volatility = atr * multiplier
                volatility = fallback_volatility.bfill().ffill()
            else:
                volatility = volatility.bfill().ffill()
            
            # Final check: if still NaN, use close price * multiplier as emergency fallback
            if volatility.isna().all():
                emergency_volatility = df['close'].rolling(window=min(20, len(df)//2)).std() * multiplier
                volatility = emergency_volatility.bfill().ffill()
                self.logger.warning("Using emergency volatility calculation based on price standard deviation")
            
            # Log debug info
            non_nan_count = volatility.notna().sum()
            self.logger.debug(f"Volatility calculation: {non_nan_count}/{len(volatility)} non-NaN values (need {total_bars_needed} bars)")
            
            # Create bands
            df['upper_band'] = df['zlema'] + volatility
            df['lower_band'] = df['zlema'] - volatility
            df['volatility'] = volatility
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility bands: {e}")
            return df
    
    def _calculate_trend_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend state using Pine Script logic - COMPLETE REWRITE:
        
        var trend = 0
        if ta.crossover(close, zlema+volatility)
            trend := 1
        if ta.crossunder(close, zlema-volatility)
            trend := -1
        """
        try:
            if 'zlema' not in df.columns or 'volatility' not in df.columns:
                df['trend'] = 0
                return df
            
            # Calculate upper and lower bands
            upper_band = df['zlema'] + df['volatility']
            lower_band = df['zlema'] - df['volatility']
            
            df['upper_band'] = upper_band
            df['lower_band'] = lower_band
            
            # Initialize trend with Pine Script var logic
            # Start with -1 (bearish) as default, as most markets start below bands
            trend = pd.Series(-1, index=df.index, dtype=int)
            close = df['close']
            
            # DEBUG: Add detailed crossover detection
            bullish_crosses = []
            bearish_crosses = []
            
            # Process each bar (Pine Script simulation)
            for i in range(1, len(df)):
                curr_close = close.iloc[i]
                prev_close = close.iloc[i-1]
                curr_upper = upper_band.iloc[i]
                prev_upper = upper_band.iloc[i-1] 
                curr_lower = lower_band.iloc[i]
                prev_lower = lower_band.iloc[i-1]
                
                # Maintain previous trend as default
                trend.iloc[i] = trend.iloc[i-1]
                
                # CRITICAL FIX: Must be a GREEN candle crossing above upper band
                # Pine Script ta.crossover(close, upper_band) WITH candle color validation
                curr_open = df['open'].iloc[i] if 'open' in df.columns else curr_close
                is_green_candle = curr_close > curr_open  # Green candle: close > open
                is_red_candle = curr_close < curr_open    # Red candle: close < open
                
                # BULL: GREEN candle must close above upper band
                bullish_cross = (prev_close <= prev_upper) and (curr_close > curr_upper) and is_green_candle
                
                # BEAR: RED candle must close below lower band
                bearish_cross = (prev_close >= prev_lower) and (curr_close < curr_lower) and is_red_candle
                
                if bullish_cross:
                    trend.iloc[i] = 1
                    bullish_crosses.append(i)
                    # Enhanced debug with candle color and band movement info
                    band_diff = curr_upper - prev_upper
                    price_diff = curr_close - prev_close
                    self.logger.debug(f"🔺 BULLISH CROSS at {i}: GREEN candle close {curr_close:.5f} > upper {curr_upper:.5f}")
                    self.logger.debug(f"   Candle: Open={curr_open:.5f}, Close={curr_close:.5f} (GREEN)")
                    self.logger.debug(f"   Previous: close {prev_close:.5f} <= upper {prev_upper:.5f}")
                    self.logger.debug(f"   Band moved: {band_diff:.5f}, Price moved: {price_diff:.5f}")
                    
                elif bearish_cross:
                    trend.iloc[i] = -1
                    bearish_crosses.append(i)
                    # Enhanced debug with candle color and band movement info
                    band_diff = curr_lower - prev_lower
                    price_diff = curr_close - prev_close
                    self.logger.debug(f"🔻 BEARISH CROSS at {i}: RED candle close {curr_close:.5f} < lower {curr_lower:.5f}")
                    self.logger.debug(f"   Candle: Open={curr_open:.5f}, Close={curr_close:.5f} (RED)")
                    self.logger.debug(f"   Previous: close {prev_close:.5f} >= lower {prev_lower:.5f}")
                    self.logger.debug(f"   Band moved: {band_diff:.5f}, Price moved: {price_diff:.5f}")
                    
                # CRITICAL DEBUG: Show current trend determination for validation (debug level)
                if i >= len(df) - 5:  # Last 5 bars for debugging
                    trend_color = "🟢 GREEN" if trend.iloc[i] == 1 else "🔴 RED" if trend.iloc[i] == -1 else "⚪ NEUTRAL"
                    close_vs_bands = f"Close:{curr_close:.5f} vs Upper:{curr_upper:.5f} vs Lower:{curr_lower:.5f}"
                    self.logger.debug(f"TREND DEBUG [{i}]: {trend_color} | {close_vs_bands}")
            
            df['trend'] = trend
            
            # Add debug info
            self.logger.debug(f"Zero Lag trend calculation complete: {len(bullish_crosses)} bull crosses, {len(bearish_crosses)} bear crosses")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating trend state: {e}")
            df['trend'] = 0
            return df
    
    def detect_zero_lag_alerts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Zero Lag trend crossover signals (RIBBON COLOR CHANGES ONLY)
        
        Pine Script Logic:
        - BULL: trend crosses from -1 to 1 (ribbon changes from RED to GREEN)
        - BEAR: trend crosses from 1 to -1 (ribbon changes from GREEN to RED)
        
        This is MUCH more selective - only triggers when the ribbon color changes!
        """
        try:
            if df is None or df.empty or len(df) < 2:
                return df
            
            # Sort by time and reset index
            df = df.sort_values('start_time').reset_index(drop=True)
            
            # Initialize alert columns
            df['bull_alert'] = False
            df['bear_alert'] = False
            
            # Get previous trend for crossover detection
            prev_trend = df['trend'].shift(1)
            current_trend = df['trend']
            
            # TREND CROSSOVERS ONLY (Pine Script logic)
            # Bull alert: trend crosses from -1/0 to 1 (ribbon turns GREEN)
            df['bull_alert'] = (prev_trend <= 0) & (current_trend == 1)
            
            # Bear alert: trend crosses from 1/0 to -1 (ribbon turns RED)
            df['bear_alert'] = (prev_trend >= 0) & (current_trend == -1)
            
            # Add debug information
            if df['bull_alert'].any() or df['bear_alert'].any():
                bull_count = df['bull_alert'].sum()
                bear_count = df['bear_alert'].sum()
                self.logger.debug(f"Zero Lag crossovers detected: {bull_count} BULL, {bear_count} BEAR")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in Zero Lag alert detection: {e}")
            return df
    
    def validate_ema200_trend_filter(self, row: pd.Series, signal_type: str) -> bool:
        """
        EMA 200 Trend Filter: Ensure signals align with major trend direction
        
        Args:
            row: DataFrame row with price and EMA data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            True if trend is correct for signal type
        """
        try:
            if not getattr(config, 'EMA_200_TREND_FILTER_ENABLED', True):
                return True
            
            ema_200 = row.get('ema_200', 0)
            close_price = row.get('close', 0)
            
            if ema_200 == 0 or close_price == 0:
                self.logger.debug("EMA 200 or close price not available")
                return True  # Allow signal if data not available
            
            # Buffer for noise reduction (1 pip buffer)
            buffer_pips = getattr(config, 'EMA_200_BUFFER_PIPS', 1.0)
            if close_price > 50:  # Likely JPY pair
                pip_multiplier = 100
            else:  # Standard pair
                pip_multiplier = 10000
            
            buffer_distance = buffer_pips / pip_multiplier
            
            if signal_type == 'BULL':
                # Bull signals: price must be above EMA 200
                if close_price > ema_200 + buffer_distance:
                    distance_above = (close_price - ema_200) * pip_multiplier
                    self.logger.debug(f"EMA 200 trend OK for BULL: price {distance_above:.1f} pips above")
                    return True
                else:
                    distance_below = (ema_200 - close_price) * pip_multiplier
                    self.logger.info(f"EMA 200 trend INVALID for BULL: price {distance_below:.1f} pips below")
                    return False
            
            elif signal_type == 'BEAR':
                # Bear signals: price must be below EMA 200
                if close_price < ema_200 - buffer_distance:
                    distance_below = (ema_200 - close_price) * pip_multiplier
                    self.logger.debug(f"EMA 200 trend OK for BEAR: price {distance_below:.1f} pips below")
                    return True
                else:
                    distance_above = (close_price - ema_200) * pip_multiplier
                    self.logger.info(f"EMA 200 trend INVALID for BEAR: price {distance_above:.1f} pips above")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating EMA 200 trend: {e}")
            return True  # Allow signal on error
    
    def get_required_indicators(self, length: int = 21) -> List[str]:
        """
        Get list of required indicators for Zero Lag strategy
        
        Args:
            length: Zero Lag EMA period
            
        Returns:
            List of required indicator column names
        """
        base_indicators = [
            'zlema',
            f'ema_200',
            'close',
            'open',
            'high',
            'low',
            'start_time',
            'upper_band',
            'lower_band',
            'volatility',
            'trend',
            'zlema_slope',
            'rsi'
        ]
        
        # Add ATR if available
        base_indicators.append('atr')
        
        # Add Squeeze Momentum indicators if enabled
        if getattr(config, 'SQUEEZE_MOMENTUM_ENABLED', True):
            base_indicators.extend([
                'squeeze_momentum',
                'squeeze_state',
                'squeeze_is_lime',
                'squeeze_is_green', 
                'squeeze_is_red',
                'squeeze_is_maroon',
                'squeeze_bullish',
                'squeeze_bearish',
                'squeeze_on',
                'squeeze_off'
            ])
        
        return base_indicators
    
    def validate_data_requirements(self, df: pd.DataFrame, min_bars: int) -> bool:
        """
        Validate that DataFrame meets minimum requirements
        
        Args:
            df: DataFrame to validate
            min_bars: Minimum number of bars required
            
        Returns:
            True if requirements met, False otherwise
        """
        if df is None or df.empty:
            self.logger.debug("DataFrame is None or empty")
            return False
        
        if len(df) < min_bars:
            self.logger.debug(f"Insufficient data: {len(df)} < {min_bars}")
            return False
        
        # Check for required columns
        required_cols = ['close', 'open', 'high', 'low']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            return False

        return True

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate RSI (Relative Strength Index) using the standard formula

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss

        Args:
            df: DataFrame with price data
            period: RSI period (default 14)

        Returns:
            DataFrame with RSI added
        """
        try:
            if len(df) < period + 1:
                self.logger.debug(f"Insufficient data for RSI calculation: {len(df)} < {period + 1}")
                df['rsi'] = 50.0  # Neutral RSI value
                return df

            # Calculate price changes
            close = df['close']
            delta = close.diff()

            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)

            # Calculate average gains and losses using EMA (Wilder's smoothing)
            # Wilder's smoothing uses alpha = 1/period
            alpha = 1.0 / period
            avg_gains = gains.ewm(alpha=alpha, adjust=False).mean()
            avg_losses = losses.ewm(alpha=alpha, adjust=False).mean()

            # Calculate RS and RSI
            rs = avg_gains / avg_losses.replace(0, np.finfo(float).eps)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))

            # Handle edge cases
            rsi = rsi.fillna(50.0)  # Fill NaN with neutral value
            rsi = rsi.clip(0, 100)  # Ensure RSI stays in 0-100 range

            df['rsi'] = rsi

            self.logger.debug(f"RSI calculated with period {period}")
            return df

        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            df['rsi'] = 50.0  # Fallback to neutral RSI
            return df

    def _calculate_simple_sr_levels(self, df: pd.DataFrame, lookback: int = 100) -> Dict:
        """
        Calculate support/resistance levels for LIVE TRADING & BACKTESTING

        REALISTIC TRADING LOGIC: Only look LEFT (historical data)
        - Find local highs that are higher than N bars to the left
        - Find local lows that are lower than N bars to the left
        - Include recent highs/lows as primary levels (most relevant)

        This works correctly for BOTH:
        - Live trading: We don't have future bars
        - Backtesting: Each bar only sees historical data (bar-by-bar simulation)

        Args:
            df: DataFrame with OHLC data (up to current bar only)
            lookback: Number of bars to look back (default 100)

        Returns:
            Dictionary with support and resistance levels
        """
        try:
            if len(df) < 20:
                return {'support': [], 'resistance': []}

            # Use last N bars for S/R calculation
            recent_df = df.tail(min(lookback, len(df)))

            # Number of bars to look LEFT for pivot detection
            left_bars = 10

            # Find swing highs (resistance) - only check LEFT side
            resistance_levels = []

            # Start from left_bars to end (we can always look left)
            for i in range(left_bars, len(recent_df)):
                high = recent_df.iloc[i]['high']

                # Check if this is higher than the previous left_bars
                is_swing_high = True
                for j in range(i - left_bars, i):
                    if recent_df.iloc[j]['high'] >= high:
                        is_swing_high = False
                        break

                if is_swing_high:
                    resistance_levels.append(high)

            # Find swing lows (support) - only check LEFT side
            support_levels = []

            # Start from left_bars to end (we can always look left)
            for i in range(left_bars, len(recent_df)):
                low = recent_df.iloc[i]['low']

                # Check if this is lower than the previous left_bars
                is_swing_low = True
                for j in range(i - left_bars, i):
                    if recent_df.iloc[j]['low'] <= low:
                        is_swing_low = False
                        break

                if is_swing_low:
                    support_levels.append(low)

            # CRITICAL: Cluster nearby levels to avoid too many S/R levels
            # Merge levels within 0.2% of each other (they're essentially the same level)
            def cluster_levels(levels, tolerance=0.002):
                if not levels:
                    return []

                sorted_levels = sorted(levels)
                clustered = []
                current_cluster = [sorted_levels[0]]

                for level in sorted_levels[1:]:
                    # If within tolerance of current cluster, add to cluster
                    if abs(level - current_cluster[0]) / current_cluster[0] < tolerance:
                        current_cluster.append(level)
                    else:
                        # Save average of cluster and start new cluster
                        clustered.append(sum(current_cluster) / len(current_cluster))
                        current_cluster = [level]

                # Don't forget the last cluster
                if current_cluster:
                    clustered.append(sum(current_cluster) / len(current_cluster))

                return clustered

            # Cluster nearby levels
            resistance_levels = cluster_levels(resistance_levels)
            support_levels = cluster_levels(support_levels)

            # ALWAYS include the most recent high and low (last 20 bars)
            # These are the most relevant for immediate trading decisions
            recent_bars = min(20, len(recent_df))
            recent_high = recent_df.tail(recent_bars)['high'].max()
            recent_low = recent_df.tail(recent_bars)['low'].min()

            # Add recent high if not already in list (within 0.2% tolerance)
            add_recent_high = True
            for r in resistance_levels:
                if abs(recent_high - r) / recent_high < 0.002:
                    add_recent_high = False
                    break
            if add_recent_high:
                resistance_levels.append(recent_high)

            # Add recent low if not already in list (within 0.2% tolerance)
            add_recent_low = True
            for s in support_levels:
                if abs(recent_low - s) / recent_low < 0.002:
                    add_recent_low = False
                    break
            if add_recent_low:
                support_levels.append(recent_low)

            # Keep only the most significant levels (max 5 each)
            resistance_levels = sorted(resistance_levels)[-5:]
            support_levels = sorted(support_levels)[:5]  # Take lowest 5 support levels

            self.logger.debug(f"S/R levels found: {len(resistance_levels)} resistance, {len(support_levels)} support")

            return {
                'support': support_levels,
                'resistance': resistance_levels
            }

        except Exception as e:
            self.logger.error(f"Error calculating S/R levels: {e}")
            return {'support': [], 'resistance': []}

    def validate_sr_proximity(self, df: pd.DataFrame, current_price: float, signal_type: str, epic: str) -> tuple:
        """
        Validate if signal is too close to support/resistance levels

        Args:
            df: DataFrame with OHLC data
            current_price: Current price to check
            signal_type: 'BULL' or 'BEAR'
            epic: Trading pair epic

        Returns:
            (is_valid, reason, details) tuple
        """
        try:
            # Calculate S/R levels
            sr_levels = self._calculate_simple_sr_levels(df, lookback=200)

            if not sr_levels['support'] and not sr_levels['resistance']:
                return True, "No S/R levels found", {}

            # Calculate volatility-aware minimum distance
            pip_size = 0.01 if 'JPY' in epic.upper() else 0.0001

            # Get multiplier from config
            try:
                from configdata.strategies import config_zerolag_strategy
                multiplier = getattr(config_zerolag_strategy, 'ZERO_LAG_SR_MIN_DISTANCE_MULTIPLIER', 1.0)
            except ImportError:
                multiplier = 1.0

            # Get ATR-based minimum distance
            if 'atr' in df.columns and len(df) > 0:
                recent_atr = df['atr'].iloc[-1]
                atr_pips = (recent_atr / pip_size) if recent_atr and recent_atr > 0 else 0
                # More lenient: 1.0x ATR or 8 pips minimum
                min_distance_pips = max(8.0, atr_pips * multiplier)
            else:
                min_distance_pips = 8.0  # Fallback to 8 pips (reduced from 10)

            # Check signal type specific validation
            if signal_type == 'BULL':
                # BUY signals: Check resistance ABOVE current price
                # Reject if too close to resistance that could block upward movement
                for resistance in sr_levels['resistance']:
                    if resistance > current_price:
                        distance_pips = (resistance - current_price) / pip_size
                        if distance_pips < min_distance_pips:
                            reason = f"Too close to resistance above: {distance_pips:.1f} pips (min: {min_distance_pips:.1f})"
                            details = {
                                'resistance_level': resistance,
                                'distance_pips': distance_pips,
                                'min_required': min_distance_pips
                            }
                            return False, reason, details

            else:  # BEAR
                # SELL signals: Check BOTH resistance above AND support below
                # 1. Check if too close to resistance ABOVE (price might bounce back up)
                for resistance in sr_levels['resistance']:
                    if resistance > current_price:
                        distance_pips = (resistance - current_price) / pip_size
                        if distance_pips < min_distance_pips:
                            reason = f"Too close to resistance above: {distance_pips:.1f} pips (min: {min_distance_pips:.1f})"
                            details = {
                                'resistance_level': resistance,
                                'distance_pips': distance_pips,
                                'min_required': min_distance_pips
                            }
                            return False, reason, details

                # 2. Check if too close to support BELOW (might block downward movement)
                for support in sr_levels['support']:
                    if support < current_price:
                        distance_pips = (current_price - support) / pip_size
                        if distance_pips < min_distance_pips:
                            reason = f"Too close to support below: {distance_pips:.1f} pips (min: {min_distance_pips:.1f})"
                            details = {
                                'support_level': support,
                                'distance_pips': distance_pips,
                                'min_required': min_distance_pips
                            }
                            return False, reason, details

            return True, "Clear of S/R levels", {}

        except Exception as e:
            self.logger.error(f"Error validating S/R proximity: {e}")
            return True, "Error in S/R validation", {}  # Allow signal on error