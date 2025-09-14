# core/strategies/helpers/macd_indicator_calculator.py
"""
MACD Indicator Calculator Module
Handles MACD calculations, crossover detection and data validation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional


class MACDIndicatorCalculator:
    """Calculates MACD indicators and detects crossovers - lightweight and focused"""
    
    def __init__(self, logger: logging.Logger = None, eps: float = 1e-8):
        self.logger = logger or logging.getLogger(__name__)
        self.eps = eps
        
        # Default MACD parameters (can be overridden)
        self.default_config = {
            'fast_ema': 12,
            'slow_ema': 26, 
            'signal_ema': 9
        }
    
    def get_required_indicators(self, macd_config: Dict = None) -> List[str]:
        """Get list of required indicators for MACD strategy"""
        config = macd_config or self.default_config
        return [
            f'ema_{config["fast_ema"]}',
            f'ema_{config["slow_ema"]}',
            'macd_line',
            'macd_signal', 
            'macd_histogram',
            'ema_200'  # For trend filter
        ]
    
    def validate_data_requirements(self, df: pd.DataFrame, min_bars: int = 50) -> bool:
        """Validate that we have enough data for MACD calculations"""
        if len(df) < min_bars:
            self.logger.debug(f"Insufficient data: {len(df)} bars (need {min_bars})")
            return False
            
        required_columns = ['open', 'high', 'low', 'close']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            return False
            
        return True
    
    def ensure_macd_indicators(self, df: pd.DataFrame, macd_config: Dict = None) -> pd.DataFrame:
        """
        Calculate MACD indicators if not present
        
        Args:
            df: DataFrame with price data
            macd_config: MACD configuration (fast_ema, slow_ema, signal_ema)
            
        Returns:
            DataFrame with MACD indicators added
        """
        config = macd_config or self.default_config
        df_copy = df.copy()
        
        try:
            # Calculate EMAs needed for MACD
            fast_ema_col = f'ema_{config["fast_ema"]}'
            slow_ema_col = f'ema_{config["slow_ema"]}'
            
            if fast_ema_col not in df_copy.columns:
                df_copy[fast_ema_col] = df_copy['close'].ewm(span=config['fast_ema']).mean()
                
            if slow_ema_col not in df_copy.columns:
                df_copy[slow_ema_col] = df_copy['close'].ewm(span=config['slow_ema']).mean()
            
            # Calculate MACD line (fast EMA - slow EMA)
            if 'macd_line' not in df_copy.columns:
                df_copy['macd_line'] = df_copy[fast_ema_col] - df_copy[slow_ema_col]
            
            # Calculate MACD signal line (EMA of MACD line)
            if 'macd_signal' not in df_copy.columns:
                df_copy['macd_signal'] = df_copy['macd_line'].ewm(span=config['signal_ema']).mean()
                
            # Calculate MACD histogram (MACD line - signal line)
            if 'macd_histogram' not in df_copy.columns:
                df_copy['macd_histogram'] = df_copy['macd_line'] - df_copy['macd_signal']
            
            # Add EMA 200 for trend filter if not present
            if 'ema_200' not in df_copy.columns:
                df_copy['ema_200'] = df_copy['close'].ewm(span=200).mean()
            
            # Add enhanced filters: ADX, ATR, and RSI
            df_copy = self._add_enhanced_filters(df_copy)
            
            # Add MACD divergence detection for high-quality signals
            df_copy = self.detect_macd_divergence(df_copy)
            
            self.logger.debug("MACD indicators, enhanced filters, and divergence detection completed successfully")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD indicators: {e}")
            return df_copy
    
    def detect_macd_crossovers(self, df: pd.DataFrame, epic: str = '') -> pd.DataFrame:
        """
        Detect MACD histogram crossovers with strength filtering (main signal generation)
        
        Args:
            df: DataFrame with MACD indicators
            epic: Trading pair epic for strength threshold determination
            
        Returns:
            DataFrame with crossover signals added
        """
        df_copy = df.copy()
        
        try:
            # Ensure we have MACD data
            required_cols = ['macd_line', 'macd_signal', 'macd_histogram']
            if not all(col in df_copy.columns for col in required_cols):
                self.logger.error("Missing MACD indicators for crossover detection")
                return df_copy
            
            # Initialize signal columns
            df_copy['bull_crossover'] = False
            df_copy['bear_crossover'] = False
            df_copy['bull_alert'] = False
            df_copy['bear_alert'] = False
            
            # Detect histogram crossovers (zero line crosses)
            df_copy['histogram_prev'] = df_copy['macd_histogram'].shift(1)
            
            # Determine base strength threshold and apply volatility adjustment
            base_threshold = self.get_histogram_strength_threshold(epic)
            
            # Apply volatility-adjusted thresholds (2-3x higher for restrictive filtering)
            strength_threshold = self.get_enhanced_threshold(df_copy, epic, base_threshold)
            
            # ENHANCED RESTRICTIVE CONDITIONS: Multiple filters + divergence preference
            
            # Prioritize divergence signals (highest quality)
            bull_divergence_signal = (
                # MACD bullish divergence detected
                df_copy['bullish_divergence'] &
                
                # Basic crossover still required
                (df_copy['macd_histogram'] > 0) & 
                (df_copy['histogram_prev'] <= 0) &
                
                # Relaxed threshold for divergence signals (higher quality)
                (df_copy['macd_histogram'] >= base_threshold) &
                
                # ADX trending requirement (lower threshold for divergence)
                (df_copy['adx'] > 20) &
                
                # RSI confluence
                (df_copy['rsi'] < 70)
            )
            
            # Regular crossover signals (more restrictive)
            bull_regular_signal = (
                # No divergence detected - require stronger conditions
                ~df_copy['bullish_divergence'] &
                
                # Basic MACD crossover with enhanced threshold
                (df_copy['macd_histogram'] > 0) & 
                (df_copy['histogram_prev'] <= 0) &
                (df_copy['macd_histogram'] >= strength_threshold) &
                
                # Stronger ADX requirement for regular signals
                (df_copy['adx'] > 30) &
                
                # RSI and trend alignment
                (df_copy['rsi'] < 65) &
                (df_copy['close'] > df_copy['ema_200'])
            )
            
            # Combine divergence and regular signals
            bull_cross = bull_divergence_signal | bull_regular_signal
            
            # BEAR SIGNALS: Similar logic
            bear_divergence_signal = (
                df_copy['bearish_divergence'] &
                (df_copy['macd_histogram'] < 0) & 
                (df_copy['histogram_prev'] >= 0) &
                (df_copy['macd_histogram'] <= -base_threshold) &
                (df_copy['adx'] > 20) &
                (df_copy['rsi'] > 30)
            )
            
            bear_regular_signal = (
                ~df_copy['bearish_divergence'] &
                (df_copy['macd_histogram'] < 0) & 
                (df_copy['histogram_prev'] >= 0) &
                (df_copy['macd_histogram'] <= -strength_threshold) &
                (df_copy['adx'] > 30) &
                (df_copy['rsi'] > 35) &
                (df_copy['close'] < df_copy['ema_200'])
            )
            
            bear_cross = bear_divergence_signal | bear_regular_signal
            
            df_copy['bull_crossover'] = bull_cross
            df_copy['bear_crossover'] = bear_cross
            
            # For compatibility, also set alert flags
            df_copy['bull_alert'] = bull_cross
            df_copy['bear_alert'] = bear_cross
            
            # Log crossover detection with strength filtering info
            bull_count = bull_cross.sum()
            bear_count = bear_cross.sum()
            if bull_count > 0 or bear_count > 0:
                self.logger.debug(f"MACD crossovers detected - Bull: {bull_count}, Bear: {bear_count} (threshold: {strength_threshold})")
            
            # Log filtered signals if any crossovers were filtered out
            raw_bull = ((df_copy['macd_histogram'] > 0) & (df_copy['histogram_prev'] <= 0)).sum()
            raw_bear = ((df_copy['macd_histogram'] < 0) & (df_copy['histogram_prev'] >= 0)).sum()
            if raw_bull > bull_count or raw_bear > bear_count:
                self.logger.debug(f"Strength filter removed {raw_bull - bull_count} bull + {raw_bear - bear_count} bear signals")
            
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error detecting MACD crossovers: {e}")
            return df_copy
    
    def get_histogram_strength_threshold(self, epic: str) -> float:
        """
        Get MACD histogram strength threshold based on currency pair
        
        JPY pairs: 0.010 minimum histogram strength (calibrated from actual data)
        Non-JPY pairs: 0.00010 minimum histogram strength (calibrated from actual data)
        
        Args:
            epic: Trading pair epic (e.g., 'CS.D.USDJPY.MINI.IP')
            
        Returns:
            Minimum histogram strength threshold
        """
        try:
            # Check if this is a JPY pair
            if 'JPY' in epic.upper():
                threshold = 0.010  # JPY pairs - calibrated from actual market data
                self.logger.debug(f"Using JPY histogram threshold: {threshold}")
            else:
                threshold = 0.00010  # Non-JPY pairs - calibrated from actual market data
                self.logger.debug(f"Using non-JPY histogram threshold: {threshold}")
            
            return threshold
            
        except Exception as e:
            self.logger.error(f"Error determining histogram threshold for {epic}: {e}")
            return 0.00010  # Default to non-JPY threshold
    
    def get_enhanced_threshold(self, df: pd.DataFrame, epic: str, base_threshold: float) -> float:
        """
        Calculate volatility-adjusted threshold for restrictive signal filtering
        
        Uses ATR to dynamically adjust thresholds based on market volatility:
        - High volatility = Higher thresholds (more restrictive)
        - Low volatility = Slightly lower thresholds
        - Always applies 2-3x multiplier for restrictive filtering
        
        Args:
            df: DataFrame with ATR data
            epic: Trading pair epic
            base_threshold: Base histogram strength threshold
            
        Returns:
            Enhanced threshold adjusted for volatility and restrictiveness
        """
        try:
            # Get recent ATR for volatility measurement
            if 'atr' in df.columns and len(df) > 0:
                recent_atr = df['atr'].tail(20).mean()  # 20-period average ATR
                
                # Normalize ATR (typical forex ATR ranges 0.0001-0.01)
                if 'JPY' in epic.upper():
                    # JPY pairs have higher ATR values
                    atr_multiplier = min(3.0, max(1.5, recent_atr / 0.005))
                else:
                    # Major pairs ATR normalization
                    atr_multiplier = min(3.0, max(1.5, recent_atr / 0.001))
                
                self.logger.debug(f"ATR volatility multiplier for {epic}: {atr_multiplier:.2f}")
            else:
                atr_multiplier = 2.0  # Default multiplier if ATR unavailable
            
            # Apply restrictive multiplier (2-3x base threshold)
            restrictive_multiplier = 2.5  # Base 2.5x more restrictive
            
            # Combine volatility and restrictive adjustments
            enhanced_threshold = base_threshold * restrictive_multiplier * atr_multiplier
            
            self.logger.debug(f"Enhanced threshold for {epic}: {enhanced_threshold:.6f} "
                            f"(base: {base_threshold:.6f}, restrictive: {restrictive_multiplier}x, "
                            f"volatility: {atr_multiplier:.2f}x)")
            
            return enhanced_threshold
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced threshold for {epic}: {e}")
            return base_threshold * 2.5  # Fallback to 2.5x base threshold
    
    def validate_macd_strength(self, row: pd.Series, signal_type: str, threshold: float = 0.0001) -> bool:
        """
        Validate MACD signal strength
        
        Args:
            row: DataFrame row with MACD data
            signal_type: 'BULL' or 'BEAR'
            threshold: Minimum histogram value for valid signal
            
        Returns:
            True if signal is strong enough
        """
        try:
            histogram = row.get('macd_histogram', 0)
            
            if signal_type == 'BULL':
                # Bull signals need positive histogram above threshold
                return histogram > threshold
            elif signal_type == 'BEAR':
                # Bear signals need negative histogram below negative threshold  
                return histogram < -threshold
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating MACD strength: {e}")
            return False
    
    def get_macd_momentum_score(self, row: pd.Series) -> float:
        """
        Calculate MACD momentum score (0.0 to 1.0)
        
        Args:
            row: DataFrame row with MACD data
            
        Returns:
            Momentum score between 0.0 and 1.0
        """
        try:
            histogram = row.get('macd_histogram', 0)
            macd_line = row.get('macd_line', 0)
            macd_signal = row.get('macd_signal', 0)
            
            # Base score from histogram strength
            hist_score = min(1.0, abs(histogram) / 0.001) if histogram != 0 else 0.0
            
            # Bonus for MACD line and signal alignment
            line_signal_aligned = (
                (macd_line > macd_signal and histogram > 0) or
                (macd_line < macd_signal and histogram < 0)
            )
            alignment_bonus = 0.2 if line_signal_aligned else 0.0
            
            return min(1.0, hist_score + alignment_bonus)
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum score: {e}")
            return 0.5  # Neutral score on error
    
    def _add_enhanced_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add enhanced filters: ADX, ATR, and RSI for restrictive signal generation
        
        These filters help reduce false signals and improve quality:
        - ADX: Measures trend strength (avoid choppy markets)
        - ATR: Measures volatility (dynamic threshold adjustment) 
        - RSI: Identifies overbought/oversold conditions
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with enhanced filter indicators
        """
        try:
            df_enhanced = df.copy()
            
            # 1. ADD ADX (Average Directional Index) for trend strength
            if 'adx' not in df_enhanced.columns:
                df_enhanced = self._calculate_adx(df_enhanced)
            
            # 2. ADD ATR (Average True Range) for volatility measurement
            if 'atr' not in df_enhanced.columns:
                df_enhanced = self._calculate_atr(df_enhanced)
            
            # 3. ADD RSI (Relative Strength Index) for overbought/oversold
            if 'rsi' not in df_enhanced.columns:
                df_enhanced = self._calculate_rsi(df_enhanced)
            
            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"Error adding enhanced filters: {e}")
            return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX (Average Directional Index) for trend strength measurement"""
        try:
            df_copy = df.copy()
            
            # Calculate True Range components
            df_copy['high_minus_low'] = df_copy['high'] - df_copy['low']
            df_copy['high_minus_close_prev'] = abs(df_copy['high'] - df_copy['close'].shift(1))
            df_copy['low_minus_close_prev'] = abs(df_copy['low'] - df_copy['close'].shift(1))
            
            # True Range is the maximum of the three
            df_copy['true_range'] = df_copy[['high_minus_low', 'high_minus_close_prev', 'low_minus_close_prev']].max(axis=1)
            
            # Calculate directional movements
            df_copy['high_diff'] = df_copy['high'].diff()
            df_copy['low_diff'] = -df_copy['low'].diff()
            
            # Positive and Negative Directional Movement
            df_copy['plus_dm'] = np.where(
                (df_copy['high_diff'] > df_copy['low_diff']) & (df_copy['high_diff'] > 0),
                df_copy['high_diff'], 0
            )
            df_copy['minus_dm'] = np.where(
                (df_copy['low_diff'] > df_copy['high_diff']) & (df_copy['low_diff'] > 0),
                df_copy['low_diff'], 0
            )
            
            # Smooth the True Range and Directional Movements
            alpha = 1.0 / period
            df_copy['tr_smooth'] = df_copy['true_range'].ewm(alpha=alpha, adjust=False).mean()
            df_copy['plus_dm_smooth'] = df_copy['plus_dm'].ewm(alpha=alpha, adjust=False).mean()
            df_copy['minus_dm_smooth'] = df_copy['minus_dm'].ewm(alpha=alpha, adjust=False).mean()
            
            # Calculate Directional Indicators
            df_copy['plus_di'] = 100 * (df_copy['plus_dm_smooth'] / df_copy['tr_smooth'])
            df_copy['minus_di'] = 100 * (df_copy['minus_dm_smooth'] / df_copy['tr_smooth'])
            
            # Calculate DX (Directional Movement Index)
            df_copy['dx'] = 100 * abs(df_copy['plus_di'] - df_copy['minus_di']) / (df_copy['plus_di'] + df_copy['minus_di'])
            
            # ADX is the smoothed average of DX
            df_copy['adx'] = df_copy['dx'].ewm(alpha=alpha, adjust=False).mean()
            
            # Clean up temporary columns
            temp_cols = ['high_minus_low', 'high_minus_close_prev', 'low_minus_close_prev', 
                        'high_diff', 'low_diff', 'plus_dm', 'minus_dm', 'tr_smooth',
                        'plus_dm_smooth', 'minus_dm_smooth', 'dx']
            df_copy = df_copy.drop(columns=temp_cols, errors='ignore')
            
            self.logger.debug(f"ADX calculated successfully (period: {period})")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ATR (Average True Range) for volatility measurement"""
        try:
            df_copy = df.copy()
            
            # True Range calculation
            df_copy['high_low'] = df_copy['high'] - df_copy['low']
            df_copy['high_close'] = abs(df_copy['high'] - df_copy['close'].shift(1))
            df_copy['low_close'] = abs(df_copy['low'] - df_copy['close'].shift(1))
            
            df_copy['true_range'] = df_copy[['high_low', 'high_close', 'low_close']].max(axis=1)
            
            # ATR is the smoothed average of True Range
            df_copy['atr'] = df_copy['true_range'].ewm(span=period, adjust=False).mean()
            
            # Clean up temporary columns
            df_copy = df_copy.drop(columns=['high_low', 'high_close', 'low_close', 'true_range'], errors='ignore')
            
            self.logger.debug(f"ATR calculated successfully (period: {period})")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate RSI (Relative Strength Index) for overbought/oversold detection"""
        try:
            df_copy = df.copy()
            
            # Calculate price changes
            df_copy['price_change'] = df_copy['close'].diff()
            
            # Separate gains and losses
            df_copy['gain'] = df_copy['price_change'].where(df_copy['price_change'] > 0, 0)
            df_copy['loss'] = -df_copy['price_change'].where(df_copy['price_change'] < 0, 0)
            
            # Calculate average gains and losses using EMA
            alpha = 1.0 / period
            df_copy['avg_gain'] = df_copy['gain'].ewm(alpha=alpha, adjust=False).mean()
            df_copy['avg_loss'] = df_copy['loss'].ewm(alpha=alpha, adjust=False).mean()
            
            # Calculate RS (Relative Strength) and RSI
            df_copy['rs'] = df_copy['avg_gain'] / df_copy['avg_loss']
            df_copy['rsi'] = 100 - (100 / (1 + df_copy['rs']))
            
            # Handle division by zero
            df_copy['rsi'] = df_copy['rsi'].fillna(50)  # Neutral RSI when avg_loss is 0
            
            # Clean up temporary columns
            temp_cols = ['price_change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs']
            df_copy = df_copy.drop(columns=temp_cols, errors='ignore')
            
            self.logger.debug(f"RSI calculated successfully (period: {period})")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return df
    
    def detect_macd_divergence(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """
        Detect MACD divergence signals (advanced analysis for high-quality signals)
        
        Divergence Types:
        - Bullish Regular: Price lower lows, MACD higher lows (potential upward reversal)
        - Bearish Regular: Price higher highs, MACD lower highs (potential downward reversal) 
        - Bullish Hidden: Price higher lows, MACD lower lows (trend continuation up)
        - Bearish Hidden: Price lower highs, MACD higher highs (trend continuation down)
        
        Args:
            df: DataFrame with price and MACD data
            lookback: Number of bars to look back for divergence analysis
            
        Returns:
            DataFrame with divergence signals added
        """
        try:
            df_div = df.copy()
            
            # Initialize divergence columns
            df_div['bullish_divergence'] = False
            df_div['bearish_divergence'] = False  
            df_div['bullish_hidden_div'] = False
            df_div['bearish_hidden_div'] = False
            df_div['divergence_strength'] = 0.0
            
            # Ensure we have required data
            required_cols = ['close', 'high', 'low', 'macd_histogram', 'macd_line']
            if not all(col in df_div.columns for col in required_cols):
                self.logger.warning("Missing required columns for divergence detection")
                return df_div
            
            # Need enough data for meaningful analysis
            if len(df_div) < lookback * 2:
                return df_div
            
            # Find swing highs and lows in price and MACD
            df_div = self._identify_swing_points(df_div, lookback)
            
            # Detect regular divergences (reversal signals)
            df_div = self._detect_regular_divergence(df_div, lookback)
            
            # Detect hidden divergences (continuation signals)
            df_div = self._detect_hidden_divergence(df_div, lookback)
            
            self.logger.debug("MACD divergence detection completed")
            return df_div
            
        except Exception as e:
            self.logger.error(f"Error detecting MACD divergence: {e}")
            return df
    
    def _identify_swing_points(self, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Identify swing highs and lows in price and MACD"""
        try:
            df_swing = df.copy()
            
            # Price swing points
            df_swing['price_swing_high'] = False
            df_swing['price_swing_low'] = False
            
            # MACD swing points
            df_swing['macd_swing_high'] = False
            df_swing['macd_swing_low'] = False
            
            # Rolling windows to find local extremes
            for i in range(lookback, len(df_swing) - lookback):
                current_idx = df_swing.index[i]
                
                # Price swing high detection
                window_highs = df_swing['high'].iloc[i-lookback:i+lookback+1]
                if df_swing.loc[current_idx, 'high'] == window_highs.max():
                    df_swing.loc[current_idx, 'price_swing_high'] = True
                
                # Price swing low detection  
                window_lows = df_swing['low'].iloc[i-lookback:i+lookback+1]
                if df_swing.loc[current_idx, 'low'] == window_lows.min():
                    df_swing.loc[current_idx, 'price_swing_low'] = True
                
                # MACD swing high detection
                macd_window_high = df_swing['macd_line'].iloc[i-lookback:i+lookback+1]
                if df_swing.loc[current_idx, 'macd_line'] == macd_window_high.max():
                    df_swing.loc[current_idx, 'macd_swing_high'] = True
                
                # MACD swing low detection
                macd_window_low = df_swing['macd_line'].iloc[i-lookback:i+lookback+1]
                if df_swing.loc[current_idx, 'macd_line'] == macd_window_low.min():
                    df_swing.loc[current_idx, 'macd_swing_low'] = True
            
            return df_swing
            
        except Exception as e:
            self.logger.error(f"Error identifying swing points: {e}")
            return df
    
    def _detect_regular_divergence(self, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Detect regular bullish and bearish divergences"""
        try:
            df_reg = df.copy()
            
            # Get recent swing points
            recent_data = df_reg.tail(lookback * 3)  # Look at recent data
            
            # Find price and MACD swing points
            price_highs = recent_data[recent_data['price_swing_high']]['high']
            price_lows = recent_data[recent_data['price_swing_low']]['low']
            macd_highs = recent_data[recent_data['macd_swing_high']]['macd_line']
            macd_lows = recent_data[recent_data['macd_swing_low']]['macd_line']
            
            # Check for bullish regular divergence (price lower lows, MACD higher lows)
            if len(price_lows) >= 2 and len(macd_lows) >= 2:
                recent_price_lows = price_lows.tail(2)
                recent_macd_lows = macd_lows.tail(2)
                
                if len(recent_price_lows) == 2 and len(recent_macd_lows) == 2:
                    price_lower = recent_price_lows.iloc[-1] < recent_price_lows.iloc[-2]
                    macd_higher = recent_macd_lows.iloc[-1] > recent_macd_lows.iloc[-2]
                    
                    if price_lower and macd_higher:
                        last_idx = recent_price_lows.index[-1]
                        df_reg.loc[last_idx, 'bullish_divergence'] = True
                        # Calculate divergence strength
                        price_diff = abs(recent_price_lows.iloc[-2] - recent_price_lows.iloc[-1])
                        macd_diff = abs(recent_macd_lows.iloc[-1] - recent_macd_lows.iloc[-2])
                        df_reg.loc[last_idx, 'divergence_strength'] = min(1.0, (price_diff + macd_diff) / 0.01)
            
            # Check for bearish regular divergence (price higher highs, MACD lower highs)
            if len(price_highs) >= 2 and len(macd_highs) >= 2:
                recent_price_highs = price_highs.tail(2)
                recent_macd_highs = macd_highs.tail(2)
                
                if len(recent_price_highs) == 2 and len(recent_macd_highs) == 2:
                    price_higher = recent_price_highs.iloc[-1] > recent_price_highs.iloc[-2]
                    macd_lower = recent_macd_highs.iloc[-1] < recent_macd_highs.iloc[-2]
                    
                    if price_higher and macd_lower:
                        last_idx = recent_price_highs.index[-1]
                        df_reg.loc[last_idx, 'bearish_divergence'] = True
                        # Calculate divergence strength
                        price_diff = abs(recent_price_highs.iloc[-1] - recent_price_highs.iloc[-2])
                        macd_diff = abs(recent_macd_highs.iloc[-2] - recent_macd_highs.iloc[-1])
                        df_reg.loc[last_idx, 'divergence_strength'] = min(1.0, (price_diff + macd_diff) / 0.01)
            
            return df_reg
            
        except Exception as e:
            self.logger.error(f"Error detecting regular divergence: {e}")
            return df
    
    def _detect_hidden_divergence(self, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Detect hidden bullish and bearish divergences (trend continuation)"""
        try:
            df_hid = df.copy()
            
            # Get recent swing points (similar to regular but looking for continuation patterns)
            recent_data = df_hid.tail(lookback * 3)
            
            price_highs = recent_data[recent_data['price_swing_high']]['high']
            price_lows = recent_data[recent_data['price_swing_low']]['low']
            macd_highs = recent_data[recent_data['macd_swing_high']]['macd_line']
            macd_lows = recent_data[recent_data['macd_swing_low']]['macd_line']
            
            # Bullish hidden divergence (price higher lows, MACD lower lows - uptrend continuation)
            if len(price_lows) >= 2 and len(macd_lows) >= 2:
                recent_price_lows = price_lows.tail(2)
                recent_macd_lows = macd_lows.tail(2)
                
                if len(recent_price_lows) == 2 and len(recent_macd_lows) == 2:
                    price_higher = recent_price_lows.iloc[-1] > recent_price_lows.iloc[-2]
                    macd_lower = recent_macd_lows.iloc[-1] < recent_macd_lows.iloc[-2]
                    
                    if price_higher and macd_lower:
                        last_idx = recent_price_lows.index[-1]
                        df_hid.loc[last_idx, 'bullish_hidden_div'] = True
            
            # Bearish hidden divergence (price lower highs, MACD higher highs - downtrend continuation)
            if len(price_highs) >= 2 and len(macd_highs) >= 2:
                recent_price_highs = price_highs.tail(2)
                recent_macd_highs = macd_highs.tail(2)
                
                if len(recent_price_highs) == 2 and len(recent_macd_highs) == 2:
                    price_lower = recent_price_highs.iloc[-1] < recent_price_highs.iloc[-2]
                    macd_higher = recent_macd_highs.iloc[-1] > recent_macd_highs.iloc[-2]
                    
                    if price_lower and macd_higher:
                        last_idx = recent_price_highs.index[-1]
                        df_hid.loc[last_idx, 'bearish_hidden_div'] = True
            
            return df_hid
            
        except Exception as e:
            self.logger.error(f"Error detecting hidden divergence: {e}")
            return df