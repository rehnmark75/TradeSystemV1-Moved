# core/strategies/helpers/mean_reversion_indicator_calculator.py
"""
Mean Reversion Indicator Calculator Module
Handles multi-oscillator calculations for mean reversion strategy
Based on RAG analysis: LuxAlgo Premium Oscillator, Multi-timeframe RSI, Divergence Detection, Squeeze Momentum
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

try:
    from forex_scanner.configdata.strategies import config_mean_reversion_strategy as mr_config
except ImportError:
    import configdata.strategies.config_mean_reversion_strategy as mr_config


class MeanReversionIndicatorCalculator:
    """Calculates mean reversion indicators and detects oscillator confluence signals"""

    def __init__(self, logger: logging.Logger = None, eps: float = 1e-8):
        self.logger = logger or logging.getLogger(__name__)
        self.eps = eps

        # Global signal tracking to prevent over-signaling
        self.global_signal_tracker = {}  # {epic: {'last_signal_time': timestamp, 'signal_count': int}}

        # Load configuration
        self.config = self._load_configuration()

    def _load_configuration(self) -> Dict:
        """Load mean reversion configuration from config module"""
        return {
            # LuxAlgo Oscillator settings
            'luxalgo_length': mr_config.LUXALGO_LENGTH,
            'luxalgo_smoothing': mr_config.LUXALGO_SMOOTHING,
            'luxalgo_overbought': mr_config.LUXALGO_OVERBOUGHT_THRESHOLD,
            'luxalgo_oversold': mr_config.LUXALGO_OVERSOLD_THRESHOLD,
            'luxalgo_extreme_ob': mr_config.LUXALGO_EXTREME_OB_THRESHOLD,
            'luxalgo_extreme_os': mr_config.LUXALGO_EXTREME_OS_THRESHOLD,

            # Multi-timeframe RSI settings
            'mtf_rsi_period': mr_config.MTF_RSI_PERIOD,
            'mtf_rsi_timeframes': mr_config.MTF_RSI_TIMEFRAMES,
            'mtf_rsi_overbought': mr_config.MTF_RSI_OVERBOUGHT,
            'mtf_rsi_oversold': mr_config.MTF_RSI_OVERSOLD,

            # RSI-EMA Divergence settings
            'rsi_ema_period': mr_config.RSI_EMA_PERIOD,
            'rsi_ema_rsi_period': mr_config.RSI_EMA_RSI_PERIOD,
            'rsi_ema_divergence_sensitivity': mr_config.RSI_EMA_DIVERGENCE_SENSITIVITY,

            # Squeeze Momentum settings
            'squeeze_bb_length': mr_config.SQUEEZE_BB_LENGTH,
            'squeeze_bb_mult': mr_config.SQUEEZE_BB_MULT,
            'squeeze_kc_length': mr_config.SQUEEZE_KC_LENGTH,
            'squeeze_kc_mult': mr_config.SQUEEZE_KC_MULT,
            'squeeze_momentum_length': mr_config.SQUEEZE_MOMENTUM_LENGTH,

            # Oscillator confluence settings
            'oscillator_weights': mr_config.OSCILLATOR_WEIGHTS,
            'bull_confluence_threshold': mr_config.OSCILLATOR_BULL_CONFLUENCE_THRESHOLD,
            'bear_confluence_threshold': mr_config.OSCILLATOR_BEAR_CONFLUENCE_THRESHOLD,

            # Signal quality settings
            'min_confidence': mr_config.SIGNAL_QUALITY_MIN_CONFIDENCE,
            'max_signals_per_day': mr_config.SIGNAL_FILTER_MAX_SIGNALS_PER_DAY,
            'min_signal_spacing': mr_config.SIGNAL_FILTER_MIN_SIGNAL_SPACING
        }

    def get_required_indicators(self) -> List[str]:
        """Get list of required indicators for mean reversion strategy"""
        return [
            # LuxAlgo Premium Oscillator
            'luxalgo_oscillator',
            'luxalgo_signal',
            'luxalgo_histogram',

            # Multi-timeframe RSI
            'rsi_14',
            'mtf_rsi_15m',
            'mtf_rsi_1h',
            'mtf_rsi_4h',

            # RSI-EMA Divergence
            f'ema_{self.config["rsi_ema_period"]}',
            'rsi_ema_divergence_bull',
            'rsi_ema_divergence_bear',

            # Squeeze Momentum
            'bb_upper',
            'bb_lower',
            'kc_upper',
            'kc_lower',
            'squeeze_momentum',
            'squeeze_on',

            # Supporting indicators
            'atr',
            'adx',
            'vwap',
            'ema_200'  # For trend filter
        ]

    def validate_data_requirements(self, df: pd.DataFrame, min_bars: int = 100) -> bool:
        """Validate that we have enough data for mean reversion calculations"""
        if len(df) < min_bars:
            self.logger.debug(f"Insufficient data: {len(df)} bars (need {min_bars})")
            return False

        required_columns = ['open', 'high', 'low', 'close']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            return False

        return True

    def ensure_mean_reversion_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all mean reversion indicators if not present

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with all mean reversion indicators added
        """
        df_copy = df.copy()

        try:
            # 1. Calculate LuxAlgo Premium Oscillator
            df_copy = self._calculate_luxalgo_oscillator(df_copy)

            # 2. Calculate Multi-timeframe RSI
            df_copy = self._calculate_mtf_rsi(df_copy)

            # 3. Calculate RSI-EMA Divergence (FAST MODE: Skip if enabled)
            if not (hasattr(mr_config, 'BACKTEST_FAST_MODE') and mr_config.BACKTEST_FAST_MODE and mr_config.FAST_MODE_DISABLE_DIVERGENCE):
                df_copy = self._calculate_rsi_ema_divergence(df_copy)
            else:
                # Fast mode: Add dummy columns to prevent errors
                df_copy['rsi_ema_divergence_bull'] = False
                df_copy['rsi_ema_divergence_bear'] = False
                df_copy['divergence_strength'] = 0.0

            # 4. Calculate Squeeze Momentum Indicator
            df_copy = self._calculate_squeeze_momentum(df_copy)

            # 5. Add supporting indicators
            df_copy = self._add_supporting_indicators(df_copy)

            # 6. Calculate oscillator confluence scores
            df_copy = self._calculate_oscillator_confluence(df_copy)

            self.logger.debug("Mean reversion indicators calculated successfully")
            return df_copy

        except Exception as e:
            self.logger.error(f"Error calculating mean reversion indicators: {e}")
            return df_copy

    def _calculate_luxalgo_oscillator(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate LuxAlgo Premium Oscillator (primary mean reversion engine)

        Implementation based on stochastic oscillator with enhanced smoothing
        """
        try:
            df_luxalgo = df.copy()
            length = self.config['luxalgo_length']
            smoothing = self.config['luxalgo_smoothing']

            # Calculate Stochastic %K (raw oscillator)
            df_luxalgo['lowest_low'] = df_luxalgo['low'].rolling(window=length).min()
            df_luxalgo['highest_high'] = df_luxalgo['high'].rolling(window=length).max()

            # Avoid division by zero
            range_diff = df_luxalgo['highest_high'] - df_luxalgo['lowest_low']
            range_diff = range_diff.replace(0, self.eps)

            df_luxalgo['stoch_k'] = ((df_luxalgo['close'] - df_luxalgo['lowest_low']) / range_diff) * 100

            # Apply LuxAlgo smoothing (double smoothing for premium quality)
            df_luxalgo['luxalgo_raw'] = df_luxalgo['stoch_k'].ewm(span=smoothing).mean()
            df_luxalgo['luxalgo_oscillator'] = df_luxalgo['luxalgo_raw'].ewm(span=smoothing).mean()

            # Calculate signal line (EMA of oscillator)
            df_luxalgo['luxalgo_signal'] = df_luxalgo['luxalgo_oscillator'].ewm(span=3).mean()

            # Calculate histogram (oscillator - signal)
            df_luxalgo['luxalgo_histogram'] = df_luxalgo['luxalgo_oscillator'] - df_luxalgo['luxalgo_signal']

            # Mark extreme readings
            df_luxalgo['luxalgo_extreme_overbought'] = df_luxalgo['luxalgo_oscillator'] > self.config['luxalgo_extreme_ob']
            df_luxalgo['luxalgo_extreme_oversold'] = df_luxalgo['luxalgo_oscillator'] < self.config['luxalgo_extreme_os']

            # Clean up temporary columns
            temp_cols = ['lowest_low', 'highest_high', 'stoch_k', 'luxalgo_raw']
            df_luxalgo = df_luxalgo.drop(columns=temp_cols, errors='ignore')

            self.logger.debug(f"LuxAlgo oscillator calculated (length: {length}, smoothing: {smoothing})")
            return df_luxalgo

        except Exception as e:
            self.logger.error(f"Error calculating LuxAlgo oscillator: {e}")
            return df

    def _calculate_mtf_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Multi-timeframe RSI for confluence analysis

        Note: In real implementation, this would fetch higher timeframe data
        For now, we simulate MTF by using different periods
        """
        try:
            df_mtf = df.copy()
            base_period = self.config['mtf_rsi_period']

            # Calculate base RSI (current timeframe)
            df_mtf = self._calculate_rsi(df_mtf, f'rsi_{base_period}', base_period)

            # Simulate multi-timeframe RSI using different periods
            # 15m simulated with period 14
            df_mtf = self._calculate_rsi(df_mtf, 'mtf_rsi_15m', base_period)

            # 1h simulated with period 21 (1.5x base for higher TF effect)
            df_mtf = self._calculate_rsi(df_mtf, 'mtf_rsi_1h', int(base_period * 1.5))

            # 4h simulated with period 28 (2x base for higher TF effect)
            df_mtf = self._calculate_rsi(df_mtf, 'mtf_rsi_4h', int(base_period * 2))

            # Calculate MTF alignment score
            df_mtf = self._calculate_mtf_alignment(df_mtf)

            self.logger.debug("Multi-timeframe RSI calculated")
            return df_mtf

        except Exception as e:
            self.logger.error(f"Error calculating MTF RSI: {e}")
            return df

    def _calculate_rsi(self, df: pd.DataFrame, column_name: str, period: int) -> pd.DataFrame:
        """Calculate RSI for specified period"""
        try:
            df_rsi = df.copy()

            # Calculate price changes
            df_rsi['price_change'] = df_rsi['close'].diff()

            # Separate gains and losses
            df_rsi['gain'] = df_rsi['price_change'].where(df_rsi['price_change'] > 0, 0)
            df_rsi['loss'] = -df_rsi['price_change'].where(df_rsi['price_change'] < 0, 0)

            # Calculate average gains and losses using EMA
            alpha = 1.0 / period
            df_rsi['avg_gain'] = df_rsi['gain'].ewm(alpha=alpha, adjust=False).mean()
            df_rsi['avg_loss'] = df_rsi['loss'].ewm(alpha=alpha, adjust=False).mean()

            # Calculate RS and RSI
            df_rsi['rs'] = df_rsi['avg_gain'] / df_rsi['avg_loss'].replace(0, self.eps)
            df_rsi[column_name] = 100 - (100 / (1 + df_rsi['rs']))

            # Handle edge cases
            df_rsi[column_name] = df_rsi[column_name].fillna(50)

            # Clean up temporary columns
            temp_cols = ['price_change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs']
            df_rsi = df_rsi.drop(columns=temp_cols, errors='ignore')

            return df_rsi

        except Exception as e:
            self.logger.error(f"Error calculating RSI {column_name}: {e}")
            return df

    def _calculate_mtf_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate multi-timeframe RSI alignment score"""
        try:
            df_align = df.copy()

            # Get RSI columns
            rsi_cols = ['mtf_rsi_15m', 'mtf_rsi_1h', 'mtf_rsi_4h']

            # Calculate bullish alignment (RSI values supporting bullish direction)
            df_align['mtf_bull_count'] = 0
            df_align['mtf_bear_count'] = 0

            for col in rsi_cols:
                if col in df_align.columns:
                    # Bullish: RSI in 40-60 range (neutral to slightly bullish)
                    df_align['mtf_bull_count'] += (
                        (df_align[col] >= 40) & (df_align[col] <= 70)
                    ).astype(int)

                    # Bearish: RSI in 30-60 range (neutral to slightly bearish)
                    df_align['mtf_bear_count'] += (
                        (df_align[col] >= 30) & (df_align[col] <= 60)
                    ).astype(int)

            # Calculate alignment scores (0-1)
            total_timeframes = len(rsi_cols)
            df_align['mtf_bull_alignment'] = df_align['mtf_bull_count'] / total_timeframes
            df_align['mtf_bear_alignment'] = df_align['mtf_bear_count'] / total_timeframes

            return df_align

        except Exception as e:
            self.logger.error(f"Error calculating MTF alignment: {e}")
            return df

    def _calculate_rsi_ema_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI-EMA divergence detection for reversal signals
        """
        try:
            df_div = df.copy()
            ema_period = self.config['rsi_ema_period']
            rsi_period = self.config['rsi_ema_rsi_period']

            # Calculate EMA for divergence analysis
            if f'ema_{ema_period}' not in df_div.columns:
                df_div[f'ema_{ema_period}'] = df_div['close'].ewm(span=ema_period).mean()

            # Calculate RSI for divergence analysis
            if f'rsi_{rsi_period}' not in df_div.columns:
                df_div = self._calculate_rsi(df_div, f'rsi_{rsi_period}', rsi_period)

            # Detect divergence patterns
            df_div = self._detect_price_rsi_divergence(df_div, ema_period, rsi_period)

            self.logger.debug("RSI-EMA divergence calculated")
            return df_div

        except Exception as e:
            self.logger.error(f"Error calculating RSI-EMA divergence: {e}")
            return df

    def _detect_price_rsi_divergence(self, df: pd.DataFrame, ema_period: int, rsi_period: int, lookback: int = 20) -> pd.DataFrame:
        """Detect bullish and bearish divergence between price and RSI"""
        try:
            df_divergence = df.copy()

            # Initialize divergence columns
            df_divergence['rsi_ema_divergence_bull'] = False
            df_divergence['rsi_ema_divergence_bear'] = False
            df_divergence['divergence_strength'] = 0.0

            if len(df_divergence) < lookback * 2:
                return df_divergence

            ema_col = f'ema_{ema_period}'
            rsi_col = f'rsi_{rsi_period}'

            # Rolling window analysis for divergence
            for i in range(lookback, len(df_divergence)):
                current_idx = df_divergence.index[i]

                # Get recent data window
                start_idx = max(0, i - lookback)
                window_data = df_divergence.iloc[start_idx:i+1]

                # Find local extremes in price and RSI
                price_highs = window_data[ema_col].rolling(window=5, center=True).max()
                price_lows = window_data[ema_col].rolling(window=5, center=True).min()
                rsi_highs = window_data[rsi_col].rolling(window=5, center=True).max()
                rsi_lows = window_data[rsi_col].rolling(window=5, center=True).min()

                # Bullish divergence: price lower lows, RSI higher lows
                price_low_points = window_data[window_data[ema_col] == price_lows]
                rsi_low_points = window_data[window_data[rsi_col] == rsi_lows]

                if len(price_low_points) >= 2 and len(rsi_low_points) >= 2:
                    recent_price_lows = price_low_points[ema_col].tail(2)
                    recent_rsi_lows = rsi_low_points[rsi_col].tail(2)

                    if len(recent_price_lows) == 2 and len(recent_rsi_lows) == 2:
                        price_declining = recent_price_lows.iloc[-1] < recent_price_lows.iloc[-2]
                        rsi_rising = recent_rsi_lows.iloc[-1] > recent_rsi_lows.iloc[-2]

                        if price_declining and rsi_rising:
                            df_divergence.loc[current_idx, 'rsi_ema_divergence_bull'] = True
                            # Calculate strength
                            price_diff = abs(recent_price_lows.iloc[-2] - recent_price_lows.iloc[-1])
                            rsi_diff = abs(recent_rsi_lows.iloc[-1] - recent_rsi_lows.iloc[-2])
                            df_divergence.loc[current_idx, 'divergence_strength'] = min(1.0, (price_diff + rsi_diff) / 10)

                # Bearish divergence: price higher highs, RSI lower highs
                price_high_points = window_data[window_data[ema_col] == price_highs]
                rsi_high_points = window_data[window_data[rsi_col] == rsi_highs]

                if len(price_high_points) >= 2 and len(rsi_high_points) >= 2:
                    recent_price_highs = price_high_points[ema_col].tail(2)
                    recent_rsi_highs = rsi_high_points[rsi_col].tail(2)

                    if len(recent_price_highs) == 2 and len(recent_rsi_highs) == 2:
                        price_rising = recent_price_highs.iloc[-1] > recent_price_highs.iloc[-2]
                        rsi_declining = recent_rsi_highs.iloc[-1] < recent_rsi_highs.iloc[-2]

                        if price_rising and rsi_declining:
                            df_divergence.loc[current_idx, 'rsi_ema_divergence_bear'] = True
                            # Calculate strength
                            price_diff = abs(recent_price_highs.iloc[-1] - recent_price_highs.iloc[-2])
                            rsi_diff = abs(recent_rsi_highs.iloc[-2] - recent_rsi_highs.iloc[-1])
                            df_divergence.loc[current_idx, 'divergence_strength'] = min(1.0, (price_diff + rsi_diff) / 10)

            return df_divergence

        except Exception as e:
            self.logger.error(f"Error detecting price-RSI divergence: {e}")
            return df

    def _calculate_squeeze_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Squeeze Momentum Indicator (LazyBear style)
        """
        try:
            df_squeeze = df.copy()

            bb_length = self.config['squeeze_bb_length']
            bb_mult = self.config['squeeze_bb_mult']
            kc_length = self.config['squeeze_kc_length']
            kc_mult = self.config['squeeze_kc_mult']
            momentum_length = self.config['squeeze_momentum_length']

            # Calculate Bollinger Bands
            bb_basis = df_squeeze['close'].rolling(window=bb_length).mean()
            bb_dev = df_squeeze['close'].rolling(window=bb_length).std()
            df_squeeze['bb_upper'] = bb_basis + (bb_dev * bb_mult)
            df_squeeze['bb_lower'] = bb_basis - (bb_dev * bb_mult)

            # Calculate Keltner Channels
            kc_basis = df_squeeze['close'].ewm(span=kc_length).mean()

            # True Range for Keltner Channels
            df_squeeze['prev_close'] = df_squeeze['close'].shift(1)
            df_squeeze['tr1'] = df_squeeze['high'] - df_squeeze['low']
            df_squeeze['tr2'] = abs(df_squeeze['high'] - df_squeeze['prev_close'])
            df_squeeze['tr3'] = abs(df_squeeze['low'] - df_squeeze['prev_close'])
            df_squeeze['true_range'] = df_squeeze[['tr1', 'tr2', 'tr3']].max(axis=1)

            # Average True Range
            atr = df_squeeze['true_range'].ewm(span=kc_length).mean()
            df_squeeze['kc_upper'] = kc_basis + (atr * kc_mult)
            df_squeeze['kc_lower'] = kc_basis - (atr * kc_mult)

            # Squeeze condition: Bollinger Bands inside Keltner Channels
            df_squeeze['squeeze_on'] = (
                (df_squeeze['bb_upper'] < df_squeeze['kc_upper']) &
                (df_squeeze['bb_lower'] > df_squeeze['kc_lower'])
            )

            # Calculate momentum
            highest = df_squeeze['high'].rolling(window=momentum_length).max()
            lowest = df_squeeze['low'].rolling(window=momentum_length).min()
            mid_range = (highest + lowest) / 2
            df_squeeze['squeeze_momentum'] = df_squeeze['close'] - mid_range

            # Smooth momentum
            df_squeeze['squeeze_momentum'] = df_squeeze['squeeze_momentum'].ewm(span=3).mean()

            # Clean up temporary columns
            temp_cols = ['prev_close', 'tr1', 'tr2', 'tr3', 'true_range']
            df_squeeze = df_squeeze.drop(columns=temp_cols, errors='ignore')

            self.logger.debug("Squeeze momentum indicator calculated")
            return df_squeeze

        except Exception as e:
            self.logger.error(f"Error calculating squeeze momentum: {e}")
            return df

    def _add_supporting_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add supporting indicators (ATR, ADX, VWAP, EMA200)"""
        try:
            df_support = df.copy()

            # Add ATR (Average True Range)
            if 'atr' not in df_support.columns:
                df_support = self._calculate_atr(df_support)

            # Add ADX (Average Directional Index)
            if 'adx' not in df_support.columns:
                df_support = self._calculate_adx(df_support)

            # Add VWAP (Volume Weighted Average Price)
            if 'vwap' not in df_support.columns:
                df_support = self._calculate_vwap(df_support)

            # Add EMA200 for trend filter
            if 'ema_200' not in df_support.columns:
                df_support['ema_200'] = df_support['close'].ewm(span=200).mean()

            return df_support

        except Exception as e:
            self.logger.error(f"Error adding supporting indicators: {e}")
            return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range"""
        try:
            df_atr = df.copy()

            # True Range calculation
            df_atr['high_low'] = df_atr['high'] - df_atr['low']
            df_atr['high_close'] = abs(df_atr['high'] - df_atr['close'].shift(1))
            df_atr['low_close'] = abs(df_atr['low'] - df_atr['close'].shift(1))

            df_atr['true_range'] = df_atr[['high_low', 'high_close', 'low_close']].max(axis=1)
            df_atr['atr'] = df_atr['true_range'].ewm(span=period, adjust=False).mean()

            # Clean up
            df_atr = df_atr.drop(columns=['high_low', 'high_close', 'low_close', 'true_range'], errors='ignore')
            return df_atr

        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return df

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index"""
        try:
            df_adx = df.copy()

            # Calculate True Range and directional movements
            df_adx['high_diff'] = df_adx['high'].diff()
            df_adx['low_diff'] = -df_adx['low'].diff()

            df_adx['plus_dm'] = np.where(
                (df_adx['high_diff'] > df_adx['low_diff']) & (df_adx['high_diff'] > 0),
                df_adx['high_diff'], 0
            )
            df_adx['minus_dm'] = np.where(
                (df_adx['low_diff'] > df_adx['high_diff']) & (df_adx['low_diff'] > 0),
                df_adx['low_diff'], 0
            )

            # True Range
            df_adx['prev_close'] = df_adx['close'].shift(1)
            df_adx['tr1'] = df_adx['high'] - df_adx['low']
            df_adx['tr2'] = abs(df_adx['high'] - df_adx['prev_close'])
            df_adx['tr3'] = abs(df_adx['low'] - df_adx['prev_close'])
            df_adx['true_range'] = df_adx[['tr1', 'tr2', 'tr3']].max(axis=1)

            # Smooth the directional movements and true range
            alpha = 1.0 / period
            df_adx['tr_smooth'] = df_adx['true_range'].ewm(alpha=alpha, adjust=False).mean()
            df_adx['plus_dm_smooth'] = df_adx['plus_dm'].ewm(alpha=alpha, adjust=False).mean()
            df_adx['minus_dm_smooth'] = df_adx['minus_dm'].ewm(alpha=alpha, adjust=False).mean()

            # Calculate Directional Indicators
            df_adx['plus_di'] = 100 * (df_adx['plus_dm_smooth'] / df_adx['tr_smooth'])
            df_adx['minus_di'] = 100 * (df_adx['minus_dm_smooth'] / df_adx['tr_smooth'])

            # Calculate DX and ADX
            df_adx['dx'] = 100 * abs(df_adx['plus_di'] - df_adx['minus_di']) / (df_adx['plus_di'] + df_adx['minus_di'])
            df_adx['adx'] = df_adx['dx'].ewm(alpha=alpha, adjust=False).mean()

            # Clean up
            temp_cols = ['high_diff', 'low_diff', 'plus_dm', 'minus_dm', 'prev_close',
                        'tr1', 'tr2', 'tr3', 'true_range', 'tr_smooth', 'plus_dm_smooth',
                        'minus_dm_smooth', 'plus_di', 'minus_di', 'dx']
            df_adx = df_adx.drop(columns=temp_cols, errors='ignore')

            return df_adx

        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            return df

    def _calculate_vwap(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Volume Weighted Average Price"""
        try:
            df_vwap = df.copy()

            # Typical price
            df_vwap['typical_price'] = (df_vwap['high'] + df_vwap['low'] + df_vwap['close']) / 3

            # Use ltv as volume or create proxy
            volume_col = 'ltv' if 'ltv' in df_vwap.columns else None
            if volume_col is None:
                df_vwap['volume_proxy'] = abs(df_vwap['close'].diff()).fillna(0) * 1000
                volume_col = 'volume_proxy'

            # Calculate VWAP
            price_volume = df_vwap['typical_price'] * df_vwap[volume_col]
            rolling_pv = price_volume.rolling(window=period, min_periods=1).sum()
            rolling_volume = df_vwap[volume_col].rolling(window=period, min_periods=1).sum()

            df_vwap['vwap'] = rolling_pv / rolling_volume.replace(0, 1)
            df_vwap['vwap_deviation'] = (df_vwap['close'] - df_vwap['vwap']) / df_vwap['vwap'] * 100

            return df_vwap

        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return df

    def _calculate_oscillator_confluence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weighted oscillator confluence scores"""
        try:
            df_confluence = df.copy()

            # Initialize confluence scores
            df_confluence['oscillator_bull_score'] = 0.0
            df_confluence['oscillator_bear_score'] = 0.0

            weights = self.config['oscillator_weights']

            # 1. LuxAlgo Oscillator contribution
            if 'luxalgo_oscillator' in df_confluence.columns:
                luxalgo_bull = (
                    (df_confluence['luxalgo_oscillator'] < self.config['luxalgo_oversold']) |
                    (df_confluence['luxalgo_extreme_oversold'])
                ).astype(float)

                luxalgo_bear = (
                    (df_confluence['luxalgo_oscillator'] > self.config['luxalgo_overbought']) |
                    (df_confluence['luxalgo_extreme_overbought'])
                ).astype(float)

                df_confluence['oscillator_bull_score'] += luxalgo_bull * weights['luxalgo']
                df_confluence['oscillator_bear_score'] += luxalgo_bear * weights['luxalgo']

            # 2. Multi-timeframe RSI contribution
            if 'mtf_bull_alignment' in df_confluence.columns:
                df_confluence['oscillator_bull_score'] += df_confluence['mtf_bull_alignment'] * weights['mtf_rsi']
                df_confluence['oscillator_bear_score'] += df_confluence['mtf_bear_alignment'] * weights['mtf_rsi']

            # 3. RSI-EMA Divergence contribution
            if 'rsi_ema_divergence_bull' in df_confluence.columns:
                divergence_bull = df_confluence['rsi_ema_divergence_bull'].astype(float) * df_confluence['divergence_strength']
                divergence_bear = df_confluence['rsi_ema_divergence_bear'].astype(float) * df_confluence['divergence_strength']

                df_confluence['oscillator_bull_score'] += divergence_bull * weights['divergence']
                df_confluence['oscillator_bear_score'] += divergence_bear * weights['divergence']

            # 4. Squeeze Momentum contribution
            if 'squeeze_momentum' in df_confluence.columns and 'squeeze_on' in df_confluence.columns:
                # Only consider squeeze momentum when squeeze is releasing
                squeeze_releasing = (~df_confluence['squeeze_on']) & (df_confluence['squeeze_on'].shift(1))

                squeeze_bull = (
                    squeeze_releasing & (df_confluence['squeeze_momentum'] > 0)
                ).astype(float)

                squeeze_bear = (
                    squeeze_releasing & (df_confluence['squeeze_momentum'] < 0)
                ).astype(float)

                df_confluence['oscillator_bull_score'] += squeeze_bull * weights['squeeze']
                df_confluence['oscillator_bear_score'] += squeeze_bear * weights['squeeze']

            # Generate confluence signals
            df_confluence['oscillator_bull_confluence'] = (
                df_confluence['oscillator_bull_score'] >= self.config['bull_confluence_threshold']
            )
            df_confluence['oscillator_bear_confluence'] = (
                df_confluence['oscillator_bear_score'] >= self.config['bear_confluence_threshold']
            )

            return df_confluence

        except Exception as e:
            self.logger.error(f"Error calculating oscillator confluence: {e}")
            return df

    def get_epic_specific_thresholds(self, epic: str) -> Dict:
        """Get epic-specific thresholds for mean reversion signals"""
        try:
            return mr_config.get_mean_reversion_threshold_for_epic(epic)
        except Exception as e:
            self.logger.error(f"Error getting epic thresholds: {e}")
            return {
                'luxalgo_overbought': 80,
                'luxalgo_oversold': 20,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'squeeze_momentum_threshold': 0.1,
                'min_zone_distance': 10
            }

    def validate_oscillator_strength(self, row: pd.Series, signal_type: str, epic: str) -> bool:
        """
        Validate oscillator signal strength for the given epic

        Args:
            row: DataFrame row with oscillator data
            signal_type: 'BULL' or 'BEAR'
            epic: Trading pair epic

        Returns:
            True if signal meets strength requirements
        """
        try:
            thresholds = self.get_epic_specific_thresholds(epic)

            if signal_type == 'BULL':
                # Check if oscillators are in oversold territory
                luxalgo_oversold = row.get('luxalgo_oscillator', 50) < thresholds['luxalgo_oversold']
                rsi_oversold = row.get('rsi_14', 50) < thresholds['rsi_oversold']
                confluence_score = row.get('oscillator_bull_score', 0)

                return (luxalgo_oversold or rsi_oversold) and confluence_score >= self.config['bull_confluence_threshold']

            elif signal_type == 'BEAR':
                # Check if oscillators are in overbought territory
                luxalgo_overbought = row.get('luxalgo_oscillator', 50) > thresholds['luxalgo_overbought']
                rsi_overbought = row.get('rsi_14', 50) > thresholds['rsi_overbought']
                confluence_score = row.get('oscillator_bear_score', 0)

                return (luxalgo_overbought or rsi_overbought) and confluence_score >= self.config['bear_confluence_threshold']

            return False

        except Exception as e:
            self.logger.error(f"Error validating oscillator strength: {e}")
            return False

    def get_oscillator_quality_score(self, row: pd.Series, signal_type: str, epic: str) -> float:
        """
        Calculate quality score for oscillator signals (0.0 to 1.0)

        Args:
            row: DataFrame row with oscillator data
            signal_type: 'BULL' or 'BEAR'
            epic: Trading pair epic

        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            total_score = 0.0
            max_score = 100.0

            # Get oscillator values
            luxalgo_osc = row.get('luxalgo_oscillator', 50)
            rsi_14 = row.get('rsi_14', 50)
            confluence_score = row.get(f'oscillator_{signal_type.lower()}_score', 0)
            divergence_strength = row.get('divergence_strength', 0)
            squeeze_momentum = row.get('squeeze_momentum', 0)
            adx = row.get('adx', 0)

            # 1. Oscillator extremity (30 points)
            if signal_type == 'BULL':
                if luxalgo_osc < 10:
                    total_score += 30  # Extreme oversold
                elif luxalgo_osc < 20:
                    total_score += 25  # Very oversold
                elif luxalgo_osc < 30:
                    total_score += 15  # Oversold
            else:  # BEAR
                if luxalgo_osc > 90:
                    total_score += 30  # Extreme overbought
                elif luxalgo_osc > 80:
                    total_score += 25  # Very overbought
                elif luxalgo_osc > 70:
                    total_score += 15  # Overbought

            # 2. Confluence strength (25 points)
            confluence_points = min(25, confluence_score * 25)
            total_score += confluence_points

            # 3. Divergence bonus (20 points)
            divergence_points = divergence_strength * 20
            total_score += divergence_points

            # 4. Momentum strength (15 points)
            if signal_type == 'BULL' and squeeze_momentum > 0:
                momentum_points = min(15, abs(squeeze_momentum) * 150)
                total_score += momentum_points
            elif signal_type == 'BEAR' and squeeze_momentum < 0:
                momentum_points = min(15, abs(squeeze_momentum) * 150)
                total_score += momentum_points

            # 5. Trend strength (10 points)
            if adx > 25:
                total_score += 10
            elif adx > 20:
                total_score += 7
            elif adx > 15:
                total_score += 3

            return min(1.0, total_score / max_score)

        except Exception as e:
            self.logger.error(f"Error calculating oscillator quality score: {e}")
            return 0.5  # Neutral score on error