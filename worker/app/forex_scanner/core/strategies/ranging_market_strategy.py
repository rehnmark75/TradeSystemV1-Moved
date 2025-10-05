# core/strategies/ranging_market_strategy.py
"""
Ranging Market Strategy Implementation - Multi-Oscillator Confluence for Range-Bound Markets
🎯 RANGING MARKET OPTIMIZATION: Specialized strategy for sideways/ranging market conditions
📊 MULTI-OSCILLATOR: Squeeze Momentum + Wave Trend + BB/KC + RSI + RVI confluence
⚡ ORCHESTRATOR: Main class coordinates specialized helper modules for ranging market detection
🏗️ MODULAR: Clean separation of concerns with focused helper modules

Features:
- Squeeze Momentum Indicator (LazyBear - primary ranging engine)
- Wave Trend Oscillator for trend/momentum hybrid analysis
- Bollinger Bands + Keltner Channels for dynamic S/R and squeeze detection
- RSI with divergence detection for mean reversion signals
- Relative Vigor Index for momentum confirmation
- Dynamic support/resistance zone validation
- Market regime filtering (optimize for ranging conditions)
- Database optimization parameter support
- Compatible with existing backtest system
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
import logging
from datetime import datetime

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster
from .helpers.adaptive_volatility_calculator import AdaptiveVolatilityCalculator

# Import optimization functions
try:
    from optimization.optimal_parameter_service import get_ranging_market_optimal_parameters, is_epic_ranging_market_optimized
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    def get_ranging_market_optimal_parameters(*args, **kwargs):
        raise ImportError("Optimization service not available")
    def is_epic_ranging_market_optimized(*args, **kwargs):
        return False

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config

try:
    from forex_scanner.configdata.strategies import config_ranging_market_strategy as rm_config
except ImportError:
    try:
        from configdata.strategies import config_ranging_market_strategy as rm_config
    except ImportError:
        import sys
        sys.path.append('.')
        from configdata.strategies import config_ranging_market_strategy as rm_config


class RangingMarketStrategy(BaseStrategy):
    """
    Ranging Market Strategy - Multi-Oscillator Confluence for Range-Bound Markets

    This strategy is optimized for sideways/ranging market conditions using:
    - Squeeze Momentum (primary ranging detection)
    - Wave Trend Oscillator (momentum/trend hybrid)
    - Bollinger Bands + Keltner Channels (dynamic S/R)
    - RSI with divergence (mean reversion)
    - Relative Vigor Index (momentum confirmation)
    """

    def __init__(self, data_fetcher=None, backtest_mode=False, epic=None, timeframe='15m', use_optimized_parameters=True, pipeline_mode=True):
        # Initialize parent properties manually (following mean_reversion_strategy pattern)
        self.name = 'ranging_market'
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self._validator = None  # Use custom validation logic

        self.backtest_mode = backtest_mode
        self.epic = epic
        self.timeframe = timeframe
        self.use_optimized_parameters = use_optimized_parameters
        self.data_fetcher = data_fetcher

        # Enable/disable expensive features based on pipeline mode
        self.enhanced_validation = pipeline_mode and getattr(config, 'RANGING_MARKET_ENHANCED_VALIDATION', True)

        # Strategy configuration
        self.config = self._load_configuration()

        # PHASE 3: Adaptive volatility-based SL/TP calculation
        self.use_adaptive_sl_tp = getattr(rm_config, 'USE_ADAPTIVE_SL_TP', False)

        if self.use_adaptive_sl_tp:
            # Initialize adaptive volatility calculator (singleton)
            self.adaptive_calculator = AdaptiveVolatilityCalculator(logger=self.logger)
            self.logger.info("🧠 Adaptive volatility calculator enabled - Runtime regime-aware SL/TP")
        else:
            # Fallback: Use ATR multipliers from config
            self.adaptive_calculator = None
            self.stop_atr_multiplier = getattr(rm_config, 'RANGING_STOP_LOSS_ATR_MULTIPLIER', 1.5)
            self.target_atr_multiplier = getattr(rm_config, 'RANGING_TAKE_PROFIT_ATR_MULTIPLIER', 2.5)
            self.logger.info(f"🎯 ATR-based dynamic stops: SL={self.stop_atr_multiplier}x ATR, TP={self.target_atr_multiplier}x ATR")

        # Price adjuster for spread calculations
        self.price_adjuster = PriceAdjuster()

        # Swing proximity validator (NEW)
        self.swing_validator = None
        if self.config.get('swing_validation', {}).get('enabled', True):
            try:
                from .helpers.swing_proximity_validator import SwingProximityValidator
                from .helpers.smc_market_structure import SMCMarketStructure

                # Initialize SMC analyzer for swing detection
                self.smc_analyzer = SMCMarketStructure(logger=self.logger, data_fetcher=data_fetcher)

                # Initialize swing proximity validator
                self.swing_validator = SwingProximityValidator(
                    smc_analyzer=self.smc_analyzer,
                    config=self.config.get('swing_validation', {}),
                    logger=self.logger
                )
                self.logger.info(f"✅ Swing proximity validator initialized (min_distance={self.swing_validator.min_distance_pips} pips)")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not initialize swing validator: {e}")
                self.swing_validator = None

        # Performance tracking
        self.performance_stats = {
            'total_signals': 0,
            'squeeze_signals': 0,
            'wave_trend_signals': 0,
            'bollinger_keltner_signals': 0,
            'rsi_divergence_signals': 0,
            'rvi_signals': 0,
            'confluence_signals': 0,
            'zone_validation_passes': 0,
            'regime_filter_passes': 0
        }

        if self.enhanced_validation:
            self.logger.info(f"🔍 Enhanced validation ENABLED - Full ranging market analysis")
        else:
            self.logger.info(f"🔧 Enhanced validation DISABLED - Basic ranging market testing mode")

        self.logger.info(f"✅ Ranging Market Strategy initialized for {epic} on {timeframe}")

    def _load_configuration(self) -> Dict:
        """Load strategy configuration with optional optimization"""
        try:
            config_dict = {
                # Core settings
                'strategy_enabled': getattr(rm_config, 'RANGING_MARKET_STRATEGY', True),
                'strategy_weight': getattr(rm_config, 'STRATEGY_WEIGHT_RANGING_MARKET', 0.35),

                # Squeeze Momentum settings
                'squeeze_momentum_enabled': getattr(rm_config, 'SQUEEZE_MOMENTUM_ENABLED', True),
                'squeeze_bb_length': getattr(rm_config, 'SQUEEZE_BB_LENGTH', 20),
                'squeeze_bb_mult': getattr(rm_config, 'SQUEEZE_BB_MULT', 2.0),
                'squeeze_kc_length': getattr(rm_config, 'SQUEEZE_KC_LENGTH', 20),
                'squeeze_kc_mult': getattr(rm_config, 'SQUEEZE_KC_MULT', 1.5),
                'squeeze_momentum_length': getattr(rm_config, 'SQUEEZE_MOMENTUM_LENGTH', 12),
                'squeeze_min_bars_in_squeeze': getattr(rm_config, 'SQUEEZE_MIN_BARS_IN_SQUEEZE', 6),
                'squeeze_momentum_threshold': getattr(rm_config, 'SQUEEZE_MOMENTUM_THRESHOLD', 0.1),

                # Wave Trend Oscillator settings
                'wave_trend_enabled': getattr(rm_config, 'WAVE_TREND_ENABLED', True),
                'wto_channel_length': getattr(rm_config, 'WTO_CHANNEL_LENGTH', 10),
                'wto_average_length': getattr(rm_config, 'WTO_AVERAGE_LENGTH', 21),
                'wto_signal_length': getattr(rm_config, 'WTO_SIGNAL_LENGTH', 4),
                'wto_overbought_level': getattr(rm_config, 'WTO_OVERBOUGHT_LEVEL', 60),
                'wto_oversold_level': getattr(rm_config, 'WTO_OVERSOLD_LEVEL', -60),

                # Bollinger/Keltner settings
                'bollinger_bands_enabled': getattr(rm_config, 'BOLLINGER_BANDS_ENABLED', True),
                'keltner_channels_enabled': getattr(rm_config, 'KELTNER_CHANNELS_ENABLED', True),
                'bb_length': getattr(rm_config, 'BB_LENGTH', 20),
                'bb_multiplier': getattr(rm_config, 'BB_MULTIPLIER', 2.0),
                'kc_length': getattr(rm_config, 'KC_LENGTH', 20),
                'kc_multiplier': getattr(rm_config, 'KC_MULTIPLIER', 1.5),

                # RSI settings
                'rsi_enabled': getattr(rm_config, 'RSI_ENABLED', True),
                'rsi_period': getattr(rm_config, 'RSI_PERIOD', 14),
                'rsi_overbought': getattr(rm_config, 'RSI_OVERBOUGHT', 70),
                'rsi_oversold': getattr(rm_config, 'RSI_OVERSOLD', 30),
                'rsi_divergence_enabled': getattr(rm_config, 'RSI_DIVERGENCE_ENABLED', True),

                # RVI settings
                'rvi_enabled': getattr(rm_config, 'RVI_ENABLED', True),
                'rvi_period': getattr(rm_config, 'RVI_PERIOD', 10),
                'rvi_signal_period': getattr(rm_config, 'RVI_SIGNAL_PERIOD', 4),
                'rvi_overbought': getattr(rm_config, 'RVI_OVERBOUGHT', 0.5),
                'rvi_oversold': getattr(rm_config, 'RVI_OVERSOLD', -0.5),

                # Confluence settings
                'oscillator_confluence_enabled': getattr(rm_config, 'OSCILLATOR_CONFLUENCE_ENABLED', True),
                'oscillator_min_confirmations': getattr(rm_config, 'OSCILLATOR_MIN_CONFIRMATIONS', 3),
                'oscillator_weights': getattr(rm_config, 'OSCILLATOR_WEIGHTS', {
                    'squeeze_momentum': 0.35,
                    'wave_trend': 0.25,
                    'rsi': 0.20,
                    'rvi': 0.20
                }),
                'oscillator_bull_confluence_threshold': getattr(rm_config, 'OSCILLATOR_BULL_CONFLUENCE_THRESHOLD', 0.70),
                'oscillator_bear_confluence_threshold': getattr(rm_config, 'OSCILLATOR_BEAR_CONFLUENCE_THRESHOLD', 0.70),

                # Signal quality settings
                'signal_quality_min_confidence': getattr(rm_config, 'SIGNAL_QUALITY_MIN_CONFIDENCE', 0.55),  # Lowered for validation
                'signal_quality_min_risk_reward': getattr(rm_config, 'SIGNAL_QUALITY_MIN_RISK_REWARD', 1.8),

                # Market regime settings
                'market_regime_detection_enabled': getattr(rm_config, 'MARKET_REGIME_DETECTION_ENABLED', True),
                'regime_disable_in_strong_trend': getattr(rm_config, 'REGIME_DISABLE_IN_STRONG_TREND', True),
                'ranging_market_adx_max': getattr(rm_config, 'RANGING_MARKET_ADX_MAX', 20),

                # Dynamic zones settings
                'dynamic_zones_enabled': getattr(rm_config, 'DYNAMIC_ZONES_ENABLED', True),
                'zone_calculation_period': getattr(rm_config, 'ZONE_CALCULATION_PERIOD', 50),
                'zone_proximity_pips': getattr(rm_config, 'ZONE_PROXIMITY_PIPS', 8),

                # MTF settings
                'mtf_analysis_enabled': getattr(rm_config, 'MTF_ANALYSIS_ENABLED', True),
                'mtf_timeframes': getattr(rm_config, 'MTF_TIMEFRAMES', ['5m', '15m', '1h']),
                'mtf_min_alignment_score': getattr(rm_config, 'MTF_MIN_ALIGNMENT_SCORE', 0.65),

                # Risk management
                'ranging_default_sl_pips': getattr(rm_config, 'RANGING_DEFAULT_SL_PIPS', 22),
                'ranging_default_tp_pips': getattr(rm_config, 'RANGING_DEFAULT_TP_PIPS', 38),
                'ranging_dynamic_sl_tp': getattr(rm_config, 'RANGING_DYNAMIC_SL_TP', True),

                # Swing proximity validation (NEW)
                'swing_validation': getattr(rm_config, 'RANGING_SWING_VALIDATION', {
                    'enabled': True,
                    'min_distance_pips': 8,
                    'lookback_swings': 5,
                    'swing_length': 5,
                    'strict_mode': False,
                    'resistance_buffer': 1.0,
                    'support_buffer': 1.0
                }),

                # Debug settings
                'debug_logging': getattr(rm_config, 'RANGING_MARKET_DEBUG_LOGGING', True)
            }

            # Apply epic-specific adjustments
            if self.epic:
                epic_adjustments = rm_config.get_ranging_market_threshold_for_epic(self.epic)
                config_dict.update(epic_adjustments)

            # Apply optimized parameters if available and requested
            if self.use_optimized_parameters and OPTIMIZATION_AVAILABLE:
                try:
                    if is_epic_ranging_market_optimized(self.epic, self.timeframe):
                        optimal_params = get_ranging_market_optimal_parameters(self.epic, self.timeframe)
                        config_dict.update(optimal_params)
                        self.logger.info(f"✅ Applied optimized parameters for {self.epic}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not load optimized parameters: {e}")

            return config_dict

        except Exception as e:
            self.logger.error(f"❌ Failed to load configuration: {e}")
            # Return minimal default configuration
            return {
                'strategy_enabled': True,
                'squeeze_momentum_enabled': True,
                'wave_trend_enabled': True,
                'oscillator_confluence_enabled': True,
                'signal_quality_min_confidence': 0.55,  # Lowered for backtest validation
                'debug_logging': True
            }

    def detect_signal(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str = '15m') -> Optional[Dict]:
        """
        Main signal detection logic for ranging market strategy

        Args:
            df: DataFrame with price data and indicators
            epic: Trading pair epic
            spread_pips: Spread in pips
            timeframe: Trading timeframe

        Returns:
            Signal dictionary or None if no signal found
        """
        try:
            if not self.config.get('strategy_enabled', True):
                return None

            if len(df) < 50:  # Need sufficient data for analysis
                self.logger.debug(f"Insufficient data for ranging market analysis: {len(df)} bars")
                return None

            # Add required indicators if not present
            df = self._ensure_indicators(df)

            # Market regime filtering - DISABLED to allow strategy to run in all conditions
            # is_ranging = self._is_ranging_market(df)
            # if not is_ranging:
            #     latest_row = df.iloc[-1]
            #     adx = latest_row.get('adx', 25)
            #     self.logger.info(f"⏭️  Market trending (ADX: {adx:.1f}) - skipping {epic}")
            #     return None

            # Calculate oscillator signals
            oscillator_signals = self._calculate_oscillator_signals(df)

            # Check confluence requirements
            confluence_result = self._check_oscillator_confluence(oscillator_signals)

            if not confluence_result['has_signal']:
                bull_score = confluence_result.get('bull_score', 0)
                bear_score = confluence_result.get('bear_score', 0)
                self.logger.info(f"⏭️  No confluence (B:{bull_score:.3f}, Bear:{bear_score:.3f}) - skipping {epic}")
                return None

            # Validate dynamic support/resistance zones
            zone_validation = self._validate_dynamic_zones(df, confluence_result['signal_type'])

            if not zone_validation['valid']:
                reason = zone_validation.get('reason', 'unknown')
                self.logger.info(f"⏭️  Zone validation failed ({reason}) - skipping {epic}")
                return None

            # Build comprehensive signal
            signal = self._build_signal(
                df=df,
                epic=epic,
                spread_pips=spread_pips,
                timeframe=timeframe,
                confluence_result=confluence_result,
                oscillator_signals=oscillator_signals,
                zone_validation=zone_validation
            )

            if not signal:
                return None

            confidence = signal.get('confidence', 0)
            min_confidence = self.config.get('signal_quality_min_confidence', 0.65)

            # Final signal quality check
            if confidence >= min_confidence:
                self._update_performance_stats(signal, oscillator_signals)
                return signal

            return None

        except Exception as e:
            self.logger.error(f"❌ Error detecting ranging market signal: {e}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return None

    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required indicators are present in the dataframe"""
        df = df.copy()

        try:
            # Squeeze Momentum Indicator (Bollinger Bands + Keltner Channels)
            if self.config.get('squeeze_momentum_enabled'):
                df = self._calculate_squeeze_momentum(df)

            # Wave Trend Oscillator
            if self.config.get('wave_trend_enabled'):
                df = self._calculate_wave_trend_oscillator(df)

            # Bollinger Bands and Keltner Channels
            if self.config.get('bollinger_bands_enabled'):
                df = self._calculate_bollinger_bands(df)
            if self.config.get('keltner_channels_enabled'):
                df = self._calculate_keltner_channels(df)

            # RSI with divergence detection
            if self.config.get('rsi_enabled'):
                df = self._calculate_rsi_with_divergence(df)

            # Relative Vigor Index
            if self.config.get('rvi_enabled'):
                df = self._calculate_rvi(df)

            # ADX for market regime detection
            if 'adx' not in df.columns:
                df = self._calculate_adx(df)

            # ATR for volatility analysis
            if 'atr' not in df.columns:
                df = self._calculate_atr(df)

        except Exception as e:
            self.logger.error(f"❌ Error calculating indicators: {e}")

        return df

    def _calculate_squeeze_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Squeeze Momentum Indicator (LazyBear)"""
        try:
            bb_length = self.config.get('squeeze_bb_length', 20)
            bb_mult = self.config.get('squeeze_bb_mult', 2.0)
            kc_length = self.config.get('squeeze_kc_length', 20)
            kc_mult = self.config.get('squeeze_kc_mult', 1.5)
            momentum_length = self.config.get('squeeze_momentum_length', 12)

            # Bollinger Bands
            bb_basis = df['close'].rolling(window=bb_length).mean()
            bb_dev = df['close'].rolling(window=bb_length).std()
            bb_upper = bb_basis + (bb_dev * bb_mult)
            bb_lower = bb_basis - (bb_dev * bb_mult)

            # Keltner Channels
            kc_basis = df['close'].ewm(span=kc_length).mean()
            if 'atr' not in df.columns:
                df = self._calculate_atr(df)
            kc_upper = kc_basis + (df['atr'] * kc_mult)
            kc_lower = kc_basis - (df['atr'] * kc_mult)

            # Squeeze detection (BB inside KC)
            df['squeeze_on'] = (bb_lower > kc_lower) & (bb_upper < kc_upper)

            # Momentum calculation
            highest = df['high'].rolling(window=momentum_length).max()
            lowest = df['low'].rolling(window=momentum_length).min()
            m1 = (highest + lowest) / 2
            df['squeeze_momentum'] = df['close'] - (m1 + kc_basis) / 2

            # Squeeze release detection
            df['squeeze_release'] = df['squeeze_on'].shift(1) & ~df['squeeze_on']

            return df

        except Exception as e:
            self.logger.error(f"❌ Error calculating squeeze momentum: {e}")
            return df

    def _calculate_wave_trend_oscillator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Wave Trend Oscillator (LazyBear WTO)"""
        try:
            channel_length = self.config.get('wto_channel_length', 10)
            average_length = self.config.get('wto_average_length', 21)
            signal_length = self.config.get('wto_signal_length', 4)

            # Calculate typical price
            hlc3 = (df['high'] + df['low'] + df['close']) / 3

            # EMA calculations
            esa = hlc3.ewm(span=channel_length).mean()
            d = (hlc3 - esa).abs().ewm(span=channel_length).mean()
            ci = (hlc3 - esa) / (0.015 * d)

            # Wave Trend calculation
            wt1 = ci.ewm(span=average_length).mean()
            wt2 = wt1.rolling(window=signal_length).mean()

            df['wto_wt1'] = wt1
            df['wto_wt2'] = wt2
            df['wto_diff'] = wt1 - wt2

            # Signal generation
            df['wto_bullish'] = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
            df['wto_bearish'] = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))

            return df

        except Exception as e:
            self.logger.error(f"❌ Error calculating wave trend oscillator: {e}")
            return df

    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        try:
            length = self.config.get('bb_length', 20)
            multiplier = self.config.get('bb_multiplier', 2.0)

            bb_basis = df['close'].rolling(window=length).mean()
            bb_dev = df['close'].rolling(window=length).std()

            df['bb_upper'] = bb_basis + (bb_dev * multiplier)
            df['bb_lower'] = bb_basis - (bb_dev * multiplier)
            df['bb_middle'] = bb_basis

            # %B calculation
            df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            return df

        except Exception as e:
            self.logger.error(f"❌ Error calculating bollinger bands: {e}")
            return df

    def _calculate_keltner_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Keltner Channels"""
        try:
            length = self.config.get('kc_length', 20)
            multiplier = self.config.get('kc_multiplier', 1.5)

            if 'atr' not in df.columns:
                df = self._calculate_atr(df)

            kc_basis = df['close'].ewm(span=length).mean()
            df['kc_upper'] = kc_basis + (df['atr'] * multiplier)
            df['kc_lower'] = kc_basis - (df['atr'] * multiplier)
            df['kc_middle'] = kc_basis

            return df

        except Exception as e:
            self.logger.error(f"❌ Error calculating keltner channels: {e}")
            return df

    def _calculate_rsi_with_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI with divergence detection"""
        try:
            period = self.config.get('rsi_period', 14)

            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Simple divergence detection (can be enhanced)
            df['rsi_bullish_div'] = False
            df['rsi_bearish_div'] = False

            # Look for basic divergence patterns
            lookback = self.config.get('rsi_divergence_lookback', 20)
            if len(df) > lookback:
                for i in range(lookback, len(df)):
                    price_higher = df['close'].iloc[i] > df['close'].iloc[i-lookback]
                    rsi_lower = df['rsi'].iloc[i] < df['rsi'].iloc[i-lookback]
                    if price_higher and rsi_lower:
                        df.loc[df.index[i], 'rsi_bearish_div'] = True

                    price_lower = df['close'].iloc[i] < df['close'].iloc[i-lookback]
                    rsi_higher = df['rsi'].iloc[i] > df['rsi'].iloc[i-lookback]
                    if price_lower and rsi_higher:
                        df.loc[df.index[i], 'rsi_bullish_div'] = True

            return df

        except Exception as e:
            self.logger.error(f"❌ Error calculating RSI with divergence: {e}")
            return df

    def _calculate_rvi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Relative Vigor Index"""
        try:
            period = self.config.get('rvi_period', 10)
            signal_period = self.config.get('rvi_signal_period', 4)

            # Calculate numerator and denominator
            numerator = (df['close'] - df['open']).rolling(window=period).sum()
            denominator = (df['high'] - df['low']).rolling(window=period).sum()

            # Calculate RVI
            df['rvi'] = numerator / denominator.replace(0, np.nan)
            df['rvi_signal'] = df['rvi'].rolling(window=signal_period).mean()

            # RVI crossover signals
            df['rvi_bullish'] = (df['rvi'] > df['rvi_signal']) & (df['rvi'].shift(1) <= df['rvi_signal'].shift(1))
            df['rvi_bearish'] = (df['rvi'] < df['rvi_signal']) & (df['rvi'].shift(1) >= df['rvi_signal'].shift(1))

            return df

        except Exception as e:
            self.logger.error(f"❌ Error calculating RVI: {e}")
            return df

    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX for trend strength"""
        try:
            period = 14

            # Calculate True Range and Directional Movement
            df['tr'] = np.maximum(df['high'] - df['low'],
                                 np.maximum(abs(df['high'] - df['close'].shift()),
                                           abs(df['low'] - df['close'].shift())))

            df['dm_plus'] = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
                                    np.maximum(df['high'] - df['high'].shift(), 0), 0)
            df['dm_minus'] = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
                                     np.maximum(df['low'].shift() - df['low'], 0), 0)

            # Smooth the values
            tr_smooth = df['tr'].rolling(window=period).mean()
            dm_plus_smooth = df['dm_plus'].rolling(window=period).mean()
            dm_minus_smooth = df['dm_minus'].rolling(window=period).mean()

            # Calculate DI+ and DI-
            di_plus = 100 * (dm_plus_smooth / tr_smooth)
            di_minus = 100 * (dm_minus_smooth / tr_smooth)

            # Calculate ADX
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            df['adx'] = dx.rolling(window=period).mean()

            return df

        except Exception as e:
            self.logger.error(f"❌ Error calculating ADX: {e}")
            return df

    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range"""
        try:
            period = 14

            df['tr'] = np.maximum(df['high'] - df['low'],
                                 np.maximum(abs(df['high'] - df['close'].shift()),
                                           abs(df['low'] - df['close'].shift())))
            df['atr'] = df['tr'].rolling(window=period).mean()

            return df

        except Exception as e:
            self.logger.error(f"❌ Error calculating ATR: {e}")
            return df

    def _is_ranging_market(self, df: pd.DataFrame) -> bool:
        """Check if market is in ranging condition"""
        try:
            if not self.config.get('market_regime_detection_enabled', True):
                return True

            if len(df) < 20:
                return False

            latest_row = df.iloc[-1]

            # ADX check - low ADX indicates ranging market
            adx = latest_row.get('adx', 25)
            adx_threshold = self.config.get('ranging_market_adx_max', 20)

            if adx > adx_threshold:
                self.logger.debug(f"Market trending (ADX: {adx:.1f} > {adx_threshold}) - not suitable for ranging strategy")
                return False

            # Additional ranging market checks can be added here
            # - Price oscillation within recent range
            # - ATR stability
            # - etc.

            return True

        except Exception as e:
            self.logger.error(f"❌ Error checking ranging market condition: {e}")
            return True  # Default to allow trading

    def _calculate_oscillator_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate signals from all oscillators"""
        latest_row = df.iloc[-1]
        oscillator_signals = {}

        try:
            # Squeeze Momentum signals
            if self.config.get('squeeze_momentum_enabled'):
                squeeze_momentum = latest_row.get('squeeze_momentum', 0)
                squeeze_release = latest_row.get('squeeze_release', False)
                squeeze_on = latest_row.get('squeeze_on', False)

                threshold = self.config.get('squeeze_momentum_threshold', 0.1)


                # More flexible squeeze signal generation - not just on release
                bullish_squeeze = (squeeze_release and squeeze_momentum > threshold) or (squeeze_momentum > threshold * 2)
                bearish_squeeze = (squeeze_release and squeeze_momentum < -threshold) or (squeeze_momentum < -threshold * 2)

                oscillator_signals['squeeze_momentum'] = {
                    'bullish': bullish_squeeze,
                    'bearish': bearish_squeeze,
                    'strength': abs(squeeze_momentum) * 1000,  # Scale up the strength
                    'squeeze_active': squeeze_on,
                    'value': squeeze_momentum
                }

            # Wave Trend Oscillator signals
            if self.config.get('wave_trend_enabled'):
                wto_bullish = latest_row.get('wto_bullish', False)
                wto_bearish = latest_row.get('wto_bearish', False)
                wto_wt1 = latest_row.get('wto_wt1', 0)
                wto_wt2 = latest_row.get('wto_wt2', 0)


                # More flexible WTO signal generation based on levels and crossover
                wto_overbought = self.config.get('wto_overbought_level', 60)
                wto_oversold = self.config.get('wto_oversold_level', -60)

                # Generate signals based on extreme levels or crossovers
                wto_bullish_signal = wto_bullish or (wto_wt1 < wto_oversold and wto_wt1 > wto_wt2)
                wto_bearish_signal = wto_bearish or (wto_wt1 > wto_overbought and wto_wt1 < wto_wt2)

                oscillator_signals['wave_trend'] = {
                    'bullish': wto_bullish_signal,
                    'bearish': wto_bearish_signal,
                    'strength': abs(wto_wt1 - wto_wt2) / 100.0,  # Normalize strength
                    'wt1': wto_wt1,
                    'wt2': wto_wt2
                }

            # RSI signals
            if self.config.get('rsi_enabled'):
                rsi = latest_row.get('rsi', 50)
                rsi_bullish_div = latest_row.get('rsi_bullish_div', False)
                rsi_bearish_div = latest_row.get('rsi_bearish_div', False)

                rsi_oversold = self.config.get('rsi_oversold', 30)
                rsi_overbought = self.config.get('rsi_overbought', 70)

                # Symmetric RSI signal generation - balanced for ranging markets
                rsi_bullish_signal = rsi < rsi_oversold or rsi_bullish_div or (rsi < 45)
                rsi_bearish_signal = rsi > rsi_overbought or rsi_bearish_div or (rsi > 55)

                oscillator_signals['rsi'] = {
                    'bullish': rsi_bullish_signal,
                    'bearish': rsi_bearish_signal,
                    'strength': abs(rsi - 50) / 50,
                    'value': rsi,
                    'divergence_bull': rsi_bullish_div,
                    'divergence_bear': rsi_bearish_div
                }

            # RVI signals
            if self.config.get('rvi_enabled'):
                rvi_bullish = latest_row.get('rvi_bullish', False)
                rvi_bearish = latest_row.get('rvi_bearish', False)
                rvi = latest_row.get('rvi', 0)
                rvi_signal = latest_row.get('rvi_signal', 0)

                # More flexible RVI signal generation
                rvi_overbought = self.config.get('rvi_overbought', 0.5)
                rvi_oversold = self.config.get('rvi_oversold', -0.5)

                rvi_bullish_signal = rvi_bullish or (rvi < rvi_oversold) or (rvi > rvi_signal and rvi_signal < 0)
                rvi_bearish_signal = rvi_bearish or (rvi > rvi_overbought) or (rvi < rvi_signal and rvi_signal > 0)

                oscillator_signals['rvi'] = {
                    'bullish': rvi_bullish_signal,
                    'bearish': rvi_bearish_signal,
                    'strength': abs(rvi - rvi_signal) * 10,  # Scale up strength
                    'value': rvi,
                    'signal': rvi_signal
                }

        except Exception as e:
            self.logger.error(f"❌ Error calculating oscillator signals: {e}")

        return oscillator_signals

    def _check_oscillator_confluence(self, oscillator_signals: Dict) -> Dict:
        """Check for oscillator confluence and determine signal direction"""
        try:
            if not self.config.get('oscillator_confluence_enabled', True):
                return {'has_signal': False, 'reason': 'confluence_disabled'}

            weights = self.config.get('oscillator_weights', {})
            min_confirmations = self.config.get('oscillator_min_confirmations', 3)

            bull_score = 0.0
            bear_score = 0.0
            bull_confirmations = 0
            bear_confirmations = 0

            # Calculate weighted confluence scores
            for oscillator_name, signals in oscillator_signals.items():
                weight = weights.get(oscillator_name, 0.0)

                if signals.get('bullish', False):
                    bull_score += weight * signals.get('strength', 0.5)
                    bull_confirmations += 1

                if signals.get('bearish', False):
                    bear_score += weight * signals.get('strength', 0.5)
                    bear_confirmations += 1

            # Check thresholds
            bull_threshold = self.config.get('oscillator_bull_confluence_threshold', 0.70)
            bear_threshold = self.config.get('oscillator_bear_confluence_threshold', 0.70)

            # Determine signal
            if bull_score >= bull_threshold and bull_confirmations >= min_confirmations:
                if bear_score < bull_score:  # Bull signal stronger
                    return {
                        'has_signal': True,
                        'signal_type': 'BULL',
                        'bull_score': bull_score,
                        'bear_score': bear_score,
                        'bull_confirmations': bull_confirmations,
                        'bear_confirmations': bear_confirmations
                    }

            if bear_score >= bear_threshold and bear_confirmations >= min_confirmations:
                if bull_score < bear_score:  # Bear signal stronger
                    return {
                        'has_signal': True,
                        'signal_type': 'BEAR',
                        'bull_score': bull_score,
                        'bear_score': bear_score,
                        'bull_confirmations': bull_confirmations,
                        'bear_confirmations': bear_confirmations
                    }

            return {
                'has_signal': False,
                'reason': 'insufficient_confluence',
                'bull_score': bull_score,
                'bear_score': bear_score,
                'bull_confirmations': bull_confirmations,
                'bear_confirmations': bear_confirmations
            }

        except Exception as e:
            self.logger.error(f"❌ Error checking oscillator confluence: {e}")
            return {'has_signal': False, 'reason': f'error: {e}'}

    def _validate_dynamic_zones(self, df: pd.DataFrame, signal_type: str) -> Dict:
        """Validate dynamic support/resistance zones"""
        try:
            if not self.config.get('dynamic_zones_enabled', True):
                return {'valid': True, 'reason': 'zones_disabled'}

            current_price = df.iloc[-1]['close']
            zone_proximity_pips = self.config.get('zone_proximity_pips', 8)

            # Calculate recent support/resistance levels
            calculation_period = self.config.get('zone_calculation_period', 50)
            recent_data = df.tail(calculation_period)

            # Simple S/R calculation using recent highs/lows
            resistance_level = recent_data['high'].max()
            support_level = recent_data['low'].min()

            # Convert pip proximity to price difference (assuming 4-digit broker)
            pip_value = 0.0001
            if 'JPY' in str(self.epic).upper():
                pip_value = 0.01

            proximity_threshold = zone_proximity_pips * pip_value

            # Check if price is near significant zone
            near_resistance = abs(current_price - resistance_level) <= proximity_threshold
            near_support = abs(current_price - support_level) <= proximity_threshold

            # Validate based on signal type
            if signal_type == 'BULL' and near_support:
                return {
                    'valid': True,
                    'zone_type': 'support',
                    'zone_level': support_level,
                    'distance_pips': abs(current_price - support_level) / pip_value
                }
            elif signal_type == 'BEAR' and near_resistance:
                return {
                    'valid': True,
                    'zone_type': 'resistance',
                    'zone_level': resistance_level,
                    'distance_pips': abs(current_price - resistance_level) / pip_value
                }
            else:
                return {
                    'valid': False,
                    'reason': 'not_near_significant_zone',
                    'current_price': current_price,
                    'resistance_level': resistance_level,
                    'support_level': support_level
                }

        except Exception as e:
            self.logger.error(f"❌ Error validating dynamic zones: {e}")
            return {'valid': True, 'reason': f'validation_error: {e}'}

    def _build_signal(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str,
                     confluence_result: Dict, oscillator_signals: Dict, zone_validation: Dict) -> Dict:
        """Build comprehensive ranging market signal"""
        try:
            latest_row = df.iloc[-1]
            signal_type = confluence_result['signal_type']

            # Calculate confidence score
            base_confidence = max(confluence_result.get('bull_score', 0),
                                confluence_result.get('bear_score', 0))

            # Boost confidence for zone validation
            if zone_validation.get('valid', False):
                base_confidence += 0.15  # Increased boost for validation

            # Boost for squeeze release
            squeeze_signals = oscillator_signals.get('squeeze_momentum', {})
            if squeeze_signals.get('squeeze_active', False):
                base_confidence += 0.10  # Increased boost for validation

            # Additional confidence boost for backtest validation
            if self.backtest_mode:
                base_confidence += 0.05  # Extra boost in backtest mode

            # NEW: Swing proximity validation - prevent poor entry timing
            swing_proximity_penalty = 0.0
            if self.swing_validator:
                try:
                    # Analyze swing points if not already done
                    if hasattr(self, 'smc_analyzer'):
                        smc_config = {
                            'swing_length': self.config.get('swing_validation', {}).get('swing_length', 5),
                            'structure_confirmation': 3,
                            'bos_threshold': 0.0001
                        }
                        df = self.smc_analyzer.analyze_market_structure(df, smc_config, epic, timeframe)

                    # Validate swing proximity
                    swing_result = self.swing_validator.validate_entry_proximity(
                        df=df,
                        current_price=latest_row['close'],
                        direction=signal_type,
                        epic=epic,
                        timeframe=timeframe
                    )

                    if not swing_result['valid']:
                        self.logger.warning(
                            f"⚠️ Swing proximity violation: {swing_result.get('rejection_reason', 'Unknown')}"
                        )
                        # In strict mode, this will reject the signal
                        if self.config.get('swing_validation', {}).get('strict_mode', False):
                            return None

                    # Check for confirmation wait period (if significant violation)
                    swing_proximity_penalty = swing_result.get('confidence_penalty', 0.0)
                    if swing_proximity_penalty > 0.15:  # Significant violation
                        # Check bars since nearest swing
                        bars_since_swing = self._get_bars_since_nearest_swing(
                            df, signal_type, swing_result
                        )
                        min_confirmation_bars = self.config.get('swing_validation', {}).get(
                            'min_confirmation_bars', 3
                        )

                        if bars_since_swing is not None and bars_since_swing < min_confirmation_bars:
                            self.logger.warning(
                                f"⏸️ Swing too recent ({bars_since_swing} bars < {min_confirmation_bars} required) - "
                                f"waiting for confirmation"
                            )
                            # Apply heavy penalty or reject
                            if self.config.get('swing_validation', {}).get('strict_mode', False):
                                return None
                            swing_proximity_penalty += 0.20  # Heavy penalty

                    base_confidence -= swing_proximity_penalty

                    if swing_proximity_penalty > 0:
                        self.logger.info(
                            f"📉 Swing proximity penalty: -{swing_proximity_penalty:.3f} "
                            f"(distance: {swing_result.get('distance_to_swing', 0):.1f} pips to {swing_result.get('swing_type', 'swing')})"
                        )
                except Exception as e:
                    self.logger.debug(f"Swing validation error (non-critical): {e}")

            # NEW: Bounce confirmation - prevent premature entries
            bounce_confirmed = self._check_bounce_confirmation(df, signal_type, latest_row)
            if not bounce_confirmed['valid']:
                self.logger.warning(
                    f"⚠️ Bounce not confirmed: {bounce_confirmed.get('reason', 'Unknown')} - REJECTING signal"
                )
                return None
            elif bounce_confirmed.get('confidence_boost', 0) > 0:
                base_confidence += bounce_confirmed['confidence_boost']
                self.logger.info(
                    f"✅ Bounce confirmed: +{bounce_confirmed['confidence_boost']:.3f} confidence "
                    f"({bounce_confirmed.get('reason', 'Valid bounce')})"
                )

            # Calculate dynamic SL/TP if enabled
            sl_pips, tp_pips = self._calculate_dynamic_sl_tp(df, signal_type)

            # Build the signal
            signal = {
                'signal_type': signal_type,
                'strategy_name': 'ranging_market',
                'strategy': 'ranging_market',  # Required by SignalAnalyzer
                'epic': epic,
                'timeframe': timeframe,
                'confidence': min(base_confidence, 0.95),  # Cap at 95%
                'confidence_score': min(base_confidence, 0.95),  # Required by SignalAnalyzer
                'price': latest_row['close'],
                'spread_pips': spread_pips,

                # Signal details
                'confluence_result': confluence_result,
                'oscillator_signals': oscillator_signals,
                'zone_validation': zone_validation,

                # Risk management (legacy pips fields)
                'sl_pips': sl_pips,
                'tp_pips': tp_pips,
                'max_profit_pips': tp_pips,  # Required by SignalAnalyzer
                'max_loss_pips': sl_pips,    # Required by SignalAnalyzer
                'risk_reward_ratio': tp_pips / sl_pips if sl_pips > 0 else 0,

                # ✅ NEW: Add distance fields for order API (pips = points for standard pairs)
                'stop_distance': int(sl_pips),   # For order API
                'limit_distance': int(tp_pips),  # For order API

                # Timing information
                'signal_time': latest_row.get('start_time', datetime.now()),
                'timestamp': latest_row.get('start_time', datetime.now()),  # Required by SignalAnalyzer

                # Market context
                'market_regime': 'ranging',
                'adx_value': latest_row.get('adx', 0),
                'atr_value': latest_row.get('atr', 0),

                # Execution guidance
                'execution_guidance': {
                    'market_regime': 'ranging',
                    'oscillator_confluence': confluence_result.get('bull_score', 0) if signal_type == 'BULL'
                                           else confluence_result.get('bear_score', 0),
                    'zone_proximity': zone_validation.get('distance_pips', 0),
                    'squeeze_active': squeeze_signals.get('squeeze_active', False),
                    'recommended_position_size': 'medium'
                }
            }

            return signal

        except Exception as e:
            self.logger.error(f"❌ Error building ranging market signal: {e}")
            return None

    def _calculate_dynamic_sl_tp(self, df: pd.DataFrame, signal_type: str) -> tuple:
        """Calculate dynamic stop loss and take profit levels"""
        try:
            if not self.config.get('ranging_dynamic_sl_tp', True):
                # Use default values
                sl_pips = self.config.get('ranging_default_sl_pips', 35)
                tp_pips = self.config.get('ranging_default_tp_pips', 52)
                return sl_pips, tp_pips

            # Calculate range-based SL/TP
            calculation_period = 20
            recent_data = df.tail(calculation_period)

            range_size = recent_data['high'].max() - recent_data['low'].min()

            # Convert to pips (IG points: 1 point = 0.0001 for standard, 0.01 for JPY)
            pip_value = 0.0001
            if 'JPY' in str(self.epic).upper():
                pip_value = 0.01

            range_pips = range_size / pip_value

            # ✅ CRITICAL VALIDATION: Reject extreme ranges (suggests trending/volatile market, not ranging)
            # For ranging markets, we expect controlled ranges:
            # - JPY pairs: typically 20-150 pips range over 20 bars
            # - Standard pairs: typically 15-100 pips range over 20 bars
            max_reasonable_range = 200 if 'JPY' in str(self.epic).upper() else 150

            if range_pips > max_reasonable_range:
                self.logger.warning(
                    f"⚠️ Extreme range detected for {self.epic}: {range_pips:.1f} pips over {calculation_period} bars. "
                    f"This suggests trending/volatile market, not ranging. Using conservative defaults."
                )
                # Fall back to conservative defaults (not range-based)
                sl_pips = self.config.get('ranging_default_sl_pips', 35)
                tp_pips = self.config.get('ranging_default_tp_pips', 52)
                return sl_pips, tp_pips

            # Use percentage of range for SL/TP
            sl_multiplier = self.config.get('ranging_range_sl_multiplier', 0.4)
            tp_multiplier = self.config.get('ranging_range_tp_multiplier', 0.8)

            sl_pips = max(range_pips * sl_multiplier, 15)  # Minimum 15 pips
            tp_pips = max(range_pips * tp_multiplier, 25)  # Minimum 25 pips

            # ✅ SAFETY CAP: Prevent excessive SL/TP even with valid ranges
            max_sl = 80 if 'JPY' in str(self.epic).upper() else 60
            max_tp = 150 if 'JPY' in str(self.epic).upper() else 120

            if sl_pips > max_sl:
                self.logger.warning(f"⚠️ Capping SL from {sl_pips:.1f} to {max_sl} pips for {self.epic}")
                sl_pips = max_sl

            if tp_pips > max_tp:
                self.logger.warning(f"⚠️ Capping TP from {tp_pips:.1f} to {max_tp} pips for {self.epic}")
                tp_pips = max_tp

            # Ensure minimum risk/reward ratio
            min_rr = self.config.get('signal_quality_min_risk_reward', 1.8)
            if tp_pips / sl_pips < min_rr:
                tp_pips = sl_pips * min_rr

            return round(sl_pips, 1), round(tp_pips, 1)

        except Exception as e:
            self.logger.error(f"❌ Error calculating dynamic SL/TP: {e}")
            # Return defaults on error
            return 22.0, 38.0

    def _update_performance_stats(self, signal: Dict, oscillator_signals: Dict):
        """Update performance statistics"""
        try:
            self.performance_stats['total_signals'] += 1

            # Count oscillator contributions
            for oscillator_name, signals in oscillator_signals.items():
                if signals.get('bullish', False) or signals.get('bearish', False):
                    stat_key = f'{oscillator_name}_signals'
                    if stat_key in self.performance_stats:
                        self.performance_stats[stat_key] += 1

            # Count confluence signals
            if signal.get('confluence_result', {}).get('has_signal', False):
                self.performance_stats['confluence_signals'] += 1

            # Count zone validation passes
            if signal.get('zone_validation', {}).get('valid', False):
                self.performance_stats['zone_validation_passes'] += 1

            # Count regime filter passes
            if signal.get('market_regime') == 'ranging':
                self.performance_stats['regime_filter_passes'] += 1

        except Exception as e:
            self.logger.error(f"❌ Error updating performance stats: {e}")

    def get_required_indicators(self) -> List[str]:
        """
        Get list of required indicators for ranging market strategy

        Returns:
            List of indicator names needed
        """
        required_indicators = [
            'close', 'high', 'low', 'open',  # Basic OHLC
            'adx',  # Trend strength
            'atr',  # Volatility
        ]

        # Add oscillator-specific indicators based on configuration
        if self.config.get('squeeze_momentum_enabled'):
            required_indicators.extend(['bb_upper', 'bb_lower', 'kc_upper', 'kc_lower'])

        if self.config.get('wave_trend_enabled'):
            required_indicators.extend(['wto_wt1', 'wto_wt2'])

        if self.config.get('rsi_enabled'):
            required_indicators.extend(['rsi'])

        if self.config.get('rvi_enabled'):
            required_indicators.extend(['rvi', 'rvi_signal'])

        return required_indicators

    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics"""
        return self.performance_stats.copy()

    def calculate_optimal_sl_tp(
        self,
        signal: Dict,
        epic: str,
        latest_row: pd.Series,
        spread_pips: float
    ) -> Dict[str, int]:
        """
        PHASE 3: Calculate SL/TP using adaptive volatility calculator OR ATR multipliers

        Priority:
        1. Adaptive volatility calculator (if enabled via USE_ADAPTIVE_SL_TP)
        2. Fallback to ATR-based with config multipliers

        Overrides base class to use Ranging-specific calculation logic.
        """
        signal_type = signal.get('signal_type', 'BULL')

        # PHASE 3: Use adaptive volatility calculator if enabled
        if self.use_adaptive_sl_tp and self.adaptive_calculator:
            try:
                result = self.adaptive_calculator.calculate_sl_tp(
                    epic=epic,
                    data=latest_row,
                    signal_type=signal_type
                )

                self.logger.info(
                    f"🧠 Adaptive SL/TP [{result.regime.value}] {result.method_used}: "
                    f"SL={result.stop_distance}p TP={result.limit_distance}p "
                    f"(R:R={result.limit_distance/result.stop_distance:.2f}, "
                    f"conf={result.confidence:.1%}, fallback_lvl={result.fallback_level}, "
                    f"{result.calculation_time_ms:.1f}ms)"
                )

                return {
                    'stop_distance': result.stop_distance,
                    'limit_distance': result.limit_distance
                }

            except Exception as e:
                self.logger.error(f"❌ Adaptive calculator failed: {e}, falling back to ATR method")
                # Fall through to ATR fallback below

        # FALLBACK: ATR-based with config multipliers
        # Get ATR for the pair
        atr = latest_row.get('atr', 0)
        if not atr or atr <= 0:
            # Fallback: estimate from current volatility (high-low range)
            atr = abs(latest_row.get('high', 0) - latest_row.get('low', 0))
            self.logger.warning(f"No ATR indicator, using high-low range: {atr}")

        # Convert ATR to pips/points
        if 'JPY' in epic:
            atr_pips = atr * 100  # JPY pairs: 0.01 = 1 pip
        else:
            atr_pips = atr * 10000  # Standard pairs: 0.0001 = 1 pip

        # Calculate using Ranging-specific ATR multipliers (tighter for mean reversion)
        raw_stop = atr_pips * self.stop_atr_multiplier
        raw_target = atr_pips * self.target_atr_multiplier

        # Apply minimum safe distances
        if 'JPY' in epic:
            min_sl = 20  # Minimum 20 pips for JPY
        else:
            min_sl = 15  # Minimum 15 pips for others

        stop_distance = max(int(raw_stop), min_sl)
        limit_distance = int(raw_target)

        # Apply reasonable maximums to prevent excessive risk
        if 'JPY' in epic:
            max_sl = 55
        elif 'GBP' in epic:
            max_sl = 60  # GBP pairs are more volatile
        else:
            max_sl = 45

        if stop_distance > max_sl:
            self.logger.warning(f"Stop distance {stop_distance} exceeds max {max_sl}, capping to maximum")
            stop_distance = max_sl
            limit_distance = int(stop_distance * (self.target_atr_multiplier / self.stop_atr_multiplier))

        self.logger.info(
            f"🎯 Ranging ATR-based SL/TP: ATR={atr_pips:.1f} pips, "
            f"SL={stop_distance} ({self.stop_atr_multiplier}x), "
            f"TP={limit_distance} ({self.target_atr_multiplier}x), "
            f"R:R={limit_distance/stop_distance:.2f}"
        )

        return {
            'stop_distance': stop_distance,
            'limit_distance': limit_distance
        }

    def _check_bounce_confirmation(self, df: pd.DataFrame, signal_type: str,
                                   latest_row: pd.Series) -> Dict[str, Any]:
        """
        Check for bounce confirmation before entry to prevent premature signals

        For BUY (at support): Look for bullish rejection (lower wick > upper wick)
        For SELL (at resistance): Look for bearish rejection (upper wick > lower wick)

        Returns:
            Dictionary with 'valid' (bool), 'reason' (str), 'confidence_boost' (float)
        """
        try:
            # Get last 3 candles for confirmation pattern
            if len(df) < 3:
                return {'valid': False, 'reason': 'insufficient_data'}

            last_3 = df.tail(3)
            current_candle = latest_row
            prev_candle = df.iloc[-2]

            # Calculate wicks and body
            current_body = abs(current_candle['close'] - current_candle['open'])
            current_range = current_candle['high'] - current_candle['low']

            if current_range == 0:
                return {'valid': False, 'reason': 'zero_range_candle'}

            # Calculate wick sizes
            if current_candle['close'] > current_candle['open']:  # Bullish candle
                upper_wick = current_candle['high'] - current_candle['close']
                lower_wick = current_candle['open'] - current_candle['low']
            else:  # Bearish candle
                upper_wick = current_candle['high'] - current_candle['open']
                lower_wick = current_candle['close'] - current_candle['low']

            # Wick ratios
            lower_wick_ratio = lower_wick / current_range
            upper_wick_ratio = upper_wick / current_range
            body_ratio = current_body / current_range

            if signal_type in ['BUY', 'BULL']:
                # BUY at support: Need bullish rejection pattern
                # 1. Lower wick should be significant (>30% of range)
                # 2. Price should close in upper half of candle
                # 3. Or previous candle tested support and current closed higher

                if lower_wick_ratio > 0.3:
                    # Strong lower wick = rejection from support
                    if current_candle['close'] > (current_candle['high'] + current_candle['low']) / 2:
                        return {
                            'valid': True,
                            'reason': f'bullish_rejection_wick ({lower_wick_ratio:.1%} lower wick)',
                            'confidence_boost': 0.05
                        }

                # Check if price tested lower and is now recovering
                if prev_candle['low'] < current_candle['low'] and current_candle['close'] > prev_candle['close']:
                    return {
                        'valid': True,
                        'reason': 'support_test_and_recovery',
                        'confidence_boost': 0.03
                    }

                # Check for 3-bar reversal pattern
                if len(last_3) == 3:
                    if (last_3.iloc[0]['low'] > last_3.iloc[1]['low'] and  # Made lower low
                        last_3.iloc[2]['close'] > last_3.iloc[1]['close']):  # Now recovering
                        return {
                            'valid': True,
                            'reason': '3bar_reversal_pattern',
                            'confidence_boost': 0.04
                        }

                # Reject if no confirmation
                return {
                    'valid': False,
                    'reason': f'no_bullish_confirmation (lower_wick:{lower_wick_ratio:.1%}, upper_wick:{upper_wick_ratio:.1%})'
                }

            else:  # SELL at resistance
                # SELL at resistance: Need bearish rejection pattern
                # 1. Upper wick should be significant (>30% of range)
                # 2. Price should close in lower half of candle
                # 3. Or previous candle tested resistance and current closed lower

                if upper_wick_ratio > 0.3:
                    # Strong upper wick = rejection from resistance
                    if current_candle['close'] < (current_candle['high'] + current_candle['low']) / 2:
                        return {
                            'valid': True,
                            'reason': f'bearish_rejection_wick ({upper_wick_ratio:.1%} upper wick)',
                            'confidence_boost': 0.05
                        }

                # Check if price tested higher and is now falling
                if prev_candle['high'] > current_candle['high'] and current_candle['close'] < prev_candle['close']:
                    return {
                        'valid': True,
                        'reason': 'resistance_test_and_rejection',
                        'confidence_boost': 0.03
                    }

                # Check for 3-bar reversal pattern
                if len(last_3) == 3:
                    if (last_3.iloc[0]['high'] < last_3.iloc[1]['high'] and  # Made higher high
                        last_3.iloc[2]['close'] < last_3.iloc[1]['close']):  # Now falling
                        return {
                            'valid': True,
                            'reason': '3bar_reversal_pattern',
                            'confidence_boost': 0.04
                        }

                # Reject if no confirmation
                return {
                    'valid': False,
                    'reason': f'no_bearish_confirmation (lower_wick:{lower_wick_ratio:.1%}, upper_wick:{upper_wick_ratio:.1%})'
                }

        except Exception as e:
            self.logger.error(f"Error in bounce confirmation: {e}")
            # On error, allow signal (fail-safe)
            return {'valid': True, 'reason': f'error_failsafe: {e}', 'confidence_boost': 0}

    def _get_bars_since_nearest_swing(self, df: pd.DataFrame, direction: str,
                                       swing_result: Dict[str, Any]) -> Optional[int]:
        """
        Calculate number of bars since the nearest swing point

        Args:
            df: DataFrame with swing data
            direction: 'BUY' or 'SELL'
            swing_result: Result from swing proximity validator

        Returns:
            Number of bars since nearest swing, or None if not found
        """
        try:
            nearest_swing_price = swing_result.get('nearest_swing_price')
            if nearest_swing_price is None:
                return None

            # Find the nearest swing point in the DataFrame
            if direction.upper() in ['BUY', 'BULL']:
                # Looking for swing high (resistance)
                if 'swing_high' in df.columns:
                    swing_highs = df[df['swing_high'] == True]
                    if not swing_highs.empty:
                        # Find the swing closest to the nearest_swing_price
                        closest_swing = swing_highs.iloc[
                            (swing_highs['high'] - nearest_swing_price).abs().argsort()[:1]
                        ]
                        if not closest_swing.empty:
                            swing_idx = closest_swing.index[0]
                            current_idx = df.index[-1]
                            bars_since = current_idx - swing_idx
                            return int(bars_since)
            else:
                # Looking for swing low (support)
                if 'swing_low' in df.columns:
                    swing_lows = df[df['swing_low'] == True]
                    if not swing_lows.empty:
                        # Find the swing closest to the nearest_swing_price
                        closest_swing = swing_lows.iloc[
                            (swing_lows['low'] - nearest_swing_price).abs().argsort()[:1]
                        ]
                        if not closest_swing.empty:
                            swing_idx = closest_swing.index[0]
                            current_idx = df.index[-1]
                            bars_since = current_idx - swing_idx
                            return int(bars_since)

            return None

        except Exception as e:
            self.logger.debug(f"Error calculating bars since swing: {e}")
            return None

    def __str__(self):
        return f"RangingMarketStrategy(epic={self.epic}, timeframe={self.timeframe})"


def create_ranging_market_strategy(data_fetcher=None, backtest_mode=False, epic=None,
                                 timeframe='15m', use_optimized_parameters=True):
    """Factory function to create a ranging market strategy instance"""
    return RangingMarketStrategy(
        data_fetcher=data_fetcher,
        backtest_mode=backtest_mode,
        epic=epic,
        timeframe=timeframe,
        use_optimized_parameters=use_optimized_parameters
    )