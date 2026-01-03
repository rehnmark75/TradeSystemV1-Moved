# core/strategies/mean_reversion_strategy.py
"""
Mean Reversion Strategy Implementation - Multi-Oscillator Confluence Approach
🎯 ADVANCED MEAN REVERSION: Based on RAG analysis findings for optimal oscillator combinations
📊 MULTI-OSCILLATOR: LuxAlgo Premium + Multi-timeframe RSI + Divergence + Squeeze Momentum
⚡ ORCHESTRATOR: Main class coordinates specialized helper modules
🏗️ MODULAR: Clean separation of concerns with focused helper modules

Features:
- LuxAlgo Premium Oscillator (primary mean reversion engine)
- Multi-timeframe RSI confluence analysis
- RSI-EMA divergence detection for reversal patterns
- Squeeze Momentum Indicator for timing optimization
- Mean reversion zone validation
- Market regime filtering (avoid strong trends)
- Database optimization parameter support
- Compatible with existing backtest system
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster
from .helpers.mean_reversion_indicator_calculator import MeanReversionIndicatorCalculator
from .helpers.mean_reversion_signal_detector import MeanReversionSignalDetector
from .helpers.mean_reversion_trend_validator import MeanReversionTrendValidator

# Import optimization functions
try:
    from optimization.optimal_parameter_service import get_mean_reversion_optimal_parameters, is_epic_mean_reversion_optimized
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    def get_mean_reversion_optimal_parameters(*args, **kwargs):
        raise ImportError("Optimization service not available")
    def is_epic_mean_reversion_optimized(*args, **kwargs):
        return False

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config

try:
    from forex_scanner.configdata.strategies import config_mean_reversion_strategy as mr_config
except ImportError:
    import configdata.strategies.config_mean_reversion_strategy as mr_config


class MeanReversionStrategy(BaseStrategy):
    """
    🎯 ADVANCED MEAN REVERSION STRATEGY: Multi-oscillator confluence approach

    Multi-oscillator mean reversion strategy based on RAG analysis findings.
    Uses sophisticated oscillator confluence methodology to identify high-probability
    mean reversion opportunities in ranging and choppy market conditions.

    Core Components:
    - LuxAlgo Premium Oscillator (primary mean reversion engine)
    - Multi-timeframe RSI analysis (confluence confirmation)
    - RSI-EMA divergence detection (reversal pattern identification)
    - Squeeze Momentum Indicator (timing optimization)
    - Mean reversion zone validation (statistical support/resistance)
    - Market regime filtering (avoid inappropriate market conditions)
    """

    def __init__(self, data_fetcher=None, backtest_mode: bool = False, epic: str = None,
                 timeframe: str = '15m', use_optimized_parameters: bool = True, pipeline_mode: bool = True):
        # Initialize parent
        self.name = 'mean_reversion'
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self._validator = None  # Use helper modules instead
        self.epic = epic
        self.timeframe = timeframe
        self.use_optimized_parameters = use_optimized_parameters

        # Basic initialization
        self.backtest_mode = backtest_mode
        self.price_adjuster = PriceAdjuster()
        self.data_fetcher = data_fetcher

        # Load mean reversion configuration
        self.mr_config = self._get_mean_reversion_parameters()

        # Basic parameters
        self.eps = 1e-8  # Epsilon for stability
        self.min_confidence = self.mr_config.get('min_confidence', mr_config.SIGNAL_QUALITY_MIN_CONFIDENCE)
        self.min_bars = self.mr_config.get('min_data_periods', mr_config.MEAN_REVERSION_MIN_DATA_PERIODS)

        # Enable/disable expensive features based on pipeline mode
        self.enhanced_validation = pipeline_mode and getattr(config, 'MEAN_REVERSION_ENHANCED_VALIDATION', True)

        # Initialize helper modules (orchestrator pattern)
        self.indicator_calculator = MeanReversionIndicatorCalculator(logger=self.logger)
        self.signal_detector = MeanReversionSignalDetector(
            logger=self.logger,
            indicator_calculator=self.indicator_calculator,
            enhanced_validation=self.enhanced_validation
        )
        self.trend_validator = MeanReversionTrendValidator(logger=self.logger, enhanced_validation=self.enhanced_validation)

        self.logger.info(f"🎯 Mean Reversion Strategy initialized for {epic or 'default'} ({timeframe})")
        self.logger.info(f"🔧 Using multi-oscillator confluence approach with 3 specialized helpers")
        self.logger.info(f"📊 Config: Min confidence {self.min_confidence:.1%}, Min bars {self.min_bars}")
        self.logger.info(f"🧠 Smart Money: Disabled (traditional technical analysis strategy)")

        if self.enhanced_validation:
            self.logger.info(f"🔍 Enhanced validation ENABLED - Full mean reversion analysis")
        else:
            self.logger.info(f"🔧 Enhanced validation DISABLED - Basic mean reversion testing mode")

        if backtest_mode:
            self.logger.info("🔥 BACKTEST MODE: Enhanced signal validation enabled")

    def should_enable_smart_money(self) -> bool:
        """
        Mean Reversion strategy does NOT use smart money concepts.
        It relies on traditional technical analysis (oscillators, RSI, divergences).
        """
        return False

    def _get_mean_reversion_parameters(self) -> Dict:
        """Get mean reversion parameters from optimization database or config fallback"""
        try:
            # First priority: Use optimized parameters from database if available
            self.logger.info(f"🔍 Parameter lookup: use_optimized={self.use_optimized_parameters}, "
                           f"available={OPTIMIZATION_AVAILABLE}, epic={self.epic}, timeframe={self.timeframe}")

            if self.epic and self.timeframe:
                is_optimized = is_epic_mean_reversion_optimized(self.epic, self.timeframe) if OPTIMIZATION_AVAILABLE else False
                self.logger.info(f"🔍 is_epic_mean_reversion_optimized({self.epic}, {self.timeframe}) = {is_optimized}")
            else:
                self.logger.info(f"🔍 Missing epic or timeframe: epic={self.epic}, timeframe={self.timeframe}")
                is_optimized = False

            # Check if we should use optimization
            final_condition = (
                self.use_optimized_parameters and
                OPTIMIZATION_AVAILABLE and
                self.epic and
                is_optimized
            )

            self.logger.info(f"🔍 Final optimization condition: {final_condition}")

            if final_condition:
                self.logger.info("🎯 USING OPTIMIZED PARAMETERS")
                try:
                    optimal_params = get_mean_reversion_optimal_parameters(self.epic, self.timeframe)
                    self.logger.info(f"✅ Using OPTIMIZED mean reversion parameters for {self.epic} ({self.timeframe}): "
                                   f"Confidence: {optimal_params.confidence_threshold:.1%}, "
                                   f"Performance: {optimal_params.performance_score:.3f}")

                    return {
                        'min_confidence': optimal_params.confidence_threshold,
                        'bull_confluence_threshold': optimal_params.bull_confluence_threshold,
                        'bear_confluence_threshold': optimal_params.bear_confluence_threshold,
                        'luxalgo_length': optimal_params.luxalgo_length,
                        'luxalgo_smoothing': optimal_params.luxalgo_smoothing,
                        'mtf_rsi_period': optimal_params.mtf_rsi_period,
                        'mtf_min_alignment': optimal_params.mtf_min_alignment,
                        'rsi_ema_period': optimal_params.rsi_ema_period,
                        'squeeze_bb_length': optimal_params.squeeze_bb_length,
                        'squeeze_kc_length': optimal_params.squeeze_kc_length,
                        'min_data_periods': optimal_params.min_data_periods,
                        'stop_loss_pips': optimal_params.stop_loss_pips,
                        'take_profit_pips': optimal_params.take_profit_pips,
                        'market_regime_enabled': optimal_params.market_regime_enabled,
                        'zone_validation_enabled': optimal_params.zone_validation_enabled
                    }

                except Exception as opt_e:
                    self.logger.error(f"❌ Optimization parameter retrieval failed: {opt_e}")
                    # Fall through to config defaults

            # Fallback: Use configuration defaults
            self.logger.info(f"📊 Using CONFIG defaults for {self.epic or 'default'} ({self.timeframe})")

            return {
                'min_confidence': mr_config.SIGNAL_QUALITY_MIN_CONFIDENCE,
                'bull_confluence_threshold': mr_config.OSCILLATOR_BULL_CONFLUENCE_THRESHOLD,
                'bear_confluence_threshold': mr_config.OSCILLATOR_BEAR_CONFLUENCE_THRESHOLD,
                'luxalgo_length': mr_config.LUXALGO_LENGTH,
                'luxalgo_smoothing': mr_config.LUXALGO_SMOOTHING,
                'mtf_rsi_period': mr_config.MTF_RSI_PERIOD,
                'mtf_min_alignment': mr_config.MTF_RSI_MIN_ALIGNMENT,
                'rsi_ema_period': mr_config.RSI_EMA_PERIOD,
                'squeeze_bb_length': mr_config.SQUEEZE_BB_LENGTH,
                'squeeze_kc_length': mr_config.SQUEEZE_KC_LENGTH,
                'min_data_periods': mr_config.MEAN_REVERSION_MIN_DATA_PERIODS,
                'stop_loss_pips': mr_config.MEAN_REVERSION_DEFAULT_SL_PIPS,
                'take_profit_pips': mr_config.MEAN_REVERSION_DEFAULT_TP_PIPS,
                'market_regime_enabled': mr_config.MARKET_REGIME_DETECTION_ENABLED,
                'zone_validation_enabled': mr_config.MEAN_REVERSION_ZONE_ENABLED
            }

        except Exception as e:
            self.logger.warning(f"Could not load mean reversion config: {e}, using fallback defaults")
            # Last resort: hardcoded defaults
            return {
                'min_confidence': 0.6,
                'bull_confluence_threshold': 0.65,
                'bear_confluence_threshold': 0.65,
                'luxalgo_length': 14,
                'luxalgo_smoothing': 3,
                'mtf_rsi_period': 14,
                'mtf_min_alignment': 0.6,
                'rsi_ema_period': 21,
                'squeeze_bb_length': 20,
                'squeeze_kc_length': 20,
                'min_data_periods': 100,
                'stop_loss_pips': 25.0,
                'take_profit_pips': 40.0,
                'market_regime_enabled': True,
                'zone_validation_enabled': True
            }

    def get_required_indicators(self) -> List[str]:
        """Required indicators for mean reversion strategy"""
        return self.indicator_calculator.get_required_indicators()

    def detect_signal(
        self,
        df: pd.DataFrame,
        epic: str,
        spread_pips: float = 1.5,
        timeframe: str = '15m',
        evaluation_time: str = None
    ) -> Optional[Dict]:
        """
        🎯 CORE SIGNAL DETECTION: Multi-oscillator confluence mean reversion

        Signal Logic:
        1. Calculate all mean reversion indicators if not present
        2. Detect primary oscillator signals (LuxAlgo oversold/overbought)
        3. Apply multi-oscillator confluence filtering
        4. Validate market regime suitability (avoid strong trends)
        5. Apply mean reversion zone validation
        6. Generate bull/bear signals with confidence scoring
        """

        try:
            # Validate data requirements
            if not self.indicator_calculator.validate_data_requirements(df, self.min_bars):
                self.logger.debug(f"Insufficient data: {len(df)} bars (need {self.min_bars})")
                return None

            self.logger.debug(f"Processing {len(df)} bars for {epic} mean reversion analysis")

            # 1. Calculate all mean reversion indicators if not present
            df_enhanced = self.indicator_calculator.ensure_mean_reversion_indicators(df.copy())

            # 2. Detect mean reversion signals using multi-oscillator confluence
            df_with_signals = self.signal_detector.detect_mean_reversion_signals(
                df_enhanced, epic, is_backtest=self.backtest_mode
            )

            # 3. Check for signals - handle both backtest and live modes
            if self.backtest_mode:
                # BACKTEST MODE: Collect all signals and return the latest valid one
                all_signals = []

                for idx, row in df_with_signals.iterrows():
                    bull_signal = row.get('mean_reversion_bull', False)
                    bear_signal = row.get('mean_reversion_bear', False)

                    if bull_signal or bear_signal:
                        signal_type = 'BULL' if bull_signal else 'BEAR'
                        signal = self._check_immediate_signal(
                            row, epic, timeframe, spread_pips, signal_type, idx
                        )
                        if signal:
                            # Add timing info for backtest
                            if hasattr(row, 'name'):
                                signal['signal_time'] = row.name
                            elif 'start_time' in row:
                                signal['signal_time'] = row['start_time']
                            all_signals.append((idx, signal))

                # Return the latest signal (signal spacing already applied in detector)
                if all_signals:
                    all_signals.sort(key=lambda x: x[0])
                    latest_signal = all_signals[-1][1]
                    self.logger.debug(
                        f"🎯 Backtest: Found {len(all_signals)} valid signals, "
                        f"returning latest at {latest_signal.get('signal_time', 'unknown')}"
                    )
                    return latest_signal

                return None

            else:
                # LIVE MODE: Only check latest bar
                latest_row = df_with_signals.iloc[-1]

                # Check for signals
                bull_signal = latest_row.get('mean_reversion_bull', False)
                bear_signal = latest_row.get('mean_reversion_bear', False)

                if bull_signal or bear_signal:
                    signal_type = 'BULL' if bull_signal else 'BEAR'
                    self.logger.info(f"🎯 Mean Reversion {signal_type} signal detected!")

                    signal = self._check_immediate_signal(
                        latest_row, epic, timeframe, spread_pips, signal_type, len(df) - 1
                    )
                    if signal:
                        return signal

                return None

        except Exception as e:
            self.logger.error(f"Mean reversion signal detection error: {e}")
            return None

    def _check_immediate_signal(
        self,
        latest_row: pd.Series,
        epic: str,
        timeframe: str,
        spread_pips: float,
        signal_type: str,
        bar_idx: int
    ) -> Optional[Dict]:
        """Check immediate signal with comprehensive validation"""
        try:
            self.logger.debug(f"🎯 Validating {signal_type} mean reversion signal at bar {bar_idx}")

            # Create DataFrame context for trend validation
            # (In a real implementation, we'd pass the full DataFrame)
            # For now, we'll create a minimal context
            df_context = pd.DataFrame([latest_row])

            # 1. Validate trend conditions using trend validator
            validation_results = self.trend_validator.validate_all_trend_filters(
                df_context, 0, signal_type, epic
            )

            # Apply mean reversion validation logic
            validation_passed = self.trend_validator.apply_mean_reversion_validation(
                validation_results, min_pass_rate=0.6
            )

            if not validation_passed:
                failed_filters = [
                    name for name, result in validation_results.items()
                    if not result and name not in ['overall_pass_rate', 'all_passed']
                ]
                self.logger.info(f"❌ {signal_type} signal REJECTED: Failed validation filters: {failed_filters}")
                return None

            # 2. Validate signal strength using signal detector
            signal_strength_valid = self.signal_detector.validate_signal_strength(
                latest_row, signal_type, epic
            )

            if not signal_strength_valid:
                self.logger.info(f"❌ {signal_type} signal REJECTED: Insufficient signal strength")
                return None

            # 3. Create signal if all validations pass
            signal = self._create_signal(
                signal_type=signal_type,
                epic=epic,
                timeframe=timeframe,
                latest_row=latest_row,
                spread_pips=spread_pips
            )

            if signal:
                confidence = signal.get('confidence', 0)
                self.logger.debug(f"✅ {signal_type} mean reversion signal generated: {confidence:.1%} confidence")
                return signal
            else:
                self.logger.info(f"❌ {signal_type} signal creation failed")
                return None

        except Exception as e:
            self.logger.error(f"Error checking immediate signal: {e}")
            return None

    def _create_signal(
        self,
        signal_type: str,
        epic: str,
        timeframe: str,
        latest_row: pd.Series,
        spread_pips: float
    ) -> Optional[Dict]:
        """Create a mean reversion signal dictionary with all required fields"""
        try:
            # CRITICAL FIX: Ensure spread_pips is numeric for price calculations
            if isinstance(spread_pips, str):
                try:
                    spread_pips = float(spread_pips)
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid spread_pips value '{spread_pips}', using default 1.5")
                    spread_pips = 1.5

            # Create base signal using parent method
            signal = self.create_base_signal(signal_type, epic, timeframe, latest_row)

            # Add mean reversion specific data
            signal.update({
                'strategy': 'mean_reversion',
                'luxalgo_oscillator': latest_row.get('luxalgo_oscillator', 50),
                'luxalgo_signal': latest_row.get('luxalgo_signal', 50),
                'luxalgo_histogram': latest_row.get('luxalgo_histogram', 0),
                'oscillator_bull_score': latest_row.get('oscillator_bull_score', 0),
                'oscillator_bear_score': latest_row.get('oscillator_bear_score', 0),
                'mtf_bull_alignment': latest_row.get('mtf_bull_alignment', 0),
                'mtf_bear_alignment': latest_row.get('mtf_bear_alignment', 0),
                'rsi_ema_divergence_bull': latest_row.get('rsi_ema_divergence_bull', False),
                'rsi_ema_divergence_bear': latest_row.get('rsi_ema_divergence_bear', False),
                'divergence_strength': latest_row.get('divergence_strength', 0),
                'squeeze_momentum': latest_row.get('squeeze_momentum', 0),
                'squeeze_on': latest_row.get('squeeze_on', False),
                'mr_signal_strength': latest_row.get('mr_signal_strength', 0)
            })

            # Get confidence from signal detector's calculation
            confidence = latest_row.get('mr_confidence', 0)

            # Enhance confidence if not already calculated
            if confidence == 0:
                confidence = self.signal_detector._calculate_individual_signal_confidence(
                    latest_row, signal_type, epic
                )

            # Apply minimum confidence threshold
            if confidence < self.min_confidence:
                self.logger.debug(
                    f"{signal_type} signal rejected: confidence {confidence:.3f} < threshold {self.min_confidence:.3f}"
                )
                return None

            # Add confidence to signal
            signal['confidence'] = confidence
            signal['confidence_score'] = confidence  # For compatibility

            # Add execution prices
            signal = self.add_execution_prices(signal, spread_pips)

            # ✅ Calculate SL/TP using base class method
            sl_tp = self.calculate_optimal_sl_tp(signal, epic, latest_row, spread_pips)
            signal['stop_distance'] = sl_tp['stop_distance']
            signal['limit_distance'] = sl_tp['limit_distance']

            self.logger.info(
                f"🎯 Mean Reversion {signal_type} signal: "
                f"confidence={confidence:.1%}, SL={sl_tp['stop_distance']}p, TP={sl_tp['limit_distance']}p"
            )

            # Add mean reversion specific execution guidance
            signal['execution_guidance'] = {
                'strategy_type': 'mean_reversion',
                'market_regime': self._get_market_regime(latest_row),
                'oscillator_extremity': self._get_oscillator_extremity(latest_row, signal_type),
                'recommended_position_size': self._get_recommended_position_size(latest_row, epic),
                'risk_level': self._assess_risk_level(latest_row, signal_type)
            }

            return signal

        except Exception as e:
            self.logger.error(f"Error creating mean reversion signal: {e}")
            return None

    def _get_market_regime(self, row: pd.Series) -> str:
        """Determine current market regime for execution guidance"""
        try:
            adx = row.get('adx', 25)
            if adx > 50:
                return 'strong_trend'
            elif adx > 25:
                return 'moderate_trend'
            else:
                return 'ranging'
        except:
            return 'unknown'

    def _get_oscillator_extremity(self, row: pd.Series, signal_type: str) -> str:
        """Assess oscillator extremity level"""
        try:
            luxalgo = row.get('luxalgo_oscillator', 50)
            if signal_type == 'BULL':
                if luxalgo < 10:
                    return 'extreme_oversold'
                elif luxalgo < 20:
                    return 'oversold'
                else:
                    return 'mild_oversold'
            else:  # BEAR
                if luxalgo > 90:
                    return 'extreme_overbought'
                elif luxalgo > 80:
                    return 'overbought'
                else:
                    return 'mild_overbought'
        except:
            return 'unknown'

    def _get_recommended_position_size(self, row: pd.Series, epic: str) -> str:
        """Get recommended position size based on signal strength"""
        try:
            signal_strength = row.get('mr_signal_strength', 0.5)
            confidence = row.get('mr_confidence', 0.5)

            combined_score = (signal_strength + confidence) / 2

            if combined_score > 0.8:
                return 'large'
            elif combined_score > 0.6:
                return 'medium'
            else:
                return 'small'
        except:
            return 'small'

    def _assess_risk_level(self, row: pd.Series, signal_type: str) -> str:
        """Assess risk level for the signal"""
        try:
            # Check market regime
            adx = row.get('adx', 25)
            atr = row.get('atr', 0)
            confidence = row.get('mr_confidence', 0.5)

            risk_factors = 0

            # High ADX = higher risk for mean reversion
            if adx > 40:
                risk_factors += 2
            elif adx > 25:
                risk_factors += 1

            # High volatility = higher risk
            if atr > 0:  # Compare to recent average if available
                risk_factors += 1

            # Low confidence = higher risk
            if confidence < 0.7:
                risk_factors += 1

            if risk_factors >= 3:
                return 'high'
            elif risk_factors >= 2:
                return 'medium'
            else:
                return 'low'
        except:
            return 'medium'

    def create_enhanced_signal_data(self, latest_row: pd.Series, signal_type: str) -> Dict:
        """Create signal data for BaseStrategy compatibility"""
        try:
            # Get mean reversion specific data
            luxalgo_osc = latest_row.get('luxalgo_oscillator', 50)
            confluence_score = latest_row.get(f'oscillator_{signal_type.lower()}_score', 0)

            # Basic price data
            close = latest_row.get('close', 0)
            open_price = latest_row.get('open', close)
            high = latest_row.get('high', close)
            low = latest_row.get('low', close)

            # Calculate efficiency ratio from confluence score
            efficiency_ratio = min(1.0, confluence_score) if confluence_score > 0 else 0.5

            return {
                'mean_reversion_data': {
                    'luxalgo_oscillator': luxalgo_osc,
                    'confluence_score': confluence_score,
                    'divergence_strength': latest_row.get('divergence_strength', 0)
                },
                'oscillator_data': {
                    'rsi_14': latest_row.get('rsi_14', 50),
                    'mtf_alignment': latest_row.get(f'mtf_{signal_type.lower()}_alignment', 0),
                    'squeeze_momentum': latest_row.get('squeeze_momentum', 0)
                },
                'trend_data': {
                    'ema_200': latest_row.get('ema_200', 0),
                    'adx': latest_row.get('adx', 25),
                    'market_regime': self._get_market_regime(latest_row)
                },
                'other_indicators': {
                    'atr': latest_row.get('atr', high - low if high > low else 0.0001),
                    'vwap': latest_row.get('vwap', close),
                    'vwap_deviation': latest_row.get('vwap_deviation', 0),
                    'volume': latest_row.get('ltv', 1000),
                    'efficiency_ratio': efficiency_ratio
                },
                'price_data': {
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close
                }
            }

        except Exception as e:
            self.logger.error(f"Error creating enhanced signal data: {e}")
            return {}

    def enable_forex_integration(self, epic):
        """Enable forex integration for specific pair - compatibility method"""
        self.logger.debug(f"Forex integration enabled for {epic} (mean reversion strategy)")
        pass


class LegacyMeanReversionStrategy(MeanReversionStrategy):
    """
    🔄 LEGACY COMPATIBILITY: Wrapper for backward compatibility

    Ensures compatibility with any existing code that depends on different method names
    or interfaces while using the new implementation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("📦 Legacy MeanReversionStrategy wrapper initialized")

        # Add compatibility attributes
        self.enhanced_validator = self.signal_detector
        self.oscillator_calculator = self.indicator_calculator
        self.zone_validator = self.trend_validator


def create_mean_reversion_strategy(data_fetcher=None, **kwargs) -> MeanReversionStrategy:
    """
    🏭 FACTORY FUNCTION: Create mean reversion strategy instance

    Factory function for backward compatibility and easy instantiation.
    """
    return MeanReversionStrategy(data_fetcher=data_fetcher, **kwargs)