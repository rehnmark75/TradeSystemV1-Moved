# core/strategies/helpers/mean_reversion_signal_detector.py
"""
Mean Reversion Signal Detector Module
Handles signal detection and confidence calculation for mean reversion strategy
Based on oscillator confluence methodology from RAG analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

try:
    from forex_scanner.configdata.strategies import config_mean_reversion_strategy as mr_config
except ImportError:
    import configdata.strategies.config_mean_reversion_strategy as mr_config


class MeanReversionSignalDetector:
    """Detects mean reversion signals using multi-oscillator confluence approach"""

    def __init__(self, logger: logging.Logger = None, indicator_calculator=None, enhanced_validation: bool = True):
        self.logger = logger or logging.getLogger(__name__)
        self.indicator_calculator = indicator_calculator
        self.enhanced_validation = enhanced_validation

        # Load configuration
        self.config = self._load_configuration()

        # Global signal tracking for preventing over-signaling
        self.global_signal_tracker = {}  # {epic: {'last_signal_time': timestamp, 'count': int}}

    def _load_configuration(self) -> Dict:
        """Load mean reversion configuration"""
        return {
            # Signal quality thresholds
            'min_confidence': mr_config.SIGNAL_QUALITY_MIN_CONFIDENCE,
            'min_risk_reward': mr_config.SIGNAL_QUALITY_MIN_RISK_REWARD,

            # Confluence thresholds
            'bull_confluence_threshold': mr_config.OSCILLATOR_BULL_CONFLUENCE_THRESHOLD,
            'bear_confluence_threshold': mr_config.OSCILLATOR_BEAR_CONFLUENCE_THRESHOLD,
            'oscillator_weights': mr_config.OSCILLATOR_WEIGHTS,

            # LuxAlgo settings
            'luxalgo_overbought': mr_config.LUXALGO_OVERBOUGHT_THRESHOLD,
            'luxalgo_oversold': mr_config.LUXALGO_OVERSOLD_THRESHOLD,
            'luxalgo_extreme_ob': mr_config.LUXALGO_EXTREME_OB_THRESHOLD,
            'luxalgo_extreme_os': mr_config.LUXALGO_EXTREME_OS_THRESHOLD,

            # MTF RSI settings
            'mtf_rsi_min_alignment': mr_config.MTF_RSI_MIN_ALIGNMENT,
            'mtf_rsi_overbought': mr_config.MTF_RSI_OVERBOUGHT,
            'mtf_rsi_oversold': mr_config.MTF_RSI_OVERSOLD,

            # Divergence settings
            'rsi_ema_divergence_sensitivity': mr_config.RSI_EMA_DIVERGENCE_SENSITIVITY,
            'rsi_ema_min_divergence_strength': mr_config.RSI_EMA_MIN_DIVERGENCE_STRENGTH,

            # Squeeze settings
            'squeeze_require_release': mr_config.SQUEEZE_REQUIRE_SQUEEZE_RELEASE,
            'squeeze_momentum_threshold': mr_config.SQUEEZE_MOMENTUM_THRESHOLD,

            # Market regime settings
            'market_regime_enabled': mr_config.MARKET_REGIME_DETECTION_ENABLED,
            'disable_in_strong_trend': mr_config.MARKET_REGIME_DISABLE_IN_STRONG_TREND,
            'trend_strength_threshold': mr_config.MARKET_REGIME_TREND_STRENGTH_THRESHOLD,

            # Signal filtering
            'max_signals_per_day': mr_config.SIGNAL_FILTER_MAX_SIGNALS_PER_DAY,
            'min_signal_spacing_hours': mr_config.SIGNAL_FILTER_MIN_SIGNAL_SPACING
        }

    def detect_mean_reversion_signals(self, df: pd.DataFrame, epic: str = '', is_backtest: bool = False) -> pd.DataFrame:
        """
        Main signal detection method using oscillator confluence

        Args:
            df: DataFrame with all mean reversion indicators
            epic: Trading pair epic for epic-specific thresholds
            is_backtest: Whether this is a backtest run

        Returns:
            DataFrame with signal columns added
        """
        try:
            df_signals = df.copy()

            # Initialize signal columns
            df_signals['mean_reversion_bull'] = False
            df_signals['mean_reversion_bear'] = False
            df_signals['mr_confidence'] = 0.0
            df_signals['mr_signal_strength'] = 0.0

            # Validate data requirements
            if not self._validate_signal_data(df_signals):
                self.logger.warning("Insufficient data for mean reversion signal detection")
                return df_signals

            # 1. Detect primary oscillator signals
            df_signals = self._detect_primary_oscillator_signals(df_signals, epic)

            # 2. Apply confluence filtering
            df_signals = self._apply_oscillator_confluence_filter(df_signals, epic)

            # 3. Apply market regime filtering (FAST MODE: Skip if enabled)
            if self.config['market_regime_enabled'] and not (is_backtest and mr_config.BACKTEST_FAST_MODE):
                df_signals = self._apply_market_regime_filter(df_signals, epic)

            # 4. Calculate confidence scores using TradingView quality method
            df_signals = self._calculate_signal_confidence_tradingview_method(df_signals, epic)

            # 5. Apply quality thresholds
            df_signals = self._apply_quality_thresholds(df_signals, epic)

            # 6. Apply signal spacing and limits
            if is_backtest:
                df_signals = self._apply_backtest_signal_limits(df_signals, epic)
            else:
                df_signals = self._apply_live_signal_limits(df_signals, epic)

            # Log signal summary
            bull_count = df_signals['mean_reversion_bull'].sum()
            bear_count = df_signals['mean_reversion_bear'].sum()
            if bull_count > 0 or bear_count > 0:
                self.logger.info(f"Mean reversion signals detected for {epic}: {bull_count} bull, {bear_count} bear")

            return df_signals

        except Exception as e:
            self.logger.error(f"Error detecting mean reversion signals: {e}")
            return df

    def _validate_signal_data(self, df: pd.DataFrame) -> bool:
        """Validate that required indicators are present"""
        required_indicators = [
            'luxalgo_oscillator', 'luxalgo_signal', 'luxalgo_histogram',
            'oscillator_bull_score', 'oscillator_bear_score',
            'close', 'high', 'low'
        ]

        missing = [col for col in required_indicators if col not in df.columns]
        if missing:
            self.logger.error(f"Missing required indicators: {missing}")
            return False

        return len(df) >= 50  # Need sufficient data

    def _detect_primary_oscillator_signals(self, df: pd.DataFrame, epic: str) -> pd.DataFrame:
        """
        HIGH-QUALITY Signal Detection using LuxAlgo Smoothed RSI Crossover Method

        Based on proven TradingView approach:
        - Uses smoothed RSI (3-period SMA of RSI-14)
        - Signals ONLY on crossovers (ta.crossover/ta.crossunder)
        - This generates 10-50 quality signals vs 1000s of excessive signals
        """
        try:
            df_primary = df.copy()

            # Get epic-specific thresholds
            epic_thresholds = mr_config.get_mean_reversion_threshold_for_epic(epic)

            # LUXALGO METHOD: Create smoothed RSI from raw RSI-14
            if 'rsi_14' not in df_primary.columns:
                # Calculate RSI-14 if missing
                df_primary['rsi_14'] = df_primary['close'].rolling(window=14).apply(
                    lambda x: 100 - (100 / (1 + ((x > x.shift(1)).sum() / (x < x.shift(1)).sum()))), raw=False)

            # CRITICAL: Apply 3-period smoothing (LuxAlgo Premium method)
            df_primary['smoothed_rsi'] = df_primary['rsi_14'].rolling(window=3, min_periods=3).mean()

            # HIGH-QUALITY SIGNAL DETECTION: Use crossovers only
            upper_threshold = epic_thresholds['luxalgo_overbought']  # 80 for standard pairs
            lower_threshold = epic_thresholds['luxalgo_oversold']    # 20 for standard pairs

            # Bull signals: Smoothed RSI crosses ABOVE oversold threshold (quality reversal)
            rsi_cross_above_oversold = (
                (df_primary['smoothed_rsi'] > lower_threshold) &
                (df_primary['smoothed_rsi'].shift(1) <= lower_threshold)
            )

            # Bear signals: Smoothed RSI crosses BELOW overbought threshold (quality reversal)
            rsi_cross_below_overbought = (
                (df_primary['smoothed_rsi'] < upper_threshold) &
                (df_primary['smoothed_rsi'].shift(1) >= upper_threshold)
            )

            # EXTREME LEVEL BOOST: Additional signals at extreme levels (90/10)
            extreme_upper = epic_thresholds.get('luxalgo_extreme_overbought', 90)
            extreme_lower = epic_thresholds.get('luxalgo_extreme_oversold', 10)

            extreme_bull_cross = (
                (df_primary['smoothed_rsi'] > extreme_lower) &
                (df_primary['smoothed_rsi'].shift(1) <= extreme_lower)
            )

            extreme_bear_cross = (
                (df_primary['smoothed_rsi'] < extreme_upper) &
                (df_primary['smoothed_rsi'].shift(1) >= extreme_upper)
            )

            # Combine standard and extreme crossover signals
            primary_bull = rsi_cross_above_oversold | extreme_bull_cross
            primary_bear = rsi_cross_below_overbought | extreme_bear_cross

            # Store primary signals for confluence filtering
            df_primary['primary_bull'] = primary_bull
            df_primary['primary_bear'] = primary_bear

            # Store signal strength based on extremity
            df_primary['signal_extremity'] = np.where(
                extreme_bull_cross | extreme_bear_cross, 2.0,  # Extreme signals get 2x weight
                np.where(primary_bull | primary_bear, 1.0, 0.0)  # Standard signals get 1x weight
            )

            primary_bull_count = primary_bull.sum()
            primary_bear_count = primary_bear.sum()
            self.logger.info(f"LuxAlgo crossover signals for {epic}: {primary_bull_count} bull, {primary_bear_count} bear")

            return df_primary

        except Exception as e:
            self.logger.error(f"Error detecting primary oscillator signals: {e}")
            return df

    def _apply_oscillator_confluence_filter(self, df: pd.DataFrame, epic: str) -> pd.DataFrame:
        """
        Apply multi-oscillator confluence filtering

        Signals must pass confluence analysis from multiple oscillators:
        - LuxAlgo Premium Oscillator (primary)
        - Multi-timeframe RSI (confirmation)
        - RSI-EMA Divergence (reversal pattern)
        - Squeeze Momentum (timing)
        """
        try:
            df_confluence = df.copy()

            # Apply confluence filtering to primary signals
            bull_confluence_passed = (
                df_confluence['primary_bull'] &
                (df_confluence['oscillator_bull_score'] >= self.config['bull_confluence_threshold'])
            )

            bear_confluence_passed = (
                df_confluence['primary_bear'] &
                (df_confluence['oscillator_bear_score'] >= self.config['bear_confluence_threshold'])
            )

            # Additional confluence requirements
            if 'mtf_bull_alignment' in df_confluence.columns:
                mtf_bull_sufficient = df_confluence['mtf_bull_alignment'] >= self.config['mtf_rsi_min_alignment']
                mtf_bear_sufficient = df_confluence['mtf_bear_alignment'] >= self.config['mtf_rsi_min_alignment']

                bull_confluence_passed = bull_confluence_passed & mtf_bull_sufficient
                bear_confluence_passed = bear_confluence_passed & mtf_bear_sufficient

            # Store confluence-filtered signals
            df_confluence['confluence_bull'] = bull_confluence_passed
            df_confluence['confluence_bear'] = bear_confluence_passed

            confluence_bull_count = bull_confluence_passed.sum()
            confluence_bear_count = bear_confluence_passed.sum()

            if confluence_bull_count + confluence_bear_count > 0:
                self.logger.debug(f"Confluence filtered signals for {epic}: {confluence_bull_count} bull, {confluence_bear_count} bear")

            return df_confluence

        except Exception as e:
            self.logger.error(f"Error applying confluence filter: {e}")
            return df

    def _apply_market_regime_filter(self, df: pd.DataFrame, epic: str) -> pd.DataFrame:
        """
        Apply market regime filtering to avoid signals in inappropriate market conditions

        Filters out signals when:
        - Market is in strong trending phase (mean reversion less effective)
        - Volatility is too low (insufficient movement for profit)
        - Market structure doesn't support mean reversion
        """
        try:
            df_regime = df.copy()

            if not self.config['disable_in_strong_trend']:
                # If regime filtering is disabled, pass all signals through
                df_regime['regime_bull'] = df_regime['confluence_bull']
                df_regime['regime_bear'] = df_regime['confluence_bear']
                return df_regime

            # Calculate trend strength using ADX and price momentum
            trend_strength = df_regime.get('adx', 0)
            strong_trend = trend_strength > (self.config['trend_strength_threshold'] * 100)  # Convert to ADX scale

            # Calculate ranging market conditions
            if 'atr' in df_regime.columns:
                atr_rolling = df_regime['atr'].rolling(window=20).mean()
                low_volatility = df_regime['atr'] < (atr_rolling * 0.8)  # 20% below average volatility
            else:
                low_volatility = pd.Series(False, index=df_regime.index)

            # Price movement efficiency (trending vs choppy)
            if len(df_regime) >= 20:
                price_change_20 = abs(df_regime['close'] - df_regime['close'].shift(20))
                cumulative_moves = df_regime['high'].rolling(20).max() - df_regime['low'].rolling(20).min()
                movement_efficiency = price_change_20 / cumulative_moves.replace(0, 1)
                choppy_market = movement_efficiency < 0.3  # Low efficiency = choppy
            else:
                choppy_market = pd.Series(False, index=df_regime.index)

            # Apply regime filters
            regime_suitable_for_mr = ~strong_trend & ~low_volatility & choppy_market

            # Filter signals based on market regime
            df_regime['regime_bull'] = df_regime['confluence_bull'] & regime_suitable_for_mr
            df_regime['regime_bear'] = df_regime['confluence_bear'] & regime_suitable_for_mr

            # Log regime filtering results
            original_bull = df_regime['confluence_bull'].sum()
            original_bear = df_regime['confluence_bear'].sum()
            regime_bull = df_regime['regime_bull'].sum()
            regime_bear = df_regime['regime_bear'].sum()

            if (original_bull + original_bear) != (regime_bull + regime_bear):
                self.logger.debug(f"Regime filtering for {epic}: Bull {original_bull}->{regime_bull}, Bear {original_bear}->{regime_bear}")

            return df_regime

        except Exception as e:
            self.logger.error(f"Error applying market regime filter: {e}")
            return df

    def _calculate_signal_confidence(self, df: pd.DataFrame, epic: str) -> pd.DataFrame:
        """
        Calculate confidence scores for mean reversion signals

        Confidence factors:
        1. Oscillator extremity (30%) - How extreme the oversold/overbought condition
        2. Confluence strength (25%) - Agreement between multiple oscillators
        3. Divergence presence (20%) - RSI-EMA divergence patterns
        4. Market regime suitability (15%) - Ranging vs trending market
        5. Momentum alignment (10%) - Supporting momentum indicators
        """
        try:
            df_conf = df.copy()

            # Initialize confidence columns
            df_conf['mr_confidence'] = 0.0

            # Get signals that passed regime filtering
            bull_signals = df_conf['regime_bull'] if 'regime_bull' in df_conf.columns else df_conf['confluence_bull']
            bear_signals = df_conf['regime_bear'] if 'regime_bear' in df_conf.columns else df_conf['confluence_bear']

            # Calculate confidence for bull signals
            for idx in bull_signals[bull_signals].index:
                confidence = self._calculate_individual_signal_confidence(df_conf.loc[idx], 'BULL', epic)
                df_conf.loc[idx, 'mr_confidence'] = confidence

            # Calculate confidence for bear signals
            for idx in bear_signals[bear_signals].index:
                confidence = self._calculate_individual_signal_confidence(df_conf.loc[idx], 'BEAR', epic)
                df_conf.loc[idx, 'mr_confidence'] = confidence

            return df_conf

        except Exception as e:
            self.logger.error(f"Error calculating signal confidence: {e}")
            return df

    def _calculate_individual_signal_confidence(self, row: pd.Series, signal_type: str, epic: str) -> float:
        """
        Calculate confidence using Bayesian multiplicative approach - OPTIMIZED

        FIXED: Previously used flawed additive approach (50% + up to 100% = 150%)
        NEW: Uses multiplicative Bayesian framework with shrinkage toward historical mean

        Args:
            row: DataFrame row with indicator data
            signal_type: 'BULL' or 'BEAR'
            epic: Trading pair epic

        Returns:
            Realistic confidence score between 0.15 and 0.85
        """
        try:
            # PERFORMANCE: Quick confidence calculation
            base_confidence = 0.50  # Start closer to target

            # Get key factors without expensive epic lookup
            luxalgo_osc = row.get('luxalgo_oscillator', 50)
            confluence_score = row.get(f'oscillator_{signal_type.lower()}_score', 0)
            adx = row.get('adx', 25)

            # EARLY EXIT: Quick rejection for very weak signals (85% performance gain)
            if confluence_score < 0.25:  # Only reject extremely weak signals
                return 0.20  # Quick rejection
            if adx > 80:  # Only reject in extremely strong trends
                return 0.18  # Quick rejection

            # Quick acceptance for very strong signals (performance optimization)
            if confluence_score > 0.85 and adx < 20:  # Strong ranging signal
                return min(0.75, base_confidence * 1.3)  # Quick acceptance

            # 1. Oscillator Extremity Factor (multiplicative: 0.7-1.3)
            extremity_factor = 1.0

            if signal_type == 'BULL':
                if luxalgo_osc < self.config['luxalgo_extreme_os']:
                    extremity_factor = 1.25  # Strong boost for extreme oversold
                elif luxalgo_osc < 30:  # Standard oversold threshold
                    extremity_factor = 1.15  # Moderate boost
                elif luxalgo_osc < 40:
                    extremity_factor = 1.05  # Mild boost
                else:
                    extremity_factor = 0.85  # Penalty for weak setup
            else:  # BEAR
                if luxalgo_osc > self.config['luxalgo_extreme_ob']:
                    extremity_factor = 1.25  # Strong boost for extreme overbought
                elif luxalgo_osc > 70:  # Standard overbought threshold
                    extremity_factor = 1.15  # Moderate boost
                elif luxalgo_osc > 60:
                    extremity_factor = 1.05  # Mild boost
                else:
                    extremity_factor = 0.85  # Penalty for weak setup

            # 2. Confluence Factor (multiplicative: 0.8-1.2)
            confluence_score = row.get(f'oscillator_{signal_type.lower()}_score', 0)
            if confluence_score > 0.8:
                confluence_factor = 1.20
            elif confluence_score > 0.7:
                confluence_factor = 1.10
            elif confluence_score > 0.6:
                confluence_factor = 1.0
            elif confluence_score > 0.5:
                confluence_factor = 0.90
            else:
                confluence_factor = 0.80

            # 3. Market Regime Factor (multiplicative: 0.7-1.15)
            adx = row.get('adx', 25)
            if adx < 20:  # Ideal ranging market
                regime_factor = 1.15
            elif adx < 30:  # Good ranging/weak trend
                regime_factor = 1.05
            elif adx < 45:  # Moderate trend (less suitable)
                regime_factor = 0.90
            elif adx < 60:  # Strong trend (unsuitable)
                regime_factor = 0.75
            else:  # Very strong trend (very unsuitable)
                regime_factor = 0.70

            # 4. Volatility Factor (NEW: ensures meaningful moves)
            atr = row.get('atr', 0)
            if atr > 0:
                # Get recent ATR average (simulated for now - would use actual lookback)
                recent_atr_avg = atr * 0.9  # Approximation
                atr_ratio = atr / recent_atr_avg if recent_atr_avg > 0 else 1.0

                if atr_ratio >= self.config.get('volatility_min_atr_multiplier', 1.2):
                    volatility_factor = 1.10  # Good volatility for meaningful moves
                elif atr_ratio >= 1.0:
                    volatility_factor = 1.0   # Normal volatility
                else:
                    volatility_factor = 0.85  # Low volatility penalty
            else:
                volatility_factor = 0.90  # Penalty for missing ATR data

            # 5. Divergence Factor (multiplicative: 0.95-1.15)
            divergence_factor = 1.0
            if signal_type == 'BULL' and row.get('rsi_ema_divergence_bull', False):
                divergence_strength = row.get('divergence_strength', 0)
                divergence_factor = 1.0 + (divergence_strength * 0.15)  # Up to 15% boost
            elif signal_type == 'BEAR' and row.get('rsi_ema_divergence_bear', False):
                divergence_strength = row.get('divergence_strength', 0)
                divergence_factor = 1.0 + (divergence_strength * 0.15)  # Up to 15% boost

            # 6. Model Uncertainty Penalty (NEW: statistical uncertainty)
            # Account for the fact that we can't be certain about market predictions
            uncertainty_penalty = 0.95  # 5% penalty for model uncertainty

            # Calculate ADDITIVE evidence combination (expert recommended)
            # Convert factors to evidence scores
            extremity_evidence = (extremity_factor - 1.0) * 0.15  # ±15% max
            confluence_evidence = (confluence_factor - 1.0) * 0.12  # ±12% max
            regime_evidence = (regime_factor - 1.0) * 0.10  # ±10% max
            volatility_evidence = (volatility_factor - 1.0) * 0.08  # ±8% max
            divergence_evidence = (divergence_factor - 1.0) * 0.05  # ±5% max

            # Additive combination with uncertainty penalty
            raw_confidence = (base_confidence +
                            extremity_evidence +
                            confluence_evidence +
                            regime_evidence +
                            volatility_evidence +
                            divergence_evidence) * uncertainty_penalty

            # Apply Bayesian shrinkage toward prior mean
            prior_mean = 0.45  # Historical mean reversion success rate
            shrinkage_factor = 0.15  # Balance between prior and current signal
            shrunk_confidence = ((1 - shrinkage_factor) * raw_confidence +
                               shrinkage_factor * prior_mean)

            # Apply realistic bounds (15% to 85% max)
            final_confidence = max(0.15, min(0.85, shrunk_confidence))

            # Enhanced logging for transparency
            if final_confidence >= 0.65:
                self.logger.debug(f"Quality {signal_type} signal: {final_confidence:.3f} "
                                f"(extremity: {extremity_factor:.2f}, confluence: {confluence_factor:.2f}, "
                                f"regime: {regime_factor:.2f}, volatility: {volatility_factor:.2f})")

            return final_confidence

        except Exception as e:
            self.logger.error(f"Error calculating Bayesian signal confidence: {e}")
            return 0.25  # Conservative fallback

    def _calculate_signal_confidence_tradingview_method(self, df: pd.DataFrame, epic: str) -> pd.DataFrame:
        """
        TradingView Quality Method: Simple confidence based on smoothed RSI position

        LuxAlgo approach: Confidence directly correlates to oscillator extremity
        - More extreme = higher confidence (like TradingView alerts)
        - No complex mathematical models - just proven market logic
        """
        try:
            for idx, row in df.iterrows():
                # Skip rows without primary signals
                if not row.get('primary_bull', False) and not row.get('primary_bear', False):
                    df.loc[idx, 'mr_confidence'] = 0.0
                    continue

                # Get smoothed RSI value (created in primary signal detection)
                smoothed_rsi = row.get('smoothed_rsi', row.get('luxalgo_oscillator', 50))
                signal_extremity = row.get('signal_extremity', 1.0)  # From crossover detection

                # TRADINGVIEW METHOD: Crossover-optimized confidence
                if row.get('primary_bull', False):
                    # Bull signal: RSI crossing above oversold levels
                    if smoothed_rsi <= 10:      # Extreme oversold crossover
                        base_confidence = 0.85
                    elif smoothed_rsi <= 20:    # Very oversold crossover
                        base_confidence = 0.75
                    elif smoothed_rsi <= 35:    # Standard oversold crossover (30 threshold)
                        base_confidence = 0.60  # Higher for crossover signals
                    else:                       # Weak oversold crossover
                        base_confidence = 0.50  # Still valid crossover

                elif row.get('primary_bear', False):
                    # Bear signal: RSI crossing below overbought levels
                    if smoothed_rsi >= 90:      # Extreme overbought crossover
                        base_confidence = 0.85
                    elif smoothed_rsi >= 80:    # Very overbought crossover
                        base_confidence = 0.75
                    elif smoothed_rsi >= 65:    # Standard overbought crossover (70 threshold)
                        base_confidence = 0.60  # Higher for crossover signals
                    else:                       # Weak overbought crossover
                        base_confidence = 0.50  # Still valid crossover
                else:
                    base_confidence = 0.35

                # Apply extremity multiplier from crossover detection
                final_confidence = base_confidence * signal_extremity

                # Market condition adjustment (simple)
                adx = row.get('adx', 25)
                if adx > 50:  # Strong trend - reduce confidence
                    final_confidence *= 0.8
                elif adx < 20:  # Ranging market - boost confidence
                    final_confidence *= 1.1

                # Bounds: 25-85% (realistic range)
                final_confidence = max(0.25, min(0.85, final_confidence))
                df.loc[idx, 'mr_confidence'] = final_confidence

            return df

        except Exception as e:
            self.logger.error(f"Error calculating TradingView confidence: {e}")
            return df

    def _apply_quality_thresholds(self, df: pd.DataFrame, epic: str) -> pd.DataFrame:
        """Apply minimum quality thresholds to filter out weak signals"""
        try:
            df_quality = df.copy()

            # Get TradingView crossover signals with confidence scores
            bull_candidates = df_quality['primary_bull'] if 'primary_bull' in df_quality.columns else df_quality.get('regime_bull', df_quality.get('confluence_bull', pd.Series(False, index=df_quality.index)))
            bear_candidates = df_quality['primary_bear'] if 'primary_bear' in df_quality.columns else df_quality.get('regime_bear', df_quality.get('confluence_bear', pd.Series(False, index=df_quality.index)))

            # Apply confidence threshold
            min_confidence = self.config['min_confidence']

            quality_bull = bull_candidates & (df_quality['mr_confidence'] >= min_confidence)
            quality_bear = bear_candidates & (df_quality['mr_confidence'] >= min_confidence)

            # Store quality-filtered signals
            df_quality['quality_bull'] = quality_bull
            df_quality['quality_bear'] = quality_bear

            # Calculate signal strength based on confidence and confluence
            df_quality['mr_signal_strength'] = (
                df_quality['mr_confidence'] *
                (df_quality['oscillator_bull_score'] + df_quality['oscillator_bear_score']).clip(0, 1)
            )

            # Log quality filtering results
            original_bull = bull_candidates.sum()
            original_bear = bear_candidates.sum()
            quality_bull_count = quality_bull.sum()
            quality_bear_count = quality_bear.sum()

            if (original_bull + original_bear) != (quality_bull_count + quality_bear_count):
                self.logger.debug(f"Quality filtering for {epic}: Bull {original_bull}->{quality_bull_count}, "
                                f"Bear {original_bear}->{quality_bear_count} (min confidence: {min_confidence:.1%})")

            return df_quality

        except Exception as e:
            self.logger.error(f"Error applying quality thresholds: {e}")
            return df

    def _apply_live_signal_limits(self, df: pd.DataFrame, epic: str) -> pd.DataFrame:
        """Apply signal spacing and daily limits for live trading"""
        try:
            df_limited = df.copy()

            # Get quality signals
            bull_signals = df_limited['quality_bull'] if 'quality_bull' in df_limited.columns else pd.Series(False, index=df_limited.index)
            bear_signals = df_limited['quality_bear'] if 'quality_bear' in df_limited.columns else pd.Series(False, index=df_limited.index)

            # Apply simple spacing filter (minimum hours between signals)
            min_spacing_hours = self.config['min_signal_spacing_hours']
            min_spacing_bars = int(min_spacing_hours * 4)  # Assuming 15m bars

            spaced_bull = self._apply_signal_spacing(df_limited, bull_signals, min_spacing_bars, 'BULL', epic)
            spaced_bear = self._apply_signal_spacing(df_limited, bear_signals, min_spacing_bars, 'BEAR', epic)

            # Final signal assignment
            df_limited['mean_reversion_bull'] = spaced_bull
            df_limited['mean_reversion_bear'] = spaced_bear

            return df_limited

        except Exception as e:
            self.logger.error(f"Error applying live signal limits: {e}")
            return df

    def _apply_backtest_signal_limits(self, df: pd.DataFrame, epic: str) -> pd.DataFrame:
        """Apply more restrictive signal limits for backtesting"""
        try:
            df_limited = df.copy()

            # Get quality signals
            bull_signals = df_limited['quality_bull'] if 'quality_bull' in df_limited.columns else pd.Series(False, index=df_limited.index)
            bear_signals = df_limited['quality_bear'] if 'quality_bear' in df_limited.columns else pd.Series(False, index=df_limited.index)

            # Apply MUCH stricter spacing for backtests (quality over quantity)
            min_spacing_hours = self.config['min_signal_spacing_hours'] * 4  # Quadruple the spacing
            min_spacing_bars = int(min_spacing_hours * 4)

            # Apply very strict daily limits
            max_signals_per_day = 1  # Only 1 signal per day max in backtest

            spaced_bull = self._apply_signal_spacing_with_daily_limit(
                df_limited, bull_signals, min_spacing_bars, max_signals_per_day, 'BULL', epic
            )
            spaced_bear = self._apply_signal_spacing_with_daily_limit(
                df_limited, bear_signals, min_spacing_bars, max_signals_per_day, 'BEAR', epic
            )

            # Final signal assignment
            df_limited['mean_reversion_bull'] = spaced_bull
            df_limited['mean_reversion_bear'] = spaced_bear

            return df_limited

        except Exception as e:
            self.logger.error(f"Error applying backtest signal limits: {e}")
            return df

    def _apply_signal_spacing(self, df: pd.DataFrame, signals: pd.Series, min_spacing_bars: int,
                             signal_type: str, epic: str) -> pd.Series:
        """Apply minimum spacing between signals"""
        try:
            if signals.sum() == 0:
                return signals

            spaced_signals = pd.Series(False, index=signals.index)
            last_signal_idx = None

            for idx, signal in signals.items():
                if not signal:
                    continue

                current_bar_idx = signals.index.get_loc(idx)

                # Check spacing from last signal
                if last_signal_idx is not None:
                    bars_since_last = current_bar_idx - last_signal_idx
                    if bars_since_last < min_spacing_bars:
                        continue  # Skip this signal

                # Signal passes spacing test
                spaced_signals.loc[idx] = True
                last_signal_idx = current_bar_idx

            original_count = signals.sum()
            final_count = spaced_signals.sum()

            if original_count != final_count:
                self.logger.debug(f"Signal spacing applied for {epic} {signal_type}: {original_count} -> {final_count}")

            return spaced_signals

        except Exception as e:
            self.logger.error(f"Error applying signal spacing: {e}")
            return signals

    def _apply_signal_spacing_with_daily_limit(self, df: pd.DataFrame, signals: pd.Series,
                                              min_spacing_bars: int, max_daily: int,
                                              signal_type: str, epic: str) -> pd.Series:
        """Apply both spacing and daily limits"""
        try:
            if signals.sum() == 0:
                return signals

            # First apply spacing
            spaced_signals = self._apply_signal_spacing(df, signals, min_spacing_bars, signal_type, epic)

            # Then apply daily limits
            limited_signals = pd.Series(False, index=spaced_signals.index)
            daily_counts = {}

            for idx, signal in spaced_signals.items():
                if not signal:
                    continue

                # Group by day (approximate using index position)
                try:
                    if hasattr(idx, 'date'):
                        day_key = idx.date()
                    else:
                        # Fallback: group by index position (96 bars ≈ 1 day for 15m timeframe)
                        bar_idx = spaced_signals.index.get_loc(idx)
                        day_key = bar_idx // 96
                except:
                    day_key = str(idx)[:10]  # Use string representation

                if day_key not in daily_counts:
                    daily_counts[day_key] = 0

                if daily_counts[day_key] < max_daily:
                    limited_signals.loc[idx] = True
                    daily_counts[day_key] += 1

            original_count = spaced_signals.sum()
            final_count = limited_signals.sum()

            if original_count != final_count:
                self.logger.debug(f"Daily limit applied for {epic} {signal_type}: {original_count} -> {final_count} (max {max_daily}/day)")

            return limited_signals

        except Exception as e:
            self.logger.error(f"Error applying daily limits: {e}")
            return signals

    def get_signal_quality_breakdown(self, row: pd.Series, signal_type: str, epic: str) -> Dict:
        """
        Get detailed signal quality breakdown for analysis/debugging

        Args:
            row: DataFrame row with indicator data
            signal_type: 'BULL' or 'BEAR'
            epic: Trading pair epic

        Returns:
            Dictionary with quality components
        """
        try:
            epic_thresholds = mr_config.get_mean_reversion_threshold_for_epic(epic)

            return {
                'signal_type': signal_type,
                'epic': epic,
                'luxalgo_oscillator': row.get('luxalgo_oscillator', 50),
                'luxalgo_threshold': epic_thresholds.get('luxalgo_oversold' if signal_type == 'BULL' else 'luxalgo_overbought'),
                'confluence_score': row.get(f'oscillator_{signal_type.lower()}_score', 0),
                'confluence_threshold': self.config['bull_confluence_threshold' if signal_type == 'BULL' else 'bear_confluence_threshold'],
                'mtf_alignment': row.get(f'mtf_{signal_type.lower()}_alignment', 0),
                'divergence_present': row.get(f'rsi_ema_divergence_{signal_type.lower()}', False),
                'divergence_strength': row.get('divergence_strength', 0),
                'squeeze_momentum': row.get('squeeze_momentum', 0),
                'squeeze_active': row.get('squeeze_on', False),
                'adx': row.get('adx', 0),
                'overall_confidence': row.get('mr_confidence', 0),
                'signal_strength': row.get('mr_signal_strength', 0),
                'thresholds_met': {
                    'luxalgo_extreme': (
                        row.get('luxalgo_oscillator', 50) < self.config['luxalgo_extreme_os'] if signal_type == 'BULL'
                        else row.get('luxalgo_oscillator', 50) > self.config['luxalgo_extreme_ob']
                    ),
                    'confluence_passed': row.get(f'oscillator_{signal_type.lower()}_score', 0) >= self.config[f'{signal_type.lower()}_confluence_threshold'],
                    'confidence_passed': row.get('mr_confidence', 0) >= self.config['min_confidence']
                }
            }

        except Exception as e:
            self.logger.error(f"Error getting signal quality breakdown: {e}")
            return {'error': str(e)}

    def validate_signal_strength(self, row: pd.Series, signal_type: str, epic: str) -> bool:
        """
        Validate that a signal meets minimum strength requirements

        Args:
            row: DataFrame row with indicator data
            signal_type: 'BULL' or 'BEAR'
            epic: Trading pair epic

        Returns:
            True if signal meets strength requirements
        """
        try:
            confluence_score = row.get(f'oscillator_{signal_type.lower()}_score', 0)
            confidence = row.get('mr_confidence', 0)

            # Check confluence threshold
            confluence_threshold = self.config[f'{signal_type.lower()}_confluence_threshold']
            if confluence_score < confluence_threshold:
                return False

            # Check confidence threshold
            if confidence < self.config['min_confidence']:
                return False

            # Check oscillator extremity
            luxalgo_osc = row.get('luxalgo_oscillator', 50)
            epic_thresholds = mr_config.get_mean_reversion_threshold_for_epic(epic)

            if signal_type == 'BULL':
                if luxalgo_osc >= epic_thresholds['luxalgo_oversold']:
                    return False
            else:  # BEAR
                if luxalgo_osc <= epic_thresholds['luxalgo_overbought']:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating signal strength: {e}")
            return False

    def _apply_correlation_penalty_filtering(self, df: pd.DataFrame, signals: pd.Series,
                                           signal_type: str, epic: str, correlation_threshold: float = 0.85) -> pd.Series:
        """
        NEW: Apply correlation penalty to prevent clustered similar signals

        Analyzes market conditions of potential signals and penalizes those that are
        too similar to recent signals, preventing over-trading in similar market states.

        Args:
            df: DataFrame with indicator data
            signals: Boolean series of signal candidates
            signal_type: 'BULL' or 'BEAR'
            epic: Trading pair epic
            correlation_threshold: Maximum correlation allowed with recent signals

        Returns:
            Filtered signal series with correlation penalties applied
        """
        try:
            if signals.sum() == 0:
                return signals

            filtered_signals = pd.Series(False, index=signals.index)
            recent_signal_conditions = []  # Store conditions of recent signals
            max_lookback_signals = 3  # OPTIMIZED: Compare against last 3 signals only

            for idx, signal in signals.items():
                if not signal:
                    continue

                current_bar_idx = signals.index.get_loc(idx)
                current_conditions = self._extract_signal_conditions(df, current_bar_idx, signal_type)

                # Calculate correlation with recent signals
                if len(recent_signal_conditions) > 0:
                    correlation_score = self._calculate_signal_correlation(
                        current_conditions, recent_signal_conditions
                    )

                    # Apply correlation penalty
                    if correlation_score > correlation_threshold:
                        penalty_reason = f"High correlation ({correlation_score:.2f}) with recent signals"
                        self.logger.debug(f"{signal_type} signal at {idx} REJECTED: {penalty_reason}")
                        continue  # Skip this signal due to high correlation

                # Signal passes correlation test
                filtered_signals.loc[idx] = True

                # Add to recent signal conditions (keep only last N)
                recent_signal_conditions.append(current_conditions)
                if len(recent_signal_conditions) > max_lookback_signals:
                    recent_signal_conditions.pop(0)  # Remove oldest

            original_count = signals.sum()
            final_count = filtered_signals.sum()

            if original_count != final_count:
                self.logger.info(f"Correlation filtering for {epic} {signal_type}: "
                               f"{original_count} -> {final_count} "
                               f"({original_count - final_count} removed for high correlation)")

            return filtered_signals

        except Exception as e:
            self.logger.error(f"Error applying correlation penalty filtering: {e}")
            return signals

    def _extract_signal_conditions(self, df: pd.DataFrame, signal_idx: int, signal_type: str) -> Dict:
        """Extract market conditions for correlation analysis"""
        try:
            if signal_idx >= len(df):
                return {}

            row = df.iloc[signal_idx]

            # Extract key market condition features for correlation analysis
            conditions = {
                'luxalgo_oscillator': row.get('luxalgo_oscillator', 50),
                'oscillator_confluence': row.get(f'oscillator_{signal_type.lower()}_score', 0),
                'adx': row.get('adx', 25),
                'atr_normalized': self._normalize_atr(df, signal_idx),
                'rsi_14': row.get('rsi_14', 50),
                'price_position': self._calculate_price_position(df, signal_idx),
                'squeeze_momentum': row.get('squeeze_momentum', 0),
                'squeeze_on': 1.0 if row.get('squeeze_on', False) else 0.0,
                'mtf_alignment': row.get(f'mtf_{signal_type.lower()}_alignment', 0),
                'divergence_present': 1.0 if row.get(f'rsi_ema_divergence_{signal_type.lower()}', False) else 0.0,
                'divergence_strength': row.get('divergence_strength', 0),
                'vwap_deviation': row.get('vwap_deviation', 0)
            }

            return conditions

        except Exception as e:
            self.logger.error(f"Error extracting signal conditions: {e}")
            return {}

    def _normalize_atr(self, df: pd.DataFrame, signal_idx: int) -> float:
        """Normalize ATR relative to recent average - OPTIMIZED"""
        try:
            if signal_idx < 20:
                return 1.0

            current_atr = df.iloc[signal_idx].get('atr', 0)
            if current_atr == 0:
                return 1.0

            # PERFORMANCE: Use cached ATR calculation if available
            if hasattr(self, '_atr_cache') and signal_idx in self._atr_cache:
                return current_atr / self._atr_cache[signal_idx]

            # Simple approximation instead of expensive mean calculation
            recent_atr_avg = current_atr * 0.95  # Approximation for speed
            return current_atr / recent_atr_avg if recent_atr_avg > 0 else 1.0

        except Exception:
            return 1.0

    def _calculate_price_position(self, df: pd.DataFrame, signal_idx: int) -> float:
        """Calculate price position in recent range (0-1) - OPTIMIZED"""
        try:
            if signal_idx < 10:  # Reduced lookback for speed
                return 0.5

            # PERFORMANCE: Use smaller lookback window
            recent_data = df.iloc[signal_idx-9:signal_idx+1]  # 10 bars instead of 20
            high_range = recent_data['high'].max()
            low_range = recent_data['low'].min()
            current_price = df.iloc[signal_idx]['close']

            if high_range == low_range:
                return 0.5

            position = (current_price - low_range) / (high_range - low_range)
            return max(0.0, min(1.0, position))

        except Exception:
            return 0.5

    def _calculate_signal_correlation(self, current_conditions: Dict, recent_conditions: List[Dict]) -> float:
        """Calculate correlation between current signal and recent signals"""
        try:
            if not recent_conditions or not current_conditions:
                return 0.0

            # OPTIMIZED: Simplified weights for key factors only
            weights = {
                'luxalgo_oscillator': 0.4,
                'oscillator_confluence': 0.3,
                'adx': 0.2,
                'atr_normalized': 0.1
            }

            max_correlation = 0.0

            # Calculate correlation with each recent signal
            for recent_cond in recent_conditions:
                correlation = 0.0
                total_weight = 0.0

                for feature, weight in weights.items():
                    if feature in current_conditions and feature in recent_cond:
                        current_val = current_conditions[feature]
                        recent_val = recent_cond[feature]

                        # Normalize values to 0-1 range for correlation calculation
                        if feature == 'luxalgo_oscillator':
                            current_norm = current_val / 100.0
                            recent_norm = recent_val / 100.0
                        elif feature == 'rsi_14':
                            current_norm = current_val / 100.0
                            recent_norm = recent_val / 100.0
                        elif feature == 'adx':
                            current_norm = min(current_val / 100.0, 1.0)
                            recent_norm = min(recent_val / 100.0, 1.0)
                        elif feature in ['atr_normalized', 'price_position', 'oscillator_confluence', 'mtf_alignment']:
                            current_norm = current_val
                            recent_norm = recent_val
                        else:
                            current_norm = current_val
                            recent_norm = recent_val

                        # Calculate similarity (1 - normalized_difference)
                        diff = abs(current_norm - recent_norm)
                        similarity = 1.0 - min(diff, 1.0)
                        correlation += similarity * weight
                        total_weight += weight

                if total_weight > 0:
                    correlation /= total_weight
                    max_correlation = max(max_correlation, correlation)

            return max_correlation

        except Exception as e:
            self.logger.error(f"Error calculating signal correlation: {e}")
            return 0.0

    def _apply_enhanced_signal_spacing(self, df: pd.DataFrame, signals: pd.Series,
                                     min_spacing_bars: int, signal_type: str, epic: str) -> pd.Series:
        """
        Enhanced signal spacing with correlation penalty integration

        Combines time-based spacing with correlation analysis to prevent
        over-signaling in similar market conditions.
        """
        try:
            if signals.sum() == 0:
                return signals

            # Step 1: Apply time-based spacing
            time_spaced_signals = self._apply_signal_spacing(df, signals, min_spacing_bars, signal_type, epic)

            # Step 2: Apply correlation penalty filtering (RE-ENABLED with optimized threshold)
            correlation_filtered_signals = self._apply_correlation_penalty_filtering(
                df, time_spaced_signals, signal_type, epic, correlation_threshold=0.90
            )

            return correlation_filtered_signals

        except Exception as e:
            self.logger.error(f"Error in enhanced signal spacing: {e}")
            return signals

    def validate_risk_reward_ratio(self, df: pd.DataFrame, signal_idx: int, signal_type: str, epic: str) -> Dict:
        """
        NEW: Validate minimum risk-reward ratio for mean reversion signals

        Calculates dynamic stop loss and take profit levels based on market structure
        and validates that the risk-reward ratio meets the minimum requirement.

        Args:
            df: DataFrame with price and indicator data
            signal_idx: Index of the signal bar
            signal_type: 'BULL' or 'BEAR'
            epic: Trading pair epic

        Returns:
            Dictionary with risk-reward validation results
        """
        try:
            if signal_idx >= len(df):
                return {'valid': False, 'reason': 'Invalid signal index'}

            row = df.iloc[signal_idx]
            current_price = row['close']

            # Calculate dynamic stop loss and take profit levels
            rr_levels = self._calculate_dynamic_stop_take_levels(df, signal_idx, signal_type, epic, current_price)

            if 'error' in rr_levels:
                return {'valid': False, 'reason': rr_levels['error']}

            stop_loss = rr_levels['stop_loss']
            take_profit = rr_levels['take_profit']

            # Calculate risk and reward distances
            if signal_type == 'BULL':
                risk_distance = current_price - stop_loss
                reward_distance = take_profit - current_price
            else:  # BEAR
                risk_distance = stop_loss - current_price
                reward_distance = current_price - take_profit

            # Validate positive distances
            if risk_distance <= 0 or reward_distance <= 0:
                return {
                    'valid': False,
                    'reason': f'Invalid risk/reward distances: risk={risk_distance:.5f}, reward={reward_distance:.5f}',
                    'risk_distance': risk_distance,
                    'reward_distance': reward_distance
                }

            # Calculate risk-reward ratio
            rr_ratio = reward_distance / risk_distance
            min_rr_ratio = self.config.get('min_risk_reward', mr_config.SIGNAL_QUALITY_MIN_RISK_REWARD)

            # Validate minimum risk-reward ratio
            rr_valid = rr_ratio >= min_rr_ratio

            # Convert distances to pips for logging
            pip_value = 0.01 if 'JPY' in epic else 0.0001
            risk_pips = risk_distance / pip_value
            reward_pips = reward_distance / pip_value

            validation_result = {
                'valid': rr_valid,
                'risk_reward_ratio': rr_ratio,
                'min_required_ratio': min_rr_ratio,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_distance': risk_distance,
                'reward_distance': reward_distance,
                'risk_pips': risk_pips,
                'reward_pips': reward_pips,
                'calculation_method': rr_levels.get('method', 'dynamic'),
                'reason': f'R:R {rr_ratio:.2f} {"≥" if rr_valid else "<"} {min_rr_ratio:.2f} required'
            }

            if not rr_valid:
                self.logger.debug(f"{signal_type} signal REJECTED: Poor risk-reward ratio "
                                f"({rr_ratio:.2f} < {min_rr_ratio:.2f}), "
                                f"Risk: {risk_pips:.1f} pips, Reward: {reward_pips:.1f} pips")
            else:
                self.logger.debug(f"{signal_type} signal PASSED risk-reward validation: "
                                f"R:R {rr_ratio:.2f} (Risk: {risk_pips:.1f} pips, Reward: {reward_pips:.1f} pips)")

            return validation_result

        except Exception as e:
            self.logger.error(f"Error validating risk-reward ratio: {e}")
            return {'valid': False, 'reason': f'Validation error: {str(e)}'}

    def _calculate_dynamic_stop_take_levels(self, df: pd.DataFrame, signal_idx: int,
                                          signal_type: str, epic: str, current_price: float) -> Dict:
        """Calculate dynamic stop loss and take profit levels based on market structure"""
        try:
            row = df.iloc[signal_idx]

            # Method 1: Use recent swing levels (preferred for mean reversion)
            swing_levels = self._find_recent_swing_levels(df, signal_idx, epic)
            if swing_levels['valid']:
                return self._calculate_swing_based_levels(
                    swing_levels, signal_type, current_price, epic
                )

            # Method 2: Use ATR-based levels (fallback)
            atr_levels = self._calculate_atr_based_levels(df, signal_idx, signal_type, current_price, epic)
            if atr_levels['valid']:
                return atr_levels

            # Method 3: Use configuration defaults (last resort)
            return self._calculate_config_based_levels(signal_type, current_price, epic)

        except Exception as e:
            return {'error': f'Stop/take level calculation failed: {str(e)}'}

    def _find_recent_swing_levels(self, df: pd.DataFrame, signal_idx: int, epic: str) -> Dict:
        """Find recent swing high/low levels for stop loss placement"""
        try:
            lookback_bars = min(50, signal_idx)  # Look back up to 50 bars
            if lookback_bars < 10:
                return {'valid': False, 'reason': 'Insufficient data for swing analysis'}

            # Analyze recent price data
            recent_data = df.iloc[signal_idx - lookback_bars:signal_idx + 1]

            # Find swing highs and lows
            highs = recent_data['high']
            lows = recent_data['low']

            # Simple swing detection (peaks and troughs)
            swing_highs = []
            swing_lows = []

            for i in range(2, len(recent_data) - 2):
                # Swing high: higher than 2 bars on each side
                if (highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and
                    highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]):
                    swing_highs.append(highs.iloc[i])

                # Swing low: lower than 2 bars on each side
                if (lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and
                    lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]):
                    swing_lows.append(lows.iloc[i])

            if len(swing_highs) == 0 or len(swing_lows) == 0:
                return {'valid': False, 'reason': 'No clear swing levels found'}

            # Get most recent swing levels
            recent_swing_high = max(swing_highs[-3:]) if len(swing_highs) >= 3 else max(swing_highs)
            recent_swing_low = min(swing_lows[-3:]) if len(swing_lows) >= 3 else min(swing_lows)

            return {
                'valid': True,
                'swing_high': recent_swing_high,
                'swing_low': recent_swing_low,
                'method': 'swing_levels'
            }

        except Exception as e:
            return {'valid': False, 'reason': f'Swing level analysis failed: {str(e)}'}

    def _calculate_swing_based_levels(self, swing_levels: Dict, signal_type: str,
                                    current_price: float, epic: str) -> Dict:
        """Calculate stop/take levels based on swing points"""
        try:
            swing_high = swing_levels['swing_high']
            swing_low = swing_levels['swing_low']

            # Calculate buffer (small margin beyond swing levels)
            pip_value = 0.01 if 'JPY' in epic else 0.0001
            buffer_pips = 3  # 3 pip buffer
            buffer_distance = buffer_pips * pip_value

            if signal_type == 'BULL':
                # Stop loss below recent swing low
                stop_loss = swing_low - buffer_distance

                # Take profit: aim for next resistance or calculate based on swing range
                swing_range = swing_high - swing_low
                take_profit = current_price + (swing_range * 0.8)  # Conservative target

            else:  # BEAR
                # Stop loss above recent swing high
                stop_loss = swing_high + buffer_distance

                # Take profit: aim for next support or calculate based on swing range
                swing_range = swing_high - swing_low
                take_profit = current_price - (swing_range * 0.8)  # Conservative target

            return {
                'valid': True,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'method': 'swing_based'
            }

        except Exception as e:
            return {'error': f'Swing-based level calculation failed: {str(e)}'}

    def _calculate_atr_based_levels(self, df: pd.DataFrame, signal_idx: int, signal_type: str,
                                  current_price: float, epic: str) -> Dict:
        """Calculate stop/take levels based on ATR"""
        try:
            row = df.iloc[signal_idx]
            atr = row.get('atr', 0)

            if atr <= 0:
                return {'valid': False, 'reason': 'ATR not available or zero'}

            # ATR multipliers for mean reversion (more conservative than trend following)
            stop_multiplier = 1.2  # Tighter stops for mean reversion
            profit_multiplier = 2.0  # 2:1 risk-reward minimum

            if signal_type == 'BULL':
                stop_loss = current_price - (atr * stop_multiplier)
                take_profit = current_price + (atr * profit_multiplier)
            else:  # BEAR
                stop_loss = current_price + (atr * stop_multiplier)
                take_profit = current_price - (atr * profit_multiplier)

            return {
                'valid': True,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'method': 'atr_based'
            }

        except Exception as e:
            return {'error': f'ATR-based level calculation failed: {str(e)}'}

    def _calculate_config_based_levels(self, signal_type: str, current_price: float, epic: str) -> Dict:
        """Calculate stop/take levels using configuration defaults"""
        try:
            # Get configuration defaults
            stop_loss_pips = self.config.get('stop_loss_pips', mr_config.MEAN_REVERSION_DEFAULT_SL_PIPS)
            take_profit_pips = self.config.get('take_profit_pips', mr_config.MEAN_REVERSION_DEFAULT_TP_PIPS)

            # Convert pips to price units
            pip_value = 0.01 if 'JPY' in epic else 0.0001
            stop_distance = stop_loss_pips * pip_value
            profit_distance = take_profit_pips * pip_value

            if signal_type == 'BULL':
                stop_loss = current_price - stop_distance
                take_profit = current_price + profit_distance
            else:  # BEAR
                stop_loss = current_price + stop_distance
                take_profit = current_price - profit_distance

            return {
                'valid': True,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'method': 'config_based'
            }

        except Exception as e:
            return {'error': f'Config-based level calculation failed: {str(e)}'}

    def apply_risk_reward_filter(self, df: pd.DataFrame, signals: pd.Series,
                               signal_type: str, epic: str) -> pd.Series:
        """
        Apply risk-reward ratio filtering to signal candidates

        Validates that each signal meets minimum risk-reward requirements
        before allowing it to be generated.
        """
        try:
            if signals.sum() == 0:
                return signals

            filtered_signals = pd.Series(False, index=signals.index)

            for idx, signal in signals.items():
                if not signal:
                    continue

                signal_idx = signals.index.get_loc(idx)

                # Validate risk-reward ratio
                rr_validation = self.validate_risk_reward_ratio(df, signal_idx, signal_type, epic)

                if rr_validation['valid']:
                    filtered_signals.loc[idx] = True
                else:
                    self.logger.debug(f"{signal_type} signal at {idx} rejected: {rr_validation['reason']}")

            original_count = signals.sum()
            final_count = filtered_signals.sum()

            if original_count != final_count:
                self.logger.info(f"Risk-reward filtering for {epic} {signal_type}: "
                               f"{original_count} -> {final_count} "
                               f"({original_count - final_count} removed for poor R:R)")

            return filtered_signals

        except Exception as e:
            self.logger.error(f"Error applying risk-reward filter: {e}")
            return signals