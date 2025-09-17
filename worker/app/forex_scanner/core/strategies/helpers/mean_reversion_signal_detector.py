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

    def __init__(self, logger: logging.Logger = None, indicator_calculator=None):
        self.logger = logger or logging.getLogger(__name__)
        self.indicator_calculator = indicator_calculator

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

            # 3. Apply market regime filtering
            if self.config['market_regime_enabled']:
                df_signals = self._apply_market_regime_filter(df_signals, epic)

            # 4. Calculate confidence scores
            df_signals = self._calculate_signal_confidence(df_signals, epic)

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
        Detect primary mean reversion signals from LuxAlgo oscillator

        Primary signals occur when:
        1. LuxAlgo oscillator reaches extreme levels (overbought/oversold)
        2. Oscillator shows reversal momentum (histogram turning)
        3. Basic confluence requirements met
        """
        try:
            df_primary = df.copy()

            # Get epic-specific thresholds
            epic_thresholds = mr_config.get_mean_reversion_threshold_for_epic(epic)

            # Bull signal conditions: Extreme oversold with reversal momentum
            luxalgo_oversold = df_primary['luxalgo_oscillator'] < epic_thresholds['luxalgo_oversold']
            luxalgo_extreme_oversold = df_primary['luxalgo_oscillator'] < self.config['luxalgo_extreme_os']
            luxalgo_bull_momentum = (df_primary['luxalgo_histogram'] > 0) & (df_primary['luxalgo_histogram'].shift(1) <= 0)

            # Primary bull signals
            primary_bull = (
                (luxalgo_oversold | luxalgo_extreme_oversold) &
                (luxalgo_bull_momentum | (df_primary['luxalgo_histogram'] > 0))
            )

            # Bear signal conditions: Extreme overbought with reversal momentum
            luxalgo_overbought = df_primary['luxalgo_oscillator'] > epic_thresholds['luxalgo_overbought']
            luxalgo_extreme_overbought = df_primary['luxalgo_oscillator'] > self.config['luxalgo_extreme_ob']
            luxalgo_bear_momentum = (df_primary['luxalgo_histogram'] < 0) & (df_primary['luxalgo_histogram'].shift(1) >= 0)

            # Primary bear signals
            primary_bear = (
                (luxalgo_overbought | luxalgo_extreme_overbought) &
                (luxalgo_bear_momentum | (df_primary['luxalgo_histogram'] < 0))
            )

            # Store primary signals for confluence filtering
            df_primary['primary_bull'] = primary_bull
            df_primary['primary_bear'] = primary_bear

            primary_bull_count = primary_bull.sum()
            primary_bear_count = primary_bear.sum()
            self.logger.debug(f"Primary oscillator signals for {epic}: {primary_bull_count} bull, {primary_bear_count} bear")

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
        Calculate confidence for a single signal

        Args:
            row: DataFrame row with indicator data
            signal_type: 'BULL' or 'BEAR'
            epic: Trading pair epic

        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            base_confidence = 0.5  # Start at 50%
            epic_thresholds = mr_config.get_mean_reversion_threshold_for_epic(epic)

            # 1. Oscillator Extremity Score (30%)
            luxalgo_osc = row.get('luxalgo_oscillator', 50)
            extremity_score = 0.0

            if signal_type == 'BULL':
                if luxalgo_osc < self.config['luxalgo_extreme_os']:
                    extremity_score = 0.30  # Extreme oversold
                elif luxalgo_osc < epic_thresholds['luxalgo_oversold']:
                    extremity_score = 0.20  # Regular oversold
                elif luxalgo_osc < 40:
                    extremity_score = 0.10  # Mild oversold
            else:  # BEAR
                if luxalgo_osc > self.config['luxalgo_extreme_ob']:
                    extremity_score = 0.30  # Extreme overbought
                elif luxalgo_osc > epic_thresholds['luxalgo_overbought']:
                    extremity_score = 0.20  # Regular overbought
                elif luxalgo_osc > 60:
                    extremity_score = 0.10  # Mild overbought

            base_confidence += extremity_score

            # 2. Confluence Strength Score (25%)
            confluence_score = row.get(f'oscillator_{signal_type.lower()}_score', 0)
            confluence_normalized = min(0.25, confluence_score * 0.25)  # Cap at 25%
            base_confidence += confluence_normalized

            # 3. Divergence Presence Score (20%)
            divergence_score = 0.0
            if signal_type == 'BULL' and row.get('rsi_ema_divergence_bull', False):
                divergence_strength = row.get('divergence_strength', 0)
                divergence_score = 0.15 + (divergence_strength * 0.05)  # Up to 20%
            elif signal_type == 'BEAR' and row.get('rsi_ema_divergence_bear', False):
                divergence_strength = row.get('divergence_strength', 0)
                divergence_score = 0.15 + (divergence_strength * 0.05)  # Up to 20%

            base_confidence += divergence_score

            # 4. Market Regime Suitability Score (15%)
            regime_score = 0.0
            adx = row.get('adx', 0)
            if adx < 25:  # Ranging market (good for mean reversion)
                regime_score = 0.15
            elif adx < 35:  # Moderate trend
                regime_score = 0.10
            elif adx < 50:  # Strong trend (less suitable)
                regime_score = 0.05
            # No points for very strong trend (ADX > 50)

            base_confidence += regime_score

            # 5. Momentum Alignment Score (10%)
            momentum_score = 0.0
            squeeze_momentum = row.get('squeeze_momentum', 0)
            squeeze_on = row.get('squeeze_on', False)

            if not squeeze_on:  # Squeeze released (good for momentum)
                if signal_type == 'BULL' and squeeze_momentum > 0:
                    momentum_score = 0.10
                elif signal_type == 'BEAR' and squeeze_momentum < 0:
                    momentum_score = 0.10
                else:
                    momentum_score = 0.05  # Neutral momentum

            base_confidence += momentum_score

            # Final confidence bounds
            final_confidence = max(0.0, min(1.0, base_confidence))

            if final_confidence >= 0.7:
                self.logger.debug(f"High confidence {signal_type} signal: {final_confidence:.3f} "
                                f"(extremity: {extremity_score:.3f}, confluence: {confluence_normalized:.3f}, "
                                f"divergence: {divergence_score:.3f})")

            return final_confidence

        except Exception as e:
            self.logger.error(f"Error calculating individual signal confidence: {e}")
            return 0.5  # Neutral confidence on error

    def _apply_quality_thresholds(self, df: pd.DataFrame, epic: str) -> pd.DataFrame:
        """Apply minimum quality thresholds to filter out weak signals"""
        try:
            df_quality = df.copy()

            # Get signals with confidence scores
            bull_candidates = df_quality['regime_bull'] if 'regime_bull' in df_quality.columns else df_quality['confluence_bull']
            bear_candidates = df_quality['regime_bear'] if 'regime_bear' in df_quality.columns else df_quality['confluence_bear']

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

            # Apply stricter spacing for backtests (longer intervals)
            min_spacing_hours = self.config['min_signal_spacing_hours'] * 2  # Double the spacing
            min_spacing_bars = int(min_spacing_hours * 4)

            # Apply daily limits
            max_signals_per_day = self.config['max_signals_per_day']

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