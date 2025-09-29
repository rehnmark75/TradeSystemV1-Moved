# core/strategies/helpers/ichimoku_bayesian_confidence.py
"""
Ichimoku Bayesian Confidence System
Dynamic confidence scoring based on historical performance and Bayesian inference

This module implements:
1. Bayesian posterior confidence calculation based on historical signal success
2. Market regime-specific performance tracking (trending/ranging/volatile)
3. Pair-specific confidence adaptation based on currency characteristics
4. Dynamic confidence threshold adjustment based on recent performance
5. Prior belief updating with empirical evidence

Features:
- Beta-Binomial Bayesian model for signal success probability
- Market regime detection and regime-specific priors
- Exponential decay for recent performance weighting
- Confidence threshold adaptation based on performance drift
- Signal quality scoring with Bayesian confidence intervals
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy.stats import beta
import json


class IchimokuBayesianConfidence:
    """Bayesian confidence system for dynamic Ichimoku signal quality assessment"""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

        # Bayesian parameters
        self.bayesian_params = {
            # Beta distribution priors (alpha, beta)
            'prior_alpha': 2.0,              # Prior successful signals
            'prior_beta': 1.0,               # Prior failed signals
            'decay_factor': 0.95,            # Exponential decay for historical data
            'min_samples': 10,               # Minimum samples for Bayesian inference
            'confidence_interval': 0.95,     # Confidence interval for probability estimates
            'update_frequency_bars': 20,     # How often to update Bayesian estimates
        }

        # Market regime classification parameters
        self.regime_params = {
            'trend_strength_threshold': 0.6,    # Threshold for trending vs ranging
            'volatility_percentile_high': 75,   # High volatility threshold (percentile)
            'volatility_percentile_low': 25,    # Low volatility threshold (percentile)
            'regime_lookback': 50,              # Bars for regime classification
        }

        # Confidence adjustment parameters
        self.confidence_params = {
            'base_confidence_threshold': 0.60,   # Base confidence threshold
            'confidence_adjustment_range': 0.20, # Max adjustment range (+/-)
            'performance_window': 30,            # Window for performance evaluation
            'adaptation_sensitivity': 0.1,       # How quickly to adapt thresholds
        }

        # Initialize performance tracking
        self.performance_history = {}
        self.regime_performance = {}
        self.pair_performance = {}
        self.bayesian_estimates = {}

    def update_bayesian_confidence(self, df: pd.DataFrame, epic: str,
                                   signal_outcomes: List[Dict] = None) -> pd.DataFrame:
        """
        Update Bayesian confidence estimates based on historical performance

        Args:
            df: DataFrame with Ichimoku indicators
            epic: Currency pair identifier
            signal_outcomes: List of past signal outcomes with results

        Returns:
            DataFrame with Bayesian confidence estimates added
        """
        try:
            df_updated = df.copy()

            # Detect market regimes
            df_updated = self._detect_market_regimes(df_updated)

            # Update performance tracking with signal outcomes
            if signal_outcomes:
                self._update_performance_tracking(epic, signal_outcomes)

            # Calculate Bayesian confidence estimates
            df_updated = self._calculate_bayesian_estimates(df_updated, epic)

            # Apply confidence threshold adaptation
            df_updated = self._apply_adaptive_thresholds(df_updated, epic)

            # Calculate final Bayesian confidence scores
            df_updated = self._calculate_bayesian_confidence_scores(df_updated, epic)

            self.logger.info(f"Bayesian confidence updated for {epic}")
            return df_updated

        except Exception as e:
            self.logger.error(f"Error updating Bayesian confidence: {e}")
            return df

    def _detect_market_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect market regimes: trending/ranging and volatility levels"""
        try:
            lookback = self.regime_params['regime_lookback']

            # Calculate trend strength using linear regression slope
            def calculate_trend_strength(prices, window=20):
                """Calculate trend strength using R-squared of linear regression"""
                if len(prices) < window:
                    return 0.0

                x = np.arange(len(prices))
                correlation = np.corrcoef(x, prices)[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0

            # Rolling trend strength calculation
            df['trend_strength'] = df['close'].rolling(window=20).apply(
                lambda x: calculate_trend_strength(x.values), raw=False
            )

            # Volatility calculation (ATR-based)
            if 'atr_normalized' not in df.columns:
                df['atr'] = df[['high', 'low', 'close']].apply(
                    lambda x: max(x['high'] - x['low'],
                                abs(x['high'] - x['close']),
                                abs(x['low'] - x['close'])), axis=1
                ).rolling(window=14).mean()
                df['atr_normalized'] = df['atr'] / df['close']

            # Calculate volatility percentiles
            volatility_high_threshold = df['atr_normalized'].rolling(window=lookback).quantile(
                self.regime_params['volatility_percentile_high'] / 100.0
            )
            volatility_low_threshold = df['atr_normalized'].rolling(window=lookback).quantile(
                self.regime_params['volatility_percentile_low'] / 100.0
            )

            # Market regime classification
            trend_threshold = self.regime_params['trend_strength_threshold']

            df['market_regime'] = 'ranging'  # Default
            df.loc[df['trend_strength'] >= trend_threshold, 'market_regime'] = 'trending'

            # Volatility regime
            df['volatility_regime'] = 'medium'  # Default
            df.loc[df['atr_normalized'] >= volatility_high_threshold, 'volatility_regime'] = 'high'
            df.loc[df['atr_normalized'] <= volatility_low_threshold, 'volatility_regime'] = 'low'

            # Combined regime
            df['combined_regime'] = df['market_regime'] + '_' + df['volatility_regime']

            self.logger.debug("Market regimes detected successfully")
            return df

        except Exception as e:
            self.logger.error(f"Error detecting market regimes: {e}")
            return df

    def _update_performance_tracking(self, epic: str, signal_outcomes: List[Dict]):
        """Update performance tracking with new signal outcomes"""
        try:
            if epic not in self.performance_history:
                self.performance_history[epic] = []

            # Add new outcomes to history
            for outcome in signal_outcomes:
                outcome_record = {
                    'timestamp': outcome.get('timestamp', datetime.now()),
                    'signal_type': outcome.get('signal_type', 'unknown'),
                    'regime': outcome.get('regime', 'unknown'),
                    'success': outcome.get('success', False),
                    'confidence': outcome.get('confidence', 0.0),
                    'return': outcome.get('return', 0.0)
                }
                self.performance_history[epic].append(outcome_record)

            # Limit history size (keep last 1000 records)
            if len(self.performance_history[epic]) > 1000:
                self.performance_history[epic] = self.performance_history[epic][-1000:]

            self.logger.debug(f"Performance tracking updated for {epic}")

        except Exception as e:
            self.logger.error(f"Error updating performance tracking: {e}")

    def _calculate_bayesian_estimates(self, df: pd.DataFrame, epic: str) -> pd.DataFrame:
        """Calculate Bayesian probability estimates for signal success"""
        try:
            prior_alpha = self.bayesian_params['prior_alpha']
            prior_beta = self.bayesian_params['prior_beta']
            decay_factor = self.bayesian_params['decay_factor']
            min_samples = self.bayesian_params['min_samples']

            # Initialize Bayesian estimates for different regimes
            df['bayesian_success_prob'] = 0.5  # Default neutral probability
            df['bayesian_confidence_interval_lower'] = 0.3
            df['bayesian_confidence_interval_upper'] = 0.7

            # Get performance history for this epic
            if epic not in self.performance_history or len(self.performance_history[epic]) < min_samples:
                self.logger.debug(f"Insufficient data for Bayesian estimates for {epic}")
                return df

            history = self.performance_history[epic]

            # Calculate Bayesian estimates for each row based on regime
            for idx, row in df.iterrows():
                current_regime = row.get('combined_regime', 'unknown')

                # Filter relevant historical data for this regime
                regime_history = [h for h in history if h['regime'] == current_regime]

                if len(regime_history) < min_samples:
                    # Fall back to overall performance if insufficient regime-specific data
                    regime_history = history

                if len(regime_history) >= min_samples:
                    # Apply exponential decay to historical data
                    current_time = datetime.now()
                    weights = []
                    successes = []

                    for h in regime_history:
                        time_diff = (current_time - h['timestamp']).total_seconds() / (24 * 3600)  # Days
                        weight = decay_factor ** time_diff
                        weights.append(weight)
                        successes.append(1 if h['success'] else 0)

                    # Calculate weighted success rate
                    weighted_successes = sum(w * s for w, s in zip(weights, successes))
                    weighted_total = sum(weights)

                    # Bayesian update: Beta(alpha + successes, beta + failures)
                    alpha_posterior = prior_alpha + weighted_successes
                    beta_posterior = prior_beta + (weighted_total - weighted_successes)

                    # Calculate probability estimate and confidence interval
                    success_prob = alpha_posterior / (alpha_posterior + beta_posterior)

                    # Calculate confidence interval using Beta distribution
                    confidence_level = self.bayesian_params['confidence_interval']
                    alpha_ci = (1 - confidence_level) / 2

                    ci_lower = beta.ppf(alpha_ci, alpha_posterior, beta_posterior)
                    ci_upper = beta.ppf(1 - alpha_ci, alpha_posterior, beta_posterior)

                    df.at[idx, 'bayesian_success_prob'] = success_prob
                    df.at[idx, 'bayesian_confidence_interval_lower'] = ci_lower
                    df.at[idx, 'bayesian_confidence_interval_upper'] = ci_upper

                    # Store estimate for later use
                    regime_key = f"{epic}_{current_regime}"
                    self.bayesian_estimates[regime_key] = {
                        'alpha': alpha_posterior,
                        'beta': beta_posterior,
                        'success_prob': success_prob,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'sample_count': len(regime_history)
                    }

            self.logger.debug("Bayesian estimates calculated successfully")
            return df

        except Exception as e:
            self.logger.error(f"Error calculating Bayesian estimates: {e}")
            return df

    def _apply_adaptive_thresholds(self, df: pd.DataFrame, epic: str) -> pd.DataFrame:
        """Apply adaptive confidence thresholds based on recent performance"""
        try:
            base_threshold = self.confidence_params['base_confidence_threshold']
            adjustment_range = self.confidence_params['confidence_adjustment_range']
            performance_window = self.confidence_params['performance_window']
            sensitivity = self.confidence_params['adaptation_sensitivity']

            # Calculate performance-based threshold adjustment
            df['adaptive_confidence_threshold'] = base_threshold

            if epic in self.performance_history and len(self.performance_history[epic]) > 0:
                # Get recent performance
                recent_history = self.performance_history[epic][-performance_window:]

                if len(recent_history) >= 5:  # Minimum for meaningful adjustment
                    recent_success_rate = np.mean([h['success'] for h in recent_history])

                    # Calculate adjustment based on performance vs expected
                    expected_success_rate = 0.55  # Expected baseline success rate
                    performance_delta = recent_success_rate - expected_success_rate

                    # Apply sensitivity-weighted adjustment
                    threshold_adjustment = performance_delta * sensitivity * adjustment_range

                    # Clip adjustment to reasonable range
                    threshold_adjustment = np.clip(threshold_adjustment, -adjustment_range, adjustment_range)

                    # Apply adjustment
                    df['adaptive_confidence_threshold'] = base_threshold + threshold_adjustment

            # Ensure thresholds stay within reasonable bounds
            df['adaptive_confidence_threshold'] = np.clip(
                df['adaptive_confidence_threshold'], 0.40, 0.80
            )

            self.logger.debug("Adaptive thresholds applied successfully")
            return df

        except Exception as e:
            self.logger.error(f"Error applying adaptive thresholds: {e}")
            return df

    def _calculate_bayesian_confidence_scores(self, df: pd.DataFrame, epic: str) -> pd.DataFrame:
        """Calculate final Bayesian confidence scores"""
        try:
            # Bayesian confidence multiplier based on success probability
            df['bayesian_confidence_multiplier'] = df['bayesian_success_prob']

            # Confidence interval width (narrower = more confident)
            df['confidence_interval_width'] = (
                df['bayesian_confidence_interval_upper'] -
                df['bayesian_confidence_interval_lower']
            )

            # Precision bonus (narrower confidence intervals get higher confidence)
            max_ci_width = 0.4  # Maximum expected CI width
            df['precision_bonus'] = 1.0 + (max_ci_width - df['confidence_interval_width']) / max_ci_width * 0.2

            # Final Bayesian confidence score
            df['bayesian_confidence_score'] = (
                df['bayesian_confidence_multiplier'] * df['precision_bonus']
            )

            # Clip to reasonable range
            df['bayesian_confidence_score'] = np.clip(df['bayesian_confidence_score'], 0.3, 1.5)

            # Bayesian signal quality flag
            df['bayesian_quality_passed'] = (
                df['bayesian_success_prob'] >= df['adaptive_confidence_threshold']
            )

            self.logger.debug("Bayesian confidence scores calculated successfully")
            return df

        except Exception as e:
            self.logger.error(f"Error calculating Bayesian confidence scores: {e}")
            return df

    def get_bayesian_signal_assessment(self, latest_row: pd.Series, epic: str,
                                       signal_type: str) -> Dict:
        """
        Get comprehensive Bayesian assessment for a signal

        Args:
            latest_row: Latest data row
            epic: Currency pair
            signal_type: 'BULL' or 'BEAR'

        Returns:
            Dictionary with Bayesian assessment results
        """
        try:
            assessment = {
                'bayesian_success_probability': latest_row.get('bayesian_success_prob', 0.5),
                'confidence_interval_lower': latest_row.get('bayesian_confidence_interval_lower', 0.3),
                'confidence_interval_upper': latest_row.get('bayesian_confidence_interval_upper', 0.7),
                'adaptive_threshold': latest_row.get('adaptive_confidence_threshold', 0.6),
                'bayesian_confidence_score': latest_row.get('bayesian_confidence_score', 1.0),
                'quality_passed': latest_row.get('bayesian_quality_passed', False),
                'market_regime': latest_row.get('combined_regime', 'unknown'),
                'confidence_interval_width': latest_row.get('confidence_interval_width', 0.4),
                'precision_rating': 'low'
            }

            # Determine precision rating
            ci_width = assessment['confidence_interval_width']
            if ci_width <= 0.2:
                assessment['precision_rating'] = 'high'
            elif ci_width <= 0.3:
                assessment['precision_rating'] = 'medium'
            else:
                assessment['precision_rating'] = 'low'

            # Recommendation based on Bayesian analysis
            success_prob = assessment['bayesian_success_probability']
            threshold = assessment['adaptive_threshold']

            if success_prob >= threshold + 0.1:
                assessment['recommendation'] = 'strong_take'
            elif success_prob >= threshold:
                assessment['recommendation'] = 'take'
            elif success_prob >= threshold - 0.05:
                assessment['recommendation'] = 'weak_take'
            else:
                assessment['recommendation'] = 'skip'

            return assessment

        except Exception as e:
            self.logger.error(f"Error getting Bayesian signal assessment: {e}")
            return {
                'bayesian_success_probability': 0.5,
                'confidence_interval_lower': 0.3,
                'confidence_interval_upper': 0.7,
                'adaptive_threshold': 0.6,
                'bayesian_confidence_score': 1.0,
                'quality_passed': False,
                'market_regime': 'unknown',
                'confidence_interval_width': 0.4,
                'precision_rating': 'low',
                'recommendation': 'skip'
            }

    def get_performance_summary(self, epic: str) -> Dict:
        """Get performance summary for an epic"""
        try:
            if epic not in self.performance_history:
                return {'total_signals': 0, 'success_rate': 0.0}

            history = self.performance_history[epic]

            if len(history) == 0:
                return {'total_signals': 0, 'success_rate': 0.0}

            total_signals = len(history)
            successful_signals = sum(1 for h in history if h['success'])
            success_rate = successful_signals / total_signals

            # Recent performance (last 30 signals)
            recent_history = history[-30:]
            recent_success_rate = np.mean([h['success'] for h in recent_history]) if recent_history else 0.0

            # Performance by regime
            regime_performance = {}
            for regime in ['trending_high', 'trending_medium', 'trending_low',
                          'ranging_high', 'ranging_medium', 'ranging_low']:
                regime_signals = [h for h in history if h.get('regime') == regime]
                if regime_signals:
                    regime_performance[regime] = {
                        'total': len(regime_signals),
                        'success_rate': np.mean([h['success'] for h in regime_signals])
                    }

            return {
                'total_signals': total_signals,
                'success_rate': success_rate,
                'recent_success_rate': recent_success_rate,
                'regime_performance': regime_performance,
                'avg_return': np.mean([h.get('return', 0) for h in history]),
                'best_regime': max(regime_performance.items(),
                                 key=lambda x: x[1]['success_rate'])[0] if regime_performance else None
            }

        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {'total_signals': 0, 'success_rate': 0.0}

    def export_bayesian_state(self) -> Dict:
        """Export current Bayesian state for persistence"""
        try:
            return {
                'bayesian_estimates': self.bayesian_estimates,
                'performance_history': {
                    epic: [
                        {
                            'timestamp': h['timestamp'].isoformat(),
                            'signal_type': h['signal_type'],
                            'regime': h['regime'],
                            'success': h['success'],
                            'confidence': h['confidence'],
                            'return': h['return']
                        }
                        for h in history[-100:]  # Keep last 100 for export
                    ]
                    for epic, history in self.performance_history.items()
                }
            }
        except Exception as e:
            self.logger.error(f"Error exporting Bayesian state: {e}")
            return {}

    def import_bayesian_state(self, state: Dict):
        """Import Bayesian state from persistence"""
        try:
            if 'bayesian_estimates' in state:
                self.bayesian_estimates = state['bayesian_estimates']

            if 'performance_history' in state:
                for epic, history in state['performance_history'].items():
                    self.performance_history[epic] = []
                    for h in history:
                        record = h.copy()
                        record['timestamp'] = datetime.fromisoformat(h['timestamp'])
                        self.performance_history[epic].append(record)

            self.logger.info("Bayesian state imported successfully")

        except Exception as e:
            self.logger.error(f"Error importing Bayesian state: {e}")