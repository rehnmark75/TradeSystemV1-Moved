#!/usr/bin/env python3
"""
Signal Optimization Tools
Advanced signal detection methods to increase signal frequency while maintaining quality

Based on quantitative research: docs/research/signal_frequency_optimization_analysis.md
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm
from dataclasses import dataclass
import logging


@dataclass
class SignalScore:
    """Container for multi-factor signal scoring"""
    probability: float
    component_scores: Dict[str, float]
    confidence: float
    metadata: Dict


class MultiTimeframeMomentum:
    """
    Multi-timeframe momentum divergence detector
    Alternative to binary BOS/CHoCH detection
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def calculate_momentum_score(
        self,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        lookback: int = 20,
        pip_value: float = 0.0001
    ) -> Dict:
        """
        Calculate multi-timeframe momentum score

        Args:
            df_15m: 15-minute timeframe data
            df_1h: 1-hour timeframe data
            df_4h: 4-hour timeframe data
            lookback: Lookback period for momentum calculation
            pip_value: Pip value for the pair

        Returns:
            Dict with momentum_score, direction, and component analysis
        """
        # Ensure enough data
        if len(df_15m) < lookback or len(df_1h) < lookback // 4 or len(df_4h) < lookback // 16:
            return {
                'momentum_score': 0.0,
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'components': {}
            }

        # Calculate Rate of Change (ROC) for each timeframe
        roc_15m = ((df_15m['close'].iloc[-1] / df_15m['close'].iloc[-lookback]) - 1) * 100
        roc_1h = ((df_1h['close'].iloc[-1] / df_1h['close'].iloc[-lookback // 4]) - 1) * 100
        roc_4h = ((df_4h['close'].iloc[-1] / df_4h['close'].iloc[-max(1, lookback // 16)]) - 1) * 100

        # Volume-weighted momentum (if volume available)
        vol_weight = 1.0
        if 'volume' in df_15m.columns:
            vol_15m = df_15m['volume'].iloc[-lookback:].mean()
            vol_avg = df_15m['volume'].iloc[-100:].mean() if len(df_15m) >= 100 else vol_15m
            if vol_avg > 0:
                vol_weight = 1 + ((vol_15m / vol_avg) - 1) * 0.2

        # Weighted combination (exponential timeframe weighting)
        # HTF (4H) gets highest weight as it defines the trend
        momentum_score = (
            0.20 * roc_15m +  # 20% weight on entry timeframe
            0.30 * roc_1h +   # 30% weight on intermediate
            0.50 * roc_4h     # 50% weight on HTF (trend priority)
        ) * vol_weight

        # Calculate volatility-adjusted threshold
        atr_4h = self._calculate_atr(df_4h, 20)
        atr_pips = atr_4h / pip_value
        threshold = 0.5 + (atr_pips / 100) * 0.02  # Scale threshold with volatility

        # Determine direction and confidence
        direction = 'NEUTRAL'
        confidence = 0.0

        if abs(momentum_score) > threshold:
            direction = 'BULL' if momentum_score > 0 else 'BEAR'
            confidence = min(abs(momentum_score) / (threshold * 2), 1.0)

        self.logger.info(f"   Multi-TF Momentum Analysis:")
        self.logger.info(f"      15m ROC: {roc_15m:+.2f}%")
        self.logger.info(f"      1H ROC:  {roc_1h:+.2f}%")
        self.logger.info(f"      4H ROC:  {roc_4h:+.2f}%")
        self.logger.info(f"      Volume Weight: {vol_weight:.2f}x")
        self.logger.info(f"      Momentum Score: {momentum_score:+.2f}")
        self.logger.info(f"      Threshold: ±{threshold:.2f}")
        self.logger.info(f"      Direction: {direction}")
        self.logger.info(f"      Confidence: {confidence*100:.1f}%")

        return {
            'momentum_score': momentum_score,
            'direction': direction,
            'confidence': confidence,
            'components': {
                'roc_15m': roc_15m,
                'roc_1h': roc_1h,
                'roc_4h': roc_4h,
                'vol_weight': vol_weight,
                'threshold': threshold
            }
        }

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(df) < period + 1:
            return 0.0

        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]

        return float(atr) if not pd.isna(atr) else 0.0


class ContinuousScoringSystem:
    """
    Continuous probability scoring system
    Replaces binary filters with graduated confidence scores
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def calculate_signal_probability(
        self,
        signal_features: Dict,
        weights: Optional[Dict[str, float]] = None
    ) -> SignalScore:
        """
        Calculate multi-factor signal probability

        Args:
            signal_features: Dictionary of signal features
            weights: Optional custom weights for components

        Returns:
            SignalScore with probability, component scores, and confidence
        """
        # Default evidence-based weights (from analysis)
        if weights is None:
            weights = {
                'htf': 0.25,      # HTF most important
                'bos': 0.20,      # BOS direction critical
                'ob': 0.15,       # OB quality matters
                'pullback': 0.15, # Entry timing
                'rr': 0.15,       # Risk/reward
                'volume': 0.10    # Confirmation
            }

        scores = {}

        # 1. HTF Trend Strength (sigmoid function centered at 45%)
        htf_strength_raw = signal_features.get('htf_strength', 0.5)
        scores['htf'] = 1 / (1 + np.exp(-10 * (htf_strength_raw - 0.45)))

        # 2. BOS/CHoCH Significance (already 0-1)
        scores['bos'] = signal_features.get('bos_significance', 0.5)

        # 3. Order Block Quality (multi-factor)
        ob_size_pips = signal_features.get('ob_size_pips', 0)
        ob_touches = signal_features.get('ob_touches', 0)
        ob_age_bars = signal_features.get('ob_age_bars', 100)

        ob_size_score = min(ob_size_pips / 15.0, 1.0)  # Normalize to 15 pips
        ob_touch_score = min(ob_touches / 3.0, 1.0)  # Max 3 touches
        ob_age_score = 1 - min(ob_age_bars / 50.0, 1.0)  # Fresher = better
        scores['ob'] = (ob_size_score + ob_touch_score + ob_age_score) / 3

        # 4. Pullback Quality (Gaussian centered at 50% pullback)
        pullback_pct = signal_features.get('pullback_depth', 0.5)
        scores['pullback'] = np.exp(-((pullback_pct - 0.50) ** 2) / (2 * 0.15 ** 2))

        # 5. R:R Ratio (scaled 1.0-3.0 → 0-1)
        rr_ratio = signal_features.get('rr_ratio', 1.0)
        scores['rr'] = min((rr_ratio - 1.0) / 2.0, 1.0)

        # 6. Volume Confirmation (sigmoid at 1.2x volume)
        volume_ratio = signal_features.get('volume_surge', 1.0)
        scores['volume'] = 1 / (1 + np.exp(-3 * (volume_ratio - 1.2)))

        # Weighted combination
        total_score = sum(scores[key] * weights[key] for key in scores if key in weights)

        # Log detailed breakdown
        if self.logger:
            self.logger.info(f"   Continuous Scoring Breakdown:")
            for component, score in scores.items():
                weight = weights.get(component, 0)
                contribution = score * weight
                self.logger.info(f"      {component.upper()}: {score:.3f} × {weight:.2f} = {contribution:.3f}")
            self.logger.info(f"      TOTAL PROBABILITY: {total_score:.3f} ({total_score*100:.1f}%)")

        return SignalScore(
            probability=total_score,
            component_scores=scores,
            confidence=total_score,
            metadata={'weights': weights, 'features': signal_features}
        )


class BayesianSignalCombiner:
    """
    Bayesian inference for combining weak signals
    No single point of failure - accumulates evidence
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def combine_weak_signals(
        self,
        weak_signals: List[Tuple[str, float, str]],
        priors: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Combine weak signals using Bayesian updating

        Args:
            weak_signals: List of (signal_name, evidence_strength, direction)
                         e.g., [('htf_momentum', 0.6, 'BULL'), ('volume_surge', 0.4, 'BULL')]
            priors: Prior probabilities {'BULL': 0.33, 'BEAR': 0.33, 'NEUTRAL': 0.34}

        Returns:
            Posterior probabilities after updating with all evidence
        """
        # Default uniform priors
        if priors is None:
            priors = {'BULL': 0.33, 'BEAR': 0.33, 'NEUTRAL': 0.34}

        # Start with prior probabilities
        posterior = np.array([priors['BULL'], priors['BEAR'], priors['NEUTRAL']])

        if self.logger:
            self.logger.info(f"   Bayesian Signal Combination:")
            self.logger.info(f"      Priors: BULL={priors['BULL']:.2f}, BEAR={priors['BEAR']:.2f}, "
                           f"NEUTRAL={priors['NEUTRAL']:.2f}")
            self.logger.info(f"      Evidence:")

        for signal_name, strength, direction in weak_signals:
            # Likelihood: P(evidence | hypothesis)
            if direction == 'BULL':
                likelihood = np.array([
                    0.5 + 0.5 * strength,  # P(bull_evidence | BULL)
                    0.5 - 0.5 * strength,  # P(bull_evidence | BEAR)
                    0.5                     # P(bull_evidence | NEUTRAL)
                ])
            elif direction == 'BEAR':
                likelihood = np.array([
                    0.5 - 0.5 * strength,  # P(bear_evidence | BULL)
                    0.5 + 0.5 * strength,  # P(bear_evidence | BEAR)
                    0.5                     # P(bear_evidence | NEUTRAL)
                ])
            else:  # NEUTRAL
                likelihood = np.array([0.33, 0.33, 0.34])

            # Bayesian update: P(H|E) = P(E|H) * P(H) / P(E)
            prior_state = posterior.copy()
            posterior = likelihood * posterior
            posterior = posterior / np.sum(posterior)  # Normalize

            if self.logger:
                self.logger.info(f"         {signal_name}: {direction} (strength={strength:.2f})")
                self.logger.info(f"            Prior: BULL={prior_state[0]:.3f}, BEAR={prior_state[1]:.3f}")
                self.logger.info(f"            Posterior: BULL={posterior[0]:.3f}, BEAR={posterior[1]:.3f}")

        result = {
            'BULL': float(posterior[0]),
            'BEAR': float(posterior[1]),
            'NEUTRAL': float(posterior[2])
        }

        if self.logger:
            self.logger.info(f"      Final Posteriors:")
            self.logger.info(f"         BULL: {result['BULL']*100:.1f}%")
            self.logger.info(f"         BEAR: {result['BEAR']*100:.1f}%")
            self.logger.info(f"         NEUTRAL: {result['NEUTRAL']*100:.1f}%")

        return result


class VolatilityRegimeDetector:
    """
    Adaptive volatility-based threshold adjustment
    Adjusts filter strictness based on market conditions
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def calculate_volatility_regime(
        self,
        df_4h: pd.DataFrame,
        lookback: int = 100
    ) -> Tuple[str, float]:
        """
        Classify market into volatility regimes

        Args:
            df_4h: 4-hour timeframe data
            lookback: Lookback period for regime calculation

        Returns:
            Tuple of (regime_name, current_atr)
        """
        if len(df_4h) < lookback + 14:
            return 'normal', 0.0

        # Calculate ATR
        atr_values = []
        for i in range(len(df_4h) - 14, len(df_4h)):
            if i < 1:
                continue
            high = df_4h['high'].iloc[i]
            low = df_4h['low'].iloc[i]
            close_prev = df_4h['close'].iloc[i - 1]

            tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
            atr_values.append(tr)

        if not atr_values:
            return 'normal', 0.0

        atr_current = np.mean(atr_values[-14:])

        # Historical ATR distribution
        historical_atr = []
        for i in range(max(14, len(df_4h) - lookback), len(df_4h)):
            if i < 1:
                continue
            high = df_4h['high'].iloc[i]
            low = df_4h['low'].iloc[i]
            close_prev = df_4h['close'].iloc[i - 1]

            tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
            historical_atr.append(tr)

        if len(historical_atr) < 20:
            return 'normal', atr_current

        atr_mean = np.mean(historical_atr)
        atr_std = np.std(historical_atr)

        # Calculate z-score
        z_score = (atr_current - atr_mean) / atr_std if atr_std > 0 else 0

        # Classify regime
        if z_score > 1.5:
            regime = 'high_volatility'
        elif z_score < -1.0:
            regime = 'low_volatility'
        else:
            regime = 'normal'

        if self.logger:
            self.logger.info(f"   Volatility Regime Analysis:")
            self.logger.info(f"      Current ATR: {atr_current:.5f}")
            self.logger.info(f"      Historical Mean: {atr_mean:.5f}")
            self.logger.info(f"      Z-Score: {z_score:+.2f}")
            self.logger.info(f"      Regime: {regime.upper()}")

        return regime, atr_current

    def adaptive_thresholds(
        self,
        regime: str,
        base_config: Dict
    ) -> Dict:
        """
        Adjust thresholds based on volatility regime

        Args:
            regime: Volatility regime ('high_volatility', 'low_volatility', 'normal')
            base_config: Base configuration dictionary

        Returns:
            Adjusted configuration dictionary
        """
        adjustments = base_config.copy()

        if regime == 'high_volatility':
            # High volatility: Relax thresholds (more opportunities, accept higher noise)
            adjustments['SMC_MIN_HTF_STRENGTH'] = base_config.get('SMC_MIN_HTF_STRENGTH', 0.55) * 0.85
            adjustments['SMC_MIN_BOS_SIGNIFICANCE'] = base_config.get('SMC_MIN_BOS_SIGNIFICANCE', 0.55) * 0.90
            adjustments['SMC_MIN_RR_RATIO'] = base_config.get('SMC_MIN_RR_RATIO', 1.8) * 1.1  # Wider stops
            adjustments['SMC_SWING_EXHAUSTION_THRESHOLD'] = 0.15  # Wider exhaustion zone

            if self.logger:
                self.logger.info(f"   HIGH VOLATILITY Adjustments:")
                self.logger.info(f"      HTF Strength: {base_config.get('SMC_MIN_HTF_STRENGTH', 0.55)*100:.0f}% "
                               f"→ {adjustments['SMC_MIN_HTF_STRENGTH']*100:.0f}%")
                self.logger.info(f"      Min R:R: {base_config.get('SMC_MIN_RR_RATIO', 1.8):.1f} "
                               f"→ {adjustments['SMC_MIN_RR_RATIO']:.1f}")

        elif regime == 'low_volatility':
            # Low volatility: Tighten thresholds (fewer opportunities, higher quality)
            adjustments['SMC_MIN_HTF_STRENGTH'] = base_config.get('SMC_MIN_HTF_STRENGTH', 0.55) * 1.1
            adjustments['SMC_MIN_BOS_SIGNIFICANCE'] = base_config.get('SMC_MIN_BOS_SIGNIFICANCE', 0.55) * 1.1
            adjustments['SMC_MIN_RR_RATIO'] = base_config.get('SMC_MIN_RR_RATIO', 1.8) * 0.9  # Tighter stops
            adjustments['SMC_SWING_EXHAUSTION_THRESHOLD'] = 0.08  # Tighter exhaustion zone

            if self.logger:
                self.logger.info(f"   LOW VOLATILITY Adjustments:")
                self.logger.info(f"      HTF Strength: {base_config.get('SMC_MIN_HTF_STRENGTH', 0.55)*100:.0f}% "
                               f"→ {adjustments['SMC_MIN_HTF_STRENGTH']*100:.0f}%")
                self.logger.info(f"      Min R:R: {base_config.get('SMC_MIN_RR_RATIO', 1.8):.1f} "
                               f"→ {adjustments['SMC_MIN_RR_RATIO']:.1f}")

        else:
            # Normal volatility: Use base configuration
            if self.logger:
                self.logger.info(f"   NORMAL VOLATILITY: Using base configuration")

        return adjustments


class BayesianAdaptiveSystem:
    """
    Online learning system to adapt thresholds based on recent performance
    Implements Bayesian updating of threshold distributions
    """

    def __init__(self, update_frequency: int = 50):
        """
        Args:
            update_frequency: Update threshold beliefs after this many trades
        """
        # Prior beliefs about optimal thresholds
        self.threshold_priors = {
            'htf_strength': {'mean': 0.50, 'std': 0.10},
            'min_rr': {'mean': 1.8, 'std': 0.3},
            'confidence': {'mean': 0.65, 'std': 0.10}
        }

        # Track recent outcomes
        self.recent_outcomes = []  # List of dicts with thresholds, profit, win
        self.update_frequency = update_frequency

    def update_posteriors(self):
        """
        Update threshold distributions based on recent outcomes
        Using Bayesian updating with Gaussian conjugate priors
        """
        if len(self.recent_outcomes) < 10:
            return  # Need minimum data

        for threshold_name in self.threshold_priors:
            # Get outcomes for this threshold
            threshold_outcomes = [
                (o['thresholds'][threshold_name], o['profit'])
                for o in self.recent_outcomes
                if threshold_name in o['thresholds']
            ]

            if not threshold_outcomes:
                continue

            # Extract values
            threshold_values = np.array([t for t, _ in threshold_outcomes])
            profits = np.array([p for _, p in threshold_outcomes])

            # Weight recent outcomes more heavily (exponential decay)
            weights = np.exp(-0.05 * np.arange(len(profits))[::-1])
            weights /= weights.sum()

            # Calculate weighted statistics (only for profitable trades)
            profitable_mask = profits > 0
            if profitable_mask.sum() > 0:
                weighted_mean = np.average(
                    threshold_values[profitable_mask],
                    weights=weights[profitable_mask] / weights[profitable_mask].sum()
                )
                weighted_std = np.sqrt(
                    np.average(
                        (threshold_values[profitable_mask] - weighted_mean) ** 2,
                        weights=weights[profitable_mask] / weights[profitable_mask].sum()
                    )
                )
            else:
                # No profitable trades, use all data
                weighted_mean = np.average(threshold_values, weights=weights)
                weighted_std = np.sqrt(np.average((threshold_values - weighted_mean) ** 2, weights=weights))

            # Bayesian update (combine prior and likelihood)
            prior_mean = self.threshold_priors[threshold_name]['mean']
            prior_std = self.threshold_priors[threshold_name]['std']

            # Posterior mean (precision-weighted average)
            precision_prior = 1 / (prior_std ** 2)
            precision_likelihood = 1 / (weighted_std ** 2) if weighted_std > 0 else 1e-6
            posterior_mean = (precision_prior * prior_mean + precision_likelihood * weighted_mean) / \
                           (precision_prior + precision_likelihood)

            # Posterior std
            posterior_std = np.sqrt(1 / (precision_prior + precision_likelihood))

            # Update beliefs
            self.threshold_priors[threshold_name] = {
                'mean': posterior_mean,
                'std': posterior_std
            }

            print(f"Updated {threshold_name}:")
            print(f"  Prior: {prior_mean:.3f} ± {prior_std:.3f}")
            print(f"  Posterior: {posterior_mean:.3f} ± {posterior_std:.3f}")

    def get_adaptive_thresholds(self, exploration_rate: float = 0.1) -> Dict:
        """
        Sample from posterior distributions for current thresholds

        Args:
            exploration_rate: Probability of exploring (sampling) vs exploiting (using mean)

        Returns:
            Dictionary of adaptive thresholds
        """
        thresholds = {}
        for name, params in self.threshold_priors.items():
            # Epsilon-greedy: explore vs exploit
            if np.random.random() < exploration_rate:
                # Explore: sample from posterior
                thresholds[name] = np.random.normal(params['mean'], params['std'])
            else:
                # Exploit: use posterior mean
                thresholds[name] = params['mean']

            # Clip to reasonable ranges
            if name == 'htf_strength':
                thresholds[name] = np.clip(thresholds[name], 0.3, 0.8)
            elif name == 'min_rr':
                thresholds[name] = np.clip(thresholds[name], 1.0, 3.0)
            elif name == 'confidence':
                thresholds[name] = np.clip(thresholds[name], 0.4, 0.9)

        return thresholds

    def record_outcome(self, thresholds: Dict, profit_pips: float):
        """
        Record trade outcome for learning

        Args:
            thresholds: Thresholds used for this trade
            profit_pips: Profit/loss in pips
        """
        self.recent_outcomes.append({
            'thresholds': thresholds.copy(),
            'profit': profit_pips,
            'win': profit_pips > 0
        })

        # Keep only recent history
        if len(self.recent_outcomes) > 200:
            self.recent_outcomes = self.recent_outcomes[-200:]

        # Update beliefs if enough data
        if len(self.recent_outcomes) >= self.update_frequency:
            self.update_posteriors()
