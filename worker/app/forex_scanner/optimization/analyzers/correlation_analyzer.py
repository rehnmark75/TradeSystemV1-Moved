# optimization/analyzers/correlation_analyzer.py
"""
Correlation analyzer for unified parameter optimizer.
Analyzes parameter correlations with trade outcomes.
"""

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class ParameterCorrelation:
    """Result of parameter correlation analysis"""
    parameter: str
    epic: str
    correlation: float
    p_value: float
    optimal_range: Tuple[float, float]
    current_value: Any
    recommended_value: Any
    win_rate_improvement: float
    sample_size: int
    confidence_level: float
    evidence: Dict[str, Any]


@dataclass
class OptimizationRecommendation:
    """Single parameter optimization recommendation"""
    epic: str
    parameter: str
    direction: Optional[str]  # BULL, BEAR, or None
    current_value: Any
    recommended_value: Any
    confidence: float
    expected_impact_pips: float
    sample_size: int
    reason: str
    evidence: Dict[str, Any]


class CorrelationAnalyzer:
    """Analyzes parameter correlations with trade outcomes"""

    # COMPREHENSIVE parameter list for analysis
    # Grouped by category for organized analysis
    OPTIMIZABLE_PARAMS = {
        # Core thresholds (directly configurable)
        'core': [
            'confidence_score',
            'volume_ratio',
            'pullback_depth',
        ],
        # MACD-related (histogram STRENGTH matters, not just direction)
        'macd': [
            'macd_histogram',  # Histogram strength - key insight! (numeric value)
        ],
        # Trend strength indicators
        'trend_strength': [
            'adx_value',
            'efficiency_ratio',
            'kama_er',  # Kaufman efficiency ratio
        ],
        # Volatility measures
        'volatility': [
            'atr_percentile',
            'bb_width_percentile',
            'bb_percent_b',
        ],
        # Momentum/Oscillators
        'momentum': [
            'stoch_k',
            'stoch_d',
        ],
        # Structure
        'structure': [
            'swing_range_pips',
            'last_swing_bars_ago',
            'ema_distance_pips',
        ],
        # Risk/Reward
        'risk_reward': [
            'potential_rr_ratio',
            'potential_risk_pips',
        ],
    }

    # Categorical parameters (analyze win rate by category)
    CATEGORICAL_PARAMS = [
        'fib_level',           # fib zone category
        'volatility_state',    # low/medium/high
        'market_regime_detected',
        'adx_trend_strength',  # weak/moderate/strong
        'stoch_zone',          # oversold/neutral/overbought
        'rsi_zone',            # oversold/neutral/overbought
        'supertrend_direction',
        'price_vs_ema_200',    # above/below
        'kama_trend',          # up/down/sideways
        'macd_aligned',        # True/False
    ]

    # Mapping from analysis parameters to config parameters
    PARAM_TO_CONFIG = {
        'confidence_score': 'min_confidence',
        'volume_ratio': 'min_volume_ratio',
        'pullback_depth': 'fib_pullback_min',
        'macd_histogram': 'macd_min_histogram',  # New: minimum histogram strength
        'adx_value': 'min_adx_value',
        'efficiency_ratio': 'min_efficiency_ratio',
        'atr_percentile': 'atr_percentile_filter',
        'bb_percent_b': 'bb_percent_b_range',
        'stoch_k': 'stoch_threshold',
        'potential_rr_ratio': 'min_rr_ratio',
    }

    def __init__(self, min_sample_size: int = 20, min_confidence: float = 0.70):
        self.logger = logging.getLogger(__name__)
        self.min_sample_size = min_sample_size
        self.min_confidence = min_confidence

    def analyze_all_parameters(
        self,
        trade_df: pd.DataFrame,
        rejection_df: pd.DataFrame,
        current_config: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """
        Analyze all parameters and generate recommendations.

        Args:
            trade_df: Trade outcome data
            rejection_df: Rejection outcome data
            current_config: Current parameter configuration

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Get unique epics
        epics = set()
        if not trade_df.empty:
            epics.update(trade_df['epic'].unique())
        if not rejection_df.empty:
            epics.update(rejection_df['epic'].unique())

        for epic in epics:
            epic_recommendations = self._analyze_epic(
                epic,
                trade_df[trade_df['epic'] == epic] if not trade_df.empty else pd.DataFrame(),
                rejection_df[rejection_df['epic'] == epic] if not rejection_df.empty else pd.DataFrame(),
                current_config.get(epic, {})
            )
            recommendations.extend(epic_recommendations)

        # Sort by confidence and expected impact
        recommendations.sort(key=lambda r: (r.confidence, r.expected_impact_pips), reverse=True)

        return recommendations

    def _analyze_epic(
        self,
        epic: str,
        trade_df: pd.DataFrame,
        rejection_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Analyze parameters for a single epic"""
        recommendations = []

        # 1. Analyze confidence threshold from rejections
        if not rejection_df.empty and 'confidence_score' in rejection_df.columns:
            rec = self._analyze_confidence_threshold(epic, rejection_df, config)
            if rec:
                recommendations.append(rec)

        # 2. Analyze volume ratio from rejections
        if not rejection_df.empty and 'volume_ratio' in rejection_df.columns:
            rec = self._analyze_volume_threshold(epic, rejection_df, config)
            if rec:
                recommendations.append(rec)

        # 3. Analyze pullback depth from TIER3 rejections
        tier3_rejections = rejection_df[
            rejection_df['rejection_stage'].str.contains('TIER3|PULLBACK', case=False, na=False)
        ] if not rejection_df.empty else pd.DataFrame()

        if not tier3_rejections.empty and 'pullback_depth' in tier3_rejections.columns:
            rec = self._analyze_pullback_threshold(epic, tier3_rejections, config)
            if rec:
                recommendations.append(rec)

        # 4. Analyze SL/TP from trade outcomes (MFE/MAE)
        if not trade_df.empty:
            sl_tp_recs = self._analyze_sl_tp(epic, trade_df, config)
            recommendations.extend(sl_tp_recs)

        # 5. Analyze CONFIDENCE_CAP rejections (max_confidence too low)
        if not rejection_df.empty:
            rec = self._analyze_confidence_cap(epic, rejection_df, config)
            if rec:
                recommendations.append(rec)

        # 6. Analyze high win rate rejection stages
        if not rejection_df.empty:
            stage_recs = self._analyze_rejection_stages(epic, rejection_df, config)
            recommendations.extend(stage_recs)

        # 7. COMPREHENSIVE INDICATOR ANALYSIS - the key addition!
        # Analyzes ALL available indicators: MACD strength, ADX, stochastics,
        # Bollinger bands, efficiency ratio, volatility state, etc.
        if not rejection_df.empty:
            comprehensive_recs = self._analyze_comprehensive_indicators(epic, rejection_df, config)
            recommendations.extend(comprehensive_recs)

        # 8. Specific MACD histogram strength analysis
        if not rejection_df.empty:
            macd_rec = self._analyze_macd_strength(epic, rejection_df, config)
            if macd_rec:
                recommendations.append(macd_rec)

        # 9. ADX trend strength analysis
        if not rejection_df.empty:
            adx_rec = self._analyze_adx_strength(epic, rejection_df, config)
            if adx_rec:
                recommendations.append(adx_rec)

        return recommendations

    def _analyze_confidence_threshold(
        self,
        epic: str,
        rejection_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Optional[OptimizationRecommendation]:
        """Analyze if confidence threshold is too strict"""
        # Filter to valid data
        if 'confidence_score' not in rejection_df.columns or 'would_be_winner' not in rejection_df.columns:
            return None

        df = rejection_df[rejection_df['confidence_score'].notna()].copy()

        if len(df) < self.min_sample_size:
            return None

        # Check for sufficient unique values
        if df['confidence_score'].nunique() < 4:
            return None

        current_threshold = config.get('min_confidence', 0.48)

        # Calculate win rate by confidence bins
        try:
            df['conf_bin'] = pd.qcut(df['confidence_score'], q=4, duplicates='drop')
            bin_stats = df.groupby('conf_bin', observed=True).agg(
                count=('epic', 'count'),
                winners=('would_be_winner', 'sum')
            ).reset_index()
            bin_stats['win_rate'] = bin_stats['winners'] / bin_stats['count']
        except (ValueError, KeyError, TypeError):
            return None

        # Find bins with high win rate that are below current threshold
        high_wr_bins = bin_stats[bin_stats['win_rate'] > 0.55]

        if high_wr_bins.empty:
            return None

        # Get the lowest confidence bin that still has good win rate
        min_conf_good = df[df['would_be_winner']]['confidence_score'].quantile(0.25)

        if min_conf_good >= current_threshold:
            return None

        # Calculate expected improvement
        would_be_captured = len(df[
            (df['confidence_score'] >= min_conf_good) &
            (df['confidence_score'] < current_threshold) &
            (df['would_be_winner'])
        ])

        missed_pips = df[
            (df['confidence_score'] >= min_conf_good) &
            (df['confidence_score'] < current_threshold) &
            (df['would_be_winner'])
        ]['potential_profit_pips'].sum() if 'potential_profit_pips' in df.columns else 0

        # Statistical confidence based on sample size
        confidence = min(0.95, 0.5 + (len(df) / 200))

        if confidence < self.min_confidence:
            return None

        return OptimizationRecommendation(
            epic=epic,
            parameter='min_confidence',
            direction=None,
            current_value=current_threshold,
            recommended_value=round(min_conf_good, 3),
            confidence=confidence,
            expected_impact_pips=missed_pips,
            sample_size=len(df),
            reason=f"Rejected signals at {min_conf_good:.2f}-{current_threshold:.2f} confidence have {high_wr_bins['win_rate'].mean():.1%} win rate",
            evidence={
                'bin_stats': bin_stats.to_dict('records'),
                'would_be_captured': would_be_captured,
                'missed_pips': missed_pips
            }
        )

    def _analyze_volume_threshold(
        self,
        epic: str,
        rejection_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Optional[OptimizationRecommendation]:
        """Analyze if volume threshold is too strict"""
        # Filter to volume-related rejections
        df = rejection_df[
            (rejection_df['volume_ratio'].notna()) &
            (rejection_df['rejection_stage'].str.contains('VOLUME', case=False, na=False) |
             (rejection_df['volume_ratio'] < config.get('min_volume_ratio', 0.5)))
        ].copy()

        if len(df) < self.min_sample_size:
            return None

        current_threshold = config.get('min_volume_ratio', 0.50)

        # Check win rate for low-volume rejections
        win_rate = df['would_be_winner'].mean()

        if win_rate < 0.50:  # Not worth relaxing if win rate is poor
            return None

        # Find optimal threshold
        optimal = df[df['would_be_winner']]['volume_ratio'].quantile(0.25)

        if optimal >= current_threshold:
            return None

        missed_pips = df[df['would_be_winner']]['potential_profit_pips'].sum() if 'potential_profit_pips' in df.columns else 0
        confidence = min(0.95, 0.5 + (len(df) / 200))

        if confidence < self.min_confidence:
            return None

        return OptimizationRecommendation(
            epic=epic,
            parameter='min_volume_ratio',
            direction=None,
            current_value=current_threshold,
            recommended_value=round(optimal, 2),
            confidence=confidence,
            expected_impact_pips=missed_pips,
            sample_size=len(df),
            reason=f"Low-volume rejections have {win_rate:.1%} win rate",
            evidence={
                'win_rate': win_rate,
                'sample_size': len(df),
                'missed_pips': missed_pips
            }
        )

    def _analyze_pullback_threshold(
        self,
        epic: str,
        rejection_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Optional[OptimizationRecommendation]:
        """Analyze if pullback/fib threshold is too strict"""
        df = rejection_df[rejection_df['pullback_depth'].notna()].copy()

        if len(df) < self.min_sample_size:
            return None

        current_min = config.get('fib_pullback_min', 0.236)

        # Check win rate for shallow pullback rejections
        shallow_df = df[df['pullback_depth'] < current_min]

        if len(shallow_df) < 10:
            return None

        win_rate = shallow_df['would_be_winner'].mean()

        if win_rate < 0.50:
            return None

        # Recommend lower threshold
        optimal = shallow_df[shallow_df['would_be_winner']]['pullback_depth'].quantile(0.10)
        optimal = max(0.05, optimal)  # Don't go below 0.05

        if optimal >= current_min:
            return None

        missed_pips = shallow_df[shallow_df['would_be_winner']]['potential_profit_pips'].sum() if 'potential_profit_pips' in df.columns else 0
        confidence = min(0.95, 0.5 + (len(shallow_df) / 150))

        if confidence < self.min_confidence:
            return None

        return OptimizationRecommendation(
            epic=epic,
            parameter='fib_pullback_min',
            direction=None,
            current_value=current_min,
            recommended_value=round(optimal, 3),
            confidence=confidence,
            expected_impact_pips=missed_pips,
            sample_size=len(shallow_df),
            reason=f"Shallow pullback rejections ({optimal:.2f}-{current_min:.2f}) have {win_rate:.1%} win rate",
            evidence={
                'win_rate': win_rate,
                'sample_size': len(shallow_df),
                'missed_pips': missed_pips
            }
        )

    def _analyze_sl_tp(
        self,
        epic: str,
        trade_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Analyze optimal SL/TP from trade outcomes"""
        recommendations = []

        if len(trade_df) < self.min_sample_size:
            return recommendations

        current_sl = config.get('fixed_stop_loss_pips', 9)
        current_tp = config.get('fixed_take_profit_pips', 15)

        # Analyze winners - what was the average favorable move?
        winners = trade_df[trade_df['is_winner']]
        losers = trade_df[~trade_df['is_winner']]

        if len(winners) >= 10:
            avg_win_pips = winners['pips_gained'].mean()

            # If average win is significantly higher than TP, might extend TP
            if avg_win_pips > current_tp * 1.3:
                new_tp = round(avg_win_pips * 0.9, 1)  # 90% of average win
                confidence = min(0.90, 0.5 + (len(winners) / 100))

                if confidence >= self.min_confidence:
                    recommendations.append(OptimizationRecommendation(
                        epic=epic,
                        parameter='fixed_take_profit_pips',
                        direction=None,
                        current_value=current_tp,
                        recommended_value=new_tp,
                        confidence=confidence,
                        expected_impact_pips=(new_tp - current_tp) * len(winners) * 0.5,
                        sample_size=len(winners),
                        reason=f"Average win ({avg_win_pips:.1f} pips) exceeds current TP ({current_tp})",
                        evidence={'avg_win_pips': avg_win_pips, 'winners': len(winners)}
                    ))

        if len(losers) >= 10:
            avg_loss_pips = abs(losers['pips_gained'].mean())

            # If average loss is less than SL, might tighten SL
            if avg_loss_pips < current_sl * 0.7:
                new_sl = round(avg_loss_pips * 1.2, 1)  # 120% of average loss
                confidence = min(0.85, 0.5 + (len(losers) / 100))

                if confidence >= self.min_confidence:
                    recommendations.append(OptimizationRecommendation(
                        epic=epic,
                        parameter='fixed_stop_loss_pips',
                        direction=None,
                        current_value=current_sl,
                        recommended_value=new_sl,
                        confidence=confidence,
                        expected_impact_pips=(current_sl - new_sl) * len(losers) * 0.3,
                        sample_size=len(losers),
                        reason=f"Average loss ({avg_loss_pips:.1f} pips) less than current SL ({current_sl})",
                        evidence={'avg_loss_pips': avg_loss_pips, 'losers': len(losers)}
                    ))

        return recommendations

    def _analyze_confidence_cap(
        self,
        epic: str,
        rejection_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Optional[OptimizationRecommendation]:
        """Analyze if max_confidence cap is too restrictive (rejecting good high-confidence signals)"""
        # Filter to CONFIDENCE_CAP rejections
        cap_rejections = rejection_df[
            rejection_df['rejection_stage'].str.contains('CONFIDENCE_CAP', case=False, na=False)
        ]

        if len(cap_rejections) < 5:  # Need at least 5 samples
            return None

        win_rate = cap_rejections['would_be_winner'].mean()

        # If win rate > 60%, the max_confidence cap is too restrictive
        if win_rate < 0.60:
            return None

        current_max = config.get('max_confidence', 0.75)

        # Calculate missed opportunity
        winners = cap_rejections[cap_rejections['would_be_winner']]
        missed_pips = winners['potential_profit_pips'].sum() if 'potential_profit_pips' in winners.columns else len(winners) * 10

        # Recommend raising or removing max_confidence
        confidence = min(0.90, 0.5 + (len(cap_rejections) / 50))

        if confidence < self.min_confidence:
            return None

        return OptimizationRecommendation(
            epic=epic,
            parameter='max_confidence',
            direction=None,
            current_value=current_max,
            recommended_value=0.95,  # Raise to effectively disable
            confidence=confidence,
            expected_impact_pips=missed_pips,
            sample_size=len(cap_rejections),
            reason=f"CONFIDENCE_CAP rejections have {win_rate:.0%} win rate - cap is too restrictive",
            evidence={
                'win_rate': win_rate,
                'winners': len(winners),
                'missed_pips': missed_pips
            }
        )

    def _analyze_rejection_stages(
        self,
        epic: str,
        rejection_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Analyze rejection stages with high win rates - filters may be too strict"""
        recommendations = []

        # Group by rejection stage
        stage_stats = rejection_df.groupby('rejection_stage').agg(
            count=('epic', 'count'),
            winners=('would_be_winner', 'sum'),
            missed_pips=('potential_profit_pips', lambda x: x[rejection_df.loc[x.index, 'would_be_winner']].sum() if 'potential_profit_pips' in rejection_df.columns else 0)
        ).reset_index()

        stage_stats['win_rate'] = stage_stats['winners'] / stage_stats['count']

        # Find stages with high win rate and enough samples
        high_wr_stages = stage_stats[
            (stage_stats['win_rate'] >= 0.55) &
            (stage_stats['count'] >= 15)
        ]

        for _, row in high_wr_stages.iterrows():
            stage = row['rejection_stage']
            win_rate = row['win_rate']
            count = int(row['count'])
            missed = row['missed_pips']

            # Skip stages we already analyze specifically
            if any(s in stage.upper() for s in ['CONFIDENCE', 'VOLUME', 'PULLBACK', 'TIER3']):
                continue

            confidence = min(0.90, 0.5 + (count / 100))

            if confidence < self.min_confidence:
                continue

            # Determine which filter corresponds to this stage and provide specific recommendations
            if 'TIER2' in stage.upper() or 'SWING' in stage.upper():
                # TIER2/Swing rejections - recommend relaxing swing validation
                param = 'min_swing_atr_multiplier'
                current_val = config.get('min_swing_atr_multiplier', 0.25)
                # Recommend reducing by 40% (e.g., 0.25 -> 0.15)
                recommended_val = round(current_val * 0.6, 2)
                reason = f"Swing filter too strict: {win_rate:.0%} of {count} rejected signals would win. Reduce min_swing_atr_multiplier"

            elif 'MACD' in stage.upper():
                param = 'macd_filter_enabled'
                current_val = config.get('macd_filter_enabled', True)
                recommended_val = False
                reason = f"MACD filter rejecting {win_rate:.0%} winners ({count} samples) - consider disabling"

            elif 'SMC' in stage.upper():
                param = 'smc_conflict_tolerance'
                current_val = config.get('smc_conflict_tolerance', 0)
                recommended_val = 1  # Allow 1 conflict
                reason = f"SMC conflict filter rejecting {win_rate:.0%} winners ({count} samples) - allow minor conflicts"

            else:
                param = f'filter_{stage.lower()}'
                current_val = 'enabled'
                recommended_val = 'relaxed'
                reason = f"{stage} filter rejecting {win_rate:.0%} winners ({count} samples)"

            recommendations.append(OptimizationRecommendation(
                epic=epic,
                parameter=param,
                direction=None,
                current_value=current_val,
                recommended_value=recommended_val,
                confidence=confidence,
                expected_impact_pips=missed if missed > 0 else count * 8,  # Estimate 8 pips per missed trade
                sample_size=count,
                reason=reason,
                evidence={
                    'stage': stage,
                    'win_rate': win_rate,
                    'count': count,
                    'missed_pips': missed
                }
            ))

        return recommendations

    def _analyze_comprehensive_indicators(
        self,
        epic: str,
        rejection_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """
        Comprehensive indicator analysis - analyzes ALL available indicators.

        This is the KEY method that provides deep insights by analyzing:
        - MACD histogram STRENGTH (not just direction)
        - ADX trend strength values
        - Stochastic levels
        - Bollinger band positions
        - Efficiency ratios
        - And more...
        """
        recommendations = []

        if rejection_df.empty:
            return recommendations

        # Analyze each category of parameters
        for category, params in self.OPTIMIZABLE_PARAMS.items():
            for param in params:
                if param not in rejection_df.columns:
                    continue

                rec = self._analyze_numeric_param_bins(epic, rejection_df, param, config)
                if rec:
                    recommendations.append(rec)

        # Analyze categorical parameters
        for param in self.CATEGORICAL_PARAMS:
            if param not in rejection_df.columns:
                continue

            rec = self._analyze_categorical_param(epic, rejection_df, param, config)
            if rec:
                recommendations.append(rec)

        return recommendations

    def _analyze_numeric_param_bins(
        self,
        epic: str,
        df: pd.DataFrame,
        param: str,
        config: Dict[str, Any]
    ) -> Optional[OptimizationRecommendation]:
        """
        Analyze a numeric parameter by binning into quartiles and comparing win rates.

        This finds optimal ranges for parameters like:
        - MACD histogram strength
        - ADX values
        - Volume ratios
        - etc.
        """
        # Filter to valid values
        df_valid = df[df[param].notna() & (df[param] != 0)].copy()

        if len(df_valid) < self.min_sample_size:
            return None

        try:
            # Create quartile bins
            df_valid['bin'] = pd.qcut(df_valid[param], q=4, duplicates='drop')
        except ValueError:
            # Not enough unique values
            try:
                df_valid['bin'] = pd.cut(df_valid[param], bins=4)
            except ValueError:
                return None

        # Calculate win rate per bin
        bin_stats = df_valid.groupby('bin', observed=True).agg(
            count=('epic', 'count'),
            winners=('would_be_winner', 'sum'),
            avg_value=(param, 'mean'),
            missed_pips=('potential_profit_pips', lambda x: x[df_valid.loc[x.index, 'would_be_winner']].sum() if 'potential_profit_pips' in df_valid.columns else 0)
        ).reset_index()

        bin_stats['win_rate'] = bin_stats['winners'] / bin_stats['count']

        # Find best and worst performing bins
        if len(bin_stats) < 2:
            return None

        best_bin = bin_stats.loc[bin_stats['win_rate'].idxmax()]
        worst_bin = bin_stats.loc[bin_stats['win_rate'].idxmin()]

        # Only recommend if there's significant difference
        win_rate_diff = best_bin['win_rate'] - worst_bin['win_rate']
        if win_rate_diff < 0.15:  # At least 15% difference
            return None

        # Determine recommendation based on current rejection pattern
        # If worst bin has high win rate (>55%), the filter is too strict
        if worst_bin['win_rate'] > 0.55 and worst_bin['count'] >= 10:
            # Filter is rejecting too many winners - recommend relaxing
            config_param = self.PARAM_TO_CONFIG.get(param, param)
            current_val = config.get(config_param, worst_bin['avg_value'])

            # Recommend value from worst bin (to capture those rejected winners)
            recommended_val = round(float(worst_bin['avg_value']), 3)

            confidence = min(0.90, 0.5 + (worst_bin['count'] / 50) + (win_rate_diff / 2))

            if confidence < self.min_confidence:
                return None

            return OptimizationRecommendation(
                epic=epic,
                parameter=config_param,
                direction=None,
                current_value=current_val,
                recommended_value=recommended_val,
                confidence=confidence,
                expected_impact_pips=float(worst_bin['missed_pips']) if worst_bin['missed_pips'] > 0 else float(worst_bin['count']) * 8,
                sample_size=int(worst_bin['count']),
                reason=f"{param}: Rejected signals in range [{worst_bin['bin']}] have {worst_bin['win_rate']:.0%} win rate vs best {best_bin['win_rate']:.0%}",
                evidence={
                    'param': param,
                    'worst_bin_win_rate': float(worst_bin['win_rate']),
                    'best_bin_win_rate': float(best_bin['win_rate']),
                    'worst_bin_avg': float(worst_bin['avg_value']),
                    'best_bin_avg': float(best_bin['avg_value']),
                    'worst_bin_count': int(worst_bin['count']),
                    'all_bins': bin_stats.to_dict('records')
                }
            )

        return None

    def _analyze_categorical_param(
        self,
        epic: str,
        df: pd.DataFrame,
        param: str,
        config: Dict[str, Any]
    ) -> Optional[OptimizationRecommendation]:
        """
        Analyze categorical parameters to find which categories perform best/worst.

        Examples:
        - volatility_state: 'low', 'medium', 'high'
        - rsi_zone: 'oversold', 'neutral', 'overbought'
        - market_regime_detected: 'trending', 'ranging', 'breakout'
        """
        df_valid = df[df[param].notna()].copy()

        if len(df_valid) < self.min_sample_size:
            return None

        # Calculate win rate per category
        cat_stats = df_valid.groupby(param).agg(
            count=('epic', 'count'),
            winners=('would_be_winner', 'sum'),
            missed_pips=('potential_profit_pips', lambda x: x[df_valid.loc[x.index, 'would_be_winner']].sum() if 'potential_profit_pips' in df_valid.columns else 0)
        ).reset_index()

        cat_stats['win_rate'] = cat_stats['winners'] / cat_stats['count']

        # Filter to categories with enough samples
        cat_stats = cat_stats[cat_stats['count'] >= 10]

        if len(cat_stats) < 2:
            return None

        # Find best and worst categories
        best_cat = cat_stats.loc[cat_stats['win_rate'].idxmax()]
        worst_cat = cat_stats.loc[cat_stats['win_rate'].idxmin()]

        win_rate_diff = best_cat['win_rate'] - worst_cat['win_rate']

        # Only report if significant difference AND worst category has high win rate
        if win_rate_diff < 0.20:
            return None

        # If worst category has >55% win rate, we're rejecting winners
        if worst_cat['win_rate'] > 0.55:
            confidence = min(0.90, 0.5 + (worst_cat['count'] / 50) + (win_rate_diff / 2))

            if confidence < self.min_confidence:
                return None

            return OptimizationRecommendation(
                epic=epic,
                parameter=f"{param}_filter",
                direction=None,
                current_value=f"blocking {worst_cat[param]}",
                recommended_value=f"allow {worst_cat[param]}",
                confidence=confidence,
                expected_impact_pips=float(worst_cat['missed_pips']) if worst_cat['missed_pips'] > 0 else float(worst_cat['count']) * 8,
                sample_size=int(worst_cat['count']),
                reason=f"{param}={worst_cat[param]} rejections have {worst_cat['win_rate']:.0%} win rate vs best {best_cat[param]}={best_cat['win_rate']:.0%}",
                evidence={
                    'param': param,
                    'worst_category': str(worst_cat[param]),
                    'worst_win_rate': float(worst_cat['win_rate']),
                    'best_category': str(best_cat[param]),
                    'best_win_rate': float(best_cat['win_rate']),
                    'all_categories': cat_stats.to_dict('records')
                }
            )

        return None

    def _analyze_macd_strength(
        self,
        epic: str,
        rejection_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Optional[OptimizationRecommendation]:
        """
        Specific analysis for MACD histogram STRENGTH.

        Key insight: It's not just about positive/negative MACD,
        but HOW STRONG the momentum is. Weak histogram readings
        may incorrectly reject good trades.
        """
        if 'macd_histogram' not in rejection_df.columns:
            return None

        df = rejection_df[rejection_df['macd_histogram'].notna()].copy()

        if len(df) < self.min_sample_size:
            return None

        # Analyze by absolute MACD strength
        df['macd_abs'] = df['macd_histogram'].abs()

        # Split into weak vs strong momentum
        median_strength = df['macd_abs'].median()
        df['momentum_strength'] = df['macd_abs'].apply(
            lambda x: 'weak' if x < median_strength else 'strong'
        )

        strength_stats = df.groupby('momentum_strength').agg(
            count=('epic', 'count'),
            winners=('would_be_winner', 'sum'),
            avg_momentum=('macd_abs', 'mean'),
            missed_pips=('potential_profit_pips', 'sum')
        ).reset_index()

        strength_stats['win_rate'] = strength_stats['winners'] / strength_stats['count']

        if len(strength_stats) < 2:
            return None

        weak_stats = strength_stats[strength_stats['momentum_strength'] == 'weak']
        strong_stats = strength_stats[strength_stats['momentum_strength'] == 'strong']

        if weak_stats.empty or strong_stats.empty:
            return None

        weak_row = weak_stats.iloc[0]
        strong_row = strong_stats.iloc[0]

        # If weak momentum has HIGH win rate, MACD filter is too strict
        if weak_row['win_rate'] > 0.55 and weak_row['count'] >= 10:
            win_diff = strong_row['win_rate'] - weak_row['win_rate']

            # If weak momentum performs BETTER than strong, something is wrong
            if weak_row['win_rate'] >= strong_row['win_rate']:
                reason = f"MACD filter paradox: weak momentum ({weak_row['win_rate']:.0%} WR) outperforms strong ({strong_row['win_rate']:.0%} WR)"
            else:
                reason = f"Weak MACD momentum rejections have {weak_row['win_rate']:.0%} win rate - consider relaxing threshold"

            confidence = min(0.90, 0.5 + (weak_row['count'] / 50))

            if confidence < self.min_confidence:
                return None

            return OptimizationRecommendation(
                epic=epic,
                parameter='macd_min_histogram_strength',
                direction=None,
                current_value=float(median_strength),
                recommended_value=float(weak_row['avg_momentum']),
                confidence=confidence,
                expected_impact_pips=float(weak_row['missed_pips']) if weak_row['missed_pips'] > 0 else float(weak_row['count']) * 8,
                sample_size=int(weak_row['count']),
                reason=reason,
                evidence={
                    'weak_win_rate': float(weak_row['win_rate']),
                    'strong_win_rate': float(strong_row['win_rate']),
                    'weak_avg_momentum': float(weak_row['avg_momentum']),
                    'strong_avg_momentum': float(strong_row['avg_momentum']),
                    'median_strength': float(median_strength),
                    'weak_count': int(weak_row['count']),
                    'strong_count': int(strong_row['count'])
                }
            )

        return None

    def _analyze_adx_strength(
        self,
        epic: str,
        rejection_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Optional[OptimizationRecommendation]:
        """
        Analyze ADX values to find optimal trend strength thresholds.

        ADX < 20: weak/no trend
        ADX 20-25: developing trend
        ADX 25-50: strong trend
        ADX > 50: very strong trend (potentially overextended)
        """
        if 'adx_value' not in rejection_df.columns:
            return None

        df = rejection_df[rejection_df['adx_value'].notna()].copy()

        if len(df) < self.min_sample_size:
            return None

        # Create ADX buckets
        df['adx_bucket'] = pd.cut(
            df['adx_value'],
            bins=[0, 20, 25, 40, 100],
            labels=['weak_0_20', 'developing_20_25', 'strong_25_40', 'very_strong_40+']
        )

        bucket_stats = df.groupby('adx_bucket', observed=True).agg(
            count=('epic', 'count'),
            winners=('would_be_winner', 'sum'),
            avg_adx=('adx_value', 'mean'),
            missed_pips=('potential_profit_pips', 'sum')
        ).reset_index()

        bucket_stats['win_rate'] = bucket_stats['winners'] / bucket_stats['count']

        # Filter to buckets with enough samples
        bucket_stats = bucket_stats[bucket_stats['count'] >= 10]

        if len(bucket_stats) < 2:
            return None

        # Find best performing ADX range
        best_bucket = bucket_stats.loc[bucket_stats['win_rate'].idxmax()]

        # Check if we're over-filtering certain ADX ranges
        for _, row in bucket_stats.iterrows():
            if row['win_rate'] > 0.55 and row['adx_bucket'] != best_bucket['adx_bucket']:
                confidence = min(0.85, 0.5 + (row['count'] / 50))

                if confidence < self.min_confidence:
                    continue

                return OptimizationRecommendation(
                    epic=epic,
                    parameter='adx_threshold',
                    direction=None,
                    current_value=f"current ADX filter",
                    recommended_value=f"include {row['adx_bucket']} range",
                    confidence=confidence,
                    expected_impact_pips=float(row['missed_pips']) if row['missed_pips'] > 0 else float(row['count']) * 8,
                    sample_size=int(row['count']),
                    reason=f"ADX {row['adx_bucket']} rejections have {row['win_rate']:.0%} win rate ({int(row['count'])} samples)",
                    evidence={
                        'bucket': str(row['adx_bucket']),
                        'win_rate': float(row['win_rate']),
                        'avg_adx': float(row['avg_adx']),
                        'count': int(row['count']),
                        'all_buckets': bucket_stats.to_dict('records')
                    }
                )

        return None
