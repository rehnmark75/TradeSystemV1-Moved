# optimization/analyzers/direction_analyzer.py
"""
Direction analyzer for unified parameter optimizer.
Analyzes BULL vs BEAR performance separately to identify direction-specific optimizations.
"""

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class DirectionPerformance:
    """Performance metrics for a specific direction"""
    epic: str
    direction: str  # BULL or BEAR
    total_trades: int
    winners: int
    win_rate: float
    total_pips: float
    avg_pips: float
    profit_factor: float
    avg_confidence: float
    avg_volume_ratio: float


@dataclass
class DirectionRecommendation:
    """Recommendation for direction-specific parameter adjustment"""
    epic: str
    direction: str  # BULL or BEAR
    parameter: str
    current_value: Any
    recommended_value: Any
    confidence: float
    reason: str
    win_rate_impact: float
    sample_size: int


class DirectionAnalyzer:
    """Analyzes direction-specific performance and generates recommendations"""

    # Threshold for significant win rate difference
    SIGNIFICANT_DIFFERENCE = 0.15  # 15% difference

    def __init__(self, min_sample_size: int = 20, min_confidence: float = 0.70):
        self.logger = logging.getLogger(__name__)
        self.min_sample_size = min_sample_size
        self.min_confidence = min_confidence

    def analyze_direction_performance(
        self,
        trade_df: pd.DataFrame,
        rejection_df: pd.DataFrame
    ) -> Dict[str, Dict[str, DirectionPerformance]]:
        """
        Analyze performance by direction for each epic.

        Args:
            trade_df: Trade outcome data with 'direction' column
            rejection_df: Rejection data with 'attempted_direction' column

        Returns:
            Dict[epic, Dict[direction, DirectionPerformance]]
        """
        results = {}

        # Combine data sources
        if not trade_df.empty and 'direction' in trade_df.columns:
            for epic in trade_df['epic'].unique():
                epic_df = trade_df[trade_df['epic'] == epic]

                if epic not in results:
                    results[epic] = {}

                for direction in ['BULL', 'BEAR']:
                    dir_df = epic_df[epic_df['direction'] == direction]

                    if len(dir_df) >= self.min_sample_size:
                        results[epic][direction] = self._calculate_performance(
                            epic, direction, dir_df
                        )

        return results

    def _calculate_performance(
        self,
        epic: str,
        direction: str,
        df: pd.DataFrame
    ) -> DirectionPerformance:
        """Calculate performance metrics for a direction"""
        total = len(df)
        winners = df['is_winner'].sum() if 'is_winner' in df.columns else 0
        win_rate = winners / total if total > 0 else 0

        total_pips = df['pips_gained'].sum() if 'pips_gained' in df.columns else 0
        avg_pips = df['pips_gained'].mean() if 'pips_gained' in df.columns else 0

        # Calculate profit factor
        if 'pips_gained' in df.columns:
            gross_profit = df[df['pips_gained'] > 0]['pips_gained'].sum()
            gross_loss = abs(df[df['pips_gained'] < 0]['pips_gained'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            profit_factor = 0

        avg_confidence = df['confidence_score'].mean() if 'confidence_score' in df.columns else 0
        avg_volume_ratio = df['volume_ratio'].mean() if 'volume_ratio' in df.columns else 0

        return DirectionPerformance(
            epic=epic,
            direction=direction,
            total_trades=total,
            winners=int(winners),
            win_rate=win_rate,
            total_pips=total_pips,
            avg_pips=avg_pips,
            profit_factor=profit_factor,
            avg_confidence=avg_confidence if not pd.isna(avg_confidence) else 0,
            avg_volume_ratio=avg_volume_ratio if not pd.isna(avg_volume_ratio) else 0
        )

    def generate_direction_recommendations(
        self,
        performance: Dict[str, Dict[str, DirectionPerformance]],
        rejection_df: pd.DataFrame,
        current_config: Dict[str, Any]
    ) -> List[DirectionRecommendation]:
        """
        Generate direction-specific parameter recommendations.

        Args:
            performance: Direction performance by epic
            rejection_df: Rejection data for parameter optimization
            current_config: Current per-epic configuration

        Returns:
            List of direction-specific recommendations
        """
        recommendations = []

        for epic, dir_perf in performance.items():
            if 'BULL' not in dir_perf or 'BEAR' not in dir_perf:
                continue

            bull_perf = dir_perf['BULL']
            bear_perf = dir_perf['BEAR']

            # Check for significant win rate difference
            wr_diff = abs(bull_perf.win_rate - bear_perf.win_rate)

            if wr_diff >= self.SIGNIFICANT_DIFFERENCE:
                # Recommend enabling direction overrides
                recommendations.append(DirectionRecommendation(
                    epic=epic,
                    direction='BOTH',
                    parameter='direction_overrides_enabled',
                    current_value=current_config.get(epic, {}).get('direction_overrides_enabled', False),
                    recommended_value=True,
                    confidence=min(0.95, 0.6 + wr_diff),
                    reason=f"BULL: {bull_perf.win_rate:.1%} vs BEAR: {bear_perf.win_rate:.1%} ({wr_diff:.1%} difference)",
                    win_rate_impact=wr_diff,
                    sample_size=bull_perf.total_trades + bear_perf.total_trades
                ))

                # Generate specific recommendations for weaker direction
                weaker_dir = 'BULL' if bull_perf.win_rate < bear_perf.win_rate else 'BEAR'
                stronger_dir = 'BEAR' if weaker_dir == 'BULL' else 'BULL'
                weaker_perf = bull_perf if weaker_dir == 'BULL' else bear_perf
                stronger_perf = bear_perf if weaker_dir == 'BULL' else bull_perf

                # Analyze rejections for the weaker direction
                dir_recs = self._analyze_direction_rejections(
                    epic, weaker_dir, rejection_df, current_config.get(epic, {}),
                    weaker_perf, stronger_perf
                )
                recommendations.extend(dir_recs)

        return recommendations

    def _analyze_direction_rejections(
        self,
        epic: str,
        direction: str,
        rejection_df: pd.DataFrame,
        config: Dict[str, Any],
        weaker_perf: DirectionPerformance,
        stronger_perf: DirectionPerformance
    ) -> List[DirectionRecommendation]:
        """Analyze rejections to generate direction-specific recommendations"""
        recommendations = []

        if rejection_df.empty:
            return recommendations

        # Filter rejections for this epic and direction
        dir_rejections = rejection_df[
            (rejection_df['epic'] == epic) &
            (rejection_df['attempted_direction'] == direction)
        ].copy()

        if len(dir_rejections) < self.min_sample_size:
            return recommendations

        # 1. Recommend tightening confidence for weaker direction
        if 'confidence_score' in dir_rejections.columns:
            winner_rejections = dir_rejections[dir_rejections['would_be_winner'] == True]
            loser_rejections = dir_rejections[dir_rejections['would_be_winner'] == False]

            if len(winner_rejections) >= 10 and len(loser_rejections) >= 10:
                avg_conf_winners = winner_rejections['confidence_score'].mean()
                avg_conf_losers = loser_rejections['confidence_score'].mean()

                if avg_conf_winners > avg_conf_losers:
                    # Higher confidence = more winners
                    current_min = config.get(f'min_confidence_{direction.lower()}',
                                            config.get('min_confidence', 0.48))
                    recommended = round(avg_conf_winners * 0.95, 3)

                    if recommended > current_min:
                        recommendations.append(DirectionRecommendation(
                            epic=epic,
                            direction=direction,
                            parameter=f'min_confidence_{direction.lower()}',
                            current_value=current_min,
                            recommended_value=recommended,
                            confidence=min(0.90, 0.5 + len(dir_rejections) / 200),
                            reason=f"Winners avg conf: {avg_conf_winners:.2f} vs losers: {avg_conf_losers:.2f}",
                            win_rate_impact=stronger_perf.win_rate - weaker_perf.win_rate,
                            sample_size=len(dir_rejections)
                        ))

        # 2. Recommend adjusting volume ratio for weaker direction
        if 'volume_ratio' in dir_rejections.columns:
            winner_vol = dir_rejections[dir_rejections['would_be_winner']]['volume_ratio'].mean()
            loser_vol = dir_rejections[~dir_rejections['would_be_winner']]['volume_ratio'].mean()

            if not pd.isna(winner_vol) and not pd.isna(loser_vol) and winner_vol > loser_vol * 1.2:
                current_vol = config.get(f'min_volume_ratio_{direction.lower()}',
                                        config.get('min_volume_ratio', 0.50))
                recommended = round(winner_vol * 0.85, 2)

                if recommended > current_vol:
                    recommendations.append(DirectionRecommendation(
                        epic=epic,
                        direction=direction,
                        parameter=f'min_volume_ratio_{direction.lower()}',
                        current_value=current_vol,
                        recommended_value=recommended,
                        confidence=min(0.85, 0.5 + len(dir_rejections) / 200),
                        reason=f"Winners avg volume: {winner_vol:.2f} vs losers: {loser_vol:.2f}",
                        win_rate_impact=stronger_perf.win_rate - weaker_perf.win_rate,
                        sample_size=len(dir_rejections)
                    ))

        # 3. Recommend adjusting pullback depth for weaker direction
        if 'pullback_depth' in dir_rejections.columns:
            winner_pb = dir_rejections[dir_rejections['would_be_winner']]['pullback_depth'].mean()
            loser_pb = dir_rejections[~dir_rejections['would_be_winner']]['pullback_depth'].mean()

            if not pd.isna(winner_pb) and not pd.isna(loser_pb):
                current_min = config.get(f'fib_pullback_min_{direction.lower()}',
                                        config.get('fib_pullback_min', 0.236))
                current_max = config.get(f'fib_pullback_max_{direction.lower()}',
                                        config.get('fib_pullback_max', 0.786))

                # Recommend tighter pullback range for weaker direction
                if winner_pb > current_min * 1.1:
                    recommended_min = round(winner_pb * 0.90, 3)
                    recommendations.append(DirectionRecommendation(
                        epic=epic,
                        direction=direction,
                        parameter=f'fib_pullback_min_{direction.lower()}',
                        current_value=current_min,
                        recommended_value=max(0.10, recommended_min),
                        confidence=min(0.80, 0.5 + len(dir_rejections) / 200),
                        reason=f"Winners avg pullback: {winner_pb:.3f} vs losers: {loser_pb:.3f}",
                        win_rate_impact=stronger_perf.win_rate - weaker_perf.win_rate,
                        sample_size=len(dir_rejections)
                    ))

        return recommendations

    def get_direction_summary(
        self,
        performance: Dict[str, Dict[str, DirectionPerformance]]
    ) -> pd.DataFrame:
        """
        Get summary table of direction performance by epic.

        Returns DataFrame with columns:
            epic, bull_wr, bull_trades, bear_wr, bear_trades, wr_diff, recommended_action
        """
        rows = []

        for epic, dir_perf in performance.items():
            bull = dir_perf.get('BULL')
            bear = dir_perf.get('BEAR')

            row = {
                'epic': epic,
                'bull_wr': bull.win_rate if bull else None,
                'bull_trades': bull.total_trades if bull else 0,
                'bull_pips': bull.total_pips if bull else 0,
                'bear_wr': bear.win_rate if bear else None,
                'bear_trades': bear.total_trades if bear else 0,
                'bear_pips': bear.total_pips if bear else 0,
            }

            # Calculate difference and recommendation
            if bull and bear:
                row['wr_diff'] = bull.win_rate - bear.win_rate
                if abs(row['wr_diff']) >= self.SIGNIFICANT_DIFFERENCE:
                    if row['wr_diff'] > 0:
                        row['recommended_action'] = 'Tighten BEAR filters'
                    else:
                        row['recommended_action'] = 'Tighten BULL filters'
                else:
                    row['recommended_action'] = 'No direction-specific changes'
            else:
                row['wr_diff'] = None
                row['recommended_action'] = 'Insufficient data'

            rows.append(row)

        return pd.DataFrame(rows)
