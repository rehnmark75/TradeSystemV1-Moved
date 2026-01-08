# optimization/analyzers/regime_analyzer.py
"""
Regime analyzer for unified parameter optimizer.
Analyzes market regime correlation with trade outcomes and generates filter recommendations.
"""

import logging
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class RegimePerformance:
    """Performance metrics for a specific market regime"""
    epic: str
    regime: str
    total_trades: int
    winners: int
    win_rate: float
    total_pips: float
    avg_pips: float
    profit_factor: float


@dataclass
class RegimeFilterRecommendation:
    """Recommendation to filter (block) signals in specific regimes"""
    epic: str
    block_regimes: List[str]
    reason: str
    confidence: float
    estimated_impact_pips: float
    sample_size: int
    regime_stats: Dict[str, Dict[str, Any]]


class RegimeAnalyzer:
    """Analyzes market regime correlation with trade outcomes"""

    # Known market regimes
    REGIMES = ['trending', 'ranging', 'breakout', 'reversal', 'high_volatility', 'low_volatility']

    # Threshold for blocking a regime
    MIN_WIN_RATE_THRESHOLD = 0.40  # Block regimes below 40% win rate
    MIN_REGIME_TRADES = 10  # Minimum trades in a regime to consider blocking

    def __init__(self, min_sample_size: int = 20, min_confidence: float = 0.70):
        self.logger = logging.getLogger(__name__)
        self.min_sample_size = min_sample_size
        self.min_confidence = min_confidence

    def analyze_regime_performance(
        self,
        trade_df: pd.DataFrame,
        market_intel_df: pd.DataFrame,
        time_tolerance_minutes: int = 15
    ) -> Dict[str, Dict[str, RegimePerformance]]:
        """
        Analyze trade performance by market regime.

        Args:
            trade_df: Trade data with entry_timestamp
            market_intel_df: Market intelligence snapshots
            time_tolerance_minutes: Max time difference for regime matching

        Returns:
            Dict[epic, Dict[regime, RegimePerformance]]
        """
        results = {}

        if trade_df.empty:
            self.logger.warning("No trade data for regime analysis")
            return results

        # Enrich trade data with regime at entry
        enriched_df = self._enrich_with_regime(trade_df, market_intel_df, time_tolerance_minutes)

        if 'regime_at_entry' not in enriched_df.columns:
            self.logger.warning("Could not determine regime at entry for trades")
            return results

        # Analyze by epic and regime
        for epic in enriched_df['epic'].unique():
            epic_df = enriched_df[enriched_df['epic'] == epic]
            results[epic] = {}

            for regime in epic_df['regime_at_entry'].dropna().unique():
                regime_df = epic_df[epic_df['regime_at_entry'] == regime]

                if len(regime_df) >= self.MIN_REGIME_TRADES:
                    results[epic][regime] = self._calculate_regime_performance(
                        epic, regime, regime_df
                    )

        return results

    def _enrich_with_regime(
        self,
        trade_df: pd.DataFrame,
        market_intel_df: pd.DataFrame,
        time_tolerance_minutes: int
    ) -> pd.DataFrame:
        """Enrich trade data with market regime at entry time"""
        if market_intel_df.empty:
            return trade_df

        trade_df = trade_df.copy()
        market_intel_df = market_intel_df.copy()

        # Convert timestamps
        if 'entry_timestamp' in trade_df.columns:
            trade_df['entry_timestamp'] = pd.to_datetime(trade_df['entry_timestamp'])
        if 'scan_timestamp' in market_intel_df.columns:
            market_intel_df['scan_timestamp'] = pd.to_datetime(market_intel_df['scan_timestamp'])

        def find_regime(entry_ts):
            if pd.isna(entry_ts) or market_intel_df.empty:
                return None

            time_diffs = abs(market_intel_df['scan_timestamp'] - entry_ts)
            min_idx = time_diffs.idxmin()

            if time_diffs[min_idx].total_seconds() > time_tolerance_minutes * 60:
                return None

            return market_intel_df.loc[min_idx, 'dominant_regime']

        if 'entry_timestamp' in trade_df.columns:
            trade_df['regime_at_entry'] = trade_df['entry_timestamp'].apply(find_regime)

        return trade_df

    def _calculate_regime_performance(
        self,
        epic: str,
        regime: str,
        df: pd.DataFrame
    ) -> RegimePerformance:
        """Calculate performance metrics for a regime"""
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

        return RegimePerformance(
            epic=epic,
            regime=regime,
            total_trades=total,
            winners=int(winners),
            win_rate=win_rate,
            total_pips=total_pips,
            avg_pips=avg_pips,
            profit_factor=profit_factor
        )

    def generate_regime_filter_recommendations(
        self,
        regime_performance: Dict[str, Dict[str, RegimePerformance]],
        current_config: Dict[str, Any]
    ) -> List[RegimeFilterRecommendation]:
        """
        Generate recommendations to block signals in underperforming regimes.

        Args:
            regime_performance: Performance by epic and regime
            current_config: Current configuration (for checking existing filters)

        Returns:
            List of regime filter recommendations
        """
        recommendations = []

        for epic, regimes in regime_performance.items():
            if not regimes:
                continue

            # Calculate overall performance for comparison
            total_trades = sum(r.total_trades for r in regimes.values())
            total_winners = sum(r.winners for r in regimes.values())
            overall_wr = total_winners / total_trades if total_trades > 0 else 0

            # Find regimes to block
            block_regimes = []
            regime_stats = {}

            for regime, perf in regimes.items():
                regime_stats[regime] = {
                    'win_rate': perf.win_rate,
                    'trades': perf.total_trades,
                    'pips': perf.total_pips,
                    'profit_factor': perf.profit_factor
                }

                # Block if significantly below average and below threshold
                if (perf.win_rate < self.MIN_WIN_RATE_THRESHOLD and
                    perf.win_rate < overall_wr - 0.10 and
                    perf.total_trades >= self.MIN_REGIME_TRADES):
                    block_regimes.append(regime)

            if block_regimes:
                # Calculate estimated impact
                blocked_trades = sum(regimes[r].total_trades for r in block_regimes)
                blocked_losses = sum(
                    regimes[r].total_trades - regimes[r].winners
                    for r in block_regimes
                )
                avg_loss_pips = sum(
                    abs(regimes[r].total_pips) for r in block_regimes
                    if regimes[r].total_pips < 0
                )

                # Find best regime for comparison
                best_regime = max(regimes.items(), key=lambda x: x[1].win_rate)
                best_wr = best_regime[1].win_rate

                # Build reason string
                reason_parts = []
                for regime in block_regimes:
                    perf = regimes[regime]
                    reason_parts.append(
                        f"{regime.upper()}: {perf.win_rate:.0%} WR ({perf.total_trades} trades)"
                    )
                reason_parts.append(f"vs {best_regime[0].upper()}: {best_wr:.0%} WR")

                # Calculate confidence based on sample size
                confidence = min(0.95, 0.5 + blocked_trades / 100)

                if confidence >= self.min_confidence:
                    recommendations.append(RegimeFilterRecommendation(
                        epic=epic,
                        block_regimes=block_regimes,
                        reason=' | '.join(reason_parts),
                        confidence=confidence,
                        estimated_impact_pips=avg_loss_pips,
                        sample_size=blocked_trades,
                        regime_stats=regime_stats
                    ))

        # Sort by confidence and estimated impact
        recommendations.sort(key=lambda r: (r.confidence, r.estimated_impact_pips), reverse=True)

        return recommendations

    def get_regime_summary(
        self,
        regime_performance: Dict[str, Dict[str, RegimePerformance]]
    ) -> pd.DataFrame:
        """
        Get summary table of regime performance by epic.

        Returns DataFrame with columns:
            epic, regime, win_rate, trades, pips, profit_factor, recommendation
        """
        rows = []

        for epic, regimes in regime_performance.items():
            # Calculate overall for comparison
            total_trades = sum(r.total_trades for r in regimes.values())
            total_winners = sum(r.winners for r in regimes.values())
            overall_wr = total_winners / total_trades if total_trades > 0 else 0

            for regime, perf in regimes.items():
                row = {
                    'epic': epic,
                    'regime': regime,
                    'win_rate': perf.win_rate,
                    'trades': perf.total_trades,
                    'pips': perf.total_pips,
                    'profit_factor': perf.profit_factor,
                }

                # Determine recommendation
                if perf.win_rate >= overall_wr + 0.10:
                    row['recommendation'] = 'Best'
                elif perf.win_rate < self.MIN_WIN_RATE_THRESHOLD:
                    row['recommendation'] = 'BLOCK'
                elif perf.win_rate < overall_wr - 0.10:
                    row['recommendation'] = 'Caution'
                else:
                    row['recommendation'] = 'OK'

                rows.append(row)

        return pd.DataFrame(rows)

    def analyze_session_performance(
        self,
        trade_df: pd.DataFrame,
        market_intel_df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance by trading session.

        Returns Dict[epic, Dict[session, win_rate]]
        """
        results = {}

        if trade_df.empty:
            return results

        # Use session from trade data if available, otherwise from market intel
        if 'market_session' in trade_df.columns:
            session_col = 'market_session'
        else:
            # Enrich with session from market intel
            trade_df = self._enrich_with_session(trade_df, market_intel_df)
            session_col = 'session_at_entry'

        if session_col not in trade_df.columns:
            return results

        for epic in trade_df['epic'].unique():
            epic_df = trade_df[trade_df['epic'] == epic]
            results[epic] = {}

            for session in epic_df[session_col].dropna().unique():
                session_df = epic_df[epic_df[session_col] == session]

                if len(session_df) >= self.MIN_REGIME_TRADES:
                    win_rate = session_df['is_winner'].mean() if 'is_winner' in session_df.columns else 0
                    results[epic][session] = win_rate

        return results

    def _enrich_with_session(
        self,
        trade_df: pd.DataFrame,
        market_intel_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Enrich trade data with session at entry time"""
        if market_intel_df.empty or 'current_session' not in market_intel_df.columns:
            return trade_df

        trade_df = trade_df.copy()
        market_intel_df = market_intel_df.copy()

        trade_df['entry_timestamp'] = pd.to_datetime(trade_df['entry_timestamp'])
        market_intel_df['scan_timestamp'] = pd.to_datetime(market_intel_df['scan_timestamp'])

        def find_session(entry_ts):
            if pd.isna(entry_ts) or market_intel_df.empty:
                return None

            time_diffs = abs(market_intel_df['scan_timestamp'] - entry_ts)
            min_idx = time_diffs.idxmin()

            if time_diffs[min_idx].total_seconds() > 30 * 60:  # 30 min tolerance
                return None

            return market_intel_df.loc[min_idx, 'current_session']

        trade_df['session_at_entry'] = trade_df['entry_timestamp'].apply(find_session)

        return trade_df
