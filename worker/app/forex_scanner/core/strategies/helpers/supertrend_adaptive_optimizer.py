# core/strategies/helpers/supertrend_adaptive_optimizer.py
"""
SuperTrend Adaptive Optimizer - LuxAlgo Inspired
Implements advanced signal filtering and adaptive factor selection for SuperTrend strategy

Enhancements:
1. Performance-based filtering (tracks recent SuperTrend accuracy)
2. Trend strength filtering (prevents signals in choppy markets)
3. Slow SuperTrend stability enhancement (10-15 bars instead of 5)
4. K-means clustering for adaptive factor selection (future enhancement)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from sklearn.cluster import KMeans
from collections import deque


class SuperTrendPerformanceTracker:
    """
    Tracks SuperTrend performance over time (LuxAlgo inspired)

    Similar to LuxAlgo's performance memory system that uses exponential smoothing
    to track which SuperTrend multipliers are performing well.
    """

    def __init__(self, lookback: int = 20, alpha: float = 0.1):
        """
        Initialize performance tracker

        Args:
            lookback: Number of bars to track performance
            alpha: Smoothing factor (higher = more reactive)
        """
        self.lookback = lookback
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)

    def calculate_performance(self, df: pd.DataFrame, st_column: str = 'st_fast_trend') -> pd.Series:
        """
        Calculate rolling performance of SuperTrend signals

        Performance is measured as: Did price move in direction of SuperTrend?
        Positive performance = SuperTrend was correct
        Negative performance = SuperTrend was wrong

        Args:
            df: DataFrame with SuperTrend data
            st_column: Column name for SuperTrend trend (1 = bullish, -1 = bearish)

        Returns:
            Series with performance scores
        """
        try:
            # Get SuperTrend direction and price change
            st_trend = df[st_column].shift(1)  # Previous candle's trend
            price_change = df['close'].diff()  # Price movement

            # Performance: positive if price moved with trend, negative if against
            # trend = 1 (bullish), price up (+) = positive performance
            # trend = -1 (bearish), price down (-) = positive performance
            raw_performance = st_trend * price_change

            # Apply exponential smoothing
            performance = raw_performance.ewm(alpha=self.alpha, min_periods=1).mean()

            return performance

        except Exception as e:
            self.logger.error(f"Error calculating SuperTrend performance: {e}")
            return pd.Series(0, index=df.index)

    def is_performing_well(self, performance: float, threshold: float = 0.0) -> bool:
        """
        Check if SuperTrend is performing well

        Args:
            performance: Performance score from calculate_performance
            threshold: Minimum performance threshold (default 0 = positive performance)

        Returns:
            True if performance is above threshold
        """
        return performance > threshold


class TrendStrengthCalculator:
    """
    Calculates trend strength to filter out choppy markets

    In choppy/ranging markets, SuperTrends are too close together
    In trending markets, SuperTrends are well-separated
    """

    def __init__(self, min_separation_pct: float = 0.3):
        """
        Initialize trend strength calculator

        Args:
            min_separation_pct: Minimum separation as % of price (0.3 = 0.3%)
        """
        self.min_separation_pct = min_separation_pct
        self.logger = logging.getLogger(__name__)

    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate trend strength from SuperTrend separation

        Wide separation = strong trend
        Narrow separation = choppy market

        Args:
            df: DataFrame with SuperTrend columns (st_fast, st_slow)

        Returns:
            Series with trend strength as percentage
        """
        try:
            st_fast = df['st_fast']
            st_slow = df['st_slow']
            close = df['close']

            # Calculate separation as percentage of price
            separation_pct = abs(st_fast - st_slow) / close * 100

            return separation_pct

        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return pd.Series(0, index=df.index)

    def is_strong_trend(self, separation_pct: float) -> bool:
        """
        Check if trend is strong enough for trading

        Args:
            separation_pct: Trend strength from calculate_trend_strength

        Returns:
            True if trend is strong enough
        """
        return separation_pct > self.min_separation_pct


class SlowSupertrendStabilityFilter:
    """
    Enhanced slow SuperTrend stability filter

    Increases requirement from 5 bars to 10-15 bars of stability
    This prevents signals in markets that are just starting to chop
    """

    def __init__(self, min_stability_bars: int = 12):
        """
        Initialize stability filter

        Args:
            min_stability_bars: Minimum number of bars for stability (default 12)
        """
        self.min_stability_bars = min_stability_bars
        self.logger = logging.getLogger(__name__)

    def check_slow_stability(self, df: pd.DataFrame, direction: int) -> pd.Series:
        """
        Check if slow SuperTrend has been stable for minimum bars

        Args:
            df: DataFrame with st_slow_trend column
            direction: 1 for bullish, -1 for bearish

        Returns:
            Boolean series indicating stability
        """
        try:
            slow_trend = df['st_slow_trend']

            # Check if slow trend has been in same direction for N bars
            stable = slow_trend == direction

            # Check stability for each required bar
            for i in range(1, self.min_stability_bars):
                stable = stable & (slow_trend.shift(i) == direction)

            return stable

        except Exception as e:
            self.logger.error(f"Error checking slow SuperTrend stability: {e}")
            return pd.Series(False, index=df.index)


class AdaptiveFactorOptimizer:
    """
    LuxAlgo-style adaptive factor optimization using K-means clustering

    This is the most advanced feature from LuxAlgo:
    - Tests multiple ATR multipliers (factors)
    - Groups them by performance using K-means clustering
    - Selects factors from best-performing cluster
    - Adapts to changing market conditions
    """

    def __init__(
        self,
        min_factor: float = 1.0,
        max_factor: float = 5.0,
        step: float = 0.5,
        n_clusters: int = 3,
        lookback_bars: int = 500
    ):
        """
        Initialize adaptive factor optimizer

        Args:
            min_factor: Minimum ATR multiplier to test
            max_factor: Maximum ATR multiplier to test
            step: Step size between factors
            n_clusters: Number of clusters for K-means (default 3: best/average/worst)
            lookback_bars: Number of bars for performance evaluation
        """
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.step = step
        self.n_clusters = n_clusters
        self.lookback_bars = lookback_bars
        self.logger = logging.getLogger(__name__)

        # Generate factor range
        self.factors = np.arange(min_factor, max_factor + step, step)
        self.logger.info(f"🔬 Adaptive optimizer initialized with {len(self.factors)} factors: {self.factors}")

    def calculate_supertrends_for_all_factors(
        self,
        df: pd.DataFrame,
        atr_period: int = 14
    ) -> Dict[float, pd.DataFrame]:
        """
        Calculate SuperTrend for all factors

        Args:
            df: DataFrame with OHLC data
            atr_period: ATR period for SuperTrend calculation

        Returns:
            Dictionary mapping factor -> DataFrame with SuperTrend columns
        """
        try:
            results = {}

            # Calculate ATR once
            from .supertrend_calculator import calculate_atr
            atr = calculate_atr(df, period=atr_period)

            # Calculate SuperTrend for each factor
            for factor in self.factors:
                st_df = self._calculate_single_supertrend(df.copy(), atr, factor)
                results[factor] = st_df

            self.logger.info(f"✅ Calculated SuperTrends for {len(results)} factors")
            return results

        except Exception as e:
            self.logger.error(f"Error calculating SuperTrends for all factors: {e}")
            return {}

    def _calculate_single_supertrend(
        self,
        df: pd.DataFrame,
        atr: pd.Series,
        factor: float
    ) -> pd.DataFrame:
        """Calculate SuperTrend for a single factor"""
        try:
            from .supertrend_calculator import calculate_supertrend_bands

            # Calculate SuperTrend bands
            upper_band = df['high'] + (factor * atr)
            lower_band = df['low'] - (factor * atr)

            # Calculate trend
            trend = pd.Series(1, index=df.index)  # Start bullish
            supertrend = lower_band.copy()

            for i in range(1, len(df)):
                # Update trend
                if df['close'].iloc[i] > supertrend.iloc[i-1]:
                    trend.iloc[i] = 1
                elif df['close'].iloc[i] < supertrend.iloc[i-1]:
                    trend.iloc[i] = -1
                else:
                    trend.iloc[i] = trend.iloc[i-1]

                # Update SuperTrend value
                if trend.iloc[i] == 1:
                    supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
                else:
                    supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])

            df[f'st_{factor}'] = supertrend
            df[f'st_{factor}_trend'] = trend

            return df

        except Exception as e:
            self.logger.error(f"Error calculating SuperTrend for factor {factor}: {e}")
            return df

    def evaluate_factor_performance(
        self,
        st_results: Dict[float, pd.DataFrame]
    ) -> Dict[float, float]:
        """
        Evaluate performance of each factor

        Performance = Average profit when following SuperTrend signals

        Args:
            st_results: Dictionary of factor -> DataFrame with SuperTrend data

        Returns:
            Dictionary mapping factor -> performance score
        """
        try:
            performances = {}

            for factor, df in st_results.items():
                # Get trend and price changes
                trend_col = f'st_{factor}_trend'
                if trend_col not in df.columns:
                    continue

                trend = df[trend_col].shift(1)
                price_change = df['close'].diff()

                # Performance: did price move with trend?
                performance = (trend * price_change).mean()
                performances[factor] = performance

            self.logger.info(f"📊 Factor performances: {performances}")
            return performances

        except Exception as e:
            self.logger.error(f"Error evaluating factor performance: {e}")
            return {}

    def cluster_factors(
        self,
        performances: Dict[float, float]
    ) -> Dict[str, list]:
        """
        Cluster factors by performance using K-means

        Args:
            performances: Dictionary of factor -> performance score

        Returns:
            Dictionary with cluster names (best/average/worst) -> list of factors
        """
        try:
            if len(performances) < self.n_clusters:
                self.logger.warning(f"Not enough factors for clustering ({len(performances)} < {self.n_clusters})")
                return {'best': list(performances.keys())}

            # Prepare data for K-means
            factors = np.array(list(performances.keys())).reshape(-1, 1)
            perfs = np.array(list(performances.values())).reshape(-1, 1)

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(perfs)

            # Group factors by cluster
            clusters = {}
            for i in range(self.n_clusters):
                cluster_factors = [f for f, label in zip(performances.keys(), cluster_labels) if label == i]
                cluster_perf = np.mean([performances[f] for f in cluster_factors])
                clusters[i] = {'factors': cluster_factors, 'performance': cluster_perf}

            # Sort clusters by performance
            sorted_clusters = sorted(clusters.items(), key=lambda x: x[1]['performance'], reverse=True)

            # Assign names
            result = {
                'best': sorted_clusters[0][1]['factors'] if len(sorted_clusters) > 0 else [],
                'average': sorted_clusters[1][1]['factors'] if len(sorted_clusters) > 1 else [],
                'worst': sorted_clusters[2][1]['factors'] if len(sorted_clusters) > 2 else []
            }

            self.logger.info(f"🎯 Clustered factors:")
            self.logger.info(f"   Best: {result['best']}")
            self.logger.info(f"   Average: {result['average']}")
            self.logger.info(f"   Worst: {result['worst']}")

            return result

        except Exception as e:
            self.logger.error(f"Error clustering factors: {e}")
            return {'best': list(performances.keys())}

    def get_optimal_factor(
        self,
        cluster_choice: str = 'best'
    ) -> Optional[float]:
        """
        Get optimal factor from clustering results

        Args:
            cluster_choice: Which cluster to use ('best', 'average', 'worst')

        Returns:
            Average factor from chosen cluster
        """
        try:
            # This would be called after running the full optimization
            # For now, return middle of range
            return (self.min_factor + self.max_factor) / 2

        except Exception as e:
            self.logger.error(f"Error getting optimal factor: {e}")
            return None


def apply_enhanced_supertrend_filters(
    df: pd.DataFrame,
    performance_threshold: float = 0.0,
    min_trend_strength: float = 0.3,
    min_stability_bars: int = 12,
    enable_performance_filter: bool = True,
    enable_trend_strength_filter: bool = True,
    enable_stability_filter: bool = True
) -> pd.DataFrame:
    """
    Apply all enhanced SuperTrend filters to the DataFrame

    This is the main function that applies all improvements:
    1. Performance-based filtering (Option 3)
    2. Trend strength filtering (Option 5)
    3. Enhanced stability filtering (Option 2)

    Args:
        df: DataFrame with SuperTrend signals
        performance_threshold: Minimum performance for signals (default 0.0)
        min_trend_strength: Minimum trend strength percentage (default 0.3%)
        min_stability_bars: Minimum stability bars for slow ST (default 12)
        enable_performance_filter: Enable performance filtering
        enable_trend_strength_filter: Enable trend strength filtering
        enable_stability_filter: Enable enhanced stability filtering

    Returns:
        DataFrame with filtered signals
    """
    logger = logging.getLogger(__name__)

    try:
        original_bull_signals = df['bull_alert'].sum()
        original_bear_signals = df['bear_alert'].sum()

        logger.info(f"📊 Original signals: BULL={original_bull_signals}, BEAR={original_bear_signals}")

        # Create working copy
        df_filtered = df.copy()

        # 1. PERFORMANCE FILTER (Option 3)
        if enable_performance_filter:
            tracker = SuperTrendPerformanceTracker(lookback=20, alpha=0.1)
            df_filtered['st_performance'] = tracker.calculate_performance(df_filtered, 'st_fast_trend')

            # Filter signals with poor performance
            df_filtered.loc[df_filtered['st_performance'] < performance_threshold, 'bull_alert'] = False
            df_filtered.loc[df_filtered['st_performance'] < performance_threshold, 'bear_alert'] = False

            perf_filtered_bull = df_filtered['bull_alert'].sum()
            perf_filtered_bear = df_filtered['bear_alert'].sum()
            logger.info(f"✅ After performance filter: BULL={perf_filtered_bull}, BEAR={perf_filtered_bear}")

        # 2. TREND STRENGTH FILTER (Option 5)
        if enable_trend_strength_filter:
            strength_calc = TrendStrengthCalculator(min_separation_pct=min_trend_strength)
            df_filtered['trend_strength'] = strength_calc.calculate_trend_strength(df_filtered)

            # Filter signals in weak trends
            df_filtered.loc[df_filtered['trend_strength'] < min_trend_strength, 'bull_alert'] = False
            df_filtered.loc[df_filtered['trend_strength'] < min_trend_strength, 'bear_alert'] = False

            strength_filtered_bull = df_filtered['bull_alert'].sum()
            strength_filtered_bear = df_filtered['bear_alert'].sum()
            logger.info(f"✅ After trend strength filter: BULL={strength_filtered_bull}, BEAR={strength_filtered_bear}")

        # 3. ENHANCED STABILITY FILTER (Option 2)
        if enable_stability_filter:
            stability_filter = SlowSupertrendStabilityFilter(min_stability_bars=min_stability_bars)

            # Check stability for bull signals
            bull_stability = stability_filter.check_slow_stability(df_filtered, direction=1)
            df_filtered.loc[~bull_stability, 'bull_alert'] = False

            # Check stability for bear signals
            bear_stability = stability_filter.check_slow_stability(df_filtered, direction=-1)
            df_filtered.loc[~bear_stability, 'bear_alert'] = False

            final_bull = df_filtered['bull_alert'].sum()
            final_bear = df_filtered['bear_alert'].sum()
            logger.info(f"✅ After stability filter ({min_stability_bars} bars): BULL={final_bull}, BEAR={final_bear}")

        # Final summary
        final_bull_signals = df_filtered['bull_alert'].sum()
        final_bear_signals = df_filtered['bear_alert'].sum()
        reduction_pct = (1 - (final_bull_signals + final_bear_signals) / (original_bull_signals + original_bear_signals)) * 100

        logger.info(f"🎯 FINAL: BULL={final_bull_signals}, BEAR={final_bear_signals}")
        logger.info(f"📉 Signal reduction: {reduction_pct:.1f}%")

        return df_filtered

    except Exception as e:
        logger.error(f"Error applying enhanced SuperTrend filters: {e}", exc_info=True)
        return df
