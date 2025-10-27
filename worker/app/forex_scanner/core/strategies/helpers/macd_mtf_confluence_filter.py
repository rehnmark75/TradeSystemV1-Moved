# core/strategies/helpers/macd_mtf_confluence_filter.py
"""
Multi-Timeframe MACD Trend Filter for Confluence Strategy

Validates that entries align with higher timeframe MACD trend direction.
This prevents counter-trend trades and improves win rate.

Timeframe Hierarchy:
- H4: Major trend direction (primary filter)
- H1: Intermediate trend (structure identification)
- 15M: Entry timing (pattern detection)

Entry Rules:
- BULL signal on 15M requires H4 MACD bullish (MACD line > Signal line)
- BEAR signal on 15M requires H4 MACD bearish (MACD line < Signal line)
- H4 histogram must be expanding (building momentum)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging


class MACDMultiTimeframeFilter:
    """
    Multi-timeframe MACD trend validation.

    Ensures 15M entry signals align with H4 MACD trend direction.
    """

    def __init__(self,
                 data_fetcher=None,
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 require_histogram_expansion: bool = True,
                 min_histogram_value: float = 0.00001,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize MTF MACD filter.

        Args:
            data_fetcher: Data fetcher instance for multi-timeframe data
            fast_period: MACD fast EMA period
            slow_period: MACD slow EMA period
            signal_period: MACD signal line period
            require_histogram_expansion: Require H4 histogram to be expanding
            min_histogram_value: Minimum histogram magnitude to consider valid
            logger: Optional logger instance
        """
        self.data_fetcher = data_fetcher
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.require_histogram_expansion = require_histogram_expansion
        self.min_histogram_value = min_histogram_value
        self.logger = logger or logging.getLogger(__name__)

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD indicator.

        Args:
            df: DataFrame with price data (must have 'close' column)

        Returns:
            DataFrame with MACD columns added
        """
        if 'close' not in df.columns:
            self.logger.error("DataFrame missing 'close' column")
            return df

        # Calculate EMAs
        ema_fast = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.slow_period, adjust=False).mean()

        # MACD line
        df['macd_line'] = ema_fast - ema_slow

        # Signal line
        df['macd_signal'] = df['macd_line'].ewm(span=self.signal_period, adjust=False).mean()

        # Histogram
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']

        return df

    def _is_histogram_expanding(self, df: pd.DataFrame, lookback: int = 3) -> bool:
        """
        Check if MACD histogram is expanding.

        Args:
            df: DataFrame with MACD data
            lookback: Number of bars to check

        Returns:
            True if histogram is expanding (getting larger in absolute value)
        """
        if len(df) < lookback + 1:
            return False

        recent_histograms = df['macd_histogram'].iloc[-lookback:].abs()

        # Check if histogram is generally increasing
        is_expanding = recent_histograms.is_monotonic_increasing

        return is_expanding

    def get_h4_trend_direction(self, epic: str, current_time=None) -> Optional[Dict]:
        """
        Get H4 MACD trend direction.

        Args:
            epic: Currency pair epic
            current_time: Current timestamp (for backtesting)

        Returns:
            Dict with H4 trend data:
            {
                'trend': 'bullish' or 'bearish' or 'neutral',
                'macd_line': float,
                'signal_line': float,
                'histogram': float,
                'histogram_expanding': bool,
                'is_valid': bool
            }
        """
        if not self.data_fetcher:
            self.logger.warning("No data fetcher provided - cannot fetch H4 data")
            return None

        try:
            # Fetch H4 data - handle different data fetcher interfaces
            if hasattr(self.data_fetcher, 'fetch_data'):
                # Standard data fetcher
                h4_data = self.data_fetcher.fetch_data(
                    epic=epic,
                    timeframe='4h',
                    bars=100,
                    current_time=current_time
                )
            elif hasattr(self.data_fetcher, 'get_enhanced_data'):
                # Backtest data fetcher
                # Extract pair name from epic (e.g., CS.D.EURUSD.MINI.IP -> EURUSD)
                pair = epic.split('.')[2] if '.' in epic else epic
                # Get 50 bars of H4 data = 200 hours (enough for MACD calculation)
                # MACD needs slow_period(26) + signal_period(9) = 35 bars minimum
                h4_data = self.data_fetcher.get_enhanced_data(
                    epic=epic,
                    pair=pair,
                    timeframe='4h',
                    lookback_hours=200  # 50 bars * 4 hours (conservative buffer)
                )
            else:
                self.logger.error("Data fetcher has no compatible method")
                return None

            if h4_data is None or len(h4_data) < self.slow_period + self.signal_period:
                self.logger.warning(f"Insufficient H4 data for {epic}")
                return None

            # Calculate MACD
            h4_data = self._calculate_macd(h4_data)

            # Get latest values
            latest = h4_data.iloc[-1]
            macd_line = latest['macd_line']
            signal_line = latest['macd_signal']
            histogram = latest['macd_histogram']

            # Determine trend
            if macd_line > signal_line and histogram > self.min_histogram_value:
                trend = 'bullish'
            elif macd_line < signal_line and histogram < -self.min_histogram_value:
                trend = 'bearish'
            else:
                trend = 'neutral'

            # Check histogram expansion
            histogram_expanding = False
            if self.require_histogram_expansion:
                histogram_expanding = self._is_histogram_expanding(h4_data, lookback=3)

            # Validate trend
            is_valid = True
            if self.require_histogram_expansion and not histogram_expanding:
                is_valid = False
                self.logger.debug(f"H4 MACD histogram not expanding for {epic}")

            result = {
                'trend': trend,
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': histogram,
                'histogram_expanding': histogram_expanding,
                'histogram_abs': abs(histogram),
                'is_valid': is_valid
            }

            self.logger.info(f"📊 H4 MACD trend: {trend.upper()} "
                           f"(histogram: {histogram:.5f}, expanding: {histogram_expanding})")

            return result

        except Exception as e:
            self.logger.error(f"Error fetching H4 MACD data: {e}", exc_info=True)
            return None

    def get_h1_swing_data(self, epic: str, current_time=None) -> Optional[pd.DataFrame]:
        """
        Get H1 data for swing point detection.

        This is used by FibonacciCalculator to find swing highs/lows.

        Args:
            epic: Currency pair epic
            current_time: Current timestamp (for backtesting)

        Returns:
            H1 DataFrame or None
        """
        if not self.data_fetcher:
            self.logger.warning("No data fetcher provided - cannot fetch H1 data")
            return None

        try:
            # Fetch H1 data - handle different data fetcher interfaces
            if hasattr(self.data_fetcher, 'fetch_data'):
                # Standard data fetcher
                h1_data = self.data_fetcher.fetch_data(
                    epic=epic,
                    timeframe='1h',
                    bars=100,
                    current_time=current_time
                )
            elif hasattr(self.data_fetcher, 'get_enhanced_data'):
                # Backtest data fetcher
                pair = epic.split('.')[2] if '.' in epic else epic
                # Get approximately 100 bars of H1 data = 100 hours
                h1_data = self.data_fetcher.get_enhanced_data(
                    epic=epic,
                    pair=pair,
                    timeframe='1h',
                    lookback_hours=100  # 100 bars * 1 hour
                )
            else:
                self.logger.error("Data fetcher has no compatible method")
                return None

            if h1_data is None or len(h1_data) < 50:
                self.logger.warning(f"Insufficient H1 data for {epic}")
                return None

            return h1_data

        except Exception as e:
            self.logger.error(f"Error fetching H1 data: {e}", exc_info=True)
            return None

    def validate_signal_with_h4(self,
                               signal_direction: str,
                               epic: str,
                               current_time=None,
                               allow_neutral: bool = False) -> Tuple[bool, Optional[Dict]]:
        """
        Validate 15M signal against H4 MACD trend.

        Args:
            signal_direction: '15M signal direction ('BULL' or 'BEAR')
            epic: Currency pair epic
            current_time: Current timestamp (for backtesting)
            allow_neutral: Allow entries when H4 is neutral

        Returns:
            Tuple of (is_valid, h4_data)
        """
        h4_data = self.get_h4_trend_direction(epic, current_time)

        if not h4_data:
            self.logger.warning("No H4 data available - cannot validate signal")
            return False, None

        h4_trend = h4_data['trend']

        # Check alignment
        if signal_direction == 'BULL':
            if h4_trend == 'bullish':
                is_valid = h4_data['is_valid']
            elif h4_trend == 'neutral' and allow_neutral:
                is_valid = True
                self.logger.info("⚠️ H4 neutral - allowing BULL signal with caution")
            else:
                is_valid = False
                self.logger.info(f"❌ BULL signal rejected - H4 trend is {h4_trend}")

        elif signal_direction == 'BEAR':
            if h4_trend == 'bearish':
                is_valid = h4_data['is_valid']
            elif h4_trend == 'neutral' and allow_neutral:
                is_valid = True
                self.logger.info("⚠️ H4 neutral - allowing BEAR signal with caution")
            else:
                is_valid = False
                self.logger.info(f"❌ BEAR signal rejected - H4 trend is {h4_trend}")
        else:
            is_valid = False

        if is_valid:
            self.logger.info(f"✅ Signal validated with H4 trend: {signal_direction} aligned with {h4_trend}")

        return is_valid, h4_data

    def get_mtf_confluence_boost(self, h4_data: Dict) -> float:
        """
        Calculate confidence boost based on H4 trend strength.

        Args:
            h4_data: H4 trend data from get_h4_trend_direction()

        Returns:
            Confidence boost (0.0 to 0.15)
        """
        if not h4_data or not h4_data['is_valid']:
            return 0.0

        boost = 0.0

        # Base boost for valid H4 trend
        boost += 0.05

        # Bonus for expanding histogram (strong momentum)
        if h4_data.get('histogram_expanding', False):
            boost += 0.05

        # Bonus for strong histogram magnitude
        histogram_abs = abs(h4_data.get('histogram', 0))
        if histogram_abs > 0.0001:  # Strong trend
            boost += 0.05

        return min(boost, 0.15)  # Cap at 15%


def test_mtf_filter():
    """Quick test of MTF filter"""

    # Mock data fetcher for testing
    class MockDataFetcher:
        def fetch_data(self, epic, timeframe, bars, current_time=None):
            import pandas as pd
            dates = pd.date_range('2025-01-01', periods=bars, freq='4H')

            # Create uptrend data
            prices = [1.0900 + i*0.0001 for i in range(bars)]
            df = pd.DataFrame({
                'open': prices,
                'high': [p + 0.0005 for p in prices],
                'low': [p - 0.0003 for p in prices],
                'close': prices
            }, index=dates)

            return df

    fetcher = MockDataFetcher()
    mtf_filter = MACDMultiTimeframeFilter(data_fetcher=fetcher)

    # Test H4 trend
    h4_data = mtf_filter.get_h4_trend_direction('EURUSD')
    if h4_data:
        print(f"✅ H4 trend: {h4_data['trend']}")
        print(f"   Histogram: {h4_data['histogram']:.6f}")
        print(f"   Expanding: {h4_data['histogram_expanding']}")

    # Test signal validation
    is_valid, data = mtf_filter.validate_signal_with_h4('BULL', 'EURUSD')
    print(f"✅ BULL signal validation: {is_valid}")

    # Test confidence boost
    boost = mtf_filter.get_mtf_confluence_boost(h4_data)
    print(f"✅ Confidence boost: +{boost*100:.1f}%")


if __name__ == '__main__':
    test_mtf_filter()
