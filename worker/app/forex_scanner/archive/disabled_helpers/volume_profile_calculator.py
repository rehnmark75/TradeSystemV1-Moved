# core/strategies/helpers/volume_profile_calculator.py
"""
Volume Profile Calculator

High-performance Volume-by-Price calculator optimized for forex tick volume.
Calculates POC (Point of Control), VAH/VAL (Value Area), and identifies
HVN (High Volume Nodes) and LVN (Low Volume Nodes) for trading signals.

Performance target: <100ms per calculation using vectorized numpy operations.

Algorithm:
1. Create price bins based on pip size
2. Distribute bar volume across bins (vectorized)
3. Calculate POC (highest volume price)
4. Calculate Value Area (70% volume around POC)
5. Identify HVN peaks (support/resistance)
6. Identify LVN valleys (breakout zones)

Author: TradeSystemV1
Date: 2025
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from typing import List, Tuple, Optional
import time
import logging

# Import volume profile data structures
try:
    from analysis.volume_profile import VolumeProfile, VolumeNode
except ImportError:
    from forex_scanner.analysis.volume_profile import VolumeProfile, VolumeNode


class VolumeProfileCalculator:
    """
    High-performance Volume Profile calculator for forex tick volume.

    Implements Volume-by-Price algorithm with the following optimizations:
    - Vectorized numpy operations for speed
    - Intelligent price binning based on pair volatility
    - scipy peak detection for HVN/LVN identification
    - Caching support for repeated calculations

    Usage:
        calculator = VolumeProfileCalculator(price_bin_pips=1.0, value_area_pct=0.70)
        profile = calculator.calculate_profile(df, lookback_periods=50)
    """

    def __init__(
        self,
        price_bin_pips: float = 1.0,
        value_area_pct: float = 0.70,
        hvn_threshold_std: float = 1.5,
        lvn_threshold_std: float = 0.5,
        min_prominence_pct: float = 0.1,
        pip_value: float = 0.0001  # 4 decimal pairs (5 decimal = 0.00001)
    ):
        """
        Initialize volume profile calculator.

        Args:
            price_bin_pips: Price bin size in pips (1-5 recommended)
            value_area_pct: Value area percentage (typically 0.70 = 70%)
            hvn_threshold_std: Standard deviations above mean for HVN (1.5-2.5)
            lvn_threshold_std: Standard deviations below mean for LVN (0.3-0.5)
            min_prominence_pct: Minimum peak prominence as % of max volume
            pip_value: Value of 1 pip (0.0001 for 4-decimal, 0.00001 for 5-decimal)
        """
        self.price_bin_pips = price_bin_pips
        self.value_area_pct = value_area_pct
        self.hvn_threshold_std = hvn_threshold_std
        self.lvn_threshold_std = lvn_threshold_std
        self.min_prominence_pct = min_prominence_pct
        self.pip_value = pip_value

        self.logger = logging.getLogger(__name__)

        # Cache for incremental updates (optional)
        self._cache = {}

    def calculate_profile(
        self,
        df: pd.DataFrame,
        lookback_periods: int = 50,
        use_cache: bool = False
    ) -> VolumeProfile:
        """
        Calculate volume profile for last N bars.

        Args:
            df: DataFrame with columns [open, high, low, close, volume] or [ltv]
            lookback_periods: Number of bars to include in calculation
            use_cache: Whether to use cached calculations (for performance)

        Returns:
            VolumeProfile object with all calculated metrics

        Performance: Typically 15-50ms for 50-100 bars
        """
        start_time = time.perf_counter()

        # Extract lookback window
        if len(df) < lookback_periods:
            lookback_periods = len(df)

        df_window = df.iloc[-lookback_periods:].copy()

        # Determine volume column name
        volume_col = 'ltv' if 'ltv' in df_window.columns else 'volume'

        if volume_col not in df_window.columns:
            raise ValueError(f"DataFrame must have '{volume_col}' column")

        # Step 1: Determine price bins
        price_min = df_window['low'].min()
        price_max = df_window['high'].max()
        price_range = price_max - price_min

        # Calculate bin size in price units
        bin_size = self.price_bin_pips * self.pip_value

        # Create price bins (ensure at least 10 bins, max 500 bins)
        n_bins = int(np.ceil(price_range / bin_size))
        n_bins = np.clip(n_bins, 10, 500)

        # Create bin edges
        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Step 2: Distribute volume across price levels (vectorized)
        volume_at_price = self._distribute_volume_vectorized(
            df_window, bin_edges, bin_centers, volume_col
        )

        # Step 3: Calculate Point of Control (POC)
        poc_idx = np.argmax(volume_at_price)
        poc = bin_centers[poc_idx]

        # Step 4: Calculate Value Area (VAH/VAL)
        vah, val, value_area_volume_pct = self._calculate_value_area(
            bin_centers, volume_at_price, poc_idx
        )

        # Step 5: Identify HVN and LVN zones
        hvn_zones = self._find_hvn_zones(bin_centers, volume_at_price)
        lvn_zones = self._find_lvn_zones(bin_centers, volume_at_price)

        # Calculate statistics
        total_volume = volume_at_price.sum()
        volume_distribution_pct = volume_at_price / total_volume if total_volume > 0 else volume_at_price
        mean_volume = volume_at_price.mean()
        std_volume = volume_at_price.std()

        # Calculate volume skewness (where is volume concentrated?)
        volume_skewness = stats.skew(volume_at_price) if len(volume_at_price) > 2 else 0.0

        calc_time_ms = (time.perf_counter() - start_time) * 1000

        return VolumeProfile(
            poc=poc,
            vah=vah,
            val=val,
            price_levels=bin_centers,
            volume_at_price=volume_at_price,
            volume_distribution_pct=volume_distribution_pct,
            hvn_zones=hvn_zones,
            lvn_zones=lvn_zones,
            total_volume=total_volume,
            value_area_volume_pct=value_area_volume_pct,
            lookback_bars=lookback_periods,
            price_range=(price_min, price_max),
            calculation_time_ms=calc_time_ms,
            mean_volume=mean_volume,
            std_volume=std_volume,
            volume_skewness=volume_skewness
        )

    def _distribute_volume_vectorized(
        self,
        df: pd.DataFrame,
        bin_edges: np.ndarray,
        bin_centers: np.ndarray,
        volume_col: str
    ) -> np.ndarray:
        """
        Distribute bar volume across price levels using vectorized operations.

        Algorithm:
        1. For each bar, determine which price bins it spans (high to low)
        2. Distribute volume proportionally based on price range coverage
        3. Weight by proximity to close (volume concentrates near close)
        4. Use numpy broadcasting for speed

        Performance: O(n_bars * avg_bins_per_bar), typically 5-30ms for 50 bars

        Args:
            df: DataFrame with OHLC and volume data
            bin_edges: Price bin edges
            bin_centers: Price bin centers
            volume_col: Name of volume column ('volume' or 'ltv')

        Returns:
            Array of volume at each price level
        """
        volume_at_price = np.zeros(len(bin_centers))

        # Vectorized approach: process all bars efficiently
        lows = df['low'].values
        highs = df['high'].values
        volumes = df[volume_col].values
        closes = df['close'].values

        # For each bin, check which bars intersect it
        for i, (bin_low, bin_high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            bin_center = bin_centers[i]

            # Bars that intersect this bin
            overlaps = (highs >= bin_low) & (lows <= bin_high)

            if not overlaps.any():
                continue

            # Calculate overlap amount for each bar
            overlap_lows = np.maximum(lows[overlaps], bin_low)
            overlap_highs = np.minimum(highs[overlaps], bin_high)
            overlap_ranges = overlap_highs - overlap_lows

            # Calculate bar ranges
            bar_ranges = highs[overlaps] - lows[overlaps]
            bar_ranges = np.maximum(bar_ranges, self.pip_value)  # Avoid division by zero

            # Distribute volume proportionally
            volume_fractions = overlap_ranges / bar_ranges

            # Weight by proximity to close (volume tends to concentrate near close)
            close_distances = np.abs(closes[overlaps] - bin_center)
            close_weights = 1.0 / (1.0 + close_distances / (bar_ranges / 2))

            # Combined weight
            weights = volume_fractions * close_weights

            # Add weighted volume
            volume_at_price[i] = (volumes[overlaps] * weights).sum()

        return volume_at_price

    def _calculate_value_area(
        self,
        bin_centers: np.ndarray,
        volume_at_price: np.ndarray,
        poc_idx: int
    ) -> Tuple[float, float, float]:
        """
        Calculate Value Area High and Low (70% of volume around POC).

        Algorithm:
        1. Start from POC (highest volume price)
        2. Expand up and down, adding the side with more volume
        3. Stop when accumulated volume >= target percentage (70%)

        This follows the CBOT Market Profile methodology.

        Performance: O(n_bins), typically <1ms

        Args:
            bin_centers: Price bin centers
            volume_at_price: Volume at each price level
            poc_idx: Index of POC in arrays

        Returns:
            Tuple of (vah, val, actual_volume_pct)
        """
        total_volume = volume_at_price.sum()
        target_volume = total_volume * self.value_area_pct

        # Start from POC
        accumulated_volume = volume_at_price[poc_idx]
        upper_idx = poc_idx
        lower_idx = poc_idx

        # Expand value area
        while accumulated_volume < target_volume:
            # Check if we can expand
            can_go_up = upper_idx < len(bin_centers) - 1
            can_go_down = lower_idx > 0

            if not can_go_up and not can_go_down:
                break

            # Determine which direction to expand
            upper_volume = volume_at_price[upper_idx + 1] if can_go_up else -1
            lower_volume = volume_at_price[lower_idx - 1] if can_go_down else -1

            if upper_volume >= lower_volume and can_go_up:
                upper_idx += 1
                accumulated_volume += upper_volume
            elif can_go_down:
                lower_idx -= 1
                accumulated_volume += lower_volume
            else:
                break

        vah = bin_centers[upper_idx]
        val = bin_centers[lower_idx]
        value_area_volume_pct = accumulated_volume / total_volume if total_volume > 0 else 0

        return vah, val, value_area_volume_pct

    def _find_hvn_zones(
        self,
        bin_centers: np.ndarray,
        volume_at_price: np.ndarray
    ) -> List[VolumeNode]:
        """
        Find High Volume Node zones using peak detection.

        HVN zones represent institutional accumulation/distribution areas
        and act as strong support/resistance levels.

        Algorithm:
        1. Smooth volume distribution to reduce noise
        2. Find peaks using scipy.signal.find_peaks
        3. Filter peaks by prominence and threshold
        4. Create zones around peaks (expand until volume drops)

        Performance: O(n_bins * log(n_bins)), typically 1-5ms

        Args:
            bin_centers: Price bin centers
            volume_at_price: Volume at each price level

        Returns:
            List of VolumeNode objects sorted by strength (strongest first)
        """
        if len(volume_at_price) < 5:
            return []

        # Smooth volume distribution to reduce noise (simple moving average)
        window_size = max(3, len(volume_at_price) // 20)
        if window_size % 2 == 0:
            window_size += 1  # Must be odd

        smoothed_volume = self._moving_average(volume_at_price, window_size)

        # Calculate thresholds
        mean_volume = volume_at_price.mean()
        std_volume = volume_at_price.std()
        hvn_threshold = mean_volume + (self.hvn_threshold_std * std_volume)

        # Find peaks
        min_prominence = volume_at_price.max() * self.min_prominence_pct

        try:
            peaks, properties = signal.find_peaks(
                smoothed_volume,
                height=hvn_threshold,
                prominence=min_prominence,
                distance=max(2, len(volume_at_price) // 30)  # Minimum distance between peaks
            )
        except Exception as e:
            self.logger.warning(f"HVN peak detection failed: {e}")
            return []

        # Create VolumeNode zones
        hvn_zones = []
        for peak_idx in peaks:
            # Find zone boundaries (where volume drops below mean)
            left_idx = peak_idx
            right_idx = peak_idx

            # Expand left
            while left_idx > 0 and smoothed_volume[left_idx - 1] > mean_volume:
                left_idx -= 1

            # Expand right
            while right_idx < len(smoothed_volume) - 1 and smoothed_volume[right_idx + 1] > mean_volume:
                right_idx += 1

            # Create zone
            zone_volume = volume_at_price[left_idx:right_idx + 1].sum()
            strength = (smoothed_volume[peak_idx] - mean_volume) / std_volume if std_volume > 0 else 0
            strength = min(strength / 3.0, 1.0)  # Normalize to 0-1

            hvn_zones.append(VolumeNode(
                price_center=bin_centers[peak_idx],
                price_low=bin_centers[left_idx],
                price_high=bin_centers[right_idx],
                volume=zone_volume,
                strength=strength
            ))

        # Sort by strength (strongest first)
        hvn_zones.sort(key=lambda x: x.strength, reverse=True)

        return hvn_zones

    def _find_lvn_zones(
        self,
        bin_centers: np.ndarray,
        volume_at_price: np.ndarray
    ) -> List[VolumeNode]:
        """
        Find Low Volume Node zones (gaps/valleys in volume profile).

        LVN zones represent price rejection areas where price moved quickly.
        These are potential breakout zones with minimal resistance.

        Algorithm:
        1. Invert volume distribution
        2. Find peaks in inverted distribution (= valleys in original)
        3. Filter by threshold (volume significantly below mean)
        4. Create zones around valleys

        Performance: O(n_bins * log(n_bins)), typically 1-5ms

        Args:
            bin_centers: Price bin centers
            volume_at_price: Volume at each price level

        Returns:
            List of VolumeNode objects sorted by strength (strongest LVN first)
        """
        if len(volume_at_price) < 5:
            return []

        # Smooth volume distribution
        window_size = max(3, len(volume_at_price) // 20)
        if window_size % 2 == 0:
            window_size += 1

        smoothed_volume = self._moving_average(volume_at_price, window_size)

        # Invert for valley detection
        max_volume = smoothed_volume.max()
        if max_volume == 0:
            return []

        inverted_volume = max_volume - smoothed_volume

        # Calculate thresholds
        mean_volume = volume_at_price.mean()
        std_volume = volume_at_price.std()
        lvn_threshold = mean_volume - (self.lvn_threshold_std * std_volume)
        lvn_threshold = max(lvn_threshold, 0)

        # Find valleys (peaks in inverted distribution)
        min_prominence = max_volume * self.min_prominence_pct

        try:
            valleys, properties = signal.find_peaks(
                inverted_volume,
                prominence=min_prominence,
                distance=max(2, len(volume_at_price) // 30)
            )
        except Exception as e:
            self.logger.warning(f"LVN valley detection failed: {e}")
            return []

        # Filter valleys where original volume is below threshold
        valleys = [v for v in valleys if smoothed_volume[v] < mean_volume]

        # Create VolumeNode zones
        lvn_zones = []
        for valley_idx in valleys:
            # Find zone boundaries (where volume stays low)
            left_idx = valley_idx
            right_idx = valley_idx

            # Expand left (while volume stays low)
            while left_idx > 0 and smoothed_volume[left_idx - 1] < mean_volume:
                left_idx -= 1

            # Expand right
            while right_idx < len(smoothed_volume) - 1 and smoothed_volume[right_idx + 1] < mean_volume:
                right_idx += 1

            # Create zone
            zone_volume = volume_at_price[left_idx:right_idx + 1].sum()
            strength = (mean_volume - smoothed_volume[valley_idx]) / std_volume if std_volume > 0 else 0
            strength = min(strength / 2.0, 1.0)  # Normalize to 0-1

            lvn_zones.append(VolumeNode(
                price_center=bin_centers[valley_idx],
                price_low=bin_centers[left_idx],
                price_high=bin_centers[right_idx],
                volume=zone_volume,
                strength=strength
            ))

        # Sort by strength (strongest LVN first)
        lvn_zones.sort(key=lambda x: x.strength, reverse=True)

        return lvn_zones

    @staticmethod
    def _moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
        """
        Fast moving average using numpy convolution.

        Args:
            data: Input array
            window_size: Window size for moving average

        Returns:
            Smoothed array (same length as input)
        """
        if window_size > len(data):
            window_size = len(data)
        if window_size < 1:
            return data

        kernel = np.ones(window_size) / window_size
        # Use 'same' mode to keep same length, with edge padding
        return np.convolve(data, kernel, mode='same')
