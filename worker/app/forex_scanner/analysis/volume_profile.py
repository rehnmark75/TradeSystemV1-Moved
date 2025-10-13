# analysis/volume_profile.py
"""
Volume Profile Data Structures and Core Types

This module provides the core data structures for Volume Profile analysis:
- VolumeNode: Represents High/Low Volume Node zones
- VolumeProfile: Complete volume profile with all calculated metrics

These structures are used by the Volume Profile strategy for identifying
institutional support/resistance levels and trading opportunities.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class VolumeNode:
    """
    Represents a High Volume Node (HVN) or Low Volume Node (LVN) zone.

    HVN zones indicate strong institutional interest (support/resistance).
    LVN zones indicate price rejection areas (potential breakout zones).

    Attributes:
        price_center: Center price of the volume node
        price_low: Lower boundary of the zone
        price_high: Upper boundary of the zone
        volume: Total volume within this zone
        strength: Normalized strength 0-1 (how strong this node is)
    """
    price_center: float
    price_low: float
    price_high: float
    volume: float
    strength: float  # Normalized 0-1

    def __repr__(self):
        return f"VolumeNode(center={self.price_center:.5f}, strength={self.strength:.2f})"


@dataclass
class VolumeProfile:
    """
    Complete volume profile data structure.

    Contains all calculated metrics from Volume-by-Price analysis including:
    - POC (Point of Control): Highest volume price level
    - VAH/VAL (Value Area High/Low): Boundaries of 70% volume
    - HVN zones: High Volume Nodes (support/resistance)
    - LVN zones: Low Volume Nodes (breakout opportunities)

    This structure is returned by VolumeProfileCalculator and used by
    trading strategies for signal generation.
    """

    # Core metrics
    poc: float  # Point of Control (highest volume price)
    vah: float  # Value Area High (top of 70% volume)
    val: float  # Value Area Low (bottom of 70% volume)

    # Volume distribution
    price_levels: np.ndarray  # Price bin centers
    volume_at_price: np.ndarray  # Volume at each price level
    volume_distribution_pct: np.ndarray  # Percentage distribution

    # Key levels
    hvn_zones: List[VolumeNode]  # High Volume Nodes
    lvn_zones: List[VolumeNode]  # Low Volume Nodes

    # Metadata
    total_volume: float
    value_area_volume_pct: float  # Should be ~70%
    lookback_bars: int
    price_range: Tuple[float, float]  # (min, max)
    calculation_time_ms: float

    # Statistics
    mean_volume: float
    std_volume: float
    volume_skewness: float  # Positive = volume skewed to higher prices

    def __repr__(self):
        return (f"VolumeProfile(POC={self.poc:.5f}, VAH={self.vah:.5f}, VAL={self.val:.5f}, "
                f"HVN={len(self.hvn_zones)}, LVN={len(self.lvn_zones)}, "
                f"lookback={self.lookback_bars})")

    def get_summary(self) -> dict:
        """Get summary statistics for logging/debugging"""
        return {
            'poc': round(self.poc, 5),
            'vah': round(self.vah, 5),
            'val': round(self.val, 5),
            'value_area_width_pips': round((self.vah - self.val) / 0.0001, 1),
            'hvn_count': len(self.hvn_zones),
            'lvn_count': len(self.lvn_zones),
            'total_volume': int(self.total_volume),
            'value_area_volume_pct': round(self.value_area_volume_pct * 100, 1),
            'calculation_time_ms': round(self.calculation_time_ms, 2),
            'volume_skewness': round(self.volume_skewness, 3),
        }
