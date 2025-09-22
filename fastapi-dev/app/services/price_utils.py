"""
Price Utility Functions
Shared utility functions for price and points calculations across the trading system.
"""


def ig_points_to_price(points: float, epic: str) -> float:
    """
    Convert points to price based on epic type.

    Args:
        points: Number of points to convert
        epic: Trading symbol (e.g., 'CS.D.GBPUSD.MINI.IP')

    Returns:
        Price value as float
    """
    if "JPY" in epic:
        return round(points * 0.01, 5)
    elif any(x in epic for x in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
        return round(points * 0.0001, 5)
    else:
        return round(points * 1.0, 5)


def price_to_ig_points(price: float, epic: str) -> float:
    """
    Convert price to points based on epic type (inverse of ig_points_to_price).

    Args:
        price: Price value to convert
        epic: Trading symbol

    Returns:
        Points as float
    """
    if "JPY" in epic:
        return round(price / 0.01, 1)
    elif any(x in epic for x in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
        return round(price / 0.0001, 1)
    else:
        return round(price / 1.0, 1)


def get_point_multiplier(epic: str) -> float:
    """
    Get the point multiplier for a given epic.

    Args:
        epic: Trading symbol

    Returns:
        Point multiplier as float
    """
    if "JPY" in epic:
        return 0.01
    elif any(x in epic for x in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
        return 0.0001
    else:
        return 1.0