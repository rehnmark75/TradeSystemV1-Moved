"""
Trading Utility Functions — SINGLE SOURCE OF TRUTH for price/point conversions.

================================================================================
PIP SIZE (a.k.a. "point value") — MARKET CONSTANT
================================================================================
Pip size is the price-per-pip for an instrument. It is fixed by the broker /
instrument specification and is NOT a tunable parameter.

Reference values:
    EUR/USD, GBP/USD, etc.  : 0.0001 USD per pip   (4-decimal forex)
    USD/JPY, EUR/JPY, etc.  : 0.01   per pip       (2-decimal forex)
    Gold (CFEGOLD/XAU)      : 0.1    USD per pip
    Indices, other commodit.: 1.0    (default)

⚠️  DO NOT CHANGE these values unless the broker changes the instrument spec.
    Changing pip_size silently corrupts ALL SL/TP, trailing-stop, breakeven,
    and pip-distance math — across both live trading and backtests.

================================================================================
NOT TO BE CONFUSED WITH:
================================================================================
1. POSITION / CONTRACT SIZE  →  configured in `config.py` → `EPIC_ORDER_SIZES`.
   To trade BIGGER contracts (more lots), edit that dict — e.g., gold from
   0.05 → 0.1 lots. Pip size stays the same; only your $/pip exposure changes.

2. IG API "POINT" UNIT       →  IG's stopDistance / limitDistance API
   parameters use IG points. For most FX, 1 IG point = 1 pip; for IG CFEGOLD,
   1 IG point = 0.5 price units = 5 XAU pips. The system converts to IG
   points only at the API boundary; everywhere else uses pips.

If you add a new instrument:
    - Add its pattern to `get_point_value` below with the broker's pip size.
    - Add its order size to `config.py → EPIC_ORDER_SIZES`.
    - Verify on demo that SL/TP placed at "N pips" actually moves N × pip_size.
"""


def get_point_value(epic: str) -> float:
    """
    Resolve pip size (price-per-pip) from an epic string.

    Match order matters: JPY → forex majors → gold/XAU → fallback.
    Gold patterns include "CFEGOLD", "GOLD", and "XAU" so that any epic /
    symbol form ("CS.D.CFEGOLD.CEE.IP", "CFEGOLD.1.CEE", "XAUUSD", "GOLD")
    resolves to the same pip size of 0.1.

    Examples:
        >>> get_point_value("CS.D.USDJPY.MINI.IP")
        0.01
        >>> get_point_value("CS.D.EURUSD.CEEM.IP")
        0.0001
        >>> get_point_value("CS.D.CFEGOLD.CEE.IP")
        0.1
        >>> get_point_value("XAUUSD")
        0.1
        >>> get_point_value("US500")
        1.0
    """
    if not epic:
        return 1.0
    epic_upper = epic.upper()
    # Gold first — must precede USD-pair check so "XAUUSD" doesn't fall into 0.0001.
    if "CFEGOLD" in epic_upper or "GOLD" in epic_upper or "XAU" in epic_upper:
        return 0.1
    # JPY pairs (2-decimal price)
    if "JPY" in epic_upper:
        return 0.01
    # Standard 4-decimal forex
    if any(pair in epic_upper for pair in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
        return 0.0001
    # Indices, untracked commodities, etc.
    return 1.0


def get_ig_point_size(epic: str) -> float:
    """
    Resolve IG broker-point size (price units per `stopDistance` point on the IG REST API).

    For most instruments this equals the pip size returned by ``get_point_value``,
    but for IG CFEGOLD an IG point equals 0.5 price units (= 5 XAU pips). Mixing
    the two units when persisting a stop level produces a 5× error on gold orders.
    """
    if not epic:
        return 1.0
    epic_upper = epic.upper()
    if "CFEGOLD" in epic_upper or "GOLD" in epic_upper or "XAU" in epic_upper:
        return 0.5
    return get_point_value(epic)


def convert_stop_distance_to_price(entry_price: float, stop_distance_points: int,
                                 direction: str, epic: str) -> float:
    """
    Convert stop distance in points to actual stop price level.
    
    Args:
        entry_price (float): The entry price of the trade
        stop_distance_points (int): Stop distance in points (e.g., 10)
        direction (str): Trade direction ("BUY" or "SELL")
        epic (str): The trading instrument epic
        
    Returns:
        float: Actual stop price level
        
    Examples:
        >>> convert_stop_distance_to_price(145.460, 10, "SELL", "USDJPY")
        145.560
        >>> convert_stop_distance_to_price(1.1000, 15, "BUY", "EURUSD")
        1.0985
    """
    point_value = get_ig_point_size(epic)
    stop_distance_price = stop_distance_points * point_value
    
    if direction.upper() == "BUY":
        # For BUY: stop is below entry price
        stop_price = entry_price - stop_distance_price
    else:  # SELL
        # For SELL: stop is above entry price  
        stop_price = entry_price + stop_distance_price
    
    return round(stop_price, 5)  # Round to 5 decimal places for forex


def convert_limit_distance_to_price(entry_price: float, limit_distance_points: int, 
                                  direction: str, epic: str) -> float:
    """
    Convert limit distance in points to actual limit price level.
    
    Args:
        entry_price (float): The entry price of the trade
        limit_distance_points (int): Limit distance in points (e.g., 15)
        direction (str): Trade direction ("BUY" or "SELL")
        epic (str): The trading instrument epic
        
    Returns:
        float: Actual limit price level
        
    Examples:
        >>> convert_limit_distance_to_price(145.460, 15, "SELL", "USDJPY")
        145.310
        >>> convert_limit_distance_to_price(1.1000, 20, "BUY", "EURUSD")
        1.1020
    """
    point_value = get_ig_point_size(epic)
    limit_distance_price = limit_distance_points * point_value
    
    if direction.upper() == "BUY":
        # For BUY: limit is above entry price
        limit_price = entry_price + limit_distance_price
    else:  # SELL
        # For SELL: limit is below entry price  
        limit_price = entry_price - limit_distance_price
    
    return round(limit_price, 5)  # Round to 5 decimal places for forex


def is_jpy_pair(epic: str) -> bool:
    """
    Check if the trading pair involves JPY currency.

    Args:
        epic (str): The trading instrument epic (e.g., "USDJPY", "CS.D.USDJPY.MINI.IP")

    Returns:
        bool: True if JPY pair, False otherwise

    Examples:
        >>> is_jpy_pair("USDJPY")
        True
        >>> is_jpy_pair("CS.D.USDJPY.MINI.IP")
        True
        >>> is_jpy_pair("EURUSD")
        False
    """
    return "JPY" in epic.upper()


def convert_price_to_points(price_distance: float, epic: str) -> int:
    """
    Convert price distance to points.
    
    Args:
        price_distance (float): Distance in price units (e.g., 0.0015)
        epic (str): The trading instrument epic
        
    Returns:
        int: Distance in points
        
    Examples:
        >>> convert_price_to_points(0.10, "USDJPY")
        10
        >>> convert_price_to_points(0.0015, "EURUSD")
        15
    """
    point_value = get_point_value(epic)
    return int(round(abs(price_distance) / point_value))


def calculate_move_points(from_price: float, to_price: float, direction: str, epic: str) -> int:
    """
    Calculate movement in points, considering trade direction.
    
    Args:
        from_price (float): Starting price
        to_price (float): Ending price
        direction (str): Trade direction ("BUY" or "SELL")
        epic (str): The trading instrument epic
        
    Returns:
        int: Movement in points (positive = in favor, negative = against)
        
    Examples:
        >>> calculate_move_points(1.1000, 1.1015, "BUY", "EURUSD")
        15
        >>> calculate_move_points(145.460, 145.400, "SELL", "USDJPY")
        60
    """
    if direction.upper() == "BUY":
        move = to_price - from_price
    else:  # SELL
        move = from_price - to_price
    
    return convert_price_to_points(move, epic)


def validate_epic(epic: str) -> bool:
    """
    Validate if epic is a recognized trading instrument.
    
    Args:
        epic (str): The trading instrument epic
        
    Returns:
        bool: True if epic is recognized, False otherwise
    """
    known_patterns = [
        "JPY", "USD", "EUR", "GBP", "AUD", "NZD", "CAD", "CHF",  # Forex
        "US500", "US30", "UK100", "GER40", "JPN225",  # Indices
        "GOLD", "SILVER", "OIL", "COPPER"  # Commodities
    ]
    
    return any(pattern in epic for pattern in known_patterns)


def format_price(price: float, epic: str) -> str:
    """
    Format price with appropriate decimal places for the instrument.
    
    Args:
        price (float): Price to format
        epic (str): The trading instrument epic
        
    Returns:
        str: Formatted price string
        
    Examples:
        >>> format_price(145.456789, "USDJPY")
        "145.457"
        >>> format_price(1.123456, "EURUSD")
        "1.1235"
    """
    if "JPY" in epic:
        return f"{price:.3f}"
    elif any(pair in epic for pair in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
        return f"{price:.4f}"
    else:
        return f"{price:.2f}"


# Constants for common instruments
FOREX_MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
FOREX_JPY_PAIRS = ["USDJPY", "EURJPY", "AUDJPY"]
MAJOR_INDICES = ["US500", "US30", "UK100", "GER40", "JPN225", "AUS200"]
MAJOR_COMMODITIES = ["GOLD", "SILVER", "OIL.WTI", "OIL.BRENT", "COPPER"]


if __name__ == "__main__":
    # Test the functions
    print("Testing utility functions:")
    
    # Test USDJPY
    print(f"USDJPY point value: {get_point_value('USDJPY')}")
    stop_price = convert_stop_distance_to_price(145.460, 10, "SELL", "USDJPY")
    print(f"USDJPY SELL stop price (10 points): {stop_price}")
    
    # Test EURUSD
    print(f"EURUSD point value: {get_point_value('EURUSD')}")
    stop_price = convert_stop_distance_to_price(1.1000, 15, "BUY", "EURUSD")
    print(f"EURUSD BUY stop price (15 points): {stop_price}")
    
    # Test point calculation
    points = convert_price_to_points(0.0015, "EURUSD")
    print(f"0.0015 EURUSD = {points} points")
    
    # Test move calculation
    move = calculate_move_points(1.1000, 1.1015, "BUY", "EURUSD")
    print(f"BUY EURUSD move from 1.1000 to 1.1015 = {move} points")
