"""
Trading Utility Functions

This module contains utility functions for price/point conversions and other
trading-related calculations that are used across the application.
"""

def get_point_value(epic: str) -> float:
    """
    Get point value for the instrument based on epic name.

    Args:
        epic (str): The trading instrument epic (e.g., "USDJPY", "EURUSD")

    Returns:
        float: Point value for the instrument

    Examples:
        >>> get_point_value("USDJPY")
        0.01
        >>> get_point_value("CS.D.USDJPY.MINI.IP")
        0.01
        >>> get_point_value("EURUSD")
        0.0001
        >>> get_point_value("CS.D.EURUSD.CEEM.IP")
        0.0001
        >>> get_point_value("US500")
        1.0
    """
    # NOTE: CEEM epics use regular forex pricing (e.g., 1.1646), NOT scaled format.
    # The previous comment was incorrect - prices in ig_candles are standard forex format.
    # JPY pairs use 2 decimal places (0.01 per pip)
    if "JPY" in epic:
        return 0.01
    # Standard forex pairs use 4 decimal places (0.0001 per pip)
    elif any(pair in epic for pair in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
        return 0.0001
    else:
        # For indices, commodities, etc.
        return 1.0


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
    point_value = get_point_value(epic)
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
    point_value = get_point_value(epic)
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
FOREX_JPY_PAIRS = ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"]
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